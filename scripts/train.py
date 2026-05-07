#!/usr/bin/env python3
"""Train Mistral 7B with QLoRA on ROCm/AMD with high-performance optimizations."""
import os
import sys
import argparse
from pathlib import Path
from urllib.parse import urlparse
import importlib.util
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


def get_args():
    parser = argparse.ArgumentParser(description="High-performance fine-tuning with optimum-amd")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training JSONL dataset"
    )
    parser.add_argument(
        "--output",
        default="./fine-tuned-model",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--prompt-style",
        default="inst",
        choices=["inst", "chat", "cot"],
        help="Prompt format"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA)"
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 for faster training"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--use-fused-optimizer",
        action="store_true",
        default=True,
        help="Use fused optimizer for better performance"
    )
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluation steps interval"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint steps"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging steps interval"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="https://huggingface.co/Ekemainai12",
        help="HuggingFace repo target. Accepts repo ID ('user/model') or URL ('https://huggingface.co/user[/model]'). Requires HF_TOKEN env var."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Ekemainai",
        help="Name of the fine-tuned model for upload metadata"
    )
    return parser.parse_args()


def resolve_hf_repo(repo_target: str, model_name: str) -> str:
    """Normalize HF target into repo_id format: namespace/model."""
    if not repo_target:
        return ""

    target = repo_target.strip().rstrip("/")

    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        parts = [p for p in parsed.path.split("/") if p]
    else:
        parts = [p for p in target.split("/") if p]

    if len(parts) == 1:
        return f"{parts[0]}/{model_name}"
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return ""


PROMPT_TEMPLATES = {
    "inst": "[INST] {instruction}\n{input} [/INST]\n{output}",
    "chat": "<|system|>\nYou are an expert coding assistant with reasoning and tool-use capabilities.<|end|>\n<|user|>\n{instruction}\n{input}<|end|>\n<|assistant|>\n{output}",
    "cot": "## Task\n{instruction}\n\n## Input\n{input}\n\n## Reasoning\nLet me analyze this step by step:\n\n{output}",
}


def format_sample(sample, tokenizer, max_seq, prompt_style="inst"):
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    template = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATES["inst"])
    prompt = template.format(
        instruction=instruction,
        input=input_text,
        output=output
    )
    
    enc = tokenizer(
        prompt,
        max_length=max_seq,
        truncation=True,
        padding="max_length",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].copy()
    }


def load_tokenize_dataset(dataset_path, tokenizer, max_seq, seed=42, prompt_style="inst"):
    print(f"Loading dataset from {dataset_path}...")

    if dataset_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_path, split="train")
    else:
        ds = load_dataset(dataset_path, split="train")

    print(f"Dataset size: {len(ds)} samples")

    dataset_name = Path(dataset_path).name
    if "cot" in dataset_name:
        print("Detected: Chain-of-Thought dataset")
    elif "reasoning" in dataset_name:
        print("Detected: Reasoning + Tools dataset")
    else:
        print("Detected: Standard dataset")

    def wrapped(s):
        return format_sample(s, tokenizer, max_seq, prompt_style)

    ds = ds.map(wrapped, remove_columns=ds.column_names, desc="Formatting")

    ds = ds.shuffle(seed=seed)

    print(f"Dataset size: {len(ds)} samples")
    
    split = ds.train_test_split(test_size=0.05)
    train_ds = split["train"]
    eval_ds = split["test"]
    
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


def setup_model(model_name, rank, alpha, device, use_4bit=True, use_flash_attn=False, gradient_checkpointing=False):
    print(f"Loading model: {model_name}")

    flash_attn_enabled = False
    if use_flash_attn and device == "cuda":
        flash_attn_enabled = importlib.util.find_spec("flash_attn") is not None
        if not flash_attn_enabled:
            print("⚠️  flash_attn package not found. Falling back to eager attention.")
    elif use_flash_attn and device != "cuda":
        print("⚠️  FlashAttention requested but CUDA/ROCm device not available. Falling back to eager attention.")

    attn_impl = "flash_attention_2" if flash_attn_enabled else "eager"

    if use_4bit:
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_impl,
        )

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def get_device():
    if torch.cuda.is_available():
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "rocm" in cuda_device.lower() or os.path.exists("/opt/rocm"):
            return "cuda"
        return "cuda"
    return "cpu"


def compute_metrics(eval_pred):
    """Compute perplexity and loss metrics."""
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits.reshape(-1, logits.shape[-1])),
        torch.tensor(labels.reshape(-1)),
        ignore_index=-100,
    )
    return {"eval_loss": loss.item(), "eval_perplexity": torch.exp(loss).item()}


class PerformanceTrainer(Trainer):
    """Enhanced trainer with AMD optimizations and performance metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_start_time = None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Enhanced evaluation with detailed metrics."""
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if "eval_loss" in output:
            output["eval_perplexity"] = torch.exp(torch.tensor(output["eval_loss"])).item()
            output["eval_ppl"] = output["eval_perplexity"]

        print(f"\n📊 Evaluation Metrics:")
        for key, value in output.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        return output


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 70)
    print("EkemainaiAgent - High-Performance QLoRA Fine-tuning with optimum-amd")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (grad accum: {args.grad_accum})")
    print(f"Max sequence: {args.max_seq}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Flash Attention: {args.use_flash_attn}")
    print(f"Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"Fused Optimizer: {args.use_fused_optimizer}")
    print("=" * 70)

    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.version.hip:
            print("ROCm Version:", torch.version.hip)

    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    model = setup_model(
        args.model,
        args.rank,
        args.alpha,
        device,
        args.use_4bit,
        args.use_flash_attn,
        args.gradient_checkpointing
    )

    train_ds, eval_ds = load_tokenize_dataset(
        args.dataset,
        tokenizer,
        args.max_seq,
        args.seed,
        args.prompt_style
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="adamw_torch_fused" if args.use_fused_optimizer else "paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="no",
        max_grad_norm=args.max_grad_norm,
        dataloader_pin_memory=False,
        torch_compile=False,
        logging_first_step=True,
        ddp_find_unused_parameters=False,
        ddp_gradient_as_bucket_view=True,
    )

    trainer = PerformanceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Starting training...")
    print(f"Training samples: {len(train_ds)}")
    print(f"Evaluation samples: {len(eval_ds)}")
    print(f"Total steps per epoch: {len(train_ds) // (args.batch_size * args.grad_accum)}")

    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n💾 Saving model...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "=" * 70)
    print(f"✅ Training complete! Model saved to: {args.output}")
    print("=" * 70)

    if eval_ds:
        print("\n📈 Final Evaluation:")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        print(f"\n  Final eval loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Final eval perplexity: {eval_metrics.get('eval_perplexity', 'N/A'):.2f}")

    if args.hf_repo:
        hf_token = os.getenv("HF_TOKEN")
        repo_id = resolve_hf_repo(args.hf_repo, args.model_name)
        if not repo_id:
            print("\n⚠️  Invalid HuggingFace repo target. Skipping upload.")
            print("   Use format: user/model or https://huggingface.co/user[/model]")
        elif hf_token:
            print(f"\n📤 Uploading model to HuggingFace Hub: {repo_id}")
            try:
                api = HfApi(token=hf_token)
                create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
                api.upload_folder(
                    folder_path=args.output,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message=f"Upload {args.model_name} fine-tuned model"
                )
                print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"⚠️  HuggingFace upload failed: {e}")
                print("   Model saved locally but not uploaded to HuggingFace Hub")
        else:
            print("\n⚠️  HF_TOKEN not set. Skipping HuggingFace upload.")
            print("   Set HF_TOKEN environment variable to enable upload.")

    print("\nTo use the fine-tuned model:")
    print(f"  python main.py --model-dir {args.output}")


if __name__ == "__main__":
    main()
