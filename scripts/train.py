#!/usr/bin/env python3
"""Train Mistral 7B with QLoRA on ROCm for coding assistant with reasoning and tool-use."""
import os
import sys
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune coding assistant with QLoRA")
    parser.add_argument(
        "--model", 
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Path to training JSONL dataset (e.g., data/cot_final.jsonl or data/reasoning_final.jsonl)"
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
        help="Maximum sequence length (increase for CoT datasets with longer outputs)"
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
        help="Prompt format: inst=Mistral [INST], chat=ChatML, cot=Chain-of-Thought"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA)"
    )
    return parser.parse_args()


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
    
    result = tokenizer(
        prompt,
        max_length=max_seq,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def load_tokenize_dataset(dataset_path, tokenizer, max_seq, seed=42, prompt_style="inst"):
    print(f"Loading dataset from {dataset_path}...")
    
    if dataset_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_path, split="train")
    else:
        ds = load_dataset(dataset_path, split="train")
    
    print(f"Dataset size: {len(ds)} samples")
    
    # Detect dataset type for logging
    dataset_name = Path(dataset_path).name
    if "cot" in dataset_name:
        print("Detected: Chain-of-Thought dataset")
    elif "reasoning" in dataset_name:
        print("Detected: Reasoning + Tools dataset")
    else:
        print("Detected: Standard dataset")
    
    def wrapped(s):
        return format_sample(s, tokenizer, max_seq, prompt_style)
    
    ds = ds.map(wrapped, remove_columns=ds.column_names, desc="Tokenizing")
    
    # Filter out empty samples
    ds = ds.filter(
        lambda x: x["labels"][0] != tokenizer.pad_token_id,
        desc="Filtering empty samples"
    )
    
    ds = ds.shuffle(seed=seed)
    
    print(f"After filtering: {len(ds)} samples")
    return ds.train_test_split(test_size=0.05)


def setup_model(model_name, rank, alpha, device, use_4bit=True):
    print(f"Loading model: {model_name}")
    
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
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("Codebase Intelligence Assistant - QLoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence: {args.max_seq}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")
    print(f"Prompt style: {args.prompt_style}")
    print("=" * 60)
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Setup model
    model = setup_model(args.model, args.rank, args.alpha, device, args.use_4bit)
    
    # Load and tokenize dataset
    train_ds, eval_ds = load_tokenize_dataset(
        args.dataset, 
        tokenizer, 
        args.max_seq, 
        args.seed,
        args.prompt_style
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=bool(device == "cuda"),
        bf16=bool(device == "cuda"),
        optim="paged_adamw_8bit",
        group_by_length=True,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving model...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to: {args.output}")
    print("=" * 60)
    print("\nTo use the fine-tuned model:")
    print(f"  python scripts/train.py --model {args.model} --dataset {args.output}")


if __name__ == "__main__":
    main()