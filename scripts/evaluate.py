#!/usr/bin/env python3
"""Evaluate fine-tuned model on benchmarks (HumanEval, MBPP, custom tests)."""
import os
import sys
import json
import argparse
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_PROMPT = "[INST] {instruction}\n{input} [/INST]\n"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./fine-tuned-model", help="Path to fine-tuned model or base model")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruction-v0.3", help="Base model name")
    parser.add_argument("--test-set", default="data/test_set.jsonl", help="Test JSONL file")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "custom", "all"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=10, help="Num samples for quick eval")
    return parser.parse_args()


def extract_code(text):
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```kotlin" in text:
        start = text.find("```kotlin") + 10
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            code_lines.append(line)
    return "\n".join(code_lines[:100])


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_test_set(model, tokenizer, test_path, max_new_tokens, temperature, top_p, device, num_samples=100):
    results = []
    with open(test_path) as f:
        samples = [json.loads(line) for line in f][:num_samples]
    for sample in tqdm(samples, desc="Evaluating"):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        expected = sample.get("output", "")
        prompt = DEFAULT_PROMPT.format(instruction=instruction, input=input_text)
        generated = generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device)
        result = {
            "instruction": instruction,
            "input": input_text[:100],
            "expected": expected[:100] if expected else "",
            "generated": generated[len(prompt):][:200],
        }
        results.append(result)
    return results


def run_humaneval(model, tokenizer, max_new_tokens, temperature, top_p, device, num_samples):
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        print(f"Failed to load HumanEval: {e}")
        return []
    results = []
    sample_count = min(num_samples, len(ds))
    for item in tqdm(ds.select(range(sample_count))):
        prompt_text = item["prompt"]
        canonical = item["canonical_solution"]
        test_case = item["test"]
        full_prompt = DEFAULT_PROMPT.format(
            instruction="Complete the function. Write the complete Python code.",
            input=prompt_text
        )
        generated = generate(model, tokenizer, full_prompt, max_new_tokens, temperature, top_p, device)
        code = extract_code(generated)
        results.append({
            "prompt": prompt_text[:80],
            "generated_code": code[:200] if code else generated[:200],
            "has_code": bool(code),
        })
    pass_rate = sum(1 for r in results if r["has_code"]) / len(results) * 100 if results else 0
    print(f"HumanEval pass@code: {pass_rate:.1f}% ({sum(1 for r in results if r['has_code'])}/{len(results)})")
    return results


def run_mbpp(model, tokenizer, max_new_tokens, temperature, top_p, device, num_samples):
    try:
        ds = load_dataset("scooper16/mbpp", split="test")
    except Exception as e:
        print(f"Failed to load MBPP: {e}")
        return []
    results = []
    sample_count = min(num_samples, len(ds))
    for item in tqdm(ds.select(range(sample_count))):
        text = item["text"]
        full_prompt = DEFAULT_PROMPT.format(
            instruction="Write the Python function to solve this problem.",
            input=text
        )
        generated = generate(model, tokenizer, full_prompt, max_new_tokens, temperature, top_p, device)
        code = extract_code(generated)
        results.append({"prompt": text[:80], "generated_code": code[:200] if code else generated[:200], "has_code": bool(code)})
    pass_rate = sum(1 for r in results if r["has_code"]) / len(results) * 100 if results else 0
    print(f"MBPP pass@code: {pass_rate:.1f}% ({sum(1 for r in results if r['has_code'])}/{len(results)})")
    return results


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map=device)
    if os.path.exists(os.path.join(args.model, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, args.model)
        model.eval()
    else:
        model = base
    all_results = {}
    if args.benchmark in ("humaneval", "all"):
        all_results["humaneval"] = run_humaneval(model, tokenizer, args.max_new_tokens, args.temperature, args.top_p, device, args.num_samples)
    if args.benchmark in ("mbpp", "all"):
        all_results["mbpp"] = run_mbpp(model, tokenizer, args.max_new_tokens, args.temperature, args.top_p, device, args.num_samples)
    if args.benchmark in ("custom", "all") and os.path.exists(args.test_set):
        all_results["custom"] = evaluate_test_set(model, tokenizer, args.test_set, args.max_new_tokens, args.temperature, args.top_p, device, args.num_samples)
    results_path = "data/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()