#!/usr/bin/env python3
"""Evaluate fine-tuned model on benchmarks with pass@k metrics."""
import os
import sys
import json
import argparse
import time
from pathlib import Path
import torch
import subprocess
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import random

DEFAULT_PROMPT = "[INST] {instruction}\n{input} [/INST]\n"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model with pass@k metrics")
    parser.add_argument("--model", default="./fine-tuned-model", help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model name")
    parser.add_argument("--test-set", default="data/test_set.jsonl", help="Test JSONL file")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "custom", "all"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=164, help="Num samples for HumanEval (default: all)")
    parser.add_argument("--pass-at-k", type=int, default=10, help="k for pass@k calculation")
    parser.add_argument("--num-generations", type=int, default=10, help="Number of generations per prompt for pass@k")
    parser.add_argument("--compare-base", action="store_true", help="Compare with base model")
    parser.add_argument("--output", default="data/evaluation_results.json", help="Output file")
    return parser.parse_args()


def extract_code(text, language="python"):
    lang_tags = {
        "python": ["```python", "```py", "```"],
        "kotlin": ["```kotlin", "```kt", "```"],
    }

    for tag in lang_tags.get(language, ["```"]):
        if tag in text:
            start = text.find(tag) + len(tag)
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

    lines = text.split("\n")
    code_lines = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
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
            repetition_penalty=1.1,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_multiple(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device, num_generations):
    """Generate multiple samples for pass@k calculation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    results = []
    with torch.no_grad():
        for _ in range(num_generations):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return results


def run_python_code(code):
    """Execute Python code and return success status."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ['python3', f.name],
                capture_output=True,
                text=True,
                timeout=10
            )
            os.unlink(f.name)
            return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


def calculate_pass_at_k(results, k):
    """Calculate pass@k metric."""
    n = len(results)
    if n == 0:
        return 0.0

    c = sum(1 for r in results if r["passed"])
    if c == 0:
        return 0.0

    return c / n


def evaluate_test_set(model, tokenizer, test_path, max_new_tokens, temperature, top_p, device, num_samples=100):
    results = []
    with open(test_path) as f:
        samples = [json.loads(line) for line in f][:num_samples]

    print(f"Evaluating {len(samples)} custom test samples...")

    for sample in tqdm(samples, desc="Custom Eval"):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        expected = sample.get("output", "")

        prompt = DEFAULT_PROMPT.format(instruction=instruction, input=input_text)
        generated = generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device)

        generated_text = generated[len(prompt):]
        code = extract_code(generated_text)

        result = {
            "instruction": instruction[:50],
            "generated": generated_text[:200],
            "extracted_code": code[:200] if code else "",
            "has_code": bool(code),
        }

        if code:
            passed, error = run_python_code(code)
            result["passed"] = passed
            result["error"] = error[:100] if error else ""
        else:
            result["passed"] = False

        results.append(result)

    pass_rate = sum(1 for r in results if r.get("passed", False)) / len(results) * 100
    print(f"Custom test pass rate: {pass_rate:.1f}% ({sum(1 for r in results if r.get('passed', False))}/{len(results)})")

    return results


def run_humaneval(model, tokenizer, max_new_tokens, temperature, top_p, device, num_samples, pass_at_k=10):
    """Run HumanEval benchmark with pass@k metrics."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        print(f"Failed to load HumanEval: {e}")
        return []

    print(f"\n📊 Running HumanEval benchmark (pass@{pass_at_k})...")

    results = []
    sample_count = min(num_samples, len(ds))

    for idx, item in enumerate(tqdm(ds.select(range(sample_count)), desc="HumanEval")):
        prompt_text = item["prompt"]
        test_case = item["test"]

        full_prompt = DEFAULT_PROMPT.format(
            instruction="Complete the function. Write only the Python code, no explanation.",
            input=prompt_text
        )

        generated_samples = generate_multiple(
            model, tokenizer, full_prompt, max_new_tokens, temperature, top_p, device, pass_at_k
        )

        problem_results = []
        for gen in generated_samples:
            code = extract_code(gen)
            test_passed = False

            if code:
                full_code = code + "\n" + test_case
                test_passed, _ = run_python_code(full_code)

            problem_results.append({
                "code": code[:200] if code else "",
                "passed": test_passed
            })

        passed = any(r["passed"] for r in problem_results)
        results.append({
            "task_id": item["task_id"],
            "prompt": prompt_text[:50],
            "num_passed": sum(1 for r in problem_results if r["passed"]),
            "total": pass_at_k,
            "passed": passed,
            "problem_results": problem_results[:3]
        })

    pass_at_k_result = calculate_pass_at_k(results, pass_at_k)
    total_passed = sum(1 for r in results if r["passed"])

    print(f"\n📈 HumanEval Results:")
    print(f"  Pass@{pass_at_k}: {pass_at_k_result*100:.2f}%")
    print(f"  Total passed: {total_passed}/{len(results)}")
    print(f"  Pass@1: {total_passed/len(results)*100:.2f}%")

    return results


def run_mbpp(model, tokenizer, max_new_tokens, temperature, top_p, device, num_samples, pass_at_k=10):
    """Run MBPP benchmark with pass@k metrics."""
    try:
        ds = load_dataset("scooper16/mbpp", split="test")
    except Exception as e:
        print(f"Failed to load MBPP: {e}")
        return []

    print(f"\n📊 Running MBPP benchmark (pass@{pass_at_k})...")

    results = []
    sample_count = min(num_samples, len(ds))

    for idx, item in enumerate(tqdm(ds.select(range(sample_count)), desc="MBPP")):
        text = item["text"]
        test_case = item["test"]

        full_prompt = DEFAULT_PROMPT.format(
            instruction="Write a Python function to solve this problem. Write only the code, no explanation.",
            input=text
        )

        generated_samples = generate_multiple(
            model, tokenizer, full_prompt, max_new_tokens, temperature, top_p, device, pass_at_k
        )

        problem_results = []
        for gen in generated_samples:
            code = extract_code(gen)
            test_passed = False

            if code:
                full_code = code + "\n" + test_case
                test_passed, _ = run_python_code(full_code)

            problem_results.append({
                "code": code[:200] if code else "",
                "passed": test_passed
            })

        passed = any(r["passed"] for r in problem_results)
        results.append({
            "text": text[:50],
            "num_passed": sum(1 for r in problem_results if r["passed"]),
            "total": pass_at_k,
            "passed": passed,
            "problem_results": problem_results[:3]
        })

    pass_at_k_result = calculate_pass_at_k(results, pass_at_k)
    total_passed = sum(1 for r in results if r["passed"])

    print(f"\n📈 MBPP Results:")
    print(f"  Pass@{pass_at_k}: {pass_at_k_result*100:.2f}%")
    print(f"  Total passed: {total_passed}/{len(results)}")
    print(f"  Pass@1: {total_passed/len(results)*100:.2f}%")

    return results


def load_model(model_path, base_model_name, device):
    """Load model with PEFT adapter support."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
    else:
        model = base

    return model, tokenizer


def main():
    args = parse_args()
    device = get_device()

    print("=" * 70)
    print("EkemainaiAgent - Model Evaluation with pass@k metrics")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Base model: {args.base_model}")
    print(f"Benchmarks: {args.benchmark}")
    print(f"pass@{args.pass_at_k} with {args.num_generations} generations per prompt")
    print(f"Compare with base: {args.compare_base}")
    print("=" * 70)

    all_results = {}

    model_suffix = ""
    if args.compare_base:
        print("\n🔄 Evaluating BASE model first...")
        base_model, base_tokenizer = load_model(args.model, args.base_model, device)

        base_results = {}
        if args.benchmark in ("humaneval", "all"):
            base_results["humaneval"] = run_humaneval(
                base_model, base_tokenizer, args.max_new_tokens,
                args.temperature, args.top_p, device, args.num_samples, args.pass_at_k
            )
        if args.benchmark in ("mbpp", "all"):
            base_results["mbpp"] = run_mbpp(
                base_model, base_tokenizer, args.max_new_tokens,
                args.temperature, args.top_p, device, args.num_samples, args.pass_at_k
            )
        if args.benchmark in ("custom", "all") and os.path.exists(args.test_set):
            base_results["custom"] = evaluate_test_set(
                base_model, base_tokenizer, args.test_set, args.max_new_tokens,
                args.temperature, args.top_p, device, args.num_samples
            )

        all_results["base_model"] = base_results
        model_suffix = "_fine_tuned"
        del base_model
        torch.cuda.empty_cache()

    print("\n🔄 Evaluating FINE-TUNED model...")
    model, tokenizer = load_model(args.model, args.base_model, device)

    if args.benchmark in ("humaneval", "all"):
        all_results[f"humaneval{model_suffix}"] = run_humaneval(
            model, tokenizer, args.max_new_tokens,
            args.temperature, args.top_p, device, args.num_samples, args.pass_at_k
        )
    if args.benchmark in ("mbpp", "all"):
        all_results[f"mbpp{model_suffix}"] = run_mbpp(
            model, tokenizer, args.max_new_tokens,
            args.temperature, args.top_p, device, args.num_samples, args.pass_at_k
        )
    if args.benchmark in ("custom", "all") and os.path.exists(args.test_set):
        all_results[f"custom{model_suffix}"] = evaluate_test_set(
            model, tokenizer, args.test_set, args.max_new_tokens,
            args.temperature, args.top_p, device, args.num_samples
        )

    if args.compare_base:
        print("\n" + "=" * 70)
        print("📊 COMPARISON SUMMARY")
        print("=" * 70)

        for benchmark in ["humaneval", "mbpp", "custom"]:
            base_key = benchmark
            ft_key = f"{benchmark}_fine_tuned"

            if base_key in all_results and ft_key in all_results:
                base_passed = sum(1 for r in all_results[base_key] if r.get("passed", False))
                ft_passed = sum(1 for r in all_results[ft_key] if r.get("passed", False))
                total = len(all_results[base_key])

                base_rate = base_passed / total * 100
                ft_rate = ft_passed / total * 100
                improvement = ft_rate - base_rate

                print(f"\n{benchmark.upper()}:")
                print(f"  Base model:      {base_rate:.2f}% ({base_passed}/{total})")
                print(f"  Fine-tuned:      {ft_rate:.2f}% ({ft_passed}/{total})")
                print(f"  Improvement:     {improvement:+.2f}%")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ Results saved to {args.output}")

    print("\n" + "=" * 70)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 70)

    for key, value in all_results.items():
        if isinstance(value, list) and len(value) > 0:
            passed = sum(1 for r in value if r.get("passed", False))
            total = len(value)
            rate = passed / total * 100
            print(f"{key}: {rate:.2f}% ({passed}/{total})")


if __name__ == "__main__":
    main()