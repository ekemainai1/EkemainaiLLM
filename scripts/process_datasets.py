#!/usr/bin/env python3
"""Process datasets into unified instruction-tuning format for training."""
import os
import sys
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

SUPPORTED_LANGS = {".py", ".js", ".ts", ".java", ".kt", ".cpp", ".xml"}
MAX_LINES = 40


def chunk_code(code, max_lines=MAX_LINES):
    lines = code.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        if len(chunk.strip()) > 20:
            yield chunk


def convert_apps_sample(item):
    samples = []
    question = item.get("question", "")
    solutions = item.get("solutions", [])
    for sol in solutions[:1]:
        samples.append({
            "instruction": "Solve this coding problem. Write the complete Python solution.",
            "input": question,
            "output": sol if isinstance(sol, str) else str(sol),
        })
    starter_code = item.get("starter_code", "")
    if starter_code:
        samples.append({
            "instruction": "Complete this coding problem using the starter code provided.",
            "input": f"Problem:\n{question}\n\nStarter code:\n{starter_code}",
            "output": "",
        })
    return samples


def convert_codesearchnet(item):
    code = item.get("func_documentation_string", "") or item.get("code", "")
    doc = item.get("english_documentation", "")
    if not code or not doc:
        return []
    return [{
        "instruction": "Explain this code based on the documentation.",
        "input": f"Documentation:\n{doc}\n\nCode:\n{code}",
        "output": "",
    }]


def convert_codealpaca(item):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    if not instruction:
        return []
    return [{
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }]


def convert_oss_instruct(item):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    if not instruction:
        return []
    return [{
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }]


def convert_humaneval(item):
    prompt = item.get("prompt", "")
    canonical = item.get("canonical_solution", "")
    if not prompt or not canonical:
        return []
    return [{
        "instruction": "Complete the function. Write the complete Python code.",
        "input": prompt,
        "output": canonical,
    }]


def convert_mbpp(item):
    text = item.get("text", "")
    code = item.get("code", "")
    if not text or not code:
        return []
    return [{
        "instruction": "Write the Python function to solve this problem.",
        "input": text,
        "output": code,
    }]


def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def save_jsonl(dataset, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_path}")


def merge_and_process(input_files, output_file, sample_limit=None):
    all_samples = []
    for f in input_files:
        if f.endswith(".jsonl"):
            items = load_jsonl(f)
        elif "apps" in f.lower():
            try:
                ds = load_dataset("princeton-nlp/APPS", split="train")
                items = [{"question": x["question"], "solutions": x["solutions"], "starter_code": x.get("starter_code", "")}
                        for x in ds]
            except Exception:
                items = []
        else:
            continue
        for item in items:
            samples = []
            if "apps" in f.lower():
                samples = convert_apps_sample(item)
            elif "codesearchnet" in f.lower():
                samples = convert_codesearchnet(item)
            elif "alpaca" in f.lower() or "oss" in f.lower():
                samples = convert_codealpaca(item)
            else:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                if instruction:
                    samples = [{"instruction": instruction, "input": input_text, "output": output}]
            all_samples.extend(samples)
    seen = set()
    deduped = []
    for s in all_samples:
        key = s["instruction"][:50] + s["input"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    if sample_limit:
        deduped = deduped[:sample_limit]
    save_jsonl(deduped, output_file)
    return len(deduped)


def download_core_datasets(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    downloads = []
    print("Downloading datasets...")
    try:
        print("  APPS dataset...")
        apps = load_dataset("princeton-nlp/APPS", split="train[:50000]")
        apps_path = f"{data_dir}/apps_50k.arrow"
        apps.to_parquet(apps_path)
        downloads.append(f"Downloaded APPS to {apps_path}")
    except Exception as e:
        print(f"  Failed to download APPS: {e}")
    try:
        print("  CodeSearchNet dataset...")
        csn = load_dataset("code_search_net", split="train[:50000]")
        csn.to_parquet(f"{data_dir}/codesearchnet.arrow")
        downloads.append("Downloaded CodeSearchNet")
    except Exception as e:
        print(f"  Failed to download CodeSearchNet: {e}")
    try:
        print("  CodeAlpaca dataset...")
        ca = load_dataset("HuggingFaceH4/code_alpaca", split="train")
        ca.to_parquet(f"{data_dir}/codealpaca.arrow")
        downloads.append("Downloaded CodeAlpaca")
    except Exception as e:
        print(f"  Failed to download CodeAlpaca: {e}")
    return downloads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets for training")
    parser.add_argument("--input", nargs="+", help="Input JSONL or dataset paths")
    parser.add_argument("--output", default="data/final_training.jsonl")
    parser.add_argument("--sample-limit", type=int, help="Limit total samples")
    parser.add_argument("--download", action="store_true", help="Download core datasets")
    args = parser.parse_args()
    if args.download:
        download_core_datasets()
    elif args.input:
        total = merge_and_process(args.input, args.output, args.sample_limit)
        print(f"Done! {total} samples processed.")
    else:
        parser.print_help()