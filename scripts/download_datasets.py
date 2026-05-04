#!/usr/bin/env python3
"""Download and process core datasets for training."""
import json
import os

DATASETS_CONFIG = [
    {"name": "codealpaca", "id": "HuggingFaceH4/CodeAlpaca_20K", "split": "train", "limit": 20000},
    {"name": "humaneval", "id": "openai/openai_humaneval", "split": "test", "limit": 164},
    {"name": "mbpp", "id": "google-research-datasets/mbpp", "split": "test", "limit": 500},
    {"name": "python", "id": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "limit": 18000},
]


def convert_sample(ds_name, item):
    """Convert dataset item to instruction format."""
    ds_name = ds_name.lower()
    
    if ds_name == "codealpaca":
        return {
            "instruction": item.get("prompt", "").split("\n")[0][:200] if item.get("prompt") else "Complete this coding task.",
            "input": item.get("prompt", "")[item.get("prompt", "").find("\n")+1:].strip()[:500] if "\n" in item.get("prompt", "") else item.get("prompt", "")[:500],
            "output": item.get("completion", "")[:500],
        }
    
    elif ds_name == "humaneval":
        return {
            "instruction": "Complete the function.",
            "input": item.get("prompt", "")[:500],
            "output": item.get("canonical_solution", "")[:500],
        }
    
    elif ds_name == "mbpp":
        return {
            "instruction": "Write the Python function.",
            "input": item.get("text", "")[:500],
            "output": item.get("code", "")[:500],
        }
    
    elif ds_name == "python":
        return {
            "instruction": item.get("instruction", "")[:200],
            "input": item.get("input", "")[:500],
            "output": item.get("output", "")[:500],
        }
    
    return None


def download_dataset(ds_config):
    """Download a single dataset."""
    from datasets import load_dataset
    
    name = ds_config["name"]
    ds_id = ds_config["id"]
    split = ds_config["split"]
    limit = ds_config["limit"]
    
    print(f"Loading {name} ({ds_id})...")
    
    try:
        ds = load_dataset(ds_id, split=split)
        
        samples = []
        for item in ds:
            sample = convert_sample(name, item)
            if sample and sample.get("instruction"):
                samples.append(sample)
            if len(samples) >= limit:
                break
        
        print(f"  -> {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"  -> Failed: {e}")
        return []


def save_jsonl(data, path):
    """Save to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_samples = []
    
    for ds_config in DATASETS_CONFIG:
        if args.limit:
            ds_config = ds_config.copy()
            ds_config["limit"] = args.limit
        
        samples = download_dataset(ds_config)
        all_samples.extend(samples)
        
        if samples:
            out_path = f"{args.output_dir}/{ds_config['name']}.jsonl"
            count = save_jsonl(samples, out_path)
            print(f"  Saved: {out_path} ({count})")
    
    combined_path = f"{args.output_dir}/combined.jsonl"
    count = save_jsonl(all_samples, combined_path)
    print(f"\nTotal: {count} samples -> {combined_path}")
    print(f"Breakdown:")
    for name in sorted(set(s["instruction"][:30] for s in all_samples[:1000])):
        cnt = sum(1 for s in all_samples if s["instruction"][:30] == name)
        print(f"  {name[:30]}: {cnt}")


if __name__ == "__main__":
    main()