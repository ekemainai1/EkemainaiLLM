#!/usr/bin/env python3
"""Download more code datasets."""
import json
import os

DATASETS = [
    {"name": "leetcodesmall", "id": "newfacade/LeetCodeDataset", "split": "train", "limit": 1000},
    {"name": "leetcostandalone", "id": "ziwenyd/leetcode-standalone", "split": "train", "limit": 1000},
    {"name": "scicode", "id": "SciCode/SciCode-Programming-Problems", "split": "train", "limit": 1000},
    {"name": "coding_challenge", "id": "Leskille/coding_challenge", "split": "train", "limit": 500},
    {"name": "codegen_token", "id": "google/code_x_glue_cc_code_completion_token", "split": "train", "limit": 1000},
    {"name": "coderefine_java", "id": "semeru/code-code-CodeRefinement-Java-Medium", "split": "train", "limit": 500},
    {"name": "software_eng", "id": "JuanjoLopez19/Software-Engineering-Dataset_90_10", "split": "train", "limit": 500},
]


def convert_sample(ds_name, item):
    """Convert to instruction format."""
    ds_name = ds_name.lower()
    
    # Try common fields
    instruction = item.get("instruction", "") or item.get("prompt", "") or item.get("question", "")
    input_text = item.get("input", "") or item.get("text", "") or item.get("code", "")
    output = item.get("output", "") or item.get("solution", "") or item.get("completion", "") or item.get("answer", "")
    
    if instruction and output:
        return {"instruction": instruction[:200], "input": input_text[:500], "output": output[:500]}
    return None


def download(ds_config):
    from datasets import load_dataset
    
    name = ds_config["name"]
    ds_id = ds_config["id"]
    split = ds_config["split"]
    limit = ds_config["limit"]
    
    print(f"Loading {name} ({ds_id})...")
    
    try:
        ds = load_dataset(ds_id, split=split, streaming=True)
        
        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            
            sample = convert_sample(name, item)
            if sample:
                samples.append(sample)
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}")
        
        print(f"  -> {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    
    all_samples = []
    
    for ds in DATASETS:
        samples = download(ds)
        all_samples.extend(samples)
        
        if samples:
            out_path = f"{args.output_dir}/{ds['name']}.jsonl"
            count = save_jsonl(samples, out_path)
            print(f"  Saved: {out_path}")
    
    combined_path = f"{args.output_dir}/combined.jsonl"
    count = save_jsonl(all_samples, combined_path)
    print(f"\nTotal new: {count} -> {combined_path}")


if __name__ == "__main__":
    main()