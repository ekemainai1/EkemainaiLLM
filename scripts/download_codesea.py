#!/usr/bin/env python3
"""Download CodeSearchNet with HF_TOKEN."""
import json
import os
import sys

def convert_codesea_sample(item, language="python"):
    """Convert CodeSearchNet sample to instruction format."""
    code = item.get("code", "")
    doc = item.get("docstring", "") or item.get("english_documentation", "")
    
    if not code or not doc:
        return None
    
    return {
        "instruction": f"Explain this {language} code with its documentation.",
        "input": code[:600],
        "output": doc[:400],
    }


def download_codesea(language="python", limit=50000):
    """Download CodeSearchNet dataset."""
    from datasets import load_dataset
    
    print(f"Loading CodeSearchNet ({language})...")
    
    try:
        # Use streaming for large dataset
        ds = load_dataset(
            "code-search-net/code_search_net",
            language,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        samples = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            
            sample = convert_codesea_sample(item, language)
            if sample:
                samples.append(sample)
            
            if (i + 1) % 5000 == 0:
                print(f"  Progress: {i+1}/{limit}")
        
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
    parser.add_argument("--language", default="python")
    parser.add_argument("--limit", type=int, default=50000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    samples = download_codesea(args.language, args.limit)
    
    if samples:
        out_path = f"{args.output_dir}/codesea_{args.language}.jsonl"
        count = save_jsonl(samples, out_path)
        print(f"Saved: {out_path} ({count})")


if __name__ == "__main__":
    main()