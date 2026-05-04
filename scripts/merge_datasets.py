#!/usr/bin/env python3
"""Merge multiple datasets into a single training file for fine-tuning."""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "combined_final.jsonl"

DATASET_GROUPS = {
    "core": {
        "apps_train": "data/apps_train.jsonl",
        "codealpaca": "data/codealpaca.jsonl",
        "humaneval": "data/humaneval.jsonl",
        "mbpp": "data/mbpp.jsonl",
    },
    "synthetic": {
        "synthetic": "data/synthetic.jsonl",
        "synthetic2": "data/synthetic2.jsonl",
        "synthetic3": "data/synthetic3.jsonl",
        "synthetic4": "data/synthetic4.jsonl",
    },
    "github": {
        "github_code": "data/github_code.jsonl",
        "more_repos": "data/more_repos.jsonl",
        "more_repos2": "data/more_repos2.jsonl",
        "flask_samples": "data/flask_samples.jsonl",
        "gunicorn_samples": "data/gunicorn_samples.jsonl",
        "github_discussions": "data/github_discussions.jsonl",
    },
    "extra": {
        "python": "data/python.jsonl",
        "software_eng": "data/software_eng.jsonl",
        "leetcodesmall": "data/leetcodesmall.jsonl",
        "leetcodesmall2": "data/leetcodesmall2.jsonl",
        "mbpp_extra": "data/mbpp_extra.jsonl",
        "ossinstruct": "data/ossinstruct.jsonl",
    },
    "codesearchnet": {
        "codesearchnet1": "data/codesearchnet1.jsonl",
        "codesearchnet2": "data/codesearchnet2.jsonl",
    },
}


def load_jsonl(path: Path) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if sample.get("instruction") or sample.get("prompt"):
                    samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error in {path}:{line_num} - {e}")
    return samples


def save_jsonl(samples: List[Dict], path: Path) -> int:
    """Save samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def deduplicate(samples: List[Dict]) -> List[Dict]:
    """Remove duplicate samples based on instruction+input keys."""
    seen: Set[str] = set()
    unique = []
    for sample in samples:
        # Use full instruction + first 200 chars of input for better deduplication
        key = (sample.get("instruction", "") + 
              sample.get("input", "")[:200])
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    return unique


def validate_sample(sample: Dict) -> bool:
    """Validate that a sample has required fields."""
    if not sample.get("instruction") and not sample.get("prompt"):
        return False
    return True


def merge_datasets(
    dataset_keys: List[str],
    sample_limit: int = None,
    deduplicate_samples: bool = True,
) -> List[Dict]:
    """Merge multiple datasets."""
    all_samples = []
    
    for key in dataset_keys:
        path = DATA_DIR / f"{key}.jsonl"
        
        # Also check direct path
        if not path.exists():
            path = Path(key)
        if not path.exists():
            path = Path("data") / Path(key).name
        
        if not path.exists():
            print(f"  Warning: {key} not found, skipping")
            continue
        
        print(f"Loading {key}...")
        samples = load_jsonl(path)
        
        # Normalize to instruction/input/output format
        normalized = []
        for s in samples:
            if not validate_sample(s):
                continue
            
            # Handle different formats
            if "prompt" in s and "completion" in s:
                normalized.append({
                    "instruction": s.get("instruction", "") or s.get("prompt", "").split("\n")[0][:200],
                    "input": s.get("prompt", "")[s.get("prompt", "").find("\n")+1:].strip() if "\n" in s.get("prompt", "") else s.get("prompt", ""),
                    "output": s.get("completion", ""),
                })
            elif "text" in s and "code" in s:
                normalized.append({
                    "instruction": s.get("instruction", "Write the Python function."),
                    "input": s.get("text", ""),
                    "output": s.get("code", ""),
                })
            elif "question" in s and "solutions" in s:
                normalized.append({
                    "instruction": s.get("instruction", "Solve this coding problem."),
                    "input": s.get("question", ""),
                    "output": s.get("solutions", ""),
                })
            else:
                # Already in standard format
                normalized.append({
                    "instruction": s.get("instruction", ""),
                    "input": s.get("input", ""),
                    "output": s.get("output", ""),
                })
        
        print(f"  -> {len(normalized)} samples")
        all_samples.extend(normalized)
    
    print(f"\nTotal before processing: {len(all_samples)} samples")
    
    # Filter valid samples
    all_samples = [s for s in all_samples if s.get("instruction") or s.get("output")]
    print(f"Total after filtering: {len(all_samples)} samples")
    
    # Deduplicate
    if deduplicate_samples:
        print("Deduplicating...")
        all_samples = deduplicate(all_samples)
        print(f"Total after deduplication: {len(all_samples)} samples")
    
    # Apply limit
    if sample_limit and len(all_samples) > sample_limit:
        print(f"Limiting to {sample_limit} samples...")
        all_samples = all_samples[:sample_limit]
    
    return all_samples


def list_available_datasets() -> None:
    """List all available datasets in the data directory."""
    print("Available datasets in data/:")
    print()
    
    # List files
    files = list(DATA_DIR.glob("*.jsonl"))
    print(f"Total JSONL files: {len(files)}")
    print()
    
    for group_name, datasets in DATASET_GROUPS.items():
        print(f"\n{group_name.upper()}:")
        for name, path in datasets.items():
            full_path = DATA_DIR / f"{name}.jsonl"
            if full_path.exists():
                count = sum(1 for _ in open(full_path))
                print(f"  {name}: {count:,} samples ({path})")
            else:
                print(f"  {name}: MISSING")


def main():
    parser = argparse.ArgumentParser(
        description="Merge datasets into combined_final.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python scripts/merge_datasets.py --list
  
  # Merge using group names
  python scripts/merge_datasets.py --groups core synthetic
  
  # Merge specific datasets
  python scripts/merge_datasets.py --datasets apps_train codealpaca humaneval mbpp
  
  # Merge all datasets
  python scripts/merge_datasets.py --all
  
  # Merge with sample limit
  python scripts/merge_datasets.py --all --limit 100000
        """
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=list(DATASET_GROUPS.keys()),
        help="Dataset groups to merge (core, synthetic, github, extra)"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Specific datasets to merge (without .jsonl extension)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Merge all available datasets"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit total samples"
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--output", default="data/combined_final.jsonl",
        help="Output file path"
    )
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return
    
    # Collect dataset keys
    dataset_keys = []
    
    if args.all:
        # Get all available datasets
        for group in DATASET_GROUPS.values():
            dataset_keys.extend(group.keys())
    elif args.groups:
        for group_name in args.groups:
            dataset_keys.extend(DATASET_GROUPS[group_name].keys())
    elif args.datasets:
        dataset_keys = args.datasets
    else:
        # Default: merge core + synthetic + github
        print("No datasets specified, using default: core + synthetic + github")
        for group_name in ["core", "synthetic", "github"]:
            dataset_keys.extend(DATASET_GROUPS[group_name].keys())
    
    print(f"\nDatasets to merge: {dataset_keys}\n")
    
    # Merge
    samples = merge_datasets(
        dataset_keys,
        sample_limit=args.limit,
        deduplicate_samples=not args.no_dedup,
    )
    
    # Save
    output_path = Path(args.output)
    count = save_jsonl(samples, output_path)
    
    print(f"\nSaved {count:,} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()