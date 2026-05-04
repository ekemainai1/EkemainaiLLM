#!/usr/bin/env python3
"""Convert APPS dataset to instruction format."""
import json
import os
import sys

def convert_apps_sample(item):
    """Convert APPS sample to instruction format."""
    question = item.get("question", "")
    solutions = item.get("solutions", "")
    difficulty = item.get("difficulty", "")
    
    # Parse solutions
    try:
        solutions_list = json.loads(solutions) if isinstance(solutions, str) else solutions
    except:
        solutions_list = [solutions]
    
    first_solution = solutions_list[0] if solutions_list else ""
    
    if not question or not first_solution:
        return None
    
    return {
        "instruction": f"Solve this coding problem (Difficulty: {difficulty}).",
        "input": question[:800],
        "output": first_solution[:800],
    }


def process_apps_jsonl(input_path, output_path, limit=None):
    """Process APPS JSONL file."""
    count = 0
    processed = 0
    
    with open(input_path, "r") as f:
        for line in f:
            if limit and count >= limit:
                break
            
            try:
                item = json.loads(line)
                sample = convert_apps_sample(item)
                
                if sample:
                    with open(output_path, "a") as o:
                        o.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    processed += 1
                
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} items, {processed} samples")
            except json.JSONDecodeError:
                continue
    
    return processed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/var/folders/32/8f2s5_1d3nz8vzvdz345fttc0000gn/T/apps_dataset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process train split
    train_input = f"{args.input_dir}/train.jsonl"
    train_output = f"{args.output_dir}/apps_train.jsonl"
    
    if os.path.exists(train_input):
        print(f"Processing {train_input}...")
        count = process_apps_jsonl(train_input, train_output, args.limit)
        print(f"Saved: {train_output} ({count} samples)")
    
    # Process test split
    test_input = f"{args.input_dir}/test.jsonl"
    test_output = f"{args.output_dir}/apps_test.jsonl"
    
    if os.path.exists(test_input):
        print(f"Processing {test_input}...")
        count = process_apps_jsonl(test_input, test_output, args.limit)
        print(f"Saved: {test_output} ({count} samples)")
    
    # Merge into combined
    print("Merging into combined dataset...")
    combined = []
    for fpath in [train_output, test_output]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    combined.append(line)
    
    combined_path = f"{args.output_dir}/combined.jsonl"
    with open(combined_path, "w") as f:
        f.writelines(combined)
    
    print(f"Total combined: {len(combined)} samples -> {combined_path}")


if __name__ == "__main__":
    main()