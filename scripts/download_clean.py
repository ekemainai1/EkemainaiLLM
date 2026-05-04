#!/usr/bin/env python3
"""Download datasets without torch errors - use datasets without streaming."""
import json
import os
import subprocess

# Workaround: Avoid torch import by using subprocess
def download_clean(ds_id, output_file, limit=2000):
    """Download using subprocess to avoid torch init."""
    code = f'''
import datasets
ds = datasets.load_dataset("{ds_id}", split="train")
samples = []
for i, item in enumerate(ds):
    if i >= {limit}: break
    # Try to convert
    instruct = item.get("instruction") or item.get("prompt") or item.get("question", "")
    inp = item.get("input") or item.get("text", "") or ""
    out = item.get("output") or item.get("completion") or item.get("solution", "") or ""
    if instruct and out:
        samples.append({{"instruction": instruct[:200], "input": inp[:500], "output": out[:500]}})

import json
with open("{output_file}", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\\n")
print(f"Saved {{len(samples)}} samples")
'''
    
    # Write temp script
    with open("/tmp/download_ds.py", "w") as f:
        f.write(code)
    
    # Run in subprocess
    result = subprocess.run(
        ["python3", "/tmp/download_ds.py"],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:200])


# Try datasets without streaming (load small amount first)
DATASETS = [
    ("newfacade/LeetCodeDataset", "data/leetcodesmall2.jsonl", 2000),
    ("jeffmeloy/python_documentation_code", "data/python_doc.jsonl", 2000),
    ("loubnabnl/code-generations-bigcode", "data/bigcode_gen.jsonl", 2000),
]

for ds_id, out_file, limit in DATASETS:
    if os.path.exists(out_file):
        print(f"Skipping {ds_id} (exists)")
        continue
    print(f"\nDownloading {ds_id}...")
    try:
        download_clean(ds_id, out_file, limit)
    except Exception as e:
        print(f"Failed: {e}")