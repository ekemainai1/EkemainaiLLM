#!/usr/bin/env python3
"""Simple download using subprocess - no torch import."""
import subprocess
import sys

code = '''
import pandas as pd
import io
import requests
import json

# Python train parquet
url = "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/refs%2Fconvert%2Fparquet/python/train-00000-of-00001-0.parquet"

print("Downloading...")
resp = requests.get(url, timeout=180)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    # Read parquet
    df = pd.read_parquet(io.BytesIO(resp.content))
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to instruction format
    samples = []
    for i, row in df.head(1000).iterrows():
        code = str(row.get("func_code_string", ""))[:500]
        doc = str(row.get("func_documentation_string", ""))[:300]
        if code and doc and len(code) > 50:
            samples.append({
                "instruction": "Explain this code with its documentation.",
                "input": code,
                "output": doc
            })
    
    # Save
    with open("data/codesea_samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\\n")
    
    print(f"Saved {len(samples)} samples")
else:
    print(f"Failed: {resp.text[:200]}")
'''

# Write to temp file
with open('/tmp/download_codesea.py', 'w') as f:
    f.write(code)

# Run in subprocess
result = subprocess.run([sys.executable, '/tmp/download_codesea.py'], 
                        capture_output=True, text=True, timeout=300)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])