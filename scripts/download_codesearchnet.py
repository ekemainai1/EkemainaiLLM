#!/usr/bin/env python3
"""Download CodeSearchNet dataset."""
import json
import os
import subprocess

DATA_DIR = "/Users/ekeministephen/PycharmProjects/EkemainaiAgent/data"
TEMP_DIR = "/var/folders/32/8f2s5_1d3nz8vzvdz345fttc0000gn/T/opencode/codesearchnet"

LANGUAGES = ["python", "javascript", "java", "go", "php", "ruby"]

BASE_URL = "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main"

def download_with_hf_cli():
    """Use huggingface-cli to download."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    total = 0
    
    for lang in LANGUAGES:
        print(f"Downloading {lang}...")
        for split in ["train", "test", "validation"]:
            url = f"{BASE_URL}/{lang}/{split}-00000-of-00001.parquet"
            dest = os.path.join(TEMP_DIR, f"{lang}_{split}.parquet")
            
            cmd = f"huggingface-cli download {lang} --filename {split}-00000-of-00001.parquet --local-dir {TEMP_DIR}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Downloaded {lang}/{split}")
            else:
                # Try curl
                curl_cmd = f"curl -L {url} -o {dest}"
                r = subprocess.run(curl_cmd.split(), capture_output=True)
                if r.returncode == 0 and os.path.exists(dest):
                    print(f"Downloaded {lang}/{split} via curl")
                    total += 1
                else:
                    print(f"Failed: {lang}/{split}")
    
    return total

def download_with_wget():
    """Try wget for each file."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    for lang in LANGUAGES:
        for split in ["train", "test", "validation"]:
            url = f"{BASE_URL}/{lang}/{split}-00000-of-00001.parquet"
            dest = os.path.join(TEMP_DIR, f"{lang}_{split}.parquet")
            
            if os.path.exists(dest):
                print(f"Already exists: {lang}_{split}")
                continue
            
            print(f"Downloading {lang}/{split}...")
            cmd = f"wget -q -O {dest} {url}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(dest):
                size = os.path.getsize(dest)
                print(f"Downloaded {lang}_{split}: {size/1024/1024:.1f} MB")
            else:
                print(f"Failed: {lang}_{split}")

def parse_parquet():
    """Parse downloaded parquet files."""
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        subprocess.run(["pip3", "install", "pandas", "pyarrow"], check=True)
        import pandas as pd
    
    samples = []
    for lang in LANGUAGES:
        for split in ["train", "test", "validation"]:
            path = os.path.join(TEMP_DIR, f"{lang}_{split}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                print(f"Loaded {lang}/{split}: {len(df)} rows")
                for _, row in df.iterrows():
                    code = row.get('func_code_string', '')
                    doc = row.get('func_documentation_string', '')
                    if code and len(code) > 50:
                        samples.append({
                            "instruction": f"Explain this {lang} code:",
                            "input": code[:2000],
                            "output": doc[:500] if doc else f"Function from {row.get('repository_name', 'codebase')}"
                        })
            except Exception as e:
                print(f"Error parsing {lang}/{split}: {e}")
    
    out_file = os.path.join(DATA_DIR, "codesearchnet.jsonl")
    with open(out_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"Wrote {len(samples)} samples to {out_file}")
    return len(samples)

if __name__ == "__main__":
    download_with_wget()
    parse_parquet()