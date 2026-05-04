#!/usr/bin/env python3
"""Download CodeSearchNet via direct HTTP (no torch)."""
import json
import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def download_file(url, dest_path):
    """Download a file from HF."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {dest_path}")


def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # CodeSearchNet files (from the dataset page)
    base_url = "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main"
    files = [
        "python_train.tar.gz",
    ]
    
    for fname in files:
        url = f"{base_url}/{fname}"
        dest = f"{output_dir}/{fname}"
        
        if os.path.exists(dest):
            print(f"Skipping {fname} (exists)")
            continue
        
        print(f"Downloading {fname}...")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    main()