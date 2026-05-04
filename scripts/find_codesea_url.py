#!/usr/bin/env python3
"""Try to find correct parquet URL."""
import requests
import subprocess
import sys

# Try different URL patterns
urls = [
    # Main branch
    "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python/train-00000-of-00001.parquet",
    # Converted parquet
    "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/refs/convert/parquet/all/train-00000-of-00002.parquet",
    # Try listing files first
    "https://huggingface.co/api/datasets/code-search-net/code_search_net",
]

for url in urls:
    print(f"\nTrying: {url[:70]}...")
    try:
        resp = requests.get(url, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Success! Length: {len(resp.content)}")
            if "json" in resp.headers.get("content-type", ""):
                print("JSON response:", resp.text[:200])
    except Exception as e:
        print(f"Error: {e}")