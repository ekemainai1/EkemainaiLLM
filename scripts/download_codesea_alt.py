#!/usr/bin/env python3
"""Download CodeSearchNet via direct HTTP - no torch needed."""
import os
import requests
import io
import pandas as pd

# Try without token first
def try_download():
    # Known parquet file URL from the dataset page
    # Python train split
    url = "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/refs%2Fconvert%2Fparquet/python/train-00000-of-00001-0.parquet"
    
    print(f"Trying: {url[:80]}...")
    
    try:
        resp = requests.get(url, timeout=120)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            # Save to file
            dest = "data/codesea/python_train.parquet"
            os.makedirs("data/codesea", exist_ok=True)
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"Saved: {dest} ({len(resp.content)} bytes)")
            
            # Try to read
            df = pd.read_parquet(io.BytesIO(resp.content))
            print(f"Rows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
        else:
            print(f"Failed: {resp.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    return None

# Try alternative URLs
def try_alternatives():
    urls = [
        # Original URL pattern
        "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/refs%2Fconvert%2Fparquet/python/train-00000-of-00001-0.parquet",
        # Alternative
        "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python/train-00000-of-00001.parquet",
    ]
    
    for url in urls:
        print(f"\nTrying: {url[:80]}...")
        try:
            resp = requests.get(url, timeout=60)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"Success! Size: {len(resp.content)}")
                return resp.content
        except Exception as e:
            print(f"Error: {e}")
    
    return None

if __name__ == "__main__":
    df = try_download()
    if df is None:
        try_alternatives()