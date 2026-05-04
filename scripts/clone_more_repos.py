#!/usr/bin/env python3
"""Clone popular repos for training data."""
import json
import os
import subprocess
import tempfile

REPOS = [
    "golang/go",
    "rust-lang/rust",
    "facebook/react-native",
    "flutter/flutter",
    "ansible/ansible",
    "pandas-dev/pandas",
    "numpy/numpy",
    "matplotlib/matplotlib",
    "requests/requests",
    "twbs/bootstrap",
    "jquery/jquery",
    "nodejs/node",
    "expressjs/express",
    "redis/redis",
    "postgres/postgres",
]

SUPPORTED_EXTS = [".py", ".js", ".ts", ".go", ".rs", ".java", ".kt"]
MAX_SIZE = 30000


def clone_and_extract(url, limit=200):
    """Clone repo and extract code."""
    name = url.split("/")[-1]
    print(f"Cloning {name}...", end=" ")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Shallow clone
            subprocess.run(["git", "clone", "--depth", "1", url, tmpdir],
                        capture_output=True, timeout=60)
            
            files = []
            for root, dirs, filenames in os.walk(tmpdir):
                dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "build", "__pycache__", ".venv", "venv"]]
                for f in filenames:
                    if any(f.endswith(ext) for ext in SUPPORTED_EXTS):
                        fpath = os.path.join(root, f)
                        try:
                            if os.path.getsize(fpath) < MAX_SIZE:
                                files.append(fpath)
                        except:
                            pass
            
            samples = []
            for fpath in files[:limit]:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.read().split("\n")[:50]
                        if len("\n".join(lines)) > 100:
                            ext = os.path.splitext(fpath)[1]
                            samples.append({
                                "instruction": f"Explain this {ext.lstrip('.')} code.",
                                "input": "\n".join(lines)[:600],
                                "output": f"Code snippet.",
                            })
                except:
                    pass
            
            print(f"{len(samples)} samples")
            return samples
            
    except Exception as e:
        print(f"Failed: {e}")
        return []


def main():
    output_path = "data/more_repos.jsonl"
    
    all_samples = []
    for repo in REPOS:
        samples = clone_and_extract(f"https://github.com/{repo}", limit=200)
        all_samples.extend(samples)
        
        if all_samples:
            with open(output_path, "a") as f:
                for s in all_samples:
                    f.write(json.dumps(s) + "\n")
            all_samples = []
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()