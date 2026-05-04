#!/usr/bin/env python3
"""Clone and extract code from GitHub repos for additional training data."""
import json
import os
import subprocess
import tempfile
import shutil

REPOS = [
    "https://github.com/torvalds/linux",
    "https://github.com/python/cpython",
    "https://github.com/django/django",
    "https://github.com/jmportnet/Prophet",
]

SUPPORTED_EXTS = [".py", ".js", ".ts", ".java", ".kt", ".cpp", ".go", ".rs"]
MAX_SIZE = 50000


def clone_repo(url, dest):
    """Clone a repo."""
    cmd = ["git", "clone", "--depth", "1", url, dest]
    subprocess.run(cmd, capture_output=True, timeout=120)


def extract_code_files(repo_path):
    """Extract code files from repo."""
    code_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "build", "dist", "__pycache__", ".venv", "venv", "test", "tests"]]
        
        for f in files:
            if any(f.endswith(ext) for ext in SUPPORTED_EXTS):
                fpath = os.path.join(root, f)
                try:
                    size = os.path.getsize(fpath)
                    if size < MAX_SIZE and size > 50:
                        code_files.append(fpath)
                except:
                    pass
    return code_files


def sample_code(fpath):
    """Read and sample code."""
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
            # Sample first 40 lines
            sample = "\n".join(lines[:40])
            if len(sample) > 100:
                return sample
    except:
        pass
    return None


def repo_to_samples(repo_url, name):
    """Convert repo to training samples."""
    print(f"Processing {name}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        
        try:
            clone_repo(repo_url, repo_path)
        except Exception as e:
            print(f"  Clone failed: {e}")
            return []
        
        code_files = extract_code_files(repo_path)
        print(f"  Found {len(code_files)} code files")
        
        samples = []
        for fpath in code_files[:500]:  # Limit to 500 files per repo
            code = sample_code(fpath)
            if code:
                ext = os.path.splitext(fpath)[1]
                samples.append({
                    "instruction": f"Explain this {ext.lstrip('.')} code snippet.",
                    "input": code[:600],
                    "output": f"This is {ext.lstrip('.')} code that needs explanation.",
                })
        
        return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/github_code.jsonl")
    args = parser.parse_args()
    
    all_samples = []
    for repo in REPOS:
        name = repo.rstrip("/").split("/")[-1]
        samples = repo_to_samples(repo, name)
        all_samples.extend(samples)
        print(f"  -> {len(samples)} samples")
    
    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"Saved: {args.output} ({len(all_samples)} samples)")


if __name__ == "__main__":
    main()