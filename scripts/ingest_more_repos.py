#!/usr/bin/env python3
"""Clone more GitHub repos for additional training data."""
import json
import os
import subprocess
import tempfile

REPOS = [
    ("https://github.com/facebook/react", "react"),
    ("https://github.com/angular/angular", "angular"),
    ("https://github.com/vuejs/vue", "vue"),
    ("https://github.com/microsoft/vscode", "vscode"),
    ("https://github.com/fastapi/fastapi", "fastapi"),
    ("https://github.com/pytorch/pytorch", "pytorch"),
    ("https://github.com/tensorflow/tensorflow", "tensorflow"),
    ("https://github.com/kubernetes/kubernetes", "kubernetes"),
]

SUPPORTED_EXTS = [".py", ".js", ".ts", ".java", ".kt", ".cpp", ".go", ".rs"]
MAX_SIZE = 50000


def clone_repo(url, dest):
    subprocess.run(["git", "clone", "--depth", "1", url, dest], 
                 capture_output=True, timeout=120, cwd=os.path.dirname(dest))


def extract_code(repo_path):
    code_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "build", "dist", "__pycache__", ".venv"]]
        for f in files:
            if any(f.endswith(ext) for ext in SUPPORTED_EXTS):
                fpath = os.path.join(root, f)
                try:
                    if os.path.getsize(fpath) < MAX_SIZE:
                        code_files.append(fpath)
                except:
                    pass
    return code_files


def sample_code(fpath):
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().split("\n")[:40]
            content = "\n".join(lines)
            if len(content) > 100:
                return content
    except:
        pass
    return None


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/more_code.jsonl")
    args = parser.parse_args()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for repo_url, name in REPOS:
            print(f"Processing {name}...")
            repo_path = os.path.join(tmpdir, name)
            
            try:
                subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path],
                           capture_output=True, timeout=120)
                
                files = extract_code(repo_path)[:300]
                samples = []
                
                for fpath in files:
                    code = sample_code(fpath)
                    if code:
                        ext = os.path.splitext(fpath)[1]
                        samples.append({
                            "instruction": f"Explain this {ext.lstrip('.')} code.",
                            "input": code[:600],
                            "output": f"Code snippet in {ext.lstrip('.')}.",
                        })
                
                print(f"  -> {len(samples)} samples")
                
                with open(args.output, "a") as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                        
            except Exception as e:
                print(f"  Failed: {e}")
    
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    run()