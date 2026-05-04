#!/usr/bin/env python3
"""Clone more GitHub repos for code."""
import json
import os
import subprocess
import tempfile

REPOS = [
    "https://github.com/rails/rails",
    "https://github.com/laravel/laravel", 
    "https://github.com/spring-projects/spring-boot",
    "https://github.com/rails/rails",
    "https://github.com/gorhill/uBlock",
    "https://github.com/ace/ace",
    "https://github.com/chartjs/Chart.js",
    "https://github.com/highcharts/highcharts",
    "https://github.com/moment/moment",
    "https://github.com/lodash/lodash",
    "https://github.com/axios/axios",
    "https://github.com/moment/moment.js",
    "https://github.com/Automattic/mongoose",
    "https://github.com/sequelize/sequelize",
    "https://github.com/reduxjs/redux",
    "https://github.com/reactjs/redux",
    "https://github.com/facebook/flux",
    "https://github.com/graphql/graphql-js",
    "https://github.com/vercel/next.js",
    "https://github.com/nuxt/nuxt.js",
]

SUPPORTED_EXTS = [".py", ".js", ".ts", ".go", ".rs", ".java", ".kt", ".rb", ".php"]
MAX_SIZE = 25000


def clone_one(url, name, output_file):
    """Clone one repo and extract code."""
    print(f"Cloning {name}...")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, tmpdir],
                capture_output=True, timeout=45
            )
            if result.returncode != 0:
                print(f"  Failed to clone")
                return
            
            # Extract code files
            files = []
            for root, dirs, filenames in os.walk(tmpdir):
                dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__", "vendor", "dist", "build"]]
                for f in filenames:
                    if any(f.endswith(ext) for ext in SUPPORTED_EXTS):
                        fpath = os.path.join(root, f)
                        try:
                            if os.path.getsize(fpath) < MAX_SIZE:
                                files.append(fpath)
                        except:
                            pass
            
            # Sample from files
            samples = []
            for fpath in files[:150]:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.read().split("\n")[:40]
                        if len("\n".join(lines)) > 100:
                            ext = os.path.splitext(fpath)[1]
                            samples.append({
                                "instruction": f"Explain this {ext.lstrip('.')} code.",
                                "input": "\n".join(lines)[:500],
                                "output": f"Code snippet in {ext.lstrip('.')}.",
                            })
                except:
                    pass
            
            # Append to file
            if samples:
                with open(output_file, "a") as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                print(f"  -> {len(samples)} samples")
                
    except Exception as e:
        print(f"  Error: {e}")


def main():
    output = "data/more_repos2.jsonl"
    
    # Clear/create output
    open(output, "w").close()
    
    for repo in REPOS:
        name = repo.split("/")[-1]
        clone_one(repo, name, output)
    
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()