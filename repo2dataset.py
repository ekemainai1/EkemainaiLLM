#!/usr/bin/env python3
"""
repo2dataset.py - Generate instruction-tuning dataset from GitHub repos.
Usage:
    python repo2dataset.py --repo <GITHUB_URL> --output dataset.jsonl --workers 8
    python repo2dataset.py --repo <GITHUB_URL> --output dataset.jsonl --include-discussions
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SUPPORTED_EXT = [".py", ".js", ".ts", ".java", ".kt", ".cpp", ".xml"]
MAX_LINES = 40
MAX_WORKERS = 8


def clone_repo(repo_url, target_dir):
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, target_dir],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def get_files(repo_path):
    files = []
    for root, _, filenames in os.walk(repo_path):
        for f in filenames:
            if any(f.endswith(ext) for ext in SUPPORTED_EXT):
                if any(skip in f for skip in ["node_modules", ".gradle", "build", "__pycache__", ".git"]):
                    continue
                files.append(os.path.join(root, f))
    return files


def chunk_code(code):
    lines = code.split("\n")
    for i in range(0, len(lines), MAX_LINES):
        chunk = "\n".join(lines[i : i + MAX_LINES])
        if chunk.strip():
            yield chunk


def generate_samples(code_chunk, file_path=""):
    ext = Path(file_path).suffix if file_path else ""
    lang_hint = f" ({ext.lstrip('.')} code)" if ext else ""

    return [
        {
            "instruction": "Explain what this code does" + lang_hint,
            "input": code_chunk,
            "output": "",
        },
        {
            "instruction": "Find potential bugs, errors, or security issues in this code",
            "input": code_chunk,
            "output": "",
        },
        {
            "instruction": "Refactor and improve this code for better performance and readability",
            "input": code_chunk,
            "output": "",
        },
        {
            "instruction": "Write comprehensive unit tests for this code",
            "input": code_chunk,
            "output": "",
        },
    ]


def process_file(file_path):
    samples = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        if len(code) < 50 or len(code) > 50000:
            return samples

        for chunk in chunk_code(code):
            if len(chunk.strip()) > 20:
                samples.extend(generate_samples(chunk, file_path))
    except Exception:
        pass
    return samples


def build_dataset_parallel(files, workers=8, desc="Processing"):
    dataset = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            dataset.extend(future.result())
    return dataset


def save_jsonl(dataset, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate instruction dataset from GitHub repo")
    parser.add_argument("--repo", required=True, help="GitHub repo URL or local path")
    parser.add_argument("--output", default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--include-discussions", action="store_true", help="Include GitHub discussions")
    parser.add_argument("--local", action="store_true", help="Treat repo as local directory")
    args = parser.parse_args()

    if args.local:
        repo_path = args.repo
        print(f"Processing local repo: {repo_path}")
        files = get_files(repo_path)
        print(f"Found {len(files)} files")
        dataset = build_dataset_parallel(files, args.workers, "Processing files")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = os.path.join(tmpdir, "repo")
            print(f"Cloning {args.repo}...")
            clone_repo(args.repo, repo_path)
            files = get_files(repo_path)
            print(f"Found {len(files)} files")
            dataset = build_dataset_parallel(files, args.workers, "Processing files")

            if args.include_discussions:
                try:
                    from github_ingest import ingest_github_discussions
                    gh_data = ingest_github_discussions(args.repo, max_items=50)
                    dataset.extend(gh_data)
                    print(f"Added {len(gh_data)} discussion samples")
                except ImportError:
                    print("github_ingest.py not found, skipping discussions")

    save_jsonl(dataset, args.output)
    print(f"Done! Total: {len(dataset)} samples")


if __name__ == "__main__":
    main()