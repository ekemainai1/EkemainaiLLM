#!/usr/bin/env python3
"""Fetch GitHub discussions for reasoning data."""
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

REPOS = [
    "android/compose-",
    "android/jetpack-",
    "kotlin/kotlinx.coro",
    "flutter/flutte",
    "facebook/rea",
    "vuejs/v",
    "microsoft/vsco",
    "python/cpytho",
    "google/",
    "dart-lang/sd",
    "apple/swif",
    "tensorflow/tensorfl",
    "pytorch/pytorc",
    "facebook/react-nativ",
    "rails/rails",
]

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def fetch_discussions(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/discussions"
    params = {"per_page": 100, "state": "all"}
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            return []
        
        discussions = resp.json()
        samples = []
        
        for d in discussions[:50]:
            body = d.get("body", "")
            title = d.get("title", "")
            if body and len(body) > 50:
                samples.append({
                    "instruction": "Analyze this developer discussion and provide reasoning.",
                    "input": f"Title: {title}\n\nDiscussion: {body[:500]}",
                    "output": "Technical analysis and solution reasoning.",
                })
        
        print(f"  {len(samples)} discussions")
        return samples
        
    except Exception as e:
        print(f"  Failed: {e}")
        return []

def main():
    output = "data/github_discussions.jsonl"
    all_samples = []
    
    for repo in REPOS:
        owner, name = repo.split("/")
        print(f"Fetching {repo}...", end=" ")
        samples = fetch_discussions(owner, name)
        all_samples.extend(samples)
    
    with open(output, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"\nTotal: {len(all_samples)} discussions -> {output}")

if __name__ == "__main__":
    main()