#!/usr/bin/env python3
"""
github_ingest.py - Fetch issues, PRs, and comments from GitHub API for reasoning data.
Usage:
    from github_ingest import ingest_github_discussions
    dataset = ingest_github_discussions("https://github.com/user/repo", max_items=50)
"""
import os
import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"


def get_headers():
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_repo_url(repo_url):
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1].replace(".git", "")


def fetch_items(url, key, max_items=50):
    items = []
    params = {"state": "all", "per_page": 100, "sort": "updated"}
    while url and len(items) < max_items:
        try:
            res = requests.get(url, headers=get_headers(), params=params, timeout=30)
            if res.status_code == 403:
                print(f"Rate limited. Consider setting GITHUB_TOKEN.")
                break
            res.raise_for_status()
            data = res.json()
            if isinstance(data, list):
                items.extend(data)
            elif key in data:
                items.extend(data[key])
            url = res.links.get("next", {}).get("url")
            params = {}
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            break
    return items[:max_items]


def fetch_issues(owner, repo, max_items=50):
    return fetch_items(f"{GITHUB_API}/repos/{owner}/{repo}/issues", "issues", max_items)


def fetch_prs(owner, repo, max_items=50):
    return fetch_items(f"{GITHUB_API}/repos/{owner}/{repo}/pulls", None, max_items)


def fetch_comments(url):
    try:
        res = requests.get(url, headers=get_headers(), timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception:
        return []


def issue_to_sample(issue):
    return {
        "instruction": "Analyze and suggest fixes for this bug report or feature request.",
        "input": f"Title: {issue.get('title', '')}\n\n{issue.get('body', '')}",
        "output": "",
    }


def pr_to_sample(pr):
    return {
        "instruction": "Summarize, review, and suggest improvements for this pull request.",
        "input": f"Title: {pr.get('title', '')}\n\n{pr.get('body', '')}",
        "output": "",
    }


def comment_to_sample(comment):
    return {
        "instruction": "Respond to this code review comment or developer discussion.",
        "input": comment.get("body", ""),
        "output": "",
    }


def ingest_github_discussions(repo_url, max_items=50):
    owner, repo = parse_repo_url(repo_url)
    dataset = []

    print(f"Fetching issues from {owner}/{repo}...")
    issues = fetch_issues(owner, repo, max_items)
    for issue in tqdm(issues, desc="Processing issues"):
        if "pull_request" not in issue:
            dataset.append(issue_to_sample(issue))
            try:
                comments = fetch_comments(issue.get("comments_url", ""))
                for c in comments[:5]:
                    dataset.append(comment_to_sample(c))
            except Exception:
                pass

    print(f"Fetching PRs from {owner}/{repo}...")
    prs = fetch_prs(owner, repo, max_items // 2)
    for pr in tqdm(prs, desc="Processing PRs"):
        dataset.append(pr_to_sample(pr))
        try:
            comments = fetch_comments(pr.get("comments_url", ""))
            for c in comments[:5]:
                dataset.append(comment_to_sample(c))
        except Exception:
            pass

    print(f"Total discussion samples: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", default="discussions.jsonl")
    parser.add_argument("--max-items", type=int, default=50)
    args = parser.parse_args()

    import json

    dataset = ingest_github_discussions(args.repo, args.max_items)
    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {args.output}")