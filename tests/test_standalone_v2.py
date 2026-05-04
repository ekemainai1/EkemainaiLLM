#!/usr/bin/env python3
"""Standalone tests for repo2dataset and github_ingest functions."""
import os
import sys


def chunk_code(code, max_lines=40):
    lines = code.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        if chunk.strip():
            yield chunk


def parse_repo_url(repo_url):
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1].replace(".git", "")


def issue_to_sample(issue):
    return {
        "instruction": "Analyze and suggest fixes for this bug report or feature request.",
        "input": "Title: " + issue.get("title", "") + "\n\n" + issue.get("body", ""),
        "output": "",
    }


def pr_to_sample(pr):
    return {
        "instruction": "Summarize, review, and suggest improvements for this pull request.",
        "input": "Title: " + pr.get("title", "") + "\n\n" + pr.get("body", ""),
        "output": "",
    }


def comment_to_sample(comment):
    return {
        "instruction": "Respond to this code review comment or developer discussion.",
        "input": comment.get("body", ""),
        "output": "",
    }


# === repo2dataset tests ===

def test_chunk_code():
    code = "line1\nline2\nline3\n"
    chunks = list(chunk_code(code, max_lines=2))
    assert len(chunks) == 2, "Expected 2 chunks"
    assert chunks[0] == "line1\nline2"
    assert chunks[1] == "line3\n"
    print("test_chunk_code: PASS")


def test_chunk_code():
    code = "line1\nline2\nline3\n"
    chunks = list(chunk_code(code, max_lines=2))
    assert len(chunks) == 2, "Expected 2 chunks"
    assert chunks[0] == "line1\nline2"
    assert chunks[1] == "line3\n"
    print("test_chunk_code: PASS")


def test_chunk_code_empty():
    code = ""
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 0
    print("test_chunk_code_empty: PASS")


def test_chunk_code_large():
    code = "\n".join(["line_" + str(i) for i in range(100)])
    chunks = list(chunk_code(code, max_lines=30))
    assert len(chunks) == 4
    print("test_chunk_code_large: PASS")


# === github_ingest tests ===

def test_parse_repo_url():
    owner, repo = parse_repo_url("https://github.com/user/repo")
    assert owner == "user"
    assert repo == "repo"
    print("test_parse_repo_url: PASS")


def test_parse_repo_url_with_git():
    owner, repo = parse_repo_url("https://github.com/user/repo.git")
    assert owner == "user"
    assert repo == "repo"
    print("test_parse_repo_url_with_git: PASS")


def test_parse_repo_url_deep():
    owner, repo = parse_repo_url("https://github.com/android/compose-samples")
    assert owner == "android"
    assert repo == "compose-samples"
    print("test_parse_repo_url_deep: PASS")


def test_issue_to_sample():
    issue = {"title": "Bug in login", "body": "Cannot login"}
    sample = issue_to_sample(issue)
    assert "Bug in login" in sample["input"]
    assert "Analyze" in sample["instruction"]
    print("test_issue_to_sample: PASS")


def test_issue_to_sample_empty():
    issue = {"title": "", "body": ""}
    sample = issue_to_sample(issue)
    assert sample["input"] == "Title: \n\n"
    print("test_issue_to_sample_empty: PASS")


def test_pr_to_sample():
    pr = {"title": "Add feature", "body": "New feature"}
    sample = pr_to_sample(pr)
    assert "Add feature" in sample["input"]
    assert "Summarize" in sample["instruction"]
    print("test_pr_to_sample: PASS")


def test_comment_to_sample():
    comment = {"body": "LGTM"}
    sample = comment_to_sample(comment)
    assert "LGTM" in sample["input"]
    assert "Respond" in sample["instruction"]
    print("test_comment_to_sample: PASS")


def test_comment_to_sample_empty():
    comment = {"body": ""}
    sample = comment_to_sample(comment)
    assert sample["input"] == ""
    print("test_comment_to_sample_empty: PASS")


if __name__ == "__main__":
    print("Running standalone tests for repo2dataset and github_ingest...")
    test_chunk_code()
    test_chunk_code_empty()
    test_chunk_code_large()
    test_parse_repo_url()
    test_parse_repo_url_with_git()
    test_parse_repo_url_deep()
    test_issue_to_sample()
    test_issue_to_sample_empty()
    test_pr_to_sample()
    test_comment_to_sample()
    test_comment_to_sample_empty()
    print("\nAll tests passed!")