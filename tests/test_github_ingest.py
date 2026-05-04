#!/usr/bin/env python3
"""Unit tests for github_ingest."""
import pytest
from github_ingest import (
    parse_repo_url,
    issue_to_sample,
    pr_to_sample,
    comment_to_sample,
)


def test_parse_repo_url():
    owner, repo = parse_repo_url("https://github.com/user/repo")
    assert owner == "user"
    assert repo == "repo"


def test_parse_repo_url_with_git():
    owner, repo = parse_repo_url("https://github.com/user/repo.git")
    assert owner == "user"
    assert repo == "repo"


def test_issue_to_sample():
    issue = {
        "title": "Bug in login",
        "body": "Cannot login with special chars",
    }
    sample = issue_to_sample(issue)
    assert "Bug in login" in sample["input"]
    assert "Analyze" in sample["instruction"]


def test_pr_to_sample():
    pr = {
        "title": "Add feature",
        "body": "New feature description",
    }
    sample = pr_to_sample(pr)
    assert "Add feature" in sample["input"]
    assert "Summarize" in sample["instruction"]


def test_comment_to_sample():
    comment = {"body": "LGTM"}
    sample = comment_to_sample(comment)
    assert "LGTM" in sample["input"]
    assert "Respond" in sample["instruction"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])