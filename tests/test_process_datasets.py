#!/usr/bin/env python3
"""Unit tests for datasets processing."""
import json
import tempfile
import pytest
from scripts.process_datasets import (
    chunk_code,
    convert_apps_sample,
    convert_codesearchnet,
    convert_codealpaca,
    convert_oss_instruct,
    convert_humaneval,
    convert_mbpp,
    load_jsonl,
    save_jsonl,
)


def test_chunk_code():
    code = "\n".join([f"line_{i}" for i in range(50)])
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 2
    assert len(chunks[0].split("\n")) == 40
    assert len(chunks[1].split("\n")) == 10


def test_chunk_code_empty():
    code = "short"
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 0


def test_convert_apps_sample():
    item = {
        "question": "Sum two numbers",
        "solutions": ["def add(a,b): return a+b"],
        "starter_code": "def add(a, b):",
    }
    samples = convert_apps_sample(item)
    assert len(samples) == 2
    assert samples[0]["instruction"] == "Solve this coding problem. Write the complete Python solution."
    assert "Sum two numbers" in samples[0]["input"]
    assert "def add(a,b): return a+b" in samples[0]["output"]


def test_convert_codesearchnet():
    item = {
        "code": "def foo(): pass",
        "english_documentation": "This does something",
        "func_documentation_string": None,
    }
    samples = convert_codesearchnet(item)
    assert len(samples) == 1
    assert "This does something" in samples[0]["input"]


def test_convert_codesearchnet_empty():
    item = {"code": "", "english_documentation": ""}
    samples = convert_codesearchnet(item)
    assert samples == []


def test_convert_codealpaca():
    item = {
        "instruction": "Write hello world",
        "input": "",
        "output": "print('hello world')",
    }
    samples = convert_codealpaca(item)
    assert len(samples) == 1
    assert samples[0]["instruction"] == "Write hello world"


def test_convert_codealpaca_no_instruction():
    item = {"instruction": "", "input": "", "output": ""}
    samples = convert_codealpaca(item)
    assert samples == []


def test_convert_oss_instruct():
    item = {
        "instruction": "Fix bug",
        "input": "code here",
        "output": "fixed code",
    }
    samples = convert_oss_instruct(item)
    assert len(samples) == 1
    assert samples[0]["output"] == "fixed code"


def test_convert_humaneval():
    item = {
        "prompt": "def foo(x): return x",
        "canonical_solution": "def foo(x): return x",
    }
    samples = convert_humaneval(item)
    assert len(samples) == 1
    assert "def foo(x): return x" in samples[0]["output"]


def test_convert_humaneval_empty():
    item = {"prompt": "", "canonical_solution": ""}
    samples = convert_humaneval(item)
    assert samples == []


def test_convert_mbpp():
    item = {
        "text": "Write a function",
        "code": "def foo(): pass",
    }
    samples = convert_mbpp(item)
    assert len(samples) == 1
    assert "def foo(): pass" in samples[0]["output"]


def test_convert_mbpp_empty():
    item = {"text": "", "code": ""}
    samples = convert_mbpp(item)
    assert samples == []


def test_load_jsonl_nonexistent():
    items = load_jsonl("/nonexistent/file.jsonl")
    assert items == []


def test_load_jsonl_save_roundtrip():
    data = [
        {"instruction": "Test", "input": "Test input", "output": "Test output"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name
    
    loaded = load_jsonl(temp_path)
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "Test"


def test_save_jsonl():
    data = [{"instruction": "Test"}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name
    
    save_jsonl(data, temp_path)
    loaded = load_jsonl(temp_path)
    assert len(loaded) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])