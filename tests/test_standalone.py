#!/usr/bin/env python3
"""Standalone test for process_datasets functions."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chunk_code(code, max_lines=40):
    lines = code.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        if len(chunk.strip()) > 20:
            yield chunk


def convert_apps_sample(item):
    samples = []
    question = item.get("question", "")
    solutions = item.get("solutions", [])
    for sol in solutions[:1]:
        samples.append({
            "instruction": "Solve this coding problem. Write the complete Python solution.",
            "input": question,
            "output": sol if isinstance(sol, str) else str(sol),
        })
    starter_code = item.get("starter_code", "")
    if starter_code:
        samples.append({
            "instruction": "Complete this coding problem using the starter code provided.",
            "input": f"Problem:\n{question}\n\nStarter code:\n{starter_code}",
            "output": "",
        })
    return samples


def convert_codesearchnet(item):
    code = item.get("func_documentation_string", "") or item.get("code", "")
    doc = item.get("english_documentation", "")
    if not code or not doc:
        return []
    return [{
        "instruction": "Explain this code based on the documentation.",
        "input": f"Documentation:\n{doc}\n\nCode:\n{code}",
        "output": "",
    }]


def convert_codealpaca(item):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    if not instruction:
        return []
    return [{
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }]


def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def test_chunk_code():
    code = "\n".join([f"line_{i}" for i in range(50)])
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert len(chunks[0].split("\n")) == 40
    assert len(chunks[1].split("\n")) == 10
    print("  test_chunk_code: PASS")


def test_chunk_code_empty():
    code = "short"
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 0
    print("  test_chunk_code_empty: PASS")


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
    print("  test_convert_apps_sample: PASS")


def test_convert_codesearchnet():
    item = {
        "code": "def foo(): pass",
        "english_documentation": "This does something",
        "func_documentation_string": None,
    }
    samples = convert_codesearchnet(item)
    assert len(samples) == 1
    assert "This does something" in samples[0]["input"]
    print("  test_convert_codesearchnet: PASS")


def test_convert_codesearchnet_empty():
    item = {"code": "", "english_documentation": ""}
    samples = convert_codesearchnet(item)
    assert samples == []
    print("  test_convert_codesearchnet_empty: PASS")


def test_convert_codealpaca():
    item = {
        "instruction": "Write hello world",
        "input": "",
        "output": "print('hello world')",
    }
    samples = convert_codealpaca(item)
    assert len(samples) == 1
    assert samples[0]["instruction"] == "Write hello world"
    print("  test_convert_codealpaca: PASS")


def test_convert_codealpaca_no_instruction():
    item = {"instruction": "", "input": "", "output": ""}
    samples = convert_codealpaca(item)
    assert samples == []
    print("  test_convert_codealpaca_no_instruction: PASS")


def test_load_jsonl_nonexistent():
    items = load_jsonl("/nonexistent/file.jsonl")
    assert items == []
    print("  test_load_jsonl_nonexistent: PASS")


def test_save_and_load():
    data = [{"instruction": "Test", "input": "Test input", "output": "Test output"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name
    
    loaded = load_jsonl(temp_path)
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "Test"
    os.unlink(temp_path)
    print("  test_save_and_load: PASS")


if __name__ == "__main__":
    print("Running standalone tests for process_datasets...")
    test_chunk_code()
    test_chunk_code_empty()
    test_convert_apps_sample()
    test_convert_codesearchnet()
    test_convert_codesearchnet_empty()
    test_convert_codealpaca()
    test_convert_codealpaca_no_instruction()
    test_load_jsonl_nonexistent()
    test_save_and_load()
    print("\nAll tests passed!")