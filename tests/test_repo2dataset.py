#!/usr/bin/env python3
"""Unit tests for repo2dataset."""
import os
import tempfile
import pytest
from repo2dataset import (
    chunk_code,
    get_files,
    generate_samples,
    process_file,
)


def test_chunk_code():
    code = "line1\nline2\nline3\n"
    chunks = list(chunk_code(code, max_lines=2))
    assert len(chunks) == 2
    assert chunks[0] == "line1\nline2"
    assert chunks[1] == "line3\n"


def test_chunk_code_empty():
    code = "short"
    chunks = list(chunk_code(code, max_lines=40))
    assert len(chunks) == 0


def test_get_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")
        
        txt_file = os.path.join(tmpdir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("text file")
        
        files = get_files(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".py")


def test_get_files_excludes():
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = os.path.join(tmpdir, "build")
        os.makedirs(build_dir)
        with open(os.path.join(build_dir, "test.py"), "w") as f:
            f.write("print('hello')")
        
        files = get_files(tmpdir)
        assert len(files) == 0


def test_generate_samples():
    code = "def foo(): pass"
    samples = generate_samples(code, "/path/test.py")
    assert len(samples) == 4
    assert samples[0]["instruction"] == "Explain what this code does (.py code)"
    assert code in samples[0]["input"]


def test_process_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        f.write("def bar():\n    return 0\n")
        temp_path = f.name
    
    try:
        samples = process_file(temp_path)
        assert len(samples) > 0
    finally:
        os.unlink(temp_path)


def test_process_file_too_small():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1")
        temp_path = f.name
    
    try:
        samples = process_file(temp_path)
        assert samples == []
    finally:
        os.unlink(temp_path)


def test_process_file_too_large():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = " + "1," * 10000)
        temp_path = f.name
    
    try:
        samples = process_file(temp_path)
        assert samples == []
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])