#!/usr/bin/env python3
"""Unit tests for evaluate script functions."""
import json
import pytest
from scripts.evaluate import (
    extract_code,
    DEFAULT_PROMPT,
)


def test_extract_code_python():
    text = '''Here is the code:
```python
def hello():
    return "hello"
```
End of response.'''
    code = extract_code(text)
    assert 'def hello():' in code
    assert 'return "hello"' in code


def test_extract_code_kotlin():
    text = '''Code:
```kotlin
fun hello() = "hello"
```
Done.'''
    code = extract_code(text)
    assert 'fun hello()' in code


def test_extract_code_no_fence():
    text = "def foo():\n    return 42"
    code = extract_code(text)
    assert 'def foo()' in code


def test_extract_code_with_comments():
    text = "# This is comment\ndef foo():\n    return 42"
    code = extract_code(text)
    assert "# This is comment" not in code
    assert "def foo()" in code


def test_default_prompt_format():
    prompt = DEFAULT_PROMPT.format(instruction="Test", input="test input")
    assert "[INST]" in prompt
    assert "Test" in prompt
    assert "test input" in prompt
    assert "[/INST]" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])