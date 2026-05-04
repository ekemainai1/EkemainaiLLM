#!/usr/bin/env python3
"""Unit tests for add_reasoning.py script (CoT and tools transformation)."""
import json
import pytest
from scripts.add_reasoning import (
    add_chain_of_thought,
    determine_action,
    summarize_problem,
    generate_strategy,
    generate_consideration,
    summarize_input,
    create_tool_calling_example,
    TOOL_DEFINITIONS,
)


class TestChainOfThought:
    """Test Chain-of-Thought generation."""
    
    def test_add_chain_of_thought_basic(self):
        """Test basic CoT addition."""
        output = "def add(a, b): return a + b"
        instruction = "Write a function to add two numbers"
        input_text = "Add 1 and 2"
        
        result = add_chain_of_thought(output, instruction, input_text)
        
        assert "Let me analyze this problem step by step" in result
        assert "Step 1:" in result
        assert "Step 2:" in result
        assert "Step 3:" in result
        assert "Step 4:" in result
        assert "def add(a, b): return a + b" in result
    
    def test_add_chain_of_thought_with_execution_trace(self):
        """Test CoT with code execution trace."""
        output = "def reverse_string(s): return s[::-1]"
        instruction = "Write function to reverse string"
        input_text = "Reverse 'hello'"
        
        result = add_chain_of_thought(output, instruction, input_text)
        
        assert "Execution trace" in result
        assert "-> Initializing" in result
        assert "-> Processing input" in result
        assert "-> Generating output" in result
    
    def test_add_chain_of_thought_empty_output(self):
        """Test CoT with empty output returns empty."""
        output = ""
        result = add_chain_of_thought(output, "test", "test")
        # Empty outputs should return minimal reasoning
        assert "Let me analyze" in result or result == ""


class TestProblemClassification:
    """Test problem classification."""
    
    def test_write_instruction(self):
        """Test classification for write/create instructions."""
        inst = "Write a function to add numbers"
        assert determine_action(inst) == "create new code"
        
        inst = "Create a class"
        assert determine_action(inst) == "create new code"
        
        inst = "Implement a sorting algorithm"
        assert determine_action(inst) == "create new code"
    
    def test_fix_instruction(self):
        """Test classification for bug fix instructions."""
        inst = "Fix the bug in this code"
        assert determine_action(inst) == "debug existing code"
        
        inst = "Debug this function"
        assert determine_action(inst) == "debug existing code"
    
    def test_explain_instruction(self):
        """Test classification for explanation instructions."""
        inst = "Explain what this code does"
        assert determine_action(inst) == "explain code"
        
        inst = "Describe the function"
        assert determine_action(inst) == "explain code"
    
    def test_optimize_instruction(self):
        """Test classification for optimization instructions."""
        inst = "Optimize this code for performance"
        assert determine_action(inst) == "optimize code"
        
        inst = "Improve the algorithm"
        assert determine_action(inst) == "optimize code"
    
    def test_search_instruction(self):
        """Test classification for search instructions."""
        inst = "Locate the function"
        action = determine_action(inst)
        assert action is not None
    
    def test_default_instruction(self):
        """Test default classification."""
        inst = "Process this data"
        action = determine_action(inst)
        assert action is not None


class TestReasoningGeneration:
    """Test reasoning step generation."""
    
    def test_summarize_problem(self):
        """Test problem summarization."""
        text = "Write a function to add two numbers"
        result = summarize_problem(text)
        assert "add two numbers" in result
    
    def test_summarize_problem_long_text(self):
        """Test problem summarization truncates long text."""
        text = "A" * 200
        result = summarize_problem(text)
        assert len(result) == 100
    
    def test_summarize_problem_empty(self):
        """Test problem summarization with empty text."""
        result = summarize_problem("")
        assert result is not None
    
    def test_generate_strategy(self):
        """Test strategy generation."""
        instruction = "Write a function"
        output = "def foo(): pass"
        result = generate_strategy(instruction, output)
        assert result is not None
        assert len(result) > 0
    
    def test_generate_consideration(self):
        """Test edge case generation."""
        inst = "Sort a list"
        result = generate_consideration(inst)
        assert "empty" in result.lower() or "duplicate" in result.lower()
    
    def test_generate_consideration_search(self):
        """Test edge case generation for search."""
        inst = "Find element"
        result = generate_consideration(inst)
        assert "not found" in result.lower() or "empty" in result.lower()
    
    def test_summarize_input(self):
        """Test input summarization."""
        text = "Add numbers 1 and 2"
        result = summarize_input(text)
        assert result is not None
    
    def test_summarize_input_empty(self):
        """Test input summarization with empty text."""
        result = summarize_input("")
        assert "no additional" in result.lower() or result == ""


class TestToolDefinitions:
    """Test tool definitions."""
    
    def test_tool_definitions_exist(self):
        """Test all required tools are defined."""
        assert "python_executor" in TOOL_DEFINITIONS
        assert "file_reader" in TOOL_DEFINITIONS
        assert "file_writer" in TOOL_DEFINITIONS
        assert "web_search" in TOOL_DEFINITIONS
        assert "git_clone" in TOOL_DEFINITIONS
        assert "code_search" in TOOL_DEFINITIONS
    
    def test_tool_has_name(self):
        """Test each tool has a name."""
        for tool_name, tool_def in TOOL_DEFINITIONS.items():
            assert "name" in tool_def
            assert tool_def["name"] == tool_name
    
    def test_tool_has_description(self):
        """Test each tool has a description."""
        for tool_def in TOOL_DEFINITIONS.values():
            assert "description" in tool_def
            assert len(tool_def["description"]) > 0
    
    def test_tool_has_input_schema(self):
        """Test each tool has input schema."""
        for tool_def in TOOL_DEFINITIONS.values():
            assert "input_schema" in tool_def
            schema = tool_def["input_schema"]
            assert "type" in schema
            assert "properties" in schema
            assert "required" in schema
    
    def test_python_executor_schema(self):
        """Test python_executor has correct schema."""
        tool = TOOL_DEFINITIONS["python_executor"]
        assert "code" in tool["input_schema"]["properties"]
        assert "code" in tool["input_schema"]["required"]
    
    def test_file_reader_schema(self):
        """Test file_reader has correct schema."""
        tool = TOOL_DEFINITIONS["file_reader"]
        props = tool["input_schema"]["properties"]
        assert "path" in props
        assert "lines" in props
    
    def test_web_search_schema(self):
        """Test web_search has correct schema."""
        tool = TOOL_DEFINITIONS["web_search"]
        assert "query" in tool["input_schema"]["properties"]
        assert "query" in tool["input_schema"]["required"]


class TestToolCalling:
    """Test tool calling generation."""
    
    def test_create_tool_calling_fetch(self):
        """Test tool calling for fetch/API instructions."""
        instruction = "Fetch data from API"
        input_text = "GET /api/users"
        output = '{"users": []}'
        
        result = create_tool_calling_example(instruction, input_text, output)
        
        assert result is not None
        data = json.loads(result)
        assert "tool" in data
        assert "args" in data
    
    def test_create_tool_calling_file_read(self):
        """Test tool calling for file read instructions."""
        instruction = "Read file contents"
        input_text = "Read main.py"
        output = "file content"
        
        result = create_tool_calling_example(instruction, input_text, output)
        
        data = json.loads(result)
        assert data["tool"] in ["file_reader", "file_writer", "python_executor"]
    
    def test_create_tool_calling_search(self):
        """Test tool calling for search instructions."""
        instruction = "Search for function"
        input_text = "Find def foo"
        output = "found"
        
        result = create_tool_calling_example(instruction, input_text, output)
        
        data = json.loads(result)
        assert "tool" in data["args"] or "pattern" in data["args"]
    
    def test_create_tool_calling_clone(self):
        """Test tool calling for clone/repo instructions."""
        instruction = "Clone repository"
        input_text = "https://github.com/example/repo"
        output = "cloned"
        
        result = create_tool_calling_example(instruction, input_text, output)
        
        data = json.loads(result)
        assert "tool" in data
    
    def test_tool_calling_json_format(self):
        """Test tool calling returns valid JSON."""
        instruction = "Run this code"
        input_text = ""
        output = "print('hello')"
        
        result = create_tool_calling_example(instruction, input_text, output)
        
        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, dict)


class TestReasoningOutput:
    """Test reasoning output formatting."""
    
    def test_code_generation_format(self):
        """Test output format for code generation."""
        output = "def foo(): return 42"
        instruction = "Write a function"
        input_text = "Return 42"
        
        result = add_chain_of_thought(output, instruction, input_text)
        
        assert "Problem Analysis" in result or "Let me analyze" in result
        assert output in result
    
    def test_bug_fix_format(self):
        """Test output format for bug fix."""
        output = "# Bug fixed here"
        instruction = "Fix the bug"
        input_text = "Bug in code"
        
        result = add_chain_of_thought(output, instruction, input_text)
        
        assert "Let me analyze" in result
    
    def test_explanation_format(self):
        """Test output format for explanation."""
        output = "This code does X"
        instruction = "Explain this"
        input_text = "def foo(): pass"
        
        result = add_chain_of_thought(output, instruction, input_text)
        
        assert "Let me analyze" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])