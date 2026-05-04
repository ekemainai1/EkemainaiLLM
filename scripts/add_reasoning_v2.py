#!/usr/bin/env python3
"""Transform dataset to add Chain-of-Thought reasoning and tool-use capabilities (optimized)."""
import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SYSTEM_PROMPT ="""You are an expert coding assistant with Chain-of-Thought reasoning and tool-use capabilities.

## Available Tools
You have access to the following tools to help complete tasks:

### 1. python_executor
Execute Python code and get results.
```json
{"tool": "python_executor", "args": {"code": "print('hello')"}}
```

### 2. file_reader
Read contents from a file.
```json
{"tool": "file_reader", "args": {"path": "main.py", "lines": 50}}
```

### 3. file_writer
Write content to a file.
```json
{"tool": "file_writer", "args": {"path": "output.py", "content": "print('hello')"}}
```

### 4. web_search
Search for information online.
```json
{"tool": "web_search", "args": {"query": "python list comprehension"}}
```

## Chain-of-Thought Reasoning
When solving problems, think step by step:
1. Understand what needs to be done
2. Determine the approach
3. Consider edge cases  
4. Implement the solution

Always explain your reasoning before giving the final answer."""

def create_reasoning_output(instruction: str, input_text: str, output: str) -> str:
    """Add Chain-of-Thought reasoning to output."""
    # Determine the problem type
    problem_type = classify_problem(instruction)
    
    # Generate reasoning steps
    steps = generate_reasoning_steps(problem_type, instruction, input_text)
    
    # Add execution trace if code
    execution_trace = ""
    if "def " in output or "class " in output or "import " in output:
        execution_trace = """## Execution Trace
```
-> Initializing...
-> Processing input...
-> Generating output...
-> Complete!
```
"""
    
    return f"""## Problem Analysis
{steps}

{output}

{execution_trace}"""

def classify_problem(instruction: str) -> str:
    """Classify the problem type."""
    inst = instruction.lower()
    if "write" in inst or "create" in inst or "implement" in inst:
        return "code_generation"
    elif "fix" in inst or "debug" in inst or "bug" in inst:
        return "bug_fix"
    elif "explain" in inst or "describe" in inst:
        return "explanation"
    elif "optimize" in inst or "improve" in inst:
        return "optimization"
    elif "find" in inst or "search" in inst:
        return "search"
    else:
        return "general"

def generate_reasoning_steps(problem_type: str, instruction: str, input_text: str) -> str:
    """Generate reasoning steps based on problem type."""
    type_templates = {
        "code_generation": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Write clean, efficient code following best practices
3. Edge cases: Input validation, error handling, performance
4. Implementation: Generate complete solution""",
        
        "bug_fix": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Identify the bug and understand the root cause
3. Edge cases: Test with various inputs
4. Implementation: Fix the issue while preserving existing functionality""",
        
        "explanation": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Explain the code clearly with examples
3. Edge cases: Make it understandable for different skill levels
4. Implementation: Provide comprehensive explanation""",
        
        "optimization": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Identify performance bottlenecks
3. Edge cases: Ensure correctness is maintained
4. Implementation: Optimize while keeping code readable""",
        
        "search": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Find relevant code patterns
3. Edge cases: Multiple matches, no matches
4. Implementation: Return best match""",
        
        "general": f"""Let me analyze this step by step:
1. Problem: {instruction[:100]}
2. Approach: Generate appropriate solution
3. Edge cases: Handle edge cases
4. Implementation: Complete the task"""
    }
    return type_templates.get(problem_type, type_templates["general"])

def transform_sample(sample: Dict) -> Dict:
    """Transform a single sample."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    if not output or len(output) < 10:
        return None
    
    # Transform output with CoT
    new_output = create_reasoning_output(instruction, input_text, output)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": new_output
    }

def transform_sample_with_tools(sample: Dict, include_tools: bool = False) -> Tuple[Dict, bool]:
    """Transform sample, optionally adding tool calling format."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    if not output or len(output) < 10:
        return None, False
    
    # Transform output with CoT
    new_output = create_reasoning_output(instruction, input_text, output)
    
    # Determine if tools should be added (selective)
    should_add_tools = include_tools and should_use_tools(instruction)
    
    return {
        "instruction": SYSTEM_PROMPT + f"\n\n## Task\n{instruction}",
        "input": input_text + ("\n\nUse appropriate tools to complete this task." if should_add_tools else ""),
        "output": new_output
    }, should_add_tools

def should_use_tools(instruction: str) -> bool:
    """Determine if this sample should include tool calling."""
    inst = instruction.lower()
    tool_keywords = ["fetch", "api", "http", "search", "read", "write", "file", "create", "execute", "run"]
    return any(kw in inst for kw in tool_keywords)

def load_jsonl(path: Path) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples

def save_jsonl(samples: List[Dict], path: Path) -> int:
    """Save samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)

def main():
    parser = argparse.ArgumentParser(description="Add Chain-of-Thought to dataset")
    parser.add_argument("--input", default="data/combined_final.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/cot_final.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, help="Limit samples")
    parser.add_argument("--tools", action="store_true", help="Add tool calling to relevant samples")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading {input_path}...")
    samples = load_jsonl(input_path)
    print(f"Loaded {len(samples)} samples")
    
    if args.limit:
        samples = samples[:args.limit]
    
    print("Transforming with Chain-of-Thought...")
    
    transformed = []
    tool_count = 0
    for i, sample in enumerate(samples):
        if i % 50000 == 0:
            print(f"Processing {i}/{len(samples)}...")
        
        if args.tools:
            new_sample, has_tools = transform_sample_with_tools(sample, include_tools=True)
            if has_tools:
                tool_count += 1
        else:
            new_sample = transform_sample(sample)
        
        if new_sample:
            transformed.append(new_sample)
    
    print(f"Transformed: {len(transformed)} samples")
    if args.tools:
        print(f"Samples with tool calling: {tool_count}")
    
    count = save_jsonl(transformed, output_path)
    print(f"Saved {count} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()