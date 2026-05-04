#!/usr/bin/env python3
"""Transform dataset to add Chain-of-Thought reasoning and tool-use capabilities."""
import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Tool definitions for code assistant
TOOL_DEFINITIONS = {
    "python_executor": {
        "name": "python_executor",
        "description": "Execute Python code and return the result. Use for running calculations, data processing, and algorithm implementation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    },
    "file_reader": {
        "name": "file_reader", 
        "description": "Read contents of a file from the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full path to file"},
                "lines": {"type": "integer", "description": "Number of lines to read (default 100)"}
            },
            "required": ["path"]
        }
    },
    "file_writer": {
        "name": "file_writer",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full path to file"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information, documentation, or code examples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    "git_clone": {
        "name": "git_clone",
        "description": "Clone a GitHub repository to get code samples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_url": {"type": "string", "description": "GitHub repository URL"}
            },
            "required": ["repo_url"]
        }
    },
    "code_search": {
        "name": "code_search",
        "description": "Search code in a local repository for specific patterns or functions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search"},
                "path": {"type": "string", "description": "Directory to search in"}
            },
            "required": ["pattern"]
        }
    }
}

REASONING_TEMPLATES = [
    "Let me think about this step by step:\n",
    "I'll analyze this systematically:\n",
    "Here's my reasoning process:\n",
    "Let me break this down:\n",
    "First, I'll understand the problem, then solve it:\n",
]

THINK_PATTERNS = [
    "Let me think about this. The input suggests we need to {action}. ",
    "Looking at the problem: {problem}. ",
    "I need to consider {consideration}. ",
    "The approach should be to {approach}. \n",
    "Let me analyze what we have: {input_summary}. ",
    "My strategy: {strategy}. ",
]

def add_chain_of_thought(output: str, instruction: str, input_text: str) -> str:
    """Add Chain-of-Thought reasoning to the output."""
    action = determine_action(instruction)
    problem = summarize_problem(input_text)
    strategy = generate_strategy(instruction, output)
    consideration = generate_consideration(instruction)
    input_summary = summarize_input(input_text)
    
    reasoning = [
        f"Let me analyze this problem step by step:\n",
        f"Step 1: Understand what needs to be done - {problem}\n",
        f"Step 2: Determine the approach - {strategy}\n", 
        f"Step 3: Consider edge cases - {consideration}\n",
        f"Step 4: Implement the solution\n",
        f"\nBased on this reasoning:\n",
    ]
    
    # Add execution trace if code in output
    if "def " in output or "class " in output or "import " in output:
        code_match = re.search(r'```python\n(.*?)```', output, re.DOTALL)
        if not code_match:
            code_match = re.search(r'(def .+|class .+|import .+)', output, re.MULTILINE)
        
        if code_match:
            reasoning.append("\nExecution trace:\n")
            reasoning.append(f"  -> Initializing...\n")
            reasoning.append(f"  -> Processing input...\n")
            reasoning.append(f"  -> Generating output...\n")
    
    reasoning.append(f"\n{output}")
    return "".join(reasoning)

def determine_action(instruction: str) -> str:
    """Determine what action the instruction requires."""
    inst_lower = instruction.lower()
    if "write" in inst_lower or "create" in inst_lower or "implement" in inst_lower:
        return "create new code"
    elif "fix" in inst_lower or "debug" in inst_lower or "bug" in inst_lower:
        return "debug existing code"
    elif "explain" in inst_lower or "describe" in inst_lower:
        return "explain code"
    elif "optimize" in inst_lower or "improve" in inst_lower:
        return "optimize code"
    elif "find" in inst_lower or "search" in inst_lower:
        return "find/search code"
    else:
        return "process the request"

def summarize_problem(text: str) -> str:
    """Summarize the problem from input text."""
    if not text:
        return "process the given instruction"
    text = text[:100]
    return text.replace("\n", " ").strip()

def generate_strategy(instruction: str, output: str) -> str:
    """Generate strategy based on instruction and output type."""
    inst_lower = instruction.lower()
    if "write" in inst_lower or "create" in inst_lower:
        return "write the code from scratch following best practices"
    elif "fix" in inst_lower or "bug" in inst_lower:
        return "identify and fix the issue"
    elif "explain" in inst_lower:
        return "explain the code clearly with examples"
    elif "optimize" in inst_lower:
        return "improve performance and readability"
    else:
        return "generate appropriate solution"

def generate_consideration(instruction: str) -> str:
    """Generate edge case considerations."""
    inst_lower = instruction.lower()
    if "sort" in inst_lower:
        return "empty lists, single elements, duplicate values"
    elif "search" in inst_lower or "find" in inst_lower:
        return "element not found, empty input"
    elif "reverse" in inst_lower:
        return "empty strings, single characters"
    elif "parse" in inst_lower or "validate" in inst_lower:
        return "invalid input, edge cases, error handling"
    else:
        return "basic functionality, edge cases, error handling"

def summarize_input(text: str) -> str:
    """Summarize the input."""
    if not text:
        return "no additional input provided"
    text = text[:80].replace("\n", " ")
    return text.strip()

def create_tool_calling_example(instruction: str, input_text: str, output: str) -> str:
    """Create a tool calling example based on the task."""
    inst_lower = instruction.lower()
    
    # Determine which tool to use
    if "fetch" in inst_lower or "api" in inst_lower or "http" in inst_lower:
        tool = "web_search"
        args = {"query": input_text[:100] if input_text else instruction}
    elif "read" in inst_lower or "file" in inst_lower:
        tool = "file_reader"
        args = {"path": extract_file_path(input_text) or "example.py", "lines": 50}
    elif "save" in inst_lower or "write" in inst_lower or "create" in inst_lower:
        tool = "file_writer"
        args = {"path": "output.py", "content": output[:200]}
    elif "search" in inst_lower or "find" in inst_lower:
        tool = "code_search"
        args = {"pattern": extract_pattern(instruction), "path": "."}
    elif "clone" in inst_lower or "repo" in inst_lower or "github" in inst_lower:
        tool = "git_clone"
        args = {"repo_url": extract_github_url(input_text) or "https://github.com/example/repo"}
    else:
        tool = "python_executor"
        args = {"code": output[:200] if output else "pass"}
    
    return json.dumps({"tool": tool, "args": args})

def extract_file_path(text: str) -> Optional[str]:
    """Extract file path from text."""
    patterns = [
        r'["\']([\w/]+\.py)["\']',
        r'([\w/]+\.py)',
    ]
    for p in patterns:
        match = re.search(p, text or "")
        if match:
            return match.group(1)
    return None

def extract_pattern(text: str) -> Optional[str]:
    """Extract search pattern from text."""
    if not text:
        return "def .*"
    words = text.split()
    if words:
        return f"def {words[0]}"
    return "def .*"

def extract_github_url(text: str) -> Optional[str]:
    """Extract GitHub URL from text."""
    if not text:
        return None
    match = re.search(r'github\.com/[\w-]+/[\w-]+', text)
    if match:
        return f"https://{match.group()}"
    match = re.search(r'https?://[\w./-]+', text)
    if match and "github" in match.group().lower():
        return match.group()
    return None

def transform_sample(sample: Dict, add_cot: bool = True, add_tools: bool = True) -> Dict:
    """Transform a single sample with CoT and tool-use."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    # Skip empty samples
    if not output or len(output) < 10:
        return None
    
    new_instruction = instruction
    new_input = input_text
    new_output = output
    
    # Add Chain-of-Thought
    if add_cot:
        new_output = add_chain_of_thought(output, instruction, input_text)
    
    # Add tool definitions and calling format
    if add_tools:
        tool_defs = json.dumps(list(TOOL_DEFINITIONS.values()), indent=2)
        new_input = f"{input_text}\n\n## Available Tools\n```json\n{tool_defs}\n```\n\n## Your Task\n{instruction}"
        new_instruction = "Use Chain-of-Thought reasoning and available tools to complete this task."
        
        # Add tool calling to output
        if add_cot:
            tool_call = create_tool_calling_example(instruction, input_text, output)
            new_output = f"{new_output}\n\n## Tool Call\n```json\n{tool_call}\n```"
    
    return {
        "instruction": new_instruction,
        "input": new_input,
        "output": new_output
    }

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
    parser = argparse.ArgumentParser(description="Add Chain-of-Thought and tools to dataset")
    parser.add_argument("--input", default="data/combined_final.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/reasoning_final.jsonl", help="Output JSONL file")
    parser.add_argument("--no-cot", action="store_true", help="Skip Chain-of-Thought")
    parser.add_argument("--no-tools", action="store_true", help="Skip tool definitions")
    parser.add_argument("--limit", type=int, help="Limit samples")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading {input_path}...")
    samples = load_jsonl(input_path)
    print(f"Loaded {len(samples)} samples")
    
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")
    
    print(f"Transforming with Chain-of-Thought: {not args.no_cot}")
    print(f"Transforming with Tools: {not args.no_tools}")
    
    transformed = []
    for i, sample in enumerate(samples):
        if i % 10000 == 0:
            print(f"Processing {i}/{len(samples)}...")
        
        new_sample = transform_sample(
            sample, 
            add_cot=not args.no_cot,
            add_tools=not args.no_tools
        )
        if new_sample:
            transformed.append(new_sample)
    
    print(f"Transformed to {len(transformed)} samples")
    
    count = save_jsonl(transformed, output_path)
    print(f"Saved {count} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()