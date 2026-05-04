#!/usr/bin/env python3
"""Generate synthetic code training data."""
import json
import random
import os

CATEGORIES = [
    # Python functions
    ("python", "Write a Python function to {task}", [
        "reverse a string",
        "check if a number is prime",
        "find the factorial of a number",
        "sort a list in ascending order",
        "find the maximum element in a list",
        "calculate the sum of a list",
        "check if a string is a palindrome",
        "remove duplicates from a list",
        "find the index of an element",
        "count occurrences of each element",
    ]),
    ("python", "Implement a Python class for {task}", [
        "a stack data structure",
        "a queue data structure", 
        "a linked list",
        "a binary search tree",
        "a caching system",
    ]),
    ("python", "Fix the bug in this Python code", [
        "def divide(a, b): return a / b",
        "for i in range(10): print(i)",
        "def foo(l): return l[0]",
    ]),
    ("python", "Optimize this Python code for performance", [
        "result = [x for x in range(1000) if x % 2 == 0]",
        "for i in range(len(l)): print(l[i])",
    ]),
    # JavaScript
    ("javascript", "Write a JavaScript function to {task}", [
        "reverse an array",
        "check if a string is a palindrome",
        "find the maximum value",
        "remove duplicates",
    ]),
    ("javascript", "Implement a JavaScript class for {task}", [
        "a simple counter",
        "a storage manager",
        "an event handler",
    ]),
    # Java
    ("java", "Write a Java method to {task}", [
        "reverse a string",
        "sort an array",
        "find the largest element",
    ]),
    ("java", "Implement a Java class for {task}", [
        "a vehicle class",
        "a person class",
        "a bank account",
    ]),
    # Kotlin
    ("kotlin", "Write a Kotlin function to {task}", [
        "reverse a string",
        "filter a list",
        "map values",
    ]),
    ("kotlin", "Write a Kotlin coroutine for {task}", [
        "async data fetching",
        "parallel processing",
    ]),
    # General
    ("general", "Explain what this code does", [
        "def foo(): return 42",
        "class Bar: pass",
        "import os",
    ]),
    ("general", "Find performance issues in", [
        "nested loop code",
        "repeated database calls",
        "inefficient sorting",
    ]),
]

def generate_samples(count=5000):
    """Generate synthetic samples."""
    samples = []
    for _ in range(count):
        cat, template, tasks = random.choice(CATEGORIES)
        task = random.choice(tasks)
        
        instruction = template.format(task=task)
        
        # Generate input based on category
        if cat == "python":
            input_code = f"# {task}\n" + random.choice([
                "def solution():\n    pass",
                "def process(data):\n    return data",
                "class Handler:\n    def handle(self):\n        pass",
            ])
            output = "# Implementation here\n"
        elif cat == "javascript":
            input_code = f"// {task}\nfunction process() {{\n  // code here\n}}"
            output = "// Implementation\n"
        elif cat == "java":
            input_code = f"// {task}\npublic void process() {{\n    // code\n}}"
            output = "// Implementation\n"
        elif cat == "kotlin":
            input_code = f"// {task}\nfun process() {{\n    // code\n}}"
            output = "// Implementation\n"
        else:
            input_code = f"# {task}"
            output = "# Explanation"
        
        samples.append({
            "instruction": instruction,
            "input": input_code[:500],
            "output": output[:500],
        })
    
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--output", default="data/synthetic.jsonl")
    args = parser.parse_args()
    
    samples = generate_samples(args.count)
    
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Generated {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()