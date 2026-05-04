#!/usr/bin/env python3
"""Generate synthetic training data for demonstration."""
import json
import random
import os

PYTHON_EXAMPLES = [
    {
        "instruction": "Write a function to reverse a string",
        "input": "Create a function that reverses the input string",
        "output": "def reverse_string(s): return s[::-1]"
    },
    {
        "instruction": "Write a function to check if a number is prime",
        "input": "Create a function that checks for primality",
        "output": "def is_prime(n): return n > 1 and all(n % i for i in range(2, int(n**0.5) + 1))"
    },
    {
        "instruction": "Write a function to find factorial",
        "input": "Create a function that calculates factorial",
        "output": "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)"
    },
    {
        "instruction": "Explain what this code does",
        "input": "def fetch_data(url): return requests.get(url).json()",
        "output": "This function makes an HTTP GET request to the specified URL and returns the JSON response."
    },
    {
        "instruction": "Find bugs in this code",
        "input": "def divide(a, b): return a / b",
        "output": "Bug: No check for division by zero. Add: if b == 0: raise ValueError('Cannot divide by zero')"
    },
]

KOTLIN_EXAMPLES = [
    {
        "instruction": "Write a Kotlin function to reverse a string",
        "input": "Create a function that reverses a string in Kotlin",
        "output": "fun reverseString(s: String) = s.reversed()"
    },
    {
        "instruction": "Write a Jetpack Compose Composable",
        "input": "Create a simple greeting Composable",
        "output": "@Composable fun Greeting(name: String) { Text(\"Hello $name\") }"
    },
    {
        "instruction": "Explain this Kotlin code",
        "input": "val flow = flowOf(1, 2, 3).map { it * 2 }",
        "output": "This creates a Kotlin Flow that emits 1, 2, 3 and transforms each value by multiplying by 2."
    },
]

ANDROID_EXAMPLES = [
    {
        "instruction": "Create an Android ViewModel",
        "input": "Create a ViewModel for a counter app",
        "output": "class CounterViewModel : ViewModel() { private val _count = MutableLiveData(0); val count: LiveData get() = _count; fun increment() { _count.value = (_count.value ?: 0) + 1 } }"
    },
    {
        "instruction": "Write a Retrofit API interface",
        "input": "Create a Retrofit interface for fetching users",
        "output": "interface UserApi { @GET(\"users/{id}\") suspend fun getUser(@Path(\"id\") id: Int): User }"
    },
]

SAMPLE_DATA = PYTHON_EXAMPLES + KOTLIN_EXAMPLES + ANDROID_EXAMPLES


def generate_synthetic_data(num_samples=1000):
    """Generate synthetic training data."""
    samples = []
    
    for _ in range(num_samples):
        template = random.choice(SAMPLE_DATA)
        sample = {
            "instruction": template["instruction"],
            "input": template["input"],
            "output": template["output"]
        }
        samples.append(sample)
    
    return samples


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output", default="data/training.jsonl")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    
    data = generate_synthetic_data(args.samples)
    save_jsonl(data, args.output)
    print(f"Generated {len(data)} samples")


if __name__ == "__main__":
    main()