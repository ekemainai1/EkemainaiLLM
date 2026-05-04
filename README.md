# Codebase Intelligence Assistant

A fine-tuned LLM system for understanding specific codebases and answering developer questions accurately. Built with a focus on mobile/Android development.

## Project Overview

This project trains a Mistral-7B-based coding assistant using QLoRA fine-tuning on AMD ROCm (MI300X). The model specializes in code generation, code understanding, and developer assistance, with particular focus on Android/Jetpack Compose development.

**Stack**: Mistral 7B + PyTorch + ROCm + QLoRA + vLLM + FastAPI + Gradio

---

## Architecture

```
EkemainaiAgent/
├── main.py                 # FastAPI backend server
├── app.py                 # Gradio frontend interface
├── repo2dataset.py       # CLI for generating datasets from GitHub repos
├── github_ingest.py      # GitHub API ingestion utilities
├── requirements.txt      # Python dependencies
├── terraform/            # Terraform configs for AMD GPU provisioning
├── scripts/
│   ├── train.py           # QLoRA fine-tuning script
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── process_datasets.py # Dataset processing and merging
│   ├── merge_datasets.py  # Merge multiple JSONL datasets
│   ├── add_reasoning.py   # Add CoT + tools + tool calls
│   ├── add_reasoning_v2.py # Add CoT reasoning (optimized)
│   ├── download_datasets.py # Download core datasets from HuggingFace
│   ├── convert_apps.py    # Convert APPS dataset to instruction format
│   ├── generate_synthetic.py # Generate synthetic training data
│   ├── ingest_github_repos.py # Clone and extract code from GitHub repos
│   ├── clone_more_repos.py # Clone popular repositories for training
│   ├── fetch_discussions.py # Fetch GitHub discussions for reasoning
│   └── download_extra.py  # Download Flask/Gunicorn samples
└── data/                # Training datasets (JSONL format)
   ├── combined_final.jsonl   # 257K samples (all sources)
    ├── cot_final.jsonl      # 256K samples (CoT enabled)
    └── reasoning_final.jsonl # 256K samples (full CoT + tools)
```

---

## Dataset Sources

The training data consists of **800,000+ samples** from the following sources:

### Core Datasets (HuggingFace)

| Dataset | Source | Samples | Description |
|---------|--------|---------|-------------|
| **APPS** | `princeton-nlp/APPS` | 5,000 | Python coding problems from Codeforces, AtCoder, etc. with difficulty ratings |
| **CodeAlpaca** | `HuggingFaceH4/CodeAlpaca_20K` | 18,019 | Instruction-following code generation dataset |
| **HumanEval** | `openai/openai_humaneval` | 164 | Python function completion benchmark |
| **MBPP** | `google-research-datasets/mbpp` | 500 | Mostly Basic Python Programming problems |
| **CodeSearchNet** | `code_search_net` | 119,547 | Python, Java, Go, JavaScript code with documentation |

### Synthetic Datasets

| Dataset | Samples | Generation Method |
|---------|---------|-------------------|
| **synthetic.jsonl** | 20,000 | Template-based Python/JS/Java/Kotlin function generation |
| **synthetic2.jsonl** | 30,000 | Template-based code generation with varied tasks |
| **synthetic3.jsonl** | 25,000 | Template-based code generation with bug fixing |
| **synthetic4.jsonl** | 13,000 | Template-based optimization tasks |

### GitHub Code Datasets

| Dataset | Samples | Source Repositories |
|---------|---------|---------------------|
| **github_code.jsonl** | 993 | torvalds/linux, python/cpython, django/django, jmportnet/Prophet |
| **more_repos.jsonl** | 1,000 | golang/go, rust-lang/rust, facebook/react-native, flutter/flutter |
| **more_repos2.jsonl** | 921 | ansible/ansible, pandas-dev/pandas, numpy/numpy, requests, etc. |
| **flask_samples.jsonl** | 225 | github.com/pallets/flask |
| **gunicorn_samples.jsonl** | 1,080 | github.com/benoitc/gunicorn |

### GitHub Discussions

| Dataset | Samples | Source |
|---------|---------|--------|
| **github_discussions.jsonl** | 581 | Android, Kotlin, Flutter, React Native, Vue, Django, PyTorch discussions |

### Additional Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| **python.jsonl** | 18,000 | `iamtarun/python_code_instructions_18k_alpaca` |
| **software_eng.jsonl** | 500 | Software engineering concepts |
| **leetcodesmall.jsonl** | 1,000 | LeetCode problems |
| **leetcodesmall2.jsonl** | 2,000 | Additional LeetCode problems |
| **ossinstruct.jsonl** | 100 | `njucc/OSS-Instruct` |
| **mbpp_extra.jsonl** | 300 | Extra MBPP-like samples |

### Combined Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| **combined.jsonl** | 141,069 | Merged core datasets |
| **combined_v2_*.jsonl** | 87,407 each | Version 2 combined |
| **combined_final.jsonl** | 257,930 | Final merged training set (all sources) |
| **cot_final.jsonl** | 256,542 | Chain-of-Thought enabled |
| **reasoning_final.jsonl** | 256,542 | Full CoT + tools + tool calls |

---

## Chain-of-Thought and Tool-Use Enhancement

The base model was enhanced with **Chain-of-Thought (CoT) reasoning** and **tool-use capabilities** through dataset transformation.

### Enhancement Scripts

| Script | Purpose |
|--------|---------|
| `scripts/add_reasoning.py` | Add CoT + tool definitions + JSON tool calls |
| `scripts/add_reasoning_v2.py` | Add CoT reasoning only (optimized) |
| `scripts/merge_datasets.py` | Merge multiple datasets |

### Available Tools

The model has access to these tools for enhanced task completion:

| Tool | Description |
|------|-------------|
| **python_executor** | Execute Python code and return results |
| **file_reader** | Read contents from a file |
| **file_writer** | Write content to a file |
| **web_search** | Search the web for information |
| **git_clone** | Clone GitHub repositories |
| **code_search** | Search code patterns in repositories |

### Tool Schema Format

Each tool is defined with JSON schema:

```json
{
  "name": "python_executor",
  "description": "Execute Python code and return the result",
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "Python code to execute"}
    },
    "required": ["code"]
  }
}
```

### Chain-of-Thought Reasoning Format

Each output now includes step-by-step reasoning:

```markdown
## Problem Analysis
Let me analyze this step by step:
1. Problem: [what needs to be done]
2. Approach: [solution strategy]
3. Edge cases: [handling edge cases]
4. Implementation: [how it's solved]

[original code output]

## Execution Trace
-> Initializing...
-> Processing input...
-> Generating output...
-> Complete!
```

### Tool Calling Format

Tool calls are formatted as JSON:

```json
{"tool": "python_executor", "args": {"code": "def reverse_string(s): return s[::-1]"}}
```

### How It Works

1. **Problem Classification**: Each sample is classified by problem type (code_generation, bug_fix, explanation, optimization, search)
2. **Reasoning Generation**: Appropriate reasoning steps are generated based on problem type
3. **Output Transformation**: Original output is wrapped with CoT reasoning
4. **Tool Integration** (optional): Tool definitions and calls are added to relevant samples

### Usage

```bash
# Create reasoning-enabled dataset (CoT only, optimized)
python3 scripts/add_reasoning_v2.py \
  --input data/combined_final.jsonl \
  --output data/cot_final.jsonl

# Create full dataset with tools and tool calls
python3 scripts/add_reasoning.py \
  --input data/combined_final.jsonl \
  --output data/reasoning_final.jsonl

# Merge specific datasets first
python3 scripts/merge_datasets.py \
  --groups core synthetic github \
  --output data/my_training.jsonl
```

### Dataset Sizes After Enhancement

| Dataset | Samples | Size |
|--------|---------|------|
| `combined_final.jsonl` | 257,930 | 136 MB |
| `cot_final.jsonl` | 256,542 | 208 MB |
| `reasoning_final.jsonl` | 256,542 | 991 MB |

The reasoning-enhanced datasets are ready for fine-tuning to produce models with:

---

## Dataset Format

All datasets follow the instruction-tuning format:

```json
{
  "instruction": "Task description or question",
  "input": "Context, code, or problem statement",
  "output": "Expected response or solution"
}
```

---

## Training Configuration

### Model
- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Quantization**: 4-bit NF4 (QLoRA)

### QLoRA Parameters
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Dropout**: 0.05

### Training Arguments
- **Epochs**: 3
- **Per-device batch size**: 4
- **Gradient accumulation**: 4
- **Learning rate**: 2e-4
- **Warmup**: 5%
- **Max sequence length**: 4,096 tokens
- **Optimizer**: paged_adamw_8bit

---

## Installation

```bash
# Install dependencies
pip install torch transformers accelerate peft datasets optimum vllm fastapi uvicorn sentence-transformers faiss-cpu requests tqdm

# Set environment variables
export GITHUB_TOKEN=your_github_token
export HF_TOKEN=your_huggingface_token
```

---

## Usage

### Dataset Generation

```bash
# Generate dataset from GitHub repository
python repo2dataset.py --repo https://github.com/android/compose-samples --output data/compose.jsonl --workers 8

# Generate with GitHub discussions
python repo2dataset.py --repo https://github.com/android/compose-samples --include-discussions --output data/compose.jsonl
```

### Training

```bash
# Fine-tune model on AMD ROCm
python scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/combined_final.jsonl \
  --output ./fine-tuned-model \
  --epochs 3 \
  --batch_size 4
```

### Evaluation

```bash
# Run benchmark evaluation
python scripts/evaluate.py --model ./fine-tuned-model --test-set data/test_set.jsonl

# Evaluate on specific benchmark
python scripts/evaluate.py --benchmark humaneval
python scripts/evaluate.py --benchmark mbpp
```

---

## Deployment

### Start FastAPI Backend

```bash
python main.py --model-dir ./fine-tuned-model --host 0.0.0.0 --port 8000
```

### Start vLLM Server (for production)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./fine-tuned-model \
  --device rocm \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000
```

### Start Gradio Frontend

```bash
python app.py
```

---

## API Endpoints

### POST /chat

```json
{
  "instruction": "Write a function to reverse a string",
  "input": "",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

Response:
```json
{
  "response": "def reverse_string(s): return s[::-1]",
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "device": "cuda"
}
```

### GET /health

Returns service health status.

---

## Benchmark Targets

| Benchmark | Pass@1 Target |
|-----------|---------------|
| HumanEval | > 50% |
| MBPP | > 55% |
| Android/Kotlin | > 40% |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub API token for rate limit increase |
| `HF_TOKEN` | HuggingFace access token for dataset downloads |
| `MODEL_DIR` | Path to fine-tuned model (default: `./fine-tuned-model`) |

---

## Project Timeline

| Phase | Status |
|-------|--------|
| Dataset Acquisition | ✅ Complete (257K samples) |
| Data Processing + Merging | ✅ Complete |
| CoT + Tools Enhancement | ✅ Complete |
| Fine-tuning | 🔲 Pending (requires AMD ROCm) |
| Evaluation | 🔲 Pending |
| Deployment | 🔲 Pending |

---

## Model Capabilities After Fine-Tuning

The model will have:

- **Chain-of-Thought Reasoning**: Step-by-step problem analysis in outputs
- **Tool Use**: Ability to call appropriate tools (python_executor, file_reader, web_search, etc.)
- **Code Generation**: Python, JavaScript, Java, Kotlin, Go, Rust
- **Code Understanding**: Explanation, documentation generation
- **Bug Fixing**: Identification and correction of issues
- **Code Optimization**: Performance and readability improvements

---

## References

### Datasets
- APPS: https://github.com/princeton-nlp/APPS
- CodeAlpaca: https://github.com/huggingface/CodeAlpaca
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research-datasets/mbpp
- CodeSearchNet: https://github.com/github/CodeSearchNet
- OSS-Instruct: https://github.com/njucc/lei/OSS-Instruct

### Libraries
- PyTorch: https://pytorch.org/
- QLoRA: https://github.com/artidoro/qlora
- vLLM: https://github.com/vllm-project/vllm
- FastAPI: https://fastapi.tiangolo.com/
- Gradio: https://gradio.app/

---

## License

MIT License
