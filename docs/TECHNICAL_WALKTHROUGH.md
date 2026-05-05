# Technical Walkthrough: Building EkemainaiAgent

## Overview

This document provides a comprehensive technical walkthrough of how EkemainaiAgent was built—a fine-tuned LLM for codebase intelligence running on AMD GPUs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │   Gradio UI      │    │     FastAPI Backend (main.py)    │   │
│  │   (app.py)       │───▶│     POST /chat, GET /health      │   │
│  │   Port 7860      │    │     Port 8000                    │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL INFERENCE LAYER                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Mistral-7B-Instruct-v0.3 + LoRA Adapter (PEFT)            │ │
│  │  Device: ROCm (AMD MI300X)                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
│  ┌─────────┐  ┌─────────────┐  ┌────────────┐  ┌──────────┐     │
│  │ Raw     │→ │ merge_      │→ │ add_       │→ │ train.py │     │
│  │ Datasets│  │ datasets.py │  │ reasoning_*│  │ (QLoRA)  │     │
│  └─────────┘  └─────────────┘  └────────────┘  └──────────┘     │
│       │              │                  │              │        │
│       ▼              ▼                  ▼              ▼        │
│  800K+ samples   257K merged      256K CoT       Fine-tuned     │
│                   combined_final   cot_final      model         │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CI/CD (GitHub Actions)                      │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐       │
│  │test.yml  │  │dataset_    │  │infra     │  │training  │       │
│  │          │  │pipeline.yml│  │.yml      │  │.yml      │       │
│  └──────────┘  └────────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Dataset Acquisition

### Data Sources

The project combines multiple data sources totaling 800K+ samples:

| Source | Type | Samples |
|--------|------|---------|
| APPS | Coding problems | 5,000 |
| CodeAlpaca | Instructions | 18,019 |
| HumanEval | Python functions | 164 |
| MBPP | Python problems | 500 |
| CodeSearchNet | Multi-language | 119,547 |
| Synthetic | Generated | 88,000+ |
| GitHub Repos | Linux, CPython, Django, etc. | 100,000+ |
| GitHub Discussions | Android, Kotlin, Flutter, etc. | 50,000+ |

### Acquisition Scripts

```bash
# Download core datasets from HuggingFace
python scripts/download_datasets.py

# Clone GitHub repos for code extraction
python scripts/clone_more_repos.py

# Fetch GitHub discussions
python scripts/fetch_discussions.py
```

---

## Step 2: Dataset Merging

### merge_datasets.py

Merges all JSONL files into a single unified dataset with deduplication:

```bash
python scripts/merge_datasets.py --all --no-dedup --output data/combined_final.jsonl
```

**Output:** 257,930 samples in standard JSONL format with `instruction`, `input`, `output` fields.

---

## Step 3: Chain-of-Thought Enhancement

### add_reasoning_v2.py (Optimized CoT)

Adds structured reasoning to each sample:

```bash
python scripts/add_reasoning_v2.py \
  --input data/combined_final.jsonl \
  --output data/cot_final.jsonl
```

### Reasoning Format

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

**Output:** 256,542 Chain-of-Thought enhanced samples.

### add_reasoning.py (Full + Tools)

Adds tool definitions and JSON tool calls to relevant samples.

---

## Step 4: QLoRA Fine-tuning

### train.py

Fine-tunes Mistral-7B-Instruct-v0.3 using QLoRA:

```bash
python scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/cot_final.jsonl \
  --output ./fine-tuned-model \
  --epochs 3 \
  --batch-size 4 \
  --prompt-style cot \
  --max-seq 4096
```

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Learning Rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Max Sequence | 4096 tokens |

### AMD GPU Training

Install PyTorch with ROCm support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

---

## Step 5: Infrastructure Provisioning

### Terraform Configuration

Located in `terraform/main.tf`:

```hcl
# Single MI300X GPU Droplet (192GB VRAM)
resource "digitalocean_droplet" "mi300x_single" {
  name     = "mi300x-training"
  region   = "nyc3"
  image    = "gpu-amd-base"
  size     = "gpu-mi300x1-192gb"
  ssh_keys = [var.ssh_key_fingerprint]
}
```

### Required Variables

| Variable | Description |
|----------|-------------|
| `do_token` | DigitalOcean API token |
| `ssh_key_fingerprint` | SSH key fingerprint |

### Provision via GitHub Actions

The `deploy.yml` workflow automates:
1. Running tests
2. Generating datasets
3. Provisioning GPU infrastructure
4. Running training

---

## Step 6: Deployment

### FastAPI Backend

```bash
python main.py --port 8000 --host 0.0.0.0
```

**Endpoints:**
- `POST /chat` - Chat completions
- `GET /health` - Health check

### Gradio Frontend

```bash
python app.py
```

Runs on port 7860 with chat interface.

### vLLM Production Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./fine-tuned-model \
  --device rocm
```

---

## Step 7: CI/CD Pipeline

### GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test.yml` | push/PR | Unit tests + linting |
| `dataset_pipeline.yml` | manual | Merge + CoT transformation |
| `infrastructure.yml` | manual + workflow_run | Terraform provisioning |
| `training.yml` | manual | Run training |
| `deploy.yml` | manual | Full pipeline (tests → dataset → infra → train) |

---

## Model Capabilities

After fine-tuning with CoT dataset:

- **Chain-of-Thought reasoning** - Step-by-step problem analysis
- **Code generation** - Python, JavaScript, Java, Kotlin, Go, Rust
- **Code explanation** - Detailed documentation and comments
- **Bug detection** - Identify issues and suggest fixes
- **Code optimization** - Performance improvements
- **Tool invocation** - python_executor, file_reader, web_search, git_clone, code_search

---

## Costs

| Resource | Cost |
|----------|------|
| Single MI300X (192GB) | ~$1.99/hr |
| 8x MI300X Cluster | ~$15.92/hr |
| DigitalOcean Volume (100GB) | $5/mo |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ekemainai/EkemainaiAgent.git
cd EkemainaiAgent

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python scripts/merge_datasets.py --all

# Train model
python scripts/train.py --dataset data/combined_final.jsonl

# Run locally
python main.py  # Backend
python app.py   # Frontend
```

---

## Conclusion

EkemainaiAgent demonstrates a complete, production-ready LLM fine-tuning pipeline running entirely on AMD GPUs. From dataset acquisition to deployment, every component is automated and open-source. This enables developers to build custom AI coding assistants without proprietary dependencies.