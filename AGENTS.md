# Codebase Intelligence Assistant - Agent Guidelines

## Project Overview

Fine-tuned LLM system with Chain-of-Thought reasoning and tool-use capabilities for understanding codebases and answering developer questions.

**Stack**: Mistral 7B + PyTorch + ROCm + QLoRA + vLLM + FastAPI + Gradio

---

## Key Commands

### Development
```bash
# Install dependencies
pip install torch transformers accelerate peft datasets optimum vllm fastapi uvicorn sentence-transformers faiss-cpu requests tqdm python-dotenv

# Merge datasets
python3 scripts/merge_datasets.py --all --output data/combined_final.jsonl

# Add Chain-of-Thought reasoning (optimized)
python3 scripts/add_reasoning_v2.py \
  --input data/combined_final.jsonl \
  --output data/cot_final.jsonl

# Add full reasoning + tools
python3 scripts/add_reasoning.py \
  --input data/combined_final.jsonl \
  --output data/reasoning_final.jsonl

# Fine-tune with CoT dataset
python3 scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/cot_final.jsonl \
  --output ./fine-tuned-model \
  --epochs 3 \
  --batch_size 4 \
  --max-seq 4096 \
  --prompt-style cot
```

### Training Options
```bash
# Standard training (without CoT)
python3 scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/combined_final.jsonl \
  --output ./fine-tuned-model

# Chain-of-Thought training
python3 scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/cot_final.jsonl \
  --output ./fine-tuned-cot \
  --prompt-style cot \
  --max-seq 4096

# Reasoning + Tools training
python3 scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/reasoning_final.jsonl \
  --output ./fine-tuned-reasoning \
  --prompt-style cot \
  --max-seq 4096
```

### Generate Dataset from Repo
```bash
python repo2dataset.py --repo <GITHUB_URL> --output dataset.jsonl --workers 8 --include-discussions
```

### Run vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server --model ./fine-tuned-model --device rocm
```

### Testing
```bash
# Run evaluation tests
python scripts/evaluate.py --model ./fine-tuned-model --test-set test_set.jsonl
```

---

## Code Conventions

- **File extensions**: Python (`.py`), Kotlin (`.kt`), Java (`.java`), JavaScript (`.js`)
- **Dataset format**: JSONL with `instruction`, `input`, `output` fields
- **Code chunking**: 40 lines max per chunk
- **Max tokens**: 4096 for CoT datasets (longer outputs)
- **QLoRA config**: `r=16`, `lora_alpha=32`, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`

---

## Supported File Types

`.py`, `.js`, `.ts`, `.java`, `.kt`, `.cpp`, `.go`, `.rs`

---

## Project Structure

```
EkemainaiAgent/
├── AGENTS.md              # This file
├── README.md              # Project documentation
├── main.py               # FastAPI backend
├── app.py                # Gradio frontend
├── repo2dataset.py        # Dataset generator CLI
├── github_ingest.py      # GitHub API ingestion
├── requirements.txt     # Dependencies
├── terraform/           # AMD GPU Terraform configs
├── scripts/
│   ├── train.py           # QLoRA fine-tuning (supports CoT)
│   ├── evaluate.py       # Benchmark evaluation
│   ├── merge_datasets.py # Merge multiple JSONL files
│   ├── add_reasoning.py   # Add CoT + tools + tool calls
│   ├── add_reasoning_v2.py # Add CoT reasoning (optimized)
│   ├── process_datasets.py # Dataset processing
│   ├── download_datasets.py # Download HuggingFace datasets
│   └── ...
└─�� data/
    ├── combined_final.jsonl  # 257K merged samples
    ├── cot_final.jsonl         # 256K CoT reasoning samples
    └── reasoning_final.jsonl  # 256K full reasoning + tools
```

---

## Dataset Options

| Dataset | Samples | Purpose |
|---------|---------|---------|
| `combined_final.jsonl` | 257,930 | Standard training |
| `cot_final.jsonl` | 256,542 | Chain-of-Thought reasoning |
| `reasoning_final.jsonl` | 256,542 | Full tools + tool calls |

---

## Chain-of-Thought & Tools

The model includes reasoning and tool-use capabilities through dataset transformation.

### Available Tools
- `python_executor` - Execute Python code
- `file_reader` - Read files
- `file_writer` - Write files
- `web_search` - Search web
- `git_clone` - Clone GitHub repos
- `code_search` - Search code patterns

### Output Format (CoT)
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

---

## Deployment

```bash
# Start backend
python main.py --model-dir ./fine-tuned-model --host 0.0.0.0 --port 8000

# Start vLLM (AMD GPU)
python -m vllm.entrypoints.openai.api_server --model ./fine-tuned-model --device rocm --tensor-parallel-size 1

# Start Gradio frontend
python app.py
```

---

## Environment Variables

- `GITHUB_TOKEN` - GitHub API rate limit token
- `HF_TOKEN` - Hugging Face access token
- `MODEL_DIR` - Path to fine-tuned model (default: `./fine-tuned-model`)

---

## Implementation Status

### ✅ Completed
1. Dataset Acquisition (257K+ samples from multiple sources)
2. Dataset Merging (`merge_datasets.py`)
3. Chain-of-Thought Enhancement (`add_reasoning.py`, `add_reasoning_v2.py`)
4. Model Training Script (updated with CoT support)
5. Terraform configs for AMD GPU provisioning

### 🔲 Pending
1. Fine-tuning on AMD ROCm (MI300X)
2. Evaluation
3. Deployment

---

## Training on AMD Developer Cloud

### Option 1: Manual Provisioning
1. Create GPU Droplet at amd.cloud
2. SSH into instance
3. Run training script

### Option 2: Terraform
```bash
cd terraform
terraform init
terraform plan -var="do_token=$DO_TOKEN" -var="ssh_key_fingerprint=xx:xx"
terraform apply -var="do_token=$DO_TOKEN" -var="ssh_key_fingerprint=xx:xx"
```

### GPU Options
| Config | Slug | VRAM | Cost |
|--------|------|------|------|
| Single MI300X | `gpu-mi300x1-192gb` | 192 GB | ~$1.99/hr |
| 8x MI300X | `gpu-mi300x8-1536gb` | 1,536 GB | ~$15.92/hr |

---

## Expected Model Capabilities

After fine-tuning with CoT dataset:
- Chain-of-Thought reasoning in outputs
- Step-by-step problem analysis
- Appropriate tool selection and usage
- Code generation (Python, JS, Java, Kotlin)
- Code explanation and documentation
- Bug identification and fixing
- Code optimization

---

## CI/CD Pipelines

GitHub Actions workflows for automation:

### Workflow Files
```
.github/workflows/
├── ci.yml                    # Main CI (combined)
├── test.yml                   # Unit tests + lint
├── dataset.yml               # Dataset validation
├── dataset_pipeline.yml      # Dataset processing
├── training.yml              # Model training
├── quality.yml              # Code quality checks
└── WORKFLOWS.md            # Workflow documentation
```

### Usage

```bash
# Run tests locally
pytest tests/ -v

# Run CI locally (simulated)
gh workflow run test.yml

# Trigger dataset pipeline
gh workflow run dataset_pipeline.yml -f action=all

# Trigger training (requires AMD GPU)
gh workflow run training.yml \
  -f model=mistralai/Mistral-7B-Instruct-v0.3 \
  -f dataset=data/cot_final.jsonl \
  -f epochs=3
```

### Pipeline Summary

| Pipeline | Trigger | Description |
|----------|---------|-------------|
| test.yml | push/PR | Unit tests, linting |
| dataset.yml | data/ push | Validate JSONL format |
| dataset_pipeline.yml | manual | Merge → CoT → Reasoning |
| training.yml | manual | Train model |
| quality.yml | push/PR | Code quality checks |