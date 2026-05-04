# Build in Public: EkemainaiAgent Technical Journey

## AMD Developer Hackathon 2025 - Build in Public Documentation

### Project Overview

**EkemainaiAgent** is a fine-tuned code assistant using Mistral 7B with Chain-of-Thought reasoning and tool-use capabilities, optimized for AMD ROCm (MI300X).

---

## Phase 1: Dataset Acquisition & Processing

### What We Built

**26 dataset files** from multiple sources totaling **257,930 training samples**:

| Source | Samples | Purpose |
|--------|---------|---------|
| APPS (HuggingFace) | 5,000 | Python coding problems |
| CodeAlpaca | 18,019 | Instruction-following code |
| HumanEval + MBPP | 664 | Function completion |
| CodeSearchNet | 119,547 | Code with docs |
| Synthetic | 88,000 | Template-based generation |
| GitHub repos | 4,800 | Real-world code |
| GitHub discussions | 581 | Reasoning data |
| Additional | 21,319 | LeetCode, Python, etc. |

### Key Challenges Solved

1. **Dataset format standardization**: Created `merge_datasets.py` to merge multiple JSONL files with deduplication
2. **Memory efficiency**: Processed 257K samples without loading all into memory
3. **Quality filtering**: Removed duplicates using instruction+input key hashing

### Technical Insights

```bash
# Merge all datasets with one command
python scripts/merge_datasets.py --all --output data/combined_final.jsonl
# Output: 257,930 samples in 136 MB
```

**Lesson**: Batch processing and streaming are essential for large datasets.

---

## Phase 2: Chain-of-Thought Reasoning

### What We Added

Enhanced the dataset with **Chain-of-Thought reasoning** through transformation:

| Feature | Description |
|--------|-------------|
| Problem Analysis | 4-step breakdown (Problem → Approach → Edge Cases → Implementation) |
| Execution Trace | Visual trace showing code execution flow |
| Tool Definitions | JSON schemas for 6 tools |

### Tools Added

```python
TOOL_DEFINITIONS = {
    "python_executor": {"description": "Execute Python code", "args": {"code": "string"}},
    "file_reader": {"description": "Read files", "args": {"path": "string", "lines": "int"}},
    "file_writer": {"description": "Write files", "args": {"path": "string", "content": "string"}},
    "web_search": {"description": "Search web", "args": {"query": "string"}},
    "git_clone": {"description": "Clone GitHub repos", "args": {"repo_url": "string"}},
    "code_search": {"description": "Search code patterns", "args": {"pattern": "string"}},
}
```

### Output Format Example

```markdown
## Problem Analysis
Let me analyze this step by step:
1. Problem: Write a function to add numbers
2. Approach: Write clean, efficient code
3. Edge cases: Input validation, error handling
4. Implementation: Generate complete solution

def add(a, b):
    return a + b

## Execution Trace
-> Initializing...
-> Processing input...
-> Generating output...
-> Complete!
```

### Key Challenges Solved

1. **Output length**: Increased max-seq to 4096 tokens for CoT outputs
2. **Execution traces**: Added visual feedback for code generation
3. **Selective tools**: Only add tool calls where relevant (not for every sample)

### Technical Insights

```bash
# Add CoT reasoning (optimized - 208 MB)
python scripts/add_reasoning_v2.py \
  --input data/combined_final.jsonl \
  --output data/cot_final.jsonl

# Add full tools + tool calls (991 MB)
python scripts/add_reasoning.py \
  --input data/combined_final.jsonl \
  --output data/reasoning_final.jsonl
```

---

## Phase 3: Training Pipeline

### QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Balance between quality and size |
| Alpha | 32 | 2x rank for better adaptation |
| Target Modules | q,k,v,o_proj | Attention layers |
| Quantization | 4-bit NF4 | Reduces 14GB → 4GB |
| Max Seq | 4096 | Accommodate CoT outputs |

### Training Command

```bash
python scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/cot_final.jsonl \
  --output ./fine-tuned-model \
  --epochs 3 \
  --batch_size 4 \
  --max-seq 4096 \
  --prompt-style cot
```

### Prompt Styles Supported

1. **inst**: `[INST] instruction [/INST] output`
2. **chat**: `<|system|>...<|user|>...<|assistant|>...`
3. **cot**: `## Task ## Input ## Reasoning output`

---

## Phase 4: Infrastructure (Terraform)

### AMD Developer Cloud Provisioning

```hcl
resource "digitalocean_droplet" "mi300x_single" {
  name   = "mi300x-training"
  region = "nyc3"
  image  = "gpu-amd-base"  # ROCm 7.0.2 pre-installed
  size   = "gpu-mi300x1-192gb"  # 192 GB VRAM
  # Auto-installs PyTorch, transformers, peft
}
```

### GPU Options

| Config | Slug | VRAM | Cost |
|--------|------|-----|------|
| Single MI300X | `gpu-mi300x1-192gb` | 192 GB | ~$1.99/hr |
| 8x MI300X | `gpu-mi300x8-1536gb` | 1,536 GB | ~$15.92/hr |

---

## Feedback on AMD Developer Experience

### ✅ What Worked Well

1. **ROCm pre-installed**: `gpu-amd-base` image has ROCm 7.0.2 ready
2. **4-bit quantization**: QLoRA works perfectly with ROCm PyTorch
3. **Large VRAM**: 192 GB allows loading full 7B model in memory
4. **Documentation**: AMD Developer Cloud docs are comprehensive
5. **Pricing**: $1.99/hr is competitive vs. NVIDIA alternatives

### Challenges & Workarounds

1. **Initial setup**: Required DigitalOcean account + SSH key upload
   - **Fix**: Used `doctl` CLI for automation
   
2. **PyTorch ROCm version**: Needed specific index URL
   - **Fix**: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1`

3. **vLLM on ROCm**: Required specific version
   - **Fix**: `pip install vllm==0.15.0+rocm700`

### Suggestions for Improvement

1. **Pre-built Docker images**: More 1-Click options for common frameworks
2. **JupyterLab integration**: Built-in Jupyter would speed up experimentation
3. **Instance templates**: Save configured templates for reuse

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Dataset Samples | 257,930 |
| CoT-Enhanced Samples | 256,542 |
| Training Scripts | 10+ |
| Test Coverage | 106 tests |
| Documentation Pages | 2 (README + AGENTS) |
| Terraform Configs | 1 |

---

## How to Replicate

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/EkemainaiAgent.git

# 2. Install dependencies
pip install torch transformers accelerate peft datasets

# 3. Merge datasets
python scripts/merge_datasets.py --all

# 4. Add Chain-of-Thought
python scripts/add_reasoning_v2.py \
  --input data/combined_final.jsonl \
  --output data/cot_final.jsonl

# 5. Train (requires AMD MI300X)
python scripts/train.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/cot_final.jsonl \
  --output ./fine-tuned-model
```

---

## Build in Public Updates

### Update 1: Dataset Pipeline Complete
> 🚀 Just built a 257K sample code training dataset! Merged APPS, CodeAlpaca, HumanEval, CodeSearchNet & more. Added Chain-of-Thought reasoning for better code generation. #AMD #Hackathon @lablab @AIatAMD

### Update 2: Training Pipeline Ready  
> 🎯 Fine-tuned Mistral 7B with QLoRA for code assistance. CoT reasoning + tool-use capabilities. Ready for AMD MI300X training. Full pipeline: merge ��� enhance → train #AIatAMD @lablabai

---

## Files Contributed

- `scripts/merge_datasets.py` - Dataset merging with deduplication
- `scripts/add_reasoning.py` - Full CoT + tools transformation  
- `scripts/add_reasoning_v2.py` - Optimized CoT transformation
- `scripts/train.py` - Updated QLoRA training with prompt styles
- `terraform/main.tf` - AMD GPU provisioning
- `tests/test_*.py` - 106 unit tests
- `README.md` - Full documentation

---

## License

MIT License - Open for contributions!

---

*Built for AMD Developer Hackathon 2025*