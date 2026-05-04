# GitHub Actions Workflows Overview

## Available CI/CD Pipelines

### 1. Test Pipeline (test.yml)
Runs on: push, pull_request
- Unit tests (pytest)
- Linting (black, ruff)
- Python syntax check

### 2. Dataset Validation (dataset.yml)
Runs on: push to data/
- Validates JSONL format
- Checks schema
- Reports dataset sizes

### 3. Dataset Pipeline (dataset_pipeline.yml)
Trigger: workflow_dispatch
Actions:
- merge: Combine all datasets
- add_cot: Add Chain-of-Thought
- add_reasoning: Add full tools + reasoning
- all: Run full pipeline

### 4. Training Pipeline (training.yml)
Trigger: workflow_dispatch
Parameters:
- model: Base model (default: mistralai/Mistral-7B-Instruct-v0.3)
- dataset: Training dataset
- epochs: 1-10
- batch_size: 1-16
- prompt_style: inst/chat/cot

### 5. Code Quality (quality.yml)
Runs on: push, pull_request
- Black formatting check
- isort import order
- Ruff linting
- Bandit security scan

### 6. CI Combined (ci.yml)
Combines all workflows with proper dependencies

---

## Manual Triggers

### Run Dataset Pipeline
```bash
gh workflow run dataset_pipeline.yml -f action=all
```

### Run Training (requires AMD GPU)
```bash
gh workflow run training.yml \
  -f model=mistralai/Mistral-7B-Instruct-v0.3 \
  -f dataset=data/cot_final.jsonl \
  -f epochs=3
```

---

## Usage

### Local Testing
```bash
# Run tests locally
pytest tests/ -v

# Run specific test file
pytest tests/test_merge_datasets.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

### GitHub Actions

1. Tests run automatically on push/PR
2. Dataset validation runs when data/ changes
3. Manually trigger dataset pipeline: Actions → Dataset Pipeline → Run workflow
4. Manually trigger training: Actions → Training → Run workflow

---

## Workflow Files

```
.github/workflows/
├── ci.yml                    # Main CI (combined)
├── test.yml                   # Unit tests + lint
├── dataset.yml               # Dataset validation
├── dataset_pipeline.yml      # Dataset processing
├── training.yml              # Model training
├── quality.yml              # Code quality checks
└── trigger.yml             # Generic trigger
```