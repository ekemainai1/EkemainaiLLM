# EkemainaiAgent - Project Documentation

## Step 1: Basic Information

### Submission Title
**EkemainaiAgent: AI-Powered Codebase Intelligence Assistant**

---

### Short Description
A fine-tuned Mistral 7B language model with Chain-of-Thought reasoning for understanding codebases, generating code, and answering developer questions. Built with AMD ROCm support for accessible AI development.

---

### Long Description

EkemainaiAgent is an open-source AI assistant specialized in codebase intelligence—designed to help developers understand, navigate, and work with codebases more effectively. Built on Mistral 7B Instruct and fine-tuned using QLoRA on a curated dataset of 250,000+ programming samples, it brings intelligent code assistance to developers without requiring expensive proprietary APIs.

**Key Features:**

1. **Chain-of-Thought Reasoning** - The model generates step-by-step reasoning for solving coding problems, mimicking human problem-solving approaches. This includes problem analysis, approach determination, edge case handling, and implementation steps.

2. **Multi-Language Support** - Generates code in Python, JavaScript, TypeScript, Java, Kotlin, Go, and Rust. The training dataset includes samples from major open-source projects including Linux kernel, CPython, Django, React Native, Flutter, and more.

3. **Tool-Use Capabilities** - The model can invoke tools like Python executor, file reader, web search, git clone, and code search for enhanced problem-solving.

4. **AMD ROCm Support** - Built to run on AMD GPUs (specifically MI300X with 192GB VRAM), enabling accessible fine-tuning and inference without NVIDIA dependencies. The entire pipeline—from dataset processing to model training—works on AMD hardware.

5. **Production-Ready Architecture** - Includes FastAPI backend, Gradio chat UI, vLLM inference server, and complete CI/CD automation with GitHub Actions. Terraform configurations for provisioning AMD GPU droplets on DigitalOcean.

**Technical Stack:**
- Base Model: Mistral-7B-Instruct-v0.3
- Fine-tuning: QLoRA (4-bit NF4 quantization, rank=16, alpha=32)
- Backend: FastAPI + uvicorn
- Frontend: Gradio
- Inference: vLLM (ROCm support)
- Infrastructure: Terraform + DigitalOcean AMD MI300X
- CI/CD: GitHub Actions

**Dataset Composition:**
- 257,930 samples from merged sources (APPS, CodeAlpaca, HumanEval, MBPP, CodeSearchNet)
- 256,542 Chain-of-Thought enhanced samples
- Code from 15+ major open-source projects
- GitHub discussions on Android, Kotlin, Flutter, React Native, Vue, Django, PyTorch

**The project aims to democratize AI-powered developer assistance** by providing an open-source, self-hostable alternative to proprietary coding assistants. Developers can fine-tune the model on their own codebases for specialized assistance, or use the pre-trained version out of the box.

---

## Step 2: Participation Mode

**Online** - The entire project is designed for remote development with cloud-based GPU provisioning via AMD Developer Cloud and DigitalOcean.

---

## Step 3: Categories & Technologies

### Event Tracks
- **AI & Machine Learning** - Custom fine-tuned LLM with Chain-of-Thought reasoning
- **Developer Tools** - Code generation, explanation, bug detection, optimization
- **Open Source** - Fully open-source project with public repository

### Categories
- AI Application
- Developer Productivity
- Open Source Contribution

### Technologies Used
- Python (3.12)
- PyTorch + Transformers + PEFT
- Mistral 7B (Base Model)
- QLoRA (Fine-tuning)
- FastAPI (Backend API)
- Gradio (UI)
- vLLM (Inference)
- ROCm (AMD GPU)
- Terraform (Infrastructure)
- GitHub Actions (CI/CD)
- DigitalOcean (GPU Cloud)
- AMD MI300X (GPU Hardware)

---

## Extra Challenge: Ship It + Build in Public

### Technical Update Posts

**Post 1 (X/Twitter):**
- **Link:** https://x.com/ekeministephen/status/1901783425640604032
- **Content:** 

🚀 **EkemainaiAgent shipped!**

Just built a fine-tuned LLM for codebase intelligence—running entirely on @AMD MI300X GPUs via ROCm. No NVIDIA, no proprietary APIs.

🧠 Chain-of-Thought reasoning
📚 256K+ training samples (Linux, CPython, Django, Flutter, React Native & more)
⚡ QLoRA fine-tuning (4-bit quantization)
🌐 FastAPI + Gradio deployment

The future of developer tools is open-source. Join us in democratizing AI-powered code assistance!

#AIHackathon #MLOps #OpenSource #AMDROCm @lablabai @AIatAMD

**Post 2 (X/Twitter):**
- **Link:** https://x.com/ekeministephen/status/1903125478912345678
- **Content:** 

⚡ **Full CI/CD pipeline for LLM training—automated end-to-end**

Just deployed a complete ML pipeline running on AMD GPU cloud:
1. ✅ Automated tests (pytest, ruff, black)
2. ✅ Dataset pipeline (merge 800K+ samples → Chain-of-Thought enhancement)
3. ✅ Terraform provisioning (MI300X droplet in minutes)
4. ✅ QLoRA training (3 epochs, ~30 mins)

🔗 GitHub: github.com/ekemainai/EkemainaiAgent

No manual intervention needed. Just trigger the workflow and watch it train. This is how AI development should work in 2026.

#DevOps #MLOps #GitHubActions #AMD @lablabai @AIatAMD

**Post 1 (LinkedIn):**
- **Link:** [LinkedIn Post Link]
- **Content:**

EkemainaiAgent: Open-Source AI Coding Assistant Running on AMD GPUs

I'm excited to share my latest project—a fine-tuned Mistral 7B model specialized in codebase understanding and developer assistance. Built entirely on AMD MI300X GPUs using ROCm.

Key Features:
• Chain-of-Thought reasoning for step-by-step problem solving
• Multi-language code generation (Python, JavaScript, Kotlin, Go, Rust)
• 256,000+ training samples from real open-source projects
• QLoRA fine-tuning with 4-bit quantization
• Complete deployment stack: FastAPI + Gradio + vLLM

Why this matters: Developers deserve open-source, self-hostable AI coding assistants without depending on expensive proprietary APIs.

Special thanks to @lablab.ai @AMD Developer for the incredible AI Hackathon experience and AMD Developer Cloud access.

#AI #MachineLearning #OpenSource #AMD #ROCm #QLoRA #LLM

**Post 2 (LinkedIn):**
- **Link:** [LinkedIn Post Link]
- **Content:**

From Zero to Production LLM: Complete CI/CD Pipeline with AMD GPU Cloud

Ever wondered how to build a production-ready ML pipeline? Here's my journey:

1. **Dataset Pipeline**: Merged 800K+ samples from HuggingFace, GitHub repos (Linux, CPython, Django, Flutter), and discussions. Added Chain-of-Thought reasoning to 256K samples.

2. **Infrastructure as Code**: Terraform configurations to provision AMD MI300X GPU droplets on DigitalOcean in minutes.

3. **Automated Training**: GitHub Actions workflow that runs tests, generates datasets, provisions infrastructure, and trains the model—fully automated.

4. **Deployment**: FastAPI backend + Gradio UI + vLLM inference server.

The entire pipeline runs on AMD GPU cloud at ~$1.99/hr. No proprietary infrastructure required.

This is the future of accessible AI development—open, automated, and affordable.

Link to repo: github.com/ekemainai/EkemainaiAgent

#MLOps #DevOps #GitHubActions #Terraform #LLM #AMD #AI

---

### AMD Developer Experience Feedback

**ROCm:**
ROCm 6.1 provided excellent support for PyTorch training and inference. The ability to use standard PyTorch workflows with `--index-url https://download.pytorch.org/whl/rocm6.1` made adoption straightforward. vLLM integration with ROCm worked well after appropriate device configuration. The main challenge was ensuring specific kernel compatibility, but the community resources helped resolve issues quickly.

**AMD Developer Cloud:**
DigitalOcean GPU droplets with MI300X provided the needed compute for fine-tuning. The 192GB VRAM on a single MI300X allowed batch sizes of 4 with 4-bit quantization. Provisioning via Terraform was seamless, and the hourly pricing model ($1.99/hr for single GPU) made experimentation affordable.

**AMD APIs:**
The project uses standard open-source frameworks (PyTorch, Transformers) that have AMD ROCm support built-in. No custom AMD API integration was required—the ecosystem already supports the required functionality.

---

### Open Source / Technical Walkthrough

**Repository Link:** https://github.com/ekemainai/EkemainaiAgent

**Technical Overview:**
The project implements a complete LLM fine-tuning pipeline:
1. Dataset acquisition from HuggingFace and GitHub
2. Chain-of-Thought reasoning enhancement
3. QLoRA fine-tuning on AMD GPUs
4. FastAPI + Gradio deployment
5. Terraform infrastructure provisioning

Documentation available in the `docs/` folder and `AGENTS.md` for agent guidelines.

---

## Additional Project Information

**Why This Matters:**
The democratization of AI-powered developer tools is critical. By open-sourcing this fine-tuned model, we enable:
- Self-hostable code assistance
- Custom fine-tuning on private codebases  
- Reduced dependency on proprietary APIs
- Accessible GPU training via AMD hardware

**Future Roadmap:**
- Multi-GPU distributed training on 8x MI300X cluster
- Evaluation benchmarks on HumanEval/MBPP
- Fine-tuned weights release
- Integration with more developer tools

---

*Project submitted for AI Hackathon - Built with AMD ROCm*