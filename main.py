#!/usr/bin/env python3
"""FastAPI backend for Codebase Intelligence Assistant."""
import os
import sys
import json
import argparse
import torch
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


app = FastAPI(title="Codebase Intelligence Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL = None
TOKENIZER = None
DEVICE = get_device()
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM_PROMPT = "[INST] You are an expert coding assistant. Provide accurate, well-commented code with explanations. Specialize in Python, Kotlin, Java, and Android/Jetpack Compose. [/INST]\n"
ANDROID_PROMPT = "[INST] You are an expert Android/Kotlin/Jetpack Compose developer. Provide accurate, production-ready code with best practices for mobile development. Focus on modern Android architecture, coroutines, and Jetpack Compose. [/INST]\n"

DEFAULT_PROMPT = "[INST] {instruction}\n{input} [/INST]\n"


class ChatRequest(BaseModel):
    instruction: str = "Answer this coding question."
    input: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    model: str
    device: str


def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return
    model_dir = os.environ.get("MODEL_DIR", "./fine-tuned-model")
    print(f"Loading model from {model_dir}...")
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        MODEL = PeftModel.from_pretrained(base, model_dir)
    else:
        MODEL = base
    MODEL.eval()
    print(f"Model loaded on {DEVICE}")


def generate_response(instruction, input_text, max_tokens=2048, temperature=0.7):
    prompt = DEFAULT_PROMPT.format(instruction=instruction, input=input_text)
    full_prompt = SYSTEM_PROMPT + prompt
    inputs = TOKENIZER(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0.01,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return response[len(full_prompt):].strip()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    load_model()
    try:
        response = generate_response(
            req.instruction,
            req.input,
            req.max_tokens,
            req.temperature,
        )
        return ChatResponse(response=response, model=BASE_MODEL, device=DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./fine-tuned-model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = args.model_dir
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()