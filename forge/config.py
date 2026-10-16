"""Paths, defaults and the curated base-model catalog."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FORGE_HOME", Path(__file__).resolve().parent.parent / "forge_data"))
DATASETS_DIR = ROOT / "datasets"
RUNS_DIR = ROOT / "runs"
DB_PATH = ROOT / "forge.db"

for _d in (ROOT, DATASETS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Base models offered in the UI. Any Hugging Face model id or local path can
# also be used. Reasoning-capable models are preferred: fine-tuning with LoRA
# keeps their general language and reasoning skills while adding domain knowledge.
MODEL_CATALOG = [
    {"id": "Qwen/Qwen3-0.6B", "params": "0.6B", "reasoning": True,
     "notes": "Native <think> mode. Fast to iterate; good for prototypes."},
    {"id": "Qwen/Qwen3-1.7B", "params": "1.7B", "reasoning": True,
     "notes": "Native <think> mode. Fits a 16 GB GPU with LoRA."},
    {"id": "Qwen/Qwen3-4B", "params": "4B", "reasoning": True,
     "notes": "Strong reasoning for its size. QLoRA on 16 GB, LoRA on 24 GB."},
    {"id": "Qwen/Qwen3-8B", "params": "8B", "reasoning": True,
     "notes": "Best quality in the catalog. QLoRA on 24 GB."},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "params": "1.5B", "reasoning": True,
     "notes": "Distilled from R1; always thinks before answering."},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "params": "7B", "reasoning": True,
     "notes": "Distilled from R1; strong math/logic chains."},
    {"id": "microsoft/Phi-4-mini-reasoning", "params": "3.8B", "reasoning": True,
     "notes": "Compact model tuned for step-by-step math reasoning."},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "params": "3B", "reasoning": False,
     "notes": "General instruct model; learns <think> format from your data. Gated."},
    {"id": "google/gemma-3-1b-it", "params": "1B", "reasoning": False,
     "notes": "Small general instruct model. Gated."},
    {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "params": "135M", "reasoning": False,
     "notes": "Tiny; runs on CPU. For smoke-testing the pipeline only."},
]

# Default hyperparameters; every field can be overridden per job.
DEFAULT_HPARAMS = {
    "method": "lora",            # lora | qlora | full
    "epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "grad_accum": 8,
    "max_seq_len": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": "all-linear",
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "eval_ratio": 0.05,          # fraction held out for validation loss / perplexity
    "seed": 42,
    "system_prompt": "",         # injected into examples that have no system message
    "train_on_prompt": False,    # False = loss only on assistant (answer + reasoning) tokens
}

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
