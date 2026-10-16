"""Load a trained job (base model + adapter) and chat with it; merge/export adapters."""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data, db
from .config import RUNS_DIR
from .trainer import pick_dtype


class ModelCache:
    """Keeps the single most recently used model in memory."""

    def __init__(self):
        self.key = None
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def get(self, job_id: str | None, base: str | None = None):
        key = job_id or f"base:{base}"
        if key != self.key:
            self.model = self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model, self.tokenizer = load(job_id, base)
            self.key = key
        return self.model, self.tokenizer


def artifact_dir(job_id: str) -> tuple[Path, bool]:
    run = RUNS_DIR / job_id
    if (run / "adapter").exists():
        return run / "adapter", True
    if (run / "model").exists():
        return run / "model", False
    raise FileNotFoundError(f"job {job_id} has no trained weights yet")


def load(job_id: str | None, base: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if job_id is None:  # untuned base model, for side-by-side comparison
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=pick_dtype()).to(device)
    else:
        path, is_adapter = artifact_dir(job_id)
        tok = AutoTokenizer.from_pretrained(path)
        if is_adapter:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(db.get_job(job_id)["base_model"],
                                                              dtype=pick_dtype())
            model = PeftModel.from_pretrained(base_model, path)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=pick_dtype())
        model = model.to(device)
    data.ensure_chat_template(tok)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens=512, temperature=0.6,
             top_p=0.95, enable_thinking=True) -> dict:
    # enable_thinking is understood by Qwen3-style templates and ignored by others.
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=enable_thinking)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None,
                         top_p=top_p if temperature > 0 else None,
                         pad_token_id=tok.pad_token_id)
    new = out[0, enc["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    # The opening <think> may live in the prompt (some templates prefill it).
    if prompt.rstrip().endswith(data.THINK_OPEN) and data.THINK_OPEN not in text:
        text = data.THINK_OPEN + text
    thinking, answer = data.split_reasoning(text)
    return {"thinking": thinking, "answer": answer, "raw": text, "new_tokens": int(new.shape[0])}


def merge(job_id: str) -> Path:
    """Fold a LoRA adapter into the base weights -> standalone HF model folder."""
    path, is_adapter = artifact_dir(job_id)
    if not is_adapter:
        return path
    from peft import PeftModel
    target = RUNS_DIR / job_id / "merged"
    base = AutoModelForCausalLM.from_pretrained(db.get_job(job_id)["base_model"], dtype=pick_dtype())
    merged = PeftModel.from_pretrained(base, path).merge_and_unload()
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(path).save_pretrained(target)
    return target
