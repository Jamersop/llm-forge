"""Load a trained job (base model + adapter) and chat with it; merge/export adapters."""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data
from .config import RUNS_DIR
from .hardware import pick_device, pick_dtype


class ModelCache:
    """Keeps the single most recently used model in memory."""

    def __init__(self):
        self.key = None
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def get(self, job_id: str | None, base: str):
        key = job_id or f"base:{base}"
        if key != self.key:
            self.model = self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if job_id is None:
                self.model, self.tokenizer = load_weights(base)
            else:
                path, is_adapter = artifact_dir(job_id)
                self.model, self.tokenizer = load_weights(base, path, is_adapter)
            self.key = key
        return self.model, self.tokenizer


def artifact_dir(job_id: str) -> tuple[Path, bool]:
    run = RUNS_DIR / job_id
    if (run / "adapter").exists():
        return run / "adapter", True
    if (run / "model").exists():
        return run / "model", False
    raise FileNotFoundError(f"job {job_id} has no trained weights yet")


def load_weights(base: str, path: Path | None = None, is_adapter: bool = False):
    """Base model alone, base + LoRA adapter at `path`, or a full model at `path`."""
    device = pick_device()
    dtype = torch.float16 if device == "mps" else pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(path or base)
    if path is not None and not is_adapter:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        if path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
    data.ensure_chat_template(tok)
    return model.to(device).eval(), tok


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


def merge(job_id: str, base_model: str) -> Path:
    """Fold a LoRA adapter into the base weights -> standalone HF model folder."""
    path, is_adapter = artifact_dir(job_id)
    if not is_adapter:
        return path
    from peft import PeftModel
    target = RUNS_DIR / job_id / "merged"
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype="auto")  # keep native precision
    merged = PeftModel.from_pretrained(base, path).merge_and_unload()
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(path).save_pretrained(target)
    return target
