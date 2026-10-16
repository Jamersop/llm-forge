"""Fine-tuning run for one job. Executed in its own process by the worker:

    python -m forge.trainer <job_id>

Writes into runs/<job_id>/:  train.log, metrics.jsonl (one JSON object per log step),
adapter/ (LoRA weights + tokenizer) or model/ (full fine-tune), summary.json.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from . import data, db
from .config import RUNS_DIR

log = logging.getLogger("forge.trainer")


class Collator:
    """Right-pad input_ids/labels to the longest example in the batch."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(feats), n), -100, dtype=torch.long)
        mask = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            labels[i, :k] = torch.tensor(f["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class MetricsCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        rec = {"step": state.global_step, "max_steps": state.max_steps,
               "epoch": round(state.epoch or 0, 3), "time": time.time(), **logs}
        if "eval_loss" in logs:
            rec["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(json.dumps(rec))


def _warmup_kwarg(ratio: float) -> dict:
    # transformers v5 folded warmup_ratio into warmup_steps (a float < 1 means a ratio).
    if "warmup_ratio" in TrainingArguments.__dataclass_fields__:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}


def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(base: str, hp: dict):
    dtype = pick_dtype()
    kwargs = {"dtype": dtype}
    if hp["method"] == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs a CUDA GPU; choose 'lora' or 'full' instead")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.config.use_cache = False

    if hp["method"] in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if hp["method"] == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        targets = hp["lora_targets"]
        if isinstance(targets, str) and targets != "all-linear":
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
            target_modules=targets, task_type="CAUSAL_LM"))
    return model


def run(job_id: str):
    job = db.get_job(job_id)
    hp, base = job["hparams"], job["base_model"]
    out = RUNS_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    set_seed(hp["seed"])
    log.info("job %s: base=%s method=%s cuda=%s", job_id, base, hp["method"], torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(base)
    data.ensure_chat_template(tokenizer)
    sets = []
    for ds_id in job["dataset_ids"]:
        ds = db.get_dataset(ds_id)
        if ds is None:
            raise RuntimeError(f"dataset {ds_id} no longer exists")
        sets.append((ds["kind"], data.load(ds["path"])))
    train, evals, info = data.build_features(tokenizer, sets, hp)
    log.info("data: %s", info)

    model = load_model(base, hp)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    cuda = torch.cuda.is_available()
    dtype = pick_dtype()
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        **_warmup_kwarg(hp["warmup_ratio"]),
        weight_decay=hp["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="epoch" if evals else "no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=cuda and dtype == torch.bfloat16,
        fp16=cuda and dtype == torch.float16,
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cuda else None,
        report_to=[],
        seed=hp["seed"],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evals or None,
                      data_collator=Collator(tokenizer.pad_token_id),
                      callbacks=[MetricsCallback(out / "metrics.jsonl")])
    result = trainer.train()

    summary = {"data": info, "train_loss": result.training_loss,
               "train_runtime_s": result.metrics.get("train_runtime")}
    if evals:
        ev = trainer.evaluate()
        summary["eval_loss"] = ev["eval_loss"]
        summary["eval_perplexity"] = math.exp(min(ev["eval_loss"], 20))

    target = out / ("adapter" if hp["method"] in ("lora", "qlora") else "model")
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved to %s; summary %s", target, summary)
    return summary


def main():
    job_id = sys.argv[1]
    db.init()
    try:
        summary = run(job_id)
    except Exception as e:  # report failure to the DB, then exit non-zero
        logging.getLogger("forge.trainer").exception("training failed")
        db.update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
        sys.exit(1)
    db.update_job(job_id, status="completed", metrics=summary, finished_at=time.time())


if __name__ == "__main__":
    main()
