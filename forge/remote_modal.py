"""Run training (and playground chat) on rented cloud GPUs via Modal (https://modal.com).

One-time setup on the machine running the Forge server:
    pip install modal
    modal setup                                   # log in
    modal deploy forge/remote_modal.py            # from the repo root
    # gated models (Llama, Gemma): set HF_TOKEN where the Forge server runs; it is
    # forwarded to the GPU container per call.

Then pick "Cloud GPU (Modal)" under Compute when starting a job. You pay per second of GPU
time only while a job or chat is running.

Data flow: the job spec (datasets included) is sent to the GPU container; metrics stream
back through a Modal Queue; weights are written to the `llm-forge-runs` Volume and
downloaded into runs/<job_id>/ when training finishes, so the rest of the app (export,
local playground) works the same as for local jobs.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import modal

APP_NAME = "llm-forge"
VOLUME_ROOT = "/runs"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.51", "peft>=0.13", "accelerate>=0.34",
                 "bitsandbytes>=0.43", "tokenizers")
    .env({"FORGE_HOME": "/tmp/forge", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("forge")
)
runs_vol = modal.Volume.from_name("llm-forge-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("llm-forge-hf-cache", create_if_missing=True)
VOLUMES = {VOLUME_ROOT: runs_vol, "/root/.cache/huggingface": hf_cache}

log = logging.getLogger("forge.remote")


# ---- remote side --------------------------------------------------------------

def _use_token(hf_token: str | None):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


@app.function(image=image, gpu="A10G", timeout=24 * 3600, volumes=VOLUMES)
def train_remote(spec: dict, queue: modal.Queue, hf_token: str | None = None) -> dict:
    from forge.trainer import setup_logging, train
    _use_token(hf_token)
    out = Path(VOLUME_ROOT) / spec["job_id"]
    setup_logging(out)
    try:
        summary = train(spec, out, on_log=queue.put)
    finally:
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
        runs_vol.commit()
        hf_cache.commit()
    return summary


@app.cls(image=image, gpu="A10G", timeout=3600, volumes=VOLUMES, scaledown_window=300)
class Chat:
    """Keeps the last-used model on the GPU; the container shuts down after 5 idle minutes."""

    @modal.enter()
    def setup(self):
        self.key, self.model, self.tok = None, None, None

    @modal.method()
    def generate(self, job_id: str | None, base_model: str, messages: list[dict], params: dict,
                 hf_token: str | None = None) -> dict:
        from forge import inference
        _use_token(hf_token)
        key = job_id or f"base:{base_model}"
        if key != self.key:
            self.model = None
            if job_id:
                runs_vol.reload()
                run = Path(VOLUME_ROOT) / job_id
                adapter = run / "adapter"
                self.model, self.tok = inference.load_weights(
                    base_model, adapter if adapter.exists() else run / "model", adapter.exists())
            else:
                self.model, self.tok = inference.load_weights(base_model)
            self.key = key
        return inference.generate(self.model, self.tok, messages, **params)


# ---- client side (runs inside the Forge server / trainer process) -------------

def _deployed(name: str):
    try:
        return modal.Function.from_name(APP_NAME, name) if name != "Chat" \
            else modal.Cls.from_name(APP_NAME, name)
    except Exception as e:
        raise RuntimeError(f"Modal app not found ({e}). Run: modal deploy forge/remote_modal.py") from e


def _timeout_errors():
    return tuple({TimeoutError, getattr(modal.exception, "TimeoutError", TimeoutError)})


def run_remote(spec: dict, out: Path, gpu: str = "A10G") -> dict:
    """Train on a Modal GPU; stream metrics into out/metrics.jsonl; download weights."""
    fn = _deployed("train_remote").with_options(gpu=gpu)
    out.mkdir(parents=True, exist_ok=True)
    metrics = open(out / "metrics.jsonl", "a")

    def record(rec):
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        log.info(json.dumps(rec))

    log.info("submitting job %s to Modal on %s", spec["job_id"], gpu)
    with modal.Queue.ephemeral() as queue:
        call = fn.spawn(spec, queue, os.environ.get("HF_TOKEN"))
        try:
            while True:
                for rec in queue.get_many(100, block=True, timeout=5):
                    record(rec)
                try:
                    summary = call.get(timeout=0)
                    break
                except _timeout_errors():
                    continue
        except BaseException:  # cancelled (SIGTERM -> KeyboardInterrupt) or connection lost
            log.info("cancelling remote call %s", call.object_id)
            call.cancel()
            raise
        for rec in queue.get_many(1000, block=False):
            record(rec)
    metrics.close()

    _download(spec["job_id"], out)
    summary["compute"] = f"modal:{gpu}"
    log.info("remote training finished: %s", summary)
    return summary


def _download(job_id: str, out: Path):
    from modal.volume import FileEntryType
    vol = modal.Volume.from_name("llm-forge-runs")
    for entry in vol.listdir(job_id, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = Path(entry.path).relative_to(job_id)
        if rel.parts[0] in ("adapter", "model", "merged") or rel.name == "summary.json":
            dest = out / rel
        elif rel.name == "train.log":
            dest = out / "remote_train.log"
        else:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
    log.info("downloaded weights for %s", job_id)


def chat_remote(job_id: str | None, base_model: str, messages: list[dict], params: dict,
                gpu: str = "A10G") -> dict:
    chat_cls = _deployed("Chat").with_options(gpu=gpu)
    return chat_cls().generate.remote(job_id, base_model, messages, params, os.environ.get("HF_TOKEN"))


def delete_remote_run(job_id: str):
    try:
        modal.Volume.from_name("llm-forge-runs").remove_file(job_id, recursive=True)
    except Exception:
        pass
