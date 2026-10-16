"""HTTP API + web UI.  Run:  uvicorn forge.api:app --port 8000"""
from __future__ import annotations

import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data, db
from .config import DATASETS_DIR, DEFAULT_HPARAMS, MODEL_CATALOG, RUNS_DIR
from .worker import Worker

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

worker = Worker()
models = None  # inference.ModelCache, created lazily so the API starts without torch import cost


@asynccontextmanager
async def lifespan(app):
    db.init()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="LLM Forge", lifespan=lifespan)


def _404(what: str):
    raise HTTPException(404, f"{what} not found")


# ---- catalog ------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    return {"catalog": MODEL_CATALOG, "defaults": DEFAULT_HPARAMS}


# ---- datasets -----------------------------------------------------------------

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), name: str = Form("")):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file larger than 500 MB")
    try:
        kind, records, stats = data.ingest(file.filename or "upload.jsonl", raw)
    except (data.DatasetError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(400, str(e))
    ds_id = db.new_id()
    path = DATASETS_DIR / f"{ds_id}.jsonl"
    data.save(path, records)
    return db.add_dataset(ds_id, name or file.filename, str(path), kind, stats)


@app.get("/api/datasets")
def list_datasets():
    return db.list_datasets()


@app.get("/api/datasets/{ds_id}/preview")
def preview_dataset(ds_id: str, n: int = 5):
    ds = db.get_dataset(ds_id) or _404("dataset")
    return data.load(ds["path"])[:n]


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    ds = db.get_dataset(ds_id) or _404("dataset")
    Path(ds["path"]).unlink(missing_ok=True)
    db.delete_dataset(ds_id)
    return {"ok": True}


# ---- jobs ---------------------------------------------------------------------

class JobRequest(BaseModel):
    name: str = ""
    base_model: str
    dataset_ids: list[str] = Field(min_length=1)
    hparams: dict = {}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    for ds_id in req.dataset_ids:
        db.get_dataset(ds_id) or _404(f"dataset {ds_id}")
    unknown = set(req.hparams) - set(DEFAULT_HPARAMS)
    if unknown:
        raise HTTPException(400, f"unknown hyperparameters: {sorted(unknown)}")
    hp = {**DEFAULT_HPARAMS, **req.hparams}
    if hp["method"] not in ("lora", "qlora", "full"):
        raise HTTPException(400, "method must be lora, qlora or full")
    name = req.name or f"{req.base_model.split('/')[-1]}-ft"
    return db.add_job(name, req.base_model.strip(), req.dataset_ids, hp)


@app.get("/api/jobs")
def list_jobs():
    return db.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = db.get_job(job_id) or _404("job")
    mpath = RUNS_DIR / job_id / "metrics.jsonl"
    job["history"] = [json.loads(l) for l in mpath.read_text().splitlines()] if mpath.exists() else []
    return job


@app.get("/api/jobs/{job_id}/logs", response_class=PlainTextResponse)
def job_logs(job_id: str, tail: int = 200):
    db.get_job(job_id) or _404("job")
    p = RUNS_DIR / job_id / "stdout.log"
    return "\n".join(p.read_text(errors="replace").splitlines()[-tail:]) if p.exists() else ""


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not worker.cancel(job_id):
        raise HTTPException(409, "job is not queued or running")
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = db.get_job(job_id) or _404("job")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "cancel the job first")
    shutil.rmtree(RUNS_DIR / job_id, ignore_errors=True)
    with db.connect() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/export")
async def export_job(job_id: str):
    """Merge LoRA into the base model and return a downloadable zip."""
    from . import inference
    job = db.get_job(job_id) or _404("job")
    if job["status"] != "completed":
        raise HTTPException(409, "job has not completed")
    folder = await run_in_threadpool(inference.merge, job_id)
    zip_path = await run_in_threadpool(shutil.make_archive, str(folder), "zip", folder)
    return FileResponse(zip_path, filename=f"{job['name']}.zip")


# ---- playground ---------------------------------------------------------------

class ChatRequest(BaseModel):
    job_id: str | None = None       # None + base_model = chat with the untuned base model
    base_model: str | None = None
    messages: list[dict]
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.6, ge=0, le=2)
    enable_thinking: bool = True


@app.post("/api/chat")
async def chat(req: ChatRequest):
    global models
    from . import inference
    if models is None:
        models = inference.ModelCache()
    if req.job_id:
        job = db.get_job(req.job_id) or _404("job")
        if job["status"] != "completed":
            raise HTTPException(409, "job has not completed")
    elif not req.base_model:
        raise HTTPException(400, "give job_id or base_model")

    def _go():
        with models.lock:
            model, tok = models.get(req.job_id, req.base_model)
            return inference.generate(model, tok, req.messages, req.max_new_tokens,
                                      req.temperature, enable_thinking=req.enable_thinking)
    try:
        return await run_in_threadpool(_go)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
