"""Tiny SQLite store for datasets and training jobs."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,          -- chat | text
    stats TEXT NOT NULL,         -- json
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    dataset_ids TEXT NOT NULL,   -- json list
    hparams TEXT NOT NULL,       -- json
    status TEXT NOT NULL,        -- queued | running | completed | failed | cancelled
    error TEXT,
    pid INTEGER,
    metrics TEXT,                -- json summary (final losses, perplexity)
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
"""

JSON_FIELDS = {"stats", "dataset_ids", "hparams", "metrics"}


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as c:
        c.executescript(SCHEMA)


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_FIELDS & d.keys():
        if d[k] is not None:
            d[k] = json.loads(d[k])
    return d


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---- datasets -------------------------------------------------------------

def add_dataset(id: str, name: str, path: str, kind: str, stats: dict) -> dict:
    with connect() as c:
        c.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                  (id, name, path, kind, json.dumps(stats), time.time()))
    return get_dataset(id)


def get_dataset(id: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM datasets WHERE id=?", (id,)).fetchone())


def list_datasets() -> list[dict]:
    with connect() as c:
        return [_row(r) for r in c.execute("SELECT * FROM datasets ORDER BY created_at DESC")]


def delete_dataset(id: str):
    with connect() as c:
        c.execute("DELETE FROM datasets WHERE id=?", (id,))


# ---- jobs -----------------------------------------------------------------

def add_job(name: str, base_model: str, dataset_ids: list[str], hparams: dict) -> dict:
    id = new_id()
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id,name,base_model,dataset_ids,hparams,status,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (id, name, base_model, json.dumps(dataset_ids), json.dumps(hparams), "queued", time.time()))
    return get_job(id)


def get_job(id: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone())


def list_jobs() -> list[dict]:
    with connect() as c:
        return [_row(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC")]


def next_queued_job() -> dict | None:
    with connect() as c:
        return _row(c.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone())


def update_job(id: str, **fields):
    for k in JSON_FIELDS & fields.keys():
        fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), id))
