"""End-to-end tests on a tiny local model (no GPU or network needed)."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny") / "model"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tiny_model.py"), str(out)], check=True)
    return str(out)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from forge import api
    api.worker.poll_s = 0.2
    with TestClient(api.app) as c:
        yield c


def test_normalize_formats():
    from forge.data import ingest, DatasetError
    rows = [
        {"question": "2+2?", "answer": "4", "reasoning": "two plus two"},
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]},
        {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]},
    ]
    kind, recs, stats = ingest("x.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
    assert kind == "chat" and stats["records"] == 3 and stats["with_reasoning"] == 1
    assert recs[0]["messages"][1]["content"].startswith("<think>\ntwo plus two\n</think>")

    kind, recs, _ = ingest("c.csv", b"instruction,output\nSay hi,hi\n")
    assert kind == "chat" and recs[0]["messages"][0]["content"] == "Say hi"

    kind, recs, _ = ingest("doc.txt", b"Some domain text.")
    assert kind == "text"

    with pytest.raises(DatasetError):
        ingest("x.jsonl", b'{"foo": 1}')


def test_split_reasoning():
    from forge.data import split_reasoning
    assert split_reasoning("<think>a\nb</think>\n\nans") == ("a\nb", "ans")
    assert split_reasoning("plain") == ("", "plain")
    assert split_reasoning("<think>unfinished") == ("unfinished", "")


def test_label_masking(tiny_model):
    from transformers import AutoTokenizer
    from forge.data import tokenize_chat
    tok = AutoTokenizer.from_pretrained(tiny_model)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "<think>\nwhy\n</think>\n\nanswer one"},
            {"role": "user", "content": "question two"}, {"role": "assistant", "content": "answer two"}]
    f = tokenize_chat(tok, msgs, 512)
    trained = tok.decode([i for i, l in zip(f["input_ids"], f["labels"]) if l != -100])
    assert "answer one" in trained and "answer two" in trained and "why" in trained
    assert "question" not in trained and "sys" not in trained


def test_end_to_end(client, tiny_model):
    r = client.post("/api/datasets", files={"file": ("triage.jsonl",
                    (ROOT / "examples/medical_triage_reasoning.jsonl").read_bytes())})
    assert r.status_code == 200, r.text
    qa = r.json()
    assert qa["kind"] == "chat" and qa["stats"]["with_reasoning"] == 5
    corpus = client.post("/api/datasets", files={"file": ("corpus.txt",
                         (ROOT / "examples/domain_corpus.txt").read_bytes())}).json()
    assert corpus["kind"] == "text"

    r = client.post("/api/jobs", json={"base_model": tiny_model, "dataset_ids": [qa["id"], corpus["id"]],
                                       "hparams": {"epochs": 2, "batch_size": 2, "grad_accum": 1,
                                                   "max_seq_len": 256, "learning_rate": 1e-3}})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "completed", job.get("error") or client.get(f"/api/jobs/{job_id}/logs").text
    assert job["metrics"]["train_loss"] > 0 and job["history"]

    r = client.post("/api/chat", json={"job_id": job_id, "max_new_tokens": 16,
                                       "messages": [{"role": "user", "content": "chest pain?"}]})
    assert r.status_code == 200, r.text
    assert set(r.json()) >= {"thinking", "answer"}

    r = client.post(f"/api/jobs/{job_id}/export")
    assert r.status_code == 200 and r.content[:2] == b"PK"
