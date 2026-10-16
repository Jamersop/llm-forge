"""Command-line interface: train and chat without the web server (notebooks, SSH boxes, CI).

    python -m forge.cli info
    python -m forge.cli train --base-model Qwen/Qwen3-1.7B --data qa.jsonl --data manual.txt \\
        --out runs/my-model --set epochs=2 --set max_seq_len=4096
    python -m forge.cli train ... --compute modal --gpu A100-40GB     # rent a cloud GPU
    python -m forge.cli chat --run runs/my-model
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v  # plain string


def cmd_info(_):
    from .hardware import compute_info
    print(json.dumps(compute_info(), indent=2))


def cmd_train(a):
    from . import data
    from .config import DEFAULT_HPARAMS
    from .trainer import setup_logging, train

    hp = {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        if k not in DEFAULT_HPARAMS:
            sys.exit(f"unknown setting {k!r}; choose from {sorted(DEFAULT_HPARAMS)}")
        hp[k] = _parse_value(v)
    sets = []
    for f in a.data:
        kind, records, stats = data.ingest(f, Path(f).read_bytes())
        print(f"{f}: {kind}, {stats['records']} records ({stats['skipped']} skipped)")
        sets.append([kind, records])

    out = Path(a.out)
    setup_logging(out)
    spec = {"job_id": f"cli-{uuid.uuid4().hex[:8]}", "base_model": a.base_model, "hparams": hp,
            "datasets": sets}
    if a.compute == "modal":
        from .remote_modal import run_remote
        summary = run_remote(spec, out, gpu=a.gpu)
    else:
        summary = train(spec, out)
    (out / "forge_run.json").write_text(json.dumps({"base_model": a.base_model, "hparams": hp}))
    print(json.dumps(summary, indent=2))


def cmd_chat(a):
    from .inference import generate, load_weights
    run = Path(a.run)
    meta = json.loads((run / "forge_run.json").read_text()) if (run / "forge_run.json").exists() else {}
    base = a.base_model or meta.get("base_model")
    if (run / "adapter").exists():
        if not base:
            sys.exit("pass --base-model (the model this adapter was trained on)")
        model, tok = load_weights(base, run / "adapter", is_adapter=True)
    else:
        model, tok = load_weights(base or str(run), run / "model" if (run / "model").exists() else None)
    history = []
    print("Chat with your model. Empty line or Ctrl-D to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if not q:
            break
        history.append({"role": "user", "content": q})
        r = generate(model, tok, history, max_new_tokens=a.max_new_tokens, enable_thinking=not a.no_think)
        if r["thinking"]:
            print(f"\n[thinking]\n{r['thinking']}\n[/thinking]")
        print(f"\nmodel> {r['answer']}")
        history.append({"role": "assistant", "content": r["answer"]})


def main(argv=None):
    p = argparse.ArgumentParser(prog="forge", description="LLM Forge command-line interface")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show local hardware and cloud GPU availability").set_defaults(fn=cmd_info)

    t = sub.add_parser("train", help="fine-tune a model on one or more data files")
    t.add_argument("--base-model", required=True, help="HF model id or local path")
    t.add_argument("--data", action="append", required=True, help="dataset file (repeatable)")
    t.add_argument("--out", required=True, help="output folder")
    t.add_argument("--compute", choices=["local", "modal"], default="local")
    t.add_argument("--gpu", default="A10G", help="GPU type for --compute modal")
    t.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting (repeatable)")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("chat", help="chat with a trained run folder")
    c.add_argument("--run", required=True)
    c.add_argument("--base-model")
    c.add_argument("--max-new-tokens", type=int, default=768)
    c.add_argument("--no-think", action="store_true")
    c.set_defaults(fn=cmd_chat)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
