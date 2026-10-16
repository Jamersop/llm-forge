"""Dataset ingestion, normalization and tokenization.

Every uploaded file is normalized to one of two kinds:

* ``chat`` – a list of ``{"messages": [{"role", "content"}, ...]}`` records. Assistant
  turns may carry reasoning, stored inline as ``<think>…</think>`` before the answer,
  which is the format reasoning models (Qwen3, DeepSeek-R1) are trained on.
* ``text`` – raw domain documents (``{"text": ...}``) used for domain adaptation
  (continued pre-training): the model reads the corpus and learns its facts and style.

Accepted inputs: ``.jsonl``/``.json``/``.csv`` with any of the field layouts below, or
``.txt``/``.md`` files (treated as raw text; documents separated by two blank lines).

  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
  {"instruction"|"prompt"|"question": ..., "input"?: ..., "output"|"response"|"answer": ...,
   "reasoning"|"thinking"|"rationale"|"chain_of_thought"?: ..., "system"?: ...}
  {"text": ...}
"""
from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from .config import THINK_CLOSE, THINK_OPEN

PROMPT_KEYS = ("instruction", "prompt", "question", "query", "input_text")
INPUT_KEYS = ("input", "context")
ANSWER_KEYS = ("output", "response", "answer", "completion", "target")
REASONING_KEYS = ("reasoning", "thinking", "rationale", "chain_of_thought", "cot", "explanation")
TEXT_KEYS = ("text", "content", "document", "body")
ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant",
                "user": "user", "assistant": "assistant", "system": "system"}


class DatasetError(ValueError):
    pass


def _first(rec: dict, keys) -> str | None:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def with_reasoning(answer: str, reasoning: str | None) -> str:
    if not reasoning:
        return answer
    return f"{THINK_OPEN}\n{reasoning.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"


def split_reasoning(text: str) -> tuple[str, str]:
    """Split generated text into (thinking, answer)."""
    if THINK_CLOSE in text:
        think, answer = text.split(THINK_CLOSE, 1)
        return think.replace(THINK_OPEN, "").strip(), answer.strip()
    if THINK_OPEN in text:  # thinking never finished (hit token limit)
        return text.split(THINK_OPEN, 1)[1].strip(), ""
    return "", text.strip()


def _normalize_messages(raw) -> list[dict]:
    msgs = []
    for m in raw:
        role = ROLE_ALIASES.get(str(m.get("role") or m.get("from") or "").lower())
        content = m.get("content", m.get("value"))
        if role is None or not isinstance(content, str):
            raise DatasetError(f"bad message: {m!r:.120}")
        reasoning = _first(m, REASONING_KEYS + ("reasoning_content",))
        if role == "assistant" and reasoning:
            content = with_reasoning(content, reasoning)
        msgs.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in msgs):
        raise DatasetError("conversation has no assistant turn")
    return msgs


def normalize_record(rec: dict) -> tuple[str, dict]:
    """Return (kind, normalized_record) for one raw record."""
    if isinstance(rec.get("messages") or rec.get("conversations"), list):
        return "chat", {"messages": _normalize_messages(rec.get("messages") or rec["conversations"])}

    prompt, answer = _first(rec, PROMPT_KEYS), _first(rec, ANSWER_KEYS)
    if prompt and answer:
        extra = _first(rec, INPUT_KEYS)
        user = f"{prompt}\n\n{extra}" if extra else prompt
        msgs = []
        if system := _first(rec, ("system", "system_prompt")):
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": "user", "content": user},
                 {"role": "assistant", "content": with_reasoning(answer, _first(rec, REASONING_KEYS))}]
        return "chat", {"messages": msgs}

    if text := _first(rec, TEXT_KEYS):
        return "text", {"text": text}

    raise DatasetError(f"unrecognized record fields: {sorted(rec)[:8]}")


def read_raw(filename: str, data: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()
    text = data.decode("utf-8-sig")
    if ext in (".txt", ".md"):
        return [{"text": p.strip()} for p in text.split("\n\n\n") if p.strip()] or [{"text": text}]
    if ext == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    if ext == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):  # {"data": [...]} style wrappers
            obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
        return obj
    if ext == ".jsonl":
        out = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(f"line {i}: invalid JSON ({e.msg})") from e
        return out
    raise DatasetError(f"unsupported file type {ext!r}; use .jsonl, .json, .csv, .txt or .md")


def ingest(filename: str, data: bytes) -> tuple[str, list[dict], dict]:
    """Parse + normalize an upload. Returns (kind, records, stats)."""
    raw = read_raw(filename, data)
    if not raw:
        raise DatasetError("file contains no records")
    kinds, records, errors = {}, [], []
    for i, rec in enumerate(raw):
        try:
            kind, norm = normalize_record(rec)
        except DatasetError as e:
            errors.append(f"record {i}: {e}")
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
        records.append((kind, norm))
    if not records:
        raise DatasetError("no usable records. " + "; ".join(errors[:5]))
    if len(kinds) > 1:
        raise DatasetError(f"file mixes chat and raw-text records {kinds}; upload them separately")
    kind = next(iter(kinds))
    records = [r for _, r in records]

    stats = {"records": len(records), "skipped": len(errors), "errors": errors[:10]}
    if kind == "chat":
        stats["with_reasoning"] = sum(
            any(THINK_OPEN in m["content"] for m in r["messages"] if m["role"] == "assistant")
            for r in records)
        stats["avg_turns"] = round(sum(len(r["messages"]) for r in records) / len(records), 2)
        chars = sum(len(m["content"]) for r in records for m in r["messages"])
    else:
        chars = sum(len(r["text"]) for r in records)
    stats["chars"] = chars
    stats["approx_tokens"] = chars // 4
    return kind, records, stats


def save(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- tokenization -----------------------------------------------------------

FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def ensure_chat_template(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _render(tokenizer, messages, add_generation_prompt=False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)


def tokenize_chat(tokenizer, messages: list[dict], max_len: int, train_on_prompt=False) -> dict | None:
    """Tokenize a conversation; labels are -100 everywhere except assistant turns.

    Assistant spans are located by rendering the conversation prefix up to each turn
    and using character offsets, so it works with any chat template. Templates that
    rewrite history (e.g. Qwen3 strips <think> from earlier turns) break the prefix
    property; in that case we fall back to training on the final assistant turn only.
    """
    full = _render(tokenizer, messages)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    spans = []  # character spans of assistant content to train on
    if train_on_prompt:
        spans = [(0, len(full))]
    else:
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            start_text = _render(tokenizer, messages[:i], add_generation_prompt=True)
            end_text = _render(tokenizer, messages[: i + 1])
            if full.startswith(start_text) and full.startswith(end_text):
                spans.append((len(start_text), len(end_text)))
        if not spans:
            last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
            start_text = _render(tokenizer, messages[:last], add_generation_prompt=True)
            if not full.startswith(start_text):
                return None
            spans = [(len(start_text), len(full))]

    labels = [-100] * len(ids)
    for t, (a, b) in enumerate(offsets):
        if any(s <= a < e for s, e in spans):
            labels[t] = ids[t]
    if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id) \
            and tokenizer.eos_token not in full[-20:]:
        ids, labels = ids + [tokenizer.eos_token_id], labels + [tokenizer.eos_token_id]

    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None  # answer got truncated away entirely
    return {"input_ids": ids, "labels": labels}


def tokenize_text_corpus(tokenizer, texts: list[str], block: int) -> list[dict]:
    """Concatenate documents and cut into fixed-size blocks (continued pre-training)."""
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    stream: list[int] = []
    for t in texts:
        stream += tokenizer(t, add_special_tokens=False)["input_ids"] + eos
    blocks = [stream[i:i + block] for i in range(0, len(stream), block)]
    blocks = [b for b in blocks if len(b) >= min(32, block)]
    return [{"input_ids": b, "labels": list(b)} for b in blocks]


def build_features(tokenizer, datasets: list[tuple[str, list[dict]]], hp: dict) -> tuple[list, list, dict]:
    """Turn (kind, records) pairs into shuffled train/eval feature lists."""
    ensure_chat_template(tokenizer)
    max_len, system = hp["max_seq_len"], hp.get("system_prompt") or ""
    feats, dropped = [], 0
    for kind, records in datasets:
        if kind == "text":
            feats += tokenize_text_corpus(tokenizer, [r["text"] for r in records], max_len)
            continue
        for r in records:
            msgs = r["messages"]
            if system and msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": system}] + msgs
            f = tokenize_chat(tokenizer, msgs, max_len, hp.get("train_on_prompt", False))
            if f is None:
                dropped += 1
            else:
                feats.append(f)
    if not feats:
        raise DatasetError("no trainable examples after tokenization (max_seq_len too small?)")

    random.Random(hp.get("seed", 42)).shuffle(feats)
    n_eval = int(len(feats) * hp.get("eval_ratio", 0)) if len(feats) >= 20 else 0
    info = {"train_examples": len(feats) - n_eval, "eval_examples": n_eval, "dropped": dropped,
            "train_tokens": sum(len(f["input_ids"]) for f in feats[n_eval:])}
    return feats[n_eval:], feats[:n_eval], info
