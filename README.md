# LLM Forge

A self-hosted platform for fine-tuning open LLMs on your own domain data. You pick a
base model, upload data, train, and then chat with the result. Training uses LoRA,
QLoRA or a full fine-tune, built on Hugging Face `transformers` + `peft`.

```
 1 · Data         2 · Train                  3 · Jobs                 4 · Playground
 upload JSONL/ →  pick base model,       →   queue, live loss/   →    chat with the model, see
 CSV/TXT          datasets, settings         perplexity, logs,        its <think> reasoning,
                                             export merged model      compare against the base
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # add bitsandbytes on GPU machines for QLoRA
uvicorn forge.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Data, runs and the SQLite DB go to `./forge_data`. Set `FORGE_HOME` to put them somewhere else.

No GPU or Hugging Face access? Generate a tiny local model to try the full flow:
`python scripts/make_tiny_model.py ./tiny-model`, then enter its absolute path as the base
model. Run the tests with `pytest tests`.

## How the model ends up good at language, reasoning *and* your domain

1. **Start from a reasoning model.** General language and reasoning skill comes from
   pre-training, not from your fine-tune. The catalog favours models that already think
   in `<think>…</think>`: Qwen3, DeepSeek-R1-Distill and Phi-4-mini-reasoning.
2. **Use LoRA by default.** It trains about 1% of the weights, so the model gains domain
   skill without forgetting general skills (catastrophic forgetting). Use `qlora` for big
   models on small GPUs. Use `full` only with lots of data and compute.
3. **Train on reasoning traces.** Add a `reasoning` field to Q&A examples. The platform
   formats it as `<think>reasoning</think> answer`, and the loss covers both, so the
   model learns *how* to reach the answer in your domain, not just the answer.
4. **Teach knowledge with a domain corpus.** Upload manuals, papers or policies as
   `.txt`/`.md` (or `{"text": ...}`). They train as continued pre-training (next-token
   prediction over the corpus). Combine one corpus dataset with one Q&A dataset in a
   single job.
5. **Loss on answers only.** Prompt and system tokens are masked by default
   (`train_on_prompt=false`). The model learns to respond; it does not learn to imitate
   the questions.

### Data formats (auto-detected)

```jsonc
// Q&A with reasoning (recommended). Aliases: prompt/question, response/answer, thinking/rationale/cot
{"instruction": "...", "input": "optional context", "reasoning": "step by step...", "output": "final answer"}
// Multi-turn chat (OpenAI or ShareGPT style)
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
// Raw domain text
{"text": "..."}
```

CSV with the same column names also works. See `examples/` for samples.

### Practical data advice
- 500–5,000 high-quality reasoning examples beat 50k noisy ones.
- No reasoning traces? Generate them: give each Q&A pair to a strong model and ask it to
  explain the steps from question to the known answer. Then check a sample by hand.
- Mix in some general instruction data (5–20%) if you see the model losing general skills.
- Watch **eval loss**. If it rises while train loss keeps falling, you are overfitting:
  cut epochs or the learning rate.

## Settings cheat-sheet

| Setting | Default | Notes |
|---|---|---|
| method | lora | `qlora` = 4-bit base model, CUDA + bitsandbytes only |
| epochs | 3 | 1–2 for big datasets or corpora |
| learning_rate | 2e-4 | LoRA. Use ~1e-5 for `full` |
| batch_size × grad_accum | 2 × 8 | effective batch of 16 |
| max_seq_len | 2048 | reasoning traces are long; raise to 4096+ if the GPU allows |
| lora_r / alpha | 16 / 32 | raise r to 64 for heavy domain shift |
| eval_ratio | 0.05 | held-out split, used once there are ≥20 examples |

Rough GPU needs with LoRA (bf16): 0.6–2B → 8–16 GB, 4B → 24 GB, 8B → 40 GB.
QLoRA roughly halves these.

## API

| | |
|---|---|
| `GET /api/models` | catalog + default hyperparameters |
| `POST /api/datasets` (multipart `file`, `name`) | upload + validate |
| `GET /api/datasets`, `GET /api/datasets/{id}/preview`, `DELETE /api/datasets/{id}` | |
| `POST /api/jobs` `{base_model, dataset_ids, hparams, name}` | queue a training job |
| `GET /api/jobs`, `GET /api/jobs/{id}` (includes loss history), `GET /api/jobs/{id}/logs` | |
| `POST /api/jobs/{id}/cancel`, `DELETE /api/jobs/{id}` | |
| `POST /api/jobs/{id}/export` | merge LoRA into the base model → zip in HF format (load with vLLM, TGI or transformers) |
| `POST /api/chat` `{job_id \| base_model, messages, enable_thinking, ...}` | returns `{thinking, answer}` |

## Architecture

```
forge/
  api.py        FastAPI app + static UI
  worker.py     job queue thread; runs each job as a separate `python -m forge.trainer` process
  trainer.py    model loading (LoRA/QLoRA/full), Trainer loop, metrics.jsonl
  data.py       format detection, normalization, chat-template tokenization with answer-only masking
  inference.py  adapter loading, generation with <think> parsing, merge/export
  db.py         SQLite (datasets, jobs)
  static/       single-page UI
```

Training runs in a child process, so an OOM or crash cannot take down the server, and
cancelling a job kills that process. The queue runs one job at a time on one machine.

## Not built yet (natural next steps)
- Authentication and per-user workspaces (currently single-tenant)
- Multi-GPU / remote GPU workers (e.g. distribute jobs to a Ray cluster or cloud GPUs)
- Built-in reasoning-trace generation from a teacher model for plain Q&A data
- Preference tuning (DPO/GRPO) after SFT, for further gains in reasoning quality
- Automatic benchmark evals (e.g. a held-out domain quiz, scored before and after)
