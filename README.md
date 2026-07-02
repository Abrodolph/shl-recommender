# Conversational SHL Assessment Recommender

A stateless FastAPI service that acts as a conversational agent, helping recruiters
go from a vague hiring intent (e.g. "I'm hiring a Java developer") to a grounded
shortlist of SHL assessments through dialogue.

> See [CLAUDE.md](./CLAUDE.md) for the full specification, hard constraints, and
> design decisions. It is the single source of truth for this project.

## What it does

- **Clarifies** vague queries before recommending.
- **Recommends** 1–10 SHL assessments (name + catalog URL + test_type) once it has enough context.
- **Refines** the shortlist when the user changes constraints mid-conversation.
- **Compares** assessments using catalog evidence.
- **Refuses** off-topic requests and prompt-injection attempts.
- **Never** returns a URL that isn't in the scraped catalog.

## API

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → stateless; full conversation history in, `{reply, recommendations, end_of_conversation}` out.

## Quickstart

```bash
# 1. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate        # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env         # then fill in GROQ_API_KEY

# 3. Run the service (catalog.json + embeddings.npy already ship in data/)
uvicorn app.main:app --reload
```

`data/catalog.json` and `data/embeddings.npy` are committed, so no build step is
needed to run. To regenerate them from source:

```bash
pip install beautifulsoup4         # build-time only
python scripts/scrape_catalog.py   # SHL catalog -> data/catalog.json
python scripts/build_embeddings.py # catalog.json -> data/embeddings.npy (+ ids)
```

Verify your LLM key works: `python scripts/smoke_llm.py "say ready"`.

## Evaluation

```bash
python eval/recall_eval.py            # deterministic retrieval Recall@10
python eval/recall_eval.py --assemble # + test_type-aware battery assembly
python eval/recall_eval.py --router --assemble   # live router query (shipped config)
```

See [`eval/REPORT.md`](./eval/REPORT.md) for the current numbers and
[`eval/TUNING_LOG.md`](./eval/TUNING_LOG.md) for before/after per change
(baseline **0.50 → ~0.72** Mean Recall@10). Approach write-up:
[`APPROACH.md`](./APPROACH.md).

## Deployment

The service is 12-factor: config via env vars, binds `$PORT`, and ships the
catalog + embeddings in the image so nothing large downloads at boot. `/health`
returns 200 immediately while the embedding model + indexes warm in a background
thread (`app/main.py` lifespan).

### Docker (portable — Render / Fly / Railway / HF Spaces)

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=sk_... shl-recommender
curl http://localhost:8000/health           # {"status":"ok"}
```

### Render (one-click via blueprint)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point at the repo. `render.yaml` provisions a
   Docker web service with `healthCheckPath: /health`.
3. Set `GROQ_API_KEY` in the dashboard (it's `sync: false`, never committed).
4. Deploy. Render injects `$PORT`; the container's `CMD` binds to it.

**Env vars:** `GROQ_API_KEY` (required), `LLM_PROVIDER=groq`,
`LLM_MODEL=llama-3.3-70b-versatile`. Other platforms: use the same Dockerfile
(Fly `fly launch` autodetects it; Railway/HF Spaces select "Docker").

> Note: the image includes CPU-only `torch` (needed to encode queries for dense
> retrieval). The `starter`/512MB tier is tight — bump to 1GB+ if the build OOMs.

## Testing

```bash
python -m pytest -q                   # 118 tests (schema, retrieval, agent, robustness)
```
