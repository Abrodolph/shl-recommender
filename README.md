---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

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
> retrieval). Render's free tier (512MB) is tight — bump to `starter`/1GB+ if the
> build OOMs. Render also requires a card on file to deploy any web service
> (free or paid) as an anti-abuse measure — if you'd rather not add one, use
> Hugging Face Spaces below instead (no card required).

### Hugging Face Spaces (no card required)

The YAML frontmatter at the very top of this README (`sdk: docker`,
`app_port: 8000`) is what Spaces reads to configure the deploy — nothing else
to write.

1. Create a free account at huggingface.co (no card needed).
2. **New → Space** → pick a name → **Docker** as the SDK → **CPU basic** hardware
   (free, 2 vCPU / 16GB — more headroom than Render's free tier).
3. Push this repo to the Space's git remote (shown on the Space's page after
   creation; Spaces use `main` as the default branch):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space master:main
   ```
   Authenticate with a Hugging Face **write** access token (Settings → Access
   Tokens) when prompted for a password.
4. In the Space's **Settings → Variables and secrets**, add `GROQ_API_KEY` as a
   **Secret**, and `LLM_PROVIDER=groq` / `LLM_MODEL=llama-3.3-70b-versatile` as
   plain **Variables**.
5. The Space rebuilds automatically. Once live, health/chat are reachable at
   `https://<your-username>-<space-name>.hf.space/health` etc.

## Testing

```bash
python -m pytest -q                   # 118 tests (schema, retrieval, agent, robustness)
```
