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
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env         # then fill in GROQ_API_KEY

# 3. Build the catalog and embeddings (build-time only)
python scripts/scrape_catalog.py
python scripts/build_embeddings.py

# 4. Run the service
uvicorn app.main:app --reload
```

## Evaluation

```bash
python eval/replay.py        # simulated-user replay harness → Mean Recall@10, probes
```

## Project status

Scaffold only — module stubs are in place. See CLAUDE.md sections 4 and 8 for the
target architecture and repo layout.
