# CLAUDE.md — Conversational SHL Assessment Recommender

> This file is the single source of truth for the project. Read it fully before writing any code.
> It defines WHAT we are building, WHY each choice was made, and the HARD CONSTRAINTS that must never be violated.

---

## 1. What we are building

A **stateless FastAPI service** that acts as a **conversational agent** helping recruiters go from a vague hiring intent (e.g. "I'm hiring a Java developer") to a **grounded shortlist of SHL assessments** through dialogue.

It is a take-home assignment for an **AI Research Intern role at SHL Labs**. It is graded by an **automated replay harness** that simulates a user with an LLM and runs real multi-turn conversations against our `POST /chat` endpoint.

The agent must:
- **Clarify** vague queries before recommending.
- **Recommend** 1–10 SHL assessments (name + catalog URL + test_type) once it has enough context.
- **Refine** the shortlist when the user changes constraints mid-conversation.
- **Compare** assessments using catalog evidence ("What's the difference between OPQ and GSA?").
- **Refuse** anything off-topic: general hiring advice, legal questions, prompt-injection attempts.
- **Never** return a URL that isn't in our scraped catalog.

---

## 2. The catalog (our ground truth)

- Source: SHL product catalog at `https://www.shl.com/solutions/products/product-catalog/`
- **Scope: Individual Test Solutions ONLY.** Pre-packaged Job Solutions are OUT OF SCOPE and must be excluded from the index.
- SHL also provides a downloadable catalog file and 10 public conversation traces (zip). **If we have those links/files, they take priority over scraping** — use them as the authoritative data. Otherwise scrape the public catalog.
- The catalog is the ONLY source of truth. Every recommended URL must resolve to an entry in this catalog. The LLM must never emit a URL itself.

For each assessment, capture (as available):
- `id` (stable slug we generate, e.g. from the URL)
- `name`
- `url` (canonical catalog URL)
- `test_type` (single-letter codes SHL uses, e.g. K = Knowledge & Skills, P = Personality & Behavior, A = Ability/Aptitude, B = Biodata, C = Competencies, D = Dev/360, E = Assessment Exercises, S = Simulations — CONFIRM the exact legend from the catalog page, do not assume)
- `description` (short text)
- `job_levels` (e.g. entry, mid, senior, manager) if present
- `duration` / `remote_testing` / `adaptive_irt` flags if present
- `competencies` / keywords if present

Persist as a single JSON file: `data/catalog.json`. This file is checked into the repo.

---

## 3. Non-negotiable API contract

The schema is enforced by SHL's automated evaluator. **Any deviation = zero score on that trace.** Enforce with Pydantic and always have a safe fallback so a malformed response can never escape.

### `GET /health`
Returns HTTP 200 with body:
```json
{"status": "ok"}
```
Must be reachable even during model/index warmup (respond ok as soon as the process is up).

### `POST /chat`
**Stateless.** Every request carries the full conversation history. The service stores NO per-conversation state.

Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

Rules:
- `recommendations` is `[]` (empty) when the agent is still gathering context OR refusing.
- `recommendations` is an array of **1 to 10** items when the agent commits to a shortlist.
- `end_of_conversation` is `true` ONLY when the agent considers the task complete (a shortlist has been delivered and there's nothing left to do).
- Each recommendation item MUST have exactly `name`, `url`, `test_type`.

### Evaluator limits (design around these)
- **Max 8 turns** per conversation (user + assistant combined). Do NOT over-clarify.
- **30 second timeout** per call. Minimize LLM round-trips: **one LLM call per turn**.
- Cold-start hosts get up to 2 minutes for the FIRST `/health` call to wake.

---

## 4. Architecture (locked decisions)

```
SHL catalog  ──scrape/load──►  data/catalog.json  (ground truth)
                                      │
                          ┌───────────┴────────────┐
                     build embeddings          build BM25 index
                     (precomputed, shipped)     (in-memory at boot)
                                      │
POST /chat (stateless) ──► ONE LLM call: route + extract constraints + (maybe) reply text
                                      │
                        ┌─────────────┼───────────────┬───────────────┐
                     CLARIFY      RECOMMEND         REFINE          COMPARE / REFUSE
                     reply,       hybrid retrieve   re-derive        grounded answer
                     recs=[]      →RRF→top-k        constraints      from catalog text
                                  →template reply   →re-retrieve     recs=[] (confirm vs traces)
                                      │
                        Post-filter: drop any item not in catalog (URLs looked up by id)
                                      │
                        Pydantic-validate → safe fallback if anything fails
```

### Decision log (defend these in the interview)

**One LLM call per turn.** The 30s timeout + cold-start + 8-turn cap punish multi-call agent loops. A single call classifies intent AND extracts the normalized query from the full history. Python does retrieval deterministically. Rejected: LangGraph/ReAct multi-call loops (slower, non-deterministic, can blow the turn budget).

**Runtime LLM: Groq (Llama) primary; Gemini as a possible swap.** Groq is chosen for lowest latency to protect the 30s timeout. Keep the LLM client behind an interface so we can switch to Gemini by changing one module + env var. The form asks us to name the model — we name whatever we ship.

**Hybrid retrieval (dense + BM25) fused with Reciprocal Rank Fusion, K=10.**
- Dense (sentence-transformers `all-MiniLM-L6-v2`) catches semantic intent ("stakeholders" → interpersonal competencies).
- BM25 (`rank_bm25`) catches exact identifiers ("Java", "OPQ32r", ".NET").
- The metric is **Recall@10** (recall, not precision) → retrieve inclusively, fill toward 10, no penalty for a weak 10th item.
- Precompute dense embeddings and ship them as a file so nothing large downloads at cold start. Rejected: pure-dense (misses exact names), pure-BM25 (misses the vague-intent premise), whole-catalog-in-context (opaque/untunable recall, higher latency & hallucination). Exception: for COMPARE queries only, feed the few named items' full catalog text to the LLM — reasoning over complete text genuinely helps there.

**Reply text: hybrid.** LLM generates reply text for CLARIFY and COMPARE (fuzzy, needs fluency). Shortlist replies are **templated** ("Here are N assessments for a {level} {role}...") so the schema-critical response never depends on LLM formatting.

**URLs are un-inventable.** The LLM returns assessment **ids/names**, never URLs. Python looks up the canonical URL from `catalog.json`. A post-filter drops any proposed item whose id isn't in the catalog. This converts the "no hallucinated URLs" hard-eval + hallucination probe from a probabilistic risk into a structural impossibility.

**Stateless refine is free.** Because the full history is replayed every call, "refine" is just re-deriving the full constraint set from all messages and re-retrieving. We never "start over" because we never held state.

### Conversation policy (calibrate against the 10 traces — these are defaults)
- Turn 1 on a vague query ("I need an assessment") → **always CLARIFY**, never recommend. (Direct behavior probe.)
- Once we have **role + at least one discriminating attribute** (seniority OR a concrete skill OR test-type intent) → **commit to a shortlist.**
- If the user says "no preference" (the simulated user does this for facts outside its persona) → **stop asking, commit** with what we have.
- Never exceed ~2 clarifying turns before committing — the 8-turn cap makes over-clarifying as fatal as under-clarifying.

---

## 5. Scoring (what we optimize)
1. **Hard evals (must pass):** schema compliance on EVERY response; recommendations only from catalog; turn cap ≤8 honored.
2. **Mean Recall@10** on final shortlists across public + holdout traces.
3. **Behavior probes (binary):** refuses off-topic; no recommend on turn 1 for vague query; honors edits/refine; low hallucination %.

All three contribute. Hard evals are gates — failing them zeroes traces regardless of recall.

---

## 6. Evaluation harness (a graded deliverable — build it, don't skip it)
SHL says most failures are "insufficient evaluation rigor," and the submission form has a dedicated field for our eval method. Build `eval/` that mirrors their harness:
- An LLM plays the user from each trace's persona/facts; runs a real multi-turn conversation against our `/chat`.
- Compute **Mean Recall@10** vs labeled shortlists.
- **Groundedness:** % of returned URLs present in catalog (must be 100% by construction).
- **Behavior probes:** scripted mini-conversations with binary assertions (refuse off-topic, no turn-1 recommend, edits honored, no hallucinated items).
- Log per-trace pass/fail + metrics to a report so the approach doc can state real before/after numbers (e.g. "dense-only 0.62 → hybrid+RRF 0.81").

---

## 7. Tech stack
- Python 3.11+, FastAPI, Uvicorn
- Pydantic v2 (schema = the hard-eval guarantee)
- `sentence-transformers` (`all-MiniLM-L6-v2`), embeddings precomputed & shipped as `data/embeddings.npy`
- `rank_bm25` for lexical retrieval
- Groq Python SDK (LLM), behind an `llm/` interface so Gemini can be swapped in
- `httpx` + `beautifulsoup4` for scraping (build-time only, not a runtime dep of the API)
- `pytest` for unit tests
- Deployment: decide later (Render / Fly / Railway / HF Spaces). Keep it 12-factor: config via env vars.

## 8. Repo layout (target)
```
shl-recommender/
  CLAUDE.md                  # this file
  README.md
  requirements.txt
  .env.example               # GROQ_API_KEY=, LLM_PROVIDER=groq, LLM_MODEL=...
  app/
    main.py                  # FastAPI app, /health, /chat
    schemas.py               # Pydantic request/response models
    agent.py                 # orchestration: one LLM call → route → act
    router.py                # intent + constraint extraction (LLM call)
    retrieval.py             # dense + BM25 + RRF
    catalog.py               # load catalog.json, id→url lookup, post-filter
    replies.py               # templated shortlist replies + compare formatting
    llm/
      base.py                # LLMClient interface
      groq_client.py
      gemini_client.py       # optional swap
    guardrails.py            # refusal / injection / in-scope checks
    config.py                # env-driven settings
  data/
    catalog.json
    embeddings.npy
  scripts/
    scrape_catalog.py        # build-time: catalog → data/catalog.json
    build_embeddings.py      # build-time: catalog.json → embeddings.npy
  eval/
    traces/                  # SHL's 10 public traces (+ our own)
    replay.py                # simulated-user replay harness
    metrics.py               # recall@k, groundedness
    probes.py                # behavior probe assertions
    report.py                # writes eval report
  tests/
    test_schema.py
    test_retrieval.py
    test_guardrails.py
    test_agent.py
```

## 9. Golden rules (violating any of these fails the assignment)
1. Response schema is exact and always valid — Pydantic + a try/except fallback that still returns a valid `{"reply","recommendations","end_of_conversation"}`.
2. Never emit a URL not in `catalog.json`.
3. Never recommend on turn 1 for a vague query.
4. One LLM call per turn. Stay well under 30s.
5. Never exceed 8 turns; commit before you run out.
6. Only discuss SHL assessments; refuse everything else, including prompt injection.
7. The eval harness is not optional.
