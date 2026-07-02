# Retrieval Tuning Log — Mean Recall@10

Metric: **Mean Recall@10** = mean over the 10 public traces of
`|expected ∩ top-10| / |expected|`, where the labeled shortlist is mapped to
catalog ids. Measured by `python eval/recall_eval.py [--assemble] [--router]`.

Two query modes are reported:
- **user-text** (deterministic): query = concatenation of the trace's user
  messages. Reproducible; isolates retrieval changes from LLM noise.
- **router** (live): query = the router's synthesized `search_query` (one LLM
  call, temperature 0; ~±0.01 run-to-run).

| Step | Change | Query | Mean Recall@10 | Δ |
|------|--------|-------|---------------:|----:|
| 0 | **Baseline** hybrid dense+BM25+RRF (n=50, rrf_k=60) | user-text | **0.5012** | — |
| a | Sweep RRF constant × candidate N | user-text | 0.48–0.52 | ~0 |
| d | **test_type-aware assembly** (inject OPQ32r `P` + Verify G+ `A` defaults) | user-text | **0.6931** | **+0.1919** |
| b | Enrich embedded text (name emphasis; job_levels already present) | user-text | 0.6931 | 0 |
| c | Router `search_query` synthesis | router | 0.7131 | +0.0200 |
| c′ | Router prompt: "include every distinct skill" | router | ~0.72 (0.713–0.727) | +0.01 |

**Baseline 0.5012 → shipped ~0.72** (router query + assembly). Groundedness is
1.0 by construction (URLs looked up from `catalog.json`; non-catalog ids dropped).

---

## (a) RRF constant + dense/BM25 N — *flat, kept defaults*
Swept `rrf_k ∈ {10,20,40,60,100}` × `n ∈ {20,30,50,80,120,200,370}`. The whole
grid sits in 0.48–0.52; the single best cell (n=30, rrf_k=20 → 0.5155) is not
robust (neighbouring cells drop to 0.4955), i.e. noise. **Kept n=50, rrf_k=60.**
Fusion params are not the bottleneck — the misses are missing *documents*, not
mis-ranked ones.

## (b) Enrich embedded text — *no runway, flat*
CLAUDE.md suggested adding `job_levels` + `competencies`. `job_levels` is already
in `assessment_text`; the catalog has **no** competencies/keywords field beyond
the test-type `keys` (already included). Tested name-emphasis (name embedded
twice) to lift crowded short-named tests (`sql-new`, `ms-excel-new`): **0.6931 →
0.6931, no change.** Reverted.

## (c) Router search_query synthesis — *small, real gain*
Using the router's synthesized query instead of raw user text: **0.6931 →
0.7131**. Adding "include EVERY distinct skill/technology, each as its own term"
to the router prompt (targets the C9 Java-flood that drops `sql-new`/`docker-new`)
nudged to ~0.72. Small but real; kept.

## (d) test_type-aware assembly — *the lever, +0.19*
The labeled shortlists are **batteries**: a role-skills spine + two recurring
defaults the user rarely names — a personality measure (**OPQ32r**, `P`, in 7/10
finals) and a cognitive measure (**SHL Verify Interactive G+**, `A`). After
retrieval we reserve slots and guarantee these defaults for hiring-style requests,
unless the user opts out (`app/assembly.py`). This recovered the single largest
miss class: **0.5012 → 0.6931**. Opt-out uses proximity matching so "keep Verify
G+" alongside "drop the OPQ" only drops personality (see `test_assembly.py`).

---

## What didn't work
- **Per-message / per-skill multi-query fusion** (RRF across sub-queries):
  **0.486**, *worse* than a single query — sub-queries pulled unrelated items and
  diluted the exact-name signal. Abandoned.
- **RRF/N tuning** and **text enrichment**: both flat (above).

## Remaining misses (not recoverable by retrieval alone)
Near-duplicate product families where the short canonical variant loses to
`-365`/`-essentials` siblings (C8 `ms-excel-new`), multi-skill JDs where one
skill family floods the pool (C9 `sql-new`, `docker-new`), and second personality
*reports* (C1 `opq-universal-competency-report`). These are ranking/diversity
problems; the next lever would be per-family de-duplication, not more fusion
tuning. **Returns have flattened — stopped here.**
