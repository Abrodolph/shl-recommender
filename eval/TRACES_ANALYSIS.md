# SHL Public Traces — Analysis

Derived from the 10 public conversation traces in `eval/traces/` via
`eval/traces_loader.py`. These are behavioural targets for the agent and the
replay harness. All numbers below are produced by the loader, not eyeballed.

| Trace | Turns | Commit turn | Final size | Type mix | Opener character |
|------|------:|------------:|-----------:|----------|------------------|
| C1  | 4 | 3 | 3 | P×3 | **Vague** ("solution for senior leadership") |
| C2  | 3 | 2 | 5 | K×3, A, P | Specific role (senior Rust eng) |
| C3  | 5 | 3 | 4 | S×2, B, C, K, P | Specific role, missing a discriminator (language) |
| C4  | 3 | 1 | 5 | K×2, A, B, P, S | Detailed + names the tests it wants |
| C5  | 3 | 1 | 5 | P×3, C, D, K | Moderately vague ("re-skill Sales org") |
| C6  | 3 | 1 | 2 | K, P | Specific role + priority (safety) |
| C7  | 4 | 2 | 5 | K×3, P×2, S | Specific + constraint (Spanish/HIPAA) |
| C8  | 3 | 1 | 5 | K×4, P, S | Moderately vague ("screen admin assistants") |
| C9  | 7 | 3 | 7 | K×5, A, P | Very detailed (full JD paste) |
| C10 | 3 | 1 | 2 | A, B | Specific ("full battery… all graduates") |

Aggregate: shortlist sizes `[3,5,4,5,5,2,5,5,7,2]` → **min 2 / median 5 / max 7,
mean 4.3**. Global type mix `K=20, P=14, S=6, A=4, B=3, C=2, D=1`.
**Groundedness: 43/43** expected items resolve to `data/catalog.json` (recall
ceiling = 1.0; excluding Job Solutions cost us nothing).

---

## (a) How much context the agent needs before committing

**Little — the trigger is "role + at least one discriminating attribute," not a
full interview.** Commit turns: five traces commit on **turn 1** (C4, C5, C6, C8,
C10), two on turn 2 (C2, C7), three on turn 3 (C1, C3, C9). **No trace clarifies
more than twice before producing a first shortlist.**

The agent commits as soon as it has a **role plus one attribute that changes the
shortlist**: seniority, a concrete skill, an explicit test-type request, or a hard
constraint (language, safety). It keeps clarifying only when a *single* missing
discriminator would materially change the picks:
- C3 asks call **language → accent** (drives which SVAR variant).
- C9 asks **backend-vs-frontend**, then **IC-vs-lead** (drives which knowledge
  tests + whether to add a leadership layer).
- C1 asks **who/what for → selection-vs-development** (drives report choice).
- C7 asks **hybrid-vs-Spanish-only** (drives whether English-only knowledge tests
  are admissible).

Implication for our agent: after the single router LLM call, **commit whenever
role + ≥1 discriminating attribute is present**; otherwise ask exactly one targeted
question aimed at the discriminator that would move the shortlist. Cap at ~2
clarifiers — the 8-turn budget and these traces both punish over-clarifying. This
matches CLAUDE.md §4 policy and is now confirmed empirically.

## (b) How vague the openers are

**A spectrum, and mostly not very vague.** Only **C1** is genuinely vague with no
actionable attribute. The rest arrive with a role and at least one constraint —
several are highly detailed (C9 pastes a full JD; C4 names the exact tests it
wants). So:
- Turn-1 vagueness should trigger clarify **only** when no discriminating attribute
  is present (C1-like). Do **not** reflexively clarify a specific opener — C4, C6,
  C10 would be penalised for a turn-1 question.
- The realistic failure mode here is **over-clarifying a specific query**, not
  under-clarifying a vague one. The router must distinguish "vague" (no attribute)
  from "specific" (role + attribute) and only gate turn-1 recommendations for the
  former (CLAUDE.md golden rule #3).

## (c) Do COMPARE queries expect items echoed in recommendations?

**No — a comparison never introduces new recommendations.** COMPARE/Q&A turns do
one of two things, both of which leave the committed shortlist unchanged:
- **Return `recommendations: null`** and answer purely in prose: C3 T4
  ("Is the Contact Center Call Sim different from the Customer Service Phone Sim?"),
  C6 T2 ("difference between DSI and Safety & Dependability 8.0?"), and the legal
  refusal C7 T3.
- **Echo the current standing shortlist unchanged** while answering in prose:
  C5 T2 ("difference between OPQ and OPQ MQ Sales Report?"), C9 T5/T6
  ("is Advanced the right level?", "do we need Verify G+?").

Two invariants hold across every comparison:
1. The items being compared are **already members of the standing shortlist** — the
   comparison is grounded in things already recommended (or already in catalog),
   never a fresh retrieval.
2. `end_of_conversation` stays **false** on a comparison turn.

Implication: on COMPARE, generate a grounded prose answer from catalog text and
**re-emit the current shortlist unchanged** (safe, schema-carries-the-list) — or
`[]` if we prefer; both match the traces. Never treat a comparison as a request for
new picks, and never flip `end_of_conversation` on it.

## (d) Granularity and size of expected shortlists

**Batteries of 2–7 items (median 5); never 1, never near 10.** Recall@10 has plenty
of headroom — the risk is *composition*, not list length.

- **Granularity = one test per distinct dimension.** A JD is decomposed into a test
  per skill: C9 → Core Java, Spring, SQL, AWS, Docker as five separate K tests.
  Skills are not bundled into one "developer" test.
- **Shortlists are multi-dimensional "spines," not single-type lists.** The common
  pattern is: **role-specific knowledge/skill tests (K)** + **one cognitive/ability
  test** (SHL Verify Interactive G+ recurs) + **a default personality layer**
  (OPQ32r appears in 7/10 final shortlists) + sometimes an **SJT/simulation** (B/S).
  Even knowledge-only asks get OPQ32r added as a default (C8), with the agent
  offering to drop it.
- **Type mix skews K then P** (K=20, P=14 of 50 items ≈ 68%), mirroring the catalog
  skew (K and P dominate). A, S, B, C, D fill specific needs (cognitive, simulation,
  SJT, competency/development).

Implication for retrieval/assembly: retrieve inclusively toward ~5–7, then assemble
a battery that **spans dimensions** rather than returning the top-k of a single
type. Include a cognitive (G+) and a personality (OPQ32r) default for hiring-style
requests unless the user constrains otherwise, and split multi-skill roles into
per-skill knowledge tests. Because every labeled item is in our catalog, a miss will
come from **assembly/ranking**, not from a coverage gap.
