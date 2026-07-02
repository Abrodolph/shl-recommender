"""Simulated-user replay harness (a graded deliverable — CLAUDE.md §6).

Runs each trace in ``eval/traces/`` as a REAL multi-turn conversation against
``app.agent.handle`` (which makes one router LLM call per turn), collects the
final shortlist, and computes **conversational** Mean Recall@10 + groundedness.
Unlike ``eval/recall_eval.py`` (a single-shot retrieval proxy), this exercises the
whole agent turn-by-turn: clarify handling, refine/edits, defaults, the turn cap,
and end_of_conversation.

Two user modes:
- ``scripted`` (default): replay the trace's actual user turns in order —
  deterministic and faithful to the labeled data.
- ``llm`` (``--llm-user``): an LLM plays the recruiter from the trace's facts,
  conversing freely (mirrors SHL's harness; nondeterministic, more LLM calls).

    python eval/replay.py                 # scripted, all traces
    python eval/replay.py --llm-user      # LLM-simulated user
    python eval/replay.py --trace C9      # one trace, prints the transcript
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import handle  # noqa: E402
from app.router import _SAFE_CLARIFY_TEXT  # noqa: E402
from app.schemas import SAFE_FALLBACK  # noqa: E402
from eval.metrics import groundedness, mean_recall_at_k, recall_at_k  # noqa: E402
from eval.recall_eval import _catalog_maps, expected_ids_for  # noqa: E402
from eval.traces_loader import Trace, load_traces  # noqa: E402

MAX_TURNS = 8  # evaluator hard cap (user+assistant combined)

# Replies that mean the router/LLM call failed (e.g. rate limit) rather than a
# genuine CLARIFY — used to flag, not silently score, degraded conversations.
_FALLBACK_REPLIES = {_SAFE_CLARIFY_TEXT, SAFE_FALLBACK.reply}


@dataclass
class TurnRecord:
    user: str
    reply: str
    rec_ids: list[str]
    rec_urls: list[str]
    end: bool


@dataclass
class ReplayResult:
    trace_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    final_ids: list[str] = field(default_factory=list)
    final_urls: list[str] = field(default_factory=list)
    turns_used: int = 0
    ended: bool = False
    llm_error: bool = False   # a turn fell back due to an LLM/router failure


def _resp_ids(response, url2id) -> tuple[list[str], list[str]]:
    ids, urls = [], []
    for rec in response.recommendations:
        urls.append(rec.url)
        cid = url2id.get(rec.url.rstrip("/").lower())
        if cid:
            ids.append(cid)
    return ids, urls


# --- scripted user ------------------------------------------------------------
def replay_scripted(trace: Trace, url2id) -> ReplayResult:
    """Feed the trace's real user turns in order; stop at end_of_conversation or
    the turn cap. Final shortlist = recs at the ending turn, else last non-empty."""
    res = ReplayResult(trace_id=trace.id)
    history: list[dict] = []
    for user_msg in trace.user_messages:
        if res.turns_used >= MAX_TURNS:
            break
        history.append({"role": "user", "content": user_msg})
        response = handle(history)
        history.append({"role": "assistant", "content": response.reply})
        ids, urls = _resp_ids(response, url2id)
        if response.reply in _FALLBACK_REPLIES and not ids:
            res.llm_error = True
        res.turns.append(
            TurnRecord(user_msg, response.reply, ids, urls, response.end_of_conversation)
        )
        res.turns_used += 1
        if ids:
            res.final_ids, res.final_urls = ids, urls
        if response.end_of_conversation:
            res.ended = True
            break
    return res


# --- LLM-simulated user -------------------------------------------------------
_USER_SYSTEM = """\
You are role-playing a busy recruiter talking to an assessment-recommendation \
assistant. Your hiring need and the facts you know are below. Answer the \
assistant's questions consistently with these facts; if it asks something the \
facts don't cover, pick a reasonable answer and stay consistent. Keep each reply \
to one or two sentences. When the assistant has given you a shortlist you're \
happy with, reply exactly "that's good, thanks" to end. Do not invent assessment \
names.

YOUR HIRING NEED AND FACTS:
{facts}
"""


def replay_llm(trace: Trace, url2id, client=None) -> ReplayResult:
    from app.llm import get_client

    client = client or get_client()
    facts = "\n".join(f"- {m}" for m in trace.user_messages)
    user_system = _USER_SYSTEM.format(facts=facts)

    res = ReplayResult(trace_id=trace.id)
    history: list[dict] = []
    # The recruiter opens with the trace's first message (its genuine intent).
    next_user = trace.opener
    while res.turns_used < MAX_TURNS:
        history.append({"role": "user", "content": next_user})
        response = handle(history)
        history.append({"role": "assistant", "content": response.reply})
        ids, urls = _resp_ids(response, url2id)
        res.turns.append(
            TurnRecord(next_user, response.reply, ids, urls, response.end_of_conversation)
        )
        res.turns_used += 1
        if ids:
            res.final_ids, res.final_urls = ids, urls
        if response.end_of_conversation:
            res.ended = True
            break
        # The simulated recruiter replies to the assistant (roles inverted).
        sim_history = [
            {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
            for m in history
        ]
        try:
            next_user = client.complete(user_system, sim_history).strip()
        except Exception:
            break
        if "that's good" in next_user.lower():
            # Let the assistant see the confirmation and close the turn.
            history.append({"role": "user", "content": next_user})
            response = handle(history)
            ids, urls = _resp_ids(response, url2id)
            if ids:
                res.final_ids, res.final_urls = ids, urls
            res.turns.append(
                TurnRecord(next_user, response.reply, ids, urls, response.end_of_conversation)
            )
            res.ended = response.end_of_conversation
            break
    return res


# --- runner -------------------------------------------------------------------
def run(mode: str = "scripted", only: str | None = None) -> dict:
    traces = load_traces()
    if only:
        traces = [t for t in traces if t.id == only]
    url2id, name2id = _catalog_maps()
    catalog_ids = set(url2id.values())

    rows, pairs, all_returned = [], [], []
    for tr in traces:
        expected, _ = expected_ids_for(tr, url2id, name2id)
        res = replay_llm(tr, url2id) if mode == "llm" else replay_scripted(tr, url2id)
        rec = recall_at_k(res.final_ids, expected, 10)
        all_returned.extend(res.final_ids)
        # A conversation is only unscorable if an LLM failure left it with NO
        # shortlist at all. If it recovered and delivered a final shortlist, the
        # final list is exactly what the evaluator grades — so score it, even if
        # an earlier clarify turn hit a transient rate limit.
        excluded = res.llm_error and not res.final_ids
        if not excluded:
            pairs.append((res.final_ids, expected))
        rows.append(
            {
                "id": tr.id,
                "recall": rec,
                "hits": len(expected & set(res.final_ids)),
                "expected": len(expected),
                "turns": res.turns_used,
                "ended": res.ended,
                "n_recs": len(res.final_ids),
                "llm_error": res.llm_error,
                "excluded": excluded,
                "result": res,
            }
        )
    n_err = sum(1 for r in rows if r["excluded"])
    return {
        "mode": mode,
        "mean_recall": mean_recall_at_k(pairs, 10),
        "n_scored": len(pairs),
        "n_errored": n_err,
        "groundedness": groundedness(all_returned, catalog_ids),
        "rows": rows,
    }


def print_report(report: dict, show_transcript: bool = False) -> None:
    line = "=" * 74
    print(line)
    print(f"CONVERSATIONAL REPLAY  (user={report['mode']})")
    print(line)
    for row in report["rows"]:
        err = ("  LLM_ERROR(excluded)" if row["excluded"]
               else "  (recovered after transient error)" if row["llm_error"] else "")
        print(f"  {row['id']:<4} recall={row['recall']:.2f}  {row['hits']}/{row['expected']}"
              f"  turns={row['turns']}  ended={row['ended']}  recs={row['n_recs']}{err}")
        if show_transcript:
            for t in row["result"].turns:
                print(f"      U: {t.user[:80]}")
                print(f"      A: {t.reply[:80]}  [{len(t.rec_ids)} recs, end={t.end}]")
    print(line)
    print(f"  MEAN RECALL@10 = {report['mean_recall']:.4f}  "
          f"(scored {report['n_scored']}/{len(report['rows'])} traces; "
          f"{report['n_errored']} LLM-errored)")
    print(f"  GROUNDEDNESS   = {report['groundedness']:.4f}")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-user", action="store_true", help="LLM plays the user.")
    ap.add_argument("--trace", help="Run a single trace and show the transcript.")
    args = ap.parse_args()
    mode = "llm" if args.llm_user else "scripted"
    report = run(mode=mode, only=args.trace)
    print_report(report, show_transcript=bool(args.trace))


if __name__ == "__main__":
    main()
