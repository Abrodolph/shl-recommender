"""Deterministic retrieval Recall@10 harness (the tuning testbed).

Mean Recall@10 is our headline metric (CLAUDE.md §5). To tune retrieval fast and
reproducibly we measure RETRIEVAL recall directly, without the LLM in the loop:

  - query per trace = the concatenation of that trace's user messages (the full
    context the agent has by commit time), OR the router's synthesized query when
    ``--router`` is passed;
  - expected shortlist items are mapped to catalog ids by URL (fallback: name);
  - recall@10 = |expected ∩ top-10 retrieved ids| / |expected|, averaged.

This isolates retrieval changes (RRF/N, embedded text, test_type bias) from the
router's nondeterminism. Run:

    python eval/recall_eval.py            # deterministic, user-text query
    python eval/recall_eval.py --router   # uses the live router query (LLM)
    python eval/recall_eval.py --write    # also (re)write eval/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import get_retriever  # noqa: E402
from eval.metrics import mean_recall_at_k, recall_at_k  # noqa: E402
from eval.traces_loader import load_traces  # noqa: E402

CATALOG_PATH = ROOT / "data" / "catalog.json"
REPORT_PATH = ROOT / "eval" / "REPORT.md"


def _norm_url(url: str) -> str:
    return url.rstrip("/").lower()


def _norm_name(name: str) -> str:
    return " ".join(name.split()).lower()


def _catalog_maps() -> tuple[dict[str, str], dict[str, str]]:
    recs = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    url2id = {_norm_url(r["url"]): r["id"] for r in recs}
    name2id = {_norm_name(r["name"]): r["id"] for r in recs}
    return url2id, name2id


def expected_ids_for(trace, url2id, name2id) -> tuple[set[str], list[str]]:
    """Map a trace's expected shortlist to catalog ids. Returns (ids, unmapped)."""
    ids: set[str] = set()
    unmapped: list[str] = []
    for rec in trace.expected_shortlist:
        cid = url2id.get(_norm_url(rec.url)) or name2id.get(_norm_name(rec.name))
        if cid:
            ids.add(cid)
        else:
            unmapped.append(rec.name)
    return ids, unmapped


def trace_query(trace) -> str:
    """Deterministic query = all user messages joined (full revealed context)."""
    return " ".join(trace.user_messages)


def router_query(trace) -> str:
    """Live router-synthesized query (one LLM call). Falls back to user text."""
    from app.router import route

    messages = [{"role": "user", "content": m} for m in trace.user_messages]
    result = route(messages)
    return result.search_query or trace_query(trace)


def evaluate(k: int = 10, use_router: bool = False, assemble: bool = False) -> dict:
    traces = load_traces()
    url2id, name2id = _catalog_maps()
    retriever = get_retriever()

    rows = []
    pairs: list[tuple[list[str], set[str]]] = []
    for tr in traces:
        expected, unmapped = expected_ids_for(tr, url2id, name2id)
        query = router_query(tr) if use_router else trace_query(tr)
        if assemble:
            from app.assembly import assemble_ids, default_flags

            add_p, add_c = default_flags(None, " ".join(tr.user_messages))
            retrieved = assemble_ids(
                retriever.retrieve_ids(query, k=k * 2),
                k=k,
                add_personality=add_p,
                add_cognitive=add_c,
            )
        else:
            retrieved = retriever.retrieve_ids(query, k=k)
        r = recall_at_k(retrieved, expected, k)
        pairs.append((retrieved, expected))
        rows.append(
            {
                "id": tr.id,
                "expected": len(expected),
                "unmapped": unmapped,
                "hits": len(expected & set(retrieved[:k])),
                "recall": r,
                "missed": sorted(expected - set(retrieved[:k])),
            }
        )
    mean = mean_recall_at_k(pairs, k)
    return {"k": k, "use_router": use_router, "mean_recall": mean, "rows": rows}


def print_report(result: dict) -> None:
    line = "=" * 74
    print(line)
    print(f"RETRIEVAL RECALL@{result['k']}  (query="
          f"{'router' if result['use_router'] else 'user-text'})")
    print(line)
    for row in result["rows"]:
        flag = " (unmapped: %s)" % row["unmapped"] if row["unmapped"] else ""
        print(f"  {row['id']:<4} recall={row['recall']:.2f}  "
              f"{row['hits']}/{row['expected']}{flag}")
        if row["missed"]:
            print(f"       missed: {row['missed']}")
    print(line)
    print(f"  MEAN RECALL@{result['k']} = {result['mean_recall']:.4f}")
    print(line)


def write_report(result: dict) -> None:
    lines = [
        "# Evaluation Report — Retrieval Recall@10",
        "",
        f"Query mode: **{'router-synthesized' if result['use_router'] else 'user-text (deterministic)'}**  ",
        f"**Mean Recall@{result['k']} = {result['mean_recall']:.4f}**",
        "",
        "| Trace | Recall@10 | Hits/Expected | Missed ids |",
        "|-------|----------:|--------------:|------------|",
    ]
    for row in result["rows"]:
        missed = ", ".join(row["missed"]) if row["missed"] else "—"
        lines.append(
            f"| {row['id']} | {row['recall']:.2f} | "
            f"{row['hits']}/{row['expected']} | {missed} |"
        )
    lines += [
        "",
        "Recall is measured on retrieval (top-10 catalog ids vs the labeled "
        "shortlist mapped to ids). Items the agent adds as defaults (e.g. OPQ32r, "
        "Verify G+) that the user never mentions are the main miss source — see "
        "`eval/TUNING_LOG.md`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--router", action="store_true", help="Use live router query.")
    ap.add_argument("--assemble", action="store_true",
                    help="Apply test_type-aware assembly (default injection).")
    ap.add_argument("--write", action="store_true", help="Write eval/REPORT.md.")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    result = evaluate(k=args.k, use_router=args.router, assemble=args.assemble)
    print_report(result)
    if args.write:
        write_report(result)


if __name__ == "__main__":
    main()
