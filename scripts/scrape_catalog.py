"""Build-time: parse SHL's official catalog file into ``data/catalog.json``.

Per CLAUDE.md §2, SHL's downloadable catalog file takes priority over scraping the
public site, so this script *parses the provided file* rather than crawling. (An
httpx + BeautifulSoup scrape would be the fallback if no file were available.)

Responsibilities (CLAUDE.md §2):
- Load the source catalog (``shl_product_catalog.json`` by default).
- Restrict to Individual Test Solutions ONLY; exclude pre-packaged Job Solutions.
- Emit one record per assessment with a stable ``id`` (slug from the URL) plus the
  fields CLAUDE.md asks for (name, url, test_type, description, job_levels,
  duration, remote_testing, adaptive_irt, ...).
- Persist a single ``data/catalog.json`` (checked into the repo).
- Print a summary: total count, breakdown by test_type, and any rows missing
  name / url / test_type.

Build-time only — NOT a runtime dependency of the API.

--------------------------------------------------------------------------------
NOTES ON DECISIONS DERIVED FROM THE ACTUAL DATA (not assumed):

test_type legend
    The source file stores test types as full-text labels in a ``keys`` array
    (e.g. "Personality & Behavior"), while the API schema and SHL's own
    conversation traces use single-letter codes. The mapping below was CONFIRMED
    against the traces (which show both letter and label): A, B, K, P, S are
    directly attested; C, D, E follow SHL's standard legend. Note the non-obvious
    one: "Assessment Exercises" -> E (NOT A).

multi-key items
    39 of 377 items carry more than one key. The ``keys`` array is stored
    alphabetically, so position gives no priority signal. We therefore keep the
    FULL set of letters in ``test_types`` and expose a single ``test_type`` (the
    alphabetically-first letter) purely for schema compliance. Downstream
    retrieval/compare can use the full ``test_types`` / ``keys``.

Job Solutions exclusion
    The file has no explicit solution-type field. Pre-packaged Job Solutions are
    identified by SHL's naming convention (the whole word "Solution" in the name,
    e.g. "Entry Level Sales Solution"). Every excluded item is printed in the
    summary so the exclusion can be reviewed.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "shl_product_catalog.json"
DEFAULT_OUTPUT = ROOT / "data" / "catalog.json"

# --- test_type legend (confirmed from data + traces; see module docstring) ---
LABEL_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# --- Job Solutions exclusion --------------------------------------------------
# Pre-packaged Job Solutions follow SHL's "... Solution" naming convention.
JOB_SOLUTION_RE = re.compile(r"\bsolution\b", re.IGNORECASE)


def clean_text(value: str) -> str:
    """Collapse embedded newlines / runs of whitespace (some names contain them)."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def slug_from_url(url: str) -> str:
    """Stable id = the ``/view/<slug>/`` path segment of the canonical catalog URL."""
    m = re.search(r"/view/([^/]+)/?", url or "")
    if m:
        return m.group(1).strip().lower()
    # Fallback: last non-empty path segment.
    tail = [p for p in (url or "").rstrip("/").split("/") if p]
    return (tail[-1] if tail else "").strip().lower()


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def letters_for(keys: list[str]) -> tuple[list[str], list[str]]:
    """Return (unknown_labels, sorted_unique_letters) for a record's ``keys``."""
    letters, unknown = set(), []
    for k in keys or []:
        label = clean_text(k)
        letter = LABEL_TO_LETTER.get(label)
        if letter is None:
            unknown.append(label)
        else:
            letters.add(letter)
    return unknown, sorted(letters)


def load_source(path: Path) -> list[dict]:
    # strict=False tolerates raw control chars found inside some string values.
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def build_catalog(source: list[dict]) -> tuple[list[dict], dict]:
    """Transform source rows into catalog records. Returns (records, report)."""
    records: list[dict] = []
    excluded_job_solutions: list[str] = []
    missing: list[dict] = []
    unknown_labels = collections.Counter()
    seen_ids: dict[str, str] = {}
    id_collisions: list[str] = []

    for row in source:
        name = clean_text(row.get("name", ""))
        url = (row.get("link") or "").strip()

        # Exclude pre-packaged Job Solutions (Individual Test Solutions only).
        if JOB_SOLUTION_RE.search(name):
            excluded_job_solutions.append(name)
            continue

        unk, letters = letters_for(row.get("keys"))
        for u in unk:
            unknown_labels[u] += 1
        test_type = letters[0] if letters else ""

        record = {
            "id": slug_from_url(url),
            "name": name,
            "url": url,
            "test_type": test_type,          # single primary letter (schema field)
            "test_types": letters,           # full set (multi-key items keep all)
            "keys": [clean_text(k) for k in (row.get("keys") or [])],
            "description": clean_text(row.get("description", "")),
            "job_levels": row.get("job_levels") or [],
            "languages": row.get("languages") or [],
            "duration": clean_text(row.get("duration", "")),
            "remote_testing": to_bool(row.get("remote")),
            "adaptive_irt": to_bool(row.get("adaptive")),
        }

        # Track rows missing any hard-required field.
        missing_fields = [f for f in ("name", "url", "test_type") if not record[f]]
        if missing_fields:
            missing.append({"name": name or "<no name>", "url": url,
                            "missing": missing_fields})

        # Detect id collisions (slugs must be stable AND unique).
        if record["id"] in seen_ids:
            id_collisions.append(f"{record['id']}  ({seen_ids[record['id']]}  vs  {name})")
        else:
            seen_ids[record["id"]] = name

        records.append(record)

    records.sort(key=lambda r: r["name"].lower())
    report = {
        "excluded_job_solutions": excluded_job_solutions,
        "missing": missing,
        "unknown_labels": unknown_labels,
        "id_collisions": id_collisions,
    }
    return records, report


def print_summary(records: list[dict], report: dict, source_count: int) -> None:
    line = "=" * 66
    print(line)
    print("SHL CATALOG BUILD SUMMARY")
    print(line)
    print(f"Source rows            : {source_count}")
    print(f"Excluded Job Solutions : {len(report['excluded_job_solutions'])}")
    print(f"Individual Test count  : {len(records)}")
    print()

    # test_type legend (as used).
    print("test_type legend (label -> letter):")
    for label, letter in sorted(LABEL_TO_LETTER.items(), key=lambda kv: kv[1]):
        print(f"    {letter} = {label}")
    print()

    # Breakdown by primary test_type.
    by_primary = collections.Counter(r["test_type"] or "<none>" for r in records)
    print("Breakdown by primary test_type:")
    for letter, count in sorted(by_primary.items()):
        print(f"    {letter}: {count}")
    print()

    # Breakdown counting every letter (multi-key items counted in each).
    by_any = collections.Counter(l for r in records for l in r["test_types"])
    multi = sum(1 for r in records if len(r["test_types"]) > 1)
    print(f"Breakdown by ANY test_type (multi-key counted in each; {multi} multi-key items):")
    for letter, count in sorted(by_any.items()):
        print(f"    {letter}: {count}")
    print()

    # Excluded job solutions (for review).
    print("Excluded Job Solutions (name-based heuristic):")
    for n in report["excluded_job_solutions"]:
        print(f"    - {n}")
    print()

    # Rows missing hard-required fields.
    print(f"Rows missing name/url/test_type: {len(report['missing'])}")
    for m in report["missing"]:
        print(f"    - {m['name']!r}  missing={m['missing']}  url={m['url']!r}")
    print()

    if report["unknown_labels"]:
        print("WARNING: unmapped test-type labels (not in legend):")
        for label, count in report["unknown_labels"].most_common():
            print(f"    - {label!r} x{count}")
        print()

    if report["id_collisions"]:
        print("WARNING: duplicate slug ids:")
        for c in report["id_collisions"]:
            print(f"    - {c}")
        print()

    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"source catalog file (default: {DEFAULT_SOURCE.name})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    source = load_source(args.source)
    records, report = build_catalog(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(records, report, source_count=len(source))
    print(f"Wrote {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
