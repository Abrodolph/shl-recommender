"""Catalog loading, id->url lookup, and the hallucination post-filter.

Responsibilities (CLAUDE.md §4, §9):
- Load ``data/catalog.json`` (the single source of truth) at boot.
- Provide id -> canonical URL / name / test_type and full-record lookups.
- Post-filter: given proposed items from the agent, drop any whose id is not in
  the catalog, so a recommended URL can never be one that isn't in the catalog.
- Expose full text of a few named items for COMPARE grounding.

Golden rule enforced here: never emit a URL not in ``catalog.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "catalog.json"


def _display_test_type(record: dict) -> str:
    """Contract ``test_type`` string. Prefer the full list joined ("K,S") to
    mirror how the traces render multi-key items; fall back to the primary
    letter. Always non-empty for a valid catalog record."""
    types = record.get("test_types") or []
    if types:
        return ",".join(types)
    return record.get("test_type") or ""


@dataclass
class Catalog:
    records: list[dict]

    def __post_init__(self) -> None:
        self._by_id: dict[str, dict] = {r["id"]: r for r in self.records}
        self._urls: set[str] = {r["url"] for r in self.records}

    # --- lookups --------------------------------------------------------------
    def has(self, item_id: str) -> bool:
        return item_id in self._by_id

    def get(self, item_id: str) -> dict | None:
        return self._by_id.get(item_id)

    def url_for(self, item_id: str) -> str | None:
        rec = self._by_id.get(item_id)
        return rec["url"] if rec else None

    def name_for(self, item_id: str) -> str | None:
        rec = self._by_id.get(item_id)
        return rec["name"] if rec else None

    def test_type_for(self, item_id: str) -> str | None:
        rec = self._by_id.get(item_id)
        return _display_test_type(rec) if rec else None

    @property
    def ids(self) -> list[str]:
        return list(self._by_id.keys())

    # --- recommendation building (canonical name/url/test_type) ---------------
    def to_recommendation(self, item_id: str) -> dict | None:
        """Build a contract recommendation dict from the catalog, so name/url/
        test_type are always canonical (the LLM never supplies these)."""
        rec = self._by_id.get(item_id)
        if not rec:
            return None
        return {
            "name": rec["name"],
            "url": rec["url"],
            "test_type": _display_test_type(rec),
        }

    def recommendations_for(self, item_ids: list[str]) -> list[dict]:
        """Map ids -> canonical recommendation dicts, dropping unknown ids and
        de-duplicating while preserving order (the anti-hallucination filter)."""
        out: list[dict] = []
        seen: set[str] = set()
        for item_id in item_ids:
            if item_id in seen:
                continue
            rec = self.to_recommendation(item_id)
            if rec is not None:
                out.append(rec)
                seen.add(item_id)
        return out

    # --- post-filter ----------------------------------------------------------
    def filter_valid(self, items: list) -> list:
        """Drop any proposed item whose id/url is not in the catalog.

        Accepts items as dicts or objects exposing ``id`` and/or ``url``. An item
        is kept only if its id resolves to a catalog record (and, when a url is
        present, it matches that record's canonical url)."""
        kept: list = []
        for item in items:
            item_id = _attr(item, "id")
            url = _attr(item, "url")
            rec = self._by_id.get(item_id) if item_id else None
            if rec is None:
                # No id (or unknown id): fall back to matching a known url.
                if url and url in self._urls:
                    kept.append(item)
                continue
            if url and url != rec["url"]:
                continue  # id/url disagree -> treat as hallucinated, drop
            kept.append(item)
        return kept

    # --- COMPARE grounding ----------------------------------------------------
    def record_text(self, item_id: str) -> str:
        """Full catalog text for one item, for the COMPARE grounding call."""
        rec = self._by_id.get(item_id)
        if not rec:
            return ""
        lines = [
            f"Name: {rec['name']}",
            f"Test type: {_display_test_type(rec)} "
            f"({', '.join(rec.get('keys') or [])})",
        ]
        if rec.get("duration"):
            lines.append(f"Duration: {rec['duration']}")
        if rec.get("job_levels"):
            lines.append(f"Job levels: {', '.join(rec['job_levels'])}")
        if rec.get("description"):
            lines.append(f"Description: {rec['description']}")
        return "\n".join(lines)


def _attr(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    records = json.loads(path.read_text(encoding="utf-8"))
    return Catalog(records=records)


_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    """Process-wide catalog singleton (loaded once at first use / boot)."""
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog
