"""Build-time: encode ``data/catalog.json`` into ``data/embeddings.npy``.

Responsibilities (CLAUDE.md §4, §7):
- Load the catalog, build a text representation per assessment (via
  ``app.retrieval.assessment_text`` so it matches the BM25 documents exactly),
  and encode it with sentence-transformers ``all-MiniLM-L6-v2``.
- Save the L2-normalized dense embedding matrix to ``data/embeddings.npy`` and
  the aligned id order to ``data/embeddings_ids.json`` (both shipped in the repo)
  so nothing large downloads at cold start and ids map to embedding rows.

Build-time only — NOT a runtime dependency of the API.

    python scripts/build_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import (  # noqa: E402  (after sys.path insert)
    EMBEDDING_MODEL,
    assessment_text,
)

DEFAULT_CATALOG = ROOT / "data" / "catalog.json"
DEFAULT_EMB = ROOT / "data" / "embeddings.npy"
DEFAULT_IDS = ROOT / "data" / "embeddings_ids.json"


def build(catalog_path: Path, emb_path: Path, ids_path: Path) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ids = [r["id"] for r in catalog]
    texts = [assessment_text(r) for r in catalog]
    print(f"Loaded {len(catalog)} assessments from {catalog_path.name}")

    from sentence_transformers import SentenceTransformer

    print(f"Encoding with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors → dot product == cosine
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")

    print(f"Saved {embeddings.shape[0]}x{embeddings.shape[1]} matrix -> {emb_path}")
    print(f"Saved {len(ids)} ids -> {ids_path}")
    # Sanity: rows should be unit-norm.
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Row-norm min/max: {norms.min():.4f} / {norms.max():.4f} (expect ~1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--out", type=Path, default=DEFAULT_EMB)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    args = ap.parse_args()
    build(args.catalog, args.out, args.ids)


if __name__ == "__main__":
    main()
