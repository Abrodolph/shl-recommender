"""Smoke test the configured LLM provider end-to-end.

Sends one prompt through the factory-selected client and prints the response so
you can verify your key / model work before wiring the agent.

    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --json "List two SHL test types as JSON"

Reads provider/model/key from env (see .env / app.config). Exits non-zero on
failure with a readable message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import LLMError, get_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "prompt",
        nargs="?",
        default="Say 'ready' if you can read this.",
        help="User prompt to send.",
    )
    ap.add_argument("--json", action="store_true", help="Request JSON-mode output.")
    args = ap.parse_args()

    s = get_settings()
    print(f"provider={s.llm_provider}  model={s.llm_model}  key_set={s.has_llm_key}")
    if not s.has_llm_key:
        print("No API key configured. Set GROQ_API_KEY (or GEMINI_API_KEY) in .env.")
        return 2

    system = (
        "You are a JSON API. Reply with a JSON object."
        if args.json
        else "You are a terse assistant."
    )
    try:
        client = get_client()
        out = client.complete(
            system=system,
            messages=[{"role": "user", "content": args.prompt}],
            json_mode=args.json,
        )
    except LLMError as exc:
        print(f"LLM call failed: {exc}")
        return 1

    print("-" * 60)
    print(out)
    print("-" * 60)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
