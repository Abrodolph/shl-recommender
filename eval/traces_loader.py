"""Load SHL's public conversation traces into a normalized structure.

Each trace in ``eval/traces/*.md`` is a multi-turn recruiter<->agent conversation
(markdown). This module parses them into a stable shape the replay harness and the
analysis can consume:

    Trace
      id                 e.g. "C1"
      turns[]            per-turn {turn, user, agent_text, recommendations,
                                   has_recs, end_of_conversation}
      opener             first user message (the vague/greeting intent)
      user_messages[]    every user message = the persona/facts revealed
      expected_shortlist final labeled shortlist (recs at end_of_conversation=true,
                         else the last recommendations seen)

The traces have no explicit "persona" block, so persona/facts are DERIVED from the
user turns (the opener plus every constraint the user later reveals). The labeled
expected shortlist is the final recommendations table.

Recommendation item shape (mirrors the API contract): {name, url, test_type}. We
also keep ``test_type_letters`` (parsed set) since the traces render multi-key
items as comma-joined codes (e.g. "P,C", "K,S", "C, K").

Run directly to print a per-trace summary (persona, #expected, test_type mix) and
a groundedness check against ``data/catalog.json``:

    python eval/traces_loader.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = ROOT / "eval" / "traces"
CATALOG_PATH = ROOT / "data" / "catalog.json"

# A "### Turn N" header starts each turn block.
TURN_RE = re.compile(r"^###\s+Turn\s+(\d+)\s*$", re.MULTILINE)
END_TRUE_RE = re.compile(r"end_of_conversation`?:\s*\*\*true\*\*", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")  # the |---|---| divider


@dataclass
class Recommendation:
    name: str
    url: str
    test_type: str                      # raw as rendered, e.g. "K" or "P,C"
    test_type_letters: list[str] = field(default_factory=list)

    def as_contract(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


@dataclass
class Turn:
    turn: int
    user: str
    agent_text: str
    recommendations: list[Recommendation]
    has_recs: bool
    end_of_conversation: bool


@dataclass
class Trace:
    id: str
    path: Path
    turns: list[Turn]

    @property
    def opener(self) -> str:
        return self.turns[0].user if self.turns else ""

    @property
    def user_messages(self) -> list[str]:
        return [t.user for t in self.turns if t.user]

    @property
    def expected_shortlist(self) -> list[Recommendation]:
        """Final labeled shortlist: recs at the end_of_conversation turn, else the
        last non-empty recommendations seen."""
        for t in self.turns:
            if t.end_of_conversation and t.recommendations:
                return t.recommendations
        last: list[Recommendation] = []
        for t in self.turns:
            if t.recommendations:
                last = t.recommendations
        return last

    @property
    def commit_turn(self) -> int | None:
        """1-indexed position of the first turn that emits a shortlist."""
        for i, t in enumerate(self.turns, start=1):
            if t.has_recs:
                return i
        return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _parse_user(block: str) -> str:
    """User message = the blockquoted (`> `) lines after **User**."""
    m = re.search(r"\*\*User\*\*(.*?)(?:\*\*Agent\*\*|$)", block, re.DOTALL)
    if not m:
        return ""
    quoted = [re.sub(r"^\s*>\s?", "", ln) for ln in m.group(1).splitlines()
              if ln.lstrip().startswith(">")]
    return _clean(" ".join(quoted))


def _split_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _parse_letters(test_type: str) -> list[str]:
    return [p.strip().upper() for p in re.split(r"[,\s/]+", test_type)
            if p.strip() and re.fullmatch(r"[A-Za-z]", p.strip())]


def _parse_table(agent_block: str) -> list[Recommendation]:
    """Parse the markdown recommendation table, if present, into Recommendations."""
    lines = agent_block.splitlines()
    table_lines = [ln for ln in lines if TABLE_ROW_RE.match(ln)]
    if len(table_lines) < 2:
        return []

    header = _split_cells(table_lines[0])
    # Map required columns by header name (case-insensitive, fuzzy).
    def col(*names: str) -> int | None:
        for i, h in enumerate(header):
            hl = h.lower()
            if any(n in hl for n in names):
                return i
        return None

    i_name, i_type, i_url = col("name"), col("test type", "type"), col("url", "link")
    if i_name is None or i_url is None:
        return []

    recs: list[Recommendation] = []
    for ln in table_lines[1:]:
        if SEPARATOR_RE.match(ln):
            continue
        cells = _split_cells(ln)
        if len(cells) <= max(i_name, i_url):
            continue
        name = _clean(cells[i_name])
        url = cells[i_url].strip().strip("<>").strip()
        # Strip markdown link syntax if present: [text](url) or <url>.
        m = re.search(r"https?://\S+", url)
        url = m.group(0).rstrip(">) ") if m else url
        test_type = _clean(cells[i_type]) if i_type is not None and i_type < len(cells) else ""
        if not name or not url:
            continue
        recs.append(Recommendation(name=name, url=url, test_type=test_type,
                                   test_type_letters=_parse_letters(test_type)))
    return recs


def parse_trace(path: Path) -> Trace:
    text = path.read_text(encoding="utf-8")
    matches = list(TURN_RE.finditer(text))
    turns: list[Turn] = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        turn_no = int(m.group(1))

        user = _parse_user(block)
        agent_m = re.search(r"\*\*Agent\*\*(.*)$", block, re.DOTALL)
        agent_block = agent_m.group(1) if agent_m else ""
        # Agent prose = agent block minus table lines and metadata footers.
        prose_lines = [ln for ln in agent_block.splitlines()
                       if not TABLE_ROW_RE.match(ln)
                       and "recommendations" not in ln.lower()
                       and "end_of_conversation" not in ln.lower()]
        agent_text = _clean(" ".join(prose_lines))

        recs = _parse_table(agent_block)
        turns.append(Turn(
            turn=turn_no,
            user=user,
            agent_text=agent_text,
            recommendations=recs,
            has_recs=bool(recs),
            end_of_conversation=bool(END_TRUE_RE.search(block)),
        ))
    return Trace(id=path.stem, path=path, turns=turns)


def load_traces(traces_dir: Path = TRACES_DIR) -> list[Trace]:
    def sort_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 0, p.stem)
    paths = sorted(traces_dir.glob("*.md"), key=sort_key)
    return [parse_trace(p) for p in paths]


# --- summary / groundedness ---------------------------------------------------
def _load_catalog_urls() -> set[str]:
    if not CATALOG_PATH.exists():
        return set()
    recs = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {r["url"].rstrip("/").lower() for r in recs}


def _test_type_mix(recs: list[Recommendation]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for r in recs:
        for letter in (r.test_type_letters or ["?"]):
            mix[letter] = mix.get(letter, 0) + 1
    return dict(sorted(mix.items()))


def print_summary(traces: list[Trace]) -> None:
    catalog_urls = _load_catalog_urls()
    line = "=" * 74
    print(line)
    print("SHL TRACE SUMMARY")
    print(line)

    sizes, all_missing = [], []
    for tr in traces:
        shortlist = tr.expected_shortlist
        sizes.append(len(shortlist))
        mix = _test_type_mix(shortlist)
        n_turns = len(tr.turns)

        print(f"\n[{tr.id}]  turns={n_turns}  commit_turn={tr.commit_turn}")
        print(f"  persona   : {tr.opener[:110]}")
        print(f"  facts     : {len(tr.user_messages)} user message(s)")
        print(f"  expected  : {len(shortlist)} assessment(s)")
        print(f"  type mix  : {mix if mix else '{}'}")

        if catalog_urls:
            missing = [r.name for r in shortlist
                       if r.url.rstrip('/').lower() not in catalog_urls]
            if missing:
                all_missing.append((tr.id, missing))
                print(f"  NOT IN CATALOG: {missing}")

    print("\n" + line)
    print("AGGREGATE")
    print(line)
    if sizes:
        print(f"  shortlist sizes    : {sizes}")
        print(f"  min/median/max     : {min(sizes)} / "
              f"{sorted(sizes)[len(sizes)//2]} / {max(sizes)}")
        print(f"  mean               : {sum(sizes)/len(sizes):.1f}")
    global_mix: dict[str, int] = {}
    for tr in traces:
        for letter, c in _test_type_mix(tr.expected_shortlist).items():
            global_mix[letter] = global_mix.get(letter, 0) + c
    print(f"  global type mix    : {dict(sorted(global_mix.items()))}")
    if catalog_urls:
        total = sum(len(tr.expected_shortlist) for tr in traces)
        missing_ct = sum(len(m) for _, m in all_missing)
        print(f"  groundedness       : {total - missing_ct}/{total} expected items "
              f"found in catalog.json")
        if all_missing:
            print(f"  UNGROUNDED (recall ceiling < 1.0): {all_missing}")
    else:
        print("  groundedness       : catalog.json not found (skipped)")
    print(line)


if __name__ == "__main__":
    print_summary(load_traces())
