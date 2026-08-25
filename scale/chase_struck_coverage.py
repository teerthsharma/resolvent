"""CHASE — what the struck-constant absence check does NOT look at.

WHAT IT LOOKS AT. `tests/loop/test_no_struck_constant_ships.py`:

    LEAD_DOCS  = ("README.md", "MODEL_CARD.md", "MODEL_CARD.md", "PROGNOSIS.md",
                  "CHECKLIST.md", "LOOP_PROMPT.md", "STATE.md")
    SHIPPED_SRC = ("ceq/hf/modeling_ceq.py", "ceq/hf/configuration_ceq.py",
                   "ceq/diagnose.py")

`MODEL_CARD.md` is listed TWICE and the parametrize is over `sorted(set(...))`,
so the scan runs over SIX distinct documents and THREE source files — nine
params. `inspector.py:416` labels the check "struck-constant absence (9 documents
+ shipped code)". Nine is the param count, not the document count.

Layer 1 — the part the test itself calls "the bind", the only part that compares
numbers to numbers — walks exactly ONE object: `ceq.hf.modeling_ceq.COSTS`.
Everything else in this repository is covered by layer 2, which the file's own
docstring labels "weaker by construction".

WHAT THIS FILE DOES. It applies layer 2's OWN logic — the same `STRUCK` registry,
the same `STRIKE_MARKERS`, the same paragraph-unit `_block` rule, all imported
from the test rather than reimplemented — to every markdown and python file in
the repository that the test does not scan. Append-only history (`DONE.md`,
`DONE_ARCHIVE_ROUND1.md`, the `LOOP_PROMPT_ROUND*_ARCHIVE.md` files, `done3.md`,
`done4.md`, `workdone2.md`) is excluded for the same reason the test excludes it:
a strike must be RECORDED somewhere, and that is where.

MUST-FIRE CONTROL: a synthetic document asserting `1.471448` with no strike
marker in its paragraph must be caught, and the same text WITH a marker must not.
Both are printed. A scanner that cannot see a planted hit is not a scanner.

Exit 1 if any struck constant is asserted without a strike marker in a file the
shipped check does not cover. `python scale/chase_struck_coverage.py`
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "_struck", ROOT / "tests" / "loop" / "test_no_struck_constant_ships.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
STRUCK, MARKERS, _block = _m.STRUCK, _m.STRIKE_MARKERS, _m._block
COVERED = set(_m.LEAD_DOCS) | set(_m.SHIPPED_SRC)

#: append-only history: a strike must appear here, exactly as the shipped test says
HISTORY = {"DONE.md", "DONE_ARCHIVE_ROUND1.md", "done3.md", "done4.md",
           "workdone2.md", "LOOP_PROMPT_ROUND2_ARCHIVE.md",
           "LOOP_PROMPT_ROUND3_ARCHIVE.md", "LOOP_PROMPT_ROUND4_ARCHIVE.md"}


def scan_text(text: str, rel: str):
    """Layer 2's rule, verbatim in behaviour: paragraph-unit marker, then match."""
    lines = text.splitlines()
    hits = []
    for n, line in enumerate(lines, 1):
        if any(m in _block(lines, n).lower() for m in MARKERS):
            continue
        for v in STRUCK:
            if f"{abs(v)}".rstrip("0").rstrip(".") in line or f"{abs(v)}" in line:
                hits.append((rel, n, v, line.strip()[:100]))
                break
    return hits


def control() -> bool:
    print("=== MUST-FIRE CONTROL ===")
    bad = scan_text("A paragraph that just says the tail norm is 1.471448 flat out.",
                    "<planted>")
    good = scan_text("The 1.471448 previously printed here was struck as fabricated.",
                     "<planted-with-marker>")
    f1, f2 = len(bad) == 1, len(good) == 0
    print(f"  planted assertion, no marker   -> {len(bad)} hit(s)  "
          f"{'FIRED' if f1 else 'DID NOT FIRE'}")
    print(f"  same number WITH strike marker -> {len(good)} hit(s)  "
          f"{'FIRED (correctly silent)' if f2 else 'DID NOT FIRE (false positive)'}")
    print()
    return f1 and f2


def main() -> int:
    if not control():
        print("CONTROL DID NOT FIRE -> scanner proves nothing. STOP.")
        return 1

    print("=== REGISTRY ===")
    for v, why in STRUCK.items():
        print(f"  {v:<12} {why.split('.')[0]}.")
    print(f"\n  {len(STRUCK)} struck constants.")
    print(f"  layer 1 (numbers vs numbers, 'the bind') walks: "
          f"ceq.hf.modeling_ceq.COSTS -- ONE object.")
    print(f"  layer 2 (text) covers {len(set(_m.LEAD_DOCS))} distinct documents "
          f"+ {len(_m.SHIPPED_SRC)} source files "
          f"= {len(COVERED)} paths ({len(_m.LEAD_DOCS) + len(_m.SHIPPED_SRC)} "
          f"params, MODEL_CARD.md listed twice).")
    print(f"  covered: {sorted(COVERED)}\n")

    targets = []
    for p in sorted(ROOT.rglob("*.md")) + sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if any(part in (".git", "__pycache__", ".pytest_cache", ".claude",
                        ".benchmarks") for part in p.parts):
            continue
        if rel in COVERED or p.name in HISTORY:
            continue
        targets.append((p, rel))

    print(f"=== SCANNING {len(targets)} PATHS THE SHIPPED CHECK DOES NOT COVER ===")
    hits = []
    for p, rel in targets:
        try:
            hits += scan_text(p.read_text(encoding="utf-8"), rel)
        except (UnicodeDecodeError, OSError) as e:
            print(f"  UNREADABLE {rel}: {e}")
    print(f"  scanned. uncovered .md: "
          f"{sum(1 for _, r in targets if r.endswith('.md'))}, "
          f"uncovered .py: {sum(1 for _, r in targets if r.endswith('.py'))}\n")

    notable = [r for _, r in targets
               if r in ("M2_TRAINED_PREREGISTERED_READING.md",
                        "M2_PREREGISTERED_READING.md",
                        "M2PRIME_PREREGISTERED_READING.md", "ceq/bench.py",
                        "CONTRACT.md", "TRAINING.md", "RESEARCH.md")]
    print("  UNCOVERED PATHS THAT ARE LOAD-BEARING FOR A PUBLISHED NUMBER:")
    for r in notable:
        print(f"    {r}")
    print("    (inspector.py:340-372 PARSES M2_TRAINED_PREREGISTERED_READING.md to")
    print("     re-derive the published M2 slope. A document that is an INPUT to the")
    print("     published-number check sits outside the struck-constant scan.)\n")

    if hits:
        print(f"=== {len(hits)} STRUCK CONSTANT(S) ASSERTED WITHOUT A STRIKE MARKER ===")
        print("  LAYER 2 IS A TEXT SCAN AND TEXT SCANS CRY WOLF -- the shipped test's")
        print("  own docstring counts SIX instruments that did. Every hit below is a")
        print("  CANDIDATE requiring the line to be read, not a finding.")
        by_file: dict[str, int] = {}
        for rel, n, v, txt in hits:
            by_file[rel] = by_file.get(rel, 0) + 1
            # stdout here is cp1252 on this box; the repo's prose carries U+2212.
            safe = txt.encode("ascii", "backslashreplace").decode("ascii")
            print(f"  {rel}:{n}  {v}\n      {safe}")
        print("\n  by file: " + ", ".join(f"{k}={v}" for k, v in sorted(by_file.items())))
        return 1
    print("=== NO STRUCK CONSTANT IS ASSERTED IN ANY UNCOVERED PATH ===")
    print("The gap is LATENT, not live. Nothing to retract today; the scan simply")
    print("would not see it if it happened tomorrow in any of these paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
