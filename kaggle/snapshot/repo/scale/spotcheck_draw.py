"""The iteration-2 spot-check draw, fixed before anyone runs it.

WHY THIS IS A FILE AND NOT A JUDGEMENT CALL. SATURN built AUDIT.md. A census
author who also picks which of his own rows get verified has chosen the sample
that flatters the sheet, and no amount of good faith makes that measurable. The
draw is therefore mechanical, seeded, and committed BEFORE the verification runs,
so the sample cannot move after anyone sees a result.

The seed is not arbitrary and is not tuned: SEED = 10002 = round 10, iteration 2.
Anyone re-running this file reproduces the identical fifteen rows.

WHAT THE DRAW IS FOR. Under H0 "the sheet is at least 95% right", a uniform draw
of 10 KEEPs has P(at least one fails) = 1 - 0.95**10 = 0.4013. So one failure
AMENDS a row and is expected; two failures reject the census at p < 0.09 and it
re-runs. The 5 ATTICs must be dead.

WHAT THIS DRAW CANNOT DO, stated here so the verdict cannot overclaim: a uniform
draw is powered against errors spread across the sheet. If the sheet's errors are
CONCENTRATED -- one directory, one class, one presumption -- a uniform 10 can miss
the cluster entirely. That probability is MARS's iteration-2 calculation and it is
not answered here. If he measures this draw underpowered, the verdict it produces
is caveated, not discarded, and iteration 3 re-draws stratified.
"""
from __future__ import annotations

import pathlib
import random
import re
import sys

SEED = 10002
N_KEEP = 10
N_ATTIC = 5
ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "AUDIT.md"
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def rows() -> list[tuple[str, str, str]]:
    """(path, class, disposition) for every classed row. Journals have 3 cells and are skipped."""
    out = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:          # the two journals, listed and unclassed by design
            continue
        out.append((m.group(1), cells[4], cells[5]))
    return out


def draw(all_rows):
    keep = sorted(p for p, _c, d in all_rows if d == "KEEP")
    attic = sorted(p for p, _c, d in all_rows if d == "ATTIC")
    rng = random.Random(SEED)
    return rng.sample(keep, N_KEEP), rng.sample(attic, N_ATTIC), len(keep), len(attic)


def main() -> int:
    all_rows = rows()
    k, a, n_keep, n_attic = draw(all_rows)
    print(f"AUDIT.md classed rows: {len(all_rows)}  (KEEP {n_keep} / ATTIC {n_attic})")
    print(f"seed={SEED}  P(>=1 KEEP fails | sheet >=95% right) = {1 - 0.95 ** N_KEEP:.4f}")
    print(f"\nKEEP drawn ({N_KEEP}) -- every one MUST PASS:")
    for p in k:
        print(f"  {p}")
    print(f"\nATTIC drawn ({N_ATTIC}) -- every one MUST BE DEAD:")
    for p in a:
        print(f"  {p}")
    return 0


def demo() -> None:
    """Self-check: the draw is reproducible, disjoint, and inside the sheet."""
    all_rows = rows()
    k1, a1, n_keep, n_attic = draw(all_rows)
    k2, a2, _, _ = draw(all_rows)
    assert k1 == k2 and a1 == a2, "draw is not reproducible under its own seed"
    assert len(set(k1)) == N_KEEP and len(set(a1)) == N_ATTIC, "draw has duplicates"
    assert not (set(k1) & set(a1)), "a row was drawn as both KEEP and ATTIC"
    paths = {p for p, _c, _d in all_rows}
    assert set(k1) | set(a1) <= paths, "drew a row that is not on the sheet"
    assert n_keep + n_attic == len(all_rows), "KEEP + ATTIC does not close on the row count"
    print(f"demo OK: {len(all_rows)} rows, draw reproducible, {n_keep} KEEP / {n_attic} ATTIC")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
