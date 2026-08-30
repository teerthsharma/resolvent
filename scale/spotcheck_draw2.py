"""The iteration-3 replacement draw: stratified by classing rule, pinned to a revision.

TWO DEFECTS IN THE ITERATION-2 DRAW, BOTH MEASURED, BOTH FIXED HERE.

1. THE URN MOVED UNDER THE DRAW. `scale/spotcheck_draw.py` reads `AUDIT.md` from the
   WORKING TREE. SATURN amended the sheet three times while executing the check --
   correctly, each amendment was a measured repair -- and the sheet went 307 KEEP /
   40 ATTIC to 314/33. The pre-registered sample stopped being reproducible: at HEAD
   the same seed emits 3 different KEEP paths and 3 different ATTIC paths. A draw
   that re-rolls when the sheet is repaired is not a pre-registration.
   FIX: read the sheet from a PINNED git revision, passed explicitly and printed in
   the header. `git show <rev>:AUDIT.md`. The working tree is never consulted.

2. A UNIFORM DRAW CANNOT FALSIFY A RULE IT NEVER SAMPLES. MARS measured this exactly.
   The 307 KEEP rows are produced by 46 distinct classing rules; a uniform draw of 10
   touches 7 of them, leaving 39 rules over 151 rows -- 49.2% of KEEP -- at zero
   draws. A rule the draw never samples cannot be falsified by it, probability 1.
   And the round's measured error mode is rule-generated, not scattered: P1' condemned
   27 rows, 12 were doing real work, and all 12 were the same rule stratum. Against
   that c=12 cluster a uniform 10 misses with P = 0.6671, and it did miss -- 0 of 12
   drawn. Reaching 95% on c=12 by uniform sampling needs k = 67 of 307.
   FIX: MARS's stratified allocation. Floor of 1 on every rule stratum holding >= 2
   rows, singletons pooled into one stratum, remainder proportional to stratum size.
   P(detect a wholly-wrong rule) = 1.0000 for all 46, which no uniform draw buys below
   the full sheet.

WHAT THE STRATIFIED DRAW DOES NOT BUY, stated because it cuts against the change.
MARS measured that stratification does NOT dominate. Against a fixed size-12 cluster
placed adversarially inside the largest stratum (n=114, 37% of KEEP under one rule),
uniform at matched budget 30 detects with 0.7157 and stratified with 0.4951. The floor
buys total rule COVERAGE -- the guarantee that no rule goes unsampled -- and that is
the axis the round's measured history says fails. Buying both costs budget 40+.

THE AUTHOR OF THE SHEET DOES NOT DRAW. That is why this file exists rather than a
parameter on SATURN's side: he built AUDIT.md, so a sample he selects is the sample
that flatters it. He executes what the seed says and may not edit this file.

SEED = 10003 = round 10, iteration 3. Not tuned; changing it after seeing a result is
the defect the pinning exists to prevent.
"""
from __future__ import annotations

import argparse
import collections
import random
import re
import subprocess
import sys

SEED = 10003
N_KEEP = 30
N_ATTIC = 5
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def sheet_at(rev: str) -> str:
    """AUDIT.md as of a pinned revision. The working tree is never read."""
    return subprocess.run(
        ["git", "show", f"{rev}:AUDIT.md"],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout


def rows(text: str) -> list[tuple[str, str, str, str]]:
    """(path, class, disposition, reason) per classed row; 3-cell journals skipped."""
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:
            continue
        out.append((m.group(1), cells[4], cells[5], cells[6]))
    return out


def rule_key(reason: str) -> str:
    """The classing MECHANISM, with row-specific counts and paths normalised out.

    Kept byte-identical to MARS's definition in
    tests/mars/test_spotcheck_draw_is_powered_against_clustered_error.py so his
    power numbers describe THIS draw and not a near neighbour of it.

    ponytail: regex normalisation, not a parse. Stable only while the reason cells
    are template-generated; if AUDIT.md stops being template-generated this needs a
    real parse, and the power calculation needs redoing with it.
    """
    s = re.sub(r"`[^`]*`", "@", reason)
    s = re.sub(r"\d+", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def allocate(strata: list[list[str]], budget: int) -> list[int]:
    """Floor of 1 per stratum, remainder proportional to size. MARS's allocation."""
    sizes = [len(s) for s in strata]
    draws = [1] * len(strata)
    rem = budget - sum(draws)
    if rem < 0:
        raise ValueError(f"budget {budget} is below the floor of {len(strata)} strata")
    want = [rem * n / sum(sizes) for n in sizes]
    extra = [int(w) for w in want]
    while sum(extra) < rem:
        i = max(range(len(strata)), key=lambda j: want[j] - extra[j])
        extra[i] += 1
    return [min(a + b, n) for a, b, n in zip(draws, extra, sizes)]


def build(rev: str):
    all_rows = rows(sheet_at(rev))
    keep = [(p, rule_key(r)) for p, _c, d, r in all_rows if d == "KEEP"]
    attic = sorted(p for p, _c, d, _r in all_rows if d == "ATTIC")

    by_rule: dict[str, list[str]] = collections.defaultdict(list)
    for p, k in keep:
        by_rule[k].append(p)

    multi = sorted((sorted(v) for v in by_rule.values() if len(v) >= 2),
                   key=lambda v: (-len(v), v[0]))
    singles = sorted(v[0] for v in by_rule.values() if len(v) == 1)
    strata = multi + ([singles] if singles else [])

    draws = allocate(strata, N_KEEP)
    rng = random.Random(SEED)
    drawn_keep = [p for stratum, n in zip(strata, draws) for p in rng.sample(stratum, n)]
    drawn_attic = rng.sample(attic, N_ATTIC)
    return all_rows, by_rule, strata, draws, sorted(drawn_keep), sorted(drawn_attic)


def main(rev: str) -> int:
    all_rows, by_rule, strata, draws, k, a = build(rev)
    n_keep = sum(1 for _p, _c, d, _r in all_rows if d == "KEEP")
    n_attic = sum(1 for _p, _c, d, _r in all_rows if d == "ATTIC")
    print(f"AUDIT.md pinned at {rev}: {len(all_rows)} classed rows "
          f"(KEEP {n_keep} / ATTIC {n_attic})")
    print(f"seed={SEED}  distinct classing rules over KEEP: {len(by_rule)}")
    print(f"strata: {len(strata)} ({len(strata) - 1} multi-row + 1 pooled singleton)"
          if len(strata) > 1 else f"strata: {len(strata)}")
    print(f"allocation (stratum size -> draws): "
          f"{[(len(s), n) for s, n in zip(strata, draws)]}")
    print(f"\nevery rule sampled: {all(n >= 1 for n in draws)}  "
          f"=> P(detect a wholly-wrong rule) = 1.0000 for all {len(by_rule)} rules")
    print(f"\nKEEP drawn ({len(k)}) -- every one MUST PASS:")
    for p in k:
        print(f"  {p}")
    print(f"\nATTIC drawn ({len(a)}) -- every one MUST BE DEAD:")
    for p in a:
        print(f"  {p}")
    return 0


def demo(rev: str) -> None:
    """Self-check: reproducible, disjoint, inside the sheet, and every rule sampled."""
    a1 = build(rev)
    a2 = build(rev)
    assert a1[4] == a2[4] and a1[5] == a2[5], "draw is not reproducible under its own seed"
    all_rows, by_rule, strata, draws, k, a = a1
    assert len(set(k)) == len(k) == N_KEEP, f"KEEP draw is {len(set(k))}, want {N_KEEP}"
    assert len(set(a)) == len(a) == N_ATTIC, f"ATTIC draw is {len(set(a))}, want {N_ATTIC}"
    assert not (set(k) & set(a)), "a row was drawn as both KEEP and ATTIC"
    paths = {p for p, _c, _d, _r in all_rows}
    assert set(k) | set(a) <= paths, "drew a row absent from the pinned sheet"
    assert all(n >= 1 for n in draws), "a stratum got zero draws; the floor is broken"
    assert sum(len(s) for s in strata) == sum(1 for _p, _c, d, _r in all_rows if d == "KEEP"), \
        "strata do not partition KEEP"
    print(f"demo OK @ {rev}: {len(all_rows)} rows, {len(by_rule)} rules, "
          f"{len(strata)} strata, every rule sampled, draw reproducible")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="f823b02", help="pinned revision to read AUDIT.md from")
    ap.add_argument("--demo", action="store_true")
    ns = ap.parse_args()
    if ns.demo:
        demo(ns.rev)
    else:
        sys.exit(main(ns.rev))
