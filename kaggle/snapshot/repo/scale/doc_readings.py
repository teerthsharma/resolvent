"""The census's OWN reading extractor, identified from the sheet rather than guessed.

WHY THIS EXISTS. The iteration-2 doc arm re-measured `workdone2.md` and reported
"0 of 4 readings reproduce" against a sheet cell of 1/10, could not explain the
denominator gap, and amended the row anyway. The extractor it used counted numbers
with >= 6 SIGNIFICANT DIGITS. It was calibrated on exactly one control -- AUDIT.md's
8/23 cell for `LOOP_PROMPT_ROUND6_ARCHIVE.md` -- which it reproduced exactly, and that
agreement was a coincidence of that one document: >= 6 significant digits and the rule
below return the same 23 readings there and disagree almost everywhere else. A
single-point calibration on the largest available document is not an identification,
and this file is the correction.

WHAT THE CENSUS ACTUALLY COUNTS. A reading is a number written with >= 5 DECIMAL
PLACES, deduplicated by value. Identified by exhausting a 32-rule grid -- {significant
digits, decimal places} x {threshold 2..9} x {deduplicated, with multiplicity} --
against the 64 AUDIT.md rows that carry an `N/M readings reproduce` cell and are NOT
in the iteration-3 draw:

    dec >= 5, deduplicated   54/64 exact      <- identified
    dec >= 6, deduplicated   32/64
    dec >= 4, deduplicated   21/64
    dec >= 5, multiplicity   19/64
    sig >= 6, deduplicated    9/64            <- what iteration 2 used

THE RESIDUAL, AND WHY IT IS A TOLERANCE RATHER THAN A DEFECT. Ten of the 64 miss, and
every one is a near miss: eight by +-1, `PRIOR_ART.md` by +4, `scale/twodof.py` by -3.
None of the ten has been touched since the census revision f823b02, so the residual is
in the rule and not in the documents -- the census's extractor differs from this one in
some boundary case (most likely which characters terminate a number) that these ten
documents contain and the other 54 do not. The consequence is stated as a band, not
waved away: this instrument reproduces the census denominator EXACTLY on 84% of rows
and within +-4 on all 64, so a denominator disagreement of <= 4 is an amendment and
one of > 4 is a disagreement about method that no tolerance covers.

The NUMERATOR carries no such caveat. "Does this reading reproduce from results/*.jsonl
at abs=5e-7" is a lookup against a pool of journal leaves, and it is exact for whatever
set of readings is handed to it.

ponytail: regex over the rendered text, not a markdown parse. Numbers inside fenced
code blocks and inside tables count alike, which is what the census did.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scale.journal_scan import walk_numbers          # noqa: E402  (reused, not rebuilt)

NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)")
CELL = re.compile(r"^\|\s*`([^`]+)`\s*\|")
READS = re.compile(r"(\d+)\s*/\s*(\d+)\s+readings reproduce")

#: The identified rule. Changing either number invalidates the 54/64 identification
#: above, so they are named rather than inlined.
MIN_DECIMALS = 5
TOL = 5e-7

#: Denominator agreement measured on 64 held-out rows: exact on 54, |delta| <= 4 on all
#: 64. A gap inside the band is an amendment; a gap outside it is a method disagreement.
DENOM_BAND = 4


def decimals(s: str) -> int:
    m = re.match(r"-?\d+\.(\d+)", s)
    return len(m.group(1)) if m else 0


def readings(path: pathlib.Path) -> list[float]:
    """The census's readings for one document, deduplicated by value."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return sorted({float(s) for s in NUM.findall(text)
                   if decimals(s) >= MIN_DECIMALS})


def pool() -> set[float]:
    """Every numeric leaf in every results/*.jsonl journal."""
    out: set[float] = set()
    for j in sorted((ROOT / "results").glob("*.jsonl")):
        for line in j.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for _p, v in walk_numbers(rec):
                out.add(v)
    return out


def reproduce(path: pathlib.Path, jpool: set[float]) -> tuple[list[float], list[float]]:
    """(readings that reproduce at abs=5e-7, all readings)."""
    reads = readings(path)
    return [r for r in reads if any(abs(r - v) <= TOL for v in jpool)], reads


def sheet_cells() -> dict[str, tuple[int, int]]:
    """{path: (numerator, denominator)} for every row carrying a readings cell."""
    out = {}
    for line in (ROOT / "AUDIT.md").read_text(encoding="utf-8").splitlines():
        m = CELL.match(line)
        if not m:
            continue
        c = [x.strip() for x in line.split("|")]
        if len(c) < 7:
            continue
        r = READS.search(c[3])
        if r:
            out[m.group(1)] = (int(r.group(1)), int(r.group(2)))
    return out


def demo() -> None:
    """Pins the identification. If a future edit changes the rule, this fails."""
    cells = sheet_cells()
    assert len(cells) >= 70, f"only {len(cells)} readings cells parsed; check is vacuous"
    # The control the iteration-2 arm calibrated on -- it agrees under BOTH rules, which
    # is precisely why it identified nothing.
    assert len(readings(ROOT / "LOOP_PROMPT_ROUND6_ARCHIVE.md")) == 23

    # `workdone2.md`: the sheet's ORIGINAL cell was 1/10 and the iteration-2 arm
    # measured 4. Under the census's own rule the denominator is 10, exactly.
    assert len(readings(ROOT / "workdone2.md")) == 10, \
        f"workdone2.md denominator is {len(readings(ROOT / 'workdone2.md'))}, sheet said 10"

    exact = inband = seen = 0
    for p, (_n, d) in cells.items():
        f = ROOT / p
        if not f.exists():
            continue
        # The one cell this file exists to correct. Its 4 was written INTO the sheet by
        # the iteration-2 arm using the >= 6-significant-digit rule; scoring this rule
        # against that number would be scoring it against the error it is repairing.
        if p == "workdone2.md":
            continue
        seen += 1
        got = len(readings(f))
        exact += got == d
        inband += abs(got - d) <= DENOM_BAND
    assert seen >= 70, seen
    assert exact / seen >= 0.80, f"identification decayed: {exact}/{seen} exact"
    assert inband == seen, f"{seen - inband} rows outside the +-{DENOM_BAND} band"
    print(f"demo OK: census extractor = decimals>={MIN_DECIMALS}, deduplicated; "
          f"{exact}/{seen} sheet denominators exact, {inband}/{seen} within "
          f"+-{DENOM_BAND}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("docs", nargs="*")
    ns = ap.parse_args()
    if ns.demo:
        demo()
    else:
        jp = pool()
        cells = sheet_cells()
        for d in ns.docs:
            hits, reads = reproduce(ROOT / d, jp)
            n, m = cells.get(d, (None, None))
            print(f"{d}: {len(hits)}/{len(reads)} reproduce at abs={TOL:g}"
                  f"   sheet says {n}/{m}")
