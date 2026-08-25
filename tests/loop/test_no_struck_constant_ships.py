"""No number this project has struck may be asserted by shipped code or a lead document.

THE GAP THIS CLOSES — the fourth appearance of one shape, and the first one
caught by a bind instead of by an audit.

Iteration 34 declared "six documents, one story" after a consistency script
reported every document carried the kill. Iteration 35 found that claim wrong by
two, and the reason is structural:

    the script tested for the PRESENCE of the kill string
    it never tested for the ABSENCE of struck ones

**A grep for what should be there cannot find what should not be.** Three
existing binds cover operators (iteration 18), arms (31) and theorem hypotheses
(33) — all of them presence checks. Nothing asserted absence, so struck numbers
survived in the governing prompt for eighteen iterations and in `COSTS` for
longer.

WHY THIS ONE IS ABOUT CODE FIRST. Prose gets read sceptically; a dict gets
imported. `ceq/hf/modeling_ceq.py::COSTS` is the object that travels to
HuggingFace, and it asserted `slope: -1.389` while `README.md:136` withdrew that
exponent as a `floor = 1e-6` artifact. The module contradicted its own card.

TWO LAYERS, AND THEY ARE NOT EQUALLY STRONG — stated because this project has
had four structure-by-regex instruments break or cry wolf (the LOCK slice
boundary, the LOCK-line scraper matching prose, the provenance audit missing a
line break, the ARMS-DISTINCT dispatch slice).

    layer 1  VALUE.  Imports the shipped objects and walks them. Compares
                     numbers to numbers. Cannot cry wolf. This is the bind.
    layer 2  TEXT.   Scans lead documents for struck constants quoted without a
                     strike marker. Weaker by construction, and labelled so.

Append-only history is excluded from layer 2 by design: `DONE.md` and the round
archive RECORD the strikes, so a struck number must appear there.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every constant this project has struck, why, and what replaced it.
#: A number enters here the moment an audit strikes it — that is the point.
STRUCK: dict[float, str] = {
    -1.389: (
        "M2 decay exponent. WITHDRAWN as a `floor = 1e-6` artifact (README.md:136). "
        "NO REPLACEMENT IS PUBLISHED — least squares on the floor=0 rates gives "
        "-0.958 (R^2 0.9990) while the audit that forced the correction reports "
        "-1.221 (R^2 0.9662), and those disagree. The honest repair is DELETION "
        "of the claim, never substitution of a number nobody can defend."
    ),
    0.9938: "R^2 belonging to the withdrawn -1.389; withdrawn with it.",
    -1.826: (
        "M2 slope as first reported. Contradicted by measurement at iteration 20 — "
        "the shipped operator reads -1.298. Propagated in a RUN voice at iteration 17."
    ),
    1.471448: (
        "M5 tail norm at s=128. FABRICATED — iteration 35 swept 1,800 settings "
        "(dim, seed, generator layout, rho, lam, hops) and ZERO produced it. "
        "Measured: 0.880500 at hops=2, 0.882030 at hops=4."
    ),
    1.343174: "M5 tail norm at s=512, same fabrication. Measured: 1.292741.",
    5.4944e-13: ("Karcher residual in float64. STRUCK by the health inspection at "
        "round 5 iteration 21: asserted in a [RUN] voice with NO LIVE PRODUCER. "
        "It appeared only in a code comment and in prose. It came from a "
        "throwaway float64 check whose OWN output was defective -- that script "
        "printed `nan` for both means because it omitted the norm clamp the "
        "probe has, and 5.4944e-13 was its `min` at settings nobody recorded. "
        "The probe reproduces 7.481e-09 / 8.155e-09 / 8.405e-09 at published "
        "settings, and 7.307e-13 / 7.958e-13 / 8.405e-13 at --tol 1e-15 "
        "--steps 400. 5.4944e-13 is not the mean, min or max at any k. Order of "
        "magnitude right, so stale rather than fabricated -- but a number "
        "asserted [RUN] with no producer is the 1.471448 class."),
}

#: Words that mark a mention as a citation-of-a-strike rather than an assertion.
STRIKE_MARKERS = (
    "struck", "withdrawn", "superseded", "fabricat", "not reproducible",
    "re-scoped", "rescoped", "tgate", "dead", "no longer", "corrected",
    "previously", "first printed", "disagree", "instrument #",
)

#: Documents a stranger reads as current claims.
#: TWO REPAIRS HERE, both of them coverage rather than logic.
#:
#: 1. `MODEL_CARD.md` WAS LISTED TWICE. The parametrisation deduplicates via
#:    `sorted(set(...))`, so the duplicate never doubled a check -- but the
#:    Inspector reports "9 documents" from the PARAM COUNT, which made the tuple
#:    look like it covered one more document than it did. A count that is a
#:    property of a typo is not a count.
#: 2. `D1.md` WAS NOT COVERED AT ALL. It is the contract's negative-result
#:    DELIVERABLE (`LOOP_PROMPT.md` clause 10) and it ships. A shipped document
#:    outside the struck-constant scan is exactly how `1.471448` survived
#:    eighteen iterations -- a number in an artifact nothing checked.
LEAD_DOCS = ("README.md", "MODEL_CARD.md", "PROGNOSIS.md", "D1.md",
             "CHECKLIST.md", "LOOP_PROMPT.md", "STATE.md")

#: Source that ships.
SHIPPED_SRC = ("ceq/hf/modeling_ceq.py", "ceq/hf/configuration_ceq.py",
               "ceq/diagnose.py")


def _block(lines: list[str], n: int) -> str:
    """The blank-line-delimited paragraph containing 1-indexed line `n`."""
    i = n - 1
    lo = i
    while lo > 0 and lines[lo - 1].strip():
        lo -= 1
    hi = i
    while hi + 1 < len(lines) and lines[hi + 1].strip():
        hi += 1
    return "\n".join(lines[lo:hi + 1])


def _walk(obj, path=""):
    """Yield (path, number) for every number reachable in a shipped structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


# ==========================================================================
# LAYER 1 — VALUE. The bind.
# ==========================================================================

def test_no_struck_value_is_reachable_in_the_shipped_costs_dict():
    """`COSTS` travels to HuggingFace. It may not assert a withdrawn number.

    This compares numbers to numbers on the imported object — no regex, no
    slice, nothing that can drift when a file is reformatted.
    """
    from ceq.hf.modeling_ceq import COSTS

    hits = [(p, v, STRUCK[v]) for p, v in _walk(COSTS, "COSTS") if v in STRUCK]
    assert not hits, "STRUCK CONSTANTS REACHABLE IN THE SHIPPED COSTS DICT:\n" + "\n".join(
        f"  {p} = {v}\n      {why}" for p, v, why in hits
    )


def test_the_struck_registry_would_actually_fire():
    """Calibration: a control that cannot be nonzero is not a control (#15).

    Exhibits the input that must trigger layer 1, so a green above means the
    dict is clean rather than the walker being blind.
    """
    probe = {"a": {"slope": -1.389}, "b": [1.0, {"deep": 1.471448}]}
    hits = [v for _, v in _walk(probe) if v in STRUCK]
    assert sorted(hits) == [-1.389, 1.471448], f"walker missed planted values: {hits}"


# ==========================================================================
# LAYER 2 — TEXT. Weaker by construction. Labelled.
# ==========================================================================

@pytest.mark.parametrize("rel", sorted(set(LEAD_DOCS + SHIPPED_SRC)))
def test_struck_constants_in_lead_documents_carry_a_strike_marker(rel):
    """A struck number may be NAMED as struck; it may not be ASSERTED.

    Weaker than layer 1 — it reads text. `DONE.md` and the round archive are
    excluded on purpose: append-only history is where strikes are RECORDED, so
    the numbers must appear there.
    """
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} absent")
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = []
    for n, line in enumerate(lines, 1):
        # PARAGRAPH, NOT LINE. The first version scanned one line at a time and
        # reported five false hits: STATE.md's "The 1.471448 printed here since
        # iteration" carries FABRICATED on the NEXT line, and README's
        # `s^-1.389` mentions are withdrawn one line down. That is the FIFTH
        # structure-by-regex instrument to cry wolf here — the LOCK slice
        # boundary, the LOCK-line scraper, the provenance audit (also a line
        # break), the ARMS-DISTINCT dispatch slice, and this — and it was built
        # one iteration after the class was named, which is the point: layer 2
        # reads text and text checks drift. Layer 1 is the bind.
        #
        # A ±k line window is a tuning knob. The paragraph is the unit a strike
        # is actually discussed in, so that is what gets searched.
        if any(m in _block(lines, n).lower() for m in STRIKE_MARKERS):
            continue
        for v in STRUCK:
            if f"{abs(v)}".rstrip("0").rstrip(".") in line or f"{abs(v)}" in line:
                bad.append((n, v, line.strip()[:95]))
                break
    assert not bad, f"{rel}: struck constants asserted without a strike marker:\n" + "\n".join(
        f"  line {n}: {v}\n      {txt}" for n, v, txt in bad
    )


def test_no_collection_count_is_asserted_without_a_measurement_date():
    """A test count is a MEASUREMENT. It gets a date or it does not get asserted.

    THE FIRST VERSION OF THIS TEST WAS THE SIXTH WOLF, and it was mine.
    It searched for the bare substring `809` in a document whose tables are full
    of six-decimal confidence intervals, and reported `MODEL_CARD.md:300` —
    which is `0.038097`. A substring check for a COUNT, run against decimals,
    cannot tell 809-the-number from 809-inside-0.038097. `\\b809\\b` does not
    match `0.038097` and does match `**809 tests collect**`.

    But the deeper defect is that pinning any count is wrong. README.md:937
    records this repository carrying **794, 809 and 829** at different points;
    the live measure at iteration 38 is **955**. It grows every time a test is
    added — including the tests written to catch drift. **A pinned count is
    guaranteed to go stale, and a test that fails whenever you add a test gets
    switched off.**

    So this asserts the only property that stays true: a count claimed as
    current carries the date it was measured.
    """
    import re
    pat = re.compile(r"\b(\d{3,5})\b[^\n]{0,40}?\bcollect", re.I)
    undated = []
    for rel in ("MODEL_CARD.md", "README.md"):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            if not pat.search(line):
                continue
            block = _block(lines, n)
            if not re.search(r"\b20\d\d-\d\d-\d\d\b", block):
                undated.append(f"{rel}:{n}  {line.strip()[:88]}")
    assert not undated, (
        "collection counts asserted with no measurement date:\n  "
        + "\n  ".join(undated)
        + "\n(this repository has carried 794, 809, 829 and now 955)"
    )
