"""it.27 SATURN REPAIR 1 -- `WING_ARM` derives from citations that RESOLVE AGAINST SOURCE.

MARS's STRIKE 1 (it.22, `V20_R15_IT22_MARS.md:23-79`) is five iterations old and has been
answered twice, both answers withdrawn by this office:

  it.21  re-imported the hand-typed pin            -- struck: a fifth clause, not a witness.
  it.26  moved the pin onto a six-office consensus -- WITHDRAWN BY ITS OWN AUTHOR:
         the 32 votes all descend from the manifest, so a founding mistake at it.1
         reads GREEN 32 times. Consensus defeats an editor, never a founding mistake.

This is the third answer and it is a different KIND of evidence.

    A vote cannot be wrong in a way that shows; a resolved line citation can.

MARS's own route -- a regex for a wing id sitting next to an arm name in the ledger -- is
prose ADJACENCY. That is one more vote, cast by one more office, and it too descends from
the manifest. This node consumes no office's opinion. For each wing it takes the ledger rows
filed under that wing, extracts their CODE citations, and requires each to land inside the
body of a symbol the same row names -- in source, at the cited line numbers.

  L-10 (W1) cites `ceq/arm_smprime.py:163-172` and names `path_product`. `path_product` is
       defined at `:144` and its body ends at `:174`, so the cited range is INSIDE it.
  L-13 (W1) names `ArmSMPrime.forward`; `class ArmSMPrime` is at `ceq/arm_smprime.py:497`.
  L-14 (W3) names `ArmPL.forward`;      `class ArmPL`      is at `ceq/arm_pl.py:358`.

  RED 1  `WING_ARM` is still the hand-typed literal MARS struck. It is not derived from
         anything that can fail on code.

RED 1 is measured against UNMUTATED code. The remaining nodes are the resolver's own
control and its two planted negatives -- the three-clause manifest swap, and the founding
mistake the consensus witness could not see.

Run:  python -m pytest tests/saturn/test_v20_r15_it27_wing_arm_citation.py -x -q
"""
from __future__ import annotations

import ast
import functools
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"
CEQ = ROOT / "ceq"

#: `| **L-10** | **Q4 / W1 sparsity** | ... |` -- id, then the subject cell.
ROW = re.compile(r"^\|\s*\*\*(L-\d+)\*\*\s*\|(.*)$")
#: A CODE citation: a path into an arm module and a 1-based line or range.
LINE_CITE = re.compile(r"`ceq/(arm_[a-z0-9_]+)\.py:(\d+)(?:[-–](\d+))?`")
#: A backticked symbol, bare or dotted -- `path_product`, `ArmPL.forward`.
SYM = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?`")
WING = re.compile(r"\bW(\d+)\b")


@functools.lru_cache(maxsize=None)
def arm_modules() -> tuple:
    return tuple(sorted(p.stem for p in CEQ.glob("arm_*.py")))


@functools.lru_cache(maxsize=None)
def spans(stem: str) -> dict:
    """`name` -> (first line, last line) for every def/class in `ceq/<stem>.py`.

    Methods are keyed `Class.method`, so a dotted citation resolves to a real body
    and not merely to a name that occurs somewhere in the file.
    """
    src = (CEQ / f"{stem}.py").read_text(encoding="utf-8", errors="replace")
    out = {}

    def walk(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                out[name] = (child.lineno, child.end_lineno or child.lineno)
                if isinstance(child, ast.ClassDef):
                    walk(child, name + ".")

    walk(ast.parse(src))
    return out


def span_of(stem: str, symbol: str):
    """The body span of `symbol` in `ceq/<stem>.py`, or None.

    A bare name may be a method (`zero_hop_mask` is `ArmSMPrime.zero_hop_mask`), so
    a unique dotted suffix counts. A suffix shared by two definitions does NOT --
    `forward` names a method on both arms and must resolve to neither.
    """
    tbl = spans(stem)
    if symbol in tbl:
        return tbl[symbol]
    hits = [v for k, v in tbl.items() if k.rsplit(".", 1)[-1] == symbol]
    return hits[0] if len(hits) == 1 else None


def defining_modules(symbol: str) -> list:
    """Every arm module that DEFINES `symbol`. Ambiguity is refused, not guessed."""
    return [s for s in arm_modules() if span_of(s, symbol) is not None]


def rows(text: str) -> list:
    """(id, subject cell, whole row) for every `L-n` row of the ledger."""
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            out.append((m.group(1), m.group(2).split("|")[0], line))
    return out


def resolved_arms(row: str) -> dict:
    """The arm modules this row's citations RESOLVE INTO, keyed by citation.

    A symbol citation resolves when exactly one arm module defines it. A line
    citation resolves only when the file has that many lines AND the cited range
    lies inside the body of a symbol THE SAME ROW NAMES -- which is the half that
    fails on source rather than on agreement. A path typed next to a line number
    is not evidence; a line number landing in the described code is.
    """
    syms = {}
    for head, attr in SYM.findall(row):
        name = f"{head}.{attr}" if attr else head
        mods = defining_modules(name)
        if len(mods) == 1:
            syms[name] = mods[0]

    out = {}
    for name, stem in sorted(syms.items()):
        out.setdefault(f"symbol {name}", []).append(stem)

    for stem, lo_s, hi_s in LINE_CITE.findall(row):
        lo, hi = int(lo_s), int(hi_s or lo_s)
        key = f"line ceq/{stem}.py:{lo_s}" + (f"-{hi_s}" if hi_s else "")
        path = CEQ / f"{stem}.py"
        if not path.is_file():
            out[key] = []
            continue
        n = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        anchored = [nm for nm, s in syms.items()
                    if s == stem and span_of(stem, nm)[0] <= lo <= hi <= span_of(stem, nm)[1]]
        out[key] = [stem] if (hi <= n and anchored) else []
    return out


def ledger_wing_arm(text: str = None) -> dict:
    """wing -> arm, derived from resolved citations alone. REFUSES ambiguity.

    Rows naming two wings ("Q4 / W1 + W3") are skipped: a citation in such a row
    cannot be attributed to one wing, and attributing it to both would invent the
    joint this node exists to check.
    """
    text = LEDGER.read_text(encoding="utf-8", errors="replace") if text is None else text
    per = {}
    for _lid, subject, row in rows(text):
        wings = set(WING.findall(subject))
        if len(wings) != 1:
            continue
        wing = "W" + next(iter(wings))
        for stems in resolved_arms(row).values():
            for stem in stems:
                per.setdefault(wing, set()).add(stem)
    out = {}
    for wing, stems in sorted(per.items()):
        assert len(stems) == 1, (
            f"REFUSAL: {wing} resolves into {sorted(stems)}. The ledger's citations "
            "do not agree on which arm this wing is; no binding is derivable.")
        out[wing] = next(iter(stems))
    return out


# ==========================================================================
#  Census -- the channel is checked before the fact is asked of it
# ==========================================================================

def test_the_resolver_has_something_to_resolve():
    """Non-vacuity. A resolver that finds no citations passes every later node."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    all_rows = rows(text)
    # it.32 RULE 1, MARS-31-D: a floor is blind to deletion -- a ledger that
    # loses L-9 and gains two rows is still `>= 17`. The fact this node needs is
    # that the it.27 census rows are all STILL THERE, and only a relation over
    # their NAMES can carry it. Growth is still allowed; deletion is not.
    it27_rows = frozenset(f"L-{i}" for i in range(1, 18))
    gone = tuple(sorted(it27_rows - {r[0] for r in all_rows}))
    assert gone == (), (
        f"ledger rows deleted since the it.27 census: {gone}; "
        f"read {sorted(r[0] for r in all_rows)}")
    cites = {lid: resolved_arms(row) for lid, _s, row in all_rows}
    landed = {lid: c for lid, c in cites.items() if any(c.values())}
    #: the it.32 census reading of which rows carry a resolving citation.
    lost = tuple(sorted({"L-10", "L-13", "L-14"} - set(landed)))
    assert lost == (), (
        f"a row that carried a resolving citation at the it.32 census no longer "
        f"does: {lost}; landed is now {sorted(landed)}")
    # The line-citation half must be exercised, not just the symbol half.
    line_ok = [k for c in cites.values() for k, v in c.items() if k.startswith("line") and v]
    assert line_ok, f"no LINE citation resolved; only symbols did: {sorted(landed)}"


def test_the_anchor_discriminates_between_the_two_arms():
    """THE CHANNEL CHECK, executed rather than asserted.

    The fact asked of this channel is WHICH ARM a wing is. The channel can only
    carry it if the anchoring symbols exist in one arm module and not the other.
    Three instruments of this office this round read a channel that could not
    carry the fact asked of it; this node is the check that prevents a fourth.
    """
    assert set(arm_modules()) >= {"arm_pl", "arm_smprime"}, arm_modules()
    for sym in ("path_product", "zero_hop_mask", "ArmSMPrime.forward"):
        assert defining_modules(sym) == ["arm_smprime"], (
            f"{sym} is defined in {defining_modules(sym)}; it cannot anchor W1")
    for sym in ("ArmPL", "ArmPL.forward"):
        assert defining_modules(sym) == ["arm_pl"], (
            f"{sym} is defined in {defining_modules(sym)}; it cannot anchor W3")
    # And the cited RANGE must discriminate too, not merely the file name: `:163-172`
    # is inside `path_product` in one module and inside nothing the row names in the other.
    lo, hi = span_of("arm_smprime", "path_product")
    assert lo <= 163 and 172 <= hi, f"path_product spans {lo}-{hi}; L-10 cites 163-172"
    assert [n for n in ("path_product", "zero_hop_mask") if span_of("arm_pl", n)] == [], (
        "arm_pl defines L-10's anchors too; the range cannot discriminate")


# ==========================================================================
#  RED 1 -- the pin MARS struck is still hand-typed
# ==========================================================================

def test_the_wing_pin_is_derived_from_a_resolved_citation():
    """RED 1, against unmutated code. MARS's STRIKE 1, third answer.

    The pin must not be a literal, and the value it carries must be the one the
    resolver derives. Either half alone is defeatable: a derivation nobody checks
    against the pin is decoration, and a pin nobody derives is a fifth clause.
    """
    from tests.saturn.test_v20_r15_it14_saturn import WING_ARM

    derived = ledger_wing_arm()
    src = (ROOT / "tests" / "saturn" / "test_v20_r15_it14_saturn.py").read_text(
        encoding="utf-8", errors="replace")
    node = next(a.value for a in ast.walk(ast.parse(src))
                if isinstance(a, ast.Assign)
                and any(getattr(t, "id", None) == "WING_ARM" for t in a.targets))
    literal = (isinstance(node, ast.Dict)
               and all(isinstance(k, ast.Constant) for k in node.keys)
               and all(isinstance(v, ast.Constant) for v in node.values))
    assert not literal, (
        f"WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:{node.lineno}. "
        f"MARS struck it at it.22 and it is unchanged five iterations later. The resolver "
        f"in this module derives {derived} from ledger citations that land INSIDE the "
        f"source they claim -- L-10's `ceq/arm_smprime.py:163-172` falls in `path_product`, "
        f"{span_of('arm_smprime', 'path_product')}; L-14's `ArmPL.forward` is "
        f"`ceq/arm_pl.py:{span_of('arm_pl', 'ArmPL')[0]}`. That derivation fails on CODE. The "
        f"it.26 answer to this strike -- a six-office consensus -- was withdrawn by its own "
        f"author because all 32 votes descend from the manifest, so a founding mistake at "
        f"it.1 reads GREEN 32 times. The literal was never repaired.")
    assert WING_ARM == derived, (
        f"the pin says {WING_ARM} and the citations resolve to {derived}")


# ==========================================================================
#  The two planted negatives
# ==========================================================================

def test_the_three_clause_manifest_swap_contradicts_the_derivation():
    """PLANTED NEGATIVE 1 -- V-26, the editor who moves every manifest clause.

    The derivation reads no manifest clause, so the swap cannot move it. The pin it
    feeds then contradicts the manifest-derived arm, which is the contradiction MARS
    asked for in place of a fifth edit.
    """
    derived = ledger_wing_arm()
    swapped = {"W1": derived["W3"], "W3": derived["W1"]}
    assert swapped != derived, "the two wings share an arm; the swap is invisible"
    assert sorted(w for w in derived if swapped[w] != derived[w]) == ["W1", "W3"]


def test_a_founding_swap_of_the_citation_fails_on_source_not_on_agreement():
    """PLANTED NEGATIVE 2 -- THE CASE CONSENSUS COULD NOT SEE.

    Suppose it.1 wrote the map backwards and every office wrote it backwards after it,
    the ledger office included. Then L-10 reads `Q4 / W1 ... ceq/arm_pl.py:163-172`.
    Every vote still agrees; every vote is still wrong. The citation is not, because
    `ceq/arm_pl.py:163-172` is a scalar recurrence loop and the row is describing a
    masked reverse-cumprod. The resolution fails on the bytes.
    """
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    l10 = next(row for lid, _s, row in rows(text) if lid == "L-10")
    assert "arm_smprime.py:163-172" in l10, "premise gone: L-10 no longer cites that range"

    control = resolved_arms(l10)
    assert control["line ceq/arm_smprime.py:163-172"] == ["arm_smprime"], control

    founding = l10.replace("ceq/arm_smprime.py:163-172", "ceq/arm_pl.py:163-172")
    assert founding != l10, "the planted swap did not apply; the test is vacuous"
    after = resolved_arms(founding)
    assert after.get("line ceq/arm_pl.py:163-172") == [], (
        "a founding swap of the arm module still resolves. The node is reading the "
        f"path string, not the source: {after}")
    # ...and the row is then UNDERIVABLE rather than silently re-bound to arm_pl.
    assert "arm_pl" not in [s for v in after.values() for s in v], after
