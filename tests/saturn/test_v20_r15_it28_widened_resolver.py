"""it.28 SATURN REPAIR 3 -- widen the citation resolver past `ceq/arm_*.py`.

it.27 closed MARS's STRIKE 1 with a resolver that reads only `ceq/arm_*.py`, and
recorded its own boundary:

    "W3's binding rests on ONE symbol citation, `ArmPL.forward`. W1 has three
     independent resolutions; W3 has one. That asymmetry is real and is not
     repaired here."

One resolving citation is a pin with a single point of failure. The named route was
to widen the resolver to `lean/`, `scripts/` and `tests/` citations, which the
ledger does carry. This module widens it and MEASURES what the widening buys.

The widened rule is the it.27 rule, unchanged in kind: a citation resolves to an arm
when the cited line lands INSIDE the body of a named symbol, and that body names
exactly one arm. A path typed next to a line number is not evidence. Neither is a
line number landing in a docstring or a comment -- which is exactly what it.27 ruled
about `ceq/arm_pl.py:1`, and the same rule must apply to the widened set or the
widening is an amnesty rather than a resolver.

  RED 3  W3's binding still rests on one resolving citation after the widening.
         The two W3 candidates the widened set exposes -- L-14's
         `tests/jupiter/test_v20_r15_it9_q6.py:155` and L-17's
         `tests/jupiter/test_v20_r15_it12_constants.py:11` -- BOTH land outside
         every symbol body: the first inside a comment block recording a node
         killed at it.11, the second inside a module docstring. The single point
         of failure is now measured rather than inspected, and it is not repaired
         by a wider resolver.

Run:  python -m pytest tests/saturn/test_v20_r15_it28_widened_resolver.py -x -q
"""
from __future__ import annotations

import ast
import pathlib
import re

from tests.saturn import test_v20_r15_it27_wing_arm_citation as it27

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"

#: Any repo-relative `.py`/`.lean` citation with a 1-based line or range.
WIDE_CITE = re.compile(
    r"`((?:ceq|lean|scale|scripts|tests)/[A-Za-z0-9_./-]+\.(?:py|lean))"
    r":(\d+)(?:[-–](\d+))?`")
#: Which arm a body NAMES. Module stem or class name; nothing else counts.
ARM_NAME = {
    "arm_smprime": re.compile(r"\barm_smprime\b|\bArmSMPrime\b"),
    "arm_pl": re.compile(r"\barm_pl\b|\bArmPL\b"),
}
#: A top-level Lean declaration head.
LEAN_HEAD = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+)?"
    r"(theorem|lemma|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_.]*)")


def py_spans(path: pathlib.Path) -> dict:
    """`name` -> (lo, hi) for every def/class. Methods keyed `Class.method`."""
    out: dict = {}

    def walk(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                out[name] = (child.lineno, child.end_lineno or child.lineno)
                if isinstance(child, ast.ClassDef):
                    walk(child, name + ".")

    walk(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
    return out


def lean_spans(path: pathlib.Path) -> dict:
    """`name` -> (lo, hi) for every top-level Lean declaration.

    A declaration runs from its keyword line to the line before the next one. There
    is no Lean AST in this tree, so this is a line scanner and is stated as such:
    it can over-extend a body across trailing comments, which makes it MORE
    permissive than the Python side, not less. The measurement below survives that.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    heads = []
    for i, line in enumerate(lines):
        m = LEAN_HEAD.match(line)
        if m:
            heads.append((i + 1, m.group(2)))
    out = {}
    for k, (lo, name) in enumerate(heads):
        hi = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
        out[name] = (lo, hi)
    return out


def spans_of(path: pathlib.Path) -> dict:
    return lean_spans(path) if path.suffix == ".lean" else py_spans(path)


def wide_resolve(row: str) -> dict:
    """citation -> the arms it resolves into, over the WIDE file set.

    Resolves only when the cited range lies inside one symbol's body AND that body
    names exactly one arm. Not landing, and naming both arms, both yield `[]`.
    """
    out: dict = {}
    for rel, lo_s, hi_s in WIDE_CITE.findall(row):
        lo, hi = int(lo_s), int(hi_s or lo_s)
        key = f"{rel}:{lo_s}" + (f"-{hi_s}" if hi_s else "")
        path = ROOT / rel
        if not path.is_file():
            out[key] = []
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if hi > len(lines):
            out[key] = []
            continue
        landed = [(nm, s) for nm, s in spans_of(path).items() if s[0] <= lo <= hi <= s[1]]
        if not landed:
            out[key] = []          # docstring, comment, or module scope -- it.27's rule
            continue
        # innermost body wins, so a method beats the class that contains it
        _nm, (blo, bhi) = min(landed, key=lambda kv: kv[1][1] - kv[1][0])
        body = "\n".join(lines[blo - 1:bhi])
        arms = sorted(a for a, rx in ARM_NAME.items() if rx.search(body))
        out[key] = arms if len(arms) == 1 else []
    return out


def wide_census() -> dict:
    """wing -> {citation: arms}, over single-wing ledger rows only."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    per: dict = {}
    for _lid, subject, row in it27.rows(text):
        wings = set(it27.WING.findall(subject))
        if len(wings) != 1:
            continue
        per.setdefault("W" + next(iter(wings)), {}).update(wide_resolve(row))
    return per


def resolving_citations(wing: str) -> list:
    """Every citation that RESOLVES for `wing`: it.27's narrow set plus the wide."""
    narrow = it27.ledger_wing_arm()          # asserts non-ambiguity as a side effect
    assert wing in narrow, f"{wing} has no narrow binding: {narrow}"
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    keys = set()
    for _lid, subject, row in it27.rows(text):
        if set(it27.WING.findall(subject)) != {wing[1:]}:
            continue
        keys |= {("narrow", k) for k, v in it27.resolved_arms(row).items() if v}
        keys |= {("wide", k) for k, v in wide_resolve(row).items() if v}
    return sorted(keys)


# ==========================================================================
#  The channel check, before the fact is asked of it
# ==========================================================================

def test_the_widened_channel_reaches_files_the_narrow_one_could_not():
    """Non-vacuity. A widening that sees no new file cannot buy anyone anything,
    and the count below would be about the regex rather than about the ledger."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    seen = {k for _l, _s, row in it27.rows(text) for k in wide_resolve(row)}
    outside = sorted(k for k in seen if not k.startswith("ceq/arm_"))
    #: MARS-31-D's own prescription: a sorted path tuple, read at the it.32
    #: census. A floor of 5 was green on any deletion the widening backfilled.
    it32_outside = frozenset((
        "ceq/kdata.py:475", "lean/CEQ/V16Domain.lean:129",
        "scale/negation_scope.py:300-304", "scripts/v15_r1.py:137",
        "scripts/v15_r1.py:17-19",
        "tests/jupiter/test_v20_r15_it12_constants.py:11",
        "tests/jupiter/test_v20_r15_it12_constants.py:12",
        "tests/jupiter/test_v20_r15_it9_q6.py:155"))
    dropped = tuple(sorted(it32_outside - set(outside)))
    assert dropped == (), (
        f"the widened set lost a target it reached at the it.32 census: "
        f"{dropped}; it now reaches {outside}")
    assert any(k.startswith("lean/") for k in outside), outside
    assert any(k.startswith("tests/") for k in outside), outside
    assert any(k.startswith("scripts/") for k in outside), outside


def test_the_widened_rule_still_refuses_a_line_outside_every_body():
    """The widening must not become an amnesty.

    it.27 ruled that `ceq/arm_pl.py:1` contributes nothing because it is the module
    docstring, inside no symbol body. Applying the same rule to the widened set is
    what makes RED 3 a measurement rather than a concession.
    """
    assert wide_resolve("`ceq/arm_pl.py:1`") == {"ceq/arm_pl.py:1": []}
    for cite in ("`tests/jupiter/test_v20_r15_it9_q6.py:155`",
                 "`tests/jupiter/test_v20_r15_it12_constants.py:11`"):
        got = wide_resolve(cite)
        assert list(got.values()) == [[]], (
            f"{cite} now resolves: {got}. Re-derive RED 3 -- W3's asymmetry may be "
            "repaired, and if so this node must be rewritten, not deleted.")


def test_a_body_that_names_both_arms_resolves_to_neither():
    """PLANTED NEGATIVE: the widened set is full of files that discuss both arms.

    A test module comparing W1 to W3 names both, and a resolver that took the first
    match would bind a wing to whichever arm happened to be mentioned first. It must
    refuse instead.
    """
    src = (ROOT / "tests" / "saturn" / "test_v20_r15_it28_widened_resolver.py")
    spans = py_spans(src)
    lo, hi = spans["test_a_body_that_names_both_arms_resolves_to_neither"]
    both = f"`tests/saturn/{src.name}:{lo}-{hi}`"
    # this very body names arm_smprime and arm_pl, below, so the citation is real
    _ = ("arm_smprime", "arm_pl")
    got = wide_resolve(both)
    assert list(got.values()) == [[]], (
        f"a body naming both arms resolved to {got}; the resolver is guessing")


# ==========================================================================
#  RED 3 -- the widening does not rescue W3
# ==========================================================================

def test_w3s_pin_still_rests_on_one_resolving_citation():
    """RED 3, measured against unmutated code and an unmutated ledger.

    Filed as the finding, not argued away: widening the resolver to `lean/`,
    `scripts/` and `tests/` buys W1 additional resolutions and buys W3 none,
    because both of W3's widened candidates land outside every symbol body.
    """
    w1, w3 = resolving_citations("W1"), resolving_citations("W3")
    assert len(w3) >= 2, (
        f"W3's binding rests on {len(w3)} resolving citation(s) after the widening "
        f"to lean/, scripts/ and tests/ -- {w3}; W1 has {len(w1)} -- {w1}. L-14's "
        "`tests/jupiter/test_v20_r15_it9_q6.py:155` lands inside a comment block "
        "recording a node KILLED at it.11, and L-17's "
        "`tests/jupiter/test_v20_r15_it12_constants.py:11` lands inside a module "
        "docstring; neither is inside a symbol body, so neither resolves under the "
        "it.27 content rule. The repair is a CITATION the ledger does not yet "
        "carry -- a W3 row naming a symbol whose body names `arm_pl` -- and not a "
        "wider resolver.")


#: THE WIDENED CHANNEL'S OWN CHECK, measured 12:55Z, and it came back EMPTY.
#: Eight citations outside `ceq/arm_*.py` exist in the ledger and NONE resolves.
#: Recorded per citation with its reason so that a later citation which DOES
#: resolve fires this node instead of quietly enlarging the pin.
WIDE_CENSUS = {
    "scripts/v15_r1.py:137": "outside every symbol body",
    "scripts/v15_r1.py:17-19": "outside every symbol body",
    "lean/CEQ/V16Domain.lean:129": "in `pathProd_eq_zero_iff`; body names neither arm",
    "scale/negation_scope.py:300-304": "in `equilibrium_oracle`; body names neither arm",
    "tests/jupiter/test_v20_r15_it9_q6.py:155": "outside every symbol body (killed-node comment)",
    "tests/jupiter/test_v20_r15_it12_constants.py:12": "outside every symbol body (module docstring)",
    "tests/jupiter/test_v20_r15_it12_constants.py:11": "outside every symbol body (module docstring)",
    "ceq/kdata.py:475": "outside every symbol body",
}


def test_the_widened_channel_resolves_nothing_for_either_wing():
    """THE CHANNEL CHECK ON THE WIDENING ITSELF -- a measured NULL, filed as one.

    RED 3 says W3 still has one resolving citation. On its own that reads as a fact
    about W3. It is not: the widened channel resolves NOTHING, for EITHER wing. Six
    of the eight non-arm citations land outside every symbol body, and the two that
    land inside one (`pathProd_eq_zero_iff` in Lean, `equilibrium_oracle` in
    `scale/`) sit in bodies that name neither arm.

    So the route named at it.27 -- "widen to lean/, scripts/ and tests/" -- is a
    null instrument on the ledger AS IT STANDS, and this office says so rather than
    shipping the widening as a repair. The pin's single point of failure is
    unrepaired, and the repair is a citation the ledger does not yet carry.
    """
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    got = {k: v for _l, _s, row in it27.rows(text)
           for k, v in wide_resolve(row).items() if not k.startswith("ceq/arm_")}
    assert set(got) == set(WIDE_CENSUS), (
        f"the non-arm citation set moved: added={sorted(set(got) - set(WIDE_CENSUS))} "
        f"removed={sorted(set(WIDE_CENSUS) - set(got))}; re-derive the census")
    resolving = {k: v for k, v in got.items() if v}
    assert resolving == {}, (
        f"a widened citation now RESOLVES: {resolving}. The it.28 record says the "
        "widened channel is empty and that W3's pin therefore rests on one "
        "citation; re-derive both. This is not a failure of the widening -- it is "
        "the widening finally buying something, and the record must say so.")
    # ...and the narrow asymmetry the null leaves standing, stated as a number.
    assert (len(resolving_citations("W1")), len(resolving_citations("W3"))) == (5, 1)
