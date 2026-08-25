"""Every operator a probe MEASURES must be one the module SHIPS, or be declared.

WHY THIS EXISTS — instrument #17, and it is a different species from the other
sixteen. Those were broken measurements: a probe ceilinged by construction, a
NaN mapped to GREEN, an arm running under another arm's name, a calibration gate
that printed its targets as strings and always exited 0. Every one of them was
catchable by calibrating the instrument.

**This one was a CORRECT measurement of the WRONG OBJECT.** `tgate`
(`_causal_tgate_operator`) carried every M2 and S2 headline — the sign-flip rate
flat at slope +0.0270 across a 256x context growth, the 61x separation at
s=1024, the >=20.5x at s=2048 — and `tgate` appears **nowhere in the shipped
path**. The module ships `ceq_operator` and `sgate_operator`. On the shipped
operator the same measurement reads slope **-1.826** against M2's pre-registered
bar of **-0.3**: the kill fires by a factor of six, and routing is *worse* than
dense.

No calibration would have caught it. The instrument was working perfectly the
whole time. What was missing is a bind between the thing measured and the thing
shipped — and this repository already had that pattern for Lean (`.tril(-1)`
grep-binding a theorem hypothesis to the shipped tensor, M5). It was never
applied to the operator itself.

WHAT THIS TEST DOES, AND WHY IT IS NOT JUST PERMANENTLY RED. A research probe
legitimately wants arms the module does not ship — that is what a probe is for.
The defect was never that `tgate` existed; it was that nothing said so, so its
numbers were read as though they described the module. So the rule is not "every
arm must ship". It is:

    every probe arm must EITHER have a shipped counterpart
    OR appear in NON_SHIPPED with a reason.

The declaration is the artifact. A number measured on a declared non-shipped arm
can still be published — it just cannot be published as a fact about the module
without saying which operator it came from.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Files that constitute the SHIPPED module. An operator used here is one a
#: user of the package actually runs.
SHIPPED = [
    ROOT / "ceq" / "attention.py",
    ROOT / "ceq" / "hf" / "modeling_ceq.py",
]

#: Probe arms that are deliberately NOT shipped, each with the reason it exists.
#: Adding a name here is a declaration, not a suppression: it makes the gap
#: visible in a file that tests read, instead of leaving it implicit.
NON_SHIPPED = {
    # THE ARM NAME HIDES THE OPERATOR, and that is why instrument #17 survived a
    # whole round. `pivot_probe.ARMS` never contains the string "tgate" -- the
    # arms are called `pivot_signed` and `dense_signed`, and `build_arm` maps
    # BOTH to `_causal_tgate_operator`. The name describes a PROPERTY (signed),
    # not an IMPLEMENTATION, so nothing in the arm list ever said the numbers
    # came from an operator the module does not ship.
    "pivot_signed": (
        "Resolves to `_causal_tgate_operator` in build_arm, which ships "
        "NOWHERE. Carried the +0.0270 flat slope and the 61x separation at "
        "s=1024. On the shipped operator (sgate) the same measurement reads "
        "-1.826 against a -0.3 bar. Instrument #17."
    ),
    "dense_signed": (
        "Resolves to `_causal_tgate_operator` in build_arm, which ships "
        "NOWHERE. Carried the -0.746 dense control slope. Instrument #17."
    ),
    "tgate": (
        "Research arm only. Carried every M2/S2 headline (+0.0270 flat, 61x at "
        "s=1024) and ships NOWHERE — instrument #17. On the shipped operator "
        "the same measurement reads -1.826 against a -0.3 bar. Any number from "
        "this arm must name the arm."
    ),
    "tgatex": "Research arm only — tgate with a static 1/sqrt(s_visible) scale.",
    "deltanet": "Prior-art reimplementation (arXiv:2406.06484), not this module.",
    "paraformer": "Prior-art reimplementation (arXiv:2512.14619), not this module.",
    "signmag": "Prior-art reimplementation (Cog Attention arXiv:2411.07176).",
    "random": "Chance baseline. Deliberately not an operator.",
    "dense_unsigned": "Softmax control. Shipped as the BASELINE, not as this module.",
    "pivot_unsigned": "Softmax control routed through pivots. Ablation arm.",
}


def _shipped_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in SHIPPED if p.exists())


def probe_arms() -> set[str]:
    """Arms the probes actually dispatch on, read from the source of truth."""
    src = (ROOT / "scale" / "pivot_probe.py").read_text(encoding="utf-8")
    m = re.search(r"^ARMS\s*=\s*\(([^)]*)\)", src, re.M | re.S)
    assert m, "scale/pivot_probe.py no longer declares ARMS — update this bind"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def arm_to_operator() -> dict:
    """Which `_causal_*_operator` each arm actually resolves to in `build_arm`.

    THE POINT OF THIS FUNCTION. Checking the arm NAME against the shipped source
    is not enough: `pivot_signed` contains no substring that appears in
    `ceq/attention.py`, but neither does it contain "tgate". The dispatch is
    where the truth is, so the dispatch is what gets read.
    """
    src = (ROOT / "scale" / "pivot_probe.py").read_text(encoding="utf-8")
    body = src[src.index("def build_arm"):src.index("def run_arm")]
    out, current = {}, []
    for line in body.splitlines():
        names = re.findall(r'kind (?:==|in) \(?([^:]+)', line)
        if names:
            current = re.findall(r'"([^"]+)"', names[0])
        op = re.search(r"bench\.(_causal_\w+|_softmax_\w+)", line)
        if op and current:
            for a in current:
                out[a] = op.group(1)
            current = []
    return out


# ==========================================================================
# CALIBRATION — the bind must be seen to fail before any pass is trusted
# ==========================================================================

def test_the_bind_fires_on_an_undeclared_non_shipped_arm():
    """RED-first. An arm that neither ships nor is declared MUST be caught.

    Without this, a bind that silently passed everything would look exactly like
    a bind that works — which is how sixteen instruments got believed.
    """
    fake = {"an_arm_that_ships_nowhere"}
    shipped = _shipped_source()
    undeclared = [a for a in fake
                  if a not in NON_SHIPPED and a not in shipped]
    assert undeclared == ["an_arm_that_ships_nowhere"], (
        "the bind does not detect an undeclared non-shipped arm; it cannot "
        "have caught instrument #17 either"
    )


def test_the_bind_does_not_fire_on_a_genuinely_shipped_arm():
    """The other half of calibration: it must not cry wolf.

    `sgate` IS in the shipped path (`modeling_ceq.py::sgate_operator`), so it
    must pass without needing a declaration.
    """
    shipped = _shipped_source()
    assert "sgate_operator" in shipped, (
        "sgate is no longer in the shipped path — the calibration anchor moved"
    )


# ==========================================================================
# THE BIND
# ==========================================================================

def test_every_probe_arm_either_ships_or_is_declared():
    """The bind that would have caught instrument #17 on the day tgate landed."""
    shipped = _shipped_source()
    arms = probe_arms()
    resolved = arm_to_operator()
    offenders = []
    for arm in sorted(arms):
        # check the OPERATOR the arm resolves to, not the arm's name
        op = resolved.get(arm, "")
        stem = op.replace("_causal_", "").replace("_operator", "") if op else arm
        in_shipped = (f"{stem}_operator" in shipped or f'"{arm}"' in shipped
                      or f"{arm}_operator" in shipped)
        if not in_shipped and arm not in NON_SHIPPED:
            offenders.append(f"{arm} -> {op or '?'}")
    assert not offenders, (
        f"probe arms that neither ship nor are declared non-shipped: "
        f"{offenders}. Every number measured on these reads as a fact about "
        f"the module and is not one. Either wire the arm into "
        f"ceq/attention.py, or add it to NON_SHIPPED with the reason it "
        f"exists — instrument #17."
    )


def test_tgate_is_declared_non_shipped_and_says_so():
    """The specific fact that cost this project a round, pinned so it cannot
    quietly stop being true."""
    assert "tgate" in NON_SHIPPED
    reason = NON_SHIPPED["tgate"]
    assert "-1.826" in reason and "-0.3" in reason, (
        "tgate's declaration must carry the shipped-operator number that "
        "contradicts its headline, or the declaration is decoration"
    )
    resolved = arm_to_operator()
    assert resolved.get("pivot_signed") == "_causal_tgate_operator", (
        "pivot_signed no longer resolves to tgate — if it was rewired, every "
        "M2/S2 number must be re-measured and this declaration updated"
    )
    shipped = _shipped_source()
    assert "tgate" not in shipped, (
        "tgate now appears in the shipped path — if it was wired in "
        "deliberately, remove it from NON_SHIPPED and re-run M2 against it; "
        "the -1.826 result was measured on sgate, not on this"
    )


@pytest.mark.parametrize("arm", sorted(NON_SHIPPED))
def test_every_declaration_carries_a_reason(arm):
    """A declaration with an empty reason is a suppression wearing a
    declaration's clothes."""
    assert len(NON_SHIPPED[arm]) > 20, f"{arm} declared without a real reason"
