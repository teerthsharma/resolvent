"""MARS R10 P0 i1. The census presumption P1 misclassifies a live control.

P1, AS PRE-REGISTERED FOR THE ROUND-10 CENSUS: "a control that constructs its
own input = VACUOUS-BY-SCOPE until shown otherwise."

P1 is a distillation of five real deaths in this repository -- V-2 (hand-built
2x2 where greedy and optimal coincide), V-5 (slice-to-nothing), V-7
(`journal_scan` reading the wrong nesting level), V-10 (0.1 <= 1.0) and above
all V-14, where `scale/chase_struck_coverage.py:107-110` planted its positive by
calling `scan_text` on a LITERAL STRING, so the plant never traversed
`collect_targets` and 366 candidate files became 0 unnoticed.

WHAT V-14'S OWN RULE ACTUALLY SAYS, and it is narrower than P1: "A planted
positive must enter at the instrument's front door and traverse every stage the
real input traverses." The discriminating property is the PATH the constructed
input takes, not the fact that the control constructed it. Every planted
positive is constructed -- that is what planting means. A rule keyed on "did the
control build its own input" fires on the cure as readily as on the disease.

THE FALSIFIER. P1's operative claim, restated so it can be measured: a control
that constructs its own input has no rejection region on production-domain
inputs. One counterexample retires that claim. This file measures the rejection
region of `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` -- which
constructs every tensor it asserts on, through `M3_TASKS` -- by restoring the
V-1 defect IN PROCESS and requiring the control's own predicate to flip.

The mutation is not a mock. The V-1 defect was that both registry keys bound the
same function object, so `impact_hetero` returned byte-identically what `impact`
returned. Drawing `M3_TASKS["impact"][0]` twice at one seed reproduces that
registry's output exactly, which is why no file has to be edited to measure it.

RED IS THE POINT. Against P1 as stated this test fails, and that failure is the
attack. It goes green when the census records the measured class for this row
rather than the presumed one, so it is also the gate on the amendment: leave it
in and a later sheet that re-presumes VACUOUS-BY-SCOPE here fails CI.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import impact as IMP                               # noqa: E402
from scale import negation_scope as NS                        # noqa: E402

TARGET = "tests/cameron/test_impact_hetero_is_not_its_own_baseline.py"
AUDIT = ROOT / "AUDIT.md"

#: P1's verdict for TARGET, pre-registered before SATURN's sheet landed. The
#: file draws its tensors by calling the registry builder, so P1 fires.
P1_PRESUMED_CLASS = "vacuous"

S_NODES = IMP.IMPACT_MIN_NODES      # 1024, the admission floor, read not copied
N_DRAWS = 4
D_HOUSE = 24
SEED = 0


def _p1_fires(path: pathlib.Path) -> bool:
    """P1's antecedent, applied mechanically: does the control call production
    code to build the object it then asserts on?"""
    src = path.read_text(encoding="utf-8")
    return "M3_TASKS" in src and re.search(r"batch_fn\(", src) is not None


def _sheet_class(path: str) -> str | None:
    """The class AUDIT.md records for `path`, if the sheet exists yet. The row
    is `path | ... | class | ...`; None means the census has not landed."""
    if not AUDIT.exists():
        return None
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        if path in line and "|" in line:
            cells = [c.strip().lower() for c in line.split("|")]
            for c in cells:
                if c in ("live", "vacuous", "orphan", "superseded", "struck"):
                    return c
    return None


@pytest.fixture(scope="module")
def readings():
    """Three draws at the admission floor, one seed. `homo` and `homo_again`
    are the two draws the V-1 registry produced for its two keys."""
    homo_fn = NS.M3_TASKS["impact"][0]
    het_fn = NS.M3_TASKS["impact_hetero"][0]
    kw = dict(d_model=IMP.D_MODEL, seed=SEED)
    x_h, y_h, _f, _p = homo_fn(N_DRAWS, S_NODES, D_HOUSE, **kw)
    x_h2, y_h2, _f, _p = homo_fn(N_DRAWS, S_NODES, D_HOUSE, **kw)
    x_t, y_t, _f, _p = het_fn(N_DRAWS, S_NODES, D_HOUSE, **kw)
    return dict(repaired=(x_h, y_h, x_t, y_t), v1_defect=(x_h, y_h, x_h2, y_h2))


def _control_passes(x_a, y_a, x_b, y_b) -> bool:
    """The predicate TARGET asserts: the two registered keys draw different
    tensors and disagree on every drawn label."""
    return (not torch.equal(x_a, x_b)) and int((y_a != y_b).sum()) == N_DRAWS


def test_the_target_control_constructs_its_own_input(readings):
    """P1's antecedent holds for TARGET -- the attack is not against a strawman."""
    assert _p1_fires(ROOT / TARGET), f"{TARGET} does not construct its own input"


def test_the_constructed_input_has_a_rejection_region_on_a_production_mutation(readings):
    """The measurement. Both readings printed, as V-9 requires of any claim that
    a repair moves the object it repairs."""
    passes_repaired = _control_passes(*readings["repaired"])
    passes_defect = _control_passes(*readings["v1_defect"])
    print(f"\n  control on the repaired registry : {passes_repaired}")
    print(f"  control on the V-1 defect        : {passes_defect}")
    assert passes_repaired, "the control does not pass on the shipped registry"
    assert not passes_defect, (
        "the control passes on the restored V-1 defect -- it has no rejection "
        "region and P1 would be right about it"
    )


def test_the_census_class_for_this_row_is_the_measured_one_not_the_presumed_one(readings):
    """THE ATTACK. RED against P1 as stated.

    Ground truth is measured here, never asserted from a literal: the class is
    `live` exactly when the control passes on the shipped registry and fires on
    the production-domain mutation."""
    measured = ("live" if _control_passes(*readings["repaired"])
                and not _control_passes(*readings["v1_defect"]) else "vacuous")
    recorded = _sheet_class(TARGET) or P1_PRESUMED_CLASS
    source = "AUDIT.md" if _sheet_class(TARGET) else "P1 as pre-registered"
    print(f"\n  measured class: {measured}")
    print(f"  recorded class: {recorded}  (source: {source})")
    assert recorded == measured, (
        f"{TARGET}: {source} classes this row {recorded!r}; the mutation "
        f"measures it {measured!r}. P1 keys on whether the control built its "
        "own input; V-14's rule keys on whether the plant traverses the "
        "production path. This row builds its own input THROUGH M3_TASKS, so "
        "the path is production's and the control fires."
    )
