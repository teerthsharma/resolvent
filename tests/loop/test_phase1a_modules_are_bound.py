"""Phase 1a's modules carry self-checks that nothing collects.

THE DEFECT, found by the Health Inspector at the Phase 1a boundary:

    "1 of 11 phase-1a findings has any runnable binding; 0 were RED before
     repair. No test file exists for any phase-1a module."

`scale/it11_verdict.py`, `scale/idle_gate.py` and `scale/vram_gate.py` each end in
an assert-based `demo()` and a `__main__` guard. Every one of them passes. None of
them runs unless a human types the path, so none is part of any suite, and the
round's standing rule -- a finding is bound by a test that was RED before the
repair -- was satisfied for exactly one finding out of eleven.

A `demo()` behind `if __name__ == "__main__":` is a binding in the same sense that
a fire extinguisher in a locked cupboard is fire protection. This file gives each
one a collectible node id, so `pytest tests/loop` runs them and a regression in
any of the three gates fails the suite rather than waiting to be noticed.

WHAT THIS FILE DOES NOT CLAIM. It does not make the findings themselves RED-first;
that history cannot be rewritten and the Inspector's count stands as recorded. It
makes the repairs *executable from now on*, which is the part that is still
available. The three demos were each independently broken by the Inspector and by
this file's author and each now exits non-zero when broken -- that evidence is in
the modules' own comments, at the assertion that catches it.

WHY NOT JUST IMPORT AND CALL. Because `demo()` prints, and because a demo that
silently degrades to a no-op on a machine without the tools it probes is the exact
vacuity this round catalogues. `test_every_demo_asserts_something` plants a
failure into each module in a COPY and requires the copy to exit non-zero, so a
demo that has quietly become a no-op is caught here rather than trusted.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: (module path, a substring of a line whose removal must break the self-check).
#: The second element is the PLANTED FAILURE: it names a specific guarantee, so a
#: demo that has stopped checking that guarantee is detected instead of assumed.
MODULES = [
    ("scale/idle_gate.py", "    return count == 0"),
    ("scale/it11_verdict.py", "    if n < REQUIRED_SEEDS:"),
    ("scale/vram_gate.py", "    if not job_mib and host_mib is None:"),
]


def run(path: pathlib.Path, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(path)], cwd=str(cwd),
                          capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("rel,_plant", MODULES, ids=[m[0] for m in MODULES])
def test_the_self_check_passes_as_shipped(rel: str, _plant: str):
    """Each Phase 1a gate's own demo must pass. This is the collectible node the
    Inspector found missing -- until now these ran only when typed by hand."""
    r = run(ROOT / rel, ROOT)
    assert r.returncode == 0, (
        f"{rel} self-check fails as shipped:\n{(r.stderr or r.stdout)[-1500:]}"
    )
    assert "demo OK" in r.stdout, (
        f"{rel} exited 0 without reaching its own OK line; it may have returned "
        f"early rather than checked anything:\n{r.stdout[-800:]}"
    )


@pytest.mark.parametrize("rel,plant", MODULES, ids=[m[0] for m in MODULES])
def test_every_demo_asserts_something(rel: str, plant: str, tmp_path: pathlib.Path):
    """MUST-FIRE for the tests above. A demo that exits 0 while checking nothing
    would satisfy them, which is the vacuous-control shape this whole round
    catalogues -- and the Inspector found exactly that in three of these modules
    before they were repaired.

    The plant deletes one guarantee-bearing line in a COPY of the tree. If the
    demo still exits 0, it was not checking that guarantee.
    """
    work = tmp_path / "tree"
    shutil.copytree(ROOT / "scale", work / "scale")
    (work / "results").mkdir(exist_ok=True)
    for j in (ROOT / "results").glob("r10_it8_capacity_softmax_t*.jsonl"):
        shutil.copy2(j, work / "results" / j.name)

    target = work / rel
    src = target.read_text(encoding="utf-8")
    assert plant in src, (
        f"the planted line {plant!r} is not in {rel}; this must-fire is aimed at "
        "a line that no longer exists and would pass without testing anything"
    )
    target.write_text(src.replace(plant, plant.split("return")[0] + "return True"
                                  if "return count == 0" in plant
                                  else plant.replace("if ", "if False and ")),
                      encoding="utf-8")

    r = run(target, work)
    assert r.returncode != 0, (
        f"{rel} still exits 0 with {plant!r} disabled, so its demo does not "
        f"actually check that guarantee:\n{r.stdout[-800:]}"
    )
