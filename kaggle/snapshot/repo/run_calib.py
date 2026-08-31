#!/usr/bin/env python
"""Calibration gate for `ceq/bench.py::sign_flip_rate`. EXITS NONZERO ON DRIFT.

WHY THIS FILE WAS REWRITTEN. The previous version measured the four values,
printed them, then printed four hardcoded target STRINGS underneath, and
returned exit 0 unconditionally. It never compared anything. A drifted number
would have been printed directly beneath its own target and the script would
still have exited clean, so stopping condition G2 ("any bench.py calibration
number moves when a new arm is added") could never have fired.

That is the eleventh instrument in this project that was internally consistent
and externally wrong, and it is the same class as the `sorry` detector that
fired on the sentence "No `sorry` anywhere" and the ParaFormer arm that ran
softmax: a checker whose output has the SHAPE of evidence and none of the
content.

CALIBRATE THE CALIBRATOR. `--self-test` feeds this gate a deliberately wrong
target and requires it to FAIL. A gate that has never been observed failing is
not a gate. Run it before trusting a green.

Exit codes:  0 = all four bit-identical.  1 = drift (G2 fires).  2 = self-test
did not fail when it should have, i.e. the gate itself is broken.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
from ceq import bench

#: (label, kwargs, expected). Expected values are the published calibration and
#: are compared with `!=` on the float, not with a tolerance: the claim is
#: BIT-IDENTICAL reproduction, and a tolerance would let a real drift through.
CASES = [
    ("signed  hops=3", dict(kind="signed",  depth=1, hops=3), 0.046875),
    ("sgate   hops=1", dict(kind="sgate",   depth=1, hops=1), 0.0234375),
    ("sgate   hops=2", dict(kind="sgate",   depth=1, hops=2), 0.1640625),
    ("softmax hops=3", dict(kind="softmax", depth=1, hops=3), 0.0),
]


def run(expected_override: dict[str, float] | None = None) -> int:
    device = torch.device("cpu")
    print("calibration: ceq/bench.py::sign_flip_rate, n_draws=128, s=8, cpu")
    print(f"{'case':>16} {'measured':>14} {'expected':>14}   verdict")
    bad = []
    for label, kw, want in CASES:
        if expected_override and label in expected_override:
            want = expected_override[label]
        got = bench.sign_flip_rate(n_draws=128, s=8, device=device, **kw)
        ok = (got == want)
        if not ok:
            bad.append((label, got, want))
        print(f"{label:>16} {got:>14.10f} {want:>14.10f}   {'OK' if ok else 'DRIFT'}")
    print()
    if bad:
        print(f"G2 FIRES: {len(bad)} of {len(CASES)} calibration numbers moved.")
        for label, got, want in bad:
            print(f"  {label}: measured {got!r}, published {want!r}, "
                  f"delta {got - want:+.10g}")
        print("Every number produced by this instrument is VOID until the cause")
        print("is found. The only legal work is finding what moved.")
        return 1
    print(f"CALIBRATED: {len(CASES)}/{len(CASES)} bit-identical to the published "
          f"values.")
    return 0


def self_test() -> int:
    """Feed the gate a wrong target and require it to FAIL."""
    print("=== SELF-TEST: the gate must FAIL on a deliberately wrong target ===")
    rc = run({"sgate   hops=2": 0.1640625 + 1e-9})
    print()
    if rc == 0:
        print("GATE IS BROKEN: it passed a target it should have rejected. "
              "Do not trust any green from this script.")
        return 2
    print("gate self-test OK: it rejected a wrong target (exit 1 as required).")
    print()
    print("=== now the real calibration ===")
    return run()


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else run())
