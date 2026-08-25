"""Pre-registered mechanism prediction: the pivot plateau height is set by the
k-term background, so it must scale as k^(-1/2) (equivalently: a 4x increase in
k must cut the sign-flip rate to ~1/2).

Imports the EXISTING probe (scale/pivot_probe.py) rather than reimplementing
the operator.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scale.pivot_probe import run_arm


def test_plateau_scales_as_k_to_the_minus_half():
    r8 = run_arm("pivot_signed", 128, n_draws=1024, k=8,
                 placement="in_P", protocol="SCALING", seed=0)
    r32 = run_arm("pivot_signed", 128, n_draws=1024, k=32,
                  placement="in_P", protocol="SCALING", seed=0)
    ratio = r32["rate"] / r8["rate"]
    assert 0.35 <= ratio <= 0.71, (
        f"rate(k=8)={r8['rate']} (n={r8['n']}, sigma={r8['sigma']}, "
        f"term={r8['term']})  "
        f"rate(k=32)={r32['rate']} (n={r32['n']}, sigma={r32['sigma']}, "
        f"term={r32['term']})  "
        f"ratio=rate(k=32)/rate(k=8)={ratio}"
    )


def test_plateau_moves_at_all_with_k():
    r2 = run_arm("pivot_signed", 128, n_draws=1024, k=2,
                 placement="in_P", protocol="SCALING", seed=0)
    r32 = run_arm("pivot_signed", 128, n_draws=1024, k=32,
                  placement="in_P", protocol="SCALING", seed=0)
    ratio = r2["rate"] / r32["rate"]
    assert ratio >= 2.0, (
        f"rate(k=2)={r2['rate']} (n={r2['n']})  "
        f"rate(k=32)={r32['rate']} (n={r32['n']})  "
        f"ratio=rate(k=2)/rate(k=32)={ratio}"
    )
