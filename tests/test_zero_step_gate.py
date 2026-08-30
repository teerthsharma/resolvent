"""The 0-step gate must not fire on its own null.

RED before the repair: `windowed_signed` seed 2 read 0.9999703932724174 untrained
at t*=2, n=2048 and aborted an eight-seed run, 2.96e-5 short of a bare 1.0 bound.
Measured over 16 untrained seeds, softmax's own minimum is 1.00055844 -- the
threshold sat on the edge of the distribution it was testing.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.r10_capacity_sweep import GATE_TOL


def gate(r0t, r0e):
    return r0t >= 1.0 - GATE_TOL and r0e >= 1.0 - GATE_TOL


def test_measured_null_tail_is_admitted():
    # the exact reading that aborted the run
    assert gate(1.0007840721511323, 0.9999703932724174)


def test_softmax_measured_minimum_still_passes():
    assert gate(1.00055844, 1.00055844)


def test_an_arm_that_really_beats_the_mean_is_still_rejected():
    # a tenth of the trained seed sd at this cell is ~1e-3; anything at or beyond
    # that is what the gate exists to catch
    assert not gate(1.0, 0.99)
    assert not gate(0.95, 0.95)


def test_tolerance_is_far_below_the_trained_seed_spread():
    # trained seed sd at t*=2, n=2048 is 0.010101; a gate tolerance comparable to
    # it would admit an arm that genuinely starts ahead
    assert GATE_TOL < 0.010101 / 5
