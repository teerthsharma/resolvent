"""A bootstrap helper must not depend on where it sits in a shared stream.

This binds the G2 event of round 6. `scale/arm_a_k1.py::bslope` took a
`torch.Generator` BY REFERENCE, and its two call sites shared one. The flip
bootstrap ran first and consumed 2000 * 2400 = 4,800,000 int64 draws, so the
D_FR bootstrap began at that offset rather than at zero, and the published
interval stopped reproducing.

The second consequence was worse and went unnoticed for a round: K1's clause
requires both slopes "on the SAME draws", and a shared stream hands each half a
DIFFERENT resampled index sequence. The defect silently decoupled the two halves
of a clause whose entire point is that they be coupled.

The published numbers were never wrong. The producer acquired a defect after
publication, and repairing it restored both intervals exactly.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[2]

PUBLISHED = {
    "D_FR slope in k": "-0.4137  [-0.4579,-0.3704]",
    "flip slope in k": "-0.6960  [-1.0000,+0.0000]",
}


def test_bslope_takes_a_scalar_seed_not_a_generator():
    """A helper that accepts a live generator can be positioned wrongly by a
    caller. Nine other bootstrap helpers in this repo take a scalar; this one
    now does too."""
    from scale.arm_a_k1 import bslope
    params = list(inspect.signature(bslope).parameters)
    assert params[1] == "seed", f"second parameter is {params[1]!r}, expected 'seed'"
    ann = inspect.signature(bslope).parameters["seed"].annotation
    assert ann in (int, "int"), f"seed annotated {ann!r}; a Generator would regress this"


def test_two_calls_are_position_independent():
    """Calling the helper twice must give the second call the same stream the
    first got. Under the defect the second call started 4.8 million draws in."""
    from scale.arm_a_k1 import bslope
    a = {8: [0.0, 1.0, 0.5, 0.25], 16: [0.1, 0.2, 0.4, 0.8],
         32: [0.05, 0.1, 0.2, 0.3]}
    first = bslope(a, 4242, 64)
    second = bslope(a, 4242, 64)          # same seed, called AFTER the first
    assert first[:5] == second[:5], (
        "the second call differs from the first — the helper is stream-positional")


def test_the_published_k1_pair_reproduces_end_to_end():
    """The G2 bind. Both published intervals must come back from the shipped
    producer, unpiped, with the exit code read from the process itself."""
    r = subprocess.run([sys.executable, "scale/arm_a_k1.py", "--mode", "report"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"producer exited {r.returncode}"
    for key, want in PUBLISHED.items():
        line = next((l for l in r.stdout.splitlines() if key in l), None)
        assert line is not None, f"{key!r} absent from the report"
        assert want in line, f"{key}: expected {want!r}, got {line.strip()!r}"


def test_must_fire_a_shared_generator_would_be_caught():
    """The control. A helper sharing one generator across two calls returns
    different results for the second, which is exactly what the defect did — so
    the position-independence test above can genuinely fail."""
    def shared(data, gen, b):
        idx = torch.randint(0, len(data), (b,), generator=gen)
        return float(sum(data[int(i)] for i in idx) / b)

    g = torch.Generator().manual_seed(4242)
    data = [0.0, 1.0, 0.5, 0.25, 0.75, 0.125]
    first = shared(data, g, 64)
    second = shared(data, g, 64)          # same generator, now advanced
    assert first != second, "the control cannot detect a shared-stream helper"
