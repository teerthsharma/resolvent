"""The stranger-command: one line, CPU, no GPU, no network.

`bench.sign_flip_rate` is the highest-value artifact this project produced --
an 8-second probe that tells anyone building a non-standard attention operator
whether it has content-conditional sign -- and for three rounds it was reachable
only by importing `ceq.bench` and knowing which seven keyword arguments
reproduce the calibration. That is not a shipped diagnostic, it is a private
function with a paper attached.

WHY THE DECAY CURVE IS NOT OPTIONAL OUTPUT. A single rate at a single context
length is exactly the number that misled this project for three rounds. The rate
decays as `s^-1.389` (R^2 0.9938) from 0.17480 at s=8 to 0.00391 at s=128, so
"my operator scores 0.15" means nothing until the reader knows it was measured
at s=8 and will read 2e-4 at the context their model actually uses. A diagnostic
that reports a rate without its context dependence would mislead the same way.

Every test parametrizes over cpu and cuda per the standing rule.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-registered, in code, before the module existed.
# The published decay slope, and how far a re-measurement may drift from it.
PUBLISHED_SLOPE = -1.389
SLOPE_TOL = 0.35


def _run(*args, env_extra=None):
    env = dict(os.environ, PYTHONPATH=ROOT, **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "ceq.diagnose", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=900)


def test_the_diagnostic_is_one_command_that_needs_no_gpu(device):
    """A reader with no CUDA must get the whole report, not a skipped section.

    Run with `CUDA_VISIBLE_DEVICES` emptied so torch genuinely cannot see a
    device: this box has an RTX 4060 and a diagnostic that silently depends on
    it would pass here and fail for every reader. The `device` parametrization
    selects the arm the child process is ASKED for, and the cuda arm proves the
    same command also runs where a GPU exists.
    """
    extra = {"CUDA_VISIBLE_DEVICES": ""} if device.type == "cpu" else {}
    r = _run("--fast", "--json", "--device", device.type, env_extra=extra)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert set(out) >= {"sign_flip_rate", "decay", "interventional"}, sorted(out)
    assert out["device"] == device.type, out["device"]


def test_the_report_carries_the_context_decay_and_its_slope(device):
    """A rate without its context dependence is the number that misled us."""
    r = _run("--fast", "--json", "--device", device.type)
    assert r.returncode == 0, r.stderr[-2000:]
    decay = json.loads(r.stdout)["decay"]
    seqs = sorted(int(s) for s in decay["rate"])
    assert len(seqs) >= 4, seqs
    assert decay["rate"][str(seqs[0])] > decay["rate"][str(seqs[-1])], decay
    assert abs(decay["slope"] - PUBLISHED_SLOPE) < SLOPE_TOL, decay["slope"]


def test_the_nonnegative_control_reads_exactly_zero(device):
    """Softmax is the control INSIDE the report, so a reader cannot supply one
    from memory. A non-negative operator composed with fixed linear value
    projections factors as (non-negative weight) x (fixed matrix), so no third
    token can move a sign at any depth: the probe must read exactly 0.0, not
    approximately.
    """
    r = _run("--fast", "--json", "--device", device.type)
    assert r.returncode == 0, r.stderr[-2000:]
    table = json.loads(r.stdout)["sign_flip_rate"]
    assert table["softmax"]["1"] == 0.0, table["softmax"]
    assert table["sgate"]["1"] > 0.0, table["sgate"]


def test_the_interventional_corpus_ships_with_its_oracle_and_its_scorer(device):
    """The corpus is only usable if a stranger gets the labels too.

    Every label here is what CPython printed -- there is no answer key in the
    repository -- so the report must carry the held-out cell, the arm scores,
    and the predict-the-mean line at 1.0 that says whether any arm learned
    anything at all.
    """
    r = _run("--fast", "--json", "--device", device.type)
    assert r.returncode == 0, r.stderr[-2000:]
    iv = json.loads(r.stdout)["interventional"]
    assert iv["held_out"] == ["-", 1], iv["held_out"]
    assert {"attention", "appnp", "signed"} <= set(iv["arms"]), sorted(iv["arms"])
    for k, v in iv["arms"].items():
        assert v["ood_nrmse"] > 0.0, (k, v)
    assert iv["predict_the_mean"] == 1.0, iv


def test_the_human_readable_run_prints_the_same_numbers_it_serialises(device):
    """Two output modes that can disagree are two diagnostics.

    The default run prints a table for a human; `--json` emits it for a script.
    If the printed slope and the serialised slope could drift apart, the number
    a reader quotes is not the number the tool measured.
    """
    j = _run("--fast", "--json", "--device", device.type)
    h = _run("--fast", "--device", device.type)
    assert j.returncode == 0 and h.returncode == 0, (j.stderr[-800:], h.stderr[-800:])
    slope = json.loads(j.stdout)["decay"]["slope"]
    assert f"{slope:.3f}" in h.stdout, (slope, h.stdout[-800:])
    assert "sign_flip_rate" in h.stdout


@pytest.mark.slow
def test_the_full_run_reproduces_the_published_calibration(device):
    """Without `--fast` the probe must reproduce the published rates exactly.

    Seven numbers, all at n_draws=128, published off this instrument. A
    diagnostic whose default settings give different answers from the campaign
    that produced it is a second instrument wearing the first one's name.
    """
    r = _run("--json", "--device", device.type)
    assert r.returncode == 0, r.stderr[-2000:]
    t = json.loads(r.stdout)["sign_flip_rate"]
    assert t["softmax"]["1"] == 0.0 and t["softmax"]["2"] == 0.0, t["softmax"]
    assert t["signed"]["2"] == 0.1875, t["signed"]
    assert t["sgate"]["1"] == 0.1484375, t["sgate"]
