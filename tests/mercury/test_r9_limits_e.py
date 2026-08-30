"""Limits paragraph (e) must describe the estimator, not audit other files.

It was stale on three counts, each verified this session:

  1. It accuses `CHECKLIST.md:1167` of printing the exact pair under the label
     `B=10000`. That line now reads "exact enumeration over all 5**5 = 3125
     paired resamples" -- corrected, so the accusation is false.
  2. It cites `STATE.md:19` for the same thing. `STATE.md` contains neither
     endpoint anywhere in the file, and :19 is an unrelated row.
  3. It frames exact-versus-Monte-Carlo as "a SECOND FAMILY". Measured, they are
     one estimator reaching ADJACENT ATOMS of one 126-atom lattice -- atoms #7
     and #8 for `settled - twin`, with `ci_hi` identical at atom #117 -- and the
     Monte-Carlo endpoint is not one number: over bootstrap seeds 0..99 `ci_lo`
     takes four values, atoms #7 through #10.

A paragraph that stores claims about other files goes stale every time those
files are fixed, which is what happened twice here. The rewrite describes the
estimator and lets the other files carry their own tests.

    python -m pytest tests/mercury/test_r9_limits_e.py -q
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import pathlib
import re

import pytest

from scale import capability_table as CT

ROOT = pathlib.Path(CT.__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def limits() -> str:
    return CT.LIMITS


def test_the_checklist_accusation_is_gone_because_it_is_false(limits):
    line = (ROOT / "CHECKLIST.md").read_text(encoding="utf-8").splitlines()[1166]
    assert "exact enumeration" in line, "premise: CHECKLIST.md:1167 was corrected"
    assert "CHECKLIST.md:1167" not in limits


def test_the_state_citation_is_gone_because_the_file_lacks_the_pair(limits):
    state = (ROOT / "STATE.md").read_text(encoding="utf-8")
    assert "0.042903" not in state and "0.031557" not in state
    assert "STATE.md:19" not in limits


def test_the_second_family_framing_is_replaced_by_the_lattice(limits):
    assert "SECOND FAMILY" not in limits
    assert "atom" in limits, "the lattice must be named"
    assert "126" in limits


def test_the_surviving_numeric_claims_are_still_stated(limits):
    """The rewrite must not throw away what was true."""
    assert "3125" in limits or "5**5" in limits
    assert "0.146551" in limits, "the orphan endpoint is still an open defect"


def test_the_orphan_endpoint_is_still_in_neither_family():
    """Adversarial pass: (e) keeps this claim, so it must still hold."""
    have = collections.defaultdict(dict)
    for line in (ROOT / "results" / "m3_quintuple_v2.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["key"]
        if "_ntr8192_nev512_" not in k or "_task" in k:
            continue
        m = re.match(r"([a-z]+)_k\d+_.*_sd(\d+)$", k)
        if m:
            have[m.group(1)][int(m.group(2))] = r["value"]["eval_nrmse"]
    S = (0, 1, 2, 3, 4)
    d = [have["softmax"][s] - have["settled"][s] for s in S]
    reps = [math.fsum(sorted(c)) / 5 for c in itertools.product(d, repeat=5)]
    assert not any(abs(x - 0.146551) < 5e-7 for x in reps)
