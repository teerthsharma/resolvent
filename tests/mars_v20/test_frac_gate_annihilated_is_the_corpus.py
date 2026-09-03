"""MARS, CEQ v20 R15 it.1 -- STRIKE against a per-cell column in the DECIDING
journal `results/v17k_r4_retake.jsonl`.

`frac_gate_annihilated` (and `n_zero_gates`, which is that fraction times the
draw size and is folded into ARM S-M''s manifest hash via `SMP_FIELDS`) is
published once per trained cell, beside `eval_nrmse`, as a property of the arm.

MISTAKES.md M-18 registers the CHECK this suite runs, and it is law, not
opinion:

    "Before a diagnostic is registered, run it on the CORPUS ALONE with no arm
     and report its value and its null. A diagnostic whose no-arm value equals
     its expected with-arm value is measuring the corpus."

Nothing here trains, writes, or touches the network. Every number is read from
the committed journal or counted off the corpus builder.
"""
from __future__ import annotations

import json
import pathlib

import torch

from scale import negation_scope as ns

REPO = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = REPO / "results" / "v17k_r4_retake.jsonl"

#: the retake header's own settings, read from the file rather than retyped.
S, D, D_MODEL = 64, 24, 16


def _rows():
    return [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]


def _header(rows):
    return next(r for r in rows if r.get("t") == "header")


def _cells(rows, kind):
    return [r for r in rows if r.get("kind") == kind and r.get("cell")]


def _corpus_sign_census():
    """The eval draw's own sign counts on the live band -- NO ARM.

    Built exactly as `scripts/v15_r1.py:697-701` builds it: `head = S-1-T_STAR`,
    `live = range(head+1, S)`, eval batch at the hardcoded `seed=12345`.
    """
    rows = _rows()
    h = _header(rows)
    t_star, n_eval = h["t_star"], h["n_eval"]
    head = h["s"] - 1 - t_star
    live = list(range(head + 1, h["s"]))
    x, *_ = ns.make_equilibrium_batch(n_eval, h["s"], h["d"], d_model=D_MODEL,
                                      t_star=t_star, seed=12345, device=None)
    a = x[:, live, ns.CH_DRIVE].reshape(-1)
    n = a.numel()
    return {"n": n,
            "neg": int((a < 0).sum()), "pos": int((a > 0).sum()),
            "frac_neg": int((a < 0).sum()) / n,
            "frac_pos": int((a > 0).sum()) / n}


# ------------------------------------------------------------------ CONTROL
# The instrument validates itself before it is allowed to condemn anything.

def test_control_the_census_is_not_degenerate_and_the_journal_is_readable():
    """If this fails, every reading below is an artifact of a broken probe."""
    c = _corpus_sign_census()
    assert c["n"] == 8192, c
    assert 0.4 < c["frac_neg"] < 0.6, c          # a real Rademacher split
    assert c["frac_neg"] != c["frac_pos"], c     # the two values are distinct
    cells = _cells(_rows(), "arm_smprime")
    assert len(cells) == 8, len(cells)
    assert all("frac_gate_annihilated" in r for r in cells)


def test_control_a_column_that_is_genuinely_the_arms_does_not_match_the_census():
    """`eval_nrmse` is an arm reading. It must NOT coincide with the census --
    otherwise the comparison below would condemn every column alike."""
    c = _corpus_sign_census()
    target = {c["frac_neg"], c["frac_pos"]}
    seen = {r["eval_nrmse"] for r in _cells(_rows(), "arm_smprime")}
    assert not (seen & target), seen & target


# ------------------------------------------------------------------- STRIKE

def test_frac_gate_annihilated_is_not_the_corpus_sign_census():
    """M-18's registered check, applied to ARM S-M''s published column.

    Eight independently seeded, independently trained cells. If the column is a
    property of the ARM, its published values must not be the corpus's own sign
    counts read back.
    """
    c = _corpus_sign_census()
    corpus_values = {c["frac_neg"], c["frac_pos"]}
    cells = _cells(_rows(), "arm_smprime")
    hits = [(r["cell"], r["frac_gate_annihilated"]) for r in cells
            if r["frac_gate_annihilated"] in corpus_values]
    assert not hits, (
        f"{len(hits)}/{len(cells)} trained cells publish a gate diagnostic "
        f"equal to a CORPUS-ONLY statistic. corpus: neg={c['neg']} "
        f"({c['frac_neg']!r}), pos={c['pos']} ({c['frac_pos']!r}) of n={c['n']}. "
        f"cells: {hits}")


def test_arm_pl_frac_gate_annihilated_can_take_a_second_value():
    """The same column for ARM PL. `scripts/v15_r1.py:361` reads it off
    `g = g_head(x)`, a finite `nn.Linear` output, so `~isfinite(g)` is empty at
    every parameter value -- `exp` is never zero, which is
    `no_prefix_scan_represents_a_zero_gate` in `lean/CEQ/V16Domain.lean`.

    A column that the arm's construction pins to one value across every cell is
    a restatement of a theorem, not a measurement of a run.
    """
    seen = {r["frac_gate_annihilated"] for r in _cells(_rows(), "arm_pl")}
    assert len(seen) > 1, (
        f"all 8 ARM PL cells publish frac_gate_annihilated = {seen}; "
        f"the column has zero variance across the arm's own tournament")


def test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus():
    """`n_zero_gates` is in `SMP_FIELDS` (`ceq/arm_smprime.py:95`) and is folded
    into the cell's identity hash. `scale/identity_manifest.py` states the
    purpose: "A field missing here is a field two different arms can collide
    on." A field whose every observed value is a corpus count distinguishes
    draws, not arms.
    """
    c = _corpus_sign_census()
    corpus_counts = {c["neg"], c["pos"]}
    seen = {r["manifest"]["smp_values"]["n_zero_gates"]
            for r in _cells(_rows(), "arm_smprime")}
    assert not (seen <= corpus_counts), (
        f"every published n_zero_gates {sorted(seen)} is a corpus sign count "
        f"{sorted(corpus_counts)}; the identity field carries the draw, not the arm")
