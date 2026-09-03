"""MARS, CEQ v20 R15 it.2 -- the thread left unpulled at it.1.

Two questions the it.1 report declared undetermined, and the Inspector's C12
margin note, are ONE question:

  * `arm_smprime:t2:n2048:seed2` is the only cell that ESCAPES MARS STRIKE 1
    (its `frac_gate_annihilated` is not a corpus sign count).
  * It is the only cell that CARRIES MARS STRIKE 2(b) (the `exp_scan` planted
    negative has an empty rejection region there).
  * It is the only cell that makes the it.1 GREEN control
    `test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus`
    pass -- Inspector C12: "seen = {0, 4123, 4069}, corpus = {4123, 4069};
    passes ONLY because of 0."

If all three are the same fact, the green control is not independent evidence:
it passes on the datum its own strike exempts.

Everything is read from the committed journal `results/v17k_r4_retake.jsonl`.
Nothing trains, writes, or touches the network or Kaggle.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "results" / "v17k_r4_retake.jsonl"

# The corpus's own Rademacher sign counts at the hardcoded eval seed 12345,
# n_eval=4096, live band [62, 63] -> |a_live| = 8192. Counted with no arm in
# the room at it.1 and reproduced by the Inspector (C14).
CORPUS_SIGN_COUNTS = {4123, 4069}


def _cells(arm):
    out = {}
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("cell") or rec.get("id") or rec.get("cell_id")
        if not cid or not str(cid).startswith(arm + ":"):
            continue
        out[str(cid)] = rec
    return out


def _val(rec, *path, default=None):
    cur = rec
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _smp(rec, key, default=None):
    for pre in (("manifest", "smp_values"), ("manifest", "pl_values"), ()):
        v = _val(rec, *pre, key, default=None)
        if v is not None:
            return v
    return rec.get(key, default)


SMP = _cells("arm_smprime")
PL = _cells("arm_pl")
ESCAPE = "arm_smprime:t2:n2048:seed2"


# ---------------------------------------------------------------- CONTROLS

def test_control_the_journal_parses_and_carries_both_arms_at_eight_seeds():
    """If this fails nothing below means anything."""
    assert len(SMP) == 8, f"arm_smprime cells: {sorted(SMP)}"
    assert len(PL) == 8, f"arm_pl cells: {sorted(PL)}"
    assert ESCAPE in SMP


def test_control_a_hat_min_is_a_live_column_not_structurally_zero():
    """A column that is 0.0 by construction proves nothing when it is 0.0.

    ARM PL's `a_hat_min` is strictly positive on all eight of its cells, so
    `a_hat_min == 0.0` on ARM S-M' is a MEASURED floor, not a journal artifact
    or a formatting default. This is the planted positive for the column.
    """
    pl_mins = {c: r["a_hat_min"] for c, r in PL.items()}
    assert all(v > 0.0 for v in pl_mins.values()), pl_mins
    assert min(pl_mins.values()) < 0.05, pl_mins  # and it gets close to zero


def test_control_the_gate_floor_predicts_the_zero_gate_count_on_every_cell():
    """The mechanism, stated as a biconditional and checked 8/8.

    A gate that reaches the closed cap's lower endpoint annihilates; one that
    does not, cannot. If this holds on every cell then `a_hat_min == 0` is THE
    fact and `n_zero_gates` is its shadow.
    """
    for cid, rec in sorted(SMP.items()):
        reaches_zero = rec["a_hat_min"] == 0.0
        annihilates = _smp(rec, "n_zero_gates", 0) > 0
        assert reaches_zero == annihilates, (
            f"{cid}: a_hat_min={rec['a_hat_min']} n_zero_gates="
            f"{_smp(rec, 'n_zero_gates')}"
        )


# ------------------------------------------------------------------- REDS

def test_the_escape_cell_is_not_the_unique_cell_off_the_cap_boundary():
    """RED: seed 2 is the ONLY cell whose gate never touches either endpoint.

    Seven of eight cells sit at `a_hat_min == 0.0` exactly (the closed cap's
    lower endpoint). Seed 2 alone lives strictly inside the interval. That one
    fact is what makes it escape STRIKE 1 and carry STRIKE 2(b).
    """
    off_boundary = sorted(c for c, r in SMP.items() if r["a_hat_min"] > 0.0)
    assert len(off_boundary) != 1, (
        "ONE FACT: exactly one of eight cells has a gate that never reaches the "
        f"cap's lower endpoint -- {off_boundary}. a_hat_min per cell: "
        + repr({c: r["a_hat_min"] for c, r in sorted(SMP.items())})
    )


def test_the_it1_green_control_survives_removing_the_cell_its_strike_exempts():
    """RED: the green control passes only on the datum STRIKE 1 exempts.

    it.1's `test_the_manifest_identity_field_n_zero_gates_carries_more_than_the
    _corpus` asserts `set(n_zero_gates) - CORPUS != empty`. Remove the single
    cell that STRIKE 1 already declares anomalous and the assertion dies.
    A control whose entire margin is the strike's own exception is not
    independent of the strike.
    """
    without = {_smp(r, "n_zero_gates") for c, r in SMP.items() if c != ESCAPE}
    assert without - CORPUS_SIGN_COUNTS, (
        f"with {ESCAPE} removed, every remaining n_zero_gates is a corpus sign "
        f"count: seen={sorted(without)} corpus={sorted(CORPUS_SIGN_COUNTS)}. "
        "The it.1 GREEN and the it.1 RED rest on the same cell."
    )


def test_arm_pl_gate_ceiling_does_not_partition_its_own_tournament():
    """RED: ARM PL's three worst cells are exactly its three unbounded gates.

    `pl_values.dyn_range_bound` is `inf` on every cell -- the arm ships no cap,
    unlike ARM S-M' whose `m_max <= 1.0`. `a_hat_max` then splits the eight
    cells with no overlap: {2,3,7} at 12.77 / 49.66 / 116.01 and the other five
    under 1.51; and eval_nrmse splits on the SAME partition.
    """
    hi = {c: r for c, r in PL.items() if r["a_hat_max"] > 2.0}
    lo = {c: r for c, r in PL.items() if r["a_hat_max"] <= 2.0}
    worst = max(r["eval_nrmse"] for r in lo.values())
    best_of_hi = min(r["eval_nrmse"] for r in hi.values())
    bounds = {c: _smp(r, "dyn_range_bound") for c, r in sorted(PL.items())}
    assert not (hi and lo and best_of_hi > worst), (
        f"ARM PL a_hat_max partitions eval_nrmse with a gap: unbounded-gate "
        f"cells {sorted(hi)} all score >= {best_of_hi:.6f}; bounded-gate cells "
        f"{sorted(lo)} all score <= {worst:.6f}. dyn_range_bound per cell: "
        f"{bounds}. a_hat_max: "
        + repr({c: r["a_hat_max"] for c, r in sorted(PL.items())})
    )
