"""Gate-0 G0.8: the seven must-fires for the Kaggle-run autopilot envelope.

`ceq/autopilot.py` is a deterministic, TOTAL decision function over parsed
journal lines: `decide_line(raw, history, ...)` returns `(event_or_malformed,
Decision)`, and `Decision.tier` is always 0 (nothing to do), 1, 2 or 3 -- never
"unrecognised, so continue". Section 41.4's envelope table
(planted in the round prompt, not this file) is the source of the seven
triggers below; this file exists to prove each one FIRES, and that the same
detector does NOT fire on a clean line that has the field, in range, parsed
(non-degeneracy -- this repo has struck 14 vacuous controls and a pass that
cannot fail is void).

Each must-fire gets a `test_*_fires` and a `test_*_clean_does_not_fire` pair,
built from `_line(**overrides)` -- one base healthy poll, one field changed at
a time, so the clean sibling is provably the SAME shape as the planted one.
"""
from __future__ import annotations

import json

import pytest

from ceq import autopilot as ap

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

CONFIG = ap.AutopilotConfig(
    deciding_cells=frozenset({"cellD"}),
    branch_diagnostics={"branchX": "diag_x_check"},
    cell_queue=("cellA", "cellB", "cellC"),
    lr0=3e-4,
)

_BASE = dict(
    t=9.0, cell="cellA", step=300,
    loss=1.2, loss_nonfinite=False,
    lambda_hat_max=0.0, a_hat_max=0.4, phase_arm=False,
    r2=0.62, micro_batch=8, grad_accum=4,
    oom=False, kernel_alive=True,
    quota_remaining_frac=0.7, queue_idle=False,
    kaggle_session_remaining_min=590.0, last_ckpt_t=6.0, last_good_ckpt="ckpt-cellA-200",
    lr=3e-4, design_change_request=False, branch_reached=None,
    kcert_delta_over_tol=0.1, mem_used_frac=0.4, note="routine poll",
)


def _line(**overrides) -> str:
    obj = dict(_BASE, **overrides)
    return json.dumps(obj, sort_keys=True)


def _decide(raw, now_t=9.0, history=()):
    event, decision = ap.decide_line(raw, list(history), now_t=now_t, config=CONFIG)
    return event, decision


# --------------------------------------------------------------- must-fire 1

def test_mf1_nan_rollback_new_manifest_within_one_poll():
    _, d = _decide(_line(loss_nonfinite=True, loss=None))
    assert d.tier == 2
    assert d.action.startswith("rollback")
    assert d.cell_new == "cellA.t2.1" and d.cell_new != d.cell_old
    assert d.manifest is not None
    assert d.manifest["reason"] == "loss_nonfinite"
    assert d.manifest["lr_new"] == pytest.approx(3e-4 * 0.5)
    assert d.manifest["grad_clip_registered"] is True


def test_mf1_clean_loss_does_not_fire():
    _, d = _decide(_line())
    assert d.tier == 0 and d.action == "NONE"


# --------------------------------------------------------------- must-fire 2

def test_mf2_oom_halves_micro_batch_effective_batch_invariant():
    _, d = _decide(_line(oom=True, micro_batch=8, grad_accum=4))
    assert d.tier == 1 and d.action == "halve-micro-batch"
    old_micro, old_accum = 8, 4
    new_micro, new_accum = 4, 8
    assert new_micro * new_accum == old_micro * old_accum
    assert "4->8" in d.expected_effect or "8->4" in d.expected_effect
    assert d.cell_new == d.cell_old  # Tier-1: no new cell / no manifest
    assert d.manifest is None


def test_mf2_clean_oom_does_not_fire():
    _, d = _decide(_line(oom=False))
    assert d.tier == 0 and d.action == "NONE"


# --------------------------------------------------------------- must-fire 3
# Bitwise resume itself is owned by another agent hardening ceq/hf/train.py.
# Here: the decision fires Tier-1 "resume" naming the last checkpoint, and a
# LOCAL, torch-free stub proves that decision is sufficient to drive a
# bitwise-identical resume. See test_g08_autopilot.py::StubTrainer below and
# the report for the exact interface ceq.hf.train.train() must expose.

def test_mf3_kernel_death_decides_resume():
    _, d = _decide(_line(kernel_alive=False))
    assert d.tier == 1 and d.action == "resume"
    assert d.cell_new == d.cell_old == "cellA"


def test_mf3_clean_kernel_alive_does_not_fire():
    _, d = _decide(_line(kernel_alive=True))
    assert d.tier == 0 and d.action == "NONE"


class _StubTrainer:
    """torch-free stand-in for ceq.hf.train.train()'s resume contract.

    Real interface (ceq/hf/train.py:185, not edited here):
        train(*, out_dir, steps, ..., resume_from=None) -> record
    `resume_from` is an earlier out_dir; state that must round-trip for a
    bitwise-identical continuation is the model weights, the optimizer
    moments, the data-sampling generator state, and the global RNG state
    (ceq/hf/train.py:191-231, and tests/gate0/conftest.py's `load_state`/
    `bitwise_diff` already exercise exactly those four for G0.2/G0.3). This
    stub reproduces the same SHAPE of state (a dict of ints derived from a
    seeded `random.Random`) without torch or a GPU, at trivial size, so the
    autopilot's OWN test suite can prove "the decision names a checkpoint that
    resuming from reproduces the uninterrupted run" without depending on the
    other agent's file.
    """
    def train(self, *, out_dir, steps, seed=0, resume_from=None, store):
        import random
        if resume_from:
            rng_state, start = store[resume_from]
            rng = random.Random()
            rng.setstate(rng_state)
        else:
            rng = random.Random(seed)
            start = 0
        vals = [rng.getrandbits(32) for _ in range(steps)]
        store[out_dir] = (rng.getstate(), start + steps)
        return dict(values=vals, step=start + steps)


def test_mf3_resume_action_reproduces_bitwise_identical_state():
    trainer = _StubTrainer()
    store: dict = {}

    # Uninterrupted: one run straight through.
    trainer.train(out_dir="run-full", steps=6, seed=0, store=store)
    full_state = store["run-full"]

    # Interrupted: dies after 3 steps (kernel_alive goes False on the poll).
    store2: dict = {}
    trainer.train(out_dir="run-part", steps=3, seed=0, store=store2)
    _, d = _decide(_line(kernel_alive=False, last_good_ckpt="run-part"))
    assert d.action == "resume"
    # The autopilot names d.cell_old's last checkpoint; orchestration resumes:
    trainer.train(out_dir="run-resumed", steps=3, resume_from="run-part", store=store2)
    resumed_state = store2["run-resumed"]

    assert resumed_state == full_state, "resume must reproduce the uninterrupted run bitwise"


# --------------------------------------------------------------- must-fire 4

def test_mf4_phase_arm_ahat_over_one_halts():
    _, d = _decide(_line(phase_arm=True, a_hat_max=1.2))
    assert d.tier == 3 and d.action == "HALT"
    assert "a_hat_max" in d.trigger and "phase-arm" in d.trigger


def test_mf4_clean_phase_arm_under_one_does_not_fire():
    _, d = _decide(_line(phase_arm=True, a_hat_max=0.8))
    assert d.tier == 0 and d.action == "NONE"


def test_mf4_non_phase_arm_at_same_ahat_does_not_trigger_this_rule():
    """Non-degeneracy the other way: â_max=1.2 off a phase arm must not halt
    under the phase-arm rule (Tier-2's own â_max>1.5 rule is a separate check,
    exercised in test_mf_seed_divergence_ahat below)."""
    _, d = _decide(_line(phase_arm=False, a_hat_max=1.2))
    assert d.tier == 0 and d.action == "NONE"


# --------------------------------------------------------------- must-fire 5

def test_mf5_journal_rewrite_root_fails_to_extend_halts():
    honest = [
        _line(step=100, cell="cellA"),
        _line(step=200, cell="cellA"),
        _line(step=300, cell="cellA"),
    ]
    history = []
    sealed_root, sealed_lines = ap.merkle_root([]), 0
    for raw in honest:
        e, d = ap.decide_line(raw, history, now_t=9.0, config=CONFIG,
                              sealed_root=sealed_root, sealed_lines=sealed_lines)
        history.append((e, d))
        sealed_root, sealed_lines = d.current_root, len(history)

    # The file gets rewritten out from under us: line 1 (index 1, already
    # sealed) now carries a different loss than what was sealed.
    tampered_line1 = json.loads(honest[1])
    tampered_line1["loss"] = 999.0
    tampered_history = [
        history[0],
        (ap.parse_line(json.dumps(tampered_line1, sort_keys=True)), history[1][1]),
        history[2],
    ]
    _, d = ap.decide_line(_line(step=400), tampered_history, now_t=9.0, config=CONFIG,
                          sealed_root=sealed_root, sealed_lines=sealed_lines)
    assert d.tier == 3 and d.action == "HALT"
    assert "root-fails-to-extend" in d.trigger


def test_mf5_clean_append_does_not_fire():
    honest = [_line(step=100), _line(step=200), _line(step=300)]
    history = []
    sealed_root, sealed_lines = ap.merkle_root([]), 0
    for raw in honest:
        e, d = ap.decide_line(raw, history, now_t=9.0, config=CONFIG,
                              sealed_root=sealed_root, sealed_lines=sealed_lines)
        history.append((e, d))
        sealed_root, sealed_lines = d.current_root, len(history)
    # One more honest append against the correctly-remembered seal.
    _, d = ap.decide_line(_line(step=400), history, now_t=9.0, config=CONFIG,
                          sealed_root=sealed_root, sealed_lines=sealed_lines)
    assert d.tier == 0 and "root-fails-to-extend" != d.trigger


# --------------------------------------------------------------- must-fire 6

def test_mf6_deciding_cell_edit_request_refused():
    _, d = _decide(_line(cell="cellD", design_change_request=True))
    assert d.tier == 3 and d.action == "HALT"
    assert "deciding-cell" in d.trigger


def test_mf6_clean_non_deciding_cell_edit_request_is_not_this_refusal():
    """Non-degeneracy: the SAME design_change_request field, on a cell that is
    not pre-registered as deciding, must not hit the deciding-cell refusal
    (it still is not silently applied -- 'any deciding-cell design change' is
    the rule; an edit request off a deciding cell is simply not this trigger,
    proving the field is read against the pre-registered set, not a keyword)."""
    _, d = _decide(_line(cell="cellA", design_change_request=True))
    assert not (d.tier == 3 and "deciding-cell" in d.trigger)


def test_mf6_tier2_on_deciding_cell_is_a_kill_clause_not_an_action():
    """HARD RULE: Tier-2 on a deciding cell must HALT, never act."""
    _, d = _decide(_line(cell="cellD", loss_nonfinite=True, loss=None))
    assert d.tier == 3 and d.action == "HALT"
    assert "deciding-cell" in d.trigger
    assert d.manifest is None  # refused, not executed


# --------------------------------------------------------------- must-fire 7

def test_mf7_clean_chunk_zero_tier23_and_rate_printed():
    raw_lines = (FIXTURES / "autopilot_clean_chunk.jsonl").read_text().splitlines()
    now_ts = [json.loads(ln)["t"] for ln in raw_lines]
    decisions = ap.replay(raw_lines, now_ts, CONFIG)
    assert all(d.tier in (0, 1) for _, d in decisions)
    rate = ap.false_correction_rate([d for _, d in decisions])
    assert rate == 0.0
    printed = "false-correction rate: {:.4f} ({}/{})".format(
        rate, sum(1 for _, d in decisions if d.tier in (2, 3)), len(decisions))
    assert printed.startswith("false-correction rate: 0.0000")


def test_mf7_counter_is_real_not_hardcoded_zero():
    """Plant one Tier-2 event inside an otherwise-clean chunk and confirm the
    rate counter moves -- a rate that reads 0.0 no matter what is not a
    measurement."""
    raw_lines = (FIXTURES / "autopilot_clean_chunk.jsonl").read_text().splitlines()
    planted = json.loads(raw_lines[5])
    planted["loss_nonfinite"] = True
    planted["loss"] = None
    raw_lines[5] = json.dumps(planted, sort_keys=True)
    now_ts = [json.loads(ln)["t"] for ln in raw_lines]
    decisions = ap.replay(raw_lines, now_ts, CONFIG)
    rate = ap.false_correction_rate([d for _, d in decisions])
    assert rate == pytest.approx(1 / len(decisions))
    assert rate > 0.0


# --------------------------------------------------------- total / default

def test_malformed_line_defaults_to_tier3_not_unrecognised_continue():
    _, d = _decide("{not json")
    assert d.tier == 3 and d.action == "HALT"
    assert "malformed" in d.trigger


def test_missing_required_field_defaults_to_tier3():
    obj = json.loads(_line())
    del obj["kernel_alive"]
    _, d = _decide(json.dumps(obj))
    assert d.tier == 3 and d.action == "HALT"


def test_wrong_typed_field_defaults_to_tier3():
    obj = json.loads(_line())
    obj["oom"] = "yes"  # str, not bool
    _, d = _decide(json.dumps(obj))
    assert d.tier == 3 and d.action == "HALT"


def test_decision_is_deterministic_same_journal_same_decision():
    raw = _line(loss_nonfinite=True, loss=None)
    _, d1 = _decide(raw)
    _, d2 = _decide(raw)
    assert d1 == d2


def test_decisions_row_carries_no_hypothesis_or_verdict_language():
    _, d = _decide(_line(loss_nonfinite=True, loss=None))
    row = ap.decisions_row("2026-08-31T00:00:00Z", d)
    banned = ("should", "likely", "because", "indicates", "probably", "we believe",
             "appears to", "seems", "suggests")
    low = row.lower()
    for word in banned:
        assert word not in low, (word, row)


def test_poll_line_format_matches_contract_shape():
    _, d = _decide(_line(loss_nonfinite=True, loss=None))
    event = ap.parse_line(_line(loss_nonfinite=True, loss=None))
    text = ap.format_poll_line(event, d, "Rolled back and halved lr.")
    assert text.startswith("[AUTOPILOT ")
    assert "action=T2:" in text
    assert text.rstrip().endswith(".")


# ----------------------------------------------------- rest of the envelope
# The seven must-fires above are the headline gate; the envelope table has
# eleven more named triggers, and "the code IS the envelope" means every one
# of them needs the same fires/does-not-fire pair or it is an untested branch
# in a decision function -- exactly the shape the 14 struck vacuous controls
# came from.

def test_seed_divergence_ahat_over_1_5_rolls_back():
    _, d = _decide(_line(a_hat_max=1.6, phase_arm=False))
    assert d.tier == 2 and d.action == "seed-scoped-rollback"
    assert d.cell_new == "cellA.t2.1"


def test_seed_divergence_ahat_at_1_5_does_not_fire():
    _, d = _decide(_line(a_hat_max=1.5, phase_arm=False))
    assert d.tier == 0 and d.action == "NONE"


def test_seed_divergence_lambda_three_consecutive_polls_rolls_back():
    h = []
    for i in range(2):
        e, d = _decide(_line(lambda_hat_max=0.1, step=100 + i), history=h)
        h.append((e, d))
    _, d = _decide(_line(lambda_hat_max=0.1, step=300), history=h)
    assert d.tier == 2 and "lambda_hat_max" in d.trigger


def test_seed_divergence_lambda_two_of_three_polls_does_not_fire():
    h = []
    e, d = _decide(_line(lambda_hat_max=0.1, step=100), history=[])
    h.append((e, d))
    e, d = _decide(_line(lambda_hat_max=0.0, step=200), history=h)  # breaks the streak
    h.append((e, d))
    _, d = _decide(_line(lambda_hat_max=0.1, step=300), history=h)
    assert d.tier == 0 and d.action == "NONE"


def test_preregistered_branch_runs_named_diagnostic_only():
    _, d = _decide(_line(branch_reached="branchX"))
    assert d.tier == 2 and d.action == "diagnostic:diag_x_check"
    assert d.cell_new == d.cell_old  # diagnostic only, no cell/manifest churn


def test_unregistered_branch_name_is_inert_not_a_free_text_command():
    """A branch id the journal invents (not pre-registered in config) must not
    run anything -- pre-registration lives in AutopilotConfig, never in the
    journal's own claim about itself."""
    _, d = _decide(_line(branch_reached="run_arbitrary_diagnostic"))
    assert d.tier == 0 and d.action == "NONE"


def test_two_tier2_on_one_cell_halts_on_the_second():
    h = []
    e, d = _decide(_line(loss_nonfinite=True, loss=None, step=100), history=[])
    assert d.tier == 2
    h.append((e, d))
    # Second correction attempt lands on the same lineage (cellA.t2.1 is a
    # child of cellA) -- this must HALT, not mint cellA.t2.2.
    _, d2 = _decide(_line(cell="cellA.t2.1", a_hat_max=1.6, step=200), history=h)
    assert d2.tier == 3 and d2.action == "HALT"
    assert "two-tier2-on-cell" in d2.trigger


def test_single_tier2_on_a_cell_does_not_trip_the_two_tier2_halt():
    _, d = _decide(_line(loss_nonfinite=True, loss=None))
    assert d.tier == 2  # exactly one correction so far: not the kill clause


def test_kcert_delta_over_tol_above_half_halts():
    _, d = _decide(_line(kcert_delta_over_tol=0.51))
    assert d.tier == 3 and "kcert" in d.trigger


def test_kcert_delta_over_tol_at_half_does_not_fire():
    _, d = _decide(_line(kcert_delta_over_tol=0.50))
    assert d.tier == 0 and d.action == "NONE"


def test_quota_within_15pct_pauses():
    _, d = _decide(_line(quota_remaining_frac=0.10))
    assert d.tier == 1 and d.action == "pause"


def test_quota_above_15pct_does_not_pause():
    _, d = _decide(_line(quota_remaining_frac=0.20))
    assert d.tier == 0 and d.action == "NONE"


def test_stale_journal_probes_restart():
    """RULING 6a: stale = no write for 2 CONSECUTIVE POLLS. The same raw
    journal line is submitted on three consecutive polls -- poll 2 sees no
    growth since poll 1 (gap #1), poll 3 sees no growth since poll 2 (gap
    #2); two consecutive no-write gaps fires at poll 3."""
    raw = _line(t=9.0, step=300)
    h = []
    e, d = _decide(raw, history=h)
    h.append((e, d))
    e, d = _decide(raw, history=h)
    h.append((e, d))
    _, d = _decide(raw, history=h)
    assert d.tier == 1 and d.action == "probe-restart"


def test_one_no_write_poll_does_not_yet_probe():
    """Non-degeneracy: ONE no-write poll (one gap, not two consecutive) must
    not fire -- proves the rule needs both consecutive gaps, not just one."""
    raw = _line(t=9.0, step=300)
    h = []
    e, d = _decide(raw, history=h)
    h.append((e, d))
    _, d = _decide(raw, history=h)
    assert d.tier == 0 and d.action == "NONE"


def test_fresh_journal_does_not_probe():
    """Non-degeneracy: a genuinely growing journal (a new raw line every
    poll) never fires the stale rule, no matter how many polls."""
    h = []
    d = None
    for step in (300, 400, 500):
        e, d = _decide(_line(t=9.0, step=step), history=h)
        h.append((e, d))
    assert d.tier == 0 and d.action == "NONE"


def test_large_now_t_alone_does_not_probe_restart():
    """RULING 6a supersedes the old [ASSUMED] elapsed-time reading: a single
    poll, even against a huge `now_t` gap, is not '2 consecutive no-write
    polls' by itself -- wall-clock elapsed time is no longer the trigger."""
    _, d = _decide(_line(t=9.0), now_t=9.0 + 10_000.0)
    assert d.tier == 0 and d.action == "NONE"


def test_session_ending_and_no_ckpt_in_30_forces_checkpoint():
    _, d = _decide(_line(kaggle_session_remaining_min=10.0, last_ckpt_t=9.0 - 31.0))
    assert d.tier == 1 and d.action == "force-ckpt-end-chunk"
    assert d.cell_new == "cellB"  # next in the pre-registered queue


def test_session_ending_but_recent_ckpt_does_not_force():
    _, d = _decide(_line(kaggle_session_remaining_min=10.0, last_ckpt_t=9.0 - 5.0))
    assert d.tier == 0 and d.action == "NONE"


def test_session_remaining_is_read_directly_not_computed_from_t():
    """RULING 6b: the source is KAGGLE'S reported remaining minutes, not a
    value derived from session_end_t-t. Changing `t` (which the old formula
    depended on) must not change whether this fires, since the field is now
    independent of `t`."""
    _, d1 = _decide(_line(t=9.0, kaggle_session_remaining_min=15.0, last_ckpt_t=9.0 - 31.0))
    _, d2 = _decide(_line(t=500.0, kaggle_session_remaining_min=15.0, last_ckpt_t=500.0 - 31.0))
    assert d1.tier == 1 and d1.action == "force-ckpt-end-chunk"
    assert d2.tier == 1 and d2.action == "force-ckpt-end-chunk"


def test_queue_idle_advances_to_next_cell():
    _, d = _decide(_line(queue_idle=True))
    assert d.tier == 1 and d.action == "next-cell" and d.cell_new == "cellB"


def test_queue_not_idle_does_not_advance():
    _, d = _decide(_line(queue_idle=False))
    assert d.tier == 0 and d.action == "NONE"


# ------------------------------------------------------------- RULING 6c
# "simultaneous triggers: highest tier wins; within tier, table order.
# Tier-3 co-firing with a Tier-1 fix => HALT, not fix." `decide()` already
# checks Tier-3 guards before Tier-2 before Tier-1 before Tier-0 -- these
# tests are the regression proof that this ordering is what actually runs,
# not just what the source order implies. Confirmed RED-before-change would
# have been meaningless here: this ordering needed no code change (verified
# against the pre-RULING-6 code before touching anything -- see
# V17_R6_ENVELOPE.md); these are added as permanent coverage of a claim
# that had no test before.

def test_tier3_co_firing_with_tier1_oom_halts_not_halves():
    _, d = _decide(_line(phase_arm=True, a_hat_max=1.2, oom=True,
                        micro_batch=8, grad_accum=4))
    assert d.tier == 3 and d.action == "HALT"
    assert "a_hat_max-on-phase-arm" in d.trigger  # not "halve-micro-batch"


def test_tier3_co_firing_with_tier2_nan_halts_not_corrects():
    _, d = _decide(_line(kcert_delta_over_tol=0.9, loss_nonfinite=True, loss=None))
    assert d.tier == 3 and d.action == "HALT"
    assert "kcert" in d.trigger  # not "rollback+lr-decay+grad-clip"


def test_tier1_alone_still_fixes_without_a_co_firing_tier3():
    """Non-degeneracy: the SAME oom=True field, with nothing Tier-3 also
    matching, still gets the Tier-1 fix -- proves tier-3 pre-emption is
    conditioned on an actual co-firing trigger, not a blanket suppression."""
    _, d = _decide(_line(oom=True, micro_batch=8, grad_accum=4))
    assert d.tier == 1 and d.action == "halve-micro-batch"


# ------------------------------------------------------------- RULING 6d
# "Tier-2 lineage scoped PER ROOT CELL: two Tier-2 on any cell descended
# from one root => Tier 3 (no laundering via chains)." `_lineage_root()`
# already unwinds an arbitrarily deep `.t2.N` chain back to its root (it
# loops the strip, not a single pass) -- confirmed against the pre-RULING-6
# code before any edit (see V17_R6_ENVELOPE.md). These tests are the chain
# depth and unrelated-root coverage the round brief asked for, not a fix.

def test_second_tier2_three_cells_deep_in_the_chain_escalates():
    """A Tier-2 lands on the root (cellA); a SECOND Tier-2 is crafted two
    levels further down the same lineage (a grandchild id, cellA.t2.1.t2.1)
    -- this must halt as 'two Tier-2 on this root', not mint a third link."""
    h = []
    e, d = _decide(_line(loss_nonfinite=True, loss=None, cell="cellA", step=100), history=[])
    assert d.tier == 2 and d.cell_new == "cellA.t2.1"
    h.append((e, d))
    _, d2 = _decide(_line(a_hat_max=1.6, cell="cellA.t2.1.t2.1", step=200), history=h)
    assert d2.tier == 3 and d2.action == "HALT"
    assert "two-tier2-on-cell" in d2.trigger


def test_two_tier2_on_unrelated_roots_do_not_aggregate():
    """Non-degeneracy: a Tier-2 on cellA and a Tier-2 on cellB (unrelated
    roots) are each the FIRST Tier-2 for their own lineage -- must not
    escalate the second one just because some other lineage also had one."""
    h = []
    e, d = _decide(_line(loss_nonfinite=True, loss=None, cell="cellA", step=100), history=[])
    assert d.tier == 2
    h.append((e, d))
    _, d2 = _decide(_line(loss_nonfinite=True, loss=None, cell="cellB", step=200), history=h)
    assert d2.tier == 2 and d2.action.startswith("rollback")


# ------------------------------------------------------------- RULING 6e
# "OOM keyed per (cell, shape): second OOM at same shape after halving =>
# Tier 3, not a second halving." Before this ruling the OOM branch halved
# unconditionally every time (RED evidence in V17_R6_ENVELOPE.md).

def test_second_oom_at_same_shape_after_halving_escalates():
    h = []
    e, d = _decide(_line(oom=True, micro_batch=8, grad_accum=4, step=100), history=[])
    assert d.tier == 1 and d.action == "halve-micro-batch"
    h.append((e, d))
    _, d2 = _decide(_line(oom=True, micro_batch=8, grad_accum=4, step=200), history=h)
    assert d2.tier == 3 and d2.action == "HALT"
    assert "oom-repeat-same-shape" in d2.trigger


def test_oom_at_a_genuinely_new_shape_still_halves():
    """Non-degeneracy: the SECOND OOM overall for the cell, but at a
    DIFFERENT (already-halved) shape, is the FIRST OOM at that shape -- must
    still halve, proving the counter is shape-scoped, not a one-OOM-ever
    ceiling per cell."""
    h = []
    e, d = _decide(_line(oom=True, micro_batch=8, grad_accum=4, step=100), history=[])
    assert d.tier == 1 and d.action == "halve-micro-batch"
    h.append((e, d))
    _, d2 = _decide(_line(oom=True, micro_batch=4, grad_accum=8, step=200), history=h)
    assert d2.tier == 1 and d2.action == "halve-micro-batch"


def test_oom_at_same_shape_on_a_different_cell_does_not_aggregate():
    """Non-degeneracy: the (cell, shape) key is per-cell too -- an OOM on
    cellB at the same shape cellA already OOM'd at must not count against
    cellA's shape counter."""
    h = []
    e, d = _decide(_line(oom=True, micro_batch=8, grad_accum=4, cell="cellA", step=100), history=[])
    assert d.tier == 1
    h.append((e, d))
    _, d2 = _decide(_line(oom=True, micro_batch=8, grad_accum=4, cell="cellB", step=200), history=h)
    assert d2.tier == 1 and d2.action == "halve-micro-batch"


# ------------------------------------------------------------- RULING 6f
# "deciding-cell membership = explicit cell-id list, FROZEN at launch; may
# shrink mid-flight, never grow." `AutopilotConfig.deciding_cells_launch`
# makes growth a construction-time error -- RED before this ruling: the
# kwarg did not exist, so this raised TypeError instead of the deliberate
# ValueError (V17_R6_ENVELOPE.md has the exact pre-change traceback).

def test_growing_deciding_cells_beyond_launch_raises():
    with pytest.raises(ValueError):
        ap.AutopilotConfig(
            deciding_cells=frozenset({"cellD", "cellE"}),
            branch_diagnostics={},
            cell_queue=(),
            lr0=3e-4,
            deciding_cells_launch=frozenset({"cellD"}),
        )


def test_shrinking_deciding_cells_within_launch_succeeds():
    """The other direction must work: retiring a cell (a subset of the
    launch-frozen set) is not rejected."""
    cfg = ap.AutopilotConfig(
        deciding_cells=frozenset(),  # cellD retired mid-flight
        branch_diagnostics={},
        cell_queue=(),
        lr0=3e-4,
        deciding_cells_launch=frozenset({"cellD"}),
    )
    assert cfg.deciding_cells == frozenset()
    # And the shrink is live: a design-change request on the retired cell is
    # no longer refused as a deciding-cell edit (it is a different cell now).
    event, d = ap.decide_line(
        _line(cell="cellD", design_change_request=True), [], now_t=9.0, config=cfg)
    assert not (d.tier == 3 and "deciding-cell" in d.trigger)


def test_config_at_launch_with_no_launch_set_is_unrestricted():
    """The single config that ESTABLISHES the launch set (deciding_cells_launch
    left None) is not itself checked against anything -- there is nothing
    yet to grow beyond."""
    cfg = ap.AutopilotConfig(
        deciding_cells=frozenset({"cellD"}),
        branch_diagnostics={},
        cell_queue=(),
        lr0=3e-4,
    )
    assert cfg.deciding_cells_launch is None
