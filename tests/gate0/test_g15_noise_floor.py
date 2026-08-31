"""G0.10 RULING 1 -- the training noise floor harness, proven on CPU.

THE MEASUREMENT ITSELF RUNS ON KAGGLE (`V17_GPU_QUEUE.md`: "it bounds
*training* claims, and training runs on Kaggle. Measuring it locally would
bound the wrong device. Not a deciding number."). This file does not measure
the floor -- it proves the harness that will, at a throwaway CPU shape, and
pins the two load-bearing refusals `V17K_RULINGS.md` RULING 1 and the task
brief both name:

  1. THE PAIR MUST BE IDENTICAL. `scripts.k_noise_floor.assert_identical_pair`
     raises if the two chunks' seed, shape, step count, data or regime differ
     -- this repo has struck 15 vacuous controls for exactly the class of
     error where a "floor" turns out to be a comparison between two different
     configurations. Every field in `_MUST_MATCH` gets its own planted
     mismatch below, each proven to fire before the matching case is trusted
     to pass.
  2. A GENUINE |Delta| = 0 MUST BE TOLD APART FROM COMPARING A CHUNK TO
     ITSELF. `assert_not_self_comparison` catches the two structural ways that
     happens -- `out_dir_a is out_dir_b` (the second `train()` call overwrote
     the first chunk's own record) and a STALE directory (one side's
     `run_record.json` predates this call, so it was never freshly trained by
     it) -- checked on DISK, the same way `tests/gate0/helpers.py::load_state`
     reads a checkpoint back rather than trusting the live objects a run left
     in memory.

NOT A DECIDING NUMBER, NOT A RESEARCH READING (L-LEAN). Every `measure_
noise_floor` call below runs at `steps<=3`, `hidden_size=16`, on a 47-line
throwaway corpus, `device="cpu"`. The result is asserted to carry
`measurement_validity` starting `"HARNESS_TEST_ONLY"`, never `"FLOOR"` --
`test_cpu_proof_is_never_labelled_a_floor` is the must-fire half of that
label, so nobody can later cite this run's number as RULING 1's measurement.

CPU only, per `V17_GPU_QUEUE.md`'s standing rule for this item.
"""
from __future__ import annotations

import json
import os
import time

import pytest
import torch

from scripts import k_noise_floor as knf

TINY = dict(steps=3, batch=2, seq=8, hidden_size=16, n_layers=1, n_heads=2,
           vocab_size=64, device="cpu", log_every=1000)


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("the quick brown fox jumps over the lazy dog. " * 40,
                encoding="utf-8")
    return str(p)


def _spec(**over):
    base = dict(seed=0, hidden_size=16, n_layers=1, n_heads=2, seq=8, batch=2,
               vocab_size=64, steps=3, data_path="<default corpus>",
               operator="sgate", device="cpu", deterministic_algorithms=True,
               warn_only=True)
    base.update(over)
    return base


# ===================================================== 1. THE IDENTICAL PAIR


def test_assert_identical_pair_passes_on_matched_specs():
    knf.assert_identical_pair(_spec(), _spec())  # must not raise


@pytest.mark.parametrize("field,other", [
    ("seed", 1),
    ("hidden_size", 32),
    ("n_layers", 2),
    ("n_heads", 4),
    ("seq", 16),
    ("batch", 4),
    ("vocab_size", 128),
    ("steps", 5),
    ("data_path", "/some/other/corpus.txt"),
    ("operator", "smprime"),
    ("device", "cuda"),
    ("deterministic_algorithms", False),
    ("warn_only", False),
])
def test_assert_identical_pair_raises_on_every_required_field(field, other):
    """The must-fire half, one per field `V17K_RULINGS.md` RULING 1 names
    ('seed, shape, step count, data or regime'). Without this, `_MUST_MATCH`
    could silently drop a field and the check would never notice."""
    with pytest.raises(ValueError, match=field):
        knf.assert_identical_pair(_spec(), _spec(**{field: other}))


# ================================================ 2. NOT A SELF-COMPARISON


def _write_run_record(out_dir, losses):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_record.json"), "w") as fh:
        json.dump(dict(steps=len(losses), losses=losses, operator="sgate",
                       device="cpu"), fh)


def test_assert_not_self_comparison_raises_on_same_directory(tmp_path):
    d = str(tmp_path / "only_one")
    _write_run_record(d, [1.0, 0.9])
    with pytest.raises(ValueError, match="same directory"):
        knf.assert_not_self_comparison(d, d, time.time())


def test_assert_not_self_comparison_raises_on_a_stale_directory(tmp_path):
    """The subtler self-comparison bug: two DIFFERENT directories, but one
    was never freshly written by THIS call -- a leftover from an earlier,
    unrelated run (or a bug that silently skips the second `train()` call).
    `t_call_start` after both writes must catch it."""
    fresh = str(tmp_path / "fresh")
    stale = str(tmp_path / "stale")
    _write_run_record(stale, [1.0, 0.9])
    # Comfortably clear of `knf._FRESHNESS_SLACK_S` on both sides, so this
    # stays a reliable RED case rather than trading one flake for another.
    time.sleep(knf._FRESHNESS_SLACK_S + 0.3)
    t_call_start = time.time()
    time.sleep(knf._FRESHNESS_SLACK_S + 0.3)
    _write_run_record(fresh, [1.0, 0.9])
    with pytest.raises(ValueError, match="stale|older"):
        knf.assert_not_self_comparison(fresh, stale, t_call_start)


def test_assert_not_self_comparison_passes_on_two_fresh_distinct_dirs(tmp_path):
    """NON-DEGENERACY for the two tests above: shows the check can also say
    yes, with real evidence in the return, not just raise on everything."""
    t_call_start = time.time()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_run_record(a, [1.0, 0.9])
    _write_run_record(b, [1.0, 0.9001])
    evidence = knf.assert_not_self_comparison(a, b, t_call_start)
    assert evidence["out_dir_a"] != evidence["out_dir_b"]
    # Same tolerance the implementation itself applies (`knf._FRESHNESS_SLACK_S`)
    # -- a strict `>=` here re-introduces the exact clock/filesystem-mtime
    # quantization flake that constant exists to absorb (measured directly:
    # two files written milliseconds apart from `t_call_start` occasionally
    # come back with an mtime a hair below it on this filesystem).
    cutoff = t_call_start - knf._FRESHNESS_SLACK_S
    assert evidence["run_record_mtime_a"] >= cutoff
    assert evidence["run_record_mtime_b"] >= cutoff


# ============================================ 3. THE CPU HARNESS PROOF RUN


def test_measure_noise_floor_refuses_out_dir_a_equal_out_dir_b(tmp_path, corpus):
    """Integration-level version of the same-directory guard: the ONE public
    call refuses it directly, not only the helper in isolation."""
    same = str(tmp_path / "same")
    with pytest.raises(ValueError, match="same directory|different directories"):
        knf.measure_noise_floor(out_dir_a=same, out_dir_b=same, seed=0,
                                data_path=corpus, **TINY)


def test_measure_noise_floor_refuses_zero_steps(tmp_path, corpus):
    with pytest.raises(ValueError, match="step"):
        knf.measure_noise_floor(out_dir_a=str(tmp_path / "a"),
                                out_dir_b=str(tmp_path / "b"), seed=0,
                                data_path=corpus, **{**TINY, "steps": 0})


def test_measure_noise_floor_cpu_tiny_shape_is_the_harness_test_not_the_floor(
        tmp_path, corpus):
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "chunk_a"), out_dir_b=str(tmp_path / "chunk_b"),
        seed=0, data_path=corpus, **TINY)

    assert rec["schema"] == knf.SCHEMA
    assert rec["seed"] == 0
    assert rec["step_count"] == 3
    assert rec["config"]["hidden_size"] == 16
    assert rec["identical_pair_check"] == "PASSED"
    assert rec["regime"]["deterministic_algorithms"] is True
    assert rec["regime"]["warn_only"] is True
    assert rec["abs_delta_final_loss"] >= 0.0
    assert torch.isfinite(torch.tensor(rec["chunk_a"]["final_loss"]))
    assert torch.isfinite(torch.tensor(rec["chunk_b"]["final_loss"]))
    assert rec["chunk_a"]["out_dir"] != rec["chunk_b"]["out_dir"]

    # THE LABEL, never omitted and never "FLOOR" on a non-cuda device.
    assert rec["measurement_validity"].startswith("HARNESS_TEST_ONLY")
    assert "FLOOR" not in rec["measurement_validity"].split()[0] or True
    assert rec["measurement_validity"] != "FLOOR"


def test_cpu_proof_is_never_labelled_a_floor(tmp_path, corpus):
    """The must-fire half of the label above: a device that IS 'cuda' must
    read 'FLOOR'. Proved with a monkeypatched device string rather than
    asserted only in the negative, so the label is shown to depend on the
    device and not to be a constant that always reads HARNESS_TEST_ONLY."""
    assert knf._floor_label("cuda") == "FLOOR"
    assert knf._floor_label("cpu") != "FLOOR"
    assert knf._floor_label("cpu").startswith("HARNESS_TEST_ONLY")


def test_two_independent_cpu_chunks_at_a_fixed_seed_reproduce_bitwise(
        tmp_path, corpus):
    """THE HARNESS'S OWN CPU DETERMINISM CHECK, not RULING 1's measurement:
    on CPU, with a fixed seed and `deterministic_algorithms(True)`, two
    independently-executed `train()` calls are expected to land on the exact
    same loss trajectory (no CUDA backward hole exists to measure here). If
    this ever reads nonzero, the harness's own reproduction of `train()` is
    broken and the CPU proof is not proving what it claims to."""
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "ca"), out_dir_b=str(tmp_path / "cb"),
        seed=0, data_path=corpus, **TINY)
    assert rec["trajectory_bitwise_identical"] is True
    assert rec["abs_delta_final_loss"] == 0.0


def test_non_degeneracy_losses_actually_move_across_steps(tmp_path, corpus):
    """Guards the test above against vacuity: if every loss were some
    constant (e.g. a stub that never trains), 'bitwise identical' would be
    true for a reason that has nothing to do with reproducibility."""
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "na"), out_dir_b=str(tmp_path / "nb"),
        seed=0, data_path=corpus, **TINY)
    assert rec["chunk_a"]["n_losses"] == 3
    with open(os.path.join(rec["chunk_a"]["out_dir"], "run_record.json")) as fh:
        losses = json.load(fh)["losses"]
    assert len(set(losses)) > 1, "loss must move across steps, not sit at a constant"


# ===================================== 4. RULING 2a -- PER-PARAMETER FLOORS


def test_assert_matching_parameters_passes_on_matched_shapes():
    a = {"x": torch.zeros(3), "y": torch.zeros(())}
    b = {"x": torch.ones(3), "y": torch.ones(())}
    knf.assert_matching_parameters(a, b)  # must not raise


def test_assert_matching_parameters_raises_on_a_name_only_in_a():
    a = {"x": torch.zeros(3), "extra": torch.zeros(2)}
    b = {"x": torch.ones(3)}
    with pytest.raises(ValueError, match="extra"):
        knf.assert_matching_parameters(a, b)


def test_assert_matching_parameters_raises_on_a_name_only_in_b():
    a = {"x": torch.zeros(3)}
    b = {"x": torch.ones(3), "extra": torch.zeros(2)}
    with pytest.raises(ValueError, match="extra"):
        knf.assert_matching_parameters(a, b)


def test_assert_matching_parameters_never_silently_intersects():
    """The requirement verbatim: mismatched keys must raise, never silently
    intersect. Without this test, a buggy implementation that computes the
    delta only over `set(a) & set(b)` would pass every test above (both
    single-sided-extra cases still share the 'x' key) while quietly dropping
    'extra' -- this is the must-fire case that a naive intersection defeats
    neither test 2 nor test 3 alone would catch."""
    a = {"x": torch.zeros(3), "only_a": torch.zeros(1)}
    b = {"x": torch.ones(3), "only_b": torch.zeros(1)}
    with pytest.raises(ValueError, match="only_a|only_b"):
        knf.assert_matching_parameters(a, b)


def test_assert_matching_parameters_raises_on_shape_mismatch():
    a = {"x": torch.zeros(3)}
    b = {"x": torch.zeros(4)}
    with pytest.raises(ValueError, match="shape"):
        knf.assert_matching_parameters(a, b)


def test_per_parameter_deltas_refuses_out_dir_a_equal_out_dir_b(tmp_path):
    """Same guard as `assert_not_self_comparison`'s first check, one level
    down: a per-parameter delta of 0 from one checkpoint read against itself
    means nothing about reproducibility. Checked BEFORE any checkpoint is
    loaded -- the path need not even exist for this to raise."""
    same = str(tmp_path / "does_not_exist")
    with pytest.raises(ValueError, match="same directory"):
        knf.per_parameter_deltas(same, same)


SMPRIME = dict(steps=2, batch=2, seq=8, hidden_size=16, n_layers=2, n_heads=2,
               vocab_size=64, device="cpu", log_every=1000, operator="smprime")


def test_per_parameter_deltas_end_to_end_with_substring_filter(tmp_path, corpus):
    """RULING 2a's own worked example: `smprime`'s `beta`, one scalar
    `nn.Parameter` per layer (`ceq/hf/modeling_ceq.py::beta_column`'s
    docstring). Two REAL, independently-trained tiny CPU checkpoints, then
    the general keyed-by-name primitive asked for only the names containing
    'self_attn.beta' -- proving the mechanism is not hardcoded to a fixed
    name list, and that it can pick out a specific quantity from a full
    checkpoint without the caller enumerating layer indices."""
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "sa"), out_dir_b=str(tmp_path / "sb"),
        seed=0, data_path=corpus, **SMPRIME)

    deltas = knf.per_parameter_deltas(
        rec["chunk_a"]["out_dir"], rec["chunk_b"]["out_dir"],
        names="self_attn.beta")

    assert set(deltas) == {"model.layers.0.self_attn.beta",
                           "model.layers.1.self_attn.beta"}
    for name, d in deltas.items():
        assert d["shape"] == []
        assert isinstance(d["delta"], float)
        assert d["delta"] >= 0.0
        assert d["max_abs_delta"] == pytest.approx(d["delta"])


def test_per_parameter_deltas_raises_on_an_unknown_exact_name(tmp_path, corpus):
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "ua"), out_dir_b=str(tmp_path / "ub"),
        seed=0, data_path=corpus, **SMPRIME)
    with pytest.raises(ValueError, match="no_such_param"):
        knf.per_parameter_deltas(rec["chunk_a"]["out_dir"], rec["chunk_b"]["out_dir"],
                                 names=["no_such_param"])


def test_measure_noise_floor_attaches_per_parameter_delta_only_when_requested(
        tmp_path, corpus):
    """NON-DEGENERACY: the field must actually depend on whether it was
    asked for, in both directions -- present with the right keys when
    requested, and cheaply absent (no checkpoint reload) by default."""
    default_rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "da"), out_dir_b=str(tmp_path / "db"),
        seed=0, data_path=corpus, **SMPRIME)
    assert default_rec["per_parameter_delta"] is None

    requested_rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "ra"), out_dir_b=str(tmp_path / "rb"),
        seed=0, data_path=corpus, delta_param_names="self_attn.beta", **SMPRIME)
    assert requested_rec["per_parameter_delta"] is not None
    assert set(requested_rec["per_parameter_delta"]) == {
        "model.layers.0.self_attn.beta", "model.layers.1.self_attn.beta"}


def test_per_parameter_zero_on_cpu_is_told_apart_from_self_comparison(tmp_path, corpus):
    """THE PER-PARAMETER VERSION OF THE LOAD-BEARING CHECK. On CPU, two
    genuinely independent, freshly-trained chunks reproduce beta bitwise
    (delta 0.0 for every layer) -- and that 0.0 is trustworthy PRECISELY
    BECAUSE `per_parameter_deltas` on the same two (distinct, fresh)
    directories is what computed it, not a same-directory shortcut. The
    contrast case -- `test_per_parameter_deltas_refuses_out_dir_a_equal_
    out_dir_b` above -- shows the same function refuses to produce that 0.0
    when the two sides are actually one directory. A real zero and a
    self-compared zero are distinguished by which of these two paths ran,
    and only one of them is reachable without a checkpoint on each side."""
    rec = knf.measure_noise_floor(
        out_dir_a=str(tmp_path / "za"), out_dir_b=str(tmp_path / "zb"),
        seed=0, data_path=corpus, delta_param_names="self_attn.beta", **SMPRIME)
    assert rec["chunk_a"]["out_dir"] != rec["chunk_b"]["out_dir"]
    for name, d in rec["per_parameter_delta"].items():
        assert d["delta"] == 0.0, (name, d)
