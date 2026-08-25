"""PROOF that `train()` is resumable, cpu-only and tiny.

`train()` used to `save_pretrained` at the end and nothing else: no optimizer
state, no step counter, no RNG state, no load path. Every free GPU tier has a
session cap, so without resume the largest trainable model was whatever fit
ONE uninterrupted session -- a capability ceiling, not a convenience gap.

This test is the falsifier: train N steps, "kill" (just stop calling train),
resume from `out_dir`, train N more, and check the stitched loss sequence
against one uninterrupted 2N-step run bit-for-bit (within float tolerance).
No GPU, no CUDA import, well under 60s.
"""
import os

import torch

from ceq.hf import train as T

torch.set_num_threads(1)  # CPU matmul reduction order must not vary run to run

SHAPE = dict(hidden_size=16, n_layers=1, n_heads=2, seq=8, batch=2, vocab_size=256)
N = 3


def test_split_run_matches_uninterrupted_run(tmp_path):
    ref = T.train(out_dir=str(tmp_path / "ref"), steps=2 * N, device="cpu",
                 log_every=0, **SHAPE)

    part_a = T.train(out_dir=str(tmp_path / "a"), steps=N, device="cpu",
                     log_every=0, **SHAPE)
    assert os.path.exists(tmp_path / "a" / "trainer_state.pt")
    part_b = T.train(out_dir=str(tmp_path / "b"), steps=N, device="cpu",
                     log_every=0, resume_from=str(tmp_path / "a"), **SHAPE)
    assert part_b["start_step"] == N

    stitched = part_a["losses"] + part_b["losses"]
    assert len(stitched) == len(ref["losses"]) == 2 * N

    for i, (got, want) in enumerate(zip(stitched, ref["losses"])):
        assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (i, got, want)


# ==========================================================================
# THE BITWISE END — added iteration 39
# ==========================================================================
#
# The test above compares LOSSES within `1e-5` relative. That is the tolerance
# end. It is not enough on its own for the run this project actually needs:
# a 12-hour Colab session resumed across a session cap, where a small divergence
# in optimizer moments compounds for thousands of steps while every individual
# loss still agrees to five places.
#
# This project has two instruments that have never once cried wolf — the
# calibration gate and the bitwise replay in `scale/bucket.py` — against six
# structure-by-regex instruments that have. Both survivors compare VALUES with
# no tolerance. So resume gets the same treatment: every parameter, bit for bit.

def test_resumed_weights_are_bitwise_identical(tmp_path):
    """Stitched run must reproduce the uninterrupted run's WEIGHTS exactly."""
    from ceq.hf.modeling_ceq import CEQForCausalLM

    T.train(out_dir=str(tmp_path / "ref"), steps=2 * N, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "a"), steps=N, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "b"), steps=N, device="cpu",
            log_every=0, resume_from=str(tmp_path / "a"), **SHAPE)

    ref = CEQForCausalLM.from_pretrained(str(tmp_path / "ref"))
    got = CEQForCausalLM.from_pretrained(str(tmp_path / "b"))
    rp, gp = dict(ref.named_parameters()), dict(got.named_parameters())
    assert set(rp) == set(gp)

    drift = [(n, float((rp[n] - gp[n]).abs().max())) for n in sorted(rp)
             if not torch.equal(rp[n], gp[n])]
    assert not drift, (
        "resumed weights are NOT bitwise identical to the uninterrupted run:\n"
        + "\n".join(f"  {n}: max |delta| = {d:.3e}" for n, d in drift)
        + "\nA divergence this small still compounds across a 12-hour resumed run."
    )


def test_the_bitwise_check_fires_when_resume_is_broken(tmp_path, monkeypatch):
    """Calibration. A control that cannot fail is not a control (instrument #15).

    Drops the optimizer moments on reload — the single most plausible resume
    bug, and exactly what `train()` did before `trainer_state.pt` existed — and
    requires the bitwise check to notice. Without this, a green above could mean
    the comparison is blind rather than the resume being correct.
    """
    from ceq.hf.modeling_ceq import CEQForCausalLM

    T.train(out_dir=str(tmp_path / "ref"), steps=2 * N, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "a"), steps=N, device="cpu",
            log_every=0, **SHAPE)

    real = torch.optim.AdamW.load_state_dict
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict",
                        lambda self, sd: None)      # moments silently dropped
    T.train(out_dir=str(tmp_path / "broken"), steps=N, device="cpu",
            log_every=0, resume_from=str(tmp_path / "a"), **SHAPE)
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", real)

    ref = CEQForCausalLM.from_pretrained(str(tmp_path / "ref"))
    bad = CEQForCausalLM.from_pretrained(str(tmp_path / "broken"))
    rp, bp = dict(ref.named_parameters()), dict(bad.named_parameters())
    assert any(not torch.equal(rp[n], bp[n]) for n in rp), (
        "dropping the optimizer moments produced BITWISE IDENTICAL weights. "
        "Either the comparison is blind or the optimizer state is not "
        "load-bearing — - both mean the test above proves nothing."
    )
