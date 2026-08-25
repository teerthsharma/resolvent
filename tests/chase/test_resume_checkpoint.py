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
