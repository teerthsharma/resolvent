"""G0.2 K-RESUME: interrupt at k, resume, bitwise-equal at k+m.

WHAT `tests/chase/test_resume_checkpoint.py` ALREADY COVERS, so this file does
not repeat it: the stitched loss sequence within 1e-5 relative, the MODEL
PARAMETERS bitwise, and one planted negative (a no-op'd
`AdamW.load_state_dict`). Three of the contract's four components are NOT
covered there -- optimizer state, torch RNG state, data-generator state -- and
neither is the claim in `train()`'s docstring that "the generator's state IS
the dataloader position", which is an argument and not a measurement.

k = 2 and m = 3 deliberately differ. At k == m an optimizer whose `step`
counter were reset on resume would still land on the right total and the
comparison would not notice.

CPU, hidden 16 / L1 / H2 / seq 8 / batch 2 -- the shape
`tests/chase/test_resume_checkpoint.py:21` already trains, 11,488 parameters.
This is a bitwise-equality check and not a performance check, so the shape only
has to exercise the code path; and the one 8 GB card on this box is shared.
"""
import torch

from tests.gate0.helpers import bitwise_diff, load_state
from ceq.hf import train as T

torch.set_num_threads(1)  # CPU matmul reduction order must not vary run to run

SHAPE = dict(hidden_size=16, n_layers=1, n_heads=2, seq=8, batch=2, vocab_size=256)
K, M = 2, 3

COMPONENTS = ("model", "optimizer", "torch_rng_state", "data_gen_state")


def _split_run(tmp_path, resume=True):
    """Uninterrupted K+M into `ref`, and K-then-M into `a` then `b`."""
    T.train(out_dir=str(tmp_path / "ref"), steps=K + M, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "a"), steps=K, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "b"), steps=M, device="cpu", log_every=0,
            resume_from=str(tmp_path / "a") if resume else None, **SHAPE)
    return load_state(str(tmp_path / "ref")), load_state(str(tmp_path / "b"))


# --------------------------------------------------------------- the PASS half

def test_resume_is_bitwise_on_all_four_components(tmp_path):
    ref, got = _split_run(tmp_path)
    assert int(got["step"]) == K + M
    diff = bitwise_diff(ref, got, "state")
    assert not diff, "resumed state is NOT bitwise equal:\n  " + "\n  ".join(diff)


def test_each_component_is_actually_present_and_nonempty(tmp_path):
    """A comparison over four names that resolve to nothing is a vacuous pass."""
    T.train(out_dir=str(tmp_path / "r"), steps=K, device="cpu", log_every=0, **SHAPE)
    s = load_state(str(tmp_path / "r"))
    assert set(COMPONENTS) <= set(s)
    assert len(s["model"]) == 15, len(s["model"])
    assert set(s["optimizer"]["state"][0]) == {"exp_avg", "exp_avg_sq", "step"}
    assert len(s["optimizer"]["state"]) == len(s["model"])
    assert s["torch_rng_state"].numel() == 5056
    assert s["data_gen_state"].numel() == 5056


# ------------------------------------------------------ non-degeneracy: 1 ulp
#
# This repo has struck 14 vacuous controls. The PASS above is void unless the
# comparator is shown to catch the SMALLEST difference representable in the
# dtype it compares -- not a difference someone chose to be large enough.

def test_comparator_catches_a_one_ulp_perturbation(tmp_path):
    T.train(out_dir=str(tmp_path / "r"), steps=K, device="cpu", log_every=0, **SHAPE)
    a = load_state(str(tmp_path / "r"))
    b = load_state(str(tmp_path / "r"))
    assert not bitwise_diff(a, b), "two reads of one checkpoint already differ"

    t = b["model"]["model.embed_tokens.weight"]
    old = t.flatten()[0].clone()
    nudged = torch.nextafter(old, torch.tensor(float("inf"), dtype=t.dtype))
    assert nudged != old
    # one ulp and no more: the gap is at most 2**-23 relative for float32
    assert float((nudged.double() - old.double()).abs()) <= 2.0 ** -23 * max(
        abs(float(old)), 2.0 ** -126)
    with torch.no_grad():
        t.view(-1)[0] = nudged

    diff = bitwise_diff(a, b, "state")
    assert len(diff) == 1 and "model.embed_tokens.weight" in diff[0], diff


def test_comparator_catches_a_one_ulp_perturbation_in_the_optimizer(tmp_path):
    """The same calibration on a component the model-only check cannot see."""
    T.train(out_dir=str(tmp_path / "r"), steps=K, device="cpu", log_every=0, **SHAPE)
    a = load_state(str(tmp_path / "r"))
    b = load_state(str(tmp_path / "r"))
    t = b["optimizer"]["state"][0]["exp_avg"]
    old = t.view(-1)[0].clone()
    with torch.no_grad():
        t.view(-1)[0] = torch.nextafter(
            old, torch.tensor(float("inf"), dtype=t.dtype))
    diff = bitwise_diff(a, b, "state")
    assert len(diff) == 1 and "optimizer.state.0.exp_avg" in diff[0], diff


def test_comparator_catches_a_one_byte_flip_in_the_rng_state(tmp_path):
    """And on the two uint8 buffers, where `nextafter` has no meaning."""
    T.train(out_dir=str(tmp_path / "r"), steps=K, device="cpu", log_every=0, **SHAPE)
    for comp in ("torch_rng_state", "data_gen_state"):
        a = load_state(str(tmp_path / "r"))
        b = load_state(str(tmp_path / "r"))
        b[comp][0] = (int(b[comp][0]) + 1) % 256
        diff = bitwise_diff(a, b, "state")
        assert len(diff) == 1 and comp in diff[0], (comp, diff)


# ---------------------------------------------------- the dataloader MEASURED
#
# `train()`'s docstring argues "batches are drawn by index from `gen`, so the
# generator's state IS the dataloader position". That is an argument. This
# records the actual input tensors and compares the batches the resumed run
# draws against the batches the uninterrupted run drew at the same steps.

def test_resumed_run_draws_the_same_batches_the_uninterrupted_run_drew(
        tmp_path, monkeypatch):
    seen = {}
    real = T.ByteBatches.batch

    def recording(self, split, bs, seq, gen, device):
        x, y = real(self, split, bs, seq, gen, device)
        seen.setdefault(recording.tag, []).append(x.clone())
        return x, y

    monkeypatch.setattr(T.ByteBatches, "batch", recording)

    recording.tag = "ref"
    T.train(out_dir=str(tmp_path / "ref"), steps=K + M, device="cpu",
            log_every=0, **SHAPE)
    recording.tag = "a"
    T.train(out_dir=str(tmp_path / "a"), steps=K, device="cpu",
            log_every=0, **SHAPE)
    recording.tag = "b"
    T.train(out_dir=str(tmp_path / "b"), steps=M, device="cpu", log_every=0,
            resume_from=str(tmp_path / "a"), **SHAPE)

    assert len(seen["ref"]) == K + M and len(seen["b"]) == M
    for i in range(M):
        assert torch.equal(seen["ref"][K + i], seen["b"][i]), (
            "resumed step {} drew a different batch than uninterrupted step {}"
            .format(i, K + i))
    # the batches differ from each other, so the loop above is not free
    assert not torch.equal(seen["ref"][0], seen["ref"][1])


# ------------------------------------------------------- the planted negatives

def _ref_and_broken(tmp_path, monkeypatch, break_it):
    """Train the reference and the K-chunk clean, break ONLY the resume.

    `monkeypatch.undo()` before the states are read back is load-bearing and was
    added after it bit: the injection for negative (c) patches `torch.load`,
    which is also how `load_state` reads a checkpoint off disk, so leaving it in
    place rewrote BOTH sides of the comparison identically and the negative
    reported itself as not firing. An injection that is still live during the
    measurement is measuring the injection.
    """
    T.train(out_dir=str(tmp_path / "ref"), steps=K + M, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "a"), steps=K, device="cpu",
            log_every=0, **SHAPE)
    break_it()
    T.train(out_dir=str(tmp_path / "bad"), steps=M, device="cpu", log_every=0,
            resume_from=str(tmp_path / "a"), **SHAPE)
    monkeypatch.undo()
    return load_state(str(tmp_path / "ref")), load_state(str(tmp_path / "bad"))


def _components_that_differ(ref, bad):
    # paths look like "state.optimizer.state.0.exp_avg  [max |delta| = ...]"
    return sorted({d.split(".")[1].split(" ")[0]
                   for d in bitwise_diff(ref, bad, "state")})


def test_negative_a_optimizer_load_state_dict_noop(tmp_path, monkeypatch):
    """The negative the contract names explicitly."""
    ref, bad = _ref_and_broken(tmp_path, monkeypatch, lambda: monkeypatch.setattr(
        torch.optim.AdamW, "load_state_dict", lambda self, sd: None))
    hit = _components_that_differ(ref, bad)
    assert "model" in hit and "optimizer" in hit, hit


def test_negative_b_torch_rng_state_not_restored(tmp_path, monkeypatch):
    """No-op `torch.set_rng_state`, with a KNOWN-different ambient state.

    Seeding to 987 first is what makes this a control rather than a coin flip:
    without it the ambient global RNG state at resume time is whatever the test
    session happened to leave behind, and a negative that fires by luck is not
    a negative.
    """
    def brk():
        torch.manual_seed(987)
        monkeypatch.setattr(torch, "set_rng_state", lambda s: None)

    ref, bad = _ref_and_broken(tmp_path, monkeypatch, brk)
    hit = _components_that_differ(ref, bad)
    assert "torch_rng_state" in hit, hit


def test_negative_c_data_generator_state_not_restored(tmp_path, monkeypatch):
    """`torch.Generator` is an immutable C type, so `set_state` cannot be
    no-op'd the way `AdamW.load_state_dict` can. The injection goes in one level
    earlier instead: `torch.load` hands back the state the generator would have
    had if the restore had never happened -- a fresh `manual_seed(seed + 1)`,
    which is exactly where `train()` leaves `gen` before the resume branch.
    """
    real_load = torch.load

    def stale(*a, **kw):
        ck = real_load(*a, **kw)
        if isinstance(ck, dict) and "data_gen_state" in ck:
            ck["data_gen_state"] = torch.Generator().manual_seed(0 + 1).get_state()
        return ck

    ref, bad = _ref_and_broken(
        tmp_path, monkeypatch,
        lambda: monkeypatch.setattr(T.torch, "load", stale))
    hit = _components_that_differ(ref, bad)
    assert "model" in hit and "data_gen_state" in hit, hit


def test_no_resume_at_all_differs(tmp_path):
    """The floor control: a run that never resumed must not match."""
    ref, got = _split_run(tmp_path, resume=False)
    assert bitwise_diff(ref, got), "a run that never resumed matched the reference"
