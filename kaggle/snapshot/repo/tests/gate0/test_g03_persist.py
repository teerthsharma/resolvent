"""G0.3 K-PERSIST: atomic checkpoint writes, kill-mid-write, rotation sizing.

WHAT `tests/chase/test_resume_checkpoint.py` COVERS OF THIS: nothing. It never
kills anything; it stops calling `train()`, which is a clean shutdown and not a
crash. A Kaggle session cap is not a clean shutdown.

THE FAILURE THIS FILE IS ABOUT. `torch.save(obj, path)` TRUNCATES `path` before
it writes. A kill between the truncate and the last byte leaves a
`trainer_state.pt` that exists, is the newest file in the directory, and does
not load. `train()` writes `trainer_state.pt` AFTER `save_pretrained` returns,
so the presence of a LOADABLE `trainer_state.pt` is the only thing that
certifies a directory as resumable -- and a truncated file breaks exactly that
certificate.

CPU, the same 11,488-parameter shape as `test_g02_resume.py`, except the
rotation measurement, which must run at the real training shape and is marked
`slow`.
"""
import hashlib
import io
import os

import pytest
import torch

from tests.gate0.helpers import bitwise_diff, load_state
from ceq.hf import train as T

torch.set_num_threads(1)

SHAPE = dict(hidden_size=16, n_layers=1, n_heads=2, seq=8, batch=2, vocab_size=256)
K, M = 2, 3


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _dir_fingerprint(d):
    return {n: _sha(os.path.join(d, n)) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n))}


# ------------------------------------------------------------ crash injectors

def _crash_at_rename(monkeypatch, seen):
    """Kill AFTER the temp file is written and BEFORE the rename lands."""
    real = os.replace

    def killer(src, dst, *a, **kw):
        seen.append((str(src), str(dst)))
        if str(dst).endswith("trainer_state.pt"):
            raise OSError("simulated kill between write and rename")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", killer)


def _crash_mid_write(monkeypatch):
    """Kill PART WAY THROUGH the bytes: half a serialization, then die.

    Works against a path or a file object, so it does not encode which of the
    two `train()` currently passes to `torch.save`.
    """
    real = torch.save

    def killer(obj, f, *a, **kw):
        buf = io.BytesIO()
        real(obj, buf, *a, **kw)
        half = buf.getvalue()[: len(buf.getvalue()) // 2]
        if hasattr(f, "write"):
            f.write(half)
        else:
            with open(f, "wb") as fh:
                fh.write(half)
        raise OSError("simulated kill mid-write")

    monkeypatch.setattr(torch, "save", killer)


def _good_chunk(tmp_path):
    T.train(out_dir=str(tmp_path / "ref"), steps=K + M, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "c0"), steps=K, device="cpu",
            log_every=0, **SHAPE)
    return str(tmp_path / "c0")


def _still_resumes_bitwise(tmp_path, good):
    """The KILL clause: the last good checkpoint must resume BITWISE."""
    T.train(out_dir=str(tmp_path / "after"), steps=M, device="cpu", log_every=0,
            resume_from=good, **SHAPE)
    diff = bitwise_diff(load_state(str(tmp_path / "ref")),
                        load_state(str(tmp_path / "after")), "state")
    assert not diff, ("the last good checkpoint no longer resumes bitwise:\n  "
                      + "\n  ".join(diff))


# ------------------------------------------------------------- the atomic path

def test_trainer_state_is_written_through_a_rename(tmp_path):
    """`torch.save` straight onto the target truncates it. Rename or nothing."""
    seen = []
    real = os.replace

    def watcher(src, dst, *a, **kw):
        seen.append((str(src), str(dst)))
        return real(src, dst, *a, **kw)

    import unittest.mock as mock
    with mock.patch.object(os, "replace", watcher):
        T.train(out_dir=str(tmp_path / "c0"), steps=K, device="cpu",
                log_every=0, **SHAPE)

    renames = [(s, d) for s, d in seen if d.endswith("trainer_state.pt")]
    assert renames, ("trainer_state.pt was not written through os.replace; "
                     "seen renames: {}".format(seen))
    src, dst = renames[-1]
    # os.replace is atomic only WITHIN a filesystem, so the temp file has to be
    # a sibling of the target and not in the system temp directory.
    assert os.path.dirname(os.path.abspath(src)) == os.path.dirname(
        os.path.abspath(dst)), (src, dst)


def test_atomic_write_leaves_an_existing_target_byte_identical_on_failure(tmp_path):
    """The helper on its own: crash at the rename, target untouched."""
    p = str(tmp_path / "trainer_state.pt")
    T._atomic_torch_save(dict(v=torch.arange(8)), p)
    before = _sha(p)

    import unittest.mock as mock
    with mock.patch.object(os, "replace", side_effect=OSError("kill")):
        with pytest.raises(OSError):
            T._atomic_torch_save(dict(v=torch.zeros(8)), p)
    assert _sha(p) == before
    assert torch.equal(torch.load(p, map_location="cpu",
                                  weights_only=True)["v"], torch.arange(8))


# --------------------------------------------------------- kill-mid-write tests

def test_kill_between_write_and_rename_leaves_no_trainer_state(tmp_path, monkeypatch):
    good = _good_chunk(tmp_path)
    before = _dir_fingerprint(good)

    seen = []
    _crash_at_rename(monkeypatch, seen)
    with pytest.raises(OSError):
        T.train(out_dir=str(tmp_path / "c1"), steps=M, device="cpu", log_every=0,
                resume_from=good, **SHAPE)
    monkeypatch.undo()

    assert seen, "no rename was attempted, so nothing was killed"
    assert not os.path.exists(tmp_path / "c1" / "trainer_state.pt"), (
        "a half-committed directory carries a trainer_state.pt and will be "
        "picked up as resumable")
    assert _dir_fingerprint(good) == before, "the last good checkpoint changed"
    _still_resumes_bitwise(tmp_path, good)


def test_kill_mid_write_leaves_no_unloadable_trainer_state(tmp_path, monkeypatch):
    """The truncated-file case. `trainer_state.pt` present but unloadable is
    worse than absent: absent is a loud FileNotFoundError, present is a
    directory that looks resumable."""
    good = _good_chunk(tmp_path)
    before = _dir_fingerprint(good)

    _crash_mid_write(monkeypatch)
    with pytest.raises(OSError):
        T.train(out_dir=str(tmp_path / "c1"), steps=M, device="cpu", log_every=0,
                resume_from=good, **SHAPE)
    monkeypatch.undo()

    victim = tmp_path / "c1" / "trainer_state.pt"
    if victim.exists():
        raise AssertionError(
            "a truncated trainer_state.pt was left in {}: {} bytes, and "
            "torch.load on it raises".format(victim, victim.stat().st_size))
    assert _dir_fingerprint(good) == before
    _still_resumes_bitwise(tmp_path, good)


def test_trainer_state_presence_certifies_the_whole_directory(tmp_path, monkeypatch):
    """Kill inside `save_pretrained`: trainer_state.pt must not appear.

    This is the ordering invariant the resume-picker depends on. `train()`
    writes the model first and `trainer_state.pt` last, so a directory that has
    a loadable `trainer_state.pt` has complete model files by construction --
    no separate DONE marker is needed and none is added.
    """
    good = _good_chunk(tmp_path)

    def half_saved(self, out_dir, *a, **kw):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model.safetensors"), "wb") as fh:
            fh.write(b"\x00" * 32)          # truncated shard
        raise OSError("simulated kill inside save_pretrained")

    monkeypatch.setattr(T.CEQForCausalLM, "save_pretrained", half_saved)
    with pytest.raises(OSError):
        T.train(out_dir=str(tmp_path / "c1"), steps=M, device="cpu", log_every=0,
                resume_from=good, **SHAPE)
    monkeypatch.undo()

    assert os.path.exists(tmp_path / "c1" / "model.safetensors")
    assert not os.path.exists(tmp_path / "c1" / "trainer_state.pt")
    _still_resumes_bitwise(tmp_path, good)


def test_refuses_to_resume_into_its_own_directory(tmp_path):
    """The one path where an in-place write can eat the last good checkpoint.

    `save_pretrained` is not atomic and is not being made atomic, so resuming
    from a directory and writing back into it overwrites the model shards of
    the very state being resumed. TRAINING.md 6.6 already says `out_dir` must
    be a NEW directory each chunk; this makes that a refusal instead of a
    convention.
    """
    good = _good_chunk(tmp_path)
    before = _dir_fingerprint(good)
    with pytest.raises(ValueError, match="out_dir"):
        T.train(out_dir=good, steps=M, device="cpu", log_every=0,
                resume_from=good, **SHAPE)
    assert _dir_fingerprint(good) == before


# ------------------------------------------------------------------- rotation

@pytest.mark.slow
def test_rotation_arithmetic_at_the_real_training_shape(tmp_path, capsys):
    """Measure ONE real checkpoint at `train.DEFAULTS` and print the arithmetic.

    /kaggle/working is 20 GB (TRAINING.md:142). The rotation policy this sizes
    is the one TRAINING.md 6.6 already prescribes -- a new out_dir per chunk --
    with the last good directory kept until the next one has a LOADABLE
    trainer_state.pt, so the steady-state occupancy is two directories.
    """
    out = str(tmp_path / "real")
    T.train(out_dir=out, steps=1, device="cpu", log_every=0, **dict(T.DEFAULTS))

    sizes = {n: os.path.getsize(os.path.join(out, n))
             for n in sorted(os.listdir(out))}
    total = sum(sizes.values())
    budget = 20 * 1000 ** 3

    print("\n[MEASURED] one checkpoint at train.DEFAULTS = {}".format(dict(T.DEFAULTS)))
    for n, b in sizes.items():
        print("  {:28s} {:>13,d} B".format(n, b))
    print("  {:28s} {:>13,d} B = {:.4f} GiB".format(
        "TOTAL", total, total / 1024 ** 3))
    print("[ASSUMED] /kaggle/working budget = 20 GB = {:,d} B "
          "(TRAINING.md:142, decimal GB as the platform states it)".format(budget))
    print("[MEASURED] checkpoints that fit = floor({:,d} / {:,d}) = {:,d}".format(
        budget, total, budget // total))
    tmp = sizes["trainer_state.pt"]
    for label, n, extra in (("last good + running chunk", 2, 0),
                            ("+ the running chunk's 2 periodic slots", 4, 0),
                            ("+ one in-flight .tmp (peak)", 4, tmp)):
        b = n * total + extra
        print("[MEASURED] {:42s} = {:>15,d} B = {:.4f} GiB = {:.4f} %".format(
            label, b, b / 1024 ** 3, 100.0 * b / budget))

    assert "trainer_state.pt" in sizes and "model.safetensors" in sizes
    assert budget // total >= 4, (total, budget)


# =========================================================================
# PERIODIC CHECKPOINTS — added after the notebook and autopilot agents showed
# that an atomic write ALONE does not survive this round's failure mode.
# =========================================================================
#
# The write was atomic but it happened ONCE, at the end of a chunk, and a chunk
# is up to 11 hours. A kernel death at hour 10 costs ten GPU-hours against a
# 30 GPU-h weekly quota (TRAINING.md:142) -- three such kills is the week. And
# the autopilot's Tier-1 rule "session <20 min & no ckpt in 30 -> force ckpt"
# presumes periodic checkpoints exist and can be observed.
#
# TWO SLOTS, ALTERNATING. `{out_dir}.ckpt-a` and `{out_dir}.ckpt-b`, written
# turn and turn about. `save_pretrained` is NOT atomic, so a periodic save that
# overwrote a single slot in place would destroy the only mid-chunk checkpoint
# at exactly the moment it is being replaced. Ping-pong makes the previous slot
# a complete checkpoint at every instant, with no reaper, no globbing and a
# disk cost bounded by construction at two directories.

def _steps_on_disk(out_dir):
    return {suffix: T._ckpt_step(out_dir + suffix)
            for suffix in ("", ".ckpt-a", ".ckpt-b")}


def test_save_every_writes_periodic_checkpoints(tmp_path):
    out = str(tmp_path / "c")
    T.train(out_dir=out, steps=5, save_every=2, device="cpu", log_every=0, **SHAPE)
    # steps 2 and 4 land in the two slots; step 5 is the end-of-chunk save.
    # Slots are removed once the chunk directory is complete, so read them from
    # a run that is still "in flight" -- see the kill test below for that. Here
    # only the end-of-chunk directory must survive, at the full step count.
    assert _steps_on_disk(out)[""] == 5


def test_periodic_saving_does_not_change_the_end_state(tmp_path):
    """A checkpoint is an observation. It must not perturb what it observes."""
    T.train(out_dir=str(tmp_path / "plain"), steps=5, device="cpu",
            log_every=0, **SHAPE)
    T.train(out_dir=str(tmp_path / "periodic"), steps=5, save_every=2,
            device="cpu", log_every=0, **SHAPE)
    diff = bitwise_diff(load_state(str(tmp_path / "plain")),
                        load_state(str(tmp_path / "periodic")), "state")
    assert not diff, ("save_every changed the trajectory:\n  " + "\n  ".join(diff))


def test_resume_from_a_mid_chunk_periodic_checkpoint_is_bitwise(tmp_path, monkeypatch):
    """THE case the Kaggle run actually exercises.

    Every other resume test in this tree resumes from an end-of-chunk save. A
    session cap does not land on a chunk boundary; it lands wherever it lands,
    and what is on disk then is a PERIODIC checkpoint.
    """
    T.train(out_dir=str(tmp_path / "ref"), steps=5, device="cpu",
            log_every=0, **SHAPE)

    # die after the periodic save at step 2, before the chunk ever finishes
    out = str(tmp_path / "c")
    real = T._save_checkpoint
    calls = []

    def dying(model, opt, gen, step, d):
        real(model, opt, gen, step, d)
        calls.append((step, d))
        raise KeyboardInterrupt("simulated session cap")

    monkeypatch.setattr(T, "_save_checkpoint", dying)
    with pytest.raises(KeyboardInterrupt):
        T.train(out_dir=out, steps=5, save_every=2, device="cpu",
                log_every=0, **SHAPE)
    monkeypatch.undo()

    assert calls == [(2, out + ".ckpt-a")], calls
    mid = T.latest_checkpoint(out)
    assert mid == out + ".ckpt-a" and T._ckpt_step(mid) == 2

    T.train(out_dir=str(tmp_path / "after"), steps=3, device="cpu", log_every=0,
            resume_from=mid, **SHAPE)
    diff = bitwise_diff(load_state(str(tmp_path / "ref")),
                        load_state(str(tmp_path / "after")), "state")
    assert not diff, ("resume from a MID-CHUNK checkpoint is not bitwise:\n  "
                      + "\n  ".join(diff))


def test_slots_alternate_so_a_kill_during_one_leaves_the_other(tmp_path, monkeypatch):
    """Kill during the SECOND periodic save. The first slot must survive whole."""
    T.train(out_dir=str(tmp_path / "ref"), steps=5, device="cpu",
            log_every=0, **SHAPE)
    out = str(tmp_path / "c")

    real = T._save_checkpoint
    calls = []

    def dying(model, opt, gen, step, d):
        calls.append((step, d))
        if len(calls) == 2:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "model.safetensors"), "wb") as fh:
                fh.write(b"\x00" * 32)      # truncated shard, then the kill
            raise KeyboardInterrupt("simulated session cap")
        return real(model, opt, gen, step, d)

    monkeypatch.setattr(T, "_save_checkpoint", dying)
    with pytest.raises(KeyboardInterrupt):
        T.train(out_dir=out, steps=5, save_every=2, device="cpu",
                log_every=0, **SHAPE)
    monkeypatch.undo()

    assert [d for _, d in calls] == [out + ".ckpt-a", out + ".ckpt-b"], calls
    assert T._ckpt_step(out + ".ckpt-a") == 2
    assert T._ckpt_step(out + ".ckpt-b") is None   # half-written, not a candidate
    assert T.latest_checkpoint(out) == out + ".ckpt-a"

    T.train(out_dir=str(tmp_path / "after"), steps=3, device="cpu", log_every=0,
            resume_from=out + ".ckpt-a", **SHAPE)
    assert not bitwise_diff(load_state(str(tmp_path / "ref")),
                            load_state(str(tmp_path / "after")), "state")


def test_latest_checkpoint_prefers_the_highest_step_and_skips_the_corrupt(tmp_path):
    out = str(tmp_path / "c")
    assert T.latest_checkpoint(out) is None, "an empty root is not a checkpoint"

    T.train(out_dir=out, steps=5, device="cpu", log_every=0, **SHAPE)
    T.train(out_dir=out + ".ckpt-a", steps=2, device="cpu", log_every=0, **SHAPE)
    assert T.latest_checkpoint(out) == out          # step 5 beats step 2

    # corrupt the winner: it must fall back, not raise and not pick it anyway
    with open(os.path.join(out, "trainer_state.pt"), "wb") as fh:
        fh.write(b"not a torch file")
    assert T.latest_checkpoint(out) == out + ".ckpt-a"


def test_slots_are_removed_once_the_chunk_directory_is_complete(tmp_path):
    """The disk bound. Slots are redundant the moment out_dir is loadable."""
    out = str(tmp_path / "c")
    T.train(out_dir=out, steps=5, save_every=2, device="cpu", log_every=0, **SHAPE)
    assert not os.path.exists(out + ".ckpt-a")
    assert not os.path.exists(out + ".ckpt-b")
    assert T.latest_checkpoint(out) == out


def test_refuses_a_resume_from_its_own_periodic_slot(tmp_path):
    """`{out_dir}.ckpt-a` is a name this chunk will overwrite at step save_every.

    Resuming from it and writing back into the same chunk destroys the source
    while the run is still healthy.
    """
    out = str(tmp_path / "c")
    T.train(out_dir=out + ".ckpt-a", steps=2, device="cpu", log_every=0, **SHAPE)
    with pytest.raises(ValueError, match="out_dir"):
        T.train(out_dir=out, steps=3, save_every=2, device="cpu", log_every=0,
                resume_from=out + ".ckpt-a", **SHAPE)


def test_save_every_is_in_the_signature_for_the_notebook_probe(tmp_path):
    """The notebook detects this by `inspect.signature` and warns if absent."""
    import inspect
    sig = inspect.signature(T.train)
    assert "save_every" in sig.parameters
    assert sig.parameters["save_every"].default == 0, "periodic saving is opt-in"
    assert sig.parameters["save_every"].kind is inspect.Parameter.KEYWORD_ONLY
