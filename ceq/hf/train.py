"""The training routine the Colab notebook calls.

WHY THIS IS A MODULE AND NOT NOTEBOOK CELLS. A notebook is the least testable
artifact anyone ships: cell text is not imported, not linted, and not executed
until somebody is already paying for a runtime. Everything of substance lives
here, where `tests/chase/test_colab_chain.py` runs it on cpu and cuda in seconds,
and the notebook is reduced to install / probe / call / push.

THREE THINGS THIS LOOP DOES THAT A STOCK LOOP DOES NOT, each from a measurement:

1. `preflight()` REFUSES a shape that cannot fit, before the dataset download.
   The 0.5B signed configuration at seq 2048 needs 36.31 GiB at BATCH 1 against
   an A100-40GB's 33.53 GiB budget, and 230.01 GiB at batch 8. Finding that out
   after twenty minutes of streaming is the expensive way.

2. It logs the MINIMUM ROW L1 of the operator's raw logits every step. The
   backward of `A = rho*w/sum|w|` carries a factor 1/l1, measured at exactly 10x
   per decade over six decades; over 800 steps on real text the minimum reached
   9.35e-07 while the median rose from 16.55 to 53.07. The global gradient norm
   does not show this -- it stayed at 1.21 -- because one bad row out of ~65,000
   disappears into it.

3. `grad_checkpoint` is priced by the preflight and verified by a test to change
   the memory and NOT the loss. It is not a rollback, it is a REQUIREMENT: at
   0.5B and seq 2048 the signed arm fits NO Colab GPU at batch 1 without it, and
   with it reaches batch 15 on an A100-40GB and batch 7 on an L4-24GB.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import shutil

import torch
import torch.nn.functional as F

from .. import sizing
from .configuration_ceq import CEQConfig
from .modeling_ceq import CEQForCausalLM, beta_column, beta_census

#: A shape that fits the smallest Colab GPU. Byte-level vocabulary, so there is
#: no tokenizer dependency and nothing about the comparison hides in a merge
#: table -- the same reason `ceq/lm.py` uses one.
DEFAULTS = dict(hidden_size=512, n_layers=8, n_heads=8, seq=512, batch=8,
                vocab_size=256)

_LOCAL_CORPUS = pathlib.Path(__file__).resolve().parents[2] / "data" / "tinystories_20k.txt"


# ------------------------------------------------------------------- preflight

def preflight(*, hidden_size, n_layers, n_heads, seq, batch, gpu,
              vocab_size=256, grad_checkpoint=False,
              optimizer="adamw_bf16_mixed", dtype="bf16_autocast"):
    """Return (ok, message). Called before anything is downloaded."""
    cfg = sizing.Config(d_model=hidden_size, n_layers=n_layers, n_heads=n_heads,
                        d_head=hidden_size // n_heads, vocab=vocab_size, seq=seq,
                        tied_embeddings=False)
    f = sizing.fits(cfg, batch=batch, arm="signed", gpu=gpu, optimizer=optimizer,
                    dtype=dtype, checkpointed=grad_checkpoint)
    ctrl = sizing.fits(cfg, batch=batch, arm="softmax", gpu=gpu,
                       optimizer=optimizer, dtype=dtype)
    msg = f.explain() + "\n  softmax control at the same shape: {:.2f} GiB.".format(
        ctrl.total / 1024 ** 3)
    if not f.ok and not grad_checkpoint:
        ck = sizing.fits(cfg, batch=batch, arm="signed", gpu=gpu,
                         optimizer=optimizer, dtype=dtype, checkpointed=True)
        msg += "\n  with grad_checkpoint=True it would be {:.2f} GiB ({}).".format(
            ck.total / 1024 ** 3, "fits" if ck.ok else "still does not fit")
    return f.ok, msg


# ------------------------------------------------------------------- open data

def load_open_text(dataset_id="roneneldan/TinyStories", split="train",
                   max_bytes=64 * 1024 * 1024, text_key="text"):
    """Stream an OPEN dataset and return raw text.

    Streaming rather than downloading: Colab's disk is shared with the
    checkpoints, and the whole corpus is never needed at byte level.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split, streaming=True)
    buf, n = [], 0
    for row in ds:
        t = row[text_key]
        buf.append(t)
        n += len(t)
        if n >= max_bytes:
            break
    return "\n".join(buf)


def _corpus_text(data_path, max_bytes):
    p = pathlib.Path(data_path) if data_path else _LOCAL_CORPUS
    if not p.exists():
        raise FileNotFoundError(
            "{} not found. Pass data_path=, or call load_open_text() first and "
            "write the result to a file.".format(p))
    return p.read_text(encoding="utf-8", errors="ignore")[:max_bytes]


class ByteBatches:
    """Raw bytes, 90/10 contiguous split so validation is unseen text."""

    def __init__(self, text, vocab_size=256, val_frac=0.1):
        b = torch.tensor(list(text.encode("utf-8")), dtype=torch.long) % vocab_size
        cut = int(len(b) * (1 - val_frac))
        self.train, self.val = b[:cut], b[cut:]

    def batch(self, split, bs, seq, gen, device):
        d = self.train if split == "train" else self.val
        if len(d) <= seq + 1:
            raise ValueError("corpus too small for seq={}".format(seq))
        i = torch.randint(len(d) - seq - 1, (bs,), generator=gen)
        x = torch.stack([d[j:j + seq] for j in i])
        y = torch.stack([d[j + 1:j + seq + 1] for j in i])
        return x.to(device), y.to(device)


# ----------------------------------------------------------------- the probe

def _attach_row_l1_probe(model, store):
    """Record the minimum row L1 of the raw logits, per step.

    Hooks the `qkv` Linear so its output is reused rather than recomputed; the
    probe then costs ONE extra [S,S] matmul per layer, about 3% of a forward.

    IT PREDICTS A BLOW-UP THAT ONLY THE `signed` OPERATOR HAS. `A = rho*w/sum|w|`
    divides by the row L1 of the raw logits, so its backward carries a 1/l1 and a
    row whose L1 collapses to 9.35e-07 is a real hazard invisible in the global
    gradient norm. `sgate` -- the shipped default -- divides by the constant
    `1 + lam` and by nothing data-dependent, and its raw-logit row L1 has a
    measured constructive floor of 1.227272 at every d and S. So on the default
    operator this number is a diagnostic of the LOGIT SPREAD (which is what
    decides whether the operator has any negative entries at all -- see
    `COSTS["signedness"]`) and NOT a blow-up alarm. Left on for both, described
    correctly for both.

    THE OWNING ATTENTION IS BOUND IN A CLOSURE, NOT STORED ON THE `qkv` MODULE.
    It used to be `qkv._ceq_owner = self_attn`, and assigning an `nn.Module` to
    an attribute of another `nn.Module` REGISTERS IT AS A SUBMODULE -- so `qkv`
    contained `self_attn` which contained `qkv`, a cycle that sends
    `Module.state_dict()` into unbounded recursion. It never fired because the
    only `save_pretrained` happened after the post-loop `del`; the first
    mid-loop checkpoint hit `RecursionError: maximum recursion depth exceeded`
    inside `state_dict` immediately. A closure carries the same reference with
    no registration, and the model tree is unchanged while the probe is on.
    """
    def make_hook(attn):
        def hook(mod, inputs, output):
            b, s, _ = output.shape
            q, k, _ = output.chunk(3, dim=-1)

            def shape(t):
                return t.view(b, s, attn.n_heads, attn.d_head).transpose(1, 2)

            with torch.no_grad():
                w = (shape(q) @ shape(k).transpose(-2, -1)) / math.sqrt(attn.d_head)
                m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
                l1 = w.masked_fill(~m, 0.0).abs().sum(-1)[..., 1:]  # row 0 is empty
                store.append(float(l1.min()))
        return hook

    handles = []
    for layer in model.model.layers:
        handles.append(layer.self_attn.qkv.register_forward_hook(
            make_hook(layer.self_attn)))
    return handles


# --------------------------------------------------------------------- train

#: Operator settings this function will FORWARD but never DEFAULT.
_OPERATOR_KEYS = ("operator", "rho", "lam", "hops",
                  "smp_beta", "smp_qk", "smp_g")


def _atomic_torch_save(obj, path):
    """`torch.save` that a kill cannot leave half-written.

    `torch.save(obj, path)` TRUNCATES `path` and then streams into it. A session
    cap landing between the truncate and the last byte leaves a
    `trainer_state.pt` that exists, is the newest file in its directory, and
    does not load -- measured at 57,636 bytes of a 115,272-byte state by
    `tests/gate0/test_g03_persist.py`. Present-but-unloadable is strictly worse
    than absent, because absent is a loud `FileNotFoundError` while present is a
    directory that looks resumable.

    The temp file is a SIBLING of the target, not in the system temp directory,
    because `os.replace` is only atomic within one filesystem. `fsync` before
    the rename so the rename cannot be ordered ahead of the data.

    ponytail: the containing directory is not fsynced after the rename -- there
    is no portable way to do that on Windows, where this is developed. On a
    power cut the rename itself could be lost and the PREVIOUS checkpoint
    directory is then the last good one, which is what the rotation policy
    already assumes. Add a directory fsync in the POSIX branch if a
    non-graceful host reboot ever turns up as a real loss mode on Kaggle.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


#: The two alternating periodic-checkpoint slots, as suffixes on `out_dir`.
#:
#: TWO AND NOT ONE, because `save_pretrained` is not atomic. A periodic save
#: that overwrote a single slot in place would destroy the only mid-chunk
#: checkpoint at exactly the moment it is being replaced, which is the window a
#: session cap is most likely to land in on a long chunk. Alternating leaves the
#: previous slot complete at every instant.
#:
#: TWO AND NOT N, because N needs a reaper, and a reaper needs to decide what to
#: delete. Two is bounded by construction: no globbing, no policy, no deletion
#: of anything this function did not itself just write.
_SLOTS = (".ckpt-a", ".ckpt-b")


def _ckpt_step(out_dir):
    """The step a checkpoint directory stands at, or None if it is not one.

    This IS the completeness test. `_save_checkpoint` writes the model shards
    first and `trainer_state.pt` last and atomically, so a directory whose
    `trainer_state.pt` loads has complete shards by construction. One
    `torch.load` therefore certifies the whole directory and no DONE marker
    exists or is needed.
    """
    try:
        return int(torch.load(os.path.join(out_dir, "trainer_state.pt"),
                              map_location="cpu", weights_only=True)["step"])
    except Exception:
        return None


def latest_checkpoint(out_dir):
    """The newest COMPLETE checkpoint for `out_dir`, or None.

    Looks at the chunk directory and its two periodic slots and returns the one
    standing at the highest step. Pass the result straight back as
    `resume_from`; the autopilot passes it as `event.last_good_ckpt`.
    """
    best = None
    for d in (out_dir,) + tuple(out_dir + s for s in _SLOTS):
        step = _ckpt_step(d)
        if step is not None and (best is None or step > best[0]):
            best = (step, d)
    return best[1] if best else None


def _save_checkpoint(model, opt, gen, step, out_dir):
    """One complete, resumable checkpoint directory.

    THE ORDER IS THE CERTIFICATE: shards first, `trainer_state.pt` last and
    atomically. `grad_checkpoint` is toggled off across the save and restored,
    so a mid-chunk save leaves the running loop exactly as it found it.
    """
    was = model.model.gradient_checkpointing
    model.model.gradient_checkpointing = False
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    model.model.gradient_checkpointing = was
    _atomic_torch_save(
        dict(optimizer=opt.state_dict(), step=step,
             torch_rng_state=torch.get_rng_state(), data_gen_state=gen.get_state()),
        os.path.join(out_dir, "trainer_state.pt"))


def build(*, hidden_size, n_layers, n_heads, seq, vocab_size=256, **overrides):
    """Shape here, operator in `CEQConfig`.

    THE OPERATOR KEYS ARE DELIBERATELY NOT DEFAULTED HERE. This function used to
    carry `hops=3, rho=0.9` of its own, which were the old `signed` defaults;
    once the config default moved to `sgate` at rho=1.5 hops=2, those local
    defaults would have silently trained `sgate` at rho=0.9 hops=3 -- a point
    nobody has ever measured -- while every log line said "ceq". A default in two
    places is a default that drifts, so this one lives in `CEQConfig` alone and
    anything passed here is forwarded verbatim.
    """
    unknown = sorted(set(overrides) - set(_OPERATOR_KEYS))
    if unknown:
        # SILENT DROP IS THE DEFECT CLASS, not this key. `train()` used to
        # forward no operator at all and every run built the default while
        # the caller believed otherwise; `smp_beta/qk/g` were dropped the
        # same way and `train(operator='smprime', smp_beta=0.0)` trained at
        # (1,1,1) with nothing raised. A key this function does not know is
        # a caller expectation it cannot meet, so it says so.
        raise TypeError(
            "build() does not understand {}; known operator keys are {}"
            .format(unknown, list(_OPERATOR_KEYS)))
    operator = {k: overrides[k] for k in _OPERATOR_KEYS if k in overrides}
    return CEQForCausalLM(CEQConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=n_layers,
        num_attention_heads=n_heads, max_position_embeddings=seq, **operator))


def train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
          device="cuda", vocab_size=256, data_path=None, max_bytes=64 * 1024 * 1024,
          lr=3e-4, seed=0, grad_checkpoint=False, clip=1.0, probe_every=1,
          log_every=50, gpu=None, resume_from=None, save_every=0, **overrides):
    """Train and `save_pretrained` into `out_dir`. Returns the run record.

    PERIODIC CHECKPOINTS. `save_every=N` writes a complete checkpoint every N
    steps into `{out_dir}.ckpt-a` and `{out_dir}.ckpt-b` alternately, through
    the same atomic path as the end-of-chunk save. `0`, the default, is
    end-of-chunk only. This is not a convenience: a Kaggle chunk runs up to
    11 hours against a 30 GPU-h weekly quota, so an end-of-chunk-only save
    turns one kernel death into ten lost GPU-hours and three of them into the
    entire week. `latest_checkpoint(out_dir)` returns whichever of the three
    directories stands at the highest step and is loadable -- pass it back as
    `resume_from`. Both slots are removed once `out_dir` itself is complete,
    because they are redundant from that moment; the disk cost during a chunk
    is therefore bounded at two extra directories with no reaper anywhere.

    RESUME. `resume_from`, if given, is an earlier `train()` call's `out_dir`.
    Model weights come back through `from_pretrained` (already correct); what
    was missing was everything `save_pretrained` does not know about, so
    `{out_dir}/trainer_state.pt` now also carries the optimizer's per-parameter
    moments, the data-sampling generator's state and the global torch RNG
    state. There is no scheduler and no dataloader to checkpoint -- `lr` is a
    constant and batches are drawn by index from `gen`, so the generator's
    state IS the dataloader position. `steps` counts steps ADDED, not a new
    total, and `seed` is ignored on resume: the saved state already determines
    what comes next.
    """
    if gpu is not None:
        ok, msg = preflight(hidden_size=hidden_size, n_layers=n_layers,
                            n_heads=n_heads, seq=seq, batch=batch, gpu=gpu,
                            vocab_size=vocab_size, grad_checkpoint=grad_checkpoint)
        if not ok:
            raise MemoryError("preflight refused this shape:\n" + msg)

    forbidden = {os.path.abspath(out_dir)}
    forbidden |= {os.path.abspath(out_dir + s) for s in _SLOTS}
    if resume_from and os.path.abspath(resume_from) in forbidden:
        raise ValueError(
            "out_dir must not be resume_from, nor either of its periodic slots "
            "{!r}. `save_pretrained` is not "
            "atomic, so writing back into the directory being resumed from "
            "overwrites the model shards of the very state being resumed -- and "
            "on Windows it does not even get that far, safetensors raises "
            "os error 1224 against its own memory-mapped read. Use a NEW "
            "out_dir per chunk, as TRAINING.md 6.6 prescribes.".format(resume_from))

    gen = torch.Generator().manual_seed(seed + 1)
    if resume_from and overrides:
        # A resumed run reads its operator out of the checkpoint's own
        # config, so anything passed here would be silently ignored -- the
        # same shape of defect as the missing passthrough itself.
        raise ValueError(
            "operator overrides {} cannot be applied on resume; the operator "
            "comes from {}/config.json. Drop them, or start a fresh run."
            .format(sorted(overrides), resume_from))
    if resume_from:
        model = CEQForCausalLM.from_pretrained(resume_from).to(device)
        # weights_only=True: this file is tensors, ints and dicts only, and a
        # checkpoint is untrusted input -- torch is flipping this default anyway.
        ckpt = torch.load(os.path.join(resume_from, "trainer_state.pt"),
                          map_location="cpu", weights_only=True)
        torch.set_rng_state(ckpt["torch_rng_state"])
        gen.set_state(ckpt["data_gen_state"])
        start_step = ckpt["step"]
    else:
        torch.manual_seed(seed)
        model = build(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads,
                      seq=seq, vocab_size=vocab_size, **overrides).to(device)
        start_step = 0
    if grad_checkpoint:
        model.model.gradient_checkpointing = True
    model.train()

    data = ByteBatches(_corpus_text(data_path, max_bytes), vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if resume_from:
        opt.load_state_dict(ckpt["optimizer"])

    probe_store = []
    handles = _attach_row_l1_probe(model, probe_store)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    losses, gnorms, l1mins, slot = [], [], [], 0
    beta_track, beta_acc = [], None
    for step in range(steps):
        probe_store.clear()
        x, _ = data.batch("train", batch, seq, gen, device)
        # `labels=x`: the model shifts internally (logits[:, :-1] against
        # labels[:, 1:]), so passing the pre-shifted targets would shift twice.
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        beta_acc = beta_census(model, beta_acc)
        g = torch.norm(torch.stack([p.grad.detach().norm()
                                    for p in model.parameters() if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(float(loss))
        gnorms.append(float(g))
        l1mins.append(min(probe_store) if probe_store else float("nan"))
        if log_every and step % log_every == 0:
            print("step {:6d}  loss {:.4f}  |g| {:.4e}  min row L1 {:.3e}".format(
                step, losses[-1], gnorms[-1], l1mins[-1]), flush=True)
            beta_track.append(beta_column(model))
        # `step + 1 < steps`: the end-of-chunk save below already covers the
        # last step, and at the training shape a redundant one costs 0.29 GiB
        # and a wall-clock stall.
        if save_every and (step + 1) % save_every == 0 and step + 1 < steps:
            _save_checkpoint(model, opt, gen, start_step + step + 1,
                             out_dir + _SLOTS[slot])
            slot = 1 - slot

    for h in handles:
        h.remove()

    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    model.model.gradient_checkpointing = False
    _save_checkpoint(model, opt, gen, start_step + steps, out_dir)
    # The slots are redundant the instant `out_dir` itself loads, and that is
    # verified rather than assumed before anything is deleted. Only the two
    # names this call itself wrote are ever removed.
    if save_every and _ckpt_step(out_dir) == start_step + steps:
        for s in _SLOTS:
            shutil.rmtree(out_dir + s, ignore_errors=True)

    beta_track.append(beta_column(model))
    record = dict(steps=steps, start_step=start_step,
                  beta=beta_track, beta_census=beta_acc,
                  operator=model.config.operator, losses=losses, grad_norms=gnorms,
                  row_l1_min=l1mins, peak_bytes=int(peak),
                  n_params=sum(p.numel() for p in model.parameters()),
                  grad_checkpoint=bool(grad_checkpoint), device=str(device))
    with open(os.path.join(out_dir, "run_record.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


# ---------------------------------------------------------------------- push

def push(out_dir, repo_id, *, token=None, private=True, commit_message="ceq"):
    """Upload the whole directory, .py files included.

    `upload_folder` rather than `model.push_to_hub`: push_to_hub re-runs
    `save_pretrained` into a temp dir and uploads that, which is correct but
    re-derives the artifact. Uploading the directory that was already saved AND
    already loaded back in a test is the version that was verified.

    The token is read from the argument or `HF_TOKEN`; it is never written into
    the notebook.
    """
    from huggingface_hub import HfApi

    need = {"config.json", "configuration_ceq.py", "modeling_ceq.py"}
    have = set(os.listdir(out_dir))
    missing = need - have
    if missing:
        raise FileNotFoundError(
            "{} missing from {}. Without them `auto_map` points at nothing and the "
            "repo downloads but does not load.".format(sorted(missing), out_dir))

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, private=private, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, folder_path=out_dir,
                             commit_message=commit_message)
