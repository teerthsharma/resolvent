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

import torch
import torch.nn.functional as F

from .. import sizing
from .configuration_ceq import CEQConfig
from .modeling_ceq import CEQForCausalLM

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
    """
    def hook(mod, inputs, output):
        attn = mod._ceq_owner
        b, s, _ = output.shape
        q, k, _ = output.chunk(3, dim=-1)

        def shape(t):
            return t.view(b, s, attn.n_heads, attn.d_head).transpose(1, 2)

        with torch.no_grad():
            w = (shape(q) @ shape(k).transpose(-2, -1)) / math.sqrt(attn.d_head)
            m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
            l1 = w.masked_fill(~m, 0.0).abs().sum(-1)[..., 1:]   # row 0 is empty
            store.append(float(l1.min()))

    handles = []
    for layer in model.model.layers:
        layer.self_attn.qkv._ceq_owner = layer.self_attn
        handles.append(layer.self_attn.qkv.register_forward_hook(hook))
    return handles


# --------------------------------------------------------------------- train

#: Operator settings this function will FORWARD but never DEFAULT.
_OPERATOR_KEYS = ("operator", "rho", "lam", "hops")


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
    operator = {k: overrides[k] for k in _OPERATOR_KEYS if k in overrides}
    return CEQForCausalLM(CEQConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=n_layers,
        num_attention_heads=n_heads, max_position_embeddings=seq, **operator))


def train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
          device="cuda", vocab_size=256, data_path=None, max_bytes=64 * 1024 * 1024,
          lr=3e-4, seed=0, grad_checkpoint=False, clip=1.0, probe_every=1,
          log_every=50, gpu=None, resume_from=None):
    """Train and `save_pretrained` into `out_dir`. Returns the run record.

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

    gen = torch.Generator().manual_seed(seed + 1)
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
                      seq=seq, vocab_size=vocab_size).to(device)
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

    losses, gnorms, l1mins = [], [], []
    for step in range(steps):
        probe_store.clear()
        x, _ = data.batch("train", batch, seq, gen, device)
        # `labels=x`: the model shifts internally (logits[:, :-1] against
        # labels[:, 1:]), so passing the pre-shifted targets would shift twice.
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
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

    for h in handles:
        h.remove()
    for layer in model.model.layers:
        del layer.self_attn.qkv._ceq_owner

    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    model.model.gradient_checkpointing = False
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    torch.save(dict(optimizer=opt.state_dict(), step=start_step + steps,
                    torch_rng_state=torch.get_rng_state(), data_gen_state=gen.get_state()),
              os.path.join(out_dir, "trainer_state.pt"))

    record = dict(steps=steps, start_step=start_step, losses=losses, grad_norms=gnorms,
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
