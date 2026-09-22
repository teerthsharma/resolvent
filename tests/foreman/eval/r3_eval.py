"""R3/R4/R5: eval path for ceq/hf/train.py, the eval_indices seed fix, and the
determinism measurement -- all against COPIES / monkeypatched reuse of the
real ceq/ modules. Nothing under ceq/ is edited by this file; the diff this
demonstrates is proposed in r3_eval.md.

Reused UNEDITED from the real repo: build(), _corpus_text(), _attach_row_l1_probe(),
CEQForCausalLM, beta_column, beta_census, sizing. Replaced/added: ByteBatches
(byte-offset split -> by-document split), eval_indices (seed-honouring),
the eval loop itself (did not exist), R5 asserts, repetition-factor printing.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, REPO)

import torch

from ceq.hf.train import build, _corpus_text, _attach_row_l1_probe, preflight
from ceq.hf.modeling_ceq import beta_column, beta_census
from ceq.harness import eval_indices as eval_indices_ORIGINAL_BUGGY  # read-only, for the demo

CORPUS_TOKENS = 18_167_706  # ceq/data/tinystories_20k.txt, byte-level vocab: 1 byte = 1 token
ZERO_REP_CEILING_PARAMS = CORPUS_TOKENS / 20  # == 908,385.3


# --------------------------------------------------------------- R2: repetition

def repetition_factor(n_params: int) -> float:
    """Chinchilla tokens (20 * n_params) / the fixed corpus's token count.

    >1x means training to the Chinchilla-optimal budget for this parameter
    count would require repeating this corpus more than once -- the corpus,
    not the architecture, is then the ceiling, and no generalization claim
    survives it (R2).
    """
    return (20.0 * n_params) / CORPUS_TOKENS


def print_repetition_row(label: str, n_params: int) -> float:
    rep = repetition_factor(n_params)
    tag = " MEMORIZATION REGIME" if rep > 1.0 else ""
    print("[REP] {}: n_params={:,}  repetition_factor={:.4f}x{}".format(
        label, n_params, rep, tag), flush=True)
    return rep


# ------------------------------------------------------------- R3(1): the split

class DocByteBatches:
    """Byte-level batches, split BY DOCUMENT (blank-line-separated story), seeded.

    Replaces ceq.hf.train.ByteBatches, whose 90/10 split is a byte OFFSET into
    the concatenated corpus -- the same story straddles the cut, so validation
    bytes are frequently a direct continuation of training bytes the model has
    already seen. Splitting the document list first and only then concatenating
    each half removes that leak. `data/tinystories_20k.txt` documents are
    separated by a blank line (105,094 of them); `\\n\\n` splits after `.strip()`
    on each part removes any that were blank runs.
    """

    def __init__(self, text, vocab_size=256, val_frac=0.1, split_seed=0):
        docs = [d for d in text.split("\n\n") if d.strip()]
        if len(docs) < 10:
            raise ValueError("only {} documents found; is the delimiter still "
                              "'\\n\\n' for this corpus?".format(len(docs)))
        order = list(range(len(docs)))
        random.Random(split_seed).shuffle(order)
        cut = int(len(order) * (1 - val_frac))
        train_ix, val_ix = sorted(order[:cut]), sorted(order[cut:])

        def to_bytes(ix):
            t = "\n\n".join(docs[i] for i in ix)
            return torch.tensor(list(t.encode("utf-8")), dtype=torch.long) % vocab_size

        self.train, self.val = to_bytes(train_ix), to_bytes(val_ix)
        self.split_seed, self.val_frac = split_seed, val_frac
        self.n_docs, self.n_train_docs, self.n_val_docs = len(docs), len(train_ix), len(val_ix)

    def batch(self, split, bs, seq, gen, device):
        d = self.train if split == "train" else self.val
        if len(d) <= seq + 1:
            raise ValueError("{} split too small for seq={} ({} bytes)".format(
                split, seq, len(d)))
        i = torch.randint(len(d) - seq - 1, (bs,), generator=gen)
        x = torch.stack([d[j:j + seq] for j in i])
        y = torch.stack([d[j + 1:j + seq + 1] for j in i])
        return x.to(device), y.to(device)


# ------------------------------------------------------------- R3(2): the seed fix

def eval_indices_fixed(n_test: int, max_eval: int, seed: int) -> list[int]:
    """`ceq.harness.eval_indices`, with the seed argument honoured.

    The original seeds off `n_test` alone (`20260825 + n_test`): every `seed`
    value produces the identical `torch.Generator`, so eval seeds 1 and 2
    return byte-identical subsamples -- no subsample-luck estimate is
    reachable. Adding `seed` into the same formula keeps `seed=0` byte-identical
    to the original (so existing seed-0 results, e.g. cogs_control.md, do not
    move) while making every other seed distinct.
    """
    if max_eval >= n_test:
        return list(range(n_test))
    g = torch.Generator().manual_seed(20260825 + n_test + seed)
    return torch.randperm(n_test, generator=g)[:max_eval].sort().values.tolist()


def demonstrate_seed_fix():
    n_test, max_eval = 3920, 256  # data/scan_length_test.txt row count order of magnitude
    orig = [eval_indices_ORIGINAL_BUGGY(n_test, max_eval, s) for s in (0, 1, 2)]
    fixed = [eval_indices_fixed(n_test, max_eval, s) for s in (0, 1, 2)]
    report = dict(
        n_test=n_test, max_eval=max_eval,
        original_seed1_eq_seed2=(orig[1] == orig[2]),
        original_seed0_eq_seed1=(orig[0] == orig[1]),
        fixed_seed1_eq_seed2=(fixed[1] == fixed[2]),
        fixed_seed0_matches_original_seed0=(fixed[0] == orig[0]),
        fixed_seed0_head=fixed[0][:8],
        fixed_seed1_head=fixed[1][:8],
        fixed_seed2_head=fixed[2][:8],
    )
    assert report["original_seed1_eq_seed2"] is True, "bug no longer reproduces -- re-check"
    assert report["fixed_seed1_eq_seed2"] is False, "fix did not take"
    assert report["fixed_seed0_matches_original_seed0"] is True, "fix moved seed=0 behavior"
    print(json.dumps(report, indent=2), flush=True)
    return report


# ------------------------------------------------------------------- R5: asserts

def assert_head_dim(hidden_size: int, n_heads: int) -> int:
    assert hidden_size % n_heads == 0, "hidden_size not divisible by n_heads"
    d_head = hidden_size // n_heads
    assert d_head % 8 == 0, (
        "d_head={} is not a multiple of 8: both fused SDPA backends reject this "
        "shape and a math-kernel fallthrough materialises [B,H,S,S] at 31.8x "
        "memory (R5)".format(d_head))
    return d_head


def assert_no_math_kernel_fallthrough():
    """R5's 'fused SDPA backend asserted at runtime' -- N/A for this model.

    `ceq.hf.modeling_ceq.CEQForCausalLM` (`_supports_sdpa = False`) computes its
    attention by hand -- explicit `softmax`/matmul over `[B,H,S,S]` -- and never
    calls `torch.nn.functional.scaled_dot_product_attention` at all (grepped:
    zero hits in ceq/hf/*.py). There is no fused-kernel selection to assert on
    for THIS model; the R5 SDPA-backend note is about `ceq/lm.py`'s literal
    `softmax` control and `ceq/hybrid.py`, neither of which `train.py` calls.
    Documented rather than silently skipped, per the brief.
    """
    return "not applicable: this model has no F.scaled_dot_product_attention call site"


def check_host_spill(peak_bytes: int, device: str) -> bool:
    """R5: the memory wall is silent, not an OOM (WDDM host spill). Detected by
    comparing torch's own reported peak against the device's total memory --
    the spill signature is peak_bytes > device total with nothing raised."""
    if device != "cuda" or not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    spill = peak_bytes > total
    if spill:
        print("[R5] HOST SPILL DETECTED: reported peak {:.1f} MiB > device {:.1f} MiB "
              "(WDDM silent spill)".format(peak_bytes / 1024**2, total / 1024**2), flush=True)
    return spill


# ------------------------------------------------------------ the eval-bearing loop

def train_with_eval(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
                     device="cuda", vocab_size=256, data_path=None,
                     max_bytes=64 * 1024 * 1024, lr=3e-4, seed=0, clip=1.0,
                     val_frac=0.1, split_seed=None, eval_every=50, eval_batches=4,
                     log_every=50, save_model=False, **overrides):
    """`ceq.hf.train.train()`, minus checkpoint resume/save_every (not needed
    for these short scratchpad runs -- see r3_eval.md for the full-feature
    diff proposed against the real function), plus:
      - DocByteBatches instead of ByteBatches (split BY DOCUMENT, seeded)
      - an eval loss computed every `eval_every` steps AND at the end, on a
        FIXED held-out subsample (same eval generator seed every checkpoint
        and every repeat run, so L-REPRO and the determinism experiment are
        comparing the same held-out batches)
      - val_frac, split_seed, doc counts, repetition_factor in run_record.json
      - R5 asserts and the host-spill check
    """
    d_head = assert_head_dim(hidden_size, n_heads)
    assert_no_math_kernel_fallthrough()
    if split_seed is None:
        split_seed = seed

    torch.manual_seed(seed)
    model = build(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads,
                  seq=seq, vocab_size=vocab_size, **overrides).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    rep = print_repetition_row(os.path.basename(out_dir) if out_dir else "(unnamed)", n_params)

    data = DocByteBatches(_corpus_text(data_path, max_bytes), vocab_size,
                          val_frac=val_frac, split_seed=split_seed)
    gen = torch.Generator().manual_seed(seed + 1)
    eval_gen = torch.Generator().manual_seed(20260921 + split_seed)  # fixed across checkpoints/repeats
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    probe_store = []
    handles = _attach_row_l1_probe(model, probe_store)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def eval_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            eg = torch.Generator().manual_seed(int(eval_gen.initial_seed()))
            for _ in range(eval_batches):
                xv, _ = data.batch("val", batch, seq, eg, device)
                tot += float(model(input_ids=xv, labels=xv).loss)
        model.train()
        return tot / eval_batches

    losses, gnorms, l1mins = [], [], []
    eval_losses = [dict(step=0, loss=eval_loss())]
    _train_t0 = time.time()
    for step in range(steps):
        probe_store.clear()
        x, _ = data.batch("train", batch, seq, gen, device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        beta_census(model, None)
        g = torch.norm(torch.stack([p.grad.detach().norm()
                                    for p in model.parameters() if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(float(loss))
        gnorms.append(float(g))
        l1mins.append(min(probe_store) if probe_store else float("nan"))
        if eval_every and (step + 1) % eval_every == 0:
            eval_losses.append(dict(step=step + 1, loss=eval_loss()))
        if log_every and step % log_every == 0:
            print("step {:6d}  loss {:.4f}  |g| {:.4e}".format(
                step, losses[-1], gnorms[-1]), flush=True)
    _train_seconds = time.time() - _train_t0
    if eval_losses[-1]["step"] != steps:
        eval_losses.append(dict(step=steps, loss=eval_loss()))
    for h in handles:
        h.remove()

    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    host_spill = check_host_spill(peak, device)

    record = dict(
        steps=steps, seed=seed, device=str(device), n_params=n_params,
        hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
        seq=seq, batch=batch, lr=lr,
        losses=losses, grad_norms=gnorms, row_l1_min=l1mins,
        eval_losses=eval_losses, final_eval_loss=eval_losses[-1]["loss"],
        val_frac=val_frac, split_seed=split_seed,
        n_docs=data.n_docs, n_train_docs=data.n_train_docs, n_val_docs=data.n_val_docs,
        peak_bytes=int(peak), host_spill=host_spill,
        repetition_factor=rep, memorization_regime=(rep > 1.0),
        operator=model.config.operator,
        seconds=_train_seconds,
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_record.json"), "w") as fh:
            json.dump(record, fh, indent=2)
        if save_model:
            # This function's docstring drops `save_every` deliberately, as
            # "not needed for these short scratchpad runs". That reduction
            # turned out to matter: every arm trained here was discarded
            # in-process, so no analysis needing weights -- row mass, per-token
            # abstention -- could run on the arm that produced the headline
            # number. A single save at the end restores that without restoring
            # periodic checkpointing. Off by default: no existing caller changes.
            torch.save({"state_dict": model.state_dict(),
                        "n_params": n_params,
                        "operator": model.config.operator,
                        "seed": seed, "split_seed": split_seed},
                       os.path.join(out_dir, "model.pt"))
            record["model_path"] = os.path.join(out_dir, "model.pt")
    return record


if __name__ == "__main__":
    print("=== R3(2): eval_indices seed fix ===")
    demonstrate_seed_fix()
