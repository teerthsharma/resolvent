"""The trained half of `test_scale_axes.py`, run once, recorded to JSONL.

Self-contained on purpose: it builds and trains models directly instead of going
through `lm.train_one`, so a concurrent edit to `ceq/lm.py`'s training signature
cannot silently change a recorded number. Same reason `scale/scale_sweep.py`
does it.

ONE AXIS MOVES AT A TIME. `scale/scale_sweep.py` moves d, L and H together, which
is the right shape for "does the number move" and the wrong shape for "which
thing moved it". Everything here holds the parity point --
`rho=1.5 lam=0.10 hops=2 lr=1e-3`, 600 steps, byte-level TinyStories -- and moves
exactly one thing.

    python tests/chase/axes.py heads depth seq rho

Usage note: the runs are long. Each axis appends its own records and is
idempotent by (axis, x, size), so an interrupted sweep can be resumed by
re-running the same axis.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ceq import lm

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "scale_axes.jsonl"
CORPUS = ROOT / "data" / "tinystories_20k.txt"

#: The shipped parity point, iteration 16. Nothing here re-tunes it.
STEPS, LR, RHO, LAM, HOPS = 600, 1e-3, 1.5, 0.10, 2
SEEDS = (0, 1, 2)
#: tokens per optimizer step, held constant so the sequence axis is not also a
#: token-budget axis.
TOKENS_PER_STEP = 4096


def load():
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def _append(rec):
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def train(kind, corpus, *, d, layers, heads, seq, batch, seed, device, rho):
    lm.RHO, lm.SGATE_LAM, lm.HOPS = rho, LAM, HOPS
    torch.manual_seed(seed)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=seed).to(device)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    m.train()
    for _ in range(STEPS):
        x, y = corpus.batch("train", batch, seq, g, device)
        loss = F.cross_entropy(m(x).reshape(-1, lm.VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    vg = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        vals = []
        for _ in range(20):
            a, b = corpus.batch("val", batch, seq, vg, device)
            vals.append(F.cross_entropy(m(a).reshape(-1, lm.VOCAB), b.reshape(-1)))
        val = float(torch.stack(vals).mean())
    n = m.n_params()
    del m
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return val, n


def _point(corpus, device, *, axis, x, cfg, rho=RHO, size="", seeds=SEEDS,
           softmax_from=None):
    """Train both arms at one configuration and record the ratio."""
    done = {(r["axis"], r["x"], r["size"]) for r in load()}
    if (axis, x, size) in done:
        print("skip {} x={} {} (already recorded)".format(axis, x, size), flush=True)
        return
    t0 = time.perf_counter()
    if softmax_from is None:
        sm = [train("softmax", corpus, seed=s, device=device, rho=rho, **cfg)
              for s in seeds]
        sm_vals, npar = [v for v, _ in sm], sm[0][1]
    else:
        sm_vals, npar = softmax_from
    sg = [train("sgate", corpus, seed=s, device=device, rho=rho, **cfg) for s in seeds]
    sg_vals = [v for v, _ in sg]
    ratios = [a / b for a, b in zip(sg_vals, sm_vals)]
    rec = dict(axis=axis, x=x, size=size, cfg=dict(cfg), rho=rho, lam=LAM, hops=HOPS,
               lr=LR, steps=STEPS, seeds=list(seeds), params=npar,
               batch=cfg["batch"], seq=cfg["seq"], d=cfg["d"],
               layers=cfg["layers"], heads=cfg["heads"],
               softmax=statistics.median(sm_vals), sgate=statistics.median(sg_vals),
               ratio=statistics.median(ratios), lo=min(ratios), hi=max(ratios),
               softmax_all=sm_vals, sgate_all=sg_vals,
               device=str(device), seconds=time.perf_counter() - t0)
    _append(rec)
    print("{axis:6} x={x:<6} {size:6} params {params:>10,}  softmax {softmax:.4f}  "
          "sgate {sgate:.4f}  ratio {ratio:.4f} [{lo:.4f}-{hi:.4f}]  {seconds:.0f}s"
          .format(**rec), flush=True)
    return sm_vals, npar


# ------------------------------------------------------------------- the axes

def axis_heads(corpus, device):
    """d=256 L=4 seq=128 fixed; H = 2,4,8,16 -> d_head = 128,64,32,16.

    Parameter count is IDENTICAL at every point: the head split is a reshape.
    """
    for h in (2, 4, 8, 16):
        _point(corpus, device, axis="heads", x=h,
               cfg=dict(d=256, layers=4, heads=h, seq=128, batch=32))


def axis_depth(corpus, device):
    """d=256 H=4 seq=128 fixed; L = 2,4,8,16. 300M is 24."""
    for L in (2, 4, 8, 16):
        _point(corpus, device, axis="depth", x=L,
               cfg=dict(d=256, layers=L, heads=4, seq=128, batch=32))


def axis_seq(corpus, device):
    """d=256 L=4 H=4 fixed; seq = 128,256,512,1024 with batch*seq pinned at 4096."""
    for s in (128, 256, 512, 1024):
        _point(corpus, device, axis="seq", x=s,
               cfg=dict(d=256, layers=4, heads=4, seq=s,
                        batch=TOKENS_PER_STEP // s))


def axis_rho(corpus, device):
    """Does the argmin over rho MOVE between two sizes 8x apart in parameters?

    The two sizes are separated on DEPTH, 4 layers against 16, because depth is
    where the operator's per-layer DC gain of 3.734 against softmax's 1.000
    compounds, and because 300M's largest jump from the parity point is 4 -> 24
    layers. 3.32M against 12.86M parameters.

    Two seeds rather than three: the question is where the minimum sits, not how
    tight the estimate at one rho is. softmax is trained ONCE per size -- it has
    no rho -- so the control is literally the same numbers across the grid.

    GRID EXTENDED, R10 P0 it.3. The first four points (0.9, 1.2, 1.5, 2.0) ran to
    completion and could not answer the question: the ratio was monotone
    DECREASING across the whole grid at both sizes, so both argmins sat on the
    boundary point 2.0. A quantity pinned to the edge at both ends cannot be shown
    to move or not move. 3.0 and 4.0 are one and two octaves out; the second new
    point is what distinguishes "the argmin is bracketed" from "the wall moved".
    """
    seeds = (0, 1)
    for size, cfg in (("small", dict(d=256, layers=4, heads=4, seq=128, batch=32)),
                      ("large", dict(d=256, layers=16, heads=4, seq=128, batch=32))):
        sm = [train("softmax", corpus, seed=s, device=device, rho=RHO, **cfg)
              for s in seeds]
        base = ([v for v, _ in sm], sm[0][1])
        print("rho control {} softmax {}".format(size, base[0]), flush=True)
        for r in (0.9, 1.2, 1.5, 2.0, 3.0, 4.0):
            _point(corpus, device, axis="rho", x=r, size=size, cfg=cfg, rho=r,
                   seeds=seeds, softmax_from=base)


AXES = dict(heads=axis_heads, depth=axis_depth, seq=axis_seq, rho=axis_rho)


def main(names):
    corpus = lm.ByteCorpus(CORPUS.read_text(encoding="utf-8")[:4_000_000])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device={} steps={} lr={} rho={} lam={} hops={} seeds={}".format(
        device, STEPS, LR, RHO, LAM, HOPS, SEEDS), flush=True)
    for n in names:
        print("=== axis {}".format(n), flush=True)
        AXES[n](corpus, device)


if __name__ == "__main__":
    main(sys.argv[1:] or list(AXES))
