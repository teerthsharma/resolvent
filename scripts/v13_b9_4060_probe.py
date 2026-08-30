"""B9 -- measured throughput and peak bytes on the local card.

The contract's budget line prices attention activation as `2*n*s^2*heads` bytes
in bf16, i.e. ONE retained [n, heads, s, s] tensor at 2 bytes per element, and
concludes that the binding constraint is throughput rather than memory.
`ceq/sizing.py` was calibrated against the CUDA allocator on this same card and
records 3.9 retained tensors per layer at an effective 3.4 bytes per element
under bf16 autocast. This script decides between them by measurement.

Run: python scripts/v13_b9_4060_probe.py [--steps N]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import ceq.sizing as sz
from ceq import lm


def build(kind, *, d, layers, heads, seq, device):
    torch.manual_seed(0)
    return lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)


def forward(m, idx, *, ckpt):
    h = m.tok(idx) + m.pos(torch.arange(idx.shape[1], device=idx.device))[None]
    for blk in m.blocks:
        h = checkpoint(blk, h, use_reentrant=False) if ckpt else blk(h)
    return m.head(m.norm(h))


def peak_and_rate(kind, *, d, layers, heads, seq, bs, device, steps, ckpt):
    """Returns (peak_bytes, tokens_per_sec) or (None, None) if it OOMs."""
    try:
        m = build(kind, d=d, layers=layers, heads=heads, seq=seq, device=device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randint(0, lm.VOCAB, (bs, seq), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        # one warm step, excluded from timing: allocator and autotune settle here
        F.cross_entropy(forward(m, x, ckpt=ckpt).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            F.cross_entropy(forward(m, x, ckpt=ckpt).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() - base
        del m, x, opt
        torch.cuda.empty_cache()
        return peak, bs * seq * steps / dt
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable -- B9 needs the card"); return 1
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print("card: %s  %.2f GB  sm_%d%d  SMs=%d  torch %s"
          % (p.name, p.total_memory / 1e9, p.major, p.minor,
             p.multi_processor_count, torch.__version__))
    print("shape: d=%d heads=%d seq=%d bs=%d steps=%d\n" % (a.d, a.heads, a.seq, a.bs, a.steps))

    print("%-8s %-3s %-6s %12s %12s %8s %12s" %
          ("arm", "L", "ckpt", "peak_MiB", "model_MiB", "ratio", "tok/s"))
    for layers in (4, 8, 16):
        cfg = sz.Config(d_model=a.d, n_layers=layers, n_heads=a.heads,
                        d_head=a.d // a.heads, vocab=lm.VOCAB, seq=a.seq)
        for kind in ("softmax", "signed"):
            for ckpt in (False, True):
                peak, rate = peak_and_rate(kind, d=a.d, layers=layers, heads=a.heads,
                                           seq=a.seq, bs=a.bs, device=dev,
                                           steps=a.steps, ckpt=ckpt)
                pred = sz.activation_bytes(cfg, batch=a.bs, arm=kind, checkpointed=ckpt)
                if peak is None:
                    print("%-8s %-3d %-6s %12s %12.1f %8s %12s"
                          % (kind, layers, ckpt, "OOM", pred / 2**20, "-", "-"))
                    continue
                print("%-8s %-3d %-6s %12.1f %12.1f %8.2f %12.0f"
                      % (kind, layers, ckpt, peak / 2**20, pred / 2**20,
                         peak / pred, rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
