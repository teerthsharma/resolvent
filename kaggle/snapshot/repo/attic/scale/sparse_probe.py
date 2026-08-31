"""Does a bounded-reachability schedule arrest the 1/s decay?

MECHANISM (Foreman, measured): a third token c reaches the pair (i,j) only along a
path THROUGH c, and the first term of J = sum_k A^k containing one is k=2. The
two-hop term sums over ~s intermediates of which c is one, so its share is 1/s.

PREDICTION: if a schedule bounds reachability at k << s, the sum runs over ~k
intermediates and the share is 1/k -- INDEPENDENT of s.

Two schedules, both from the author's own shipped work:
  window+sinks -- fixed k, the StreamingLLM shape that beat a dyadic schedule 3/3
  multizoom    -- k grows as log s, measured at 3.445*log2(seq), R^2 0.999833

Same construction and same probe convention as ceq/bench.sign_flip_rate so the
numbers are comparable to the published dense ones (n_draws=128 recovers
0.0469=6/128, 0.1641=21/128).
"""
from __future__ import annotations
import sys, math, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ceq import bench

RHO, LAM = 1.5, 0.10


def causal(s, device):
    return torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)


def window_sinks(s, w, n_sink, device):
    """Fixed reachability: w recent keys plus n_sink attention sinks."""
    m = causal(s, device).clone()
    idx = torch.arange(s, device=device)
    far = (idx[:, None] - idx[None, :]) > w
    m &= ~far
    m[:, :n_sink] |= causal(s, device)[:, :n_sink]
    return m


def multizoom(s, device, base=3.445):
    """Reachability growing as log2(s): geometric tiling of the past."""
    m = torch.zeros(s, s, dtype=torch.bool, device=device)
    idx = torch.arange(s, device=device)
    d = idx[:, None] - idx[None, :]
    lvls = max(1, int(base * math.log2(max(s, 2))))
    keep = torch.zeros_like(m)
    for L in range(lvls):
        lo, hi = 2 ** L, 2 ** (L + 1)
        band = (d >= lo) & (d < hi)
        first = band & (band.cumsum(-1) <= 2)      # 2 representatives per octave
        keep |= first
    keep |= (d > 0) & (d <= 4)                      # dense near field
    return keep & causal(s, device)


def sgate(q, k, mask):
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(w.dtype).min
    pp = torch.softmax(w.masked_fill(~mask, neg), -1).masked_fill(~mask, 0.0)
    pm = torch.softmax((-w).masked_fill(~mask, neg), -1).masked_fill(~mask, 0.0)
    return RHO * (pp - LAM * pm) / (1.0 + LAM)


def flip_rate(sched, s, n_draws=256, d=16, hops=2, seed=0, device=None,
              floor=1e-6, rel=0.0):
    """Fraction of draws where changing token c flips the sign of j's influence
    on i. Relative positions held fixed as s grows, exactly as the dense sweep."""
    dev = device or torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    i, j, c = s - 1, 1, s // 2
    mask = sched(s, dev)
    pairs = []
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        grads = []
        for cv in (rnd(d), rnd(d)):
            x = x0.clone(); x[c] = cv
            v = v0.clone().requires_grad_(True)
            a = sgate(x @ wq, x @ wk, mask)
            h, term = v, v
            for _ in range(hops):
                term = a @ term
                h = h + term
            grad, = torch.autograd.grad((h @ wo)[i].sum(), v)
            grads.append(float(grad[j].sum()))
        pairs.append((grads[0], grads[1]))
        # ONE discard rule, ONE place. This line used to be a verbatim copy of
        # `ceq/bench.py`'s, which is how a single ABSOLUTE floor came to be
        # applied unaudited across four files at once (Inspector S2/S3/S4).
        # `rel` reaches the scale-relative criterion; see `bench.flip_rate`.
    return bench.flip_rate(pairs, floor, rel)


def main():
    dev = torch.device("cpu")
    scheds = {
        "dense (Foreman)": lambda s, d: causal(s, d),
        "window+sinks k=8": lambda s, d: window_sinks(s, 8, 2, d),
        "window+sinks k=16": lambda s, d: window_sinks(s, 16, 2, d),
        "multizoom ~log s": multizoom,
    }
    S = (8, 16, 32, 64, 128)
    print(f"content-conditional sign rate vs context. 256 draws, hops=2, d=16.\n")
    print(f"{'schedule':>20} " + "".join(f"{s:>9}" for s in S) + f"{'  slope':>10}")
    for name, f in scheds.items():
        rates = [flip_rate(f, s, device=dev) for s in S]
        xs = [math.log(s) for s in S]
        ys = [math.log(max(r, 1e-4)) for r in rates]
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
        den = sum((a-mx)**2 for a in xs)
        slope = num/den if den else 0.0
        print(f"{name:>20} " + "".join(f"{r:>9.5f}" for r in rates) + f"{slope:>10.3f}")
    print("\n  dense slope was -1.389 (Foreman, 1024 draws).")
    print("  a schedule that arrests the decay shows a slope near 0.")


if __name__ == "__main__":
    main()
