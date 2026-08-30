"""Hierarchical (self-similar) reachability with hops scaled to depth.

THE DECAY, restated in the form that says what to change:
    rate ~ 1 / |N_in(i) ∩ N_out(j)|
the number of intermediates reachable FROM j that also reach i. Dense causal
attention on a line makes that intersection ~s, hence 1/s.

THE IDEA. Embed positions on a self-similar structure. Common ancestors are O(1)
per level over O(log s) levels, so the intersection is O(log s), not O(s).

THE CATCH, and it is why window+sinks died. On a hierarchy c reaches i in
O(log s) hops, not 2. The property lives in the 2-hop term only because that is
the shortest path containing a third token; on a tree the shortest such path is
the tree distance. Fix hops at 2 and c is not diluted, it is SEVERED.

So hops must scale with depth. Which forces the control that decides whether the
structure matters at all:

    dense + hops = log2(s)

If raising hops alone flattens the curve, the fractal is decoration.

c is placed UNIFORMLY AT RANDOM and never inserted into any schedule by hand.
SCREENING-ONLY: synthetic Q/K/V, python reference attention.
"""
from __future__ import annotations
import sys, math, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ceq import bench

RHO, LAM = 1.5, 0.10


def causal(s, dev):
    return torch.ones(s, s, dtype=torch.bool, device=dev).tril(-1)


def dyadic_ancestors(s, dev):
    """Self-similar reachability: position i reaches, at each dyadic level L, the
    block-aligned representative of its own level-L ancestor block, plus the
    near field. Branching is constant per level, depth is log2(s), so the common
    -ancestor set of any pair is O(log s)."""
    m = torch.zeros(s, s, dtype=torch.bool, device=dev)
    idx = torch.arange(s, device=dev)
    for L in range(int(math.log2(max(s, 2))) + 1):
        blk = 1 << L
        anc = (idx // blk) * blk                    # start of i's level-L block
        for off in (0, blk // 2):                   # 2 representatives per level
            tgt = (anc + off).clamp(max=s - 1)
            m[idx, tgt] = True
    m &= causal(s, dev)
    m |= causal(s, dev) & ((idx[:, None] - idx[None, :]) <= 2)   # dense near field
    return m


def sgate(q, k, mask):
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(w.dtype).min
    pp = torch.softmax(w.masked_fill(~mask, neg), -1).masked_fill(~mask, 0.0)
    pm = torch.softmax((-w).masked_fill(~mask, neg), -1).masked_fill(~mask, 0.0)
    return RHO * (pp - LAM * pm) / (1.0 + LAM)


def run(sched, s, hops, n_draws=160, d=16, seed=0, floor=1e-6, rel=0.0):
    dev = torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    mask = sched(s, dev)
    i, j = s - 1, 0
    pairs, reach = [], 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        c = int(torch.randint(1, s - 1, (1,), generator=g))     # UNIFORMLY RANDOM
        # is c on ANY path of length <= hops from j into i?
        r, ok = mask.clone(), False
        cur = mask[i].clone()
        for _ in range(hops - 1):
            if bool(cur[c]): ok = True
            cur = (mask[cur].any(0)) | cur
        reach += int(ok or bool(mask[i, c]))
        grads = []
        for cv in (rnd(d) * 0.5, rnd(d) * 0.5):
            x = x0.clone(); x[c] = cv
            v = v0.clone().requires_grad_(True)
            a = sgate(x @ wq, x @ wk, mask)
            h, t = v, v
            for _ in range(hops):
                t = a @ t; h = h + t
            gr, = torch.autograd.grad((h @ wo)[i].sum(), v)
            grads.append(float(gr[j].sum()))
        pairs.append((grads[0], grads[1]))
        # ONE discard rule, ONE place. This line used to be a verbatim copy of
        # `ceq/bench.py`'s, which is how a single ABSOLUTE floor came to be
        # applied unaudited across four files at once (Inspector S2/S3/S4).
        # `rel` reaches the scale-relative criterion; see `bench.flip_rate`.
    return bench.flip_rate(pairs, floor, rel), reach / n_draws


def main():
    S = (16, 32, 64, 128, 256)
    arms = [
        ("dense, hops=2",          lambda s: causal(s, torch.device("cpu")), lambda s: 2),
        ("dense, hops=log2 s",     lambda s: causal(s, torch.device("cpu")), lambda s: int(math.log2(s))),
        ("carpet, hops=2",         lambda s: dyadic_ancestors(s, torch.device("cpu")), lambda s: 2),
        ("carpet, hops=log2 s",    lambda s: dyadic_ancestors(s, torch.device("cpu")), lambda s: int(math.log2(s))),
    ]
    print("c uniformly random. 160 draws. rate / 2-hop-reachability.\n")
    print(f"{'arm':>22} " + "".join(f"{s:>15}" for s in S) + f"{'  slope':>9}")
    for name, sf, hf in arms:
        cells, rates = [], []
        for s in S:
            r, rc = run(lambda ss, dv: sf(ss), s, hf(s))
            cells.append(f"{r:.4f}/{rc:.2f}"); rates.append(r)
        xs = [math.log(s) for s in S]; ys = [math.log(max(r, 1e-4)) for r in rates]
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        num = sum((a-mx)*(b-my) for a, b in zip(xs, ys)); den = sum((a-mx)**2 for a in xs)
        print(f"{name:>22} " + "".join(f"{c:>15}" for c in cells) + f"{num/den:>9.3f}")
    print("\n  carpet flat AND dense-log-hops decaying => the structure does the work.")
    print("  both flat => hops did it, the fractal is decoration.")
    print("  mean reachable keys per row:", end=" ")
    for s in (64, 256):
        print(f"s={s}: carpet {float(dyadic_ancestors(s, torch.device('cpu'))[1:].sum(-1).float().mean()):.1f}"
              f" dense {float(causal(s, torch.device('cpu'))[1:].sum(-1).float().mean()):.1f}", end="  ")
    print()


if __name__ == "__main__":
    main()
