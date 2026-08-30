"""Dr House's experiment: is it routing, or a shrunken denominator?

HIS IDENTITY. For any CONTENT-BLIND schedule of size k,
    P(c reachable) * (share | reachable) = (k/s) * (1/k) = 1/s
bit-for-bit the dense rate. Sparsity moves the decay from the share factor into
the RECALL factor; it does not remove it. A schedule-conditioned measurement comes
back flat either way and proves nothing.

THE QUANTITY THAT IS NOT AN ARTIFACT: whether P(c selected) stays Theta(1) as s
grows when selection is CONTENT-CONDITIONAL. That is routing, and a random
schedule provably cannot fake it.

DESIGN, and the fairness point. c is placed UNIFORMLY AT RANDOM every draw and is
never inserted into the schedule by hand. For a content gate to have any reason to
find it, c must be content-DISTINGUISHED -- so it carries a planted feature, the
same shape as the caustic corpus's control channel. A gate that cannot find a
planted token cannot find a real one.

THE GATE is CE-BB's, verbatim in form: G_j = max |q.k| / (||q|| ||k||), keep top-k.

THREE ARMS:
  signed + content gate   -- the claim
  signed + random, same k -- the denominator control
  softmax + content gate  -- must read 0.0000, or sparsity manufactures the property
"""
from __future__ import annotations
import sys, math, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ceq import bench

RHO, LAM = 1.5, 0.10


def gate_topk(q, k, kbud, i):
    """CE-BB: cosine-magnitude score over j < i, keep top-k. Content-conditional."""
    qi = q[i:i + 1]
    kk = k[:i]
    g = (qi @ kk.T).abs().squeeze(0) / (qi.norm() * kk.norm(dim=-1) + 1e-30)
    return torch.topk(g, min(kbud, i)).indices


def random_topk(i, kbud, gen):
    return torch.randperm(i, generator=gen)[:min(kbud, i)]


def run(arm, s, kbud, n_draws=192, d=16, hops=2, seed=0, floor=1e-6, rel=0.0):
    """BUG FIXED, twice, and both bugs inflated the random arm.

    1. The random schedule was drawn from the SHARED generator, so the two
       c-values got DIFFERENT schedules and every 'flip' was schedule noise
       rather than c's content effect. The random arm now draws its schedule
       from a per-draw generator reused across both c-values, so the schedule is
       identical and only c differs -- which is the whole comparison.
    2. Recall was `m[i,c] or m[:,c].any()` -- "some row selected c" -- which with
       k=12 over s rows is near 1 by construction. It now means what it has to
       mean: c lies on a TWO-HOP path into i, i.e. i selects some m that selects
       c. That is the only path by which c can carry the property at hops=2.
    """
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    pairs, recall = [], 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        i, j = s - 1, 0
        c = int(torch.randint(1, s - 1, (1,), generator=g))   # UNIFORMLY RANDOM
        x0[c, 0] += 6.0                                       # planted, findable
        grads = []
        sched_seed = int(torch.randint(1 << 30, (1,), generator=g))
        reach = False
        for cv in (rnd(d) * 0.5, rnd(d) * 0.5):
            x = x0.clone(); x[c, 1:] = cv[1:]
            v = v0.clone().requires_grad_(True)
            q, kk = x @ wq, x @ wk
            m = torch.zeros(s, s, dtype=torch.bool)
            sg = torch.Generator().manual_seed(sched_seed)   # SAME for both c-values
            for r in range(1, s):
                sel = (gate_topk(q, kk, kbud, r) if arm.endswith("gate")
                       else random_topk(r, kbud, sg))
                m[r, sel] = True
            if arm.startswith("softmax"):
                w = (q @ kk.T) / math.sqrt(d)
                a = torch.softmax(w.masked_fill(~m, torch.finfo(w.dtype).min), -1)
                a = a.masked_fill(~m, 0.0)
            else:
                w = (q @ kk.T) / math.sqrt(d)
                neg = torch.finfo(w.dtype).min
                pp = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
                pm = torch.softmax((-w).masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
                a = RHO * (pp - LAM * pm) / (1.0 + LAM)
            h, t = v, v
            for _ in range(hops):
                t = a @ t; h = h + t
            gr, = torch.autograd.grad((h @ wo)[i].sum(), v)
            grads.append(float(gr[j].sum()))
            # c on a TWO-HOP path into i: i selects some m, and that m selects c
            reach = reach or bool(m[i, c]) or bool((m[i] & m[:, c]).any())
        recall += int(reach)
        pairs.append((grads[0], grads[1]))
        # ONE discard rule, ONE place. This line used to be a verbatim copy of
        # `ceq/bench.py`'s, which is how a single ABSOLUTE floor came to be
        # applied unaudited across four files at once (Inspector S2/S3/S4).
        # `rel` reaches the scale-relative criterion; see `bench.flip_rate`.
    return bench.flip_rate(pairs, floor, rel), recall / n_draws


def main():
    K = 12
    S = (32, 64, 128, 256)
    print(f"c placed UNIFORMLY AT RANDOM, never inserted by hand. k={K} fixed. 192 draws.\n")
    print(f"{'arm':>24} " + "".join(f"{s:>16}" for s in S))
    print(f"{'':>24} " + "".join(f"{'rate / recall':>16}" for _ in S))
    for arm in ("signed + gate", "signed + random", "softmax + gate"):
        cells = []
        for s in S:
            r, rec = run(arm, s, K)
            cells.append(f"{r:.4f}/{rec:.3f}")
        print(f"{arm:>24} " + "".join(f"{c:>16}" for c in cells))
    print("\n  gate flat + random decaying => routing, and the gap IS the recall curve.")
    print("  both decaying => the gate has no recall on the carrier; every flat number was the pool.")
    print("  SCREENING-ONLY: synthetic Q/K/V, python reference attention.")


if __name__ == "__main__":
    main()
