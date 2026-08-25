"""Dr House's falsifier: a CONSEQUENCE-scored selector, not a resemblance one.

WHY THE LAST TEST PROVED NOTHING. The only c-detector ever tried was a cosine
gate -- G_j = max|q.k|/(||q|| ||k||) -- which is a RESEMBLANCE score. It read
recall 0.432 against random's 0.406 at s=256. That is the module's own thesis
restated, not refuted: similarity does not find consequence.

THE SCORE. caustic Theorem 1: for an injective ground relation on n entities
producing m distinct values, err >= n - m, computable with NO ground truth.
Point it at blocks:

    DeltaFloor(b) = m(with b) - m(without b)

i.e. how many downstream readings STOP BEING DISTINGUISHABLE when b is removed.
Positive means b was holding outcomes apart -- it was causally load-bearing.
Resemblance appears nowhere.

WHY EVICTION AND NOT GATING. Removal is the only bitwise counterfactual measured
in this project: perturbing a crushed token moves the state 0.000000e+00 over
24/24 draws under eviction and 2.154868e-05 under post-softmax gating, because a
gate cannot leave the softmax denominator. So do(remove b) is an intervention
PERFORMED, not modelled -- and the modelled-intervention arm is the one that read
2.6151 OOD, worse than predicting the mean.

KILL NUMBERS, fixed before running:
  recall at s=256 <= 0.48                      (inside the measured 0.406+eps band)
  log-log recall-vs-s slope < -0.1 over 64/256/1024   (a decaying detector is the
                                                       -1.389 death under a new name)

SCREENING-ONLY: synthetic Q/K/V, python reference reader.
"""
from __future__ import annotations
import sys, math, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BLOCK = 32


def clopper_pearson(k, n, alpha=0.05):
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
        return float(lo), float(hi)
    except Exception:
        return float("nan"), float("nan")


def reader(x, wq, wk, wv, keep):
    """Frozen reader. `keep` is a bool mask over KEY positions -- a block that is
    False is EVICTED, absent from the softmax denominator entirely."""
    s = x.shape[0]
    q, k, v = x @ wq, x @ wk, x @ wv
    m = torch.ones(s, s, dtype=torch.bool).tril(0) & keep[None, :]
    w = (q @ k.T) / math.sqrt(x.shape[1])
    a = torch.softmax(w.masked_fill(~m, torch.finfo(w.dtype).min), -1)
    return (a.masked_fill(~m, 0.0) @ v)


def floor_m(out, tol):
    """caustic T1's m: the number of DISTINCT readings, greedy single-linkage."""
    reps = []
    for r in out:
        if not any(float((r - p).abs().max()) < tol for p in reps):
            reps.append(r)
    return len(reps)


def score_dfloor(x, wq, wk, wv, nblk, tol):
    """DeltaFloor per block: m(with) - m(without), by EXACT eviction."""
    s = x.shape[0]
    full = torch.ones(s, dtype=torch.bool)
    m_full = floor_m(reader(x, wq, wk, wv, full), tol)
    out = torch.zeros(nblk)
    for b in range(nblk):
        keep = full.clone()
        keep[b * BLOCK:(b + 1) * BLOCK] = False
        if not bool(keep.any()):
            continue
        out[b] = m_full - floor_m(reader(x, wq, wk, wv, keep), tol)
    return out


def score_cosine(x, wq, wk, nblk):
    """The resemblance control that read 0.432."""
    s = x.shape[0]
    q, k = x @ wq, x @ wk
    g = (q[-1:] @ k.T).abs().squeeze(0) / (q[-1:].norm() * k.norm(dim=-1) + 1e-30)
    return torch.stack([g[b * BLOCK:(b + 1) * BLOCK].max() for b in range(nblk)])


def run(kind, s, n_draws=64, d=16, topk_blocks=4, seed=0, tol=0.35):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    nblk = s // BLOCK
    hits = 0
    for _ in range(n_draws):
        wq, wk, wv = rnd(d, d), rnd(d, d), rnd(d, d)
        x = rnd(s, d)
        c = int(torch.randint(0, s, (1,), generator=g))
        x[c, 0] += 6.0                                   # planted carrier
        if kind == "dfloor":
            sc = score_dfloor(x, wq, wk, wv, nblk, tol)
        elif kind == "cosine":
            sc = score_cosine(x, wq, wk, nblk)
        else:
            sc = torch.rand(nblk, generator=g)
        top = torch.topk(sc, min(topk_blocks, nblk)).indices
        hits += int((top == c // BLOCK).any())
    return hits, n_draws


def main():
    print(f"planted carrier, uniformly random position. block={BLOCK}, top-4 blocks kept.")
    print(f"KILL: recall <= 0.48 at s=256, or slope < -0.1 over s=64/256/1024.\n")
    print(f"{'selector':>10} " + "".join(f"{s:>26}" for s in (64, 128, 256)))
    print(f"{'':>10} " + "".join(f"{'recall  [95% CI]':>26}" for _ in range(3)))
    curves = {}
    for kind in ("dfloor", "cosine", "random"):
        cells, rec = [], []
        for s in (64, 128, 256):
            n = 48 if s < 256 else 24
            k, tot = run(kind, s, n_draws=n)
            r = k / tot
            lo, hi = clopper_pearson(k, tot)
            rec.append(max(r, 1e-4))
            cells.append(f"{r:.3f} [{lo:.3f},{hi:.3f}]")
        curves[kind] = rec
        xs = [math.log(v) for v in (64, 128, 256)]
        ys = [math.log(v) for v in rec]
        mx, my = sum(xs)/3, sum(ys)/3
        sl = sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)
        print(f"{kind:>10} " + "".join(f"{c:>26}" for c in cells) + f"   slope {sl:+.3f}")
    print(f"\n  chance for top-4 of s/{BLOCK} blocks: "
          + "  ".join(f"s={s}: {4/(s//BLOCK):.3f}" for s in (64, 256, 1024)))


if __name__ == "__main__":
    main()
