"""DeltaFloor falsifier, rebuilt. The first version's slope was an artifact.

WHAT WAS WRONG WITH v1. It kept `min(4, nblk)` blocks while `nblk = s//32` grew
with s. At s=64 that is min(4,2)=2 of 2 blocks and at s=128 it is 4 of 4 --
EVERY BLOCK KEPT, so recall is 1.000 by construction, not by measurement. Its own
output said so: "chance for top-4 of s/32 blocks: s=64: 2.000". A chance level
above 1 is a tell that the arm cannot fail. The reported slope -0.708 was fitted
through two structural constants and one real number.

This is the seventh instrument in this project that was internally consistent and
externally wrong, and the second (after the multizoom Delta=-0.424) whose "result"
was True at every point tested by construction.

THE FIX. Hold the KEEP FRACTION fixed at 1/4 instead of the block count, so
chance is 0.250 at every s and the slope is measured against a flat baseline.
Every point below is informative; none is ceilinged.

KILL NUMBERS, unchanged from v1 and still fixed in advance:
  recall at s=256 <= 0.48
  log-log recall-vs-s slope < -0.1

SCREENING-ONLY: synthetic Q/K/V, python reference reader.
"""
from __future__ import annotations
import sys, math, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BLOCK = 32
KEEP_FRAC = 0.25


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def reader(x, wq, wk, wv, keep):
    s = x.shape[0]
    q, k, v = x @ wq, x @ wk, x @ wv
    m = torch.ones(s, s, dtype=torch.bool).tril(0) & keep[None, :]
    w = (q @ k.T) / math.sqrt(x.shape[1])
    a = torch.softmax(w.masked_fill(~m, torch.finfo(w.dtype).min), -1)
    return a.masked_fill(~m, 0.0) @ v


def floor_m(out, tol):
    """caustic T1's m: distinct readings by greedy single-linkage.

    Vectorised against v1's list-comprehension `any()`, which was the cost that
    kept v1 from reaching s=1024 at all. Same greedy order, same answer.
    """
    reps = out[:1]
    for r in out[1:]:
        if float((r - reps).abs().amax(-1).min()) >= tol:
            reps = torch.cat([reps, r[None]], 0)
    return reps.shape[0]


def score_dfloor(x, wq, wk, wv, nblk, tol):
    s = x.shape[0]
    full = torch.ones(s, dtype=torch.bool)
    m_full = floor_m(reader(x, wq, wk, wv, full), tol)
    out = torch.zeros(nblk)
    for b in range(nblk):
        keep = full.clone()
        keep[b * BLOCK:(b + 1) * BLOCK] = False
        if bool(keep.any()):
            out[b] = m_full - floor_m(reader(x, wq, wk, wv, keep), tol)
    return out


def score_cosine(x, wq, wk, nblk):
    q, k = x @ wq, x @ wk
    g = (q[-1:] @ k.T).abs().squeeze(0) / (q[-1:].norm() * k.norm(dim=-1) + 1e-30)
    return torch.stack([g[b * BLOCK:(b + 1) * BLOCK].max() for b in range(nblk)])


def run(kind, s, n_draws, d=16, seed=0, tol=0.35):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    nblk = s // BLOCK
    kb = max(1, int(round(nblk * KEEP_FRAC)))
    hits = 0
    for _ in range(n_draws):
        wq, wk, wv = rnd(d, d), rnd(d, d), rnd(d, d)
        x = rnd(s, d)
        c = int(torch.randint(0, s, (1,), generator=g))
        x[c, 0] += 6.0
        if kind == "dfloor":
            sc = score_dfloor(x, wq, wk, wv, nblk, tol)
        elif kind == "cosine":
            sc = score_cosine(x, wq, wk, nblk)
        else:
            sc = torch.rand(nblk, generator=g)
        hits += int((torch.topk(sc, kb).indices == c // BLOCK).any())
    return hits, n_draws, kb, nblk


def main():
    SS = (128, 256, 512, 1024)
    N = 96
    print(f"planted carrier, uniform position. block={BLOCK}, keep fraction={KEEP_FRAC} "
          f"(CONSTANT -- v1 kept a constant block COUNT and ceilinged itself)")
    print(f"n={N} draws per cell. KILL: recall <= 0.48 at s=256, or slope < -0.1.\n")
    hdr = "".join(f"{s:>24}" for s in SS)
    print(f"{'selector':>10} {hdr}")
    print(f"{'':>10} " + "".join(f"{'recall  [95% CI]':>24}" for _ in SS))
    for kind in ("dfloor", "cosine", "random"):
        cells, rec, geom = [], [], []
        for s in SS:
            k, tot, kb, nblk = run(kind, s, N)
            r = k / tot
            lo, hi = clopper_pearson(k, tot)
            rec.append(max(r, 1e-4)); geom.append((kb, nblk))
            cells.append(f"{r:.3f} [{lo:.3f},{hi:.3f}]")
        xs = [math.log(v) for v in SS]; ys = [math.log(v) for v in rec]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        sl = sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)
        print(f"{kind:>10} " + "".join(f"{c:>24}" for c in cells) + f"  slope {sl:+.3f}")
    print("\n  keep/total blocks per s: " + "  ".join(f"s={s}: {g[0]}/{g[1]}" for s, g in zip(SS, geom)))
    print("  chance (flat by construction): " + "  ".join(f"s={s}: {g[0]/g[1]:.3f}" for s, g in zip(SS, geom)))


if __name__ == "__main__":
    main()
