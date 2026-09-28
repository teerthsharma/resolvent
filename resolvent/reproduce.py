"""Recompute the README's results table.

    python -m resolvent.reproduce gap        # Bayes - latent distance vs shared error, K = 4   (~1 min CPU)
    python -m resolvent.reproduce bound      # D-JEPA Cor 1 ceiling vs Bayes at sigma = 1, 10    (~20 s CPU)
    python -m resolvent.reproduce closure    # learned heads at sigma = 1, 36 epochs            (minutes CPU)
"""
import argparse
import json
from functools import cache

import numpy as np

from . import shared_error as S


def cell(sigma, seed, n=4000, mix=0.0, max_rank=None):
    """(bed, Bayes pick) for one sigma and seed, memoised: gap, bound and the tests share the Monte Carlo."""
    return _cell(float(sigma), int(seed), int(n), float(mix), max_rank)


@cache
def _cell(sigma, seed, n, mix, max_rank):
    b = S.make_bed(n, sigma, seed=seed, mix=mix)
    return b, S.bayes_pick(b, max_rank=max_rank)


def gap(sigmas=(0.1, 1.0, 3.0, 10.0, 30.0), seeds=(0, 1, 2), n=4000):
    out = {}
    for sg in sigmas:
        rows = []
        for sd in seeds:
            b, bp = cell(sg, sd, n)
            rows.append((S.hit(bp, b), S.hit(S.dist_pick(b["zhat_true"]), b)))
        bayes, dist = np.mean(rows, 0)
        out[sg] = dict(bayes=float(bayes), dist=float(dist), gap=float(bayes - dist))
    return out


def bound(sigmas=(1.0, 10.0), seeds=(0, 1, 2), n=4000):
    """Bayes vs the bound-restricted Bayes rule (plug-in distance ranks {0, 1}: D-JEPA's Cor 1 reach at K = 4)."""
    out = {}
    for sg in sigmas:
        rows = []
        for sd in seeds:
            (b, bp), (_, bb) = cell(sg, sd, n), cell(sg, sd, n, max_rank=1)
            rows.append((S.hit(bp, b), S.hit(bb, b)))
        bayes, bounded = np.mean(rows, 0)
        out[sg] = dict(bayes=float(bayes), bounded=float(bounded))
    return out


def closure(sigma=1.0, seeds=(0, 1, 2), epochs=36):
    r = S.run_cell(sigma, seeds=seeds, epochs=epochs)
    mean = {f: {k: float(np.mean(v)) for k, v in r[f].items()} for f in ("hit", "agree", "closure")}
    return dict(sigma=sigma, bayes=float(np.mean(r["bayes"])), dist=float(np.mean(r["dist"])), per_seed=r, **mean)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m resolvent.reproduce")
    p.add_argument("what", choices=("gap", "bound", "closure"))
    p.add_argument("--sigmas", type=float, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n", type=int, default=4000)
    a = p.parse_args(argv)
    if a.what == "closure":
        out = closure(a.sigmas[0] if a.sigmas else 1.0, tuple(a.seeds))
    else:
        f = gap if a.what == "gap" else bound
        kw = dict(seeds=tuple(a.seeds), n=a.n) | (dict(sigmas=tuple(a.sigmas)) if a.sigmas else {})
        out = {str(k): v for k, v in f(**kw).items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
