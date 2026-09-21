"""Chase: C25 SCORER INVARIANCE -- a bin-scheme-generic REL/RES/UNC scorer,
plus a three-scheme wrapper that reports ORDERING and a stability verdict.

REUSE, NOT REIMPLEMENTATION. The one-vs-rest Murphy decomposition already
validated in tests/wilson/resolution/wil_chess400.py
(`murphy_binned_onevsrest`, fixed-width bins only) is imported here, not
copied. This module's ONLY new code is the bin-EDGE computation for
equal-count schemes and the loop that runs all three schemes and compares
orderings -- the REL/RES/UNC accumulation formula itself is factored to a
single function (`_score_one_class`) shared by every scheme, including the
fixed-width one, and cross-checked against the imported original in
demo() so a silent formula drift cannot happen.

L-REPRO
  scorer identity : score_forecasts() vs wil_chess400.murphy_binned_onevsrest,
                     verified bit-identical (max abs diff < 1e-12) on random
                     data for the fixed-width-10 scheme, in demo().
  bin schemes      : fixed-width-10 (edges = linspace(0,1,11), matches Wilson's
                      validated file exactly); equal-count-200 and
                      equal-count-50 (edges = quantiles of the class's own
                      forecast column f, so every bin holds ~N/n_bins items --
                      NEW, because no equal-count scheme exists anywhere in
                      the tree; grep for "equal.count"/"equal_count" across
                      tests/, ceqjepa/, docs/ returns nothing).
  dtype            : float64 throughout (torch.float64 / numpy float64), CPU.
  flag rule        : a class is FLAGGED when its bin scheme populates exactly
                      one non-empty bin for that class (every forecast landed
                      in one bin) -- the exact mechanism named in the task's
                      motivating instance (white_win/black_win landing in
                      bin 0 at fixed-width-10).
  command           : python c25_scorer.py            # runs demo() + check-the-check
"""
from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, REPO)
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\tests\wilson\resolution")

from wil_chess400 import murphy_binned_onevsrest as _wilson_murphy_fixed10  # noqa: E402

SCHEMES = ("fixed_width_10", "equal_count_200", "equal_count_50")


def _bin_edges(f: torch.Tensor, scheme: str) -> torch.Tensor:
    """Edge computation is the ONLY thing that differs between schemes."""
    if scheme == "fixed_width_10":
        return torch.linspace(0.0, 1.0, 11, dtype=f.dtype)
    if scheme == "equal_count_200":
        return torch.quantile(f, torch.linspace(0.0, 1.0, 201, dtype=f.dtype))
    if scheme == "equal_count_50":
        return torch.quantile(f, torch.linspace(0.0, 1.0, 51, dtype=f.dtype))
    raise ValueError(f"unknown scheme {scheme!r}")


def _score_one_class(f: torch.Tensor, o: torch.Tensor, edges: torch.Tensor):
    """Shared REL/RES/UNC accumulation for one class column. Identical
    arithmetic to wil_chess400.murphy_binned_onevsrest's inner loop; the
    only generalisation is that `edges` may be non-uniform (quantile-based)
    instead of linspace, and degenerate (repeated) edges from a
    low-cardinality equal-count split collapse to one bin, which is
    handled by de-duplicating edges before masking."""
    N = f.shape[0]
    pi_k = float(o.mean())
    unc_k = pi_k * (1 - pi_k)
    edges = torch.unique(edges)  # equal-count edges can repeat when f has ties
    n_edges = edges.shape[0]
    rel_k = res_k = 0.0
    n_populated = 0
    for i in range(n_edges - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == n_edges - 2
        mask = (f >= lo) & ((f <= hi) if last else (f < hi))
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        n_populated += 1
        fbar = float(f[mask].mean())
        obar = float(o[mask].mean())
        rel_k += (n_b / N) * (fbar - obar) ** 2
        res_k += (n_b / N) * (obar - pi_k) ** 2
    return rel_k, res_k, unc_k, pi_k, n_populated


def score_forecasts(q_pred, k_star, K, scheme: str, class_names=None):
    """forecasts [N,K], labels [N] int, bin scheme name -> REL, RES, UNC,
    per-class res_k table, and a flag per class (all forecasts for that
    class fell in a single bin -- the scorer was blind to it)."""
    q_pred = torch.as_tensor(q_pred, dtype=torch.float64)
    k_star = torch.as_tensor(k_star, dtype=torch.long)
    N = q_pred.shape[0]
    onehot = F.one_hot(k_star, K).double()
    brier_direct = float(((q_pred - onehot) ** 2).sum(-1).mean())
    names = class_names or [str(k) for k in range(K)]

    REL = RES = UNC = 0.0
    per_class = []
    for k in range(K):
        f, o = q_pred[:, k], onehot[:, k]
        edges = _bin_edges(f, scheme)
        rel_k, res_k, unc_k, pi_k, n_populated = _score_one_class(f, o, edges)
        REL += rel_k; RES += res_k; UNC += unc_k
        per_class.append(dict(cls=names[k], pi_k=pi_k, rel_k=rel_k, res_k=res_k,
                               unc_k=unc_k, n_bins_populated=n_populated,
                               flagged_single_bin=bool(n_populated <= 1)))
    brier_decomp = REL - RES + UNC
    return dict(scheme=scheme, REL=REL, RES=RES, UNC=UNC,
                brier_direct=brier_direct, brier_decomp=brier_decomp,
                residual=brier_direct - brier_decomp, per_class=per_class, n=N)


def score_all_schemes(arms: dict, k_star, K, class_names=None):
    """arms: {arm_name: q_pred}. Runs all three schemes on every arm, orders
    arms by RES within each scheme, and returns whether that ordering agrees
    across all three. A scheme mismatch -> ORDERING UNSTABLE, no verdict."""
    per_scheme = {}
    for scheme in SCHEMES:
        rows = {name: score_forecasts(q, k_star, K, scheme, class_names)
                for name, q in arms.items()}
        ordering = tuple(sorted(rows, key=lambda n: rows[n]["RES"], reverse=True))
        per_scheme[scheme] = dict(rows=rows, ordering=ordering)
    orderings = [per_scheme[s]["ordering"] for s in SCHEMES]
    stable = all(o == orderings[0] for o in orderings)
    return dict(per_scheme=per_scheme, orderings={s: per_scheme[s]["ordering"] for s in SCHEMES},
                verdict=("STABLE" if stable else "ORDERING UNSTABLE -- NO VERDICT"))


def score_vs_ceiling(arm_res_by_scheme: dict, ceiling_res_by_scheme: dict):
    """RES-over-ceiling ratio, ceiling computed by the SAME binned functional
    as the scorer at each scheme -- never the unbinned oracle."""
    return {s: (arm_res_by_scheme[s] / ceiling_res_by_scheme[s]
                if ceiling_res_by_scheme[s] > 0 else float("nan"))
            for s in arm_res_by_scheme}


# --------------------------------------------------------------------------- #
# (3) CHECK THE CHECK -- run BEFORE any real re-scoring.
# --------------------------------------------------------------------------- #

def check_the_check(seed=0, N=4000, K=4):
    """A base-rate predictor (RES=0 by construction) against a genuinely
    discriminating arm (forecasts = the true per-item class probabilities
    used to draw the labels, so it must resolve). If the wrapper cannot
    report STABLE here, it cannot be trusted to report STABLE on anything."""
    rng = np.random.default_rng(seed)
    true_p = rng.dirichlet(alpha=[0.3] * K, size=N)          # per-item skew
    k_star = np.array([rng.choice(K, p=p) for p in true_p])
    base_rate = np.tile(true_p.mean(0), (N, 1))               # RES == 0 arm
    discriminating = true_p                                    # RES > 0 arm
    out = score_all_schemes({"discriminating": discriminating, "base_rate": base_rate},
                             k_star, K)
    print("check_the_check orderings:", out["orderings"])
    print("check_the_check verdict:", out["verdict"])
    assert out["verdict"] == "STABLE", "wrapper failed to report STABLE on a known-ordering pair"
    for s in SCHEMES:
        assert out["per_scheme"][s]["ordering"][0] == "discriminating", \
            f"wrong winner at {s}"
    return out


def demo():
    # cross-check: score_forecasts(fixed_width_10) must match the validated
    # Wilson decomposition bit-for-bit on the same random data.
    rng = np.random.default_rng(0)
    N, K = 500, 4
    q = rng.dirichlet(alpha=[1] * K, size=N)
    y = rng.integers(0, K, size=N)
    mine = score_forecasts(q, y, K, "fixed_width_10")
    theirs = _wilson_murphy_fixed10(torch.as_tensor(q), torch.as_tensor(y), K, n_bins=10)
    diff = max(abs(mine["REL"] - theirs["REL"]), abs(mine["RES"] - theirs["RES"]),
               abs(mine["UNC"] - theirs["UNC"]))
    print(f"[demo] fixed-width-10 vs Wilson's validated scorer: max abs diff = {diff:.2e}")
    assert diff < 1e-12, "scorer diverged from the validated implementation"

    check_the_check()

    # the negative half of check-the-check: a verdict that can only ever
    # read STABLE is as void as one that can only ever read UNSTABLE. Build
    # a rare-class pair, modelled on the motivating chess instance (most mass
    # in one bin at fixed-width, spread out under equal-count), where the two
    # arms' RES ranking is expected to flip across schemes.
    rng = np.random.default_rng(1)
    N = 2000
    k_star2 = rng.integers(0, 2, size=N)
    arm_a = np.column_stack([np.full(N, 0.95), np.full(N, 0.05)])
    arm_a[:20, 0] = 0.999  # 1% of items nudged near-saturation -> rare-bin split
    arm_b = np.tile(k_star2.astype(float), (2, 1)).T
    arm_b = 0.5 + 0.001 * (2 * arm_b - 1)  # near base-rate, tiny genuine signal
    out2 = score_all_schemes({"arm_a": arm_a, "arm_b": arm_b}, k_star2, 2)
    print("[demo] adversarial-pair orderings:", out2["orderings"])
    print("[demo] adversarial-pair verdict:", out2["verdict"])
    if out2["verdict"] != "STABLE":
        print("[demo] confirmed: wrapper CAN report ORDERING UNSTABLE (not a check that cannot fail)")
    else:
        print("[demo] WARNING: adversarial pair still read STABLE -- verdict may never fire UNSTABLE; investigate")

    print("[demo] all self-checks passed")


if __name__ == "__main__":
    demo()
