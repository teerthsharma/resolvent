"""Phase D: every probe number in this project was taken at random init. This is the delta.

THE OPEN ITEM. Five rounds of statistics -- every kappa, every separation, every
degeneracy, every retention figure -- were measured through randomly initialised
q/k projections. Nothing established that a trained model occupies the same
regime. If these statistics move, a large part of five rounds describes a regime
the model never visits.

WHAT IS MEASURED, RANDOM-INIT AND TRAINED SIDE BY SIDE.

  logit_scale     mean |w| over causal entries, w = (q k^T)/sqrt(d). Round 5's
                  F-lam named the logit scale as the single cause behind four
                  symptoms: the signed operator at every lam, Delta ~ 100 nats,
                  the float32 underflow, and ARM S's vertex collapse. Random-init
                  mean causal |w| was 1.171e+01 against the harness's 2.682399e-03.
  delta_vertex    the EXACT projective diameter of T's image at the extreme rays,
                  via scale/foreman_hilbert.py:vertex_diameter. Random-init runs
                  101.3671 .. 311.6091 nats.
  kappa_cert      tanh(Delta/4), and one_minus_kappa in the closed form
                  2/(exp(Delta/2)+1), never by subtracting from 1.
  alpha_*         the settling weights at the fixed point. Foreman measured
                  log_alpha min -182.7498 max -0.0 at random init -- near one-hot.
                  T1's question is whether a lopsided equilibrium carries anything
                  an argmax-pivot lookup does not, and NO BIRTH GATE ANSWERS IT.
                  Nobody has ever looked at trained.
  pmin / cond     min(p_c, p_j) and the 2-dof conditioning sigma_2/sigma_1. ARM P's
                  strata depend on where the draws land, and min(p_c,p_j) -> 0
                  kills every readout family.

NO DIRECTION IS ASSERTED ANYWHERE IN THIS FILE. Either answer is worth having.
The tests bind that the comparison is CAPABLE of showing a delta -- training moved
the weights, both sides run one code path, and a planted change moves the number --
and assert nothing about which way the measurement comes out.

THE TRAINING IS THE SHIPPED PATH, NOT A REIMPLEMENTATION. `m3_capability.run_arm`
does the training; this file only captures the model object it builds, by swapping
`m3_capability.Arm` for a subclass that records its instance and snapshots the
weights before the first step. The optimiser, the loss, the batches and the metric
are Chase's, untouched. The BIND is `eval_nrmse < nrmse0_eval`: if training did not
improve the arm, no trained number is reported.

A SMALL MODEL IS SUFFICIENT AND THIS FILE SAYS SO. The question is whether these
statistics MOVE between random and trained, not what they equal at scale. The
training config is printed beside every number so nothing here is read as a scale
claim. This is deliberately NOT coupled to any in-flight large run: a measurement
that waits on another agent's schedule inherits that schedule.
"""
from __future__ import annotations

import argparse
import copy
import math
import pathlib
import sys
import typing

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                              # noqa: E402
from scale import foreman_hilbert as FH                            # noqa: E402
from scale import m3_capability as m3                              # noqa: E402
from scale import twodof as TD                                     # noqa: E402
from scale.negation_scope import make_batch                        # noqa: E402
from scale.pivot_probe import select_pivots                        # noqa: E402
from scale.torque_probe import boot_ci                             # noqa: E402

#: Small on purpose. See the module docstring: the question is whether the
#: statistics move, not what they are at scale.
#: The published F-green cell, which is a config that provably trains.
#: steps=60 / n_train=256 was tried first and the bind REFUSED it: the arm
#: got worse, eval 1.272795 against 1.019428 at 0 steps, because 4769
#: parameters on 256 examples overfits. The bind was not lowered to fit it.
#: Weights are cached, so paying for this once costs nothing afterwards.
QUICK_STEPS = 150
QUICK_NTRAIN = 8192
BETA = 0.5                     # foreman_hilbert's own default
SETTLE_ITERS = 200             # round 5 measured convergence in 28-53 steps


class Pair(typing.NamedTuple):
    """One architecture, one init, one batch. Only the projections differ."""

    x: torch.Tensor
    init_wq: torch.Tensor
    init_wk: torch.Tensor
    trained_wq: torch.Tensor
    trained_wk: torch.Tensor
    metrics: dict
    config: dict


def train_pair(*, seed: int = 0, s: int = 64, d: int = 24,
               steps: int = QUICK_STEPS, n_train: int = QUICK_NTRAIN,
               n_eval: int = 256) -> Pair:
    """Train one small arm through the shipped path and keep both weight sets.

    `m3_capability.Arm` is swapped for a subclass that records the instance and
    deep-copies its weights at construction, before the optimiser exists. The
    training loop, optimiser, loss and metric are Chase's `run_arm`, called
    verbatim and restored afterwards.
    """
    cache = (pathlib.Path(__file__).resolve().parents[1] / "results" /
             f"phaseD_weights_s{s}_d{d}_n{n_train}_st{steps}_seed{seed}.pt")
    if cache.exists():
        blob = torch.load(cache, weights_only=True)
        return Pair(x=blob["x"], init_wq=blob["init_wq"], init_wk=blob["init_wk"],
                    trained_wq=blob["trained_wq"], trained_wk=blob["trained_wk"],
                    metrics=blob["metrics"], config=blob["config"])

    x_train, y_train, _, _ = make_batch(n_train, s, d, d_model=m3.D_MODEL, seed=seed)
    x_eval, y_eval, _, _ = make_batch(n_eval, s, d, d_model=m3.D_MODEL,
                                      seed=seed + 12345)
    made = []
    original = m3.Arm

    class Recording(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._init_wq = copy.deepcopy(self.wq.weight.detach())
            self._init_wk = copy.deepcopy(self.wk.weight.detach())
            made.append(self)

    m3.Arm = Recording
    try:
        metrics = m3.run_arm("pivot_unsigned", x_train, y_train, x_eval, y_eval,
                             s=s, steps=steps, seed=seed)
    finally:
        m3.Arm = original
    arm = made[-1]
    pair = Pair(x=x_eval,
                init_wq=arm._init_wq.double(), init_wk=arm._init_wk.double(),
                trained_wq=arm.wq.weight.detach().double(),
                trained_wk=arm.wk.weight.detach().double(),
                metrics=metrics,
                config=dict(seed=seed, s=s, d=d, steps=steps, n_train=n_train,
                            n_eval=n_eval, d_model=m3.D_MODEL, arm="pivot_unsigned",
                            threads=torch.get_num_threads()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pair._asdict(), cache)
    return pair


def effective_support(w: torch.Tensor) -> float:
    """exp(Shannon entropy) of a weight vector. 1 is one-hot, len(w) is uniform.

    Reported instead of a raw max because "0.999 on one pivot" and "0.6 on one
    pivot" are the same max-based story and completely different equilibria. It
    is bounded by the alphabet size, which is asserted in the tests, because a
    spread measure that can exceed its own alphabet is broken.
    """
    p = w.double().clamp_min(0)
    p = p / p.sum()
    nz = p[p > 0]
    return float(torch.exp(-(nz * nz.log()).sum()))


def alpha_at_fixed_point(rows: torch.Tensor, gate: torch.Tensor, beta: float,
                         iters: int = SETTLE_ITERS):
    """The settling weights `w_p(m*)` after iterating T to its fixed point.

    `scale/foreman_hilbert.py:settle` computes `w = gate * <a_p, m>^beta`
    normalised, then returns `w @ rows` renormalised. The weights themselves are
    what T1 asks about, so they are recomputed here at the settled reading using
    that function's own formula rather than a restatement of it.
    """
    m = rows.mean(0)
    m = m / m.sum()
    for _ in range(iters):
        m = FH.settle(rows, gate, m, beta)
    w = gate * (rows @ m).pow(beta)
    return w / w.sum(), m


def _geometry(pair: Pair, trained: bool, idx: int):
    wq = pair.trained_wq if trained else pair.init_wq
    wk = pair.trained_wk if trained else pair.init_wk
    x = pair.x[idx].double()
    return x @ wq.T, x @ wk.T


def _one(pair: Pair, trained: bool, idx: int, k: int, beta: float,
         cj: tuple[int, int], mt: tuple[int, int]) -> dict | None:
    """Every statistic at one example. ONE code path; `trained` picks weights only."""
    q, kk = _geometry(pair, trained, idx)
    s = q.shape[0]
    i = s - 1
    dk = q.shape[-1]

    w = (q @ kk.transpose(-2, -1)) / (dk ** 0.5)
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool))
    logit_scale = float(w[causal].abs().mean())

    piv = select_pivots(kk, min(k, s - 2), exclude=(i,))
    piv = piv[piv > 0]                       # pivot 0 has an empty causal row
    if piv.numel() < 2:
        return None
    a = bench._softmax_operator(q, kk).double()
    rows, p_max = FH.pivot_readings(a, piv)
    gate = a[i, piv].clone()
    gate = gate / gate.sum()

    delta = FH.vertex_diameter(rows, gate, beta)
    kappa = FH.kappa_cert(delta) if hasattr(FH, "kappa_cert") else math.tanh(delta / 4)
    omk = FH.one_minus_kappa(delta)

    alpha, _ = alpha_at_fixed_point(rows, gate, beta)
    la = alpha.clamp_min(1e-300).log()

    c, j = cj
    mm, tgt = mt
    rep = TD.rank_report(q, kk, i=i, c=c, j=j, family="asymmetric")
    abs_i, abs_l3 = TD.leakage_ratio(q, kk, i=i, c=c, j=j, m=mm, target=tgt)

    return dict(logit_scale=logit_scale,
                abs_I=abs_i,
                abs_L3=abs_l3,
                I_exact_zero=float(abs_i == 0.0),
                delta_vertex=delta,
                kappa_cert=kappa,
                one_minus_kappa=omk,
                alpha_eff_support=effective_support(alpha),
                alpha_max=float(alpha.max()),
                log_alpha_min=float(la.min()),
                log_alpha_max=float(la.max()),
                n_pivots=float(piv.numel()),
                pmin=min(rep.p_c, rep.p_j),
                cond=rep.cond,
                rank2=float(rep.rank == 2))


def statistics(pair: Pair, *, trained: bool, k: int = 8, beta: float = BETA,
               examples: int = 24, c: int = 5, j: int = 11,
               m: int = 17, target: int = 3) -> dict:
    """Mean of every statistic over `examples` draws, one code path both sides."""
    vals: dict[str, list] = {}
    for idx in range(min(examples, pair.x.shape[0])):
        r = _one(pair, trained, idx, k, beta, (c, j), (m, target))
        if r is None:
            continue
        for key, v in r.items():
            vals.setdefault(key, []).append(v)
    if not vals:
        raise RuntimeError("no usable example: every draw had < 2 pivots")
    out = {key: sum(v) / len(v) for key, v in vals.items()}
    out["_raw"] = vals
    return out


def _ci(vals):
    finite = [v for v in vals if math.isfinite(v)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    return boot_ci(finite, seed=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--examples", type=int, default=24)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--beta", type=float, default=BETA)
    a = ap.parse_args()

    pair = train_pair(seed=a.seed, s=a.s, d=a.d, steps=a.steps,
                      n_train=a.n_train, n_eval=a.n_eval)
    cfg = pair.config
    print("PHASE D -- RANDOM-INIT vs TRAINED PROJECTIONS, SIDE BY SIDE")
    print(f"  training config: arm={cfg['arm']} s={cfg['s']} d={cfg['d']} "
          f"d_model={cfg['d_model']} steps={cfg['steps']} "
          f"n_train={cfg['n_train']} n_eval={cfg['n_eval']} seed={cfg['seed']} "
          f"threads={cfg['threads']}")
    print(f"  probe config   : examples={a.examples} k={a.k} beta={a.beta} "
          f"settle_iters={SETTLE_ITERS} dtype=float64")
    print("  A SMALL MODEL. The question is whether these statistics MOVE,")
    print("  not what they equal at scale. Do not read these as scale claims.")
    print("  Other agents share this box; NO TIMING IS REPORTED.")
    print()

    print("=== BIND: training must have improved the arm ===")
    m = pair.metrics
    moved_q = float((pair.trained_wq - pair.init_wq).abs().max())
    moved_k = float((pair.trained_wk - pair.init_wk).abs().max())
    print(f"  eval NRMSE {m['nrmse0_eval']:.6f} (0 steps) -> {m['eval_nrmse']:.6f} "
          f"({cfg['steps']} steps)   CI=[{m['ci_lo']:.6f},{m['ci_hi']:.6f}]")
    print(f"  max |dW| wq={moved_q:.6e}  wk={moved_k:.6e}")
    if not (m["eval_nrmse"] < m["nrmse0_eval"] and moved_q > 1e-6 and moved_k > 1e-6):
        print("  TRAINING DID NOT IMPROVE THE ARM. No trained number reported.")
        return 1
    print("  [OK] the trained side is genuinely trained.")
    print()

    rnd = statistics(pair, trained=False, k=a.k, beta=a.beta, examples=a.examples)
    trn = statistics(pair, trained=True, k=a.k, beta=a.beta, examples=a.examples)

    rows = [
        ("logit_scale", "mean causal |w|"),
        ("delta_vertex", "mean Delta_vertex, nats"),
        ("kappa_cert", "mean per-ex tanh(D/4)"),
        ("one_minus_kappa", "mean per-ex 1-kappa"),
        ("alpha_eff_support", "alpha exp(entropy)"),
        ("alpha_max", "alpha max"),
        ("log_alpha_min", "log alpha min"),
        ("log_alpha_max", "log alpha max"),
        ("n_pivots", "pivots used"),
        ("pmin", "min(p_c,p_j)"),
        ("abs_I", "E|I| degree-2"),
        ("abs_L3", "E|L3| degree-3"),
        ("I_exact_zero", "frac I == 0.0 exactly"),
        ("cond", "2-dof sigma2/sigma1"),
        ("rank2", "fraction reading rank 2"),
    ]
    print("=== THE TABLE. Random-init and trained, one code path, same batch ===")
    print(f"  {'statistic':>24} {'random-init':>15} {'trained':>15} "
          f"{'ratio t/r':>12} {'CI excludes overlap':>21}")
    for key, label in rows:
        r, t = rnd[key], trn[key]
        rlo, rhi = _ci(rnd["_raw"][key])
        tlo, thi = _ci(trn["_raw"][key])
        disj = "YES" if (math.isfinite(rhi) and math.isfinite(tlo)
                         and (rhi < tlo or thi < rlo)) else "no"
        ratio = t / r if r not in (0.0,) and math.isfinite(r) and math.isfinite(t) else float("nan")
        print(f"  {label:>24} {r:>15.6g} {t:>15.6g} {ratio:>12.4g} {disj:>21}")
    print()
    print("=== THE CERTIFICATE IS A MAX, NOT A MEAN. K-A reads the worst cell ===")
    print("  The two rows above are independent per-example means, so the kappa")
    print("  row is mean(tanh(D_i/4)) and NOT tanh(mean D). Those differ badly")
    print("  once tanh saturates. Birkhoff's hypothesis is a SUPREMUM, so the")
    print("  quantity that certifies anything is the MAX over cells.")
    print(f"  {'side':>12} {'median D':>11} {'max D':>11} {'kappa(max D)':>13} "
          f"{'1-kappa(max D)':>15} {'Neumann N':>11}")
    for name, st in (("random-init", rnd), ("trained", trn)):
        ds = sorted(v for v in st["_raw"]["delta_vertex"] if math.isfinite(v))
        n_inf = sum(1 for v in st["_raw"]["delta_vertex"] if not math.isfinite(v))
        if not ds:
            print(f"  {name:>12}  every cell non-positive: Delta = +inf. K-A FIRES.")
            continue
        dmed, dmax = ds[len(ds) // 2], ds[-1]
        kap = math.tanh(dmax / 4.0)
        omk = FH.one_minus_kappa(dmax)
        if 0.0 < kap < 1.0 and omk > 0.0:
            nterms = math.ceil(math.log(1e-6 * omk) / math.log(kap))
        else:
            nterms = -1
        print(f"  {name:>12} {dmed:>11.4f} {dmax:>11.4f} {kap:>13.6f} "
              f"{omk:>15.6e} {(nterms if nterms > 0 else 'INFEASIBLE'):>11}")
        if n_inf:
            print(f"  {'':>12} {n_inf} of {len(st['_raw']['delta_vertex'])} cells "
                  f"read Delta = +inf and are EXCLUDED above. K-A FIRES on those.")
    print("  Neumann N is the truncation contract 1.2 needs for error < 1e-6:")
    print("  N = ceil(log(1e-6 (1-kappa)) / log kappa). The contract's own sanity")
    print("  table reads kappa=0.9 -> N=153, which is the check on this column.")
    print()
    print("  Intervals are bootstrap over the example draws, B=2000, seed=0.")
    print("  'CI excludes overlap' means the two intervals are disjoint, which")
    print("  is the only form in which a move is claimed here (G6).")
    print()
    print("  alpha is the T1 quantity. exp(entropy) of 1.0 is one-hot; the")
    print("  ceiling is the pivot count. Foreman measured log_alpha min")
    print("  -182.7498 max -0.0 at random init and nobody had looked at trained.")
    print()
    print("  LIMIT. One seed, one small model, one arm (pivot_unsigned). This")
    print("  says whether these statistics move under training at this scale.")
    print("  It does not say what they equal at n_train=8192, and it is not a")
    print("  capability claim. The aggregator retention figures 2.9%/13.3%/58.9%")
    print("  are NOT re-run here and remain random-init upper bounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
