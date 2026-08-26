"""CONSEQUENCE FIDELITY — the capability metric that replaces the Hankel frame.

    python -m scale.foreman_consequence --steps 150 --n-train 8192 --seeds 0 1 2 3 4
    python -m scale.foreman_consequence --s 512 --d 256 --n-train 256 --seeds 0 1 2

WHY THIS PLANE AND NOT ANOTHER ADJACENCY. Six rounds of instruments measured an
object standing next to the trained network — a cone, a rank, a contraction
constant, a linear map — and every one of them died of the adjacency. The Hankel
frame died last and hardest: `rank_+ > rank` is a theorem about nonnegative
weighted automata with LINEAR value paths, and the M3 readout is
`Linear -> GELU -> Linear`, so the nonlinearity supplies the sign the operator
withholds (F-J3: the trained NON-NEGATIVE arm signs `0.492188` of drawn
third-token interventions; the same trained weights with `GELU -> nn.Identity()`
sign `0.000000`). `LOOP_PROMPT.md` 1.1 retires that frame as a capability
predictor.

This file measures the trained network's BEHAVIOUR: apply a single-token `do()`
to the input, ask whether the model's predicted change moves the way the oracle's
actual change moves. There is no proxy object between the claim and the
measurement. Three properties are what make it a capability-table column rather
than a plea:

  1. IT IS COMPUTABLE FOR EVERY ARM, softmax included. Nothing in
     `consequence_fidelity` touches an operator; it calls `model(x)` twice. A
     metric only the signed arm can be scored on is an argument for the signed
     arm, not a measurement of it.
  2. THE INTERVAL IS EXACT. The sign-match count is a binomial, and this project
     has already shipped a rate of `0.004000` that was 1 event in 250. The
     interval here is CLOPPER-PEARSON, the inversion of the two binomial tails,
     via `scipy.stats.binomtest(k, n).proportion_ci(method="exact")` — not a
     normal approximation, not Wilson, not a bootstrap.
  3. THE ORACLE EFFECT IS CLOSED FORM. For `y = payload * sign`, a single-token
     `do()` moves the label by an expression in the drawn replacement alone
     (below). The closed form is nonetheless CHECKED against re-running the
     oracle on the perturbed tensor, entrywise, on every call — a closed form
     nobody compared against the thing it claims to compute is an assumption.

THE THREE INTERVENTIONS, and their closed forms. `y = x[p, CH_PAYLOAD] *
x[f, CH_FLIP]`, `f = s-1-d`, `p = s-2`.

    kind 0  FLIPPER   x'[f, CH_FLIP] <- sigma'  (Rademacher)
                      dy = x[p, CH_PAYLOAD] * (sigma' - x[f, CH_FLIP])
    kind 1  PAYLOAD   x'[p, CH_PAYLOAD] <- v'   (N(0,1))
                      dy = (v' - x[p, CH_PAYLOAD]) * x[f, CH_FLIP]
    kind 2  NULL      x'[t, CH_NOISE:] <- fresh noise, t drawn uniformly from
                      every position except f and p
                      dy = 0, EXACTLY — the oracle reads neither t nor those
                      channels

Kind 2 carries no sign, so it is excluded from the fraction and reported
separately: it is the control that says whether a nonzero `dy_hat` means
anything. A model that answers a null `do()` as loudly as a live one is not
reading consequence, and the two medians are printed side by side.

WHY BOTH LIVE KINDS AND NOT JUST THE FLIP. Negating the flipper sends `y -> -y`,
so `sign(dy) = -sign(y)` and fidelity on flipper draws alone collapses into the
model's plain sign accuracy on the label — a quantity the NRMSE column already
half-contains. The PAYLOAD intervention does not: `sign(dy) = sign(v' - v) *
sigma` couples the drawn change at one position to the sign held at another, and
no marginal statistic of `y` determines it. Both are reported, pooled and split,
so the pooled number can never hide which half carried it.

WHAT THIS BUYS OVER NRMSE, stated so it can be checked rather than believed.
NRMSE saturates: every arm in the W4 round died ABOVE 1.0, which orders nothing
because 1.0 is the constant predictor and everything worse than it is equally
worse-than-nothing. Consequence fidelity is bounded in [0, 1] with a known null
at 0.5 and an EXACT interval at every count, so it still separates arms in the
region where NRMSE has stopped separating anything. It is not independent of
NRMSE and this file does not pretend it is — both columns are printed on the
same row, at the same weights, from the same batch.

THE PAIRING, AND THE TEST THAT USES IT. `draw_do` is a function of `(x, f, p,
seed)` and of nothing about the model, so every arm at a given seed is scored on
BYTE-IDENTICAL interventions. Two arms' sign-match counts are therefore PAIRED,
and the exact test of their difference is McNemar's — a binomial test on the
discordant pairs — not a comparison of two independent proportions. Overlapping
Clopper-Pearson intervals are reported too, but overlap is not evidence of
equality and is never read as any; the McNemar p is the number that speaks to a
difference.

THE RESOLUTION, COMPUTED BEFORE THE DATA LAND (1.8 asks for the power analysis
on the pilot, not after). At `n_eval = 512` about one draw in three is a null and
about one flipper draw in two is a Rademacher no-op, so roughly `n_used = 280`
draws carry a sign. At that `n_used`:

    Clopper-Pearson 95% width      0.1202 at rate 0.50,  0.0738 at 0.90,
                                   0.0288 at 0.9893,     0.0131 at 1.00
    exact McNemar, smallest        n_disc =  20  ->  0.0357
    |CF_a - CF_b| that rejects     n_disc =  40  ->  0.0500
    at alpha = 0.05                n_disc =  80  ->  0.0714
                                   n_disc = 160  ->  0.0929
    floor                          an all-one-sided split rejects from
                                   n_disc >= 6, since 2 * 0.5**6 < 0.05

So this cell resolves an arm-vs-softmax difference of roughly FIVE POINTS of
consequence fidelity and no better. A reported "no reject" means the difference
is under that, NOT that it is zero, and it is never written as zero.

THE PRE-REGISTERED FALSIFIER (`LOOP_PROMPT.md` 1.7c), fixed here before the run:

  FIRES — the operator contributes nothing behavioural, and the signed program
    retires on its own terms — when, at every `d >= 256` cell and every seed,
    softmax's McNemar test against `pivot_signed` AND against `settled` fails to
    reject at `alpha = 0.05`, AND the cell is INFORMATIVE.
  DOES NOT FIRE when any arm rejects against softmax at any seed.
  NOT EVALUABLE — and the falsifier is neither fired nor cleared — when the cell
    is UNINFORMATIVE: no arm's Clopper-Pearson interval excludes 0.5. Then every
    arm is at chance, softmax "matches" the others only in the sense that none
    of them measured anything, and reading a retirement out of that is the
    zero-by-construction defect this project has already shipped twice. An
    uninformative cell is a harness limit (K-5E's shape), reported as one.

PRIOR ART, FETCHED BEFORE THIS FILE WAS BUILT, AND THE DELTA STATED. Absence is
reported as NOT FOUND, never as "unoccupied".

  (a) THE INTERVAL IS NOT NEW AND IS NOT CLAIMED AS NEW. Clopper, C. J. and
      Pearson, E. S. (1934), "The use of confidence or fiducial limits
      illustrated in the case of the binomial", Biometrika 26(4):404-413. The
      interval is the inversion of the two binomial tails,
      `B(alpha/2; x, n-x+1) < p < B(1-alpha/2; x+1, n-x)`, with the degenerate
      ends `(0, 1-(alpha/2)^(1/n))` at `x = 0` and `((alpha/2)^(1/n), 1)` at
      `x = n`. `scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.95,
      method="exact")` is that interval and is what this file calls; `method`
      also accepts `"wilson"` and `"wilsoncc"`, and `"exact"` is scipy's own
      default. NOT VERIFIED: the version-pinned scipy 1.17.0 doc page was not
      fetched, only the current and a 1.15.2 mirror, both showing this
      signature. The identity is therefore checked numerically against
      `scipy.stats.binom` in `tests/foreman/test_consequence_fidelity.py`
      instead of taken on the name.

  (b) THE SLOPE IS PRIOR ART UNDER A NAME. Regressing a predicted causal effect
      on the true one and reading the coefficient is the CALIBRATION SLOPE of
      the heterogeneous-treatment-effect literature: Xu, Y. and Yadlowsky, S.,
      "Calibration Error for Heterogeneous Treatment Effects", arXiv:2203.13364,
      which defines "an analogue of the calibration slope, called the Best
      Linear Predictor of the CATE on the ML proxy predictor (the coefficient of
      the proxy is the calibration slope)", attributing the construction to
      Chernozhukov et al. 2018b. Perfect prediction reads intercept 0 and slope
      1; slopes under 1 are the overfitting direction. THE DELTA: nothing here
      is a new estimator. What differs is that this project's "true effect" is
      not estimated from observational data at all — the oracle is executable
      and the effect is closed form, so the usual identification problem that
      the CATE-calibration literature exists to solve is absent by construction.
      That makes the slope EASIER here, not novel, and it is reported as a
      description of the arm rather than as a contribution.

  (c) THE SIGN-AGREEMENT FRACTION: NOT FOUND as a named metric matching this
      description — a trained network scored by the fraction of interventions on
      which the sign of its predicted change agrees with the sign of an oracle
      causal effect. Searched: "interventional sign agreement", "sign agreement
      causal effect neural network", "directional accuracy"/"sign match" in
      causal representation learning and mechanistic interpretability,
      "counterfactual sign match", "do-operator sign accuracy", "interventional
      faithfulness", "causal consistency metric", "Intervention Consistency
      Score". The nearest occupied ground, none of it the same object:
        - INTERCHANGE INTERVENTION ACCURACY (Geiger et al., arXiv:2112.00826;
          arXiv:2303.02536) scores whether patched activations yield the
          expected value of a causal VARIABLE under a causal abstraction — an
          accuracy on the intervened variable, not on the SIGN of an effect;
        - STRUCTURAL INTERVENTION DISTANCE counts variable pairs whose
          interventional effects a predicted GRAPH gets wrong — graph-level, in
          causal discovery, not a trained predictor's response;
        - arXiv:2605.27153 was reported by a search layer to use a "sign match"
          fraction against reported HUMAN effects in social-science
          replication; the PDF body could not be extracted, so this is
          UNVERIFIED and is not cited as prior art.
      ONE NEGATIVE RESULT WORTH KEEPING: a search layer asserted that "Causal
      Evaluation of Language Models" (CaLM, arXiv:2405.00622) defines a
      "Direction Accuracy" metric as "the fraction of query tokens where the
      sign of the predicted effect equals the sign of the true effect". A direct
      fetch of that paper's Section 6 (Metrics) returned Accuracy, Robustness,
      Model Volatility, Understandability, Open-Limited Gap, Solvability and
      Prompt Volatility, and NO such metric. The summary contradicts the source
      and is treated as fabricated. It is recorded here because the sentence is
      exactly the sentence this file would want a citation for, which is the
      circumstance under which a fabricated citation gets adopted.

CPU only, threads pinned HERE, not by the launcher — the same discipline as
`scale/m3_capability.py`, because CPU matmul reduction order varies with the
thread count.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import typing

import torch
from scipy.stats import binomtest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ceq import bench                                              # noqa: E402
from scale import foreman_signfloor as SF                          # noqa: E402
from scale import m3_capability as m3                              # noqa: E402
from scale.negation_scope import (CH_FLIP, CH_NOISE, CH_PAYLOAD,   # noqa: E402
                                  make_batch, nrmse, oracle)

torch.set_num_threads(2)

__all__ = ["Do", "draw_do", "clopper_pearson", "mcnemar_exact",
           "consequence_fidelity", "arm_row", "ARM_FACTORY"]

ALPHA = 0.05
N_BOOT = 2000


class Do(typing.NamedTuple):
    """One drawn single-token `do()` per example.

    ``x_do``  the perturbed input, one token touched per example.
    ``dy``    the oracle's label change, in CLOSED FORM.
    ``kind``  0 flipper, 1 payload, 2 null.
    ``site``  the position touched, so the draw can be audited.
    """

    x_do: torch.Tensor
    dy: torch.Tensor
    kind: torch.Tensor
    site: torch.Tensor


def draw_do(x: torch.Tensor, f: int, p: int, *, seed: int) -> Do:
    """Draw one single-token `do()` per example, with its closed-form effect.

    Every example gets exactly one intervention and the kind is DRAWN, never
    assigned: nothing here picks a position or a kind for its answer. All three
    replacement tensors are drawn for every example and then selected by kind,
    so the generator stream does not depend on which kinds came up.
    """
    n, s, dm = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.arange(n)

    kind = torch.randint(0, 3, (n,), generator=g)
    sigma_new = torch.randint(0, 2, (n,), generator=g).float() * 2.0 - 1.0
    v_new = torch.randn(n, generator=g)
    cand = torch.tensor([t for t in range(s) if t not in (f, p)])
    site = cand[torch.randint(0, len(cand), (n,), generator=g)]
    noise = torch.randn(n, dm - CH_NOISE, generator=g) * 0.1

    x_do = x.clone()
    sigma_old = x[:, f, CH_FLIP]
    v_old = x[:, p, CH_PAYLOAD]

    m0, m1, m2 = kind == 0, kind == 1, kind == 2
    x_do[m0, f, CH_FLIP] = sigma_new[m0]
    x_do[m1, p, CH_PAYLOAD] = v_new[m1]
    idx2 = rows[m2]
    x_do[idx2[:, None], site[m2][:, None], torch.arange(CH_NOISE, dm)[None, :]] = noise[m2]

    dy = torch.where(m0, v_old * (sigma_new - sigma_old),
                     torch.where(m1, (v_new - v_old) * sigma_old,
                                 torch.zeros_like(v_old)))
    site = torch.where(m0, torch.full_like(site, f),
                       torch.where(m1, torch.full_like(site, p), site))
    return Do(x_do=x_do, dy=dy, kind=kind, site=site)


def clopper_pearson(k: int, n: int, *, alpha: float = ALPHA) -> tuple[float, float]:
    """The EXACT binomial interval: the inversion of the two binomial tails.

    `scipy.stats.binomtest(k, n).proportion_ci(method="exact")` is Clopper-Pearson
    (1934). Delegated rather than reimplemented — a hand-rolled Beta-quantile
    copy is a second implementation of a rule that already ships, which is the
    defect that made `m3_capability` and `negation_scope` disagree in round 2.
    The endpoints satisfy `P(X >= k | lo) = alpha/2` and `P(X <= k | hi) =
    alpha/2`, and `tests/foreman/test_consequence_fidelity.py` checks that
    identity against `scipy.stats.binom` rather than trusting the name.
    """
    if n <= 0:
        return float("nan"), float("nan")
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=1.0 - alpha,
                                                method="exact")
    return float(ci.low), float(ci.high)


def mcnemar_exact(a_match: torch.Tensor, b_match: torch.Tensor) -> dict:
    """Exact McNemar on two PAIRED sign-match indicator vectors.

    The two arms are scored on byte-identical interventions, so the concordant
    pairs carry no information about a difference and the exact test is a
    two-sided binomial test on the discordant counts at `q = 1/2`. Reported with
    both discordant counts so a p-value can never be read without its `n`.
    """
    b = int((a_match & ~b_match).sum())
    c = int((~a_match & b_match).sum())
    if b + c == 0:
        return dict(b=0, c=0, n_disc=0, p=1.0)
    p = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return dict(b=b, c=c, n_disc=b + c, p=p)


def _ols_slope(dy: torch.Tensor, dyh: torch.Tensor) -> tuple[float, float]:
    """(slope, intercept) of `dy_hat` regressed on `dy`, in float64."""
    a = dy.to(torch.float64)
    b = dyh.to(torch.float64)
    am, bm = a.mean(), b.mean()
    var = ((a - am) ** 2).sum()
    if float(var) == 0.0:
        return float("nan"), float("nan")
    sl = float(((a - am) * (b - bm)).sum() / var)
    return sl, float(bm - sl * am)


def consequence_fidelity(model_fn, x: torch.Tensor, f: int, p: int, *,
                         seed: int, sigma: float = 1.0, alpha: float = ALPHA,
                         n_boot: int = N_BOOT) -> dict:
    """The metric. `model_fn` is any callable `[n,s,d_model] -> [n]`.

    `sigma` is the training-target standard deviation: `m3_capability.run_arm`
    fits standardised targets, so the model's output is in units of `sigma` and
    `dy_hat = (model(x') - model(x)) * sigma` puts the predicted change back in
    the oracle's units. `sigma > 0` cannot move a sign, so the sign-match
    fraction is invariant to it; the SLOPE is not, which is the whole reason it
    is a parameter and not a constant.

    Zero-effect draws (kind 2, and flipper draws where the Rademacher came back
    equal) are excluded from the fraction — `sign(0)` is not a sign — and their
    count is reported. A model `dy_hat` of exactly 0.0 on a live draw counts as
    a MISMATCH, which is the conservative direction, and is counted separately.
    """
    with torch.no_grad():
        do = draw_do(x, f, p, seed=seed)
        y0 = oracle(x, f, p)
        resid = float(((oracle(do.x_do, f, p) - y0) - do.dy).abs().max())
        base = model_fn(x).reshape(-1)
        pert = model_fn(do.x_do).reshape(-1)
    dyh = (pert - base) * float(sigma)

    live = do.dy != 0
    n_used = int(live.sum())
    sdy, sdh = torch.sign(do.dy), torch.sign(dyh)
    match = (sdy == sdh) & live
    k = int(match.sum())
    lo, hi = clopper_pearson(k, n_used, alpha=alpha)

    dy_l, dyh_l = do.dy[live], dyh[live]
    slope, intercept = _ols_slope(dy_l, dyh_l)
    g = torch.Generator().manual_seed(seed + 991)
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, n_used, (n_used,), generator=g)
        s_b, _ = _ols_slope(dy_l[idx], dyh_l[idx])
        if s_b == s_b:
            boots.append(s_b)
    boots.sort()
    if boots:
        s_lo = boots[int(alpha / 2 * len(boots))]
        s_hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    else:
        s_lo = s_hi = float("nan")

    null = do.kind == 2
    per_kind = {}
    for kk, nm in ((0, "flipper"), (1, "payload")):
        sel = live & (do.kind == kk)
        nk, mk = int(sel.sum()), int((match & sel).sum())
        klo, khi = clopper_pearson(mk, nk, alpha=alpha)
        per_kind[nm] = dict(n=nk, k=mk, rate=(mk / nk) if nk else float("nan"),
                            ci_lo=klo, ci_hi=khi)

    return dict(
        sign_match=(k / n_used) if n_used else float("nan"),
        k=k, n_used=n_used, ci_lo=lo, ci_hi=hi, alpha=alpha,
        ci_method="Clopper-Pearson exact (scipy.stats.binomtest.proportion_ci)",
        slope=slope, intercept=intercept, slope_lo=s_lo, slope_hi=s_hi,
        slope_ci_method=f"paired percentile bootstrap over draws, B={len(boots)}",
        n_zero_effect=int((~live).sum()), n_null=int(null.sum()),
        n_dyhat_exact_zero=int(((dyh == 0) & live).sum()),
        oracle_closed_form_max_resid=resid,
        null_abs_median=float(dyh[null].abs().median()) if int(null.sum()) else float("nan"),
        live_abs_median=float(dyh[live].abs().median()) if n_used else float("nan"),
        per_kind=per_kind, sigma=float(sigma), seed=seed,
        _match=match, _live=live,
    )


# --------------------------------------------------------------- the arm table
def _quint(cell: str):
    from scale.m3_quintuple import BETA, QuintArm

    def make(_kind: str, s: int):
        return QuintArm("softmax", s, cell=cell, k_piv=8, beta=BETA,
                        t_max=21, n_neumann=21)
    return make


#: name -> factory. The four `m3.ARMS` entries build themselves; `settled` and
#: `twin` are the quintuple's cells, trained by the SAME loop
#: (`foreman_signfloor.train_arm`) rather than a second copy of it, so a
#: difference between rows is the arm and not the optimiser trajectory.
ARM_FACTORY = {k: None for k in m3.ARMS}
ARM_FACTORY["settled"] = "settled"
ARM_FACTORY["twin"] = "twin"


def _sign_row(model, xe: torch.Tensor, kind: str) -> dict:
    """The sign floor at the SAME trained weights, so the sign column and the
    capability column are never two different models wearing one name."""
    window = m3.W_WINDOW if kind == "windowed_signed" else 0
    with torch.no_grad():
        q, k = model.wq(xe), model.wk(xe)
        if kind in ("softmax", "pivot_unsigned", "settled", "twin"):
            a = bench._softmax_operator(q, k, window=window)
            return SF._nonneg_report(a, bench._causal_mask(xe.shape[1], q.device,
                                                           window), m3.SGATE_LAM)
        return SF.sign_margin(q, k, rho=m3.SGATE_RHO, lam=m3.SGATE_LAM,
                              window=window)


def arm_row(kind: str, *, steps: int, n_train: int, seed: int, s: int, d: int,
            n_eval: int) -> dict:
    """Train one arm through the shipped loop, then read every column at those
    weights: NRMSE, the sign floor, and consequence fidelity."""
    fac = ARM_FACTORY[kind]
    make = _quint(fac) if isinstance(fac, str) else None
    build_kind = "softmax" if isinstance(fac, str) else kind
    t0 = time.time()
    model, mu, sd = SF.train_arm(build_kind, steps=steps, n_train=n_train,
                                 seed=seed, s=s, d=d, make=make)
    xe, ye, f, p = make_batch(n_eval, s, d, d_model=SF.DMODEL, seed=seed + 12345)

    def fwd(xx):
        return model(xx)

    with torch.no_grad():
        ev = nrmse(model(xe) * sd + mu, ye)
    cf = consequence_fidelity(fwd, xe, f, p, seed=seed + 777, sigma=sd)
    row = dict(cf)
    match, live = row.pop("_match"), row.pop("_live")
    row.update(_sign_row(model, xe, kind))
    row.update(arm=kind, steps=steps, n_train=n_train, n_eval=n_eval, s=s, d=d,
               seed=seed, eval_nrmse=ev, f=f, p=p,
               seconds=round(time.time() - t0, 3))
    return row, match, live


_COLS = ("eval_nrmse", "sign_match", "ci_lo", "ci_hi", "k", "n_used",
         "slope", "slope_lo", "slope_hi", "min_entry", "neg_frac")


def _fmt(r: dict) -> str:
    out = "%-16s" % r["arm"]
    for c in _COLS:
        v = r[c]
        out += (" %12d" % v) if isinstance(v, int) else (" %12.6g" % v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scale.foreman_consequence")
    ap.add_argument("--s", type=int, default=SF.S)
    ap.add_argument("--d", type=int, default=SF.D)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--arms", nargs="+", default=list(m3.ARMS))
    ap.add_argument("--out", default="results/foreman_consequence.jsonl")
    a = ap.parse_args(argv)

    print("CONSEQUENCE FIDELITY  s=%d d=%d steps=%d n_train=%d n_eval=%d "
          "seeds=%s threads=%d" % (a.s, a.d, a.steps, a.n_train, a.n_eval,
                                   a.seeds, torch.get_num_threads()))
    print("interval: Clopper-Pearson exact, alpha=%.2f. paired arm-vs-softmax "
          "test: exact McNemar." % ALPHA)
    print("PRE-REGISTERED (module docstring): the d>=256 falsifier fires only if "
          "the cell is INFORMATIVE -- some arm's CP interval excludes 0.5.")
    hdr = ("%-16s" % "arm") + "".join(" %12s" % c for c in _COLS)

    rows, matches = [], {}
    for seed in a.seeds:
        print("\n--- seed %d ---" % seed)
        print(hdr)
        print("-" * len(hdr))
        for kind in a.arms:
            r, match, live = arm_row(kind, steps=a.steps, n_train=a.n_train,
                                     seed=seed, s=a.s, d=a.d, n_eval=a.n_eval)
            rows.append(r)
            matches[(seed, kind)] = match
            print(_fmt(r) + "   [%.0fs]" % r["seconds"])
            print("%-16s   flipper %s   payload %s   null|dyhat| %.6g vs live "
                  "%.6g   zero-effect %d   closed-form resid %.3e"
                  % ("", "%.6f (%d/%d)" % (r["per_kind"]["flipper"]["rate"],
                                           r["per_kind"]["flipper"]["k"],
                                           r["per_kind"]["flipper"]["n"]),
                     "%.6f (%d/%d)" % (r["per_kind"]["payload"]["rate"],
                                       r["per_kind"]["payload"]["k"],
                                       r["per_kind"]["payload"]["n"]),
                     r["null_abs_median"], r["live_abs_median"],
                     r["n_zero_effect"], r["oracle_closed_form_max_resid"]))

        if "softmax" in a.arms:
            print("\n  PAIRED exact McNemar vs softmax (same interventions, "
                  "byte-identical draws):")
            for kind in a.arms:
                if kind == "softmax":
                    continue
                mn = mcnemar_exact(matches[(seed, "softmax")], matches[(seed, kind)])
                rows.append(dict(arm=kind, vs="softmax", seed=seed, s=a.s, d=a.d,
                                 test="mcnemar_exact", **mn))
                print("    softmax vs %-16s b=%d c=%d n_disc=%d  p=%.6g  %s"
                      % (kind, mn["b"], mn["c"], mn["n_disc"], mn["p"],
                         "REJECT" if mn["p"] < ALPHA else "no reject"))

    cf_rows = [r for r in rows if "sign_match" in r]
    above = [r for r in cf_rows if r["ci_lo"] > 0.5]
    below = [r for r in cf_rows if r["ci_hi"] < 0.5]
    #: INFORMATIVE means some arm is ABOVE chance, not merely away from it. An
    #: arm whose interval sits entirely BELOW 0.5 is reliably ANTI-fidelitous,
    #: which is a finding but is not evidence that the cell can show capability.
    #: Reading a retirement out of a cell where nothing rose above chance is the
    #: zero-by-construction defect this project has already shipped twice, so
    #: the two counts are printed separately and never summed.
    print("\nCELL INFORMATIVE: %s   above chance %d / %d arm-seed rows "
          "(CP interval entirely > 0.5);  below chance %d"
          % (bool(above), len(above), len(cf_rows), len(below)))

    out = pathlib.Path(__file__).resolve().parents[1] / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("wrote %d rows to %s" % (len(rows), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
