"""The sign floor: where `sgate` stops being signed, as a certificate not a reading.

    python -m scale.foreman_signfloor --steps 150

WHAT THIS MEASURES, AND WHY IT IS NOT ANOTHER MIN-ENTRY PRINTOUT. `CHECKLIST.md:333`
records one reading -- at the M3 harness geometry `_causal_sgate_operator(lam=0.10)`
has min entry exactly `0.000e+00`. A min entry is a property of ONE batch at ONE
seed. `PRIOR_ART.md` states the replacement goal this file executes: "measure the
sign structure of every arm operator at the geometry it will be measured at, and
exhibit a negative entry, before one gap task is built."

THE EXACT CRITERION. For

    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam),      rho > 0, lam > 0

on the strictly-causal support of row `i`, entry `(i, j)` is negative exactly when
`softmax(w)_ij < lam * softmax(-w)_ij`. Taking logs turns that into a comparison
with no division and no multiplication in it, which is what G8 asks of a sign
decision:

    L_ij := 2 w_ij + logsumexp_l(-w_il) - logsumexp_l(w_il)
    A_ij < 0   <=>   L_ij < log lam

`rho` does not appear: it is a positive prefactor and cannot move a sign. So the
whole sign question for an operator is the single scalar `min_ij L_ij` compared
against `log lam`, and the smallest `lam` that would put a negative entry in this
operator at this geometry is `exp(min_ij L_ij)` exactly -- a value, not a search.

THE RANGE CERTIFICATE. Let `R_i = max_j w_ij - min_j w_ij` over row `i`'s support
and let `n_i` be its size. Then

    logsumexp_l(-w_il) >= -min_j w_ij - R_i          (drop all but the largest term)
    logsumexp_l( w_il) <= log n_i + min_j w_ij + R_i (bound every term by the largest)

and the `log n_i` cancels against the same factor available on the other side, so

    L_ij >= -2 R_i        hence       max_i R_i <= (1/2) log(1/lam)  =>  A >= 0 entrywise.

`L_STAR = (1/2) log(1/lam)` is `1.151292546497023` at the shipped `lam = 0.10`.
This is a condition on the LOGIT GEOMETRY ALONE: below it, no seed, no batch and
no draw can produce a negative entry. `tests/foreman/test_sign_floor.py` shows on
drawn instances that the criterion is exact and that the certificate is a
sufficient condition and NOT an iff, which is what stops it being a tautology.

WHAT IT IS FOR. The S2 chain ends in a comparison between `pivot_signed` and
`pivot_unsigned`. That comparison is evidence about SIGN only if the first arm's
operator carries a negative entry at the geometry it is scored on. This file
answers that for every arm in `scale/m3_capability.ARMS`, at initialisation and
after the harness's own training loop, and prints the exact `lam` that restores
sign.

CPU only, threads pinned to 2 -- the same discipline as `scale/m3_capability.py`,
because CPU matmul reduction order varies with the thread count.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ceq import bench                                          # noqa: E402
from ceq.attention import ceq_operator                         # noqa: E402
from scale import m3_capability as m3                          # noqa: E402
from scale.negation_scope import make_batch, nrmse             # noqa: E402
from scale.pivot_probe import (batched_select_pivots,          # noqa: E402
                               batched_pivot_hop2)

torch.set_num_threads(2)

#: The geometry every published M3 number was taken at: `scale/m3_capability.py`
#: defaults plus the harness batch that produced softmax 0.877168. `n = 256` for
#: the operator read -- the statistics below are a min and a max over examples,
#: so a larger `n` can only make the reading more extreme, never less.
N, S, D, DMODEL, SEED = 256, 64, 24, 16, 0

__all__ = ["sign_margin", "harness_operator", "harness_batch", "affine_collapse",
           "trained_report", "value_path_flip_rate", "ARM_GEOMETRIES", "l_star"]


def l_star(lam: float) -> float:
    """The certified non-negativity threshold on the row logit range."""
    return 0.5 * math.log(1.0 / lam)


def _logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])


def sign_margin(q: torch.Tensor, k: torch.Tensor, *, rho: float, lam: float,
                window: int = 0) -> dict:
    """The sign question for `_causal_sgate_operator(q, k, rho, lam, window)`.

    Reported over every entry on the strictly-causal support:

    ``min_log_ratio``  `min_ij [2 w_ij + logsumexp(-w_i.) - logsumexp(w_i.)]`.
                       A negative entry exists iff this is below `log lam`.
    ``lam_needed``     `exp(min_log_ratio)`: the smallest `lam` at which this
                       geometry has a negative entry.
    ``max_range``      `max_i (max_j w_ij - min_j w_ij)` on the support.
    ``min_entry``      the operator's actual minimum, so the identity above can
                       be checked against the operator rather than assumed.

    Rows with empty support (row 0 always, and nothing else while `window` keeps
    at least one column) are EXCLUDED rather than filled with zeros: a row with
    nothing to attend to has no sign to have, and folding its zeros in as
    non-negative evidence is how a sparse operator reads sign-free by
    construction.
    """
    m = bench._causal_mask(q.shape[-2], q.device, window)       # [s,s] bool, CACHED
    w = _logits(q, k)
    neg = torch.finfo(w.dtype).min

    lse_p = torch.logsumexp(w.masked_fill(~m, neg), dim=-1)     # [..., s]
    lse_m = torch.logsumexp((-w).masked_fill(~m, neg), dim=-1)
    log_ratio = 2.0 * w + (lse_m - lse_p)[..., None]

    hi = w.masked_fill(~m, -math.inf).amax(-1)
    lo = w.masked_fill(~m, math.inf).amin(-1)
    live = m.any(-1)                                            # rows with support
    rng = (hi - lo)[..., live]

    a = bench._causal_sgate_operator(q, k, rho=rho, lam=lam, window=window)
    sup = m.expand_as(a)
    mlr = float(log_ratio[sup].min())
    return dict(
        min_log_ratio=mlr,
        lam_needed=math.exp(mlr),
        log_lam=math.log(lam),
        max_range=float(rng.max()),
        max_abs_logit=float(w[sup].abs().max()),
        min_entry=float(a[sup].min()),
        neg_frac=float((a[sup] < 0).to(torch.float64).mean()),
        n_support=int(sup.sum()),
        row_sum_max=float(a.sum(-1).abs().max()),
        l_star=l_star(lam),
    )


# --------------------------------------------------------------- arm geometries
def harness_batch(n: int = N, seed: int = SEED):
    x, y, _f, _p = make_batch(n, S, D, d_model=DMODEL, seed=seed)
    return x, y


def _arm_qk(kind: str, x: torch.Tensor, seed: int = SEED):
    """q, k exactly as `m3_capability.Arm.forward` computes them."""
    torch.manual_seed(seed)
    arm = m3.Arm(kind, S)
    with torch.no_grad():
        return arm, arm.wq(x), arm.wk(x)


def _nonneg_report(a: torch.Tensor, m: torch.Tensor, lam: float) -> dict:
    """The subset of `sign_margin` that is defined for a non-`sgate` operator.

    `min_log_ratio`, `lam_needed`, `max_range` and `max_abs_logit` are NaN here
    rather than 0: they are quantities of the two-softmax construction, and a
    zero would read as a measurement.
    """
    sup = m.expand_as(a)
    nan = float("nan")
    return dict(min_log_ratio=nan, lam_needed=nan, log_lam=math.log(lam),
                max_range=nan, max_abs_logit=nan,
                min_entry=float(a[sup].min()),
                neg_frac=float((a[sup] < 0).to(torch.float64).mean()),
                n_support=int(sup.sum()),
                row_sum_max=float(a.sum(-1).abs().max()), l_star=l_star(lam))


#: name -> (arm kind whose q/k projections are used, lam, window).
#:
#: The four shipped M3 arms at their shipped settings, plus two controls that
#: are signed for INDEPENDENT reasons -- one by a different normalizer, one by a
#: different lam -- so "no sign anywhere" cannot be a property of this file. The
#: controls share the harness batch and the `pivot_signed` projections, so the
#: only thing that differs between a control and the arm it controls is the one
#: named quantity.
ARM_GEOMETRIES = {
    "softmax":          ("softmax", m3.SGATE_LAM, 0),
    "pivot_unsigned":   ("pivot_unsigned", m3.SGATE_LAM, 0),
    "pivot_signed":     ("pivot_signed", m3.SGATE_LAM, 0),
    "windowed_signed":  ("windowed_signed", m3.SGATE_LAM, m3.W_WINDOW),
    "CONTROL_row_l1":   ("row_l1", m3.SGATE_LAM, 0),
    "CONTROL_lam1":     ("pivot_signed", 1.00, 0),
}


def harness_operator(arm: str, *, x: torch.Tensor | None = None,
                     seed: int = SEED) -> dict:
    """Sign report for one entry of `ARM_GEOMETRIES` at its own geometry."""
    kind, lam, window = ARM_GEOMETRIES[arm]
    x = harness_batch(seed=seed)[0] if x is None else x
    if kind == "row_l1":
        # `ceq.attention.ceq_operator`, the operator `ceq/arms.py`'s `signed`
        # arm carries. Homogeneous of degree ZERO in the logits, so its sign
        # structure cannot be moved by the input scale at all. It is here as the
        # control that makes a null on the sgate arms mean something.
        _arm, q, k = _arm_qk("pivot_signed", x, seed)
        a = ceq_operator(q, k, None, rho=m3.SGATE_RHO)
        return _nonneg_report(a, bench._causal_mask(S, q.device, window), lam)
    _arm, q, k = _arm_qk(kind, x, seed)
    if kind in ("softmax", "pivot_unsigned"):
        a = bench._softmax_operator(q, k, window=window)
        return _nonneg_report(a, bench._causal_mask(S, q.device, window), lam)
    return sign_margin(q, k, rho=m3.SGATE_RHO, lam=lam, window=window)


# ------------------------------------------------------------- affine collapse
def affine_collapse(x: torch.Tensor, seed: int = SEED) -> dict:
    """How far `sgate` is from an affine image of `softmax` at this geometry.

    To first order in `w` about zero, with `n_i` the row's support size,

        softmax(w)_ij = (1/n_i)(1 + w_ij - wbar_i) + O(w^2)
        A_ij          = rho*softmax(w)_ij - rho*(2 lam/(1+lam))/n_i + O(w^2)

    so at small logits the signed arm's operator is the unsigned arm's operator
    scaled by `rho`, minus a per-row constant. Reported as the maximum absolute
    residual of that identity and its size relative to `max |A|`. This is a
    DESCRIPTION of how close the two arms are at this geometry, not a claim that
    they are equal: the residual is printed, never asserted away.
    """
    _arm, q, k = _arm_qk("pivot_signed", x, seed)
    m = bench._causal_mask(S, q.device, 0)
    with torch.no_grad():
        a = bench._causal_sgate_operator(q, k, rho=m3.SGATE_RHO, lam=m3.SGATE_LAM)
        p = bench._softmax_operator(q, k)
        n_i = m.sum(-1).clamp_min(1).to(a.dtype)                 # [s]
        c = m3.SGATE_RHO * (2.0 * m3.SGATE_LAM / (1.0 + m3.SGATE_LAM))
        pred = m3.SGATE_RHO * p - (c / n_i)[:, None] * m.to(a.dtype)
        sup = m.expand_as(a)
        resid = float((a - pred)[sup].abs().max())
        return dict(affine_abs_resid=resid,
                    affine_rel_resid=resid / float(a[sup].abs().max()),
                    gain=m3.SGATE_RHO, offset_c=c)


# ----------------------------------------------------------------- trained read
def trained_report(kind: str, *, steps: int, n_train: int, seed: int = SEED) -> dict:
    """The same sign report after the harness's own training loop.

    Every sign measurement on record in this repo is at RANDOM INIT. `wq` and
    `wk` are the only parameters that can move the logit geometry, and Adam at
    `m3.LR` moves them, so "the arm is non-negative" and "the arm starts
    non-negative" are different claims. The loop here is the shipped one: same
    optimizer, same lr, same target standardisation as `m3_capability.run_arm`.
    The operator is read on a HELD-OUT batch (`seed + 1000`), so the geometry
    reported is the one an eval number would be taken at.
    """
    x, y = harness_batch(n_train, seed)
    torch.manual_seed(seed)
    model = m3.Arm(kind, S)
    opt = torch.optim.Adam(model.parameters(), lr=m3.LR)
    mu = float(y.mean())
    sd = float(y.std(unbiased=False)) or 1.0
    yt = (y - mu) / sd
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), yt).backward()
        opt.step()
    model.eval()
    xe, ye = harness_batch(N, seed + 1000)
    window = m3.W_WINDOW if kind == "windowed_signed" else 0
    with torch.no_grad():
        q, k = model.wq(xe), model.wk(xe)
        if kind in ("softmax", "pivot_unsigned"):
            a = bench._softmax_operator(q, k, window=window)
            rep = _nonneg_report(a, bench._causal_mask(S, q.device, window),
                                 m3.SGATE_LAM)
        else:
            rep = sign_margin(q, k, rho=m3.SGATE_RHO, lam=m3.SGATE_LAM,
                              window=window)
        rep["eval_nrmse"] = nrmse(model(xe) * sd + mu, ye)
    rep.update(kind=kind, steps=steps, n_train=n_train, seed=seed)
    return rep


# ------------------------------------------------- the value path, with the MLP
def value_path_flip_rate(kind: str, *, nonlinear: bool, n: int = N,
                         seed: int = SEED, floor: float = 1e-6,
                         gain: float = 1.0, arm: torch.nn.Module | None = None) -> dict:
    """Can a THIRD token flip the sign of token `j`'s influence on the output?

    This is `scale/pivot_probe.py`'s `wrt="v"` measurement with `m3_capability`'s
    OWN readout stack in place of a bare `W_o`. The operator is built from `x`
    and the gradient is taken with respect to a separate value tensor `v`, so
    only the value path is read -- `tests/cameron/test_negation_is_the_axis.py::
    test_the_probe_is_measuring_the_value_path_and_not_the_input_path` is the
    reason that distinction is kept.

        M = I + A + hop2        (built from x, so a third token moves it)
        out = readout(MLP(M v))[s-1]
        g   = d out / d v_j , summed over the feature axis

    A flip is `g` changing SIGN between two drawn values of the third token,
    with both magnitudes above `floor`. The decision is a sign comparison, never
    a product: `lo*hi < 0` underflows to `-0.0` at spreads this project has
    already measured (`scale/valuation.py`), which is the G8 defect.

    `nonlinear=False` replaces the GELU with an identity and changes NOTHING
    else. That is the control, and it is the whole point of the pair: with a
    linear value path and a non-negative `M`, `g = M[s-1,j] * const` and its
    sign cannot move, so the rate must read exactly 0. If it does not, this
    probe is broken and its other reading means nothing.

    `c` is drawn UNIFORMLY per example from the strict interior, excluding `j`
    and the query row. No position is chosen for its answer.

    `gain` scales the harness batch. `gain=1.0` is the shipped geometry; the
    logits scale as `gain**2`, so the sweep over `gain` is the same axis the
    sign-floor certificate is stated on, and it is what shows this probe can
    read a nonzero rate at all.
    """
    x, _y = harness_batch(n, seed)
    x = x * gain
    if arm is None:
        torch.manual_seed(seed)
        arm = m3.Arm(kind, S)
    if not nonlinear:
        arm.mlp[1] = torch.nn.Identity()

    j = S - 1 - D                                   # the flipper, per make_batch
    g = torch.Generator().manual_seed(seed)
    cand = torch.tensor([t for t in range(1, S - 1) if t != j])
    c = cand[torch.randint(0, len(cand), (n,), generator=g)]
    v0 = torch.randn(n, S, DMODEL, generator=g) * 0.1
    rows = torch.arange(n)

    def read(repl: torch.Tensor) -> torch.Tensor:
        xv = x.clone()
        xv[rows, c] = repl
        q, k = arm.wq(xv), arm.wk(xv)
        a = arm._operator(q, k)
        v = v0.clone().requires_grad_(True)
        z = v + a @ v
        if kind == "windowed_signed":
            z = z + a @ (a @ v)
        elif kind != "softmax":
            z = z + batched_pivot_hop2(a, batched_select_pivots(k, arm.k_pivots)) @ v
        out = arm.readout(arm.mlp(z)).squeeze(-1)[:, S - 1]
        grad, = torch.autograd.grad(out.sum(), v)
        return grad[rows, j].sum(-1)

    lo = read(torch.randn(n, DMODEL, generator=g) * 0.1 * gain)
    hi = read(torch.randn(n, DMODEL, generator=g) * 0.1 * gain)
    live = torch.minimum(lo.abs(), hi.abs()) > floor
    flipped = ((lo < 0) != (hi < 0)) & live
    used = int(live.sum())
    #: The WEAKER statistic, and the one with resolution: is the influence ever
    #: negative at all? A flip needs the third token to move the sign; this needs
    #: only that a negative sign is reachable. Reported beside the flip rate
    #: because a flip rate of exactly 0 has two different causes -- no sign
    #: available, or a sign that is available but not third-token-conditional --
    #: and they license different conclusions.
    neg_any = float((((lo < 0) | (hi < 0)) & live).to(torch.float64).mean())
    return dict(kind=kind, nonlinear=bool(nonlinear), n=n, used=used,
                k=int(flipped.sum()),
                rate=(float(flipped.sum()) / used) if used else float("nan"),
                neg_influence_frac=neg_any,
                min_influence=float(torch.minimum(lo, hi).min()), floor=floor,
                seed=seed, j=j, gain=gain)


def train_arm(kind: str, *, steps: int, n_train: int, seed: int = SEED):
    """The shipped training loop, returning the model. Same optimizer, lr and
    target standardisation as `m3_capability.run_arm`; factored out so the sign
    report and the value-path probe read the SAME trained weights instead of two
    separately-trained models that differ by an Adam trajectory."""
    x, y = harness_batch(n_train, seed)
    torch.manual_seed(seed)
    model = m3.Arm(kind, S)
    opt = torch.optim.Adam(model.parameters(), lr=m3.LR)
    mu = float(y.mean())
    sd = float(y.std(unbiased=False)) or 1.0
    yt = (y - mu) / sd
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), yt).backward()
        opt.step()
    model.eval()
    return model, mu, sd


_COLS = ("min_entry", "neg_frac", "min_log_ratio", "log_lam", "lam_needed",
         "max_range", "l_star", "max_abs_logit", "row_sum_max")


def _row(name: str, r: dict) -> str:
    return ("%-16s" % name) + "".join("%14.6g" % r[c] for c in _COLS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scale.foreman_signfloor")
    ap.add_argument("--steps", type=int, default=0,
                    help="training steps for the trained read; 0 skips it")
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--out", default="results/foreman_signfloor.jsonl")
    a = ap.parse_args(argv)

    x, _y = harness_batch()
    hdr = ("%-16s" % "arm") + "".join("%14s" % c for c in _COLS)
    print("SIGN FLOOR at the M3 harness geometry  n=%d s=%d d=%d d_model=%d seed=%d"
          % (N, S, D, DMODEL, SEED))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name in ARM_GEOMETRIES:
        r = harness_operator(name, x=x)
        rows.append(dict(r, arm=name, phase="init"))
        print(_row(name, r))

    ac = affine_collapse(x)
    print("\naffine collapse   A_sgate  vs  %.6g*A_softmax - %.6g/n_i :"
          "   max abs resid %.6e   relative %.6e"
          % (ac["gain"], ac["offset_c"], ac["affine_abs_resid"],
             ac["affine_rel_resid"]))

    if a.steps:
        print("\nTRAINED  steps=%d n_train=%d lr=%g  (operator read on a held-out batch)"
              % (a.steps, a.n_train, m3.LR))
        print(hdr)
        print("-" * len(hdr))
        for kind in m3.ARMS:
            t0 = time.time()
            r = trained_report(kind, steps=a.steps, n_train=a.n_train)
            rows.append(dict(r, arm=kind, phase="trained"))
            print(_row(kind, r) + "   eval_nrmse %.6f  [%.0fs]"
                  % (r["eval_nrmse"], time.time() - t0))

    p = pathlib.Path(__file__).resolve().parents[1] / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r, affine=ac)) + "\n")
    print("\nwrote %d rows to %s" % (len(rows), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
