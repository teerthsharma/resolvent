"""S2 — the selection ablation, run standalone so it cannot collide with the loop.

WHY A SEPARATE MODULE. `scale/pivot_probe.py` and `scale/m2_units.py` are being
run by the bucket loop against `results/m2.jsonl` right now. Editing either to
add an arm would change the code under a running measurement (G2/G3 territory).
So this file adds NOTHING to them: it imports `pivot_probe.run_arm` and
`pivot_probe.build_arm` and drives THE SAME code path, swapping exactly one
symbol — the pivot selector — for the K4 control. A reimplementation that
"agrees" with the original is the ParaFormer class of evidence and is worth
nothing; the reproduction bind in DONE.md (five fields bit-identical to the
journalled `pivot_signed__in_P/s8`) is what makes numbers from here comparable
to numbers from there.

THE ARMS, and why these four.

    pivot_signed     signed tgate,  hop2 = A[:,P]A[P,:],  P by key-norm  (journalled)
    dense_signed     signed tgate,  hop2 = A A,           c drawn at P   (journalled)
    pivot_unsigned   SOFTMAX,       hop2 = A[:,P]A[P,:],  P by key-norm  (K1/K2)
    dense_unsigned   SOFTMAX,       hop2 = A A,           c drawn at P   (K2/K3c)
    randpivot_signed signed tgate,  hop2 = A[:,P]A[P,:],  P RANDOM       (K4)

`randpivot_signed` exists because the two axes CHECKLIST S2 names — pivots
vs dense, signed vs unsigned — do not separate the two things Star-Transformer
1902.09113 actually owns. Its relay is a virtual hub: fixed path count, global
reach, content-BLIND. So "pivots" as an axis conflates fixed-path-count (2019)
with content-selection (not 2019), and only a content-blind signed control tells
them apart. It is the third measurement, and it is pre-registered in DONE.md
under K4 with its number.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale import pivot_probe
from scale.pivot_probe import (clopper_pearson, loglog_slope, run_arm,
                               select_pivots)

_RG = torch.Generator().manual_seed(0)


def reseed(seed: int) -> None:
    """Make the content-blind selector reproducible. Used by the bind test."""
    _RG.manual_seed(seed)


def random_pivots(key: torch.Tensor, k: int, *, exclude=()) -> torch.Tensor:
    """K4's selector: |P| = k, drawn uniformly, NEVER reading `key`'s values.

    Signature-compatible with `pivot_probe.select_pivots` so it can be swapped in
    without touching `run_arm`. `key` enters only through `key.shape[0]`, which
    is what makes the control content-blind — the property the bind test checks
    by calling it twice from the same seed with different keys.
    """
    s = int(key.shape[0])
    pool = torch.tensor([t for t in range(s) if t not in set(int(e) for e in exclude)])
    if pool.numel() <= k:
        return pool
    return pool[torch.randperm(pool.numel(), generator=_RG)[:k]]


@contextlib.contextmanager
def selector(fn):
    """Swap the pivot selector for the duration of a measurement."""
    old = pivot_probe.select_pivots
    pivot_probe.select_pivots = fn
    try:
        yield
    finally:
        pivot_probe.select_pivots = old


def min_influence_entry(kind: str, s: int, n_draws: int = 8, k: int = 8,
                        d: int = 16, seed: int = 0) -> float:
    """Smallest entry of the influence matrix M = I + A + hop2 over `n_draws`.

    The probe's read is `grad[j].sum() = M[i,j] * sum(Wo)` with `Wo` fixed across
    a draw's two contexts, so a sign flip IS a sign flip of `M[i,j]`. If M has no
    negative entry, no context can flip anything and the rate is a FLOOR at 0,
    not a flat curve. Returns the min over `A + hop2` (both already strictly
    lower triangular, so masked cells read 0 and cannot fake a negative).
    """
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    worst = float("inf")
    for _ in range(n_draws):
        wq, wk = rnd(d, d), rnd(d, d)
        x0 = rnd(s, d)
        gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
        kk = x0 @ wk
        pivots = pivot_probe.select_pivots(kk, k)
        # `pivot_probe.build_arm` by ATTRIBUTE, never the name bound at import:
        # a local `from ... import build_arm` would keep pointing at the
        # unpatched function and this scan would silently report the SIGNED
        # arm's number under the absmag arm's name -- the G3 (ParaFormer) bug
        # class exactly. The bind test caught it reading -0.5365633964538574.
        a, hop2 = pivot_probe.build_arm(kind, x0 @ wq, kk, gvec, bet, pivots, gen=g)
        worst = min(worst, float((a + hop2).min()))
    return worst


# ---------------------------------------------------------------------------
# THE CORRECTED UNSIGNED COMPARATOR.
#
# The pre-registered unsigned arm is `bench._softmax_operator`, whose scores are
# UNNORMALIZED `q.k/sqrt(d)`, while `tgate` scores `qhat.khat/tau` with both
# vectors L2-normalized. MEASURED on this box, 24 draws, row i=s-1:
#
#     s      softmax logit std   max row weight   participation ratio 1/sum(a^2)
#     8            14.412           0.939805                1.145
#    32            15.609           0.899966                1.290
#   128            15.769           0.887753                1.269
#   512            15.775           0.835463                1.505
#
# A participation ratio of 1.15-1.51 means that arm attends to ~one token: it is
# a hard argmax, not a mixture. Its share metric therefore measures "did the
# argmax happen to land on a pivot", which decays like k/s for reasons that have
# nothing to do with signedness. The pre-registered K3a comparator is CONFOUNDED,
# and the number below is reported as pre-registered anyway, with the confound
# named -- not replaced quietly.
#
# `absmag` is the comparator without the confound: |tgate|. The SAME matrix,
# sign stripped. Same scores, same tau, same per-row gate, identical magnitudes
# entrywise, no denominator. The only thing that changes is whether the k-term
# (or s-term) background can CANCEL -- which is precisely the axis S2 asks about.
# The bind in `tests/cameron/test_s2_ablation.py` requires `term` to come out
# BIT-IDENTICAL to the signed arm, which is what proves nothing else moved.
# ---------------------------------------------------------------------------
_orig_build = pivot_probe.build_arm


def _absmag_build(kind, qq, kk, gvec, bet, pivots, **kw):
    a, _ = _orig_build(kind, qq, kk, gvec, bet, pivots, **kw)
    a = a.abs()
    hop2 = (pivot_probe.pivot_hop2(a, pivots) if kind.startswith("pivot")
            else a @ a)
    return a, hop2


@contextlib.contextmanager
def absmag():
    """Run the enclosed measurement on |A| instead of A."""
    pivot_probe.build_arm = _absmag_build
    try:
        yield
    finally:
        pivot_probe.build_arm = _orig_build


#: (label, arm kind, placement, selector) -- placement `in_P` everywhere so the
#: intervened token c sits at a pivot in EVERY arm. Matching c's position across
#: arms is what makes the dense/pivot contrast about ROUTING and not about which
#: token was poked; `dense_signed__at_pivots` in `m2_units.py` exists for the
#: same reason.
SPEC = {
    "pivot_signed":     ("pivot_signed",   "in_P", None),
    "pivot_absmag":     ("pivot_signed",   "in_P", None),
    "dense_absmag":     ("dense_signed",   "in_P", None),
    "pivot_unsigned":   ("pivot_unsigned", "in_P", None),
    "dense_unsigned":   ("dense_unsigned", "in_P", None),
    "dense_signed":     ("dense_signed",   "in_P", None),
    "randpivot_signed": ("pivot_signed",   "in_P", random_pivots),
}


def measure(label: str, sizes, n_draws: int, k: int = 8, seed: int = 0):
    kind, place, sel = SPEC[label]
    rows = []
    with (absmag() if label.endswith("absmag") else contextlib.nullcontext()),          (selector(sel) if sel is not None else contextlib.nullcontext()):
        if sel is not None:
            reseed(seed)
        for s in sizes:
            r = run_arm(kind, s, n_draws=n_draws, k=k, placement=place,
                        seed=seed, protocol="SCALING")
            lo, hi = clopper_pearson(r["k"], r["n"])
            r.update(lo=lo, hi=hi, s=s,
                     ratio=(r["term"] / r["sigma"]) if r["sigma"] > 0 else float("nan"))
            rows.append(r)
    return rows


def fmt(label: str, rows) -> str:
    sizes = [r["s"] for r in rows]
    sl_rate, n1 = loglog_slope(sizes, [r["rate"] for r in rows])
    sl_share, n2 = loglog_slope(sizes, [r["ratio"] for r in rows])
    out = [f"ARM {label}   n_draws={rows[0]['n']}  k=8  PROTOCOL: SCALING  c in P"]
    out.append(f"{'s':>6}{'rate':>12}{'CP95':>22}{'term':>12}{'sigma':>12}{'R=term/sigma':>15}")
    for r in rows:
        out.append(f"{r['s']:>6}{r['rate']:>12.6f}"
                   f"{'[%.5f,%.5f]' % (r['lo'], r['hi']):>22}"
                   f"{r['term']:>12.6f}{r['sigma']:>12.6f}{r['ratio']:>15.6f}")
    out.append(f"  slope(rate)  = {sl_rate:+.3f}  ({n1}/{len(sizes)} nonzero)")
    out.append(f"  slope(R)     = {sl_share:+.3f}  ({n2}/{len(sizes)} nonzero)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(SPEC))
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128, 512])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/s2_ablation.txt")
    a = ap.parse_args()
    rows = measure(a.arm, a.sizes, a.n, seed=a.seed)
    text = fmt(a.arm, rows)
    print(text)
    p = pathlib.Path(__file__).resolve().parents[1] / a.out
    with p.open("a", encoding="utf-8") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# THE NON-DEGENERATE UNSIGNED ARM: Star-Transformer's own forward.
#
# `run_arm` builds `h = v + Av + hop2 v` from two MATRICES, so a nonlinearity
# between the hops cannot go through it. This is the separate forward, and the
# only thing that makes its numbers admissible is the K5-bind in
# `tests/cameron/test_s2_ablation.py`: with `nonlin=None` it must reproduce
# `run_arm` BIT-IDENTICALLY in all five fields on four different arms, because
#
#     A[:,P] (A[P,:] v)  ==  (A[:,P] A[P,:]) v        and      A (A v) == (A A) v
#
# algebraically. Five instruments in this project were internally consistent and
# externally wrong; a second forward that merely "agrees" is the sixth.
#
# WHY THIS ARM AND NOT ANOTHER. `tests/cameron/test_negation_is_the_axis.py::
# test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back` already
# measured that ONE GELU between two softmax layers restores the sign flip. The
# semiring theorem constrains non-negative operators with LINEAR value paths, and
# a real transformer has an MLP. So the honest unsigned comparator is softmax
# WITH the nonlinearity, and gather-nonlinearity-scatter through k pivots is not
# an approximation of Star-Transformer 1902.09113 -- it is its update, verbatim:
# "a virtual hub to gather and scatter information from and to all the satellite
# nodes".
# ---------------------------------------------------------------------------
_NONLIN = {None: (lambda t: t), "gelu": torch.nn.functional.gelu}


def run_hub_arm(kind: str, s: int, *, n_draws: int, k: int, d: int = 16,
                placement: str = "in_P", protocol: str = "SCALING",
                seed: int = 0, floor: float = 1e-6, tau: float = 1.0,
                rho: float = 1.5, lam: float = 0.10, device=None,
                nonlin=None):
    """`run_arm`'s measurement with the hop-2 sum taken through a hub function.

    The draw stream is replicated call-for-call from `pivot_probe.run_arm` --
    including `bet`, which softmax never uses but which must still be drawn or
    every subsequent number in the stream shifts. That is what the identity bind
    checks.
    """
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    f = _NONLIN[nonlin]

    if protocol == "SCALING":
        i, j = s - 1, max(1, s // 4)
    else:
        i, j = min(7, s - 1), 1

    flips = 0
    terms, bgs, used = [], [], 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        gvec = torch.sigmoid(rnd(s))
        bet = torch.sigmoid(rnd(s))

        kk0 = x0 @ wk
        pivots = pivot_probe.select_pivots(kk0, k, exclude=(i, j))
        pivot_set = set(int(p) for p in pivots)
        cands_in = [p for p in pivot_set if p not in (i, j)]
        cands_out = [t for t in range(1, s - 1)
                     if t not in pivot_set and t not in (i, j)]
        if placement == "in_P":
            pool = cands_in
        elif placement == "not_in_P":
            pool = cands_out
        else:
            pool = [t for t in range(max(1, i - 8), i) if t not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        used += 1

        grads, rec = [], None
        for c_val in (rnd(d), rnd(d)):
            x = x0.clone()
            x[c] = c_val
            v = v0.clone().requires_grad_(True)
            qq, kk = x @ wq, x @ wk
            a, _ = pivot_probe.build_arm(kind, qq, kk, gvec, bet, pivots, gen=g,
                                         device=dev, tau=tau, rho=rho, lam=lam)
            if kind.startswith("pivot"):
                hop = a[:, pivots] @ f(a[pivots, :] @ v)   # gather, hub, scatter
            else:
                hop = a @ f(a @ v)
            h = (v + a @ v + hop) @ wo
            grad, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            grads.append(0.0 if grad is None else float(grad[j].sum()))

            if rec is None:
                if kind.startswith("pivot"):
                    w = a[i, pivots] * a[pivots, j]
                else:
                    w = a[i, :] * a[:, j]
                idx = ((pivots == c).nonzero().flatten()
                       if kind.startswith("pivot")
                       else torch.tensor([c], device=w.device))
                tc = float(w[idx].abs().sum()) if idx.numel() else 0.0
                rec = (tc, float(w.sum()) - float(w[idx].sum()) if idx.numel()
                       else float(w.sum()))
        terms.append(rec[0]); bgs.append(rec[1])
        lo, hi = grads
        if lo * hi < 0 and min(abs(lo), abs(hi)) > floor:
            flips += 1

    if used == 0:
        raise ValueError(f"NO DRAWS USED: arm={kind!r} s={s} placement={placement!r}")
    n = used
    mean_t = sum(terms) / n if terms else 0.0
    var = (sum(b * b for b in bgs) / n - (sum(bgs) / n) ** 2) if bgs else 0.0
    return dict(rate=flips / n, k=flips, n=used, term=mean_t,
                sigma=math.sqrt(max(var, 0.0)))


def matched_softmax_operator(q: torch.Tensor, k: torch.Tensor, tau: float = 1.0):
    """Softmax over the SAME scores `tgate` uses: `qhat . khat / tau`, L2-normalized.

    `bench._softmax_operator` scores unnormalized `q.k/sqrt(d)`, which on this
    probe's draws has logit std ~15.8 and a participation ratio of 1.15-1.51 --
    a hard argmax. Any comparison against `tgate` on that operator confounds the
    OPERATOR CLASS with the SCORE SCALE. This is the same non-negative,
    row-stochastic, strictly-causal control with the confound removed, and it is
    the fair stand-in for Star-Transformer's relay softmax.
    """
    s = q.shape[-2]
    qh = torch.nn.functional.normalize(q, dim=-1)
    kh = torch.nn.functional.normalize(k, dim=-1)
    m = torch.ones(s, s, dtype=torch.bool, device=q.device).tril(-1)
    w = (qh @ kh.transpose(-2, -1)) / tau
    a = torch.softmax(w.masked_fill(~m, torch.finfo(q.dtype).min), -1)
    return a.masked_fill(~m, 0.0)


_orig_softmax = pivot_probe.bench._softmax_operator


@contextlib.contextmanager
def matched():
    """Swap the unnormalized softmax for the score-matched one, everywhere the
    probe reaches it. Row 0 has no causal support, so its softmax over an
    all-masked row is left exactly as `bench` leaves it."""
    pivot_probe.bench._softmax_operator = (
        lambda q, k, window=0: matched_softmax_operator(q, k, tau=1.0))
    try:
        yield
    finally:
        pivot_probe.bench._softmax_operator = _orig_softmax


# ---------------------------------------------------------------------------
# THE SCOPE PROBLEM, MADE MEASURABLE.
#
# Every M2 number is measured on `tgate`, which lives in `ceq/bench.py` and
# SHIPS NOWHERE: the Hub config ships `sgate` and `ceq/attention.py` ships the
# row-L1 form. Both of those carry a denominator that sums over context -- which
# `bench.py`'s own `_causal_tgate_operator` docstring names as "the measured
# cause of the 1/s death". So the M-programme may be characterising an operator
# that is not the module, and no amount of extra draws on `tgate` can detect that.
# This swap points the SAME routed measurement at the SHIPPED matrix.
# ---------------------------------------------------------------------------
def _shipped_build(kind, qq, kk, gvec, bet, pivots, *, tau=1.0, rho=1.5,
                   lam=0.10, **kw):
    a = pivot_probe.bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam)
    hop2 = (pivot_probe.pivot_hop2(a, pivots) if kind.startswith("pivot")
            else a @ a)
    return a, hop2


@contextlib.contextmanager
def shipped():
    """Point `pivot_signed` / `dense_signed` at `sgate`, the operator on the Hub."""
    prev = pivot_probe.build_arm
    pivot_probe.build_arm = _shipped_build
    try:
        yield
    finally:
        pivot_probe.build_arm = prev
