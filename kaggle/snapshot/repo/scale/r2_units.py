"""R2 -- sign-determinacy of the k x k pivot block, and its replacement.

WHAT R2 CLAIMS. If the SIGN of `(I - A)^{-1}_ij` is fixed by the sign PATTERN of
A alone, no magnitude can change it, so context cannot dilute it -- by
combinatorics rather than by measurement. Brualdi & Shader, *Matrices of
Sign-Solvable Linear Systems*, Cambridge Tracts in Mathematics 116, CUP 1995.

WHAT THIS FILE MEASURES, and it is three different things because the first one
turned out not to be able to answer the question:

  1. `frac_det`  -- R2 AS SPECIFIED. Keep the sign pattern of the k x k pivot
     block, resample MAGNITUDES 10^4 times, call an entry of `(I-A_P)^{-1}`
     determined when its sign is BITWISE CONSTANT across all 10^4 resamples.
     Computed, not estimated: the claim is constancy, and an estimate of a
     constancy is a different quantity.

  2. `frac_det_exact` -- the same number by combinatorics instead of by
     sampling. `A` is strictly lower triangular, so `(I-A)^{-1}_ij` is the sum
     over strictly increasing chains j -> i of products of entries, and the
     entry is sign-determined IFF every chain has the same sign product. A
     three-state DP over {none, only +, only -, mixed} settles it exactly.
     THE EMPIRICAL NUMBER IS ONLY EVER AN UPPER BOUND ON THIS ONE (a resample
     grid can miss the magnitudes that cancel), so their agreement is the
     instrument's calibration and their disagreement is its RED.

  3. `coh` -- PATH COHERENCE, the replacement named in ARSENAL.md C1: not "the
     sign is determined regardless of magnitude" but "the j->i paths through c
     SHARE a sign, so c's contribution ADDS instead of cancelling". Measured at
     hop budget 3 (the calibrated configuration in `run_calib.py`), against a
     same-N same-magnitude random-sign null, because coherence at N terms is
     ~N^{-1/2} by chance alone and a number that only reproduces its own null
     measures the term COUNT, not the coherence.

WHY 1 CANNOT DECIDE ANYTHING -- stated here because it is the finding, not a
caveat. The block is k x k at every s, so anything computed on it alone is flat
in s BY CONSTRUCTION; and for a DENSE pattern every entry at distance i-j = 1
has exactly one chain, hence is determined unconditionally, so

    frac_det >= (k-1) / (k(k-1)/2) = 2/k = 0.25 at k=8

no matter what the operator does. R2's pre-registered kill is "determined
fraction ~0". That kill cannot fire. The numbers below are reported anyway,
because a floor asserted is worth less than a floor measured.

THE STRUCTURAL-ZERO TRAP, and this probe walks into it deliberately once so the
RED is on record. `A` is strictly lower triangular, so every entry of
`(I-A)^{-1}` with i <= j is EXACTLY 0 at every resample -- bitwise-constant
sign, therefore "determined" under a naive reading. For k=8 that is 36 of 64
entries, so the naive fraction starts at 0.5625 for ANY operator including one
with no determinacy at all. `frac_det_naive` is reported beside `frac_det` for
exactly this reason; it is the M2-control-was-zero-by-construction failure in a
new costume and it is worth 0.5625 of a headline number.

PROTOCOL. Every unit records `SCALING` (i=s-1, j=s/4) or `PINNED`. Pivots are
selected by `scale.pivot_probe.select_pivots`, content only, excluding i and j,
so the block this file reads is the block M2 routes through -- the operator is
never rebuilt here, `build_arm` is imported.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.bucket import Journal, run_bucket
from scale.pivot_probe import build_arm, clopper_pearson, loglog_slope, select_pivots

NAME = "r2"
SIZES = [8, 32, 128, 512, 1024, 2048]
K = 8
D = 16
#: 3 decades either side of 1. The qualitative class of a sign pattern is ALL
#: positive magnitudes; a narrow band samples a corner of it and reports
#: cancellations that do not happen there as determinacy. `--red` shows it.
WIDE = math.log(1000.0)
NARROW = 0.1

#: (label, arm, pattern transform). `random` is the chance baseline at the same
#: density; `softmax` must read 1.000 by construction or the instrument is
#: broken; `softmax_broken` is the deliberately-wrong pattern that must read low.
CELLS = [
    ("tgate", "pivot_signed", None),
    ("softmax", "pivot_unsigned", None),
    ("softmax_broken", "pivot_unsigned", "break"),
    ("deltanet", "deltanet", None),
    ("sgate", "sgate", None),
    ("random", "random", None),
]

# --------------------------------------------------------------------------
# exact sign-determinacy: three-state DP over achievable chain sign products
# --------------------------------------------------------------------------
_NONE, _POS, _NEG, _MIX = 0, 1, 2, 3
_FLIP = (_NONE, _NEG, _POS, _MIX)


def exact_states(sign: np.ndarray) -> np.ndarray:
    """`st[i,j]` in {NONE, POS, NEG, MIX} for the chains j -> i of `sign`.

    NONE  = no chain exists (structural zero -- excluded from every fraction)
    POS   = every chain has product +1  -> sign-determined, positive
    NEG   = every chain has product -1  -> sign-determined, negative
    MIX   = both products occur         -> magnitudes decide, not determined
    """
    k = sign.shape[0]
    st = np.zeros((k, k), dtype=np.int8)
    for j in range(k):
        st[j, j] = _POS                      # the empty chain, weight +1
        for i in range(j + 1, k):
            acc = _NONE
            for m in range(j, i):
                if sign[i, m] == 0 or st[m, j] == _NONE:
                    continue
                t = int(st[m, j])
                if sign[i, m] < 0:
                    t = _FLIP[t]
                acc |= t
            st[i, j] = acc
    return st


def resolvent_signs(sign: np.ndarray, n: int, rng, half_width: float) -> np.ndarray:
    """Signs of `(I-A)^{-1}` over `n` magnitude resamples of one sign pattern.

    A is strictly lower triangular, so the Neumann series terminates exactly at
    k-1 terms -- no solver, no conditioning question, no tolerance.
    """
    k = sign.shape[0]
    mag = np.exp(rng.uniform(-half_width, half_width, size=(n, k, k)))
    a = sign[None, :, :] * mag
    r = np.broadcast_to(np.eye(k), (n, k, k)).copy()
    term = a.copy()
    for _ in range(k - 1):
        r += term
        term = term @ a
    return np.sign(r)


def determinacy(sign: np.ndarray, n: int, rng, half_width: float) -> dict:
    """Both readings of the determined fraction, plus the naive one that is
    inflated by the structural zeros."""
    st = exact_states(sign)
    k = sign.shape[0]
    iu = np.tril_indices(k, -1)              # i > j only

    live = st[iu] != _NONE
    ex_det = np.isin(st[iu], (_POS, _NEG)) & live

    s = resolvent_signs(sign, n, rng, half_width)
    const = (s == s[0][None, :, :]).all(axis=0)
    emp_det = const[iu] & live

    n_live = int(live.sum())
    return dict(
        n_live=n_live,
        frac_det=float(emp_det.sum() / n_live) if n_live else float("nan"),
        frac_det_exact=float(ex_det.sum() / n_live) if n_live else float("nan"),
        # the RED: every i<=j entry is exactly 0 at every resample
        frac_det_naive=float(const.sum() / (k * k)),
        n_det=int(emp_det.sum()),
        disagree=int((emp_det != ex_det).sum()),
    )


# --------------------------------------------------------------------------
# path coherence -- the replacement
# --------------------------------------------------------------------------
def through_c_terms(a: torch.Tensor, i: int, j: int, c: int,
                    inter: np.ndarray) -> np.ndarray:
    """Weights of every j -> i path through c of length <= 3, intermediates
    restricted to `inter`.

    length 2:  j -> c -> i
    length 3:  j -> m -> c -> i   (j < m < c)   and   j -> c -> m -> i (c < m < i)

    Routed (inter = P) this bundle has at most k+1 terms at every s; dense
    (inter = everything) it has i-j-2 terms and grows with s. That difference IS
    the pivot mechanism, so the bundle is where the mechanism has to show up.
    """
    an = a.detach().cpu().numpy().astype(np.float64)
    lo = inter[(inter > j) & (inter < c)]
    hi = inter[(inter > c) & (inter < i)]
    w = [an[i, c] * an[c, j]]
    if hi.size:
        w.extend(an[i, hi] * an[hi, c] * an[c, j])
    if lo.size:
        w.extend(an[i, c] * an[c, lo] * an[lo, j])
    return np.asarray(w, dtype=np.float64)


def coherence(w: np.ndarray, rng, n_null: int = 256) -> dict:
    """|sum w| / sum|w| against a same-N same-magnitude random-sign null.

    The null matters: N random-sign terms give ~N^{-1/2} for free, so a raw
    coherence that only reproduces its own null is measuring the term COUNT.
    """
    denom = float(np.abs(w).sum())
    if denom == 0.0:
        return dict(coh=float("nan"), null=float("nan"), n_terms=int(w.size))
    coh = float(abs(w.sum()) / denom)
    mag = np.abs(w)
    sgn = rng.choice((-1.0, 1.0), size=(n_null, w.size))
    null = float(np.mean(np.abs((sgn * mag[None, :]).sum(axis=1)) / denom))
    return dict(coh=coh, null=null, n_terms=int(w.size))


# --------------------------------------------------------------------------
# one draw of the shipped operator
# --------------------------------------------------------------------------
def draw_block(kind: str, s: int, g: torch.Generator, *, k: int = K,
               protocol: str = "SCALING", transform: str | None = None):
    """(sub-block sign pattern, A, P, i, j) for one draw of the SHIPPED operator.

    `build_arm` is imported, never restated: a probe that rebuilds the operator
    proves only that two reimplementations agree, which is how this repo
    published 89,400.180 against 1.667.
    """
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    wq, wk = rnd(D, D), rnd(D, D)
    x0 = rnd(s, D)
    gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
    i, j = (s - 1, max(1, s // 4)) if protocol == "SCALING" else (min(7, s - 1), 1)

    kk = x0 @ wk
    pivots = select_pivots(kk, k, exclude=(i, j))
    pivots = torch.sort(pivots).values
    a, _ = build_arm(kind, x0 @ wq, kk, gvec, bet, pivots, gen=g)

    sub = a[pivots][:, pivots].detach().cpu().numpy().astype(np.float64)
    sign = np.sign(sub).astype(np.int8)
    if transform == "break":
        # deliberately-broken calibration: take the all-positive pattern and
        # negate half its entries. Must read LOW where the untouched pattern
        # reads 1.000, or the instrument cannot tell determinacy from anything.
        flip = np.array(
            torch.randint(0, 2, sign.shape, generator=g).numpy(), dtype=np.int8)
        sign = sign * (1 - 2 * flip)
    return sign, a, pivots, i, j


def compute(p: dict) -> dict:
    """One unit: determinacy on the pivot block + path coherence through c."""
    kind, s = p["arm"], p["s"]
    g = torch.Generator().manual_seed(p["seed"])
    rng = np.random.default_rng(p["seed"])
    n_draws, n_res = p["draws"], p["resamples"]

    det, coh_p, coh_d, null_p, null_d, npt_p, npt_d = [], [], [], [], [], [], []
    naive, exact, dis, plen = [], [], [], []
    for _ in range(n_draws):
        sign, a, pivots, i, j = draw_block(
            kind, s, g, protocol=p["protocol"], transform=p.get("transform"))
        d = determinacy(sign, n_res, rng, p["half_width"])
        det.append(d["frac_det"]); exact.append(d["frac_det_exact"])
        naive.append(d["frac_det_naive"]); dis.append(d["disagree"])
        plen.append(int(pivots.numel()))

        pv = pivots.numpy()
        pool = [int(x) for x in pv if j < x < i]
        if not pool:
            continue
        c = int(pool[int(torch.randint(0, len(pool), (1,), generator=g))])
        wp = through_c_terms(a, i, j, c, pv)
        wd = through_c_terms(a, i, j, c, np.arange(s))
        cp, cd = coherence(wp, rng), coherence(wd, rng)
        for acc, val in ((coh_p, cp["coh"]), (null_p, cp["null"]),
                         (npt_p, cp["n_terms"]), (coh_d, cd["coh"]),
                         (null_d, cd["null"]), (npt_d, cd["n_terms"])):
            if not (isinstance(val, float) and math.isnan(val)):
                acc.append(val)

    m = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(
        frac_det=m(det), frac_det_exact=m(exact), frac_det_naive=m(naive),
        det_sd=float(np.std(det)) if det else float("nan"),
        disagree=int(sum(dis)), n_draws=len(det), pivots=m(plen),
        coh_pivot=m(coh_p), null_pivot=m(null_p), n_terms_pivot=m(npt_p),
        coh_dense=m(coh_d), null_dense=m(null_d), n_terms_dense=m(npt_d),
    )


def units(sizes=None, draws=32, resamples=10_000, protocol="SCALING",
          half_width=WIDE):
    out = []
    for n, (label, arm, tf) in enumerate(CELLS):
        for s in (sizes or SIZES):
            # NOT `hash(label)`: str hashing is salted per process, so a
            # journalled unit would recompute to a different number on resume
            # and the replay assertion would fire on the harness, not on a bug.
            p = dict(arm=arm, s=s, draws=draws, resamples=resamples,
                     protocol=protocol, half_width=half_width, seed=n)
            if tf:
                p["transform"] = tf
            out.append((f"{label}/s{s}", p))
    return out


# --------------------------------------------------------------------------
# RED-first calibration
# --------------------------------------------------------------------------
def trap(s: int, n_draws: int = 4096, k: int = K, seed: int = 0,
         kind: str = "pivot_signed", floor: float = 1e-6) -> dict:
    """DOES SIGN-DETERMINACY KILL THE FLIP M2 MEASURES? Decompose every flip by
    whether the SIGN PATTERN moved.

    Sign-determinacy is invariance to MAGNITUDE at a FIXED pattern. M2's flip
    perturbs `x[c]`, which moves the magnitudes AND the pattern (row c and
    column c of A are functions of `x[c]`). So the two are only the same
    question for an operator whose pattern cannot move -- softmax, all-positive.
    This splits the measured flip rate into `flip & pattern moved` and
    `flip & pattern fixed`; the second column is the one determinacy forbids.

    BOUND TO THE SHIPPED INSTRUMENT, NOT REIMPLEMENTED TWICE. The draw sequence
    is `pivot_probe.run_arm`'s, entry for entry, and the flip test uses
    `grad[j].sum() == (A + hop2)[i,j] * wo.sum()` (BOARD.md WIN #1, entries
    agree to 1.788e-07). The check that this is the same measurement is that
    `k` here must equal the journalled M2 `k` at the same s, seed and draws --
    117 / 127 / 118 at s = 32 / 128 / 512. If it does not, this function is
    void, not M2.
    """
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    i, j = s - 1, max(1, s // 4)
    used = flips = moved = f_and_m = f_and_fixed = 0
    tmoved = ft_m = ft_fixed = 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(D, D), rnd(D, D), rnd(D, D)
        x0, v0 = rnd(s, D), rnd(s, D)
        gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
        kk0 = x0 @ wk
        pivots = select_pivots(kk0, k, exclude=(i, j))
        pivot_set = set(int(p) for p in pivots)
        pool = [p for p in pivot_set if p not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        used += 1
        grads, sg = [], []
        for c_val in (rnd(D), rnd(D)):
            x = x0.clone()
            x[c] = c_val
            a, hop2 = build_arm(kind, x @ wq, x @ wk, gvec, bet, pivots, gen=g)
            grads.append(float(a[i, j] + hop2[i, j]) * float(wo.sum()))
            sg.append((float(a[i, c]), float(a[c, j])))
        lo, hi = grads
        f = lo * hi < 0 and min(abs(lo), abs(hi)) > floor
        pm = sg[0][0] * sg[1][0] < 0 or sg[0][1] * sg[1][1] < 0
        # THE SHARP ONE. Perturbing x[c] changes exactly ONE term of the
        # readout -- t_c = A[i,c]*A[c,j] -- because A[i,j] and every p != c term
        # are functions of tokens other than c. So the only sign in the pattern
        # that the flip can use is sign(t_c). `tm` false means the flip was
        # driven by MAGNITUDE alone at a fixed sign, which is exactly the event
        # sign-determinacy forbids.
        tm = (sg[0][0] * sg[0][1]) * (sg[1][0] * sg[1][1]) < 0
        flips += f
        moved += pm
        tmoved += tm
        f_and_m += f and pm
        f_and_fixed += f and not pm
        ft_m += f and tm
        ft_fixed += f and not tm
    return dict(s=s, n=used, k=flips, rate=flips / max(used, 1),
                pattern_moved=moved / max(used, 1),
                flip_and_moved=f_and_m, flip_and_fixed=f_and_fixed,
                p_flip_given_moved=f_and_m / max(moved, 1),
                p_flip_given_fixed=f_and_fixed / max(used - moved, 1),
                term_sign_moved=tmoved / max(used, 1),
                flip_and_term_moved=ft_m, flip_and_term_fixed=ft_fixed)


def red() -> int:
    """Three REDs that must be SEEN before any real number is trusted.

    RED 1  the naive denominator reads >= 0.56 on a pattern with no determinacy
    RED 2  narrow magnitude resampling reports determinacy the exact DP denies
    RED 3  the R2 kill "determined fraction ~0" cannot fire: a dense k x k
           pattern has a hard floor at 2/k
    Plus the two GREENs the task names: softmax must read 1.000, and the
    deliberately-broken pattern must read low.
    """
    rng = np.random.default_rng(0)
    k, n = K, 10_000
    print("=== R2 RED-FIRST CALIBRATION (k=%d, %d resamples/pattern) ===\n" % (k, n))

    dense = np.tril(np.ones((k, k), dtype=np.int8), -1)
    pats = [dense * np.where(rng.random((k, k)) < 0.5, -1, 1).astype(np.int8)
            for _ in range(200)]
    for p in pats:
        p[np.triu_indices(k)] = 0

    d0 = determinacy(pats[0], n, rng, WIDE)
    print("RED 1  structural zeros inflate the naive fraction")
    print(f"       frac_det_naive = {d0['frac_det_naive']:.4f} on all {k*k} "
          f"entries, of which {(k*k+k)//2} are exactly 0 at every resample")
    print(f"       frac_det (i>j only) = {d0['frac_det']:.4f}")
    red1 = d0["frac_det_naive"] >= 0.5625 and d0["frac_det_naive"] > d0["frac_det"]
    print(f"       RED 1 {'FIRES' if red1 else 'does not fire'}\n")

    narrow = [determinacy(p, n, rng, NARROW) for p in pats[:50]]
    wide = [determinacy(p, n, rng, WIDE) for p in pats[:50]]
    nd = float(np.mean([x["frac_det"] for x in narrow]))
    ne = float(np.mean([x["frac_det_exact"] for x in narrow]))
    wd = float(np.mean([x["frac_det"] for x in wide]))
    we = float(np.mean([x["frac_det_exact"] for x in wide]))
    print("RED 2  a narrow magnitude band fakes determinacy")
    print(f"       narrow (+-{NARROW}):     empirical {nd:.4f}  exact {ne:.4f}  "
          f"disagreements {sum(x['disagree'] for x in narrow)}")
    print(f"       wide   (+-{WIDE:.2f}):  empirical {wd:.4f}  exact {we:.4f}  "
          f"disagreements {sum(x['disagree'] for x in wide)}")
    red2 = nd > ne + 1e-9
    print(f"       RED 2 {'FIRES' if red2 else 'does not fire'}\n")

    fr = [float(np.isin(exact_states(p)[np.tril_indices(k, -1)], (_POS, _NEG)).sum()
                / (k * (k - 1) / 2)) for p in
          [dense * np.where(rng.random((k, k)) < 0.5, -1, 1).astype(np.int8)
           for _ in range(20_000)]]
    print("RED 3  the R2 kill cannot fire on a dense pattern")
    print(f"       20000 random dense sign patterns, exact determined fraction:")
    print(f"       mean {np.mean(fr):.4f}  min {min(fr):.4f}  max {max(fr):.4f}"
          f"  floor 2/k = {2/k:.4f}")
    red3 = min(fr) >= 2 / k - 1e-12
    print(f"       RED 3 {'FIRES' if red3 else 'does not fire'} "
          f"(kill 'frac ~ 0' unreachable)\n")

    allpos = dense.copy()
    dsm = determinacy(allpos, n, rng, WIDE)
    half = allpos * np.where(rng.random((k, k)) < 0.5, -1, 1).astype(np.int8)
    half[np.triu_indices(k)] = 0
    dbk = determinacy(half, n, rng, WIDE)
    print("GREEN  the two the task names")
    print(f"       all-positive (softmax) pattern : frac_det = {dsm['frac_det']:.4f} "
          f"(must be 1.0000)")
    print(f"       deliberately broken pattern    : frac_det = {dbk['frac_det']:.4f} "
          f"(must be low)")
    ok = dsm["frac_det"] == 1.0 and dbk["frac_det"] < 0.7
    print(f"       {'OK' if ok else 'INSTRUMENT BROKEN'}\n")
    return 0 if (red1 and red2 and red3 and ok) else 1


def self_test() -> int:
    """Bind the closed form to the shipped autograd path, and the DP to brute
    force. Neither is assumed."""
    import itertools
    rng = np.random.default_rng(1)
    k = 6
    bad = 0
    for _ in range(300):                      # DP vs exhaustive chain enumeration
        sign = np.tril(np.where(rng.random((k, k)) < 0.5, -1, 1), -1).astype(np.int8)
        st = exact_states(sign)
        for i in range(k):
            for j in range(i):
                prods = set()
                for r in range(i - j):
                    for mid in itertools.combinations(range(j + 1, i), r):
                        ch = (j,) + mid + (i,)
                        pr = 1
                        for t in range(len(ch) - 1):
                            pr *= int(sign[ch[t + 1], ch[t]])
                        prods.add(pr)
                want = _POS if prods == {1} else _NEG if prods == {-1} else _MIX
                bad += int(st[i, j] != want)
    print(f"self-test: DP vs exhaustive enumeration, mismatches = {bad}")
    if bad:
        return 2

    g = torch.Generator().manual_seed(7)
    s = 32
    _, a, pivots, i, j = draw_block("pivot_signed", s, g)
    print(f"self-test: |P| = {int(pivots.numel())} at s={s}, i={i}, j={j}")
    return 0


def report(sizes=None) -> None:
    """Read the journal and print the curve. Refuses nothing -- prints what is
    there and says what is missing, because a verdict on a partial set is the
    failure ADR-001 exists to prevent."""
    done = Journal(NAME).done()
    want = units(sizes)
    print(f"PROTOCOL: SCALING (i=s-1, j=s/4, c drawn from P)")
    print(f"journal {len(done)}/{len(want)} units\n")
    sz = sizes or SIZES
    hdr = "".join(f"{s:>12}" for s in sz)

    # --- the determined-fraction curve, with both intervals ---------------
    #
    # CLOPPER-PEARSON IS THE ONE THE TASK ASKS FOR AND IT IS THE OPTIMISTIC ONE.
    # Its n is entries, and the entries of one block are NOT independent -- they
    # share edges of the same sign pattern. The draw-level interval
    # (mean +- 1.96 sd / sqrt(draws)) is over independent draws and is the
    # honest one. Both are printed; where they disagree, believe the wider.
    print("determined fraction on the k x k pivot block, entries with i>j, "
          "sign bitwise-constant over 10^4 magnitude resamples")
    print(f"{'arm':>16}" + "".join(f"{s:>26}" for s in sz))
    for label, _, _ in CELLS:
        cells = []
        for s in sz:
            rec = done.get(f"{label}/s{s}")
            if rec is None:
                cells.append("---")
                continue
            v = rec["value"]
            live = int(v["pivots"] * (v["pivots"] - 1) / 2)
            n = int(v["n_draws"] * live)
            kk = int(round(v["frac_det"] * n))
            lo, hi = clopper_pearson(kk, n)
            half = 1.96 * v["det_sd"] / math.sqrt(max(v["n_draws"], 1))
            cells.append(f"{v['frac_det']:.4f} CP[{lo:.3f},{hi:.3f}] "
                         f"d[{v['frac_det']-half:.3f},{v['frac_det']+half:.3f}]")
        print(f"{label:>16}" + "".join(f"{c:>26}" for c in cells))
    print(f"\n  floor for a DENSE k x k pattern = 2/k = {2/K:.4f} "
          f"(every i-j=1 entry has one chain, hence is determined for free)")
    print("  softmax reads 1.0000 BY CONSTRUCTION (all-positive pattern)")
    print("  `softmax_broken` breaks the PATTERN only; its coherence rows are "
          "softmax's and are 1.0 by construction, not a measurement\n")
    for what, key in (("determined fraction (i>j, empirical == exact)", "frac_det"),
                      ("naive fraction (RED: structural zeros counted)", "frac_det_naive"),
                      ("path coherence, PIVOT-routed bundle", "coh_pivot"),
                      ("  its same-N random-sign null", "null_pivot"),
                      ("  terms in the bundle", "n_terms_pivot"),
                      ("path coherence, DENSE bundle", "coh_dense"),
                      ("  its same-N random-sign null", "null_dense"),
                      ("  terms in the bundle", "n_terms_dense")):
        print(f"{what}\n{'arm':>16}{hdr}   slope")
        for label, _, _ in CELLS:
            vals, cells = [], []
            for s in sz:
                rec = done.get(f"{label}/s{s}")
                v = rec["value"][key] if rec else float("nan")
                vals.append(v)
                cells.append("---" if rec is None else f"{v:.4f}")
            good = [(s, v) for s, v in zip(sz, vals) if v == v and v > 0]
            sl, npts = loglog_slope([s for s, _ in good], [v for _, v in good])
            print(f"{label:>16}" + "".join(f"{c:>12}" for c in cells)
                  + f"   {sl:+.3f} ({npts}pt)")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", action="store_true", help="RED-first calibration")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trap", nargs="+", type=int, default=None,
                    help="sizes to run the flip/pattern decomposition on")
    ap.add_argument("--trap-draws", type=int, default=4096)
    ap.add_argument("--bucket", action="store_true")
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.red:
        return red()
    if a.report:
        report(a.sizes)
        return 0
    if a.trap:
        print("PROTOCOL: SCALING (i=s-1, j=s/4, c drawn from P)  arm=pivot_signed")
        print("BIND: k must match journalled M2 (117/127/118 at s=32/128/512, "
              "seed 0, 4096 draws) or this table is void.\n")
        print(f"{'s':>6}{'n':>7}{'k':>6}{'rate':>10}{'t_c sign moved':>16}"
              f"{'flip&t moved':>14}{'flip&t FIXED':>14}{'share fixed':>13}")
        for s in a.trap:
            r = trap(s, n_draws=a.trap_draws)
            print(f"{r['s']:>6}{r['n']:>7}{r['k']:>6}{r['rate']:>10.5f}"
                  f"{r['term_sign_moved']:>16.4f}{r['flip_and_term_moved']:>14}"
                  f"{r['flip_and_term_fixed']:>14}"
                  f"{r['flip_and_term_fixed']/max(r['k'],1):>13.4f}", flush=True)
        print("\n`flip & t FIXED` is the count sign-determinacy would remove: a")
        print("flip driven by MAGNITUDE alone, at an unchanged sign for the only")
        print("term c enters. The rest need the sign of t_c to move, which no")
        print("determinacy condition on a fixed pattern forbids.")
        return 0
    if a.bucket:
        u = units(a.sizes, draws=a.draws, resamples=a.resamples)
        run_bucket(NAME, u, compute, budget_s=a.budget, verify=1)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
