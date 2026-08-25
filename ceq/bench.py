"""What a benchmark can resolve, and the CPU harness that reports it.

Two independent things live here because they answer one question -- *which
benchmark is worth the GPU budget* -- and splitting them into two modules would
mean two imports for one decision.

1. **Benchmark resolving power.** Arithmetic over a scoring rule. No model, no
   GPU, no network. Given an evaluation set of n tasks scored by exact match,
   how large must the gap between two arms be before the benchmark can see it?
   For ARC-AGI this is the number that decides whether the planned run happens.

2. **The influence-sign probe.** The module's one measured distinguishing
   property is a negative influence-Jacobian entry, -9.000e-01 against exactly
   0.000e+00 for every non-negative operator. This probe asks the sharper
   question that property implies at the level of a whole attention block:
   can the SIGN of token j's influence on token i be flipped by changing a
   THIRD token? See `sign_flip_rate`.

Nothing here imports a network. Everything runs on CPU in seconds.
"""
from __future__ import annotations

import functools
import math
import statistics

import torch
from scipy.stats import fisher_exact

# --------------------------------------------------------------------------
# 1. Benchmark resolving power
# --------------------------------------------------------------------------


def arc_task_solved(guesses, target: torch.Tensor) -> bool:
    """ARC's scoring rule: exact match on the full output grid, pass@k.

    "Scoring is exact match on the full output grid: correct size, correct
    colour in every cell. One cell wrong scores zero." -- arcprize.org/guide/1

    Shape counts. A guess of the wrong dimensions is wrong even if every cell it
    does have is right, which is why the shape check comes before the value
    check rather than being left to broadcasting.
    """
    return any(g.shape == target.shape and bool(torch.equal(g, target))
               for g in guesses)


def log10_random_grid_chance(rows: int, cols: int, colors: int,
                             attempts: int = 2) -> float:
    """log10 P(at least one of `attempts` uniform random grids matches exactly).

    Returned in log10 because the linear number underflows float64 for any grid
    bigger than about 9x9 and would print as a uninformative 0.0.

    Assumes the output dimensions are already known, which is generous -- a real
    solver must also pick them -- so this is an UPPER bound on chance success.
    """
    log10_single = -rows * cols * math.log10(colors)
    return log10_single + math.log10(attempts)


def resolution_floor(n: int) -> float:
    """The smallest non-zero score representable on an n-task set: 1/n.

    Exact-match scoring is a count of solved tasks. Nothing between 0 and 1/n
    exists, so any true skill in that range reports as 0.
    """
    return 1.0 / n


def zero_success_upper_bound(n: int, conf: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on p after 0 successes in n.

    P(0 successes | p) = (1-p)^n. Setting that to 1-conf and solving gives
    p_hi = 1 - (1-conf)^(1/n). This is the exact Clopper-Pearson upper limit at
    k = 0, not the rule-of-three approximation 3/n.

    It is the honest reading of a reported "0%": the observation is consistent
    with any true skill up to this bound.
    """
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def separation_p(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """One-sided Fisher exact p for "arm B solves more than arm A".

    Fisher rather than a normal approximation because the counts are tiny and
    frequently zero, where a z-test is not merely imprecise but undefined.
    """
    table = [[k_b, n_b - k_b], [k_a, n_a - k_a]]
    return float(fisher_exact(table, alternative="greater").pvalue)


def min_successes_for_separation(n: int, k_baseline: int = 0,
                                 alpha: float = 0.05) -> int:
    """Smallest k such that k/n separates from k_baseline/n at level alpha.

    The entry price of the benchmark as a comparison instrument: below this many
    solved tasks, the better arm is not distinguishable from the control no
    matter how much compute produced it.
    """
    for k in range(k_baseline + 1, n + 1):
        if separation_p(k_baseline, n, k, n) < alpha:
            return k
    raise ValueError(f"no k <= {n} separates from {k_baseline} at alpha={alpha}")


# --------------------------------------------------------------------------
# 2. The influence-sign probe -- which benchmark could see this module?
# --------------------------------------------------------------------------
#
# W6 measured `min d(out_i)/d(v_j)` at -9.000e-01 for the signed path sum and
# exactly +0.000e+00 for softmax, APPNP and the max-plus star. That number is
# about a BARE operator. A real block has signed `W_v` and `W_o`, either of
# which supplies a minus sign, so "can produce a negative number" separates
# nothing at block level and would pick the wrong benchmark.
#
# The property that does separate them is CONTENT-CONDITIONAL SIGN: whether a
# THIRD token can decide if j helps or hurts i. `W_v` and `W_o` are constants
# and cannot; non-negative `a_ij` can only rescale. That is the property the
# word `not` needs, and it is what these probes measure.


@functools.lru_cache(maxsize=32)
def _causal_mask_pair(s: int, window: int, device: str):
    """`(m, ~m)`, memoised. A pure function of `(s, window, device)`.

    Both the allocation and the complement were being redone on EVERY operator
    call -- `_causal_sgate_operator` evaluated `~m` four times per call,
    `_softmax_operator` twice -- and neither depends on `q`, `k` or the batch.
    Memoising a pure function cannot move a bit, which is the whole reason this
    is the candidate that binds: `scale/lastrow_bind.py` measured 1.4x and 8.0x
    from slicing rows earlier and BOTH changed the number (fwd maxdiff 7.45e-09,
    grad 1.49e-08), because changing a GEMM's `m` dimension changes which BLAS
    micro-kernel runs and therefore the accumulation order. This changes no
    tensor shape and no arithmetic op.

    TWO COSTS, DECLARED.
      1. The cached tensors are SHARED, not copied. Every caller in this repo
         only reads them (`masked_fill`, `sum`, `torch.equal`), so nothing
         mutates one today; an in-place write by a future caller would corrupt
         every later call. Checked by grep over `_causal_mask` at the time of
         writing: 8 call sites, all read-only.
      2. The key is `str(device)`, so `cuda` and `cuda:0` are two entries for
         one device. Harmless (they build identical masks) and irrelevant on
         this CPU-only path, but it is a key collision by string and not by
         identity, so it is written down rather than assumed away.
    """
    m = torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)
    m = m if window <= 0 else m.triu(-window)
    return m, ~m


def _causal_mask(s: int, device, window: int = 0) -> torch.Tensor:
    """Strictly-causal boolean mask, optionally banded to a sliding window.

    `window = 0` is the unbounded causal row every published number was measured
    on. `window = w` keeps `i - w <= j < i`, the bounded receptive field of
    sliding-window attention.

    Returns the CACHED tensor from `_causal_mask_pair`; do not write to it.
    """
    return _causal_mask_pair(s, window, str(device))[0]


def _causal_signed_operator(q: torch.Tensor, k: torch.Tensor,
                            rho: float = 0.9, window: int = 0) -> torch.Tensor:
    """The shipped operator: raw logits, row-L1 normalised, strictly causal.

    Same construction as `ceq.lm.Attention.operator`, restated over plain [S, D]
    tensors so the probe does not drag a whole LM in to ask one question.
    """
    _, nm = _causal_mask_pair(q.shape[-2], window, str(q.device))
    w = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).masked_fill(nm, 0.0)
    return rho * w / w.abs().sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)


def _softmax_operator(q: torch.Tensor, k: torch.Tensor,
                      window: int = 0) -> torch.Tensor:
    """Non-negative control: same causal mask, same scaling, softmax rows."""
    _, nm = _causal_mask_pair(q.shape[-2], window, str(q.device))
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(w.masked_fill(nm, torch.finfo(q.dtype).min), -1).masked_fill(nm, 0.0)


def _causal_sgate_operator(q: torch.Tensor, k: torch.Tensor,
                           rho: float = 1.5, lam: float = 0.10,
                           window: int = 0) -> torch.Tensor:
    """The campaign's operator: a difference of two softmaxes over the SAME logits.

        A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)

    Both halves keep `exp` sharpening, which the row-L1 form lacks -- `rho*w/sum|w|`
    is homogeneous of degree zero, so it has no temperature channel at all. Signed
    because the negative half can dominate on a given entry.

    Whether the SIGN of an entry can be moved by a THIRD token is the open
    question this operator was never measured on, and is why it is added here.
    """
    _, nm = _causal_mask_pair(q.shape[-2], window, str(q.device))
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    pm = torch.softmax((-w).masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def _causal_signmag_operator(q: torch.Tensor, k: torch.Tensor,
                             rho: float = 1.5, window: int = 0) -> torch.Tensor:
    """Sign from the PAIR, magnitude from the row:  A = rho * sgn(w) * softmax(|w|).

    `sgate` decides an entry's sign by comparing two globally normalized
    quantities, so a token's leverage over any sign is its share of the prefix
    and falls as the prefix grows. Here `sgn(w_ij)` is a function of (i, j)
    alone: no third token enters it and no amount of context dilutes it. Only
    the magnitude is normalized, which is what keeps row L1 at exactly rho and
    leaves `CEQ.Nilpotent.pow_card_eq_zero` applicable unchanged.

    Identical to Cog Attention's matrix (arXiv:2411.07176 Eq. 3),
    `SignExp(p)/sum_k |SignExp(p_k)|` with `SignExp(p) = sgn(p)exp(sgn(p)p - m)`,
    which reduces to `sgn(p) * softmax(|p|)`. That work is SINGLE-HOP; the
    multi-hop path sum over this matrix is what is being probed.
    """
    _, nm = _causal_mask_pair(q.shape[-2], window, str(q.device))
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mag = torch.softmax(w.abs().masked_fill(nm, torch.finfo(q.dtype).min), -1)
    return rho * torch.sign(w) * mag.masked_fill(nm, 0.0)


def _causal_deltanet_operator(k: torch.Tensor, beta: torch.Tensor,
                             window: int = 0) -> torch.Tensor:
    """DeltaNet's WY matrix, arXiv:2406.06484 `eq:inverse`, as a path-sum base.

        T = (I + tril(diag(beta) K K^T, -1))^-1 diag(beta)

    so the matrix being inverted is `M = tril(diag(beta) K K^T, -1)` and the
    inverse is `sum_k (-M)^k`, finite because strictly-triangular is nilpotent.
    This returns `-M`: content-dependent (K and beta), signed per entry
    (`k_i . k_j` is unconstrained), strictly causal. The `diag(beta)` postfactor
    is a per-row rescale that cannot change any entry's sign, so it is dropped --
    the probe reads signs.

    Shipping: `fla/ops/utils/solve_tril.py` ("Compute the inverse of the matrix
    I + A. A should be strictly lower triangular") and sglang's
    `chunk_kda_fwd_kernel_inter_solve_fused`.
    """
    _, nm = _causal_mask_pair(k.shape[-2], window, str(k.device))
    kk = torch.nn.functional.normalize(k, dim=-1)
    return -(beta[:, None] * (kk @ kk.transpose(-2, -1))).masked_fill(nm, 0.0)


def _causal_tgate_operator(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
                           tau: float, window: int = 0,
                           static_scale: bool = False) -> torch.Tensor:
    """UNNORMALIZED, signed, QUERY-dependent, strictly causal path-sum base.

        A_ij = g_i * tanh( (qhat_i . khat_j) / tau ),   j < i

    The cell no published operator occupies. Verified against primary sources
    rather than against this repo's own reimplementations:

      * DeltaNet (arXiv:2406.06484) is denominator-free and multi-hop but its
        matrix is KEY-KEY. Paper Eq., as rendered at ar5iv: the state update is
        `S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T` and the UT
        transform inverts `I + tril(diag(beta) K K^T, -1)`. The query enters
        ONLY at readout, `o_t = S_t q_t`. Independently: in the shipping `fla`
        kernel, `prepare_wy_repr_fwd` is called with `(k, v, beta)` -- `q` is
        not passed to the function that builds the triangular matrix at all.
      * SignGT / SimA / Cog are query-dependent but single-hop AND normalized.
      * ParaFormer (arXiv:2512.14619) is multi-hop but its base matrix is a
        softmax, hence non-negative; only its hop COEFFICIENTS are signed.

    WHY NO DENOMINATOR. The measured cause of the 1/s death was never
    signedness, hop count, or the normalizer's TYPE -- it was any denominator
    that sums over context. Every normalized arm dilutes as `s` grows because a
    row's shares are REALLOCATED among more competitors; the denominator-free
    arm does not. `tanh` supplies per-entry boundedness with no sum over `s`,
    and `g_i` supplies magnitude control that is per-token and therefore local.

    `tau` is LEARNED in the module. It restores the temperature channel whose
    absence was measured as exact degree-zero homogeneity (spread 0.000e+00 over
    a 64x scale sweep, grad.q at 1e-14) and identified as the W10 widening-gap
    mechanism. With `qhat`, `khat` L2-normalized the argument lies in [-1, 1],
    so `tau` is a genuine sharpness knob rather than a rescale that cancels.

    NO CONTRACTION CERTIFICATE IS NEEDED AND NONE IS CLAIMED. Strict causality
    makes A nilpotent, and `CEQ.Nilpotent.pow_card_eq_zero` proves `A^n = 0`
    over any `CommRing` with no sign hypothesis and NO MAGNITUDE HYPOTHESIS. The
    series terminates exactly regardless of how large entries become.

    THE PRICE, STATED. Row L1 is no longer pinned at `rho`; it grows like
    `O(s * g)`, so hop-2 magnitudes scale roughly with `s`. `static_scale`
    divides row `i` by `sqrt(#visible keys)`, which is a FIXED function of
    position and of nothing else. It rescales a row uniformly rather than
    reallocating shares between its entries, which is why it is not expected to
    reintroduce the decay -- but that is a prediction, and `tgatex` exists so it
    is measured rather than assumed.
    """
    s = q.shape[-2]
    m, nm = _causal_mask_pair(s, window, str(q.device))
    qh = torch.nn.functional.normalize(q, dim=-1)
    kh = torch.nn.functional.normalize(k, dim=-1)
    w = torch.tanh((qh @ kh.transpose(-2, -1)) / tau)
    a = (g[:, None] * w).masked_fill(nm, 0.0)
    if static_scale:
        vis = m.sum(-1, keepdim=True).clamp_min(1).to(a.dtype)
        a = a / vis.sqrt()
    return a


def sign_flip_draws(kind: str, *, depth: int = 1, n_draws: int = 64, s: int = 8,
                    d: int = 16, i: int = 7, j: int = 1, c: int = 4,
                    seed: int = 0, device=None, hops: int = 3,
                    wrt: str = "v",
                    rho: float = 1.5, lam: float = 0.10,
                    gammas=None, window: int = 0, tau: float = 1.0) -> list:
    """The RAW measurement: one `(lo, hi)` gradient pair per draw, nothing discarded.

    This is the half of the old `sign_flip_rate` that is a MEASUREMENT. The other
    half -- deciding which draws are numerical dust -- is `flip_rate`, and the two
    are separate functions because fusing them is what produced Inspector strikes
    S2, S3 and S4. See `flip_rate` for the full statement.

    `lo` and `hi` are `d(out_i)/d(v_j)` (or `d(out_i)/d(x_j)`, see `wrt`) summed
    over the feature axis, at the two values of token `c`. A sign flip is
    `lo * hi < 0`; whether such a flip COUNTS is `flip_rate`'s question, not this
    function's.

    `kind` is one of "signed", "sgate", "softmax", "softmax_gelu", "paraformer".
    Depth stacks that many attention layers, each with its own random projections.

    "paraformer" is arXiv:2512.14619 Eq. 9, `Z = sum_k gamma_k Ahat^k V` with
    `Ahat = Softmax(QK^T/sqrt(d))` -- a NON-NEGATIVE base matrix carrying SIGNED
    hop coefficients, which the paper defines as "learnable weights" in R. It is
    here because it is the nearest published construction to a signed multi-hop
    path sum, and because "signed coefficients are weaker than a signed matrix"
    is a claim, not a fact, until it is measured on the same instrument. The
    gammas are drawn ONCE per draw and shared across both values of token c, so
    any flip is caused by the token and not by the coefficients moving.

    `wrt` selects which path is differentiated:

      "v" -- the VALUE path, `d(out_i)/d(v_j)` with `v` an independent leaf.
             This is the W6 quantity. For a non-negative operator with linear
             value projections it factors as (non-negative path weight) x (fixed
             matrix), so the sign cannot depend on c at any depth.
      "x" -- the INPUT path, `d(out_i)/d(x_j)`. Not sign-constrained for anyone,
             because q and k make `a_ij` itself a signed function of the input.
             Present so a test can show the two probes differ; a result measured
             on this path would prove nothing.

    No discard rule is applied here and none is available here. Every draw is
    returned, including draws whose gradients are 1e-30 -- `flip_rate` decides.

    `rho` and `lam` reach the `sgate` operator only. They default to the
    campaign's shipped point, so every previously published number off this
    probe is unchanged by their existence -- checked by
    `tests/cameron/test_parity_is_the_wrong_target.py::test_the_calibration_still_reproduces_after_the_knob_is_added`.
    `lam = 0` deletes the negative half and leaves `rho * softmax(w)`, which is
    non-negative; that is the control INSIDE the operator family.
    """
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    gelu, op_kind = kind.endswith("_gelu"), kind.split("_")[0]
    if op_kind not in ("signed", "softmax", "sgate", "paraformer", "signmag",
                       "deltanet", "tgate", "tgatex"):
        raise ValueError(kind)

    pairs = []
    for _ in range(n_draws):
        wq = [rnd(d, d) for _ in range(depth)]
        wk = [rnd(d, d) for _ in range(depth)]
        wo = [rnd(d, d) for _ in range(depth)]
        # drawn only for the arm that uses them, so every other arm's random
        # stream -- and therefore every published number -- is untouched.
        gam = (list(gammas) if gammas is not None
               else [1.0] + [float(rnd(1)) for _ in range(hops)])             if op_kind == "paraformer" else None
        bet = torch.sigmoid(rnd(s)) if op_kind == "deltanet" else None
        # same discipline as `gam` and `bet`: drawn ONLY for the arms that use
        # it, so every other arm's random stream -- and therefore every number
        # already published off this probe -- is bit-identical to before.
        gvec = (torch.sigmoid(rnd(s))
                if op_kind in ("tgate", "tgatex") else None)
        x0, v0 = rnd(s, d), rnd(s, d)
        grads = []
        for c_val in (rnd(d), rnd(d)):
            x = x0.clone()
            x[c] = c_val                       # assign BEFORE requires_grad_:
            v = v0.clone()                     # in-place on a leaf would raise
            if wrt == "x":
                x, leaf = x.requires_grad_(True), None
                leaf = x
            else:
                v, leaf = v.requires_grad_(True), None
                leaf = v
            h = v
            for layer in range(depth):
                qq, kk = x @ wq[layer], x @ wk[layer]
                if op_kind == "deltanet":
                    a = _causal_deltanet_operator(kk, bet, window=window)
                elif op_kind in ("tgate", "tgatex"):
                    a = _causal_tgate_operator(qq, kk, gvec, tau, window=window,
                                               static_scale=(op_kind == "tgatex"))
                elif op_kind == "signed":
                    a = _causal_signed_operator(qq, kk, window=window)
                elif op_kind == "sgate":
                    a = _causal_sgate_operator(qq, kk, rho=rho, lam=lam, window=window)
                elif op_kind == "signmag":
                    a = _causal_signmag_operator(qq, kk, rho=rho, window=window)
                else:
                    a = _softmax_operator(qq, kk, window=window)
                if op_kind in ("signed", "sgate", "signmag", "deltanet",
                               "tgate", "tgatex"):
                    acc, term = h, h           # h + A h + ... + A^hops h
                    for _ in range(hops):
                        term = a @ term
                        acc = acc + term
                    h = acc
                elif op_kind == "paraformer":
                    acc, term = gam[0] * h, h  # sum_k gamma_k A^k h, Eq. 9
                    for k in range(hops):
                        term = a @ term
                        acc = acc + gam[k + 1] * term
                    h = acc
                else:
                    h = a @ h
                h = h @ wo[layer]
                if gelu and layer < depth - 1:
                    h = torch.nn.functional.gelu(h)
            # `leaf` can be absent from the graph -- ParaFormer truncated to
            # its k=0 term propagates V unchanged, so the input path has no edge
            # at all. That is an exact zero gradient, not an error.
            if not h.requires_grad:
                grads.append(0.0)
                continue
            grad, = torch.autograd.grad(h[i].sum(), leaf, allow_unused=True)
            grads.append(0.0 if grad is None else float(grad[j].sum()))
        pairs.append((grads[0], grads[1]))
    return pairs


def flip_rate(draws, floor: float = 1e-6, rel: float = 0.0) -> float:
    """Fraction of `(lo, hi)` pairs that are a sign flip surviving the discard rule.

        keep if   lo * hi < 0   and   min(|lo|, |hi|) > max(floor, rel * ref)
        ref = median over the draws of max(|lo|, |hi|)

    WHY THIS IS A SEPARATE FUNCTION. `floor` alone is an ABSOLUTE threshold on a
    quantity whose scale differs between arms and drifts with `s`. sgate's row-L1
    is pinned at `rho = 1.5`; DeltaNet's grows roughly linearly (4.23 / 21.53 /
    81.62 at s = 32/128/512). One shared absolute cut therefore removes a large
    share of one arm's draws and none of the other's, which makes it an
    ARM-DEPENDENT SAMPLE FILTER rather than a dust rule. Measured at the published
    geometry `j = s/4`, `floor=1e-6` reads sgate 0.026367 vs deltanet 0.054688
    (2.07x) while `floor=0` reads 0.050781 vs 0.054688 (1.08x, overlapping):
    sgate moves 1.93x and DeltaNet does not move at all.

    `rel` is the scale-RELATIVE criterion. `ref` is estimated from the arm's own
    draws, so the cut carries the arm's units and the rule is invariant under any
    uniform rescaling of that arm's gradients -- which is the property an
    exclusion rule on a SIGN question must have, because the sign of a number is
    itself invariant under positive rescaling. A rule that is not scale-invariant
    imports the arm's magnitude into a scale-free measurement.

    `rel = 0.0` is the default and reproduces the historical absolute-floor
    behaviour bit-for-bit, which is why the four `run_calib.py` numbers do not
    move. It is NOT the recommended setting for any CROSS-ARM or CROSS-`s`
    comparison; see `tests/foreman/test_absolute_floor_is_an_arm_filter.py`.

    `floor` and `rel` compose by `max`, so `floor=0.0, rel=1e-6` is the pure
    relative rule and `floor=1e-6, rel=0.0` is the pure absolute one.
    """
    n = len(draws)
    if n == 0:
        return 0.0
    cut = floor
    if rel > 0.0:
        cut = max(cut, rel * statistics.median(
            [max(abs(a), abs(b)) for a, b in draws]))
    return sum(1 for a, b in draws
               if a * b < 0 and min(abs(a), abs(b)) > cut) / n


def discard_fraction(draws, floor: float = 1e-6, rel: float = 0.0) -> float:
    """Share of the SIGN-FLIPPED draws that the discard rule throws away.

    The number no published rate ever carried. `flip_rate` reports what survived;
    this reports what did not, which is the only way to see that a rule is
    biting two arms at different rates.
    """
    flipped = [(a, b) for a, b in draws if a * b < 0]
    if not flipped:
        return 0.0
    kept = flip_rate(draws, floor, rel) * len(draws)
    return (len(flipped) - kept) / len(flipped)


def sign_flip_rate(kind: str, *, floor: float = 1e-6, rel: float = 0.0,
                   **kw) -> float:
    """`flip_rate(sign_flip_draws(...))`. Kept as-is so every caller still works.

    Bit-identical to the pre-split function at the default `rel = 0.0`: the draw
    loop is unchanged, so the RNG stream is unchanged, and the surviving rule is
    the same comparison it always was.
    """
    return flip_rate(sign_flip_draws(kind, **kw), floor=floor, rel=rel)
