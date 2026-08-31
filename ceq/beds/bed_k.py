"""BED-K -- memory-kernel corpora: z_i = sum_j K(i, j) b_j.

Companion to ceq/corpus.py's BED-M (the chain corpus, scan-native). BED-M's
label at position i is a function of position i-1 and a running scan state;
every corpus in the campaign before this file is built that way -- the
blindness CEQ_V15_CONTRACT.md's WHERE WE ARE names explicitly: "no label has
ever contained a delayed cause." BED-K's label depends on positions the
immediately-preceding state cannot summarize:

  (a) pure delay d      K(i, j) = 1 iff j = i - d, else 0
  (b) power-law (fBm)   K(i, j) = (i - j)^(H - 3/2) for j < i, H > 0.5

(b)'s exponent is the FARIMA(0, H-1/2, 0) / fractionally-integrated-noise
tail: the moving-average representation of long-memory noise has impulse
response psi_k ~ k^{(H-1/2)-1} = k^{H-3/2} for large k (Hosking 1981). H > 0.5
gives positively-correlated, slowly-decaying weights -- long memory that no
finite window of the past summarizes.

Both are meant to be ATTENTION-NATIVE (a single fixed-offset or power-law
head reproduces the label) and SCAN-BLIND (Lean #12, `first_order_cannot_
delay`: a first-order recurrence y_i = a*y_{i-1} + b*x_i cannot express
either, because for iid b the label at any nonzero lag is population-
independent of the label's own immediate past). tests/beds/test_bed_k.py is
the must-fire for both halves of that claim, plus the Hurst covariate's
white-noise calibration and the interventional channel's exactness.

SEEDING DISCIPLINE, matching ceq/corpus.py: one `seed` argument, one RNG,
nothing else touches global state. corpus.py uses `random.Random` because its
payload is token ids; this module's payload is real-valued, so
`numpy.random.default_rng` is the equivalent instrument. `build()` returns a
manifest dict rather than an object, for the same reason corpus.py's build()
does -- every field is meant to be read by a caller that never imports this
module's internals.

INTERVENTIONAL CHANNEL. `bump` and `rebuild` are two INDEPENDENT routes to
dz/db_p: `bump` reads the analytic column straight off the kernel; `rebuild`
recomputes z through the same kernel for an arbitrary input, and is what
`finite_diff_jacobian_column` central-differences. `bump` never calls
`rebuild`, and the finite-difference path never reads `bed["K"]` directly --
so their agreement is a check, not an identity (MISTAKES.md V-3).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "kernel_matrix", "build", "build_delay", "build_powerlaw",
    "rebuild", "bump", "finite_diff_jacobian_column",
    "fit_first_order_recurrence", "hard_delay_attention", "estimate_hurst_dfa",
]


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def _delay_kernel_matrix(n: int, d: int) -> np.ndarray:
    """K[i, j] = 1 iff j == i - d, else 0."""
    if d < 0:
        raise ValueError(f"delay must be >= 0, got {d}")
    K = np.zeros((n, n), dtype=np.float64)
    if d < n:
        i = np.arange(d, n)
        K[i, i - d] = 1.0
    return K


def gl_weights(alpha: float, k_max: int) -> np.ndarray:
    """Grunwald-Letnikov / fractional-difference weights psi_0..psi_{k_max}:
    the coefficients of (1-z)^(-alpha), i.e. `(-1)^k * C(-alpha, k)` -- the
    FRACTIONAL INTEGRAL kernel (BED-K's power-law bed is long-memory, H>0.5,
    i.e. integrated; the GL DERIVATIVE kernel has the opposite sign and is
    not this).

    Computed by the stable ratio recurrence, not `scipy.special.binom(-alpha,
    k)`: binom returns NaN at `alpha=1` (the negative-integer branch of
    `binom(-1, k)`), which is exactly PART I's S-K cumulative-sum bind.
    Verified against scipy at alpha=0.3 (agrees to float equality) and at
    alpha=1 (scipy: all NaN; this recurrence: all 1.0, matching
    (1-z)^{-1} = sum z^k).

        psi_0 = 1;  psi_k = psi_{k-1} * (alpha + k - 1) / k
    """
    w = np.empty(k_max + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, k_max + 1):
        w[k] = w[k - 1] * (alpha + k - 1) / k
    return w


def _powerlaw_kernel_matrix(n: int, H: float) -> np.ndarray:
    """K[i, j] = psi_{i-j} for j < i, else 0, where `psi` are the GL weights
    at alpha = H - 1/2 -- the EXACT FARIMA(0, alpha, 0) moving-average
    coefficients, not just their large-lag asymptote. `psi_k ~ k^{alpha-1} =
    k^{H-3/2}` for large k, so this matches CEQ_V15_CONTRACT.md's stated tail
    `(i-j)^(H-3/2)` exactly in the limit, while getting the SHORT-lag weights
    right too: the naive `(i-j)^(H-1.5)` formula has `psi_1 = 1` for every H
    (`1^anything = 1`), which measurably biased the DFA Hurst read (crossover
    documented in `estimate_hurst_dfa`'s docstring); the GL weights give
    `psi_1 = alpha`, which does not have that defect.

    Domain: alpha = H - 1/2 must be in (0, 0.5), i.e. H in (0.5, 1.0) --
    ARFIMA(0, d, 0) is finite-variance/stationary only for `|d| < 0.5`, and
    the contract's `H > 0.5` alone does not state the upper bound. At
    alpha >= 0.5 the weight energy diverges (no finite second moment), so
    "the Hurst exponent" has no population value there and the bed refuses
    to build rather than emit a corpus with an undefined target.
    """
    alpha = H - 0.5
    if not (0.0 < alpha < 0.5):
        raise ValueError(
            f"power-law bed requires H strictly in (0.5, 1.0) for finite-"
            f"variance stationarity (alpha=H-0.5 must be in (0, 0.5)); got "
            f"H={H} (alpha={alpha}). At alpha>=0.5 the GL weights are not "
            "square-summable, so H has no population value to recover."
        )
    psi = gl_weights(alpha, max(n - 1, 0))
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        K[i, :i] = psi[i:0:-1]
    return K


def kernel_matrix(kind: str, n: int, **params) -> np.ndarray:
    """Full n x n K for `kind` in {'delay', 'powerlaw'}. float64."""
    if kind == "delay":
        return _delay_kernel_matrix(n, params["d"])
    if kind == "powerlaw":
        return _powerlaw_kernel_matrix(n, params["H"])
    raise ValueError(f"unknown BED-K kind {kind!r}")


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def build(kind: str, n: int, seed: int, **params) -> dict:
    """z = K @ b, oracle-labeled: K IS the label generator, never fit to it.

    Manifest dict, matching ceq/corpus.py's build() shape: every field a
    caller needs (kind, n, seed, params, K, b, z, pos) sits at the top level,
    no internal state hidden behind a class.

    POSITIONAL CHANNEL. `pos = arange(n)` is an explicit field of the
    manifest, not just an implicit array index: `b[i]`'s position is `pos[i]`
    whether or not a reader bothers to look. This matters because a pure
    delay is a POSITIONAL operation (K depends on i-j, not on any value), so
    an arm that is handed `b` as its only input `x` and is given no
    positional feature has no way to express K regardless of whether it has
    a memory kernel at all -- "no positional feature" and "no memory kernel"
    would be indistinguishable causes of the same failure. `pos` is provided
    so a caller CAN read position explicitly; whether a given arm's encoding
    actually consumes it is that arm's property, not this bed's -- verify it
    reads `pos` (or an equivalent absolute/relative positional code) before
    reading a delay-bed failure as evidence about memory rather than about
    position.
    """
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n).astype(np.float64)
    K = kernel_matrix(kind, n, **params)
    z = K @ b
    pos = np.arange(n, dtype=np.int64)
    return dict(kind=kind, n=n, seed=seed, params=dict(params), K=K, b=b, z=z, pos=pos)


def build_delay(n: int, d: int, seed: int) -> dict:
    return build("delay", n, seed, d=d)


def build_powerlaw(n: int, H: float, seed: int) -> dict:
    return build("powerlaw", n, seed, H=H)


# --------------------------------------------------------------------------
# interventional channel
# --------------------------------------------------------------------------


def rebuild(bed: dict, b: np.ndarray) -> np.ndarray:
    """The generator's forward pass on an ARBITRARY input, through the bed's
    own kernel. Used by the finite-difference cross-check and by the bump-
    movement test; never called from `bump`."""
    return bed["K"] @ b


def bump(bed: dict, p: int, eps: float = 1e-6) -> np.ndarray:
    """do(b_p -> b_p + eps): the ORACLE's response, dz/db_p.

    Exact, not approximated: z = K @ b is linear in b, so dz/db_p IS K[:, p],
    read directly off the generator's own kernel -- no finite difference on a
    model, per CEQ_V15_CONTRACT.md's INTERVENTIONAL CHANNEL clause. `eps` is
    accepted only so the call signature matches a bump on a nonlinear bed;
    on THIS bed the response does not depend on the bump's size, only that it
    is nonzero (a zero-size do() is not an intervention).
    """
    if eps == 0:
        raise ValueError("a do()-bit of size 0 bumps nothing")
    return bed["K"][:, p].copy()


def finite_diff_jacobian_column(bed: dict, p: int, eps: float = 1e-6) -> np.ndarray:
    """Central finite difference of `rebuild`, independent of `bump`."""
    b = bed["b"]
    b_plus, b_minus = b.copy(), b.copy()
    b_plus[p] += eps
    b_minus[p] -= eps
    return (rebuild(bed, b_plus) - rebuild(bed, b_minus)) / (2.0 * eps)


# --------------------------------------------------------------------------
# scan-blindness instrument (Lean #12 backing)
# --------------------------------------------------------------------------


def fit_first_order_recurrence(target: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of y_i = alpha*y_{i-1} + beta*x_i to `target`, using
    `target`'s own true previous value as y_{i-1} -- the most generous
    first-order recurrence a scan could ever be handed, since a real scan
    only has its own previous OUTPUT, never ground truth. Returns
    (alpha_hat, beta_hat, r2), r2 against target[1:]'s own mean.
    """
    target = np.asarray(target, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(target) != len(x) or len(target) < 3:
        raise ValueError("target and x must be aligned, equal-length, len>=3")
    y_prev, x_cur, y_cur = target[:-1], x[1:], target[1:]
    A = np.column_stack([y_prev, x_cur])
    coef, *_ = np.linalg.lstsq(A, y_cur, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y_cur - pred) ** 2))
    ss_tot = float(np.sum((y_cur - y_cur.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(coef[0]), float(coef[1]), r2


# --------------------------------------------------------------------------
# attention-reachability instrument (the other half of Lean #12)
# --------------------------------------------------------------------------


def hard_delay_attention(b: np.ndarray, d: int, pos: np.ndarray | None = None,
                          temperature: float = 45.0) -> np.ndarray:
    """A single hand-set attention head: one-hot query/key coding of `pos`,
    offset by d, fixed temperature, standard softmax. Reproduces
    z_i = b_{i-d} for i >= d to float64 precision -- the softmax residual off
    the true position is bounded by (n-1)*exp(-temperature), far under 1e-12
    for any n <= ~1e5 at temperature=45. Undefined (near-uniform average) for
    i < d, where the delay label itself has no source position; callers
    compare only i >= d.

    `pos` defaults to `arange(len(b))` but is a real parameter, not a
    convenience: this head reads ONLY `b` and `pos`, the two fields
    `build()`'s manifest actually exposes (see `build`'s POSITIONAL CHANNEL
    note) -- it does not reach into `bed["K"]` or any other internal state to
    manufacture position information an arm would not have. That is what
    makes "attention-reachable" here a claim about a channel a real arm could
    also be given, not a claim about what this test script alone can see.
    """
    b = np.asarray(b, dtype=np.float64)
    n = len(b)
    if pos is None:
        pos = np.arange(n)
    pos = np.asarray(pos)
    scores = np.where(pos[:, None] - d == pos[None, :], temperature, 0.0)
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=1, keepdims=True)
    return w @ b


# --------------------------------------------------------------------------
# Hurst covariate (calibrated estimator: DFA, order 1)
# --------------------------------------------------------------------------


def estimate_hurst_dfa(x: np.ndarray, min_win: int | None = None,
                        max_win: int | None = None, n_windows: int = 12) -> float:
    """Detrended Fluctuation Analysis (Peng et al. 1994), order-1 detrending.

    Chosen over R/S: CEQ_V15_CONTRACT.md PART III states the author's own R/S
    read 0.75 on AR(0.5), a short-memory process whose true H is 0.5 -- a
    known R/S failure mode (short-range correlation inflates the rescaled
    range). DFA is the standard calibrated alternative. This module's own
    white-noise and AR(0.5) must-fires (tests/beds/test_bed_k.py) are the
    check that THIS implementation does not carry that bias, not a promise
    that DFA is unbiased in general.

    `min_win` defaults to 3% of the series length rather than a small
    constant. This bed's causal, TRUNCATED power-law kernel (finite history:
    row i only sums j=0..i-1, never an infinite past) has a genuine short-
    range crossover measured directly against its own autocovariance decay,
    not just against DFA: fitting from min_win=8 measured H_hat compressed
    toward 1 (0.86-1.03 for true H in [0.6, 0.9]), useless for discriminating
    H. Restricting the fit to windows >= 3% of n (past the crossover)
    measured a +0.02..+0.10 bias instead, at n=8192, 5-seed averages. This
    was measured on the naive `(i-j)^(H-1.5)` kernel and re-measured after
    switching to the exact GL-weight kernel (`gl_weights`, below) -- the bias
    did NOT meaningfully shrink from that switch (still +0.02..+0.07), so it
    reads as a property of the finite/causal truncation itself, not of which
    short-lag coefficient formula generates the tail. See
    tests/beds/test_bed_k.py's docstrings and V15_N4_BEDK.md's "post-GREEN
    revision" section for the exact runs this default is calibrated against.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if min_win is None:
        min_win = max(8, int(0.03 * n))
    if max_win is None:
        max_win = max(min_win + 1, n // 3)
    y = np.cumsum(x - x.mean())
    sizes = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_windows).astype(int))
    sizes = sizes[sizes >= 4]
    F, used = [], []
    for w in sizes:
        n_seg = n // w
        if n_seg < 2:
            continue
        segs = y[: n_seg * w].reshape(n_seg, w)
        t = np.arange(w, dtype=np.float64)
        rms = [np.sqrt(np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2))
               for seg in segs]
        F.append(float(np.mean(rms)))
        used.append(w)
    if len(used) < 3:
        raise ValueError(f"series too short for DFA at n={n} (min_win={min_win}, max_win={max_win})")
    slope, _ = np.polyfit(np.log(np.array(used, dtype=np.float64)), np.log(np.array(F)), 1)
    return float(slope)
