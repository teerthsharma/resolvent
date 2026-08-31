"""X35' (a) the exact source solve, and (b) time-reversal localization.

CEQ_V15_2_DELTA.md poses hidden-cause inference as an INVERSE PROBLEM rather
than a detection: the residual is a field, the hidden cause is a source, and
the forward map between them is the architecture's resolvent carrier.

    (a) h_hat = (I - A) r                    `[RUN 8.9e-16, two sources]`
    (b) W^T r, Wiener-deconvolved when the
        noise is colored                     `[RUN: exact at sd 0.3]`

WHY (a) IS A SUBTRACTION AND NOT A SOLVE -- the whole content of the node.

`ceq/nonnormal.py` builds A strictly lower triangular (`.tril(-1)`, diagonal
EXCLUDED), so A^n = 0 and the forward map is the terminating occupancy sum
W = sum_{k<n} A^k, proved to be the exact two-sided inverse of (I - A) by
`CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` with the nilpotency
hypothesis discharged by `CEQ.Nilpotent.pow_card_eq_zero`. The consequence
`lean/CEQ/V15Source.lean` states is not `W * W^-1 = I` -- true of every
invertible matrix, and it licenses nothing -- but that W's inverse is
`I - A`: ONE application of A, no powers. So

    * the inverse is as sparse as A plus a diagonal (measured on the shipped
      carrier: nnz(I - A) = 251 against nnz(W) = 1703 at n=128, d=5), and
    * applying it costs one matrix-vector product against A -- O(nnz) -- with
      no factorization, no iteration, and no conditioning question, because
      nothing is solved.

`exact_source` therefore takes A and NEVER takes W. A source solve that needed
W would already have paid for the thing the closed form exists to avoid.

WHAT THE FORWARD MAP IS ON THE PRODUCTION PATH, AND WHERE THAT BINDS.

`ceq/beds/bed_k.py` adds the planted latent `u` to `z` DIRECTLY, after the
kernel; V15_X35A_RESIDUAL.md §9 files that as a limit in those words. With the
oracle visible model the production-path residual is therefore

    r  =  u + observation noise

identically, i.e. the forward map from source to residual field is the
IDENTITY and A = 0 on that path. tests/x35p measures this rather than assuming
it. Two consequences, both stated rather than worked around:

  1. On the production path the exact solve reduces to `r` and the recovery
     error is float64 rounding of `(K b + u) - K b`. That is what `8.9e-16` is,
     and it is why the delta says the figure is "not a convergence tolerance".
  2. An identity forward map has no inverse worth testing, and two estimators
     that are both the identity cannot be compared at all. `wave_field` is the
     one composition step this module adds: the SAME production-path source
     `bed["plant"]["u"]`, the SAME production-path observation stream
     (`ceq.x35.residual.observe` -- same salt, same seed), with the
     architecture's own carrier `W = occupancy(A, n)` between them, A built
     from the production-path kernel `bed_k.kernel_matrix`. Every component is
     the shipped one. Nothing here is a reimplementation of the residual
     instrument; it is the residual instrument's own pieces composed in the
     order the inverse problem needs.

WHY (b) SHIPS NORMALIZED. `W^T` is a REVERSE cumulative sum along the carrier's
hop lattice: `(W^T r)_i` sums r over the indices reachable FROM i. Its null
energy therefore falls monotonically with index (measured: energy-vs-index
correlation -0.9916 under no plant), so a first-crossing rule on the raw field
calls index 0 at every seed and every noise level, and `argmax |W^T r|` is
dominated by i = 0 whatever the source did. Dividing by the column norms
||W e_j|| -- the standard geometric-spreading correction of time-reversal
imaging, and exactly the normalization that makes the null variance
index-independent -- restores flatness (measured: +0.1582) and with it the
focusing property. `normalize=True` is the default for that reason; the raw
field is available for the record.

NO CRB, NO KK. CEQ_V15_2_DELTA.md strikes any CRB or Kramers-Kronig number
quoted before its own must-fire, and both instruments belong to a parallel
node (`kk.py`, `crb.py`). This module computes neither and quotes neither, and
reports RAW localization error -- a hit rate and a mean absolute index error --
with no floor beside it. It also makes no optimality claim: "near the floor" is
a CRB claim by implication and is struck the same way.

float64 throughout. Seeding follows `ceq/x35/residual.py`: one `seed` per run,
every stream derived from it through a fixed salt, nothing touches global
state.
"""
from __future__ import annotations

import numpy as np

from ceq.beds import bed_k
from ceq.x35 import residual as rx

__all__ = [
    "occupancy", "carrier", "exact_source", "adjoint_source", "wiener_source",
    "ar1_noise", "ar1_cov", "wave_field", "onset_sweep", "POINT_SALT",
]

#: Fixed salt for the point-source index stream, matching `residual.OBS_SALT`'s
#: role: `default_rng([POINT_SALT, seed])` shares no state with the bed's or the
#: observation channel's draws, so the source location at seed s is independent
#: of the noise realisation at seed s by construction.
POINT_SALT = 0x35B

#: Carriers are pure functions of (n, kind, params, gamma) and cost ~n/d matrix
#: products to build, so they are memoized. The arrays are shared: callers must
#: not write to them.
_CARRIERS: dict = {}


# --------------------------------------------------------------------------
# the carrier
# --------------------------------------------------------------------------


def occupancy(A: np.ndarray, N: int | None = None) -> np.ndarray:
    """`sum_{k<N} A^k` -- `CEQ.Occupancy.occupancy`, evaluated.

    The loop breaks the moment a power is identically zero, which for a
    strictly lower-triangular A happens at k = n at the latest
    (`CEQ.Nilpotent.pow_card_eq_zero`) and at k = ceil(n/d) for the delay
    carrier. The break is an early exit, not a truncation: every omitted term
    IS zero, so the sum is exact and there is no tail to bound. That is the
    difference between this and a Neumann series, and it is the hypothesis
    Lean #15 composes with.
    """
    n = A.shape[0]
    N = n if N is None else N
    W = np.eye(n, dtype=np.float64)
    T = np.eye(n, dtype=np.float64)
    for _ in range(1, N):
        T = T @ A
        if not T.any():
            break
        W = W + T
    return W


def carrier(*, n: int, kind: str, params: dict, gamma: float = 1.0) -> dict:
    """`A` from the production-path kernel, and its resolvent `W = sum_k A^k`.

    A IS `bed_k.kernel_matrix(kind, n, **params)` scaled by `gamma`, so the hop
    the inverse problem is posed on is the bed's own memory kernel and not a
    second operator invented here. Both BED-K kinds are strictly lower
    triangular (`delay` needs d >= 1; `powerlaw` is `.tril(-1)` by
    construction), hence nilpotent, hence W is exact.

    A note on which kind makes the O(nnz) claim LOCAL as well as cheap: the
    delay kernel has one nonzero per row, so `I - A` has two and the source
    solve reads `h_i = r_i - gamma*r_{i-d}` -- a first-order difference in the
    literal sense. The power-law kernel is dense lower-triangular, so `I - A`
    has ~n^2/2 nonzeros; the solve is still one mat-vec and still O(nnz) and
    still needs no factorization, but it is not local. The distinction is
    stated because "sparse and local" is a property of the hop, while "no
    solve, no conditioning question" is a property of the nilpotency and holds
    for both.
    """
    key = (n, kind, tuple(sorted(params.items())), float(gamma))
    if key not in _CARRIERS:
        A = gamma * bed_k.kernel_matrix(kind, n, **params)
        _CARRIERS[key] = dict(A=A, W=occupancy(A, n),
                              col_norm=np.linalg.norm(occupancy(A, n), axis=0))
    return _CARRIERS[key]


# --------------------------------------------------------------------------
# (a) the exact source solve
# --------------------------------------------------------------------------


def exact_source(A: np.ndarray, r: np.ndarray) -> np.ndarray:
    """`h_hat = (I - A) r = r - A r`. THE SUBTRACTION LEMMA, applied.

    One mat-vec against A. No `np.linalg.solve`, no factorization, no
    regularization parameter, and no iteration count -- which is why the
    recovery error is machine epsilon and not a tolerance. W is not an
    argument on purpose (see the module docstring).
    """
    return r - A @ r


# --------------------------------------------------------------------------
# (b) time-reversal localization
# --------------------------------------------------------------------------


def adjoint_source(W: np.ndarray, r: np.ndarray, normalize: bool = True) -> np.ndarray:
    """`W^T r`, the back-propagated field -- time reversal (Fink).

    For a Toeplitz causal W, `W^T = J W J` with J the index flip, so this IS
    "reverse the field, re-propagate, reverse again"; it is written as the
    adjoint because that is the form whose cost and null statistics are
    readable.

    `normalize` divides by the column norms ||W e_j||. Under noise alone
    `var((W^T n)_j) = sigma^2 ||W e_j||^2`, which falls with j, so the raw
    field's null is index-dependent and no fixed threshold or argmax over it
    means anything (measured in tests/x35p: raw energy-vs-index correlation
    -0.9916, normalized +0.1582). Dividing by ||W e_j|| makes the null
    variance sigma^2 at every index. Default True; pass False for the raw
    field the delta writes.
    """
    out = W.T @ r
    return out / np.linalg.norm(W, axis=0) if normalize else out


def wiener_source(W: np.ndarray, r: np.ndarray, noise_var: float,
                  source_var: float, noise_cov: np.ndarray | None = None) -> np.ndarray:
    """Wiener deconvolution of the adjoint:

        h_hat = (W^T C^-1 W + I / sigma_h^2)^-1 W^T C^-1 r

    the linear MMSE estimator of `h` from `r = W h + n` with `n ~ N(0, C)` and
    prior `h ~ N(0, sigma_h^2 I)`. `noise_cov` defaults to `noise_var * I`, in
    which case this is ridge-regularized least squares with lambda =
    noise_var / source_var -- the delta's "when the noise is colored" qualifier
    is what the argument is for, and handing it the correct C is measurably
    better than handing it `sigma^2 I` (tests/x35p, AR(1) phi=0.7).

    Two things this estimator is handed that the exact solve is not, both of
    which favour it and are therefore stated rather than buried: `noise_var`
    and `source_var` are hyperparameters supplied from outside, and the tests
    supply their TRUE values. A deployed version would have to estimate them.

    `noise_var <= 0` returns the lambda -> 0 limit, ordinary least squares.
    That branch exists because `C` is singular at zero noise, and it is worth
    noticing what it costs: the least-squares path is a linear SOLVE and
    carries its solve's rounding, while `exact_source` at the same point is a
    subtraction and is exact. That difference is the whole of the exact
    inverse's surviving territory (V15_X35P_SOURCE.md).
    """
    n = W.shape[0]
    if noise_var <= 0.0:
        return np.linalg.solve(W.T @ W, W.T @ r)
    C = noise_var * np.eye(n) if noise_cov is None else noise_cov
    Wc = W.T @ np.linalg.inv(C)
    return np.linalg.solve(Wc @ W + np.eye(n) / source_var, Wc @ r)


# --------------------------------------------------------------------------
# noise channels
# --------------------------------------------------------------------------


def ar1_noise(seed: int, n: int, sd: float, phi: float = 0.0) -> np.ndarray:
    """Observation noise with marginal sd `sd` and lag-1 correlation `phi`,
    drawn from the PRODUCTION observation stream (`residual.observe` at the
    same salt and seed) and then filtered.

    At `phi == 0` this returns `sd * g` for exactly the `g` the production path
    would have used -- bitwise, since the filter degenerates to the identity
    and the scale to `sd`. So the white sweep and the colored sweep are matched
    pairs on the same draws, not independent experiments.
    """
    e = rx.observe(dict(z=np.zeros(n, dtype=np.float64), n=n), 1.0, seed)
    if phi == 0.0:
        return sd * e
    x = np.empty(n, dtype=np.float64)
    x[0] = e[0] / np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return sd * np.sqrt(1.0 - phi * phi) * x


def ar1_cov(n: int, sd: float, phi: float = 0.0) -> np.ndarray:
    """`C_ij = sd^2 * phi^|i-j|` -- the exact stationary covariance of
    `ar1_noise`, so the Wiener estimator can be handed the truth rather than an
    estimate of it. At `phi = 0` this is bitwise `sd^2 * I` (`0.0 ** 0 == 1.0`,
    `0.0 ** k == 0.0`), which is the same matrix `wiener_source` builds by
    default -- that is what makes the phi=0 row of the colored sweep an exact
    control rather than an approximate one."""
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return sd * sd * float(phi) ** lag


# --------------------------------------------------------------------------
# the field the inverse problem reads
# --------------------------------------------------------------------------


def wave_field(seed: int, *, n: int, kind: str, params: dict, noise_sd: float,
               plant=None, gamma: float = 1.0, phi: float = 0.0) -> dict:
    """`y = W h + noise`, with `h` the production-path planted latent.

    The ONE composition step this module adds, and the reason it is needed is
    in the module docstring: on the production path the latent is added to `z`
    after the kernel, so the residual's forward map is the identity and the
    inverse problem is empty. Here the same latent is propagated by the
    architecture's own carrier before it is observed, which is the
    configuration the delta's `(I - A)^-1` names.

    Returns `bed`, `A`, `W`, the ground-truth source `h` and the field `y`.
    `h` is `bed["plant"]["u"]` verbatim -- the generator's own realisation,
    recorded for scoring only; no estimator in this module reads `bed`.
    """
    bed = bed_k.build(kind, n, seed, plant=plant, **params)
    c = carrier(n=n, kind=kind, params=params, gamma=gamma)
    h = bed["plant"]["u"]
    return dict(bed=bed, A=c["A"], W=c["W"], h=h,
                y=c["W"] @ h + ar1_noise(seed, n, noise_sd, phi))


# --------------------------------------------------------------------------
# the sweep the kill is read off
# --------------------------------------------------------------------------


def onset_sweep(noise_sds, *, n: int, kind: str, params: dict, t_star: int,
                magnitude: float, cal_seeds, eval_seeds, alpha: float = 0.01,
                gamma: float = 1.0) -> list[dict]:
    """Localization of an EXTENDED (step-onset) source by its onset, scored
    with the SAME memoryless comparator X35a uses (`residual.detect`), for the
    exact solve and for the Wiener adjoint.

    Each estimator gets its OWN threshold, calibrated as the (1-alpha) quantile
    of run-max energy over its own NO-PLANT runs on `cal_seeds` -- the same
    construction `residual.calibrate` uses, applied to the estimator's output
    instead of to the raw residual, because the estimators have different null
    scales and a shared threshold would be comparing scales rather than
    localizations. `cal_seeds` must be disjoint from `eval_seeds`.

    A run in which nothing is called is scored as error `n`, i.e. the worst a
    call could be, rather than dropped: dropping misses would let an estimator
    that alarms rarely look accurate. The miss count is returned beside the
    error so the two are never confused.

    Errors are PAIRED on seeds -- the same field goes to both estimators -- so
    `paired_diff` and the win/tie/loss counts are the reading, not the
    difference of two means over independent draws.

    The raw adjoint is not swept: it is not an onset estimator on this field at
    all (its null energy is monotone in the index; see `adjoint_source`), and
    reporting a number for it here would be reporting the operator's shape
    rather than its localization. It is swept on the point-source task, which
    is the one time reversal is for.
    """
    assert set(cal_seeds).isdisjoint(set(eval_seeds)), "M-2: blocks must be disjoint"
    c = carrier(n=n, kind=kind, params=params, gamma=gamma)
    A, W = c["A"], c["W"]
    cfg = dict(n=n, kind=kind, params=params, gamma=gamma)
    plant = [dict(index=t_star, magnitude=magnitude)]
    rows = []
    for sd in noise_sds:
        est = dict(exact=lambda y: exact_source(A, y),
                   wiener=lambda y, sd=sd: wiener_source(W, y, sd * sd, magnitude ** 2))
        thr = {k: float(np.quantile(
                   [np.max(f(wave_field(s, noise_sd=sd, plant=None, **cfg)["y"]) ** 2)
                    for s in cal_seeds], 1.0 - alpha, method="higher"))
               for k, f in est.items()}
        err = {k: [] for k in est}
        miss = {k: 0 for k in est}
        for s in eval_seeds:
            y = wave_field(s, noise_sd=sd, plant=plant, **cfg)["y"]
            for k, f in est.items():
                o = rx.detect(f(y), thr[k])["onset"]
                if o is None:
                    miss[k] += 1
                    err[k].append(float(n))
                else:
                    err[k].append(float(abs(o - t_star)))
        e_ex = np.array(err["exact"])
        e_wi = np.array(err["wiener"])
        d = e_ex - e_wi
        rows.append(dict(noise_sd=float(sd), exact_mae=float(e_ex.mean()),
                         wiener_mae=float(e_wi.mean()), paired_diff=float(d.mean()),
                         exact_wins=int((d < 0).sum()), ties=int((d == 0).sum()),
                         wiener_wins=int((d > 0).sum()),
                         exact_misses=miss["exact"], wiener_misses=miss["wiener"],
                         exact_within1=float((e_ex <= 1).mean()),
                         wiener_within1=float((e_wi <= 1).mean()),
                         thresholds=thr, runs=len(e_ex)))
    return rows
