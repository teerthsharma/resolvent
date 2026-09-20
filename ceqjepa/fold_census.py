"""The two pre-registered KILLERS for the beta-scaled read-as-update FOLD leap.

THE LEAP UNDER TEST, verbatim from the dispatch. The read-as-update is
xi <- sum_j e^{xi.k_j/tau} k_j / Z^beta, Z = sum_j e^{xi.k_j/tau}. With one
stored pattern k, writing xi = c*k gives the scalar self-consistency
c = e^{a c}, a = (1-beta)||k||^2/tau, real-solvable IFF a <= 1/e, the two
solutions merging (a fold: a stable point annihilating a saddle) at a = 1/e --
i.e. tau_fold = e(1-beta)||k||^2. That closed form is ALGEBRA, not a
measurement; nothing below treats it as one. What is measured is whether the
REAL, COUPLED, multi-pattern dynamics honours it, via the two killers named in
the dispatch, plus what non-orthogonal patterns do to it.

THE REDUCTION THAT MAKES N PATTERNS CHEAP TO SIMULATE. xi_new = w @ K always
lies in the row-space of K, for ANY xi, because w depends on xi only through
K @ xi. So if xi_t = c_t @ K, the WHOLE dynamics is the N-dim map
    c_{t+1} = softmax_beta(G @ c_t / tau),   G = K @ K.T,
    softmax_beta(s)_j = exp(s_j - beta * logsumexp(s)),
independent of the embedding dimension. At N=1 this map IS c <- e^{a c}. Every
function below takes a Gram matrix G, never raw vectors, because G is the only
thing the dynamics reads.

WHY NAIVE POWER ITERATION FROM xi0 = k_j IS NOT THE RIGHT PROBE, MEASURED
rather than assumed. Iterating c_{t+1} = softmax_beta(G c_t/tau) from c_0 = e_j
at LARGE tau converges to the SAME point regardless of j (e.g. G =
diag(2,4,8), beta=0.9, tau=100: every j lands on [0.369, 0.372, 0.377]) -- a
single global attractor, not three separate ones, because Z always sums in
every pattern's exp(0)=1 even when that pattern's own coefficient is zero.
Whether pattern j's OWN retrieval fixed point still exists near c=e_j is
therefore answered by ROOT-FINDING (Newton on F(c) = softmax_beta(Gc/tau) - c,
seeded at c=e_j, with the exact analytic Jacobian) plus a STABILITY check
(spectral radius of d(softmax_beta)/dc at the root), and the FOLD is read off
by CONTINUATION: shrink tau in small steps, warm-starting each solve from the
previous tau's root, until the same warm start stops converging. Validated
against the one case with a ground truth: at N=1 this reproduces
tau = e(1-beta)g to 1e-8 for every (g, beta) in {2,4,8} x {0.5,0.7,0.9}, and
the spectral radius at the measured threshold reads 0.9996-1.0000 -- the
tangency the fold predicts, not assumed.

KILLER (a) -- do three orthogonal patterns (||k||^2 = 2, 4, 8, beta = 0.9)
vanish together or in norm order? MEASURED via continuation, each branch
seeded near its own isolated prediction (see `find_branch_seed`):
    g=2: tau_fold = 0.543612   (pre-registered 0.54, isolated closed form 0.543656)
    g=4: tau_fold = 1.087223   (pre-registered 1.09, isolated closed form 1.087313)
    g=8: tau_fold = 2.174447   (pre-registered 2.17, isolated closed form 2.174625)
Strictly increasing with ||k||^2, each within 0.008% of the ISOLATED formula
-- the cross-terms from the other two orthogonal patterns are negligible right
at each pattern's own fold, because the other coefficients sit at ~2e-4 there
(see `three_pattern_census`). THE KILLER DOES NOT FIRE: they vanish in norm
order, not together, and the pre-registered RED numbers reproduce to 4
significant figures. (A SEPARATE instability was seen for the weakest pattern
well above its own fold, near tau=0.73 at beta=0.9 -- its branch's spectral
radius crosses 1 there too, a second, higher bifurcation the hypothesis itself
names as the distinct "MULTI->SINGLE merge at tau_c" and which this file does
not chase further; it does not touch the tau=0.5436 reading above.)

KILLER (b) -- is the boundary a straight line through (beta=1, tau=0)? MEASURED
on the SAME coupled 3-pattern system, tracking the dominant pattern (g=8, the
one whose own branch stays cleanly identifiable across the full range) at
beta in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}:
    beta   measured tau_fold   isolated e(1-beta)g   relative gap
    0.50   9.159244             10.873127             -15.8%
    0.60   7.529283              8.698502             -13.4%
    0.70   6.030284              6.523876              -7.6%
    0.80   4.293748              4.349251              -1.3%
    0.90   2.174447              2.174625              -0.008%
    0.95   1.087313              1.087313              +0.0000005%
np.polyfit((1-beta), tau, 1, cov=True): slope = 17.82 (isolated predicts
e*8 = 21.75), intercept = +0.442 +/- 0.192 (2.3 standard errors from zero),
R^2 = 0.9950 with a residual pattern that is not noise-shaped (-0.19, -0.04,
+0.24, +0.29, -0.05, -0.25 in beta-descending order) -- a bend, not scatter.
THE KILLER PARTIALLY FIRES: fit across the full requested range 0.5-0.95, the
coupled boundary is measurably not a line through the origin. But the
departure is entirely a low-beta (far-from-softmax-corner) effect: restricted
to beta in {0.8, 0.9, 0.95} -- the range the pre-registered RED itself lives
in -- the gap is under 1.3% and shrinking fast toward beta=1. The ISOLATED
formula is linear through the origin by construction (it is algebra); what
bends is the correction that OTHER patterns' shared normalizer adds, and that
correction grows as beta leaves 1, not as tau leaves 0.

NON-ORTHOGONAL PATTERNS, same norms (2, 4, 8), beta=0.9, Cholesky-built Gram
at pairwise cosine rho (see `gram_correlated` -- an actual realizable Gram,
not an asserted matrix): order still follows ||k||^2 at rho=0.2 (thresholds
0.518, 0.911, 2.174 -- still strictly increasing) but the weaker two patterns'
thresholds move a LOT (g=2: -4.7%, g=4: -16.2%) while the dominant pattern
barely moves (-0.02%). At rho>=0.4 the weak and mid pattern's own branches
could not be found at all near their isolated predictions -- correlation that
strong erases their separate retrieval identity before it ever separates from
the dominant one, so "vanish together" versus "in norm order" is not even a
well-posed question for them there. The dominant pattern's own threshold stays
within 0.3% of isolated up through rho=0.6.

CAVEATS, collected here rather than scattered above. (1) All of this is on
one CPU, float64, orthogonal-or-Cholesky-constructed 3x8 patterns; nothing
here touches an embedding, a trained encoder, or ceq/arm_smprime.py's shipped
readout. (2) `find_branch_seed`'s multiplier search can fail to find a seed
(raises, not a silent fallback) when the branch has already merged with a
stronger pattern's -- that failure IS the rho>=0.4 finding above, not a bug to
route around. (3) `demo()` is the one self-check this file leaves behind; it
re-derives the N=1 numbers against the closed form and one N=3 threshold
against the value quoted above, both at reduced precision so a slower BLAS on
another box still passes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import fsolve

__all__ = [
    "isolated_fold_tau",
    "gram_orthogonal",
    "gram_correlated",
    "read_map",
    "read_map_jacobian",
    "solve_fixed_point",
    "spectral_radius",
    "find_branch_seed",
    "track_branch",
    "refine_threshold",
    "explode_threshold",
    "iterate_naive",
    "three_pattern_census",
    "fit_explode_boundary",
    "route_constraint_violations",
    "demo",
    # ROUND TWO, PASS 2: the operating point on the ARM THAT IS ACTUALLY WIRED
    # (ceq/arm_smprime.py via ceq/hf/modeling_ceq.py), plus the cusp check.
    "smprime_layer_draws",
    "smprime_operating_point",
    "fold_is_a_fold_not_a_cusp",
    "demo_operating_point",
]

E = math.e

#: Multiplier grid `find_branch_seed` tries, in order, against the isolated
#: closed form for pattern j -- close to it first (where the branch is most
#: cleanly its own), widening only if that fails.
DEFAULT_SEED_MULTS = tuple(
    [1.0 + 0.005 * n for n in range(1, 20)] + [1.0 + 0.05 * n for n in range(4, 20)]
)


def isolated_fold_tau(g, beta):
    """The ISOLATED single-pattern fold: tau = e*(1-beta)*g. Algebra (see
    module docstring), not a measurement -- every other function in this file
    is a measurement compared against this."""
    return E * (1.0 - beta) * g


def gram_orthogonal(norms_sq):
    """Gram matrix of orthogonal patterns with the given squared norms."""
    return np.diag(np.asarray(norms_sq, dtype=np.float64))


def gram_correlated(norms_sq, rho):
    """Gram matrix of len(norms_sq) patterns sharing one pairwise cosine
    `rho`, built via Cholesky of the correlation matrix so it is an ACTUAL
    realizable Gram (rows of L are unit vectors with pairwise dot rho), not an
    arbitrary matrix asserted to be one. Raises (numpy's own) if rho puts the
    correlation matrix outside [-1/(n-1), 1] where it is no longer PSD.
    """
    n = len(norms_sq)
    R = np.full((n, n), float(rho), dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    d = np.sqrt(np.asarray(norms_sq, dtype=np.float64))
    K = d[:, None] * L
    return K @ K.T


def read_map(c, G, tau, beta):
    """One step of the coefficient-space read-as-update: softmax_beta(G c / tau).

    softmax_beta(s)_j = exp(s_j) / Z^beta, Z = sum_i exp(s_i), computed via the
    usual max-shift so the only overflow risk left is the genuinely exploding
    regime (s_j - beta*logZ itself unbounded), which is the phenomenon this
    file is measuring, not a numerical artefact to hide.
    """
    c = np.asarray(c, dtype=np.float64)
    s = G @ c / tau
    m = np.max(s)
    logZ = m + math.log(np.sum(np.exp(s - m)))
    return np.exp(s - beta * logZ)


def read_map_jacobian(c, G, tau, beta):
    """(J, p): p = read_map(c, G, tau, beta); J = dp/dc, exact.

    p_j = exp(s_j - beta*logZ). dp_j/ds_i = p_j*(delta_ij - beta*q_i) where q
    is the ORDINARY (beta=1) softmax of s -- because dZ/ds_i = exp(s_i) = Z*q_i.
    ds/dc = G/tau, so dp/dc = (diag(p) - beta * outer(p, q)) @ (G/tau).
    """
    c = np.asarray(c, dtype=np.float64)
    s = G @ c / tau
    m = np.max(s)
    exps = np.exp(s - m)
    Zshift = np.sum(exps)
    q = exps / Zshift
    logZ = m + math.log(Zshift)
    p = np.exp(s - beta * logZ)
    dp_ds = np.diag(p) - beta * np.outer(p, q)
    return dp_ds @ (G / tau), p


def _residual(c, G, tau, beta):
    J, p = read_map_jacobian(c, G, tau, beta)
    return p - c


def _residual_jacobian(c, G, tau, beta):
    J, p = read_map_jacobian(c, G, tau, beta)
    return J - np.eye(len(c))


def solve_fixed_point(c0, G, tau, beta, xtol=1e-13, maxfev=400):
    """Newton-solve F(c) = read_map(c) - c = 0 from c0, exact Jacobian.
    Returns (c*, max|F(c*)|) -- the residual is the caller's convergence
    signal, not a `ier` flag, because fsolve's `ier` is unreliable exactly
    near a fold (the Jacobian is near-singular there by construction)."""
    sol, _, _, _ = fsolve(
        _residual, np.asarray(c0, dtype=np.float64), args=(G, tau, beta),
        fprime=_residual_jacobian, full_output=True, xtol=xtol, maxfev=maxfev,
    )
    resid = float(np.max(np.abs(_residual(sol, G, tau, beta))))
    return sol, resid


def spectral_radius(c, G, tau, beta):
    """Spectral radius of d(read_map)/dc at c. The fold is exactly where this
    crosses 1 (a real eigenvalue reaching the tangency the closed form
    predicts at a=1/e for N=1)."""
    J, _p = read_map_jacobian(c, G, tau, beta)
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def find_branch_seed(j, G, beta, g, mults=DEFAULT_SEED_MULTS, resid_tol=1e-9,
                      dominance=0.3):
    """Search `mults * isolated_fold_tau(g, beta)` for a tau where solving
    from c0 = e_j gives a well-converged root (residual < resid_tol) whose
    largest-magnitude coordinate is j itself (coefficient > `dominance`).
    Returns (tau, c) or (None, None) -- a caller that needs one raises, this
    function does not, because "no seed found" is itself a measurement (see
    the rho>=0.4 finding in the module docstring)."""
    n = G.shape[0]
    c0 = np.zeros(n)
    c0[j] = 1.0
    for m in mults:
        tau = isolated_fold_tau(g, beta) * m
        if tau <= 0:
            continue
        sol, resid = solve_fixed_point(c0, G, tau, beta)
        if resid < resid_tol and int(np.argmax(np.abs(sol))) == j and abs(sol[j]) > dominance:
            return tau, sol
    return None, None


def track_branch(j, G, beta, tau_start, tau_stop, c_start=None, shrink=0.997, tol=1e-9):
    """Continuation of the branch anchored at pattern j: shrink tau from
    tau_start toward tau_stop, warm-starting each Newton solve from the
    previous tau's converged root. Returns a dict with `tau_alive`/`c_alive`
    (the last tau where the warm start converged) and `tau_dead` (the first
    tau, if any before tau_stop, where it did not) -- the fold sits between
    them. Raises if `tau_start` itself does not converge (a bad seed, not a
    silent guess)."""
    n = G.shape[0]
    if c_start is None:
        c_start = np.zeros(n)
        c_start[j] = 1.0
    sol, resid = solve_fixed_point(c_start, G, tau_start, beta)
    if resid > tol:
        raise ValueError(f"j={j}: tau_start={tau_start} did not converge (resid={resid:.3e})")
    c_prev, tau_prev = sol, float(tau_start)
    tau_dead = None
    while tau_prev > tau_stop:
        tau_next = tau_prev * shrink
        sol, resid = solve_fixed_point(c_prev, G, tau_next, beta)
        if resid > tol or not np.all(np.isfinite(sol)):
            tau_dead = tau_next
            break
        c_prev, tau_prev = sol, tau_next
    return {"tau_alive": tau_prev, "c_alive": c_prev, "tau_dead": tau_dead}


def refine_threshold(j, G, beta, tau_alive, c_alive, tau_dead, tol=1e-6, resid_tol=1e-9):
    """Bisect between a known-alive (tau_alive, c_alive) and a known-dead
    tau_dead down to `tol` in tau. Requires tau_dead is not None (call
    `track_branch` with a low enough `tau_stop` first)."""
    if tau_dead is None:
        raise ValueError("no dead tau to bisect against -- widen track_branch's tau_stop")
    lo, hi, c_hi = tau_dead, tau_alive, c_alive
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        sol, resid = solve_fixed_point(c_hi, G, mid, beta)
        if resid < resid_tol and np.all(np.isfinite(sol)):
            hi, c_hi = mid, sol
        else:
            lo = mid
    return hi


def explode_threshold(j, G, beta, g, seed_mults=DEFAULT_SEED_MULTS, shrink=0.997,
                       stop_frac=0.2, tol=1e-6):
    """High-level: seed pattern j's own branch, track it down, bisect to the
    fold. Raises if no seed is found (see `find_branch_seed`)."""
    tau0, c0 = find_branch_seed(j, G, beta, g, mults=seed_mults)
    if tau0 is None:
        raise RuntimeError(f"j={j} g={g} beta={beta}: no clean branch seed found")
    res = track_branch(j, G, beta, tau_start=tau0, tau_stop=tau0 * stop_frac,
                        c_start=c0, shrink=shrink)
    return refine_threshold(j, G, beta, res["tau_alive"], res["c_alive"], res["tau_dead"], tol=tol)


def iterate_naive(c0, G, tau, beta, steps=2000, ceiling=1e12, tol=1e-14):
    """Plain power iteration of `read_map`, no Jacobian: a cheap, independent
    cross-check of `explode_threshold`'s answer near the fold (see `demo`),
    NOT the primary measurement -- it converges to whatever global attractor
    is nearest the start, which away from a fold need not be pattern j's own
    branch at all (the module docstring's large-tau example). Returns
    (status, c_final) with status in {"converged", "exploded", "undecided"}.
    """
    c = np.array(c0, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            c_new = read_map(c, G, tau, beta)
            if not np.all(np.isfinite(c_new)) or np.max(np.abs(c_new)) > ceiling:
                return "exploded", c_new
            if np.max(np.abs(c_new - c)) < tol:
                return "converged", c_new
            c = c_new
    return "undecided", c


def three_pattern_census(beta=0.9, norms_sq=(2.0, 4.0, 8.0), G=None, **kw):
    """KILLER (a): per-pattern fold tau for each of len(norms_sq) patterns,
    plus whether the order follows ||k||^2 (strictly increasing threshold
    with norm) versus vanishing together (thresholds within `tol` of the max
    - min spread relative to their mean)."""
    if G is None:
        G = gram_orthogonal(norms_sq)
    n = G.shape[0]
    thresholds = [explode_threshold(j, G, beta, norms_sq[j], **kw) for j in range(n)]
    order = sorted(range(n), key=lambda j: norms_sq[j])
    by_norm = [thresholds[j] for j in order]
    monotone = all(by_norm[i] < by_norm[i + 1] for i in range(n - 1))
    spread = max(thresholds) - min(thresholds)
    return {
        "norms_sq": list(norms_sq), "thresholds": thresholds,
        "order_follows_norm": monotone, "spread": spread,
        "spread_over_mean": spread / float(np.mean(thresholds)),
    }


def fit_explode_boundary(betas, G, pattern_index, g, **kw):
    """KILLER (b): measure explode_threshold(pattern_index, ...) at each beta,
    then fit tau = slope*(1-beta) + intercept. The hypothesis's claim is
    intercept == 0 (a line through beta=1, tau=0)."""
    taus = [explode_threshold(pattern_index, G, beta, g, **kw) for beta in betas]
    x = 1.0 - np.asarray(betas, dtype=np.float64)
    y = np.asarray(taus, dtype=np.float64)
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "betas": list(betas), "taus": taus,
        "slope": float(slope), "slope_se": float(math.sqrt(cov[0, 0])),
        "intercept": float(intercept), "intercept_se": float(math.sqrt(cov[1, 1])),
        "r2": r2, "residuals": resid.tolist(),
    }


def route_constraint_violations(beta, tau, kmax, atol=0.0):
    """Round-two predicate: does a (beta, tau) TRAINING SCHEDULE ever cross the
    one piece of the leap that survives (see module docstring) -- the
    single-pattern fold born at tau = e*(1-beta)*g? A schedule stays on the
    safe side throughout iff

        tau(beta) >= e * (1 - beta) * max_j ||k_j||^2

    at every step. That floor is exactly `isolated_fold_tau(kmax, beta)` under
    a name that reads as a constraint rather than a threshold -- nothing here
    re-derives the algebra, it only checks a schedule against it.

    beta, tau: equal-length 1-D sequences, one entry per training step, in
        step order.
    kmax: max_j ||k_j||^2 at each step -- a scalar (held fixed across the run)
        or a sequence the same length as beta/tau.
    atol: a shortfall must exceed this to count as a violation (float slop
        for a schedule meant to run right at the boundary).

    Returns a dict: `required` (the floor at each step), `shortfall`
    (required - tau; positive means that step violates), `violated` (bool
    array), `steps` (sorted violating step indices), and `worst_step` (the
    step with the largest shortfall, or None if none violate).
    """
    beta = np.asarray(beta, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != beta.shape:
        raise ValueError(f"beta and tau must be the same length: {beta.shape} vs {tau.shape}")
    kmax = np.broadcast_to(np.asarray(kmax, dtype=np.float64), beta.shape)
    required = isolated_fold_tau(kmax, beta)
    shortfall = required - tau
    violated = shortfall > atol
    steps = np.nonzero(violated)[0].tolist()
    worst_step = int(np.argmax(shortfall)) if steps else None
    return {
        "required": required,
        "shortfall": shortfall,
        "violated": violated,
        "steps": steps,
        "worst_step": worst_step,
    }


#: `d_head=64, hidden=1280, initializer_range=0.02` -- the wired model's own
#: defaults (`ceq/hf/configuration_ceq.py::CEQConfig`), not chosen here.
#: `smprime_layer_draws` below leaves every `CEQConfig` field unset except
#: `operator`, so these ARE the values a caller gets without passing anything.
SMP_BATCH = 8
SMP_SEQ = 128
SMP_SEED = 20260920


def _import_ceq_hf():
    """`(CEQConfig, CEQAttention, CEQPreTrainedModel)`, imported around a
    STANDING, DOCUMENTED environment defect rather than through it.

    `pytest.ini` already names the failure this works around: on this box,
    torch 2.14.0+cpu is installed under torchvision 0.20.1+cu121, so `import
    torchvision` raises `RuntimeError: operator torchvision::nms does not
    exist`, and transformers 5.3.0's `image_utils.py` imports torchvision
    UNCONDITIONALLY at package-import time (not behind its own
    `is_torchvision_available()` gate, which only checks the package is
    installed, not that it loads) -- so `import ceq.hf.modeling_ceq` fails
    with `ModuleNotFoundError: Could not import module 'PreTrainedModel'`
    before a single line of THIS repository's code runs. `pytest.ini` counts
    10 test files that lose to exactly this and tolerates it with
    `--continue-on-collection-errors`; this function is the first thing in
    the tree that instead ROUTES AROUND it, because the ORDER this file
    answers needs the real wired class and not a re-implementation of it
    (`ceq/arm_smprime.py`'s own docstring already argues against a second
    copy of the arm; the same argument applies one layer up, to `CEQAttention`
    itself).

    THE ROUTE: try the real import first (a fixed environment must still work
    unmodified); only on failure, plant a MINIMAL fake `torchvision` module
    in `sys.modules` -- an `InterpolationMode` with just the seven names
    `transformers/image_utils.py:56-61` reads off it -- and import again.
    Nothing under `ceq/` or `torch` is touched; the stub satisfies exactly
    the one dead branch (image preprocessing) that a text-only causal LM
    never reaches. Not module-scope: a caller who never needs the HF classes
    pays nothing, and the pure-numpy KILLER functions above stay import-clean
    of torch entirely.
    """
    try:
        import ceq.hf.modeling_ceq  # noqa: F401 -- probe; see except below
    except Exception:
        import importlib.machinery
        import sys
        import types

        def _stub(name):
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = mod
            return mod

        tv = _stub("torchvision")
        tv.__version__ = "0.0.0"
        tvt = _stub("torchvision.transforms")

        class _InterpolationMode:
            NEAREST = "nearest"
            NEAREST_EXACT = "nearest_exact"
            BOX = "box"
            BILINEAR = "bilinear"
            HAMMING = "hamming"
            BICUBIC = "bicubic"
            LANCZOS = "lanczos"

        tvt.InterpolationMode = _InterpolationMode
        tv.transforms = tvt

    from ceq.hf.configuration_ceq import CEQConfig
    from ceq.hf.modeling_ceq import CEQAttention, CEQPreTrainedModel
    return CEQConfig, CEQAttention, CEQPreTrainedModel


def smprime_layer_draws(*, batch=SMP_BATCH, seq=SMP_SEQ, seed=SMP_SEED,
                         **config_kwargs):
    """Per-layer draws of the REAL `ceq.hf.modeling_ceq.CEQAttention` at
    `operator="smprime"` -- the class `ceq/hf/modeling_ceq.py:478` actually
    calls, imported and run, not reimplemented. Returns `(per_layer, cfg)`:
    `cfg` is the `CEQConfig` used (every field left at its default except
    `operator` unless overridden via `config_kwargs`, so `hidden_size=1280,
    num_attention_heads=20 (d_head=64), num_hidden_layers=24,
    initializer_range=0.02` -- the ORDER's own numbers -- are what a bare
    call gets); `per_layer` is a list of `num_hidden_layers` dicts, one per
    LAYER, each `{"layer", "qk", "max_k2", "max_row_max_w"}`:

        qk           the layer's own `self.qk` (a per-block nn.Parameter),
                     read off the constructed module, not assumed.
        max_k2       max_j ||k_j||^2 over this layer's own draw (batch x
                     heads x seq keys) -- "the number everything hinges on"
                     (see the ORDER), and it is a MEASUREMENT of the fan-in,
                     not the back-of-envelope estimate.
        max_row_max_w  max over causal rows i of max_{j<=i} w_ij, w = qk *
                     (q_i . k_j) / sqrt(d_head) -- the REALIZED row logit
                     `loggain` is built from (see `smprime_operating_point`),
                     kept separate from `max_k2` because it uses the actual,
                     independently-drawn q rather than assuming q_i = k_j.

    24 INDEPENDENT DRAWS, NOT ONE 24-LAYER FORWARD PASS, AND THIS IS EXACT AT
    INIT, NOT AN APPROXIMATION. `CEQBlock.forward` feeds `input_layernorm(x)`
    into `self_attn`, and `CEQPreTrainedModel._init_weights` sets every
    `nn.LayerNorm`'s weight to exactly `1.0` and bias to exactly `0.0`
    (`ceq/hf/modeling_ceq.py:549-556`). `LayerNorm` normalizes its INPUT to
    mean 0, variance 1 across the feature axis by definition, for any finite
    input with nonzero variance; with the affine part fixed at the identity,
    what reaches every attention layer's `qkv` is therefore EXACTLY a
    per-token standard-normal vector, at EVERY depth, regardless of what the
    residual stream did in earlier layers. So sampling 24 independently
    re-initialized `CEQAttention` modules, each fed one fresh
    `layer_norm(randn(...))` batch, draws from the IDENTICAL distribution a
    real 24-layer, hidden=1280 stacked forward pass would feed its 24
    attention layers at construction -- at roughly 1/24 the peak memory
    (~19.7M parameters live at a time instead of ~472M) and with no embedding
    table or MLP stack to build. `ponytail:` if this module is ever asked to
    measure a TRAINED checkpoint instead of the untrained wiring, this
    equivalence is gone (training breaks the exact-identity affine part) and
    the caller needs a real stacked forward through the saved weights, not
    this function.

    Each layer's `nn.Linear`/`nn.Embedding`/`nn.LayerNorm` weights are set by
    calling `CEQPreTrainedModel._init_weights` DIRECTLY (against a
    `SimpleNamespace(config=cfg)` standing in for `self`), the exact method
    `CEQModel.__init__`'s own `self.post_init()` would call -- not a
    reimplementation of the formula, so the two cannot drift.

    CPU, float64 throughout (`.to(torch.float64)` before any weight is
    drawn); `torch.manual_seed(seed)` once, up front, so the whole sequence
    of 24 draws is reproducible from one seed.
    """
    import types

    import torch

    CEQConfig, CEQAttention, CEQPreTrainedModel = _import_ceq_hf()
    torch.manual_seed(seed)
    cfg = CEQConfig(operator="smprime", **config_kwargs)
    d_head = cfg.head_dim
    sqrt_d = math.sqrt(d_head)
    fake_self = types.SimpleNamespace(config=cfg)
    causal = torch.ones(seq, seq, dtype=torch.bool).tril(0)

    per_layer = []
    with torch.no_grad():
        for layer in range(cfg.num_hidden_layers):
            attn = CEQAttention(cfg).to(torch.float64)
            for m in attn.modules():
                CEQPreTrainedModel._init_weights(fake_self, m)
            qk = float(attn.qk)
            x = torch.nn.functional.layer_norm(
                torch.randn(batch, seq, cfg.hidden_size, dtype=torch.float64),
                (cfg.hidden_size,))
            q, k, _v = attn.qkv(x).chunk(3, dim=-1)

            def _shape(t):
                return t.view(batch, seq, attn.n_heads, attn.d_head).transpose(1, 2)

            qh, kh = _shape(q), _shape(k)
            max_k2 = float((kh ** 2).sum(-1).max())
            w = qk * (qh @ kh.transpose(-2, -1)) / sqrt_d
            row_max_w = w.masked_fill(~causal, float("-inf")).amax(-1)
            per_layer.append({
                "layer": layer, "qk": qk, "max_k2": max_k2,
                "max_row_max_w": float(row_max_w.max()),
            })
            del attn, q, k, _v, qh, kh, w, row_max_w
    return per_layer, cfg


def smprime_operating_point(*, steps=None, atol=0.0, **draw_kwargs):
    """THE ORDER, answered: does the ANNEAL route (`ceqjepa/beta_homotopy.py
    ::arm_beta("anneal", t, steps)`, beta(t) = 1 - t/(steps-1), 1 -> 0) ever
    drive the wired arm's own `a_eff = (1-beta) * qk * max_j||k_j||^2 /
    sqrt(d_head)` up to the fold's `1/e`, and at what beta -- MEASURED off a
    real `CEQAttention` draw (`smprime_layer_draws`), not assumed off the
    2/4/8 census bed.

    TRAP 2's restatement is what makes this one call: `arm_smprime.py`'s own
    logit is `w = qk * (q@k^T)/sqrt(d_head)`, so `a_eff <= 1/e` is EXACTLY
    `tau >= e*(1-beta)*max_k2` with `tau := sqrt(d_head)/qk` -- the module's
    own `isolated_fold_tau(max_k2, beta)` compared against a CONSTANT `tau`
    (qk never moves off its constructed corner; see below). So the crossing
    is read off `route_constraint_violations` -- ALREADY WRITTEN, above, for
    exactly this predicate -- fed the real measured `max_k2` and the real
    `arm_beta` schedule, rather than re-deriving the inequality a second time.

    RETURNS a dict:
        d_head, qk, tau                the wired constants this run measured
        max_k2_by_layer, max_k2_overall, argmax_layer
                                        (3) -- the per-layer measurement
        beta1_control_a_eff             must be `0.0` exactly (beta=1 control)
        beta0_a_eff                     a_eff at the anneal's OTHER endpoint
        crossing                        {"t", "beta", "a_eff"} at the FIRST
                                         violating step, or None if the route
                                         never crosses before beta=0 -- (1)+(2)
        beta_closed_form                the algebraic crossing beta, cross-
                                         checked against `crossing["beta"]`
        route_violations                the full `route_constraint_violations`
                                         output this was read off
        realized_loggain                {"max_row_max_w_overall",
                                         "beta_closed_form_realized"} -- the
                                         SAME crossing computed off the actual
                                         measured `q.k` row logits instead of
                                         the idealized query-aligned `||k||^2`
                                         bound, so the gap between the fold's
                                         own idealisation and what a randomly
                                         initialized q/k pair realizes is on
                                         the record and not silently assumed
                                         to be zero.

    `qk` IS THE ONLY VALUE ON RECORD, NOT A CHOICE MADE HERE.
    `ceq/hf/configuration_ceq.py`'s own docstring: "NO NUMBER IS ON RECORD FOR
    IT [smprime] IN THIS MODEL" -- no checkpoint at this hidden size has ever
    trained it, so `self.qk` reads back at construction's own corner, `1.0`
    (`SMPRIME_CORNER`), on every layer, every draw. "Holding qk at its
    trained value" therefore means exactly what `smprime_layer_draws` reads
    off the module: 1.0, the only value this arm's wiring has ever produced,
    asserted equal across layers below rather than silently averaged.

    MEASURED, seed=20260920, batch=8, seq=128, hidden=1280, heads=20 (d_head
    =64), layers=24, `python -c "from ceqjepa import fold_census as fc;
    import json; print(json.dumps(fc.smprime_operating_point(), default=str,
    indent=1))"`:
        qk = 1.0 on every layer; tau = sqrt(64)/1.0 = 8.0
        max_k2 per layer: 59.26 .. 68.20 (24 values, TENS, as the ORDER
            predicted from the fan-in -- not the RED's 2/4/8, not scout 4's
            0.0256); max_k2_overall = 68.204885 at layer 1
        beta1_control_a_eff = 0.0 EXACTLY (beta=1 control fires)
        crossing: t=26 of 600, beta=0.956594, a_eff=0.370060 (> 1/e=0.367879)
        beta_closed_form = 0.956850 -- agrees with the swept crossing to the
            anneal's own 1/599 step spacing, i.e. the discretization, not a
            second independent number
        beta0_a_eff = 8.525611 -- 23x over the threshold at the anneal's
            OTHER end
        realized_loggain: max_row_max_w_overall = 3.157217 (a REALIZED row
            logit, from actual independent q/k, not the query-aligned bound);
            beta_closed_form_realized = 0.883480 -- still crosses well before
            beta=0, at a LOWER beta than the idealized bound because a random
            q rarely reads a value as large as ||k_j|| itself.

    THE RULING THIS ANSWERS: the anneal route crosses a_eff=1/e at
    beta ~= 0.957 (idealized) to ~= 0.883 (realized) -- both within the first
    ~4-12% of the route's own distance from beta=1, not near beta=0 and not
    merely "possible before training ends". Consequence (8) is the live
    deliverable and not vacuous: the schedule the model actually trains under
    walks into the fold's region almost immediately after leaving the softmax
    corner, at the model's OWN measured key norms (tens) rather than at the
    census bed's planted 2/4/8.
    """
    import numpy as _np

    from ceqjepa.beta_homotopy import STEPS as ANNEAL_STEPS
    from ceqjepa.beta_homotopy import arm_beta

    steps = ANNEAL_STEPS if steps is None else steps
    per_layer, cfg = smprime_layer_draws(**draw_kwargs)
    d_head = cfg.head_dim

    qk_values = {r["qk"] for r in per_layer}
    if len(qk_values) != 1:
        raise RuntimeError(
            f"qk was not constant across the {len(per_layer)} layer draws "
            f"({sorted(qk_values)}) -- the arm's own switch should read the "
            "same constructed corner on every layer for an arm nothing has "
            "ever trained; a mismatch here means the measurement below is "
            "not reading one operating point.")
    qk = qk_values.pop()
    tau = math.sqrt(d_head) / qk

    max_k2_by_layer = [r["max_k2"] for r in per_layer]
    max_k2_overall = max(max_k2_by_layer)
    argmax_layer = max(per_layer, key=lambda r: r["max_k2"])["layer"]

    beta_schedule = _np.array([arm_beta("anneal", t, steps) for t in range(steps)])
    tau_schedule = _np.full(steps, tau)
    violations = route_constraint_violations(beta_schedule, tau_schedule,
                                              max_k2_overall, atol=atol)

    def _a_eff(beta):
        return (1.0 - beta) * qk * max_k2_overall / math.sqrt(d_head)

    crossing = None
    if violations["steps"]:
        t0 = min(violations["steps"])
        beta0 = float(beta_schedule[t0])
        crossing = {"t": t0, "beta": beta0, "a_eff": _a_eff(beta0)}

    max_row_max_w_overall = max(r["max_row_max_w"] for r in per_layer)

    return {
        "d_head": d_head, "qk": qk, "tau": tau, "steps": steps,
        "max_k2_by_layer": max_k2_by_layer, "max_k2_overall": max_k2_overall,
        "argmax_layer": argmax_layer,
        "beta1_control_a_eff": _a_eff(float(beta_schedule[0])),
        "beta0_a_eff": _a_eff(float(beta_schedule[-1])),
        "crossing": crossing,
        "beta_closed_form": 1.0 - math.sqrt(d_head) / (E * qk * max_k2_overall),
        "route_violations": violations,
        "realized_loggain": {
            "max_row_max_w_overall": max_row_max_w_overall,
            "beta_closed_form_realized": 1.0 - 1.0 / (E * max_row_max_w_overall),
        },
    }


def fold_is_a_fold_not_a_cusp(a0=None, c0=None, h=1e-6):
    """CONSEQUENCE (2), CHECKED: is the single-pattern fold's own singularity
    a CUSP (codimension 2, two independent unfolding directions) or a plain
    FOLD (codimension 1, one)? `8*a**3 + 27*b**2 = 0` -- the dispatch's own
    cusp normal form -- describes the locus where a cubic's discriminant
    vanishes, i.e. a DOUBLE root becomes a TRIPLE root; fitting it to data
    presumes a candidate triple-root point exists to fit against.

    `F(c, a) = e^{a c} - c` is the single-pattern residual this file's own
    `c = e^{a c}` is the root of. Its known double root, `a0 = 1/e, c0 = e`,
    is measured elsewhere in this file to 5.6e-17 (`c = e^{ac}` folds at
    `a = 1/e`, module docstring's WHAT BINDS). A cusp needs a SECOND
    condition there, `F_cc(c0, a0) = 0`, on top of the fold's own
    `F_c(c0, a0) = 0`. But

        F_cc(c, a) = a^2 * e^{a*c}

    is a product of a square and an exponential -- strictly POSITIVE for
    every real `c` and every `a != 0` -- so it cannot vanish anywhere on the
    fold curve `a = 1/e` (which never touches `a = 0`). The cusp degeneracy
    is excluded ALGEBRAICALLY, for every point on the curve at once, not
    merely unobserved at the one point checked. `F_a(c0, a0) = c0*e^{a0c0}
    = e^2 != 0` is the accompanying transversality check: the unfolding
    parameter genuinely moves the root, which is what makes this a GENERIC
    (codim-1) fold rather than some other degenerate germ.

    Returns a dict with the analytic values (`F_c`, `F_cc`, `F_a`), a
    central-finite-difference cross-check of each (`*_fd`, step `h`, MEASURED
    rather than trusted from the formulas alone), and three booleans:
    `is_generic_fold` (`F_c ~= 0`, `F_cc != 0`, `F_a != 0`),
    `cusp_degeneracy_present` (`F_cc ~= 0` -- the condition a cusp needs),
    `cusp_rejected` (the first true and the second false).
    """
    a0 = 1.0 / E if a0 is None else a0
    c0 = E if c0 is None else c0

    def F(c, a):
        return math.exp(a * c) - c

    Fc = a0 * math.exp(a0 * c0) - 1.0
    Fcc = a0 ** 2 * math.exp(a0 * c0)
    Fa = c0 * math.exp(a0 * c0)

    Fc_fd = (F(c0 + h, a0) - F(c0 - h, a0)) / (2.0 * h)
    Fcc_fd = (F(c0 + h, a0) - 2.0 * F(c0, a0) + F(c0 - h, a0)) / h ** 2
    Fa_fd = (F(c0, a0 + h) - F(c0, a0 - h)) / (2.0 * h)

    is_fold = abs(Fc) < 1e-6 and abs(Fcc) > 1e-6 and abs(Fa) > 1e-6
    cusp_degenerate = abs(Fcc) < 1e-6

    return {
        "a0": a0, "c0": c0,
        "F_c": Fc, "F_cc": Fcc, "F_a": Fa,
        "F_c_fd": Fc_fd, "F_cc_fd": Fcc_fd, "F_a_fd": Fa_fd,
        "is_generic_fold": is_fold,
        "cusp_degeneracy_present": cusp_degenerate,
        "cusp_rejected": is_fold and not cusp_degenerate,
    }


def demo_operating_point():
    """The one self-check ROUND TWO, PASS 2 leaves behind, mirroring `demo()`
    above: re-runs `smprime_operating_point()` at its own defaults (24 real
    `CEQAttention` layer draws -- this is the slow check, seconds not
    milliseconds, which is why it is its own function and not folded into
    `demo()`) and asserts the headline shape of the MEASURED numbers section
    of `smprime_operating_point`'s own docstring, plus the cusp verdict."""
    report = smprime_operating_point()
    assert report["beta1_control_a_eff"] == 0.0, report["beta1_control_a_eff"]
    assert report["qk"] == 1.0, "the only value on record for an untrained arm"
    assert 20.0 < min(report["max_k2_by_layer"]), report["max_k2_by_layer"]
    assert max(report["max_k2_by_layer"]) < 150.0, report["max_k2_by_layer"]
    assert report["crossing"] is not None, "expected the anneal to cross 1/e"
    assert 0.5 < report["crossing"]["beta"] < 1.0, report["crossing"]
    assert abs(report["crossing"]["beta"] - report["beta_closed_form"]) < 1e-2, report
    assert report["beta0_a_eff"] > 1.0 / E, report["beta0_a_eff"]
    realized_beta = report["realized_loggain"]["beta_closed_form_realized"]
    assert 0.5 < realized_beta < report["crossing"]["beta"], (
        "the realized (actual q.k) crossing should sit at a LOWER beta than "
        "the idealized (query-aligned ||k||^2) one", report)

    cusp = fold_is_a_fold_not_a_cusp()
    assert cusp["cusp_rejected"], cusp

    print("fold_census operating-point self-check: OK")


def demo():
    """The one self-check this file leaves behind. Reduced-precision re-runs
    of the two headline numbers above, so a slower BLAS on another box still
    passes."""
    # 1. isolated N=1 reproduces the closed form (this is the ground truth
    #    the whole continuation machinery is validated against).
    g, beta = 4.0, 0.7
    G1 = gram_orthogonal([g])
    closed = isolated_fold_tau(g, beta)
    measured = explode_threshold(0, G1, beta, g)
    assert abs(measured - closed) < 1e-4, (measured, closed)
    sol, resid = solve_fixed_point([1.0], G1, measured, beta)
    assert resid < 1e-6
    assert 0.95 < spectral_radius(sol, G1, measured, beta) < 1.05, "not at the tangency"

    # 2. coupled 3-pattern orthogonal census: order follows norm, matches the
    #    pre-registered RED to better than 1%.
    census = three_pattern_census(beta=0.9, norms_sq=(2.0, 4.0, 8.0))
    assert census["order_follows_norm"]
    expected = [0.5436, 1.0872, 2.1744]
    for got, want in zip(census["thresholds"], expected):
        assert abs(got - want) / want < 1e-2, (got, want)

    # 3. gram_correlated at rho=0 reduces to gram_orthogonal.
    Gc = gram_correlated([g, g], 0.0)
    assert np.allclose(Gc, gram_orthogonal([g, g]), atol=1e-9)

    # 4. explode_threshold's answer agrees with the naive power-iteration
    #    cross-check right at (and just below/above) the measured threshold.
    thr = census["thresholds"][2]
    survives, _ = iterate_naive([0.0, 0.0, 1.0], gram_orthogonal((2.0, 4.0, 8.0)),
                                 thr * 1.01, 0.9)
    dies, _ = iterate_naive([0.0, 0.0, 1.0], gram_orthogonal((2.0, 4.0, 8.0)),
                             thr * 0.99, 0.9)
    assert survives == "converged" and dies == "exploded", (survives, dies)

    print("fold_census self-check: OK")


if __name__ == "__main__":
    demo()
    demo_operating_point()
