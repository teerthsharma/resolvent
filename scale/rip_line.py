"""The compressed-sensing floor under IMPACT's attribution column.

WHAT THIS IS. `scale/impact.py` plants a signed matrix B and asks an arm to recover
one of its rows from intervention draws: `impact_attribution` regresses the query
row `B[query, :]` (length `N`) against `n_samples` news vectors, exactly the
noiseless linear system `y = A b` with measurement matrix `A` and unknown signal
`b` [READ scale/impact.py:1013-1021]. That is compressed sensing, not linear
regression with a generous sample size, because `b` is sparse: each row of the
planted `B` is built by choosing `SUPPLIERS_PER_NODE` negative entries and
`COMPETITORS_PER_NODE` positive entries and leaving every other entry at zero
[READ scale/impact.py:88-89, 405-410], so `B[query, :]` has exactly
`SUPPLIERS_PER_NODE + COMPETITORS_PER_NODE = 4` nonzeros out of `N` columns.
Columns of the planted `B` carry no such guarantee -- a column's nonzero count is
the number of rows that happened to pick it as a supplier or competitor, which is
un-bounded and instance-dependent [READ scale/impact.py:396-410]; the fixed
sparsity lives in the row, not the column, and every claim below is stated for the
row that `impact_attribution` actually recovers.

THE LINE. The restricted isometry property gives the sample-complexity regime in
which recovery of an s-sparse vector in R^n from m noiseless linear measurements is
possible at all: `m >= C * s * ln(n / s)` for a dimensionless constant `C` set by
the measurement ensemble and solver, not by the corpus. Below that count, an arm's
failure to recover the planted row carries no information about the arm -- there is
provably no procedure, oracle or otherwise, that can separate a real B from an
adversarial one consistent with the same undersampled measurements. `rip_line`
below is that bound as a function of `(n, s, c)`, `c` supplied by the caller. It
imports nothing from `scale/impact.py`: the facts impact.py currently supplies are
`N = node_count >= IMPACT_MIN_NODES = 1024` [READ scale/impact.py:73,
633-635] and row-sparsity `4` [READ scale/impact.py:88-89], quoted here as prose,
not as a binding that would break under a concurrent edit to that file.

DOMAIN NOTE. `s * ln(n/s)` is increasing in `s` only while `s < n/e`; past that
point `ln(n/s)` shrinks faster than `s` grows and the closed form turns over. That
peak sits at 37% of `n` and is nowhere near a sparse regime -- compressed sensing's
whole premise is `s << n` -- so every call this module or `impact.py` makes sits
deep in the increasing branch, but `rip_line` does not clamp the input: passing a
dense `s` produces a mathematically correct, unhelpful answer, on purpose.

THE MUST-FIRE. `C` is not asserted, it is measured: `oracle_recovery_sweep` draws a
random Gaussian measurement matrix, a random s-sparse signed vector (support
uniform, sign Rademacher, magnitude uniform on [0.5, 1.5] -- never hand-built), and
recovers it by L1 basis pursuit (`scipy.optimize.linprog`, the standard
`min sum(u+v) s.t. A(u-v)=y, u,v>=0` LP form) and, independently, by orthogonal
matching pursuit (`sklearn.linear_model.OrthogonalMatchingPursuit`,
`n_nonzero_coefs=s`). A draw counts as recovered at relative L2 error under `1e-6`
for basis pursuit and under the looser `1e-3` for OMP. The sweep is run at FOUR
`(n, s)` settings. The first two share `n/s = 16` and check that `C` does not move
when `s` doubles at fixed ratio; the last two vary the RATIO, to `n/s = 64` and
`n/s = 4`, because two beds at one ratio cannot test the `log(n/s)` factor at all --
`ln(n/s)` is the same number in both, so `m_50` doubling with `s` forces `C` to come
back equal whatever the truth is. A constant that is stable only across beds sharing
its own denominator has not been shown to be stable.

  n=64,  s=4,  seed=0, draws=50, ln(n/s)=2.772589
    m:        4     8    10    12    14    16    18    20    24    32    40
    BP  :  0.00  0.00  0.08  0.16  0.44  0.60  0.82  0.92  1.00  1.00  1.00
    OMP :  0.00  0.00  0.02  0.04  0.12  0.18  0.42  0.48  0.76  0.96  1.00
    m_50 (BP) = 16 -> C = 16 / (4*2.772589)  = 1.44270
    m_50 (OMP)= 24 -> C = 24 / (4*2.772589)  = 2.16404

  n=128, s=8,  seed=1, draws=50, ln(n/s)=2.772589
    m:        8    16    20    24    28    32    36    40    48    64    80
    BP  :  0.00  0.00  0.02  0.08  0.28  0.80  0.92  0.96  1.00  1.00  1.00
    OMP :  0.00  0.00  0.00  0.00  0.16  0.20  0.34  0.42  0.76  0.98  0.96
    m_50 (BP) = 32 -> C = 32 / (8*2.772589)  = 1.44270
    m_50 (OMP)= 48 -> C = 48 / (8*2.772589)  = 2.16404

  n=256, s=4,  seed=2, draws=40, ln(n/s)=4.158883
    m:        8    12    16    20    24    28    32    40    56
    BP  :  0.00  0.00  0.07  0.40  0.80  1.00  1.00  1.00  1.00
    OMP :  0.00  0.00  0.00  0.12  0.38  0.57  0.80  0.85  1.00
    m_50 (BP) = 24 -> C = 24 / (4*4.158883)  = 1.44270
    m_50 (OMP)= 28 -> C = 28 / (4*4.158883)  = 1.68314

  n=64,  s=16, seed=3, draws=40, ln(n/s)=1.386294
    m:       16    24    32    40    44    48    52    56    64
    BP  :  0.00  0.00  0.07  0.85  0.93  0.95  1.00  1.00  1.00
    OMP :  0.00  0.00  0.00  0.03  0.10  0.20  0.17  0.35  0.62
    m_50 (BP) = 40 -> C = 40 / (16*1.386294) = 1.80337
    m_50 (OMP)= 64 -> C = 64 / (16*1.386294) = 2.88539

Every grid fails at the bottom (0.00 at its smallest `m`, all four beds, both
solvers) and reaches at least 0.96 at its top for basis pursuit, so no arm of the
sweep is degenerate.

`C` IS NOT BED-INVARIANT, AND THE TWO EQUAL VALUES IN THE FIRST TWO BEDS DO NOT
SHOW THAT IT IS. Basis pursuit fits `C = 1.44270` at `n/s = 16` (twice) and at
`n/s = 64`, but `C = 1.80337` at `n/s = 4` -- 25% higher. `s log(n/s)` is the
asymptotic scaling and its constant creeps as `n/s` falls toward 1, where the
`log(n/s)` factor stops describing the phase transition. The first two beds could
not have detected that: at fixed `n/s` the ratio `m_50 / (s ln(n/s))` is forced to
repeat as soon as `m_50` tracks `s`. The consequence for a caller is operational --
`C` must be re-swept per bed rather than the `1.44270` reused, and the grid
resolution (nearest points 32 at 0.07 and 40 at 0.85) bounds `m_50` for the
`n/s = 4` bed only to `(32, 40]`. The two solvers likewise disagree by a factor of
1.2 to 1.6 depending on the bed, which is the two-solvers-fail-differently evidence
the corpus asks for and a second reason `C` is a per-bed measurement.

`impact_attribution` defaults to `n_samples=256` draws against `N=1024`, `s=4`
[READ scale/impact.py:973]: `rip_line(1024, 4, 1.44270) = 1.44270 * 4 * ln(256) =
32.01`, so the shipped default oversamples the RIP floor (by the more permissive,
basis-pursuit-fitted constant) by roughly 8x and the corpus is comfortably
admissible today -- the point of this module is the floor itself, not a claim that
today's default sits on it.

Evidence class tagging follows the house convention: RUN (executed here), READ
(file:line), DERIVED (closed form). No `nn.Parameter`, no `torch` import: this is a
numpy/scipy/sklearn compressed-sensing check, not a model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import OrthogonalMatchingPursuit

__all__ = [
    "rip_line",
    "admissible",
    "verdict",
    "oracle_recovery_sweep",
    "SETTINGS",
    "run_all_sweeps",
    "report",
    "demo",
]


def rip_line(n: int, s: int, c: float) -> float:
    """The RIP sample-complexity floor `c * s * ln(n / s)`, natural log.

    `m >= rip_line(n, s, c)` is the regime in which recovering an s-sparse vector
    in R^n from m noiseless linear measurements is possible at all, for a
    dimensionless constant `c` fit empirically by `oracle_recovery_sweep` (never
    hard-coded here). Requires `0 < s < n`; see the module docstring's DOMAIN NOTE
    for why the result is only monotone increasing in `s` while `s < n/e`.
    """
    if not (0 < s < n):
        raise ValueError(f"rip_line requires 0 < s < n, got n={n}, s={s}")
    return c * s * math.log(n / s)


def admissible(n: int, s: int, m: int, c: float) -> bool:
    """True when `m` intervention draws clear the RIP floor for (n, s, c)."""
    return m >= rip_line(n, s, c)


def verdict(n: int, s: int, m: int, c: float) -> str:
    """`"UNDER-SAMPLED"` strictly below the RIP line, `"ADMISSIBLE"` at or above it.

    Exact strings: these are branch labels consumed elsewhere, not free text.
    """
    return "ADMISSIBLE" if admissible(n, s, m, c) else "UNDER-SAMPLED"


def _draw_sparse_signal(rng: np.random.Generator, n: int, s: int) -> np.ndarray:
    """One s-sparse signed vector in R^n: uniform support, Rademacher sign,
    magnitude uniform on [0.5, 1.5]. Never a hand-built vector."""
    support = rng.choice(n, size=s, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=s)
    mags = rng.uniform(0.5, 1.5, size=s)
    b = np.zeros(n)
    b[support] = signs * mags
    return b


def _basis_pursuit(A: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """L1 recovery via the standard LP form: min sum(u+v) s.t. A(u-v)=y, u,v>=0."""
    m, n = A.shape
    c = np.ones(2 * n)
    A_eq = np.hstack([A, -A])
    bounds = [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[:n] - res.x[n:]


def _omp_recover(A: np.ndarray, y: np.ndarray, s: int) -> np.ndarray | None:
    """OMP recovery with the true sparsity as the target support size."""
    m, n = A.shape
    k = min(s, m)
    if k <= 0:
        return None
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    omp.fit(A, y)
    return omp.coef_


def oracle_recovery_sweep(n: int, s: int, m_grid, draws: int, seed: int) -> dict:
    """Empirical recovery-fraction sweep over `m_grid`, both solvers, one signal
    and measurement draw shared between them at each trial.

    Returns per-solver recovery fractions keyed by m, the fitted `C` from each
    solver's `m_50` (smallest grid `m` with recovery fraction >= 0.5), and the
    grid/draws/seed that produced them, so the result is fully reproducible.
    """
    rng = np.random.default_rng(seed)
    m_sorted = sorted(m_grid)
    bp_frac: dict[int, float] = {}
    omp_frac: dict[int, float] = {}
    for m in m_sorted:
        bp_hits = 0
        omp_hits = 0
        for _ in range(draws):
            A = rng.standard_normal((m, n)) / math.sqrt(m)
            b = _draw_sparse_signal(rng, n, s)
            y = A @ b
            b_norm = np.linalg.norm(b)

            x_bp = _basis_pursuit(A, y)
            if x_bp is not None and np.linalg.norm(x_bp - b) / b_norm < 1e-6:
                bp_hits += 1

            x_omp = _omp_recover(A, y, s)
            if x_omp is not None and np.linalg.norm(x_omp - b) / b_norm < 1e-3:
                omp_hits += 1
        bp_frac[m] = bp_hits / draws
        omp_frac[m] = omp_hits / draws

    log_ratio = math.log(n / s)

    def _m50_and_c(frac: dict[int, float]):
        m50 = next((m for m in m_sorted if frac[m] >= 0.5), None)
        c = (m50 / (s * log_ratio)) if m50 is not None else float("nan")
        return m50, c

    m50_bp, c_bp = _m50_and_c(bp_frac)
    m50_omp, c_omp = _m50_and_c(omp_frac)

    return {
        "n": n, "s": s, "draws": draws, "seed": seed, "m_grid": m_sorted,
        "bp": bp_frac, "omp": omp_frac,
        "m_50_bp": m50_bp, "C_bp": c_bp,
        "m_50_omp": m50_omp, "C_omp": c_omp,
    }


#: Four (n, s) beds. The first two share n/s = 16 and vary s; the last two vary
#: the RATIO, to n/s = 64 and n/s = 4. Beds sharing one n/s cannot test the
#: log(n/s) factor -- see the module docstring's "C IS NOT BED-INVARIANT".
SETTINGS = (
    dict(n=64, s=4, m_grid=[4, 8, 10, 12, 14, 16, 18, 20, 24, 32, 40], draws=50, seed=0),
    dict(n=128, s=8, m_grid=[8, 16, 20, 24, 28, 32, 36, 40, 48, 64, 80], draws=50, seed=1),
    dict(n=256, s=4, m_grid=[8, 12, 16, 20, 24, 28, 32, 40, 56], draws=40, seed=2),
    dict(n=64, s=16, m_grid=[16, 24, 32, 40, 44, 48, 52, 56, 64], draws=40, seed=3),
)


def run_all_sweeps() -> list[dict]:
    return [oracle_recovery_sweep(**cfg) for cfg in SETTINGS]


def report(results: list[dict] | None = None) -> str:
    if results is None:
        results = run_all_sweeps()
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("rip_line -- the RIP sample-complexity floor under IMPACT's attribution row")
    w("=" * 78)
    w("")
    for r in results:
        n, s = r["n"], r["s"]
        log_ratio = math.log(n / s)
        w(f"n={n}, s={s}, seed={r['seed']}, draws={r['draws']}, ln(n/s)={log_ratio:.6f}")
        header = "  m    : " + " ".join(f"{m:5d}" for m in r["m_grid"])
        bp_row = "  BP   : " + " ".join(f"{r['bp'][m]:5.2f}" for m in r["m_grid"])
        omp_row = "  OMP  : " + " ".join(f"{r['omp'][m]:5.2f}" for m in r["m_grid"])
        w(header)
        w(bp_row)
        w(omp_row)
        w(f"  m_50 (BP)  = {r['m_50_bp']}  -> C = {r['C_bp']:.5f}")
        w(f"  m_50 (OMP) = {r['m_50_omp']} -> C = {r['C_omp']:.5f}")
        w(f"  bottom grid point m={r['m_grid'][0]}: BP={r['bp'][r['m_grid'][0]]:.2f}"
          f" OMP={r['omp'][r['m_grid'][0]]:.2f}")
        w(f"  top grid point    m={r['m_grid'][-1]}: BP={r['bp'][r['m_grid'][-1]]:.2f}"
          f" OMP={r['omp'][r['m_grid'][-1]]:.2f}")
        w("")
    return "\n".join(lines)


def demo(results: list[dict] | None = None) -> None:
    """Plain-assert self-check: no pytest, no fixtures.

    Checks the closed form's branch labels and monotonicity, then checks the
    empirical sweep actually transitioned from failure to success rather than
    being flat.
    """
    # verdict: strictly below the line is UNDER-SAMPLED, at or above is ADMISSIBLE
    n, s, c = 1024, 4, 1.4
    line = rip_line(n, s, c)
    assert verdict(n, s, line - 1.0, c) == "UNDER-SAMPLED", (n, s, line)
    assert verdict(n, s, line, c) == "ADMISSIBLE", (n, s, line)
    assert verdict(n, s, line + 1.0, c) == "ADMISSIBLE", (n, s, line)

    # rip_line monotone increasing in s at fixed n, deep in the s << n/e branch
    # (n/e ~ 377 here; 2,4,8,16 are nowhere near the turnover).
    vals = [rip_line(n, s_, c) for s_ in (2, 4, 8, 16)]
    assert vals == sorted(vals), vals
    assert len(set(vals)) == len(vals), vals

    if results is None:
        results = run_all_sweeps()
    for r in results:
        lo, hi = r["m_grid"][0], r["m_grid"][-1]
        assert r["bp"][lo] < r["bp"][hi], (r["n"], r["s"], r["bp"])
        assert r["omp"][lo] < r["omp"][hi], (r["n"], r["s"], r["omp"])

    print("rip_line demo OK")


if __name__ == "__main__":
    _results = run_all_sweeps()
    print(report(_results))
    demo(_results)
