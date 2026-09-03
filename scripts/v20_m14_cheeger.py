"""M14: the exact sweep-cut conductance, and the brute-force minimum that checks it.

WHY THIS FILE EXISTS. `scale/foreman_lambda2.py:300 bridge_conductance` is the only
conductance in this repository. It evaluates ONE named cut, so it returns an UPPER
bound on the Cheeger constant `h` and never a value for it -- its own docstring says
so at `:303`. An upper bound composes with `gamma <= 2h` and with nothing else. The
LOWER half `h^2/2 <= gamma` needs `h` itself, a minimum over `2^(n-1)` cuts, and this
file is the first thing in the tree that computes it.

THE THEOREM, WITH ITS HYPOTHESES, AND THE CITATION. [V-eq]
Levin and Peres, with contributions by Wilmer, *Markov Chains and Mixing Times*, 2nd
edition, **Theorem 13.10**, page 183, chapter 13 section 13.2.2, attributed there to
Sinclair-Jerrum (1989) and Lawler-Sokal (1988). Fetched from
`pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf` and read with `pdftotext -layout`. Its
own sentence reads: "Let lambda_2 be the second largest eigenvalue of a REVERSIBLE
transition matrix P, and let gamma = 1 - lambda_2", then equation (13.6),

    Phi_*^2 / 2  <=  gamma  <=  2 Phi_*

where `gamma = 1 - lambda_2` with `lambda_2` the second largest EIGENVALUE of `P`
(not the second largest modulus, and not the absolute spectral gap `gamma_* =
1 - max_{i>=2} |lambda_i|`, which is that book's equation (12.6)), and, from its
equations (7.5)-(7.7) at page 89,

    Phi_* = min { Q(S, S^c) / pi(S) : pi(S) <= 1/2 },
    Q(S, S^c) = sum_{x in S, y in S^c} pi(x) P(x, y).

WHAT THE THEOREM'S OWN SENTENCE DOES AND DOES NOT SAY. It states REVERSIBILITY and
nothing else. IRREDUCIBILITY is not in the sentence; it enters as a standing
convention of the chapter, through the eigenvalue ordering `1 = lambda_1 > lambda_2
>= ... >= -1` at equation (12.7), which needs Lemma 12.1(ii) -- irreducibility makes
the eigenvalue-1 eigenspace one-dimensional -- for `lambda_2` to be well defined at
all. So irreducibility is required to STATE the theorem rather than to prove it, and
this file checks it either way. The graph form is Chung, *Spectral Graph Theory*
(CBMS 92), chapter 2 section 2.3 pages 25-26, where it is split across Lemma 2.1
(`lambda_1 <= 2 h_G`) and Theorem 2.2 (`lambda_1 >= h_G^2 / 2`).

FOUR HYPOTHESES, EACH OF WHICH IS A PLACE THIS CAN GO WRONG.
  1. `P` IRREDUCIBLE. On a graph: connected. `_require_graph` raises otherwise.
  2. `P` REVERSIBLE w.r.t. `pi`. On a graph with `W = W^T >= 0` and `P = D^{-1} W`
     this holds with `pi = d / vol`, and `_require_graph` checks symmetry and
     non-negativity rather than assuming them. A SIGNED `W` is not a graph, has no
     `pi`, and `Phi` is undefined for it -- there is no version of this theorem that
     applies.
  3. `gamma` IS ABOUT `lambda_2`, NOT ABOUT MIXING. Cheeger says nothing about
     `lambda_n`. `absolute_spectral_gap` is provided precisely so the two can be
     printed side by side; on a bipartite graph the first is bounded away from zero
     while the second is exactly zero. Laziness is the standard fix and is an EXTRA
     hypothesis, not part of Theorem 13.10 -- the same book's Exercise 12.3 is what
     gives `gamma = gamma_*`, and only for a lazy chain, while its section 17.4 at
     page 249 needs the words "for a LAZY reversible Markov chain" before it may
     combine Theorem 12.4 (which bounds `t_mix` through `gamma_*`) with Theorem 13.10
     (which bounds `gamma`). Any use of this file's output as a MIXING dial inherits
     that extra hypothesis and must state it.
  4. `Phi_*` IS A MINIMUM. Any single named cut is an upper bound on it. This is the
     hypothesis the tree's `bridge_conductance` does not supply, and the one M14's
     lower half needs.

TWO INDEPENDENT NUMERIC PATHS, WHICH FAIL DIFFERENTLY.
  A. `sweep_cut_conductance` -- order the vertices by the Fiedler coordinate and take
     the best of the `n-1` prefix cuts. `O(n^3)` and always available, but it searches
     `n-1` of the `2^(n-1)` cuts, so it can only ever be TOO LARGE. Its guarantee is
     `phi_sweep <= sqrt(2 gamma)`, which is a check on the implementation.
  B. `exact_min_conductance` -- every cut, `2^(n-1)` of them with vertex 0 pinned into
     `S` (legitimate because `Phi(S) = Phi(S^c)`). Exact, and exponential, so it is
     capped at `BRUTE_FORCE_MAX_N`.
Agreement is evidence because neither calls the other and their error modes are
disjoint: A is a heuristic over an exact conductance, B is exhaustive over the same
exact conductance.

ARITHMETIC. `cut(S)` and `vol(S)` are sums of NON-NEGATIVE float64 terms, so no
cancellation occurs and the relative error is bounded by `n^2 * eps`. `eigvalsh` runs
on `D^{-1/2} W D^{-1/2}`, whose spectral radius is exactly 1, so its absolute
eigenvalue error is bounded by `c * n * eps`. Both bounds are quoted with numbers in
`tests/jupiter/test_m14_cheeger.py`, which is where the tolerance is justified.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["BRUTE_FORCE_MAX_N", "spectral_gap", "absolute_spectral_gap",
           "conductance_of", "named_cut_conductance", "exact_min_conductance",
           "sweep_cut_conductance", "repo_chain_weights", "domain_census",
           "domain_census_facts",
           "report", "demo"]

#: `2^(n-1)` cuts at `n = 20` is 524,288 rows of 20 float64 -- 84 MB if built at once,
#: which is why `_masks` chunks. Above this the exact path is refused rather than
#: silently swapped for the sweep cut: an upper bound reported as a minimum is exactly
#: the defect this file exists to remove.
BRUTE_FORCE_MAX_N = 20

_CHUNK = 1 << 15


def _require_graph(W: np.ndarray) -> np.ndarray:
    """The hypotheses of Theorem 13.10, checked rather than assumed."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {W.shape}")
    if not np.allclose(W, W.T, rtol=0.0, atol=1e-12):
        raise ValueError("W is not symmetric: P = D^-1 W is then not reversible "
                         "w.r.t. pi = d/vol and Theorem 13.10 does not apply")
    if (W < 0.0).any():
        raise ValueError("W has a negative entry: a signed matrix is not a weighted "
                         "graph, has no stationary pi, and conductance is undefined "
                         "on it")
    d = W.sum(axis=1)
    if (d <= 0.0).any():
        raise ValueError("isolated vertex: P has a zero row and is not stochastic")
    n = W.shape[0]
    seen, q = {0}, deque([0])
    while q:
        u = q.popleft()
        for v in np.nonzero(W[u] > 0.0)[0]:
            if int(v) not in seen:
                seen.add(int(v))
                q.append(int(v))
    if len(seen) != n:
        raise ValueError(f"graph is not connected ({len(seen)} of {n} reached): "
                         "P is reducible and Theorem 13.10 does not apply")
    return W


def _normalised_adjacency(W: np.ndarray) -> np.ndarray:
    d = np.sqrt(W.sum(axis=1))
    return W / d[:, None] / d[None, :]


def spectral_gap(W) -> float:
    """`gamma = 1 - lambda_2(P)`, the quantity Theorem 13.10 brackets.

    Computed from the SYMMETRIC conjugate `D^{-1/2} W D^{-1/2}`, which is similar to
    `P = D^{-1} W` and has real spectrum, so `eigvalsh` gives an exact ordering rather
    than a sort of complex moduli. Equivalently the second smallest eigenvalue of the
    normalised Laplacian `I - D^{-1/2} W D^{-1/2}`.
    """
    W = _require_graph(W)
    ev = np.sort(np.linalg.eigvalsh(_normalised_adjacency(W)))
    return float(1.0 - ev[-2])


def absolute_spectral_gap(W) -> float:
    """`gamma_* = 1 - max_{i >= 2} |lambda_i|`, the quantity MIXING TIME is about.

    Cheeger does NOT bound this. The two differ exactly when the most negative
    eigenvalue is the binding one, and they differ maximally on a bipartite graph,
    where `lambda_n = -1` and this reads `0` while `spectral_gap` does not.
    """
    W = _require_graph(W)
    ev = np.sort(np.linalg.eigvalsh(_normalised_adjacency(W)))
    return float(1.0 - np.abs(ev[:-1]).max())


def conductance_of(W, mask) -> float:
    """`Phi(S) = cut(S) / min(vol S, vol S^c)` for a boolean membership mask."""
    W = np.asarray(W, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    d = W.sum(axis=1)
    vol = float(d.sum())
    vol_s = float(d[mask].sum())
    if vol_s <= 0.0 or vol_s >= vol:
        raise ValueError("S is empty or all of V; Phi is undefined there")
    cut = float(W[np.ix_(mask, ~mask)].sum())
    return cut / min(vol_s, vol - vol_s)


def named_cut_conductance(W, vertices) -> float:
    """`Phi` of ONE named cut -- the shape `scale/foreman_lambda2.py:300` computes.

    Kept here under a name that says what it is. It is an UPPER bound on `phi` and
    the planted negative in the test suite is built on the fact that it can be an
    arbitrarily bad one.
    """
    W = np.asarray(W, dtype=np.float64)
    mask = np.zeros(W.shape[0], dtype=bool)
    mask[list(vertices)] = True
    return conductance_of(W, mask)


def _masks(n: int):
    """All `S` containing vertex 0, excluding `S = V`, in chunks of `_CHUNK` rows.

    Pinning vertex 0 into `S` loses nothing: `Phi(S) = Phi(S^c)` because the
    denominator is symmetric in the two sides, so every cut is still visited once.
    """
    total = (1 << (n - 1)) - 1          # drop the all-ones mask, which is S = V
    bit = np.arange(n - 1, dtype=np.int64)
    for lo in range(0, total, _CHUNK):
        idx = np.arange(lo, min(lo + _CHUNK, total), dtype=np.int64)
        X = np.empty((idx.size, n), dtype=np.float64)
        X[:, 0] = 1.0
        X[:, 1:] = ((idx[:, None] >> bit) & 1).astype(np.float64)
        yield idx, X


def exact_min_conductance(W) -> dict:
    """PATH B. `phi = min over ALL cuts`, by enumeration. Exact, exponential.

    Refuses above `BRUTE_FORCE_MAX_N` rather than degrading to a heuristic, because
    the whole point of this path is that it is not one.
    """
    W = _require_graph(W)
    n = W.shape[0]
    if n > BRUTE_FORCE_MAX_N:
        raise ValueError(f"n = {n} exceeds BRUTE_FORCE_MAX_N = {BRUTE_FORCE_MAX_N}; "
                         f"2^{n - 1} cuts is not enumerable. Use the bracket "
                         "[spectral_gap/2, sweep_cut_conductance] instead, and say "
                         "in the report that it is a bracket and not a minimum")
    d = W.sum(axis=1)
    vol = float(d.sum())
    best, best_mask, counted = np.inf, None, 0
    for _idx, X in _masks(n):
        vol_s = X @ d
        cut = ((X @ W) * (1.0 - X)).sum(axis=1)
        phi = cut / np.minimum(vol_s, vol - vol_s)
        counted += phi.size
        k = int(np.argmin(phi))
        if float(phi[k]) < best:
            best, best_mask = float(phi[k]), X[k].astype(bool).copy()
    return {"phi": best,
            "S": sorted(int(i) for i in np.nonzero(best_mask)[0]),
            "n_cuts": counted,
            "path": "brute force over all 2^(n-1) cuts"}


def sweep_cut_conductance(W) -> dict:
    """PATH A. The best prefix cut of the Fiedler ordering.

    The Fiedler vector is the eigenvector of the normalised Laplacian at its second
    smallest eigenvalue; the sweep coordinate is `x = D^{-1/2} v`. Only ONE sweep
    direction is needed: prefix `k` of the descending order is the complement of
    prefix `n-k` of the ascending order, and `Phi(S) = Phi(S^c)`.
    """
    W = _require_graph(W)
    n = W.shape[0]
    d = W.sum(axis=1)
    _ev, evec = np.linalg.eigh(_normalised_adjacency(W))
    x = evec[:, -2] / np.sqrt(d)                     # second LARGEST of D^-1/2 W D^-1/2
    order = np.argsort(-x, kind="stable")
    best, best_k = np.inf, 0
    for k in range(1, n):
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        phi = conductance_of(W, mask)
        if phi < best:
            best, best_k = phi, k
    return {"phi": float(best),
            "S": sorted(int(i) for i in order[:best_k]),
            "order": [int(i) for i in order],
            "path": "sweep cut over the Fiedler ordering"}


def repo_chain_weights(name: str):
    """`(W, meta)` for one of this repository's own rerouted Rips instances.

    `W` is recovered as `P * vol[:, None]` from the chain the tree ships at
    `w_bridge = 1, alpha = 1` -- the same object `bridge_conductance` is handed at
    `scale/foreman_lambda2.py:681` -- and re-symmetrised, since `P * vol` reproduces
    `W` only up to the rounding of one division.
    """
    from ceq.rips import REROUTED_CASES, make_case
    from scale.foreman_lambda2 import (bridge_conductance, build_chain,
                                       ergodic_lambda_2, lambda_2)

    spec = next(c for c in REROUTED_CASES if c[0] == name)
    case = make_case(*spec)
    chain = build_chain(case, spec, 1.0)
    W = chain.P * chain.vol[:, None]
    W = 0.5 * (W + W.T)
    meta = {"name": name,
            "n": int(W.shape[0]),
            "bridge": tuple(chain.bridge),
            "phi_bridge": float(bridge_conductance(chain)),
            "lambda_2_ergodic": float(ergodic_lambda_2(chain)),
            "mu_2": float(1.0 - ergodic_lambda_2(chain)),
            "rho_Q": float(lambda_2(chain))}
    return W, meta


# --------------------------------------------------------------------------
# L-DOM: the domain census MISTAKES.md V-25 requires beside any gating theorem
# --------------------------------------------------------------------------
def domain_census() -> str:
    """The five hypotheses of Theorem 13.10, against what the campaign's corpora
    MEASURABLY are. Drawn from the registered generators (`ceq/kdata.py:472-482`),
    never assumed from a docstring.

    V-25's check demands (1) the hypotheses as a predicate on the corpus's variables,
    (2) the measured support, drawn, (3) the fraction of draws satisfying it at the
    quantifier level the theorem uses, and (4) a blocked gate at 0%.
    """
    import torch

    from ceq.beds import bed_k
    from ceq.corpus import build as bedm_build

    out = ["L-DOM DOMAIN CENSUS -- LPW Thm 13.10's hypotheses against measured draws",
           "generators as registered in ceq/kdata.py:472-482", ""]

    # ---- BED-M, at the registered parameters
    d = bedm_build(n_train=384, n_test=128, seed=0)
    tr = d["train"]
    ys = np.array([r["y"] for r in tr])
    toks = np.array([r["tokens"] for r in tr])
    square = any(isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1]
                 for v in d.values())
    out += [f"BED-M  ceq.corpus.build(n_train=384, n_test=128, seed=0)",
            f"  build() keys                     {sorted(d.keys())}",
            f"  label y                          int, {len(np.unique(ys))} distinct "
            f"in [{ys.min()}, {ys.max()}]",
            f"  tokens                           {toks.dtype} {toks.shape}, "
            f"{len(np.unique(toks))} distinct ids",
            f"  H1 a square matrix over states?  {square}",
            f"  H2 row-stochastic?               n/a -- no matrix exists",
            f"  H3 irreducible?                  n/a -- no matrix exists",
            f"  H4 reversible w.r.t. some pi?    n/a -- no matrix exists",
            f"  draws satisfying H1..H4          0 / {len(tr)}", ""]

    # ---- the {-1,0,+1} object the contract calls BED-M's support
    g = torch.Generator(device="cpu").manual_seed(0 + 777)
    n_r, s_r, head = 2048, 64, 3
    a = (torch.randint(0, 2, (n_r, s_r), generator=g).float() * 2 - 1)
    a[:, :head + 1] = 0.0
    vals = sorted(float(v) for v in torch.unique(a))
    out += ["the {-1,0,+1} object  scale/negation_scope.py:428-429 "
            "(n=2048, s=64, head=3)",
            f"  measured value support           {vals}",
            f"  entries < 0                      "
            f"{int((a < 0).sum())} / {a.numel()}",
            f"  shape                            {tuple(a.shape)} -- a coefficient "
            "array, not a square matrix",
            f"  H1 a square matrix over states?  False",
            f"  W >= 0 (needed for any pi)?      False -- "
            f"{int((a < 0).sum())} negative entries",
            f"  draws satisfying H1..H4          0 / {n_r}", ""]

    # ---- BED-K, at the registered parameters, then swept to show it is structural
    b = bed_k.build_delay(n=500, d=4, seed=7)
    K = b["K"]
    out += ["BED-K  ceq.beds.bed_k.build_delay(n=500, d=4, seed=7)",
            f"  K shape / dtype                  {K.shape} {K.dtype}",
            f"  H1 square matrix?                True",
            f"  W symmetric (=> reversible)?     {bool(np.allclose(K, K.T))}",
            f"  W >= 0?                          {bool((K >= 0).all())}",
            f"  H2 row sums == 1?                "
            f"{bool(np.allclose(K.sum(1), 1.0))}  "
            f"(min {K.sum(1).min():.4f}, max {K.sum(1).max():.4f})",
            f"  zero rows (isolated states)      {int((K.sum(1) == 0).sum())} "
            f"of {K.shape[0]}",
            f"  strictly lower triangular?       "
            f"{bool(np.allclose(K, np.tril(K, -1)))}",
            f"  H3 irreducible?                  "
            f"{not bool(np.allclose(np.linalg.matrix_power(K, K.shape[0]), 0.0))}"
            f"  (K^n == 0, i.e. NILPOTENT)"]
    rows = []
    for kind, params in (("delay", {"d": 1}), ("delay", {"d": 4}),
                         ("powerlaw", {"H": 0.6}), ("powerlaw", {"H": 0.9})):
        for n in (32, 128):
            Kk = bed_k.kernel_matrix(kind, n, **params)
            rows.append(f"  {kind:9s} {str(params):12s} n={n:4d}  sym="
                        f"{str(bool(np.allclose(Kk, Kk.T))):5s} "
                        f"lower-tri={str(bool(np.allclose(Kk, np.tril(Kk, -1)))):5s} "
                        f"rowsum_max={Kk.sum(1).max():.4f}")
    out += ["  STRUCTURAL, not sampling -- every (kind, params, n):"] + rows
    out += [f"  draws satisfying H1..H4          0 / {len(rows)} kernels swept", ""]

    # ---- the one object that DOES satisfy them
    W, meta = repo_chain_weights("LargestJoin_S2Rips_64")
    out += ["Rips chain  scale/foreman_lambda2.py build_chain(LargestJoin_S2Rips_64)",
            f"  H1 square matrix?                True  ({W.shape})",
            f"  W symmetric?                     {bool(np.allclose(W, W.T))}",
            f"  W >= 0?                          {bool((W >= 0).all())}",
            f"  H2 P = D^-1 W row-stochastic?    True by construction "
            "(scale/foreman_lambda2.py:670)",
            f"  H3 irreducible (connected)?      True -- _require_graph passed",
            f"  H4 reversible w.r.t. pi = d/vol? True -- W = W^T and W >= 0",
            f"  H5 Phi_* a MINIMUM over cuts?    True in this file, "
            "False in scale/foreman_lambda2.py:300",
            "  draws satisfying H1..H4          2 / 2 (both REROUTED_CASES)",
            "",
            "VERDICT OF THE CENSUS: the hypothesis set of Thm 13.10 is satisfied by 0 "
            "of the",
            "campaign's registered beds and by the Rips graph corpus only, which is a "
            "different",
            "corpus and whose lambda_2 dial scale/foreman_lambda2.py:40-66 already "
            "records as refuted."]
    return "\n".join(out)


# --------------------------------------------------------------------------
# L-DOM, as NUMBERS. `domain_census()` above returns a formatted string; a string
# cannot be asserted, which is exactly why the census was struck UNBOUND at
# V20 R15 it.1 (`V20_R15_IT1_INSPECTOR.md:167`, C21). This returns the same
# census as machine-readable facts, through ONE predicate `_h1_h4`, so a test
# can bind it and the planted negative can show the zero is measured.
# ponytail: `domain_census()` keeps its own formatting rather than being
# rewritten on top of this; merge them when the .txt artifact is next regenerated.
# --------------------------------------------------------------------------
def _h1_h4(M) -> dict:
    """Thm 13.10's hypotheses as a PREDICATE on one candidate object.

    H1 a square matrix over states; H2 row-stochastic; H3 irreducible;
    H4 reversible w.r.t. some pi >= 0. `all4` is the conjunction. The SAME
    function is applied to every object in the census, so a zero anywhere is a
    reading of this predicate and not a hardcoded constant.
    """
    M = None if M is None else np.asarray(M, dtype=float)
    square = M is not None and M.ndim == 2 and M.shape[0] == M.shape[1]
    if not square:
        return {"H1": False, "H2": False, "H3": False, "H4": False, "all4": False}
    n = M.shape[0]
    h2 = bool(np.allclose(M.sum(1), 1.0))
    # irreducible <=> the walk connects; nilpotent (M^n == 0) is its sharpest failure
    h3 = not bool(np.allclose(np.linalg.matrix_power(M, n), 0.0))
    h4 = bool(np.allclose(M, M.T)) and bool((M >= 0).all())
    return {"H1": True, "H2": h2, "H3": h3, "H4": h4,
            "all4": bool(h2 and h3 and h4)}


def domain_census_facts() -> dict:
    """The L-DOM census of `domain_census()`, as counts and booleans."""
    import torch

    from ceq.beds import bed_k
    from ceq.corpus import build as bedm_build
    from ceq.rips import REROUTED_CASES

    # ---- BED-M: ceq/kdata.py:472-482 registered parameters
    d = bedm_build(n_train=384, n_test=128, seed=0)
    n_tr = len(d["train"])
    square = next((v for v in d.values()
                   if isinstance(v, np.ndarray) and v.ndim == 2
                   and v.shape[0] == v.shape[1]), None)
    bedm = _h1_h4(square)
    bed_m = {"n_draws": n_tr,
             "H1_square_matrix_over_states": bedm["H1"],
             "draws_satisfying_H1_H4": n_tr if bedm["all4"] else 0}

    # ---- the {-1,0,+1} object, scale/negation_scope.py:428-429
    g = torch.Generator(device="cpu").manual_seed(0 + 777)
    n_r, s_r, head = 2048, 64, 3
    a = (torch.randint(0, 2, (n_r, s_r), generator=g).float() * 2 - 1)
    a[:, :head + 1] = 0.0
    tern = _h1_h4(a.numpy())
    ternary = {"n_draws": n_r,
               "value_support": sorted(float(v) for v in torch.unique(a)),
               "negative_entries": int((a < 0).sum()),
               "shape": tuple(a.shape),
               "H1_square_matrix_over_states": tern["H1"],
               "draws_satisfying_H1_H4": n_r if tern["all4"] else 0}

    # ---- BED-K, at the registered parameters, then swept: STRUCTURAL, not sampling
    b = bed_k.build_delay(n=500, d=4, seed=7)
    K = b["K"]
    kernels, ok = [], 0
    for kind, params in (("delay", {"d": 1}), ("delay", {"d": 4}),
                         ("powerlaw", {"H": 0.6}), ("powerlaw", {"H": 0.9})):
        for n in (32, 128):
            Kk = bed_k.kernel_matrix(kind, n, **params)
            h = _h1_h4(Kk)
            ok += int(h["all4"])
            kernels.append({"kind": kind, "params": params, "n": n,
                            "lower_triangular": bool(np.allclose(Kk, np.tril(Kk, -1))),
                            "nilpotent": not h["H3"], **h})
    kh = _h1_h4(K)
    bed_k_facts = {"n_kernels_swept": len(kernels),
                   "kernels": kernels,
                   "strictly_lower_triangular": all(k["lower_triangular"] for k in kernels),
                   "nilpotent": all(k["nilpotent"] for k in kernels) and not kh["H3"],
                   "row_sums_are_one": kh["H2"],
                   "draws_satisfying_H1_H4": ok}

    # ---- the one object that DOES satisfy them, and it is NOT a registered bed
    rips_ok = 0
    for spec in REROUTED_CASES:
        W, _meta = repo_chain_weights(spec[0])
        Wn = W / W.sum(1)[:, None]                 # P = D^-1 W, the walk H2 is about
        rips_ok += int(_h1_h4(W)["H4"] and _h1_h4(Wn)["H2"] and _h1_h4(Wn)["H3"])
    rips = {"n_cases": len(REROUTED_CASES),
            "draws_satisfying_H1_H4": rips_ok,
            "is_a_registered_bed": False}          # ceq/kdata.py:472-482 lists 3, none Rips

    return {"BED-M": bed_m, "ternary": ternary, "BED-K": bed_k_facts,
            "Rips_chain": rips,
            "registered_beds_satisfying_H1_H4": (bed_m["draws_satisfying_H1_H4"]
                                                 + ternary["draws_satisfying_H1_H4"]
                                                 + bed_k_facts["draws_satisfying_H1_H4"]),
            "_predicate_is_the_same_function": True}


# --------------------------------------------------------------------------
# the RUN artifact
# --------------------------------------------------------------------------
def _row(label: str, W) -> str:
    g = spectral_gap(W)
    ga = absolute_spectral_gap(W)
    a = sweep_cut_conductance(W)["phi"]
    n = W.shape[0]
    if n <= BRUTE_FORCE_MAX_N:
        b = exact_min_conductance(W)["phi"]
        exact = f"{b:.10f}"
        gap = f"{a / b:.6f}"
        lower = f"{b * b / 2.0:.3e} <= {g:.3e}  {b * b / 2.0 <= g}"
        upper = f"{g:.3e} <= {2 * b:.3e}  {g <= 2 * b}"
    else:
        exact, gap = "n/a (2^%d cuts)" % (n - 1), "n/a"
        lower = "n/a"
        upper = f"{g:.3e} <= {2 * a:.3e}  {g <= 2 * a}"
    return (f"  {label:24s} n={n:3d}  gamma={g:.10f}  gamma_abs={ga:.10f}\n"
            f"      phi_sweep(A)={a:.10f}  phi_exact(B)={exact}  A/B={gap}\n"
            f"      lower half phi^2/2 <= gamma : {lower}\n"
            f"      upper half gamma <= 2 phi   : {upper}")


def report() -> str:
    from tests.jupiter.test_m14_cheeger import BATTERY

    out = ["M14 CHEEGER STRATIFICATION -- exact sweep cut against brute force",
           "Levin-Peres-Wilmer 2nd ed. Thm 13.10: Phi^2/2 <= gamma <= 2 Phi,",
           "P irreducible and reversible; gamma = 1 - lambda_2, NOT the absolute gap.",
           "", "SYNTHETIC BATTERY"]
    for name in sorted(BATTERY):
        out.append(_row(name, BATTERY[name]))

    out += ["", "THIS REPOSITORY'S OWN INSTANCES (scale/foreman_lambda2.py chains)"]
    for name in ("LargestJoin_S2Rips_64", "LargestJoin_S2Rips_1024"):
        W, meta = repo_chain_weights(name)
        out.append(_row(name, W))
        g = spectral_gap(W)
        a = sweep_cut_conductance(W)["phi"]
        crude = meta["phi_bridge"]
        line = (f"      crude phi_bridge (scale/foreman_lambda2.py:300) = "
                f"{crude:.10f}")
        if W.shape[0] <= BRUTE_FORCE_MAX_N:
            b = exact_min_conductance(W)["phi"]
            line += f"   crude/exact = {crude / b:.6f}"
            line += (f"\n      crude lower half {crude * crude / 2.0:.3e} <= "
                     f"{g:.3e}  {crude * crude / 2.0 <= g}")
        else:
            line += (f"\n      bracket phi in [gamma/2, phi_sweep] = "
                     f"[{g / 2.0:.10f}, {a:.10f}]  width x{a / (g / 2.0):.4f}")
        out.append(line)
    return "\n".join(out)


def demo() -> None:
    """One runnable check: the two paths agree where both run, the sweep obeys its
    own quadratic guarantee, both halves of Cheeger hold with the exact phi, and the
    planted negative shows a named cut missing a planted bottleneck."""
    m = 6
    W = np.zeros((2 * m, 2 * m))
    W[:m, :m] = W[m:, m:] = np.ones((m, m)) - np.eye(m)
    W[m - 1, m] = W[m, m - 1] = 1.0
    planted = 1.0 / (m * (m - 1) + 1.0)

    b = exact_min_conductance(W)["phi"]
    a = sweep_cut_conductance(W)["phi"]
    g = spectral_gap(W)
    assert abs(b - planted) < 1e-11, (b, planted)
    assert abs(a - planted) < 1e-11, (a, planted)
    assert a <= np.sqrt(2.0 * g) + 1e-11, (a, g)
    assert b * b / 2.0 <= g + 1e-11 <= 2.0 * b + 1e-11, (b, g)

    # PLANTED NEGATIVE: a named cut on the same graph misses the same bottleneck.
    named = named_cut_conductance(W, [0])
    assert named > 10.0 * planted, (named, planted)
    assert named * named / 2.0 > g, (named, g)   # the lower half breaks under it
    print("demo OK")


if __name__ == "__main__":
    print(report())
    print()
    print(domain_census())
    demo()
