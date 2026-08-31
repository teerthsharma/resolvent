"""V15 X8' -- Mobius inversion probe.

Delta claim under test (`CEQ_V15_1_DELTA.md`, X8'): "the four-point estimator
IS the Mobius inversion of f on the Boolean lattice", and the general k-order
interaction is mu-weighted inclusion-exclusion over subsets (Rota).

V-3 GUARD. V-3 in MISTAKES.md is an assertion that turned out to be an
algebraic identity of the prober's own construction -- both sides written
from the same understanding, so agreement proved only self-consistency. The
guard here: the Mobius inversion below is implemented from the bare poset
definition (sum over subsets S of T, sign (-1)^(|T|-|S|)), independently of
anything in scale/twodof.py, and it is compared against
`scale.twodof.leakage_ratio` -- the repo's actual estimator -- imported and
called UNMODIFIED. Where the repo has no k=4 estimator to compare against,
this file says so rather than manufacturing one and calling it a match.

Nothing trains. float64 throughout. Read-only with respect to the rest of
the repo (imports scale.twodof; does not edit it).
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)                       # pinned here, matches twodof.py

from scale import twodof  # noqa: E402  -- unmodified repo module, imported not copied


# ===========================================================================
# PART 0 -- THE POSET DEFINITION (Rota, incidence algebra of the Boolean
# lattice). Two independent derivations, cross-checked against each other
# before either touches repo data.
# ===========================================================================

def mobius_boolean(S: frozenset, T: frozenset) -> int:
    """mu(S, T) = (-1)^(|T| - |S|) for S subseteq T. The closed form that
    holds specifically on the Boolean lattice; Mobius inversion itself is a
    theorem on any locally finite poset (Rota 1964), and the alternating-sign
    inclusion-exclusion formula is what that theorem specializes to here."""
    if not S <= T:
        raise ValueError(f"{S} is not a subset of {T}")
    return -1 if (len(T) - len(S)) % 2 else 1


def mobius_boolean_recursive(S: frozenset, T: frozenset,
                              memo: dict | None = None) -> int:
    """The SAME function, derived the other way: Rota's defining recurrence
    mu(S,S) = 1, mu(S,T) = -sum_{S subseteq U subsetneq T} mu(S,U), with no
    reference to the closed-form sign rule above. Used only as an internal
    cross-check -- if this ever disagreed with mobius_boolean, the closed
    form would be the thing in error, not the repo."""
    if memo is None:
        memo = {}
    if S == T:
        return 1
    key = (S, T)
    if key in memo:
        return memo[key]
    total = 0
    rest = list(T - S)
    for r in range(len(rest)):                # strict subsets only: U != T
        for combo in itertools.combinations(rest, r):
            U = S | frozenset(combo)
            total += mobius_boolean_recursive(S, U, memo)
    memo[key] = -total
    return -total


def mobius_inversion(f, T: frozenset) -> float:
    """g(T) = sum_{S subseteq T} mu(S,T) f(S). THIS is the general k-order
    interaction: k = |T|, mu-weighted inclusion-exclusion over all 2^k
    subsets. Implemented directly from the definition -- no reference to
    scale/twodof.py's arithmetic anywhere in this function."""
    total = 0.0
    elts = list(T)
    for r in range(len(elts) + 1):
        for combo in itertools.combinations(elts, r):
            S = frozenset(combo)
            total += mobius_boolean(S, T) * f(S)
    return total


def cross_check_mu(max_k: int = 4) -> float:
    """Closed form vs recurrence, every S subseteq T with |T| <= max_k, on an
    abstract universe -- no attention geometry, no repo code involved."""
    universe = list(range(max_k))
    worst = 0.0
    for k in range(max_k + 1):
        for T in itertools.combinations(universe, k):
            T = frozenset(T)
            for r in range(k + 1):
                for S in itertools.combinations(T, r):
                    S = frozenset(S)
                    a = mobius_boolean(S, T)
                    b = mobius_boolean_recursive(S, T)
                    worst = max(worst, abs(a - b))
    return worst


# ===========================================================================
# PART 1 -- THE IDENTITY, AGAINST THE REPO'S ESTIMATOR, UNMODIFIED.
# scale/twodof.py's `leakage_ratio` is the only four-point/eight-point
# estimator found in scale/ or ceq/ (grepped for walsh/epistasis/mobius/
# four-point/two-point across both trees; nothing else defines one).
# ===========================================================================

S, D, SEED = 32, 8, 20260826
C, J, M, N4, TARGET = 5, 11, 17, 23, 3         # N4 is the 4th token for k=4
I_ROW = S - 1


def attention_f(q, k):
    """f(S) = row i's weight on `target` after masking exactly S out, via
    scale.twodof.masked_row -- the repo's own primitive, called unmodified,
    with the (c, j, delta) perturbation argument zeroed out so only `drop`
    (the S argument) matters."""
    zero = torch.zeros(2, dtype=torch.float64)

    def f(Sset: frozenset) -> float:
        return float(twodof.masked_row(q, k, I_ROW, C, J, zero, tuple(Sset))[TARGET])
    return f


def repo_signed(f, order: int) -> float:
    """The repo's own arithmetic (scale/twodof.py:245-247), reproduced ONLY
    to recover a SIGN -- `leakage_ratio` returns abs() values, which cannot
    distinguish agreement from agreement-up-to-sign. The magnitude comparison
    below uses `leakage_ratio` directly and unmodified; this function exists
    solely so the sign relationship can be reported honestly instead of
    hidden behind abs()."""
    if order == 2:
        return f(frozenset()) - f(frozenset({C})) - f(frozenset({J})) \
            + f(frozenset({C, J}))
    if order == 3:
        return (f(frozenset()) - f(frozenset({C})) - f(frozenset({J}))
                - f(frozenset({M})) + f(frozenset({C, J})) + f(frozenset({C, M}))
                + f(frozenset({J, M})) - f(frozenset({C, J, M})))
    raise ValueError(order)


def run_identity(n_geoms: int = 8) -> dict:
    """Swept over several DRAWN geometries (t = 0..n_geoms-1), not one
    hand-built instance -- MISTAKES.md's rule for V-2/V-3: draw, don't build,
    and report the count/range rather than a single flattering cell."""
    k2_signed, k2_abs, k3_signed, k3_abs, k3_flip, k4_vals = [], [], [], [], [], []
    example = None
    for t in range(n_geoms):
        q, k = twodof.geometry(S, D, SEED + t)
        q, k = q.double(), k.double()
        f = attention_f(q, k)

        T2 = frozenset({C, J})
        T3 = frozenset({C, J, M})
        T4 = frozenset({C, J, M, N4})
        mob2, mob3, mob4 = (mobius_inversion(f, T) for T in (T2, T3, T4))

        # The repo's estimator, called UNMODIFIED, exactly as it ships.
        repo_abs_i, repo_abs_l3 = twodof.leakage_ratio(
            q, k, i=I_ROW, c=C, j=J, m=M, target=TARGET)
        repo_i_signed = repo_signed(f, 2)
        repo_l3_signed = repo_signed(f, 3)

        k2_signed.append(abs(mob2 - repo_i_signed))
        k2_abs.append(abs(abs(mob2) - repo_abs_i))
        k3_signed.append(abs(mob3 - repo_l3_signed))
        k3_flip.append(abs(mob3 - (-repo_l3_signed)))
        k3_abs.append(abs(abs(mob3) - repo_abs_l3))
        k4_vals.append(mob4)

        if t == 0:
            example = dict(mob2=mob2, repo_i_signed=repo_i_signed,
                            repo_abs_i=repo_abs_i, mob3=mob3,
                            repo_l3_signed=repo_l3_signed, repo_abs_l3=repo_abs_l3,
                            mob4=mob4)

    out = {"n_geoms": n_geoms, "example": example}
    out["k2"] = dict(max_abs_diff_signed=max(k2_signed),
                      max_abs_diff_of_abs=max(k2_abs))
    out["k3"] = dict(max_abs_diff_signed_direct=max(k3_signed),
                      max_abs_diff_after_sign_flip=max(k3_flip),
                      max_abs_diff_of_abs=max(k3_abs))
    out["k4"] = dict(values=k4_vals, max_abs=max(abs(v) for v in k4_vals),
                      note="no repo estimator exists at k=4 (see PART 3 rename "
                      "plan / grep census) -- nothing to compare against; "
                      "reported for completeness only")
    return out


# ===========================================================================
# PART 2 -- MUST-FIRE AND POSITIVE CONTROL (Contract PART III EPISTASIS
# positive control; PART II item #9).
# ===========================================================================

def additive_landscape(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    c0 = float(torch.randn((), generator=g, dtype=torch.float64))
    c = torch.randn(n, generator=g, dtype=torch.float64)

    def f(S: frozenset) -> float:
        return c0 + sum(float(c[i]) for i in S)
    return f


def multiplicative_landscape(n: int, seed: int):
    """f(S) = c0 * prod_{i in S} (1 + c_i). Expanding the k=2 Mobius
    inversion of this f gives exactly c0 * c_a * c_b for T = {a,b} -- an
    epistatic term with no additive decomposition, generically nonzero."""
    g = torch.Generator().manual_seed(seed)
    c0 = float(torch.randn((), generator=g, dtype=torch.float64))
    c = torch.randn(n, generator=g, dtype=torch.float64)

    def f(S: frozenset) -> float:
        p = 1.0
        for i in S:
            p *= (1.0 + float(c[i]))
        return c0 * p
    return f


def sweep_orders(f, n: int, orders=(2, 3, 4)) -> dict:
    universe = list(range(n))
    out = {}
    for k in orders:
        vals = []
        for T in itertools.combinations(universe, k):
            vals.append(abs(mobius_inversion(f, frozenset(T))))
        out[k] = dict(mean_abs=sum(vals) / len(vals), max_abs=max(vals), n=len(vals))
    return out


def run_must_fire_and_positive_control(n: int = 6, seeds=range(5)) -> dict:
    add_worst = 0.0
    add_means = {2: [], 3: [], 4: []}
    mul_means = {2: [], 3: [], 4: []}
    for seed in seeds:
        fa = additive_landscape(n, seed)
        fm = multiplicative_landscape(n, seed + 10_000)
        ra = sweep_orders(fa, n)
        rm = sweep_orders(fm, n)
        for k in (2, 3, 4):
            add_worst = max(add_worst, ra[k]["max_abs"])
            add_means[k].append(ra[k]["mean_abs"])
            mul_means[k].append(rm[k]["mean_abs"])
    return dict(
        additive_must_fire_max_abs_over_all_orders_and_seeds=add_worst,
        additive_mean_abs_by_order={k: sum(v) / len(v) for k, v in add_means.items()},
        multiplicative_mean_abs_by_order={k: sum(v) / len(v) for k, v in mul_means.items()},
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    print("V15 X8' -- MOBIUS INVERSION PROBE")
    print(f"dtype=float64 threads={torch.get_num_threads()} "
          f"geometry: s={S} d={D} seed={SEED} c={C} j={J} m={M} n4={N4} "
          f"target={TARGET} i={I_ROW}")
    print()

    print("=== PART 0: mu closed-form vs mu recurrence (Rota), abstract, k<=4 ===")
    worst_mu = cross_check_mu(4)
    print(f"  max |closed_form - recurrence| over every (S,T) with |T|<=4: "
          f"{worst_mu:.6e}")
    print()

    print("=== PART 1: IDENTITY vs scale.twodof.leakage_ratio (unmodified) ===")
    ident = run_identity()
    ex = ident["example"]
    print(f"  swept over {ident['n_geoms']} drawn geometries "
          f"(seed={SEED}..{SEED + ident['n_geoms'] - 1}), not one hand-built cell.")
    print("  example cell (t=0):")
    print(f"    k=2  mobius_inversion(f,T) = {ex['mob2']:.12e}   "
          f"repo signed = {ex['repo_i_signed']:.12e}   "
          f"repo abs (leakage_ratio) = {ex['repo_abs_i']:.12e}")
    print(f"    k=3  mobius_inversion(f,T) = {ex['mob3']:.12e}   "
          f"repo signed L3 = {ex['repo_l3_signed']:.12e}   "
          f"repo abs (leakage_ratio) = {ex['repo_abs_l3']:.12e}")
    print(f"    k=4  mobius_inversion(f,T) = {ex['mob4']:.12e}   (no repo estimator)")

    k2 = ident["k2"]
    print("  k=2, T={c,j}, over all swept geometries:")
    print(f"    max|diff| signed                 = {k2['max_abs_diff_signed']:.3e}")
    print(f"    max|diff| of abs values          = {k2['max_abs_diff_of_abs']:.3e}")

    k3 = ident["k3"]
    print("  k=3, T={c,j,m}, over all swept geometries:")
    print(f"    max|diff| DIRECT (mobius - L3)   = {k3['max_abs_diff_signed_direct']:.3e}")
    print(f"    max|diff| AFTER (-1)^k sign flip = {k3['max_abs_diff_after_sign_flip']:.3e}")
    print(f"    max|diff| of abs values          = {k3['max_abs_diff_of_abs']:.3e}")

    k4 = ident["k4"]
    print("  k=4, T={c,j,m,n4}: NO REPO ESTIMATOR EXISTS TO COMPARE AGAINST.")
    print(f"    max|mobius_inversion(f,T)| over sweep = {k4['max_abs']:.3e}  (reported only)")
    print()

    print("=== PART 2: MUST-FIRE (additive) and POSITIVE CONTROL (multiplicative) ===")
    mfpc = run_must_fire_and_positive_control()
    print(f"  additive f, must-fire, max|interaction| over k in "
          f"(2,3,4) and 5 seeds: {mfpc['additive_must_fire_max_abs_over_all_orders_and_seeds']:.6e}")
    for k in (2, 3, 4):
        print(f"    k={k}  additive mean|I| = "
              f"{mfpc['additive_mean_abs_by_order'][k]:.6e}   "
              f"multiplicative mean|I| = "
              f"{mfpc['multiplicative_mean_abs_by_order'][k]:.6e}")
    print()
    print("  Contract's cited figure for this cell is `[RUN: 0.0375 vs 0.0000")
    print("  additive]` (CEQ_V15_CONTRACT.md:194). That exact pair does not")
    print("  reproduce from any epistasis/Mobius code in this repo -- see the")
    print("  provenance note below and V15_X8_MOBIUS.md PART 2.")
    print()

    print("=== SELF-CHECK (ponytail: the one thing that fails if this breaks) ===")
    ok = True
    if worst_mu != 0:
        print("  FAIL: mu closed-form and recurrence disagree.")
        ok = False
    if k2["max_abs_diff_signed"] > 1e-9:
        print("  FAIL: k=2 does not match the repo estimator.")
        ok = False
    if k3["max_abs_diff_after_sign_flip"] > 1e-9:
        print("  FAIL: k=3 does not match the repo estimator even up to sign.")
        ok = False
    if mfpc["additive_must_fire_max_abs_over_all_orders_and_seeds"] > 1e-9:
        print("  FAIL: additive landscape did not read exactly 0.")
        ok = False
    if mfpc["multiplicative_mean_abs_by_order"][2] < 1e-6:
        print("  FAIL: positive control read ~0, instrument is blind.")
        ok = False
    print("  PASS" if ok else "  ONE OR MORE CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
