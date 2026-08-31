"""BED-1 -- the committor-labelled multi-basin bed. CEQ_V15_CONTRACT.md PART III.

BED-1's reason to exist is one strike: **the label is the splitting probability,
never the barrier height.** PART III states the pre-v13 result -- with `m`
parallel saddles at the higher barrier, the higher barrier is the FASTER channel
above `T* = DeltaDeltaE-dagger / ln m` -- and every test below is arranged so a
bed that got that wrong fails. In particular the two labelling schemes are made
to DISAGREE inside the test suite (test 3): a bed on which barrier-height and
committor labels never differ has not been tested on the case that motivated it.

Instrument warnings that shaped this file:

  * MISTAKES.md M-19 (float64 orbit collapse). BED-1's dynamics are a REVERSIBLE
    MARKOV JUMP CHAIN on 11 integer states, not a chaotic map. Nothing here
    iterates a real-valued map, so no bit is lost per step and no
    `mantissa_bits / log2(stretch)` horizon exists to be exceeded. The state
    trajectory is an `int8` array produced by comparing PRNG uniforms against a
    fixed cumulative row -- exact. The only float64 quantities are the rate
    matrix, the committor solve, and the entropies, none of them obtained by
    iteration. That is stated here so the ABSENCE of an exact-rational orbit is
    a recorded engineering claim rather than an omission.
  * MISTAKES.md M-19, second constraint (Bollt et al. 2001: the deficit is
    NON-MONOTONE in partition misplacement). No guard in this file is chosen by
    an argmin over the deficit. Guards are the `q = 1/2` level set of the
    committor, defined before any trajectory exists. Test 7 makes that claim
    checkable rather than asserted, and measures the non-monotonicity directly.
  * MISTAKES.md V-23 (a plural claim evidenced only on its passing members). The
    conservation census in test 9 is REQUIRED to contain a non-conserving row;
    if every row conserved, the census would have checked nothing.
  * MISTAKES.md M-18 CORRECTION (a real defect promoted to explain a failure it
    does not account for). Every number this file reports is measured on the
    quantity the assertion is about -- the committor flux ratio for the strike,
    the lumped transition matrix for CK, the chain's own entropy rate for the
    deficit -- never on a sub-quantity standing in for the scored one.

Run order: this file was authored and shown RED before `ceq/beds/bed_1.py`
existed; V15_BED1.md carries that transcript. Every threshold below cites the
scratch measurement it was set from.
"""
from __future__ import annotations

import numpy as np
import pytest

from ceq.beds import bed_1


T_STAR = 1.0 / np.log(5.0)          # DeltaDeltaE = 1.0, m = 5  ->  0.6213349345596119
T_DYN = 0.25                         # the metastable temperature the dynamics batteries run at
N_WALKERS, N_STEPS = 2000, 6000      # 1.2e7 samples; see per-test tolerance notes


@pytest.fixture(scope="module")
def dyn():
    """One trajectory, shared by the CK and Pesin batteries so both read the
    same draws. T = 0.25 is the metastable regime: the chain's implied
    timescales there are 93.8 / 22.2 / 0.3 steps (measured), i.e. a slow A<->B
    interconversion well separated from the intermediate basin C's internal
    relaxation -- which is what gives the CK test a lag range to be wrong in
    and a lag range to be right in."""
    bed = bed_1.build(T_DYN, seed=0)
    traj = bed_1.simulate(bed, N_WALKERS, N_STEPS, seed=0)
    return bed, traj


# ---------------------------------------------------------------------------
# 1. the committor is harmonic in the interior -- and the check fires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jitter", [0.0, 0.05])
def test_committor_is_harmonic_in_the_interior_and_the_check_fires(jitter):
    """Contract item #14, `committor_eq_harmonic [S]`. `q` solves the backward
    Kolmogorov problem `(Lq)_i = 0` on the interior with `q = 0` on A and
    `q = 1` on B, so the discrete Laplacian of `q` must vanish there to machine
    precision. This is the bed's own correctness proof: it does not depend on
    any downstream number being interesting.

    Both halves, per MISTAKES.md's rule 6 ("a repair must be shown to change the
    object it repairs", read here as: a check must be shown to fail on a wrong
    input). Perturbing ONE interior node by 1e-6 must break it.

    Why two jitters. At `jitter = 0` the landscape is symmetric and `q` is
    dyadic (0, 1/4, 1/2, 3/4, 1) -- the residual measured exactly 0.000e+00,
    which is the strongest possible pass but also a special case. At
    `jitter = 0.05` the node energies are seeded-perturbed, `q` is no longer
    dyadic (`q(C) = 0.3356423` measured at seed 11), and the residual measured
    1.04e-17 against `max|L| = 2.0`. The general solve is exercised, not only
    the symmetric one.

    Thresholds: 1e-13 for the pass (11 orders above the measured 1.04e-17, so
    it is a machine-precision claim and not a tuned one) and 1e-9 for the
    must-fire (the perturbation drives the residual to 1.0e-06 measured, three
    orders above the fire threshold).
    """
    bed = bed_1.build(T_DYN, seed=11, jitter=jitter)
    res = bed_1.harmonic_residual(bed)
    assert res < 1e-13, f"committor is not harmonic on the interior: max|Lq| = {res}"

    # MUST-FIRE: move one interior node off the harmonic solution.
    node = bed["node_index"]["C"]
    assert node in bed["interior"], "the must-fire must perturb an INTERIOR node"
    q_bad = bed["q"].copy()
    q_bad[node] += 1e-6
    res_bad = bed_1.harmonic_residual(bed, q=q_bad)
    assert res_bad > 1e-9, (
        f"perturbing q at interior node {node!r} by 1e-6 left the discrete "
        f"Laplacian at {res_bad}; the harmonic check cannot fail, so its pass "
        "above means nothing"
    )


def test_committor_matches_its_closed_form_on_the_symmetric_landscape():
    """The default landscape's committor is analytically known and TEMPERATURE-
    INDEPENDENT: under Metropolis rates every downhill rate is exactly 1, so a
    barrier top flanked by two deeper basins has
    `q = (q_left + q_right) / 2` whatever the basin depths or T. That gives
    `q(S_lo) = q(S_hi_k) = q(C) = 1/2`, `q(S_ac) = 1/4`, `q(S_cb) = 3/4`.

    Checking the linear solve against a closed form is a stronger statement than
    checking it against its own residual (MISTAKES.md V-3: the two sides range
    over different sets -- one is `np.linalg.solve` on an 9x9 system, the other
    is arithmetic on the graph). Bitwise equality is asserted, not a tolerance:
    the values are dyadic and the solve reproduced them exactly at four
    temperatures in scratch (0.25, 0.3, 0.9T*, 1.1T*).
    """
    expected = {"A": 0.0, "B": 1.0, "C": 0.5, "S_lo": 0.5, "S_ac": 0.25, "S_cb": 0.75}
    for T in (0.25, 0.3, 0.9 * T_STAR, 1.1 * T_STAR):
        bed = bed_1.build(T, seed=0)
        q, ix = bed["q"], bed["node_index"]
        for name, val in expected.items():
            assert q[ix[name]] == val, f"T={T}: q[{name}] = {q[ix[name]]!r}, expected {val!r}"
        for k in range(bed_1.M_PARALLEL):
            assert q[ix[f"S_hi{k}"]] == 0.5, f"T={T}: q[S_hi{k}] = {q[ix[f'S_hi{k}']]!r}"


# ---------------------------------------------------------------------------
# 2. THE STRIKE: barrier-height labelling is wrong above T*
# ---------------------------------------------------------------------------


def test_crossover_temperature_is_delta_delta_over_ln_m():
    """`T* = DeltaDeltaE-dagger / ln m`. Located by bisecting the COMMITTOR
    flux difference `J_hi(T) - J_lo(T)` -- i.e. found from the harmonic solve,
    not from the closed form it is compared against. 200 bisection steps on
    [0.3, 1.5]; scratch measured |bisected - closed form| = 1.110e-16, the same
    order as the contract's own `[reproduced to 6.4e-16]`.
    """
    T_closed = bed_1.crossover_temperature(bed_1.E_HI - bed_1.E_LO, bed_1.M_PARALLEL)
    assert abs(T_closed - T_STAR) < 1e-15

    lo, hi = 0.3, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = bed_1.channel_flux(bed_1.build(mid, seed=0))
        lo, hi = (mid, hi) if f["hi"] < f["lo"] else (lo, mid)
    assert abs(mid - T_closed) < 1e-14, (
        f"committor flux crossover at T={mid!r}, closed form T*={T_closed!r}"
    )


def test_barrier_and_committor_labels_agree_below_Tstar_and_disagree_above():
    """THE TEST THE BED EXISTS FOR, and both halves of it.

    Barrier-height labelling names the channel with the smallest barrier -- `lo`
    (1.0) over `hi` (2.0) -- at every temperature, because the barriers do not
    depend on T. Committor labelling names the channel carrying the largest
    reactive flux, which at `m = 5` degenerate high saddles is `hi` above
    `T* = 0.6213349`.

    The four numbers below are the round's independently re-derived figures and
    are reproduced here THROUGH THE COMMITTOR SOLVE (`2 * J_channel`, the factor
    2 being the two edges of the channel in series; equivalently the bare
    escape rate, of which the committor keeps half at `q = 1/2`), not from the
    closed form:

        0.9 T* :  lo = 1.6725e-01   hi bundle = 1.3986e-01   -> barrier label RIGHT
        1.1 T* :  lo = 2.3151e-01   hi bundle = 2.6799e-01   -> barrier label WRONG

    A bed whose two labelling schemes never disagree has not been tested on the
    case that motivated it, so the disagreement is asserted, not just tolerated.
    """
    below, above = 0.9 * T_STAR, 1.1 * T_STAR

    b_lo = bed_1.build(below, seed=0)
    f_lo = bed_1.channel_flux(b_lo)
    assert 2 * f_lo["lo"] == pytest.approx(1.6725e-01, abs=5e-06)
    assert 2 * f_lo["hi"] == pytest.approx(1.3986e-01, abs=5e-06)
    assert bed_1.label_by_barrier(b_lo) == "lo"
    assert bed_1.label_by_committor(b_lo) == "lo", "below T* the two labels must AGREE"

    b_hi = bed_1.build(above, seed=0)
    f_hi = bed_1.channel_flux(b_hi)
    assert 2 * f_hi["lo"] == pytest.approx(2.3151e-01, abs=5e-06)
    assert 2 * f_hi["hi"] == pytest.approx(2.6799e-01, abs=5e-06)
    assert bed_1.label_by_barrier(b_hi) == "lo"
    assert bed_1.label_by_committor(b_hi) == "hi", (
        "above T* the m-fold higher barrier is the faster channel; the committor "
        "label must say so and the barrier-height label must be wrong"
    )
    assert bed_1.label_by_barrier(b_hi) != bed_1.label_by_committor(b_hi)

    # The barrier label is CONSTANT in T -- that is precisely its defect, and it
    # is checked rather than assumed, so "the labels disagree" cannot be an
    # artifact of the barrier labeller wobbling.
    assert {bed_1.label_by_barrier(bed_1.build(T, seed=0)) for T in
            (0.2, 0.4, below, above, 1.0, 2.0)} == {"lo"}


# ---------------------------------------------------------------------------
# 3. CK test: right lag passes, wrong lag fails, and neither for a trivial reason
# ---------------------------------------------------------------------------


def test_ck_passes_at_the_right_lag_and_fails_at_a_wrong_one(dyn):
    """`T-hat(n.tau) ~= T-hat(tau)^n` on the committor macrostates
    {q < 1/2, q = 1/2, q > 1/2}. The A-side macrostate contains the intermediate
    basin C's approach and the transition-state macrostate contains C itself, so
    a walker that has just entered the TS region through a saddle (residence 1
    step) and one sitting in C (residence ~ e^{0.6/T} steps) have different
    futures: the lumping carries memory that decays on C's internal timescale.
    That memory is what makes lag 1 wrong and lag 64 right.

    Measured at N_WALKERS x N_STEPS (1.2e7 samples), n = 2:

        tau =  1   CK = 0.22702
        tau =  2   CK = 0.20367
        tau = 16   CK = 0.05840
        tau = 32   CK = 0.01253
        tau = 64   CK = 0.00400

    Thresholds 0.10 (fail) and 0.02 (pass) sit between the measured 0.22702 and
    0.00400 with 2x and 5x of margin, and are 5x apart from each other.

    NON-VACUITY, both ends. A CK test passes trivially at tau ~ 0 (nothing has
    happened, `T-hat ~ I`) and again at tau -> infinity (everything is at
    equilibrium, `T-hat ~ Pi`). Both are checked at the PASSING lag so the pass
    cannot be either degeneracy: measured |T-hat(64) - I| = 0.9482 and
    |T-hat(128) - Pi| = 0.1280.
    """
    bed, traj = dyn
    sym = bed_1.macrostate_of_node(bed)[traj]

    err_wrong = bed_1.ck_error(sym, tau=1, n=2, k=3)
    err_right = bed_1.ck_error(sym, tau=64, n=2, k=3)
    assert err_wrong > 0.10, f"CK at the deliberately wrong lag 1 is only {err_wrong}"
    assert err_right < 0.02, f"CK at lag 64 is {err_right}"

    T1 = bed_1.transition_matrix(sym, 64, 3)
    T2 = bed_1.transition_matrix(sym, 128, 3)
    pi_macro = np.array([bed["pi"][bed_1.macrostate_of_node(bed) == m].sum() for m in range(3)])
    d_I = float(np.abs(T1 - np.eye(3)).max())
    d_Pi = float(np.abs(T2 - np.tile(pi_macro, (3, 1))).max())
    assert d_I > 0.5, f"T-hat(64) is within {d_I} of the identity: the pass is 'nothing happened'"
    assert d_Pi > 0.05, f"T-hat(128) is within {d_Pi} of equilibrium: the pass is 'everything relaxed'"


# ---------------------------------------------------------------------------
# 4. Pesin deficit -- near zero at the generating partition, large at a wrong guard
# ---------------------------------------------------------------------------


def test_pesin_deficit_near_zero_at_generating_partition_and_large_at_wrong_guards(dyn):
    """The deterministic Pesin identity does not apply to a stochastic chain, so
    `lambda-hat` is replaced by the quantity that plays its role: the chain's own
    Kolmogorov-Sinai entropy rate `h = -sum_i pi_i sum_j P_ij ln P_ij`, computed
    IN CLOSED FORM from `P` and `pi` (no orbit, no iteration, no horizon). The
    Kolmogorov-Sinai theorem gives `h_sym(alpha) <= h` for every partition
    `alpha`, with equality at a generating one -- and for a Markov chain the
    state partition IS generating. So `h - h_sym` is a non-negative deficit that
    vanishes at the generating partition and is positive at a coarser one.

    Measured at 1.2e7 samples, T = 0.25, exact `h = 0.081275` nats:

        state (generating), L=3      h_sym = 0.081042   deficit = 0.000233
        q = 1/2 guards, 3 symbols    h_sym = 0.071621   deficit = 0.009654
        binary cut at q >= 0.50      h_sym = 0.051386   deficit = 0.029889
        misplaced cut at q >= 0.25   h_sym = 0.050660   deficit = 0.030616
        misplaced cut at q >= 0.75   h_sym = 0.051360   deficit = 0.029915
        barrier-height cut V >= 1.0  h_sym = 0.065231   deficit = 0.016044

    which is the contract's `[RUN: 0.0003 at the right guard, 0.156 at a wrong
    one]` shape at this chain's entropy scale: normalised by `h` the readings are
    0.0029 at the generating partition against 0.119 / 0.368 / 0.377 / 0.368 at
    the coarse ones.

    Thresholds: 0.002 for the near-zero half (8.6x above the measured 0.000233,
    and 4.8x below the smallest coarse reading so the two halves cannot swap),
    0.005 for the "large" half.
    """
    bed, traj = dyn
    h = bed_1.entropy_rate_chain(bed)
    assert h > 0.0, "a chain with zero entropy rate has no deficit to measure"

    d_gen = bed_1.pesin_deficit(bed, traj.astype(np.int64), alphabet=bed["n"], L=3)
    assert 0.0 <= d_gen < 0.002, (
        f"deficit at the GENERATING (state) partition is {d_gen}; the estimator "
        "does not recover the chain's own entropy rate, so nothing it says about "
        "a coarser partition is readable"
    )

    q = bed["q"]
    wrong = {
        "q>=0.25": (q >= 0.25).astype(np.int64)[traj],
        "q>=0.75": (q >= 0.75).astype(np.int64)[traj],
        "V>=E_lo": (bed["V"] >= bed_1.E_LO).astype(np.int64)[traj],
    }
    for name, sym in wrong.items():
        d = bed_1.pesin_deficit(bed, sym, alphabet=2, L=10)
        assert d > 0.005, f"deficit at the wrong guard {name} is only {d}: no signal"
        assert d > 10 * d_gen, (
            f"wrong guard {name} deficit {d} is not separated from the generating "
            f"partition's {d_gen}"
        )


def test_guards_are_the_q_half_level_set_and_were_not_chosen_by_deficit_argmin(dyn):
    """MISTAKES.md M-19, second constraint: Bollt et al. (2001) prove the deficit
    is NON-MONOTONE in partition misplacement, so it is admissible as a
    near-zero test and inadmissible as a ranking or search criterion.

    Two things are checked. First, the guard set is exactly the `q = 1/2` level
    set -- a definition that exists before any trajectory does, and that involves
    no entropy at all. Second, that this is a CHECKABLE claim and not a
    disclaimer: the deficit is measured at three binary cuts, `q >= 0.25`,
    `q >= 0.50`, `q >= 0.75`, whose misplacements |theta - 1/2| are 0.25, 0.00,
    0.25. Measured deficits 0.030616 / 0.029889 / 0.029915 -- a spread of 2.4%
    across a 0.25 swing in misplacement, and NOT ordered by it (the two equally
    misplaced cuts differ from each other by more than one of them differs from
    the correct cut). An argmin over these would be selecting noise, which is
    exactly Bollt's point, and the assertion below records that rather than the
    ranking.
    """
    bed, traj = dyn
    q = bed["q"]
    level_set = {i for i in range(bed["n"]) if q[i] == 0.5}
    assert set(bed["guards"]) == level_set, "guards are not the q = 1/2 level set"
    assert 0 < len(level_set) < bed["n"], "the level set is degenerate (empty or everything)"

    ds = {t: bed_1.pesin_deficit(bed, (q >= t).astype(np.int64)[traj], alphabet=2, L=10)
          for t in (0.25, 0.50, 0.75)}
    spread = max(ds.values()) - min(ds.values())
    assert spread < 0.25 * min(ds.values()), (
        f"deficits across cuts at 0.25 / 0.50 / 0.75 are {ds}; if they were well "
        "separated an argmin might be defensible, and this test's premise would "
        "need re-stating"
    )
    assert abs(ds[0.25] - ds[0.75]) > 0.0, (
        f"two cuts with identical misplacement 0.25 read identically ({ds}); the "
        "non-monotonicity claim has no instance here"
    )


# ---------------------------------------------------------------------------
# 5. Morse census
# ---------------------------------------------------------------------------


def test_morse_census_closes_on_the_euler_characteristic_and_moves_when_V_moves():
    """On a graph (a 1-complex) the Morse relation is `m_0 - m_1 = chi = V - E`,
    and the Morse inequalities are `m_0 >= b_0 = 1`, `m_1 >= b_1 = E - V + 1`.
    The default landscape has 11 nodes and 16 edges, so `chi = -5`, `b_1 = 6`,
    and the census must read 3 minima (A, B, C) and 8 index-1 saddles.

    MUST-FIRE. Raising C above its two flanking saddles turns it from a minimum
    into a critical point of index 1: the census must then read 2 minima and 7
    saddles. `chi` is topological and CANNOT move -- so this half checks the
    CLASSIFIER moved while the invariant did not, which a census that simply
    returned constants would fail.
    """
    bed = bed_1.build(T_DYN, seed=0)
    c = bed_1.morse_census(bed)
    assert (c["n_nodes"], c["n_edges"], c["euler"]) == (11, 16, -5)
    assert (c["minima"], c["index_1"]) == (3, 8), c
    assert c["minima"] - c["index_1"] == c["euler"]
    assert c["minima"] >= c["betti_0"] and c["index_1"] >= c["betti_1"], c

    raised = bed_1.build(T_DYN, seed=0, V_override={"C": 2.5})
    c2 = bed_1.morse_census(raised)
    assert (c2["minima"], c2["index_1"]) == (2, 7), c2
    assert c2["euler"] == c["euler"] == -5, "chi is topological and must not move"
    assert c2["minima"] - c2["index_1"] == c2["euler"]


# ---------------------------------------------------------------------------
# 6. conservation census -- including what does NOT conserve
# ---------------------------------------------------------------------------


def test_conservation_census_names_a_non_conserving_carrier(dyn):
    """MISTAKES.md V-23: "a conservation census with no non-conserving row has
    either checked nothing or omitted something." The census is therefore
    required to contain at least one row that does not conserve, and the
    conserving rows are required to actually conserve.

    Measured at T = 0.25 (scratch): transition rows sum to 1 within 1.11e-16;
    `pi P - pi` within 1.11e-16; detailed balance `pi_i P_ij - pi_j P_ji` within
    1.36e-20; the discrete Laplacian of `q` exactly 0; TPT reactive flux is
    divergence-free on the interior within 1.36e-20 and `out(A) == in(B)` to
    5.172878e-03 on both sides.

    The rows that do NOT conserve, printed rather than omitted:
      * guard-crossing balance -- A->B and B->A crossing counts are equal only in
        expectation; measured 26666+2462+3602 forward against 26736+2456+3559
        back, an imbalance of -21 on 65k events (0.08 sqrt-units).
      * guard-crossing share vs reactive flux for the TRAP channel -- 0.11005
        of crossings against 0.05837 of the flux. A channel containing a
        metastable intermediate recrosses its own guard, so raw crossing counts
        over-count it; the two direct channels agree to 0.8%.
    """
    bed, traj = dyn
    census = bed_1.conservation_census(bed, traj)
    assert census, "empty census"

    conserving = [r for r in census if r["conserves"]]
    failing = [r for r in census if not r["conserves"]]
    assert len(conserving) >= 5, f"only {len(conserving)} conserving rows: {census}"
    assert failing, (
        "the census reports no non-conserving carrier; V-23 says it has then "
        f"either checked nothing or omitted something. rows: {[r['name'] for r in census]}"
    )
    for r in conserving:
        assert r["value"] <= r["tol"], f"row {r['name']} claims to conserve at {r['value']} > {r['tol']}"
    for r in failing:
        assert r["value"] > r["tol"], f"row {r['name']} is listed as failing but reads {r['value']}"
        assert r["note"], f"non-conserving row {r['name']} ships without its mechanism"


# ---------------------------------------------------------------------------
# 7. the itinerary reproduces the strike from simulation, independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale,expect", [(0.9, "lo"), (1.1, "hi")])
def test_guard_itinerary_channel_shares_reproduce_the_committor_flux_ratio(scale, expect):
    """A third route to the strike, and the only one that runs the dynamics: the
    itinerary of guard crossings. `lo : hi` counted from simulated A->B guard
    traversals must match `J_lo : J_hi` from the harmonic solve, which in turn
    equals the closed-form `exp(-E_lo/T) / (m exp(-E_hi/T))`. The simulation
    never reads `bed["q"]` for anything but the side labels and never reads the
    fluxes at all, so the agreement is a check and not an identity
    (MISTAKES.md V-3).

    Measured at 2000 x 6000: 0.9T* empirical 1.19715 against committor 1.19581
    (0.11%); 1.1T* empirical 0.86679 against committor 0.86389 (0.34%). Tolerance
    2% relative, ~6x the larger measured gap. At 1.1T* the empirical ratio is
    BELOW 1 -- the simulated walkers cross the m-fold higher barrier more often
    than the single lower one, which is the strike observed rather than derived.
    """
    T = scale * T_STAR
    bed = bed_1.build(T, seed=0)
    traj = bed_1.simulate(bed, N_WALKERS, N_STEPS, seed=0)
    it = bed_1.guard_itinerary(bed, traj)

    emp = it["forward"]["lo"] / it["forward"]["hi"]
    f = bed_1.channel_flux(bed)
    ref = f["lo"] / f["hi"]
    closed = (bed_1.arrhenius_channel_rate(bed_1.E_LO, T)
              / bed_1.arrhenius_channel_rate(bed_1.E_HI, T, bed_1.M_PARALLEL))
    assert ref == pytest.approx(closed, rel=1e-12)
    assert emp == pytest.approx(ref, rel=0.02), (
        f"itinerary lo:hi = {emp} against committor flux {ref} at T = {scale}T*"
    )
    dominant = "lo" if emp > 1.0 else "hi"
    assert dominant == expect, (
        f"at T = {scale}T* the simulated itinerary is dominated by {dominant}, "
        f"expected {expect}"
    )


# ---------------------------------------------------------------------------
# 8. seeded reproducibility, bitwise
# ---------------------------------------------------------------------------


def test_seeded_reproducibility_bitwise():
    """Both directions. A generator that ignored its seed would pass a
    same-seed-only check, so the different-seed half is required (BED-K's test 6
    makes the same argument)."""
    b1 = bed_1.build(T_DYN, seed=7, jitter=0.05)
    b2 = bed_1.build(T_DYN, seed=7, jitter=0.05)
    assert np.array_equal(b1["V"], b2["V"])
    assert np.array_equal(b1["q"], b2["q"])
    b3 = bed_1.build(T_DYN, seed=8, jitter=0.05)
    assert not np.array_equal(b1["V"], b3["V"])

    bed = bed_1.build(T_DYN, seed=0)
    t1 = bed_1.simulate(bed, 200, 500, seed=3)
    t2 = bed_1.simulate(bed, 200, 500, seed=3)
    t3 = bed_1.simulate(bed, 200, 500, seed=4)
    assert np.array_equal(t1, t2), "same seed, different trajectory"
    assert not np.array_equal(t1, t3), "different seed, identical trajectory"
    # jitter=0 must be a real landscape, not a silently-empty one (V-8's
    # non-degeneracy rule applied to the control arm of the seeding check).
    assert np.array_equal(bed_1.build(T_DYN, seed=0)["V"], bed_1.build(T_DYN, seed=9)["V"])
