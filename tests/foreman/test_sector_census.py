"""W5.2's GAUGE THEOREM -- measured directly, plus the census of past results
it empties.

THIS FILE IS THE WHAT-BREAKS STANCE'S OWN, NOT THE SECTOR-CENSUS LANE'S.
`ceqjepa/sector_census.py` is a SEPARATE file owned by a different agent in
this lane; this file only tests it (PART 2) and does not implement it. Part 1
below needs no such module at all -- it tests W5.2's claim directly against
`ceq/arm_smprime.py`, which already ships, so the addendum's own number does
not have to wait on anyone else's file landing.

THE QUESTION THIS FILE ANSWERS. Does W5.2's gauge theorem -- "on a
propagation graph with beta_1 = 0 the phase gate is gauge-equivalent to
unsigned gates with rephased values" -- hold NUMERICALLY on the machinery this
repo actually ships, and what does it destroy if it does. Two measurements,
not one: the pure path-product case (clause 1's own multiplicative chain) and
the harder affine-recurrence case (`ceq/arm_phase.chain_label`, which injects
a real drive `b_i` at every step and is what BIND 5 of `tests/arm_phase`
built its planted negative against). Both hold to float64 roundoff; the
worst measured gap over the sweeps below is reported by name rather than
asserted only as "small" (MISTAKES.md M-2: a threshold chosen after seeing
the number is not a threshold, so the bars below are round decades wider than
anything measured, not fitted to it).

WHY THE GRAPH IS BETA_1 = 0 UNCONDITIONALLY HERE, NOT ONLY BY DRAW.
`path_product`'s own signature is `[..., S] -> [..., S, S]`: one sequence,
indexed once. There is no argument through which two positions could carry a
SECOND, independent route between them. So the propagation graph every
existing gate in this repo runs on -- the chain 0-1-...-(S-1) -- has
`beta_1 = 0` by the SHAPE of the function, not by anything about the values
drawn onto it. `_chain_betti` below states that as arithmetic (E - V + b0 for
a connected graph) so the claim does not lean on `ceqjepa.sector_census`
existing to be checked.

RED FIRST, PART 2. `ceqjepa/sector_census.py` did not exist when this file's
first draft was written; it landed mid-investigation, from the other agent
in this lane, and this file was updated in place rather than left quoting a
stale `ModuleNotFoundError` -- W5.1's own tests above are now re-pointed at
the real API (`betti_numbers`, `sector_dims`) and PASS, verifying the landed
census against an independent chain-arithmetic route. W5.3 (holonomy) and
W5.4 (the bed-admission law) are NOT in the landed file at all -- no
`holonomy`, `interference_fraction`, `CertificateRefused` or
`assert_bed_admissible` -- so the five tests naming them still fail, now with
`AttributeError: module 'ceqjepa.sector_census' has no attribute '...'`
rather than an import error, and that is the accurate RED to hand Wilson: not
"the file is missing," but "W5.1 shipped, W5.3/W5.4 did not."

The import stays INSIDE each test function rather than at module scope for
the same reason it started that way: a module-scope import failure would
have failed PART 1's collection too (this repo's own
`--continue-on-collection-errors` note in `pytest.ini` documents exactly that
failure mode for a whole-file import), and Part 1 is the number this
addendum actually asked to be measured, independent of Part 2 entirely.

PART 3, THE HARDER HALF: which of this repo's PAST phase-carrying sign
results sit on a beta_1 = 0 bed and are therefore GAUGE-TRIVIAL BY SCOPE --
not wrong, not retracted, VACUOUS BY SCOPE, and the distinction is kept
explicit in every verdict string below rather than compressed to a verb. Two
companion tables sit beside the relabeled one and are exactly as load-bearing:
`OUT_OF_SCOPE_REAL_SIGNED_RESULTS` names sign results that use a REAL signed
operator (a difference of softmaxes, or Zaslavsky switching D = diag(+-1))
and carry no U(1) phase connection at all, so W5.2 has nothing to say about
them; and the module documents, by measurement rather than silence, that no
existing INTERFERENCE claim is being relabeled because none exists yet in
this repo -- `path_product` gives one route between any two positions, so a
two-route interference bed is something W5.4 would have to build, not
something already sitting mislabeled in the suite.
"""
from __future__ import annotations

import math
import pathlib
import re

import pytest
import torch

from ceq.arm_smprime import gate, path_product

DT = torch.float64
ROOT = pathlib.Path(__file__).resolve().parents[2]


# ===========================================================================
# PART 1 -- W5.2 measured directly. No dependency on ceqjepa.sector_census.
# ===========================================================================

def _chain_betti(n_nodes: int, extra_edges=()) -> tuple[int, int, int]:
    """(beta_0, beta_1, |E|) of the path graph 0-1-...-(n_nodes-1) plus any
    `extra_edges`. Union-find plus `E - V + beta_0` for a graph with `beta_0`
    components -- elementary, and stated here so W5.1's sector-dimension
    formula can be checked without importing anything from the sibling file.
    """
    edges = {(min(i, j), max(i, j)) for i, j in
             list(zip(range(n_nodes - 1), range(1, n_nodes))) + list(extra_edges)}
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    beta0 = len({find(i) for i in range(n_nodes)})
    beta1 = len(edges) - n_nodes + beta0
    return beta0, beta1, len(edges)


def test_the_chain_propagation_graph_is_beta1_zero_by_construction():
    """`path_product` propagates on the path 0-1-...-(S-1) only: `E = S - 1`,
    connected, so `beta_1 = 0` identically -- a property of the function's
    own shape, checked at three sizes rather than asserted once."""
    for n in (2, 8, 64):
        beta0, beta1, e = _chain_betti(n)
        assert (beta0, beta1) == (1, 0), (n, beta0, beta1)
        assert e == n - 1


def test_w5_1_must_fire_a_planted_extra_cycle_raises_beta1_by_exactly_one():
    """W5.1's own must-fire, stated as graph arithmetic alone: one edge
    joining two already-connected chain nodes adds exactly one independent
    cycle and disconnects nothing."""
    n = 8
    beta0_before, beta1_before, _ = _chain_betti(n)
    beta0_after, beta1_after, _ = _chain_betti(n, extra_edges={(0, n - 1)})
    assert beta1_after - beta1_before == 1, (beta1_before, beta1_after)
    assert beta0_after == beta0_before == 1


def test_w5_1_sector_dims_formula_matches_n_minus_beta0_e_minus_n_plus_beta0_minus_beta1_beta1():
    """The addendum's own triple `(n - beta_0, E - n + beta_0 - beta_1, beta_1)`,
    on the plain chain and on the chain-plus-one-cycle, stated in the addendum's
    own variable names rather than renamed for convenience."""
    n = 8
    for extra in ((), {(0, n - 1)}):
        beta0, beta1, e = _chain_betti(n, extra_edges=extra)
        dims = (n - beta0, e - n + beta0 - beta1, beta1)
        assert dims[0] + dims[2] == n - beta0 + beta1        # internal consistency
        assert dims[2] == beta1
        assert dims[2] == (1 if extra else 0)


def _rephase_unsigned(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """THE CLAIMED GAUGE-EQUIVALENT CONSTRUCTION, clause-for-clause out of
    W5.2's own words: `g_i = exp(i * cumsum(theta)_i)` are the vertex
    potentials a spanning-tree (here, a path -- the simplest tree) gauge
    fixes uniquely up to one global phase; `R = path_product(m)` is the real,
    non-negative -- "unsigned" -- route `ceq/arm_smprime.py` already ships as
    the modulus row; `recon_ij = g_i * R_ij * g_j^-1` for `j <= i` is the
    identity under test.
    """
    n = m.shape[-1]
    R = path_product(m)
    phi = torch.cumsum(theta, dim=-1)
    idx = torch.arange(n, device=m.device)
    le = idx.unsqueeze(-1) >= idx.unsqueeze(-2)
    phase = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    return torch.polar(R, phase).masked_fill(~le, 0)


def test_w5_2_gauge_theorem_the_path_product_case_measured_against_2_8e_15():
    """THE NUMBER THE ADDENDUM NAMES, on clause 1's own multiplicative chain.
    `path_product(gate(m, theta))` against the rephased-unsigned
    reconstruction, S=64, theta drawn wide enough to wind several full turns
    (+-10 rad), 50 seeds. `2.8e-15` is an ORDER OF MAGNITUDE for a length-64
    float64 cumulative product, not a bitwise target, so the bar asserted is
    two decades wider than anything measured and the worst seed is printed
    rather than only asserted against.
    """
    n = 64
    worst, worst_seed = 0.0, None
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        m = torch.rand(n, generator=g, dtype=DT) * 0.999 + 1e-4
        theta = (torch.rand(n, generator=g, dtype=DT) - 0.5) * 20.0
        gc = path_product(gate(m, theta))
        recon = _rephase_unsigned(m, theta)
        d = (gc - recon).abs().max().item()
        if d > worst:
            worst, worst_seed = d, seed
    print(f"\n  PART 1 measured: worst max|G - rephase(unsigned)| over 50 seeds "
          f"at S=64 = {worst:.6e} (seed {worst_seed}); addendum claims 2.8e-15")
    assert worst < 1e-13, (worst, worst_seed)
    assert torch.isfinite(torch.tensor(worst))


def test_w5_2_the_bed_m_support_case_theta_in_zero_pi_is_exact_not_approximate():
    """The DISCRETE corner of the same theorem, on BED-M's own support
    `theta in {0, pi}`: the rephased-unsigned reconstruction against
    `path_product(gate(...))` directly, S=32. Measured at seed 0:
    `2.874793289342182e-15` -- three significant figures onto the addendum's
    own `2.8e-15`, on the exact corner (`{0, pi}`, integer windings) that
    `tests/arm_smprime` and `tests/arm_phase` measure their sign results on
    (PART 3). The bar below is a full decade above the measured value on
    purpose (MISTAKES.md M-2: a bar chosen after seeing the number is not a
    bar), so this stays green if the next seed lands anywhere near this one
    and fails loudly only on a real regression.
    """
    g = torch.Generator().manual_seed(0)
    n = 32
    m = torch.rand(n, generator=g, dtype=DT) * 0.999 + 1e-4
    bits = torch.randint(0, 2, (n,), generator=g)
    theta = math.pi * bits.to(DT)
    gc = path_product(gate(m, theta))
    recon = _rephase_unsigned(m, theta)
    d = (gc - recon).abs().max().item()
    print(f"\n  PART 1 (BED-M support corner): measured {d:.6e}, addendum claims 2.8e-15")
    assert d < 3e-14, d


def test_w5_2_the_harder_case_the_affine_recurrence_with_injected_drive():
    """THE CASE BIND 5 OF `tests/arm_phase` ACTUALLY PLANTED ITS NEGATIVE ON,
    and the reconciliation the addendum asked for. `chain_label`'s recurrence
    `y_i = a_i * y_{i-1} + b_i` (complex `a`, real `b`) is NOT a bare
    path-product read -- it injects a real drive at every step -- but its
    closed form is `y_i = sum_k path_product(a)_ik * b_k`, so the SAME gauge
    identity applies to it once `b` is rephased alongside the gate.

    `drop_phase` (this repo's own planted negative, `ceq/arm_phase.py:mutate`)
    sets `theta = 0` in the GATE and leaves `b` UNTOUCHED -- it does not
    rephase the values at all -- so its measured failure (residual > 1e-6,
    fired at ~0.93 on this draw) is not a counterexample to W5.2; it is a
    different, weaker mutilation than the one the theorem describes. THIS
    TEST performs the mutilation the theorem actually describes -- drop the
    gate's phase AND rephase `b` by `exp(-i * phi)`, then restore the overall
    output phase `exp(i * phi_i)` -- and it must reproduce the label to
    float64 roundoff. Measured, not assumed: this is the test that decides
    whether "no real gate reproduces the complex label" (BIND 5's own words)
    survives as stated or only survives for the one mutilation that was
    actually tried.
    """
    from ceq import arm_phase as ap

    a, b = ap.draw(seed=15, s=8)
    y = ap.chain_label(a, b)

    m = a.abs()
    theta = torch.angle(a)
    theta[0] = 0.0                       # BOS: a[0] is poisoned nan, never read
    n = a.shape[-1]
    phi = torch.cumsum(theta, dim=0)

    b_rephased = b.to(torch.complex128) * torch.exp(-1j * phi)
    y_unsigned = torch.zeros_like(a)
    for i in range(1, n):
        y_unsigned[i] = m[i] * y_unsigned[i - 1] + b_rephased[i]
    recon = torch.exp(1j * phi) * y_unsigned

    naive_drop_phase = torch.zeros_like(a)   # BIND 5's own mutilation, for contrast
    for i in range(1, n):
        naive_drop_phase[i] = m[i] * naive_drop_phase[i - 1] + b.to(torch.complex128)[i]

    gauge_residual = float((y[1:] - recon[1:]).abs().max())
    naive_residual = float((y[1:] - naive_drop_phase[1:]).abs().max())
    print(f"\n  PART 1 (harder case): gauge-equivalent (rephased b) residual = "
          f"{gauge_residual:.6e}; BIND 5's own drop_phase (b untouched) residual = "
          f"{naive_residual:.6e}")
    assert gauge_residual < 1e-13, gauge_residual
    #: and the planted negative BIND 5 actually ran really does fire, as
    #: measured on this draw -- this is not disputing that number, only its
    #: reach.
    assert naive_residual > 1e-6, naive_residual


# ===========================================================================
# PART 2 -- the contract with ceqjepa.sector_census, owned by a different
# agent in this lane. Imports are deferred INTO each test on purpose; see the
# module docstring's RED FIRST section.
# ===========================================================================

def test_w5_1_sector_census_reports_the_asserted_dims_on_the_plain_chain():
    """UPDATED IN PLACE, mid-investigation: `ceqjepa/sector_census.py` landed
    while this file was being written (its own module docstring is dated
    against the same addendum), under names this test guessed differently
    the first time it was run (`betti`, a 3-tuple `sector_dims`) -- quoted
    verbatim in this lane's own report rather than silently patched over.
    Re-pointed at the real API, `betti_numbers` / `sector_dims`'s actual
    4-tuple (tree, curl, cycle, evidence), so this now VERIFIES the landed
    W5.1 implementation against an independent chain-arithmetic route
    instead of staying red on a naming mismatch that is no longer news."""
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)]
    beta0, beta1 = sector_census.betti_numbers(n, edges)
    assert (beta0, beta1) == (1, 0)
    tree_dim, curl_dim, cycle_dim, evidence = sector_census.sector_dims(n, edges)
    assert (tree_dim, curl_dim, cycle_dim) == (n - beta0, len(edges) - n + beta0 - beta1, beta1)
    assert evidence["ortho_residual"] < 1e-6


def test_w5_1_must_fire_reported_beta1_matches_the_projector_rank():
    """Re-pointed at `betti_numbers` for the same reason as the test above."""
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)]
    _, before = sector_census.betti_numbers(n, edges)
    _, after = sector_census.betti_numbers(n, edges + [(0, n - 1)])
    assert after - before == 1


def test_w5_3_holonomy_certificate_prints_a_generator_per_cycle():
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    theta = torch.zeros(len(edges), dtype=DT)
    theta[-1] = math.pi                              # planted holonomy pi
    cert = sector_census.holonomy(n, edges, theta)
    assert len(cert["generators"]) == 1
    assert abs(cert["H_c"][0] % (2 * math.pi) - math.pi) < 1e-9


def test_w5_3_must_fire_a_planted_holonomy_pi_reads_i_at_its_closed_form():
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    theta = torch.zeros(len(edges), dtype=DT)
    theta[-1] = math.pi
    interference = sector_census.interference_fraction(n, edges, theta)
    assert all(abs(v - 1.0) < 1e-9 for v in interference), interference


def test_w5_3_must_fire_a_flat_connection_reads_i_zero_everywhere():
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    theta = torch.zeros(len(edges), dtype=DT)         # every H_c == 0
    interference = sector_census.interference_fraction(n, edges, theta)
    assert all(v == 0.0 for v in interference), interference


def test_w5_4_bed_admission_law_refuses_a_beta1_zero_interference_claim():
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)]         # beta_1 = 0
    with pytest.raises(sector_census.CertificateRefused):
        sector_census.assert_bed_admissible(n, edges, claim="interference")


def test_w5_4_bed_admission_law_admits_a_beta1_one_bed():
    from ceqjepa import sector_census

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]  # beta_1 = 1
    sector_census.assert_bed_admissible(n, edges, claim="interference")  # must not raise


def test_w5_1_landed_but_w5_3_and_w5_4_have_not_measured_against_the_sibling_file():
    """WHAT THIS LANE'S OWN SIBLING FILE ALREADY DOES AND DOES NOT COVER,
    measured rather than assumed from its docstring. `ceqjepa/sector_census.py`
    exists and ships `betti_numbers`, `sector_dims`, `plant_extra_cycle` and
    `must_fire_planted_cycle` -- W5.1 in full, with a dual-route check
    (union-find/cyclomatic against incidence-matrix SVD) neither W5.1 nor this
    file asked for but is a strictly stronger census than the addendum
    specified. It ships NOTHING named `holonomy`, `interference_fraction`,
    `CertificateRefused` or `assert_bed_admissible` -- W5.3's certificate and
    W5.4's bed-admission law are not yet in this file, which is why PART 2's
    five other tests above are still RED at call time (AttributeError, not
    ModuleNotFoundError -- the module imports fine now)."""
    from ceqjepa import sector_census

    w5_1_shipped = {"betti_numbers", "sector_dims", "plant_extra_cycle",
                    "must_fire_planted_cycle"}
    w5_3_and_4_missing = {"holonomy", "interference_fraction",
                          "CertificateRefused", "assert_bed_admissible"}
    assert w5_1_shipped <= set(dir(sector_census))
    assert not (w5_3_and_4_missing & set(dir(sector_census))), (
        "W5.3/W5.4 landed -- PART 2's other tests should now be re-checked, "
        "not skipped, and this test's own claim is stale")


def test_the_beds_this_repo_actually_ships_are_not_all_chains():
    """THE PREMISE CHECK. W5.2's consequence ('no chain bed can test signs or
    interference') only reaches beds this repo's OWN propagation graphs put at
    beta_1 = 0. The sibling file's `run_all_beds()` measures every tracked bed
    against REAL generated data (a real BED-M sample, a real BED-K kernel
    matrix, a real BED-1 landscape build) and NOT every one of them lands
    there: BED-K's power-law kernel and BED-1's landscape graph both measure
    beta_1 > 0 here. So 'no chain bed can test signs or interference' is true
    of the beds that ARE chains and is not, by itself, a statement about every
    bed this repo owns -- the ones with beta_1 > 0 are exactly where W5.4 would
    have to look for an interference bed already sitting in the tree, rather
    than one still to be built.
    """
    from ceqjepa import sector_census

    reports = {r["bed"]: r for r in sector_census.run_all_beds(seed=0)}
    chains = {"BED-M", "BED-K-delay", "BED-H"}
    non_chains = {"BED-K-powerlaw", "BED-1"}
    assert set(reports) == chains | non_chains
    for name in chains:
        assert reports[name]["beta1"] == 0, (name, reports[name]["beta1"])
    for name in non_chains:
        assert reports[name]["beta1"] > 0, (name, reports[name]["beta1"])
    print("\n  PART 2 measured (ceqjepa.sector_census.run_all_beds, seed=0):")
    for name, r in reports.items():
        print(f"    {name:16s} n={r['n']:4d} E={r['E']:5d} beta1={r['beta1']}")


# ===========================================================================
# PART 3 -- THE HARDER HALF. Which past sign/interference results sit on a
# beta_1 = 0 bed and are GAUGE-TRIVIAL BY SCOPE -- not wrong, not retracted.
# ===========================================================================

#: Phase-CARRYING sign results, measured on a graph this repo can only ever
#: build as a chain (beta_1 = 0 by construction, PART 1 above). Each verdict
#: names the mechanism, not only the label, so "gauge-trivial" reads as a
#: scope statement and not a grade.
PAST_PHASE_RESULTS = (
    dict(file="tests/arm_smprime/test_arm_smprime.py",
         test="test_bind1_the_oracle_gates_are_reachable_on_bedm_support_exactly",
         claim="theta in {pi, 0, 0} reproduces BED-M's {-1, 0, +1} support "
               "exactly (bitwise on the real part, <=1.3e-16 on the imaginary)",
         bed="BED-M (ceq/corpus.py): straight-line executed programs, "
             "SEQ_LEN=12, one negation gate, no branch -- ceq/beds/__init__.py "
             "calls it 'the chain corpus' by name",
         verdict="GAUGE-TRIVIAL BY SCOPE. theta in {0, pi} is a Z2 potential "
                 "difference read through path_product on a chain; the +-1 it "
                 "reaches is exactly what a real per-node sign, with no phase "
                 "gate at all, already reaches (PART 1's discrete-corner test). "
                 "The measured residual is not false."),
    dict(file="tests/arm_smprime/test_arm_smprime.py",
         test="test_bind1_the_label_holds_at_both_closed_endpoints",
         claim="'a.real.min() == -1.0': both signs present on a BED-M draw, "
               "label residual <= 1e-6",
         bed="BED-M, same chain",
         verdict="GAUGE-TRIVIAL BY SCOPE, same mechanism as the row above."),
    dict(file="tests/arm_smprime/test_arm_smprime.py",
         test="test_bind1_planted_negatives_all_fire",
         claim="\"drop_phase deletes the sign carrier\" -- the docstring's own "
               "name for what phase does here",
         bed="BED-M, same chain",
         verdict="GAUGE-TRIVIAL BY SCOPE, and worth stating exactly: the "
                 "docstring's name 'the sign carrier' is the claim this "
                 "theorem empties on this bed. The carrier is real (a value- or "
                 "node-side sign); phase is one representation of it, not the "
                 "only one, once the bed has no cycle to give phase anything "
                 "gauge cannot reach."),
    dict(file="tests/arm_phase/test_arm_phase.py",
         test="test_bind3_theta_in_zero_pi_reproduces_the_signed_parity_mask_exactly",
         claim="theta_k = pi*p_k reproduces chi(pscan p i - pscan p j) BITWISE "
               "(torch.equal, not a tolerance)",
         bed="a length-64 synthetic phase sequence (theta drawn directly, no "
             "BED-M corpus draw), same path_product chain structure",
         verdict="GAUGE-TRIVIAL BY SCOPE, and by the test's OWN stated algebra: "
                 "for the order-2 character chi, chi(a - b) = chi(a) * chi(b), "
                 "so the mask IS already the outer product of a per-node sign "
                 "g_i = chi(Phi_i) with itself -- a coboundary, visible without "
                 "invoking beta_1 at all, and beta_1 = 0 here besides."),
    dict(file="tests/arm_phase/test_arm_phase.py",
         test="test_bind3_the_continuous_exponential_agrees_to_machine_precision",
         claim="exp(i(Phi_i - Phi_j)) agrees with the same parity mask to <1e-13",
         bed="same length-64 chain",
         verdict="GAUGE-TRIVIAL BY SCOPE, same mechanism as the row above."),
)

#: Phase-carrying results whose relabeling is NOT a clean strike -- the
#: theorem's own compensating construction (rephase the VALUES, not only drop
#: the gate's phase) was never the mutilation actually tried. PART 1's
#: `test_w5_2_the_harder_case_...` test resolves this one by direct
#: measurement rather than leaving it a judgment call.
AMBIGUOUS_PHASE_RESULTS_RESOLVED_IN_PART_1 = (
    dict(file="tests/arm_phase/test_arm_phase.py",
         test="test_bind5_the_oracle_setting_reproduces_the_complex_chain_label",
         claim="the complex recurrence label is reproduced to 9.155e-16",
         bed="the same chain, but via chain_label's AFFINE recurrence "
             "(y_i = a_i*y_{i-1} + b_i, a real drive injected every step) "
             "rather than a bare path-product read",
         resolution="NOT STRUCK OUTRIGHT. This number is a straightforward "
                    "measurement and unaffected either way. What needed "
                    "checking was the paired planted negative's implicit "
                    "reach (next row)."),
    dict(file="tests/arm_phase/test_arm_phase.py",
         test="test_bind5_planted_negatives_all_fire",
         claim="\"the complex label is not reproduced by any real one\" -- "
               "the docstring's own wording for the drop_phase case",
         bed="same chain, affine recurrence",
         resolution="GAUGE-TRIVIAL BY SCOPE, RESOLVED BY MEASUREMENT rather "
                    "than by category. drop_phase (theta -> 0 in the GATE, "
                    "b left UNTOUCHED) is not the gauge transform W5.2 "
                    "describes -- it drops the phase without rephasing the "
                    "values -- and PART 1's harder-case test shows the "
                    "construction the theorem actually describes (drop the "
                    "gate phase, rephase b by exp(-i*phi), restore "
                    "exp(i*phi_i) at the output) reproduces the SAME label to "
                    "5.5e-16. So 'not reproduced by any real one' is true only "
                    "for the one mutilation tried, not for real constructions "
                    "in general; the measured drop_phase residual (~0.93 on "
                    "this draw) stays correct and is not disputed."),
)

#: Explicitly NOT relabeled: sign results with no U(1) phase connection at
#: all, so W5.2 (a theorem about the PHASE gate) has nothing to say about
#: them. Named here so the exclusion is checked, not assumed.
OUT_OF_SCOPE_REAL_SIGNED_RESULTS = (
    dict(file="tests/foreman/test_sign_floor.py",
         reason="A = rho*(softmax(w) - lam*softmax(-w))/(1+lam) is REAL "
                "throughout: no theta, no complex gate, no U(1) connection. "
                "W5.2 proves a phase gate gauge-trivial on a tree; a "
                "difference of two softmaxes is not a phase gate."),
    dict(file="tests/cameron/test_the_signed_arm_is_signed_on_this_task.py",
         reason="Same real operator family. Its OWN gauge concept is "
                "switching, D = diag(+-1) acting on a real matrix by "
                "conjugation (Zaslavsky frustration) -- a discrete Z2 group, "
                "not the continuous U(1) phase gauge W5.2 proves trivial on "
                "trees. The two 'gauge's share a word and nothing else; this "
                "file's frustration statistic is unaffected by W5.2."),
    dict(file="tests/chase/test_signed_operator_trainability.py",
         reason="Same real operator family (A = rho*w/sum|w|, backward-pass "
                "hazards), no phase of any kind."),
    dict(file="tests/arm_pl/test_arm_pl.py",
         test="test_bind3_the_dag_resolvent_matches_brute_force_path_sums",
         reason="A genuine multi-route DAG (density up to 1.0 at n=9, i.e. "
                "the COMPLETE lower-triangular DAG -- beta_1 = C(9,2)-9+1 = 28 "
                "as an undirected graph, emphatically NOT beta_1 = 0) -- but "
                "the edge gates are drawn from U(lo, hi) with lo, hi > 0: "
                "real, non-negative, no phase. It cross-checks an LU solve "
                "against brute-force path enumeration and makes no sign or "
                "interference claim. Named to show that multi-route / "
                "beta_1 >= 1 structure alone does not make a result an "
                "interference claim, and that no genuine interference claim "
                "exists yet in this repo to relabel (see the test below)."),
)


def test_every_relabeled_phase_result_is_cited_at_a_test_that_still_exists():
    """The relabel names specific test ids. If a file moved or a test was
    renamed since this row was written, that is caught here rather than
    trusted."""
    for row in PAST_PHASE_RESULTS + AMBIGUOUS_PHASE_RESULTS_RESOLVED_IN_PART_1:
        path = ROOT / row["file"]
        assert path.exists(), f"{row['file']} no longer exists"
        src = path.read_text(encoding="utf-8")
        name = row["test"]
        assert re.search(rf"def {re.escape(name)}\b", src), \
            f"{name!r} not found in {row['file']}"


def test_every_excluded_real_signed_result_still_exists():
    for row in OUT_OF_SCOPE_REAL_SIGNED_RESULTS:
        path = ROOT / row["file"]
        assert path.exists(), f"{row['file']} no longer exists"
        assert row["reason"], "an exclusion with no stated reason is an assumption"
        test_name = row.get("test")
        if test_name:
            src = path.read_text(encoding="utf-8")
            assert re.search(rf"def {re.escape(test_name)}\b", src), \
                f"{test_name!r} not found in {row['file']}"


def test_no_existing_interference_result_is_being_relabeled_because_none_exists():
    """W5's own KILLS list an interference bed without beta_1 printed as void.
    This repo currently ships path_product on a 1D sequence only -- one route
    between any two positions by construction (PART 1's first test) -- so no
    existing test combines two routes with a phase read at all. The census
    above therefore relabels SIGN results only, and this states that as a
    checked fact rather than an implicit gap: every row's own text is
    grepped for the word, and none should be describing a two-route
    combination as its claim."""
    for row in PAST_PHASE_RESULTS + AMBIGUOUS_PHASE_RESULTS_RESOLVED_IN_PART_1:
        assert "two route" not in row["claim"].lower()
        assert "two path" not in row["claim"].lower()
