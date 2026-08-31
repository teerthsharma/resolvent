"""WATSON -- the K1 gate for VGPE. If these fail, nothing else in the round runs.

FOUR CHECKS, EACH PAIRED WITH THE RED IT EXISTS TO CATCH. A bind that has never
been shown to fail is a bind whose tolerance is untested, so every green here is
adjacent to a test that plants the specific defect and requires the same
machinery to report it:

  G3  RoPE recovery                 <- `test_a_perturbed_theta_breaks_...`
      the theta warp itself         <- `test_the_naive_warp_doubles_the_angle`
  K1b orthogonality at depth 512    <- the two additive controls, in the SAME test
  K1c tree round trip is identity   <- `test_a_non_orthogonal_connection_...`
      loop deviation is holonomy    <- the abelian collapse, which kills it
  K5  conjugated twin invariance    <- `test_the_wilson_probe_is_not_vacuous`
                                       and `test_a_coordinate_probe_moves_...`

WHAT THE TOLERANCES ARE AND WHERE THEY CAME FROM. Every bar below was set at
1e-12 BEFORE the runs, and every measured value is quoted in the test that owns
it so a reader can see the margin rather than trust the bar. The margins as
measured on this box, float64, torch 2.5.1+cu121:

    RoPE recovery       6.217249e-14   worst over seeds 0-3, bar 1e-12
    depth-512 norm      3.220e-15      |dev from 1|,        bar 1e-12
    tree round trip     2.286e-15      ||rho - I||_2,       bar 1e-12
    twin spectrum       7.105e-15      max |tr diff|,       bar 1e-12

THE ROPE BIND IS NOT BITWISE AND THIS FILE DOES NOT SAY IT IS. 0 of 4 instances
are bitwise identical. RoPE forms R(m) from cos/sin at angle theta_f * m in one
operation; VGPE forms the same rotation as the path-ordered product of m factors.
Those are two different roundings of one number and no amount of care makes them
the same bits. The bind is therefore a TOLERANCE result: 6.217249e-14 worst
absolute (seed 0, whose score matrix reaches 22.0735) and 3.253493e-15 worst
relative (seed 2). Calling that "exact" would be the kind of sentence this
repository has been struck for.

EVERY BIND HERE WAS RUN RED BEFORE IT WAS READ GREEN, and not only through the
paired RED tests: four mutations were planted in `scale/vgpe.py` and the suite
re-run against each, at commit 74e5590 + this working tree, python 3.11.9 /
pytest 9.0.3 / torch 2.5.1+cu121. The failures land exactly where the mutation
reaches and nowhere else, which is the part worth recording:

    M1  rope_generator drops the tan(theta/2) warp   2 failed,  9 passed
    M2  cayley -> I + Om, connection leaves SO(d)    5 failed,  6 passed
    M3  wilson_spectrum reads rho[0,0], not the tr   1 failed, 10 passed
    M4  abelian_collapse returns its input           1 failed, 10 passed
    --  unmutated                                    0 failed, 11 passed

M3 is the informative one. Reading a coordinate instead of the trace still
separates two different gauges, so `test_the_wilson_probe_is_not_vacuous` stays
green under it; only the twin test notices. That is the whole reason both halves
of K5 are needed and neither alone would do.

Measured numbers are printed as well as asserted; `pytest -s` shows them.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale import vgpe                                             # noqa: E402

S, D, SEEDS = 64, 16, (0, 1, 2, 3)
BAR = 1e-12                 # every bind below, fixed before the runs
PERTURB = 1e-9              # the RED's relative kick to theta_0


def _gauge():
    """The battery gauge, taken from the module so the two cannot drift."""
    return vgpe.default_gauge(D)


# ------------------------------------------------ 1. the theta mapping (G3)

def test_the_naive_warp_doubles_the_angle():
    """The RED for the mapping, and the reason the recovery bind can fail at all.

    Cayley(t J) = Rot(2 arctan t). Feeding theta_f straight in as the Cayley
    parameter is the natural mistake and it is NOT a small one: 2 arctan t = 2t
    + O(t^3), so the delivered angle is about double the intended one, and at
    RoPE's theta_0 = 1.0 it is 1.570796 against 1.0. If this test could pass
    with the naive parameterisation, the recovery bind below would be measuring
    nothing.
    """
    th0 = float(vgpe.rope_thetas(D)[0])
    naive = vgpe.cayley_angle(th0)
    print(f"\n  naive Cayley parameter theta_0={th0:.6f} -> angle "
          f"{naive:.6f} rad, overshoot {naive - th0:.6e}")
    assert abs(naive - th0) > 0.5, (naive, th0)
    # and the warp that fixes it, checked as an angle rather than a matrix
    import math
    assert abs(vgpe.cayley_angle(math.tan(th0 / 2.0)) - th0) < 1e-15


def test_the_warped_generator_is_the_rope_rotation_block_by_block():
    """Om = (+)_f tan(theta_f/2) J has Cayley image (+)_f Rot(theta_f), exactly.

    Checked per block against cos/sin so that a failure names the frequency
    rather than reporting one matrix norm for the whole d.
    """
    th = vgpe.rope_thetas(D)
    u = vgpe.cayley(vgpe.rope_generator(th))
    worst = 0.0
    for f in range(D // 2):
        blk = u[2 * f:2 * f + 2, 2 * f:2 * f + 2]
        c, s = torch.cos(th[f]), torch.sin(th[f])
        ref = torch.tensor([[c, -s], [s, c]], dtype=vgpe.DTYPE)
        worst = max(worst, float((blk - ref).abs().max()))
    # off-block entries must be exactly zero, not merely small: the generator
    # only ever wrote the two diagonal-adjacent entries per block.
    mask = torch.ones(D, D, dtype=torch.bool)
    for f in range(D // 2):
        mask[2 * f:2 * f + 2, 2 * f:2 * f + 2] = False
    print(f"  per-block max |U - Rot(theta_f)| = {worst:.6e}")
    assert worst < BAR, worst
    assert float(u[mask].abs().max()) == 0.0, float(u[mask].abs().max())


# ---------------------------------------- 2. the RoPE recovery bind and RED

def test_the_rope_recovery_bind_is_a_tolerance_not_a_bitwise_identity():
    """G3: on a CHAIN with ONE action type, VGPE reproduces the RoPE score matrix.

    Asserted on the whole s x s matrix, both directions of offset, and reported
    as what it is. `bitwise` is checked and recorded as FALSE rather than
    quietly omitted, because the round asked whether the equality is bitwise and
    the honest answer is no.
    """
    worst, n_bitwise, rel = 0.0, 0, 0.0
    for sd in SEEDS:
        dev, scale, bitwise = vgpe.rope_recovery(S, D, sd)
        worst, rel = max(worst, dev), max(rel, dev / scale)
        n_bitwise += int(bitwise)
        print(f"  seed {sd}: max|VGPE - RoPE| = {dev:.6e}  max|RoPE| = "
              f"{scale:.4f}  bitwise = {bitwise}")
    print(f"  worst absolute {worst:.6e}, worst relative {rel:.6e}, "
          f"bitwise {n_bitwise}/{len(SEEDS)}")
    assert worst < BAR, worst
    assert n_bitwise == 0, "bitwise equality appeared -- update the docstrings"


def test_a_perturbed_theta_breaks_the_rope_recovery_bind():
    """The RED half. A 1e-9 relative kick to theta_0 MUST break the bind.

    Same geometry, same seed, one number moved. If the bar could absorb this the
    green above would be a statement about float noise rather than about the
    algebra. Measured margin: the smallest RED deviation over the four seeds is
    3.415882e-07, a factor 5.494e+06 above the worst clean deviation.
    """
    red_min = float("inf")
    for sd in SEEDS:
        red, _, _ = vgpe.rope_recovery(S, D, sd, perturb=PERTURB)
        red_min = min(red_min, red)
        print(f"  seed {sd}: theta_0 *= 1+{PERTURB} -> max|dev| = {red:.6e}")
    print(f"  smallest RED deviation {red_min:.6e}, bar {BAR:.0e}")
    assert red_min > BAR, red_min


# ------------------------------------------- 3. orthogonality at depth 512

def test_orthogonality_holds_at_depth_512_and_the_additive_controls_do_not():
    """U orthogonal => ||rho(path)||_2 = 1 at any depth, with the control beside it.

    The controls are in THIS test rather than a neighbouring one so the
    comparison is a measurement and not an assertion: same 512-letter walk, same
    generators for the matched control, three numbers off one run.

    Measured, float64, depth 512:
        VGPE  ||rho||_2                      0.999999999999997  (dev 3.220e-15)
              ||rho^T rho - I||_2            3.402e-14
        CONTROL matched matrix prod(I + Om)  2.216254e+51
        CONTROL additive vector, 4 letters   213.535932
        CONTROL additive vector, indep dirs  19.068710 mean of 8, sqrt(512)
                                             = 22.627417 is the RMS the round
                                             spec's 22.6x quotes
    "norm 1.000000" is true to six decimal places and false at the fifteenth;
    the measured deviation is 3.220e-15 and is reported rather than rounded away.
    """
    om, us = _gauge()
    word = vgpe.path_word(vgpe.DEPTH, vgpe.N_ACTIONS, vgpe.WALK_SEED)
    r = vgpe.rho(us, word)
    n2 = float(torch.linalg.matrix_norm(r, ord=2))
    ortho = float(torch.linalg.matrix_norm(
        r.transpose(0, 1) @ r - torch.eye(D, dtype=vgpe.DTYPE), ord=2))

    add_m = float(torch.linalg.matrix_norm(
        vgpe.rho(vgpe.additive_matrix(om), word), ord=2))
    add_v = vgpe.additive_code_norm(
        vgpe.unit_vectors(vgpe.N_ACTIONS, D, 11), word)

    print(f"\n  VGPE ||rho||_2 = {n2:.15f}  |dev| = {abs(n2 - 1.0):.3e}")
    print(f"  VGPE ||rho^T rho - I||_2 = {ortho:.3e}")
    print(f"  CONTROL matched matrix ||prod(I+Om)||_2 = {add_m:.6e}")
    print(f"  CONTROL additive vector code (4 letters) = {add_v:.6f}")

    assert abs(n2 - 1.0) < BAR, n2
    assert ortho < 1e-11, ortho
    # the controls have to actually drift, or "norm 1" is a claim about nothing
    assert add_m > 100.0, add_m
    assert add_v > 100.0, add_v


# ----------------------------------- 4. relativity: tree round trip vs loop

def test_the_tree_roundtrip_is_the_identity_and_a_loop_is_holonomy():
    """The distinction the round asked for, made EXACTLY and not by tolerance.

    The discriminator is free-group reduction, not a threshold on the defect.
    A retraced walk reduces to the empty word and must transport back to the
    identity; a genuine cycle does not reduce and its deviation is holonomy,
    which this test requires to be LARGE rather than tolerating it. A checker
    that only thresholded the defect would have to call the second one a
    failure, and a checker that only looked at the word would let a broken
    tree path through.
    """
    _, us = _gauge()
    trip = vgpe.TREE_OUT + vgpe.invert_word(vgpe.TREE_OUT)
    reduced, defect = vgpe.closure(us, trip)
    print(f"\n  TREE {trip} reduced={reduced} ||rho - I||_2 = {defect:.3e}")
    assert reduced == (), reduced
    assert defect < BAR, defect

    for lp in vgpe.LOOPS:
        rd, df = vgpe.closure(us, lp)
        print(f"  LOOP {lp} reduced={rd} ||rho - I||_2 = {df:.6f}  HOLONOMY")
        assert rd != (), lp
        assert df > 1.0, (lp, df)


def test_a_non_orthogonal_connection_breaks_the_tree_roundtrip():
    """The RED for the round trip, and it is a genuine transport error.

    Transport back along a reversed edge is U^T, which is the inverse only
    because U is orthogonal. Swap the Cayley image for the additive first-order
    step I + Om -- same generators, same walk, no longer in the group -- and the
    retraced walk does not return. The word still reduces to (), so the checker
    has no holonomy reading available and must report an error. Measured defect
    2.648163e+01 against a bar of 1e-12.
    """
    om, _ = _gauge()
    trip = vgpe.TREE_OUT + vgpe.invert_word(vgpe.TREE_OUT)
    reduced, defect = vgpe.closure(vgpe.additive_matrix(om), trip)
    print(f"\n  non-orthogonal, reduced={reduced} defect = {defect:.6e}")
    assert reduced == (), reduced
    assert defect > BAR, defect


# ------------------------------- 5. K5, the conjugated twin and its vacuity

def test_the_conjugated_twin_reads_the_same_wilson_spectrum():
    """K5: {tr rho(loop)} is invariant under U_a -> V U_a V^T for fixed V.

    Conjugation is a change of frame and nothing physical moves under it, so the
    twin's spectrum must be the same numbers. Measured max |difference| over the
    five loops: 7.105e-15, against a bar of 1e-12.
    """
    _, us = _gauge()
    v = vgpe.random_orthogonal(D, vgpe.TWIN_SEED)
    sp = vgpe.wilson_spectrum(us, vgpe.LOOPS)
    tw = vgpe.wilson_spectrum(vgpe.conjugate(us, v), vgpe.LOOPS)
    # V has to be a real frame change, or the invariance is trivial
    assert float((v @ v.transpose(0, 1)
                  - torch.eye(D, dtype=vgpe.DTYPE)).abs().max()) < 1e-13
    assert float((v - torch.eye(D, dtype=vgpe.DTYPE)).abs().max()) > 0.1
    worst = float((sp - tw).abs().max())
    for lp, x, y in zip(vgpe.LOOPS, sp, tw):
        print(f"  {lp}: {float(x):.10f} vs twin {float(y):.10f}")
    print(f"  twin max |diff| = {worst:.3e}")
    assert worst < BAR, worst


def test_the_wilson_probe_is_not_vacuous():
    """The non-vacuity half. Two genuinely different gauges MUST read differently.

    Without this, a probe that returned a constant would pass the twin test and
    prove nothing. Measured separation between the battery gauge and an
    independent draw at the same scale: 3.613597 max |trace difference|, seven
    orders above the twin's 7.105e-15.
    """
    _, us = _gauge()
    other = vgpe.cayley(vgpe.random_generators(
        vgpe.N_ACTIONS, D, vgpe.OTHER_SEED, scale=vgpe.GEN_SCALE))
    sp = vgpe.wilson_spectrum(us, vgpe.LOOPS)
    ot = vgpe.wilson_spectrum(other, vgpe.LOOPS)
    sep = float((sp - ot).abs().max())
    print(f"\n  other-gauge max |diff| = {sep:.6f}")
    assert sep > 0.1, sep


def test_a_coordinate_probe_moves_under_conjugation():
    """The RED that makes the choice of probe mean something.

    rho(loop)[0, 0] is a perfectly reasonable-looking readout and it is a
    COORDINATE, so the same connection in a different frame reports a different
    number. If this could not fire, "the Wilson loops are the invariant" would
    be an empty preference between two probes that both happened to be constant.
    """
    _, us = _gauge()
    v = vgpe.random_orthogonal(D, vgpe.TWIN_SEED)
    tw = vgpe.conjugate(us, v)
    moved = max(abs(float(vgpe.rho(us, lp)[0, 0] - vgpe.rho(tw, lp)[0, 0]))
                for lp in vgpe.LOOPS)
    print(f"\n  coordinate probe rho[0,0] max |diff| under V = {moved:.6f}")
    assert moved > 0.01, moved


# ------------------------------------ 6. the K3 dial and the abelian mode

def test_the_commutator_dial_reads_nonzero_and_the_collapse_zeroes_it():
    """||[U_a, U_b]||_F per pair, and the ablation that forces it to zero.

    The dial is a MEASUREMENT of a gauge, not a property of the architecture:
    these generators are random, and a trained gauge could read anywhere on it
    including zero. What the test fixes is that the dial can distinguish the two
    regimes at all -- 4.522729 max for the random gauge against 2.428e-16 after
    the Cartan projection -- and that the collapse really is abelian rather than
    approximately so, which is checked through the holonomy it must destroy
    (6.647e-16 max over the five loops) rather than through the commutator alone.
    """
    om, us = _gauge()
    cn = vgpe.commutator_norms(us)
    assert cn.shape == (vgpe.N_ACTIONS, vgpe.N_ACTIONS)
    assert float(cn.diagonal().abs().max()) == 0.0     # [U_a, U_a] is exactly 0
    us_ab = vgpe.cayley(vgpe.abelian_collapse(om))
    cn_ab = float(vgpe.commutator_norms(us_ab).max())
    hol_ab = max(vgpe.closure(us_ab, lp)[1] for lp in vgpe.LOOPS)
    hol_full = max(vgpe.closure(us, lp)[1] for lp in vgpe.LOOPS)
    print(f"\n  full gauge max ||[U_a, U_b]||_F = {float(cn.max()):.6f}, "
          f"max loop holonomy {hol_full:.6f}")
    print(f"  collapsed  max ||[U_a, U_b]||_F = {cn_ab:.3e}, "
          f"max loop holonomy {hol_ab:.3e}")
    assert float(cn.max()) > 1.0, float(cn.max())
    assert cn_ab < BAR, cn_ab
    assert hol_ab < BAR, hol_ab
    assert hol_full > 1.0, hol_full
