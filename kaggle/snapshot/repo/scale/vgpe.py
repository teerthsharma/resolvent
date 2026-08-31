"""VGPE -- verb-gauge positional encoding, and the K1 gate that decides it.

WHAT ROPE IS, ALGEBRAICALLY, AND WHERE ITS CEILING COMES FROM. RoPE gives token
m the block-diagonal rotation R(m) = (+)_{f=1..d/2} Rot(theta_f * m) and scores
<R(m) q, R(n) k> = <q, R(n - m) k>. That identity is not a trick: R is a group
HOMOMORPHISM rho: (Z, +) -> SO(2)^{d/2}, a unitary representation of the
TRANSLATION group. The group is abelian and singly generated, so a token's
position is one integer and relative position is one subtraction. Everything
RoPE can express about structure has to be pushed through that one integer.

WHAT VGPE CHANGES. Keep the representation, drop the group. Each verb/action
type `a` carries a learned generator Om_a in so(d) (skew-symmetric), mapped into
the group by the Cayley transform

    U_a = (I - Om_a)^-1 (I + Om_a)  in SO(d)

A token's position is its CAUSAL PATH -- a word in the free monoid over the
action alphabet -- and its encoding is the path-ordered product

    rho(path) = P prod_{a in path} U_a

Relative position between i and j is parallel transport along the connecting
path, and the attention score is <q_i, rho(path i->j) k_j>. The free monoid is
non-abelian, so `open then pour` and `pour then open` are different positions
rather than the same offset 2. RoPE is the special case: one action type, a
chain graph, one generator.

THE THETA MAPPING, WHICH IS THE PLACE THIS BIND ACTUALLY FAILS. Cayley is NOT
the exponential. In one 2x2 block,

    Cayley(t J) = 1/(1+t^2) [[1 - t^2, -2t], [2t, 1 - t^2]] = Rot(2 arctan t)

so feeding Om = theta J gives a rotation by 2*arctan(theta), NOT by theta. This
is a FIRST-ORDER disagreement, not a cubic one: 2 arctan t = 2t - 2t^3/3 + ...,
so the naive parameterisation delivers roughly DOUBLE the intended angle. At
RoPE's own theta_0 = 1.0 it gives 1.570796 rad against 1.0, an overshoot of
5.707963e-01 rad -- 32.7 degrees, compounding once per token. To recover RoPE
exactly the Cayley PARAMETER has to be pre-warped:

    t_f = tan(theta_f / 2)     =>     Cayley(t_f J) = Rot(theta_f)   exactly

`rope_generator` applies exactly that warp and `cayley_angle` is the one-line
check that fixes the convention. The warp needs theta_f in (-pi, pi): the Cayley
chart cannot reach a rotation with eigenvalue -1 (t would have to be infinite),
so a half-turn is outside the chart. RoPE's own theta_f = base^(-2f/d) lies in
(0, 1], so the missing antipode costs nothing here, and this file does not claim
otherwise.

WORDS, AND WHY THE TREE/LOOP DISTINCTION NEEDS NO GRAPH OBJECT. A path is a
tuple of signed 1-based letters: `+a` traverses edge a forward, `-a` traverses
it backward. Backward traversal uses U^T, which is the inverse ONLY because U is
orthogonal -- that is a load-bearing use of the group structure, not a
convenience, and `test_a_non_orthogonal_connection_breaks_the_tree_roundtrip`
is the RED that shows what happens without it. A closed walk that retraces
itself (i -> j -> i on a TREE) is a word that FREE-REDUCES TO THE EMPTY WORD,
and its transport must be the identity: any deviation there is an error. A
closed walk that does not reduce is a genuine cycle, and its deviation is
HOLONOMY -- curvature of the connection, the thing VGPE exists to carry.
`closure` returns the reduced word beside the defect so the two can never be
confused, and a checker that calls holonomy an error is wrong in exactly the way
this distinction prevents.

THE GAUGE-INVARIANT PROBE. Coordinates are not observable: replacing every U_a
by V U_a V^T for one fixed orthogonal V is a change of frame and changes nothing
physical. The Wilson loops tr rho(loop) are invariant under it, because
conjugation passes through the whole path-ordered product and trace is a
similarity invariant. Any probe that MOVES under conjugation is measuring the
coordinate system, not the structure. K5 is that assertion, plus its non-vacuity
half: two genuinely different gauges must read DIFFERENT spectra, or the probe
proves nothing by being constant.

MEASURED NUMBERS IN THIS FILE AND IN `tests/watson/test_vgpe_binds.py` COME FROM
`python -m scale.vgpe`, float64 throughout, at s=64 d=16 seeds 0-3. float64 is
not decoration: the binds below are statements about the algebra, and in float32
the tolerance would be reporting the arithmetic rather than the mathematics.

WHAT THE BIND ACTUALLY BOUGHT, stated as the numbers rather than as a claim.
RoPE recovery is NOT bitwise -- 0/4 instances -- and holds to 6.217249e-14
absolute worst over the four seeds (seed 0, whose score matrix reaches 22.0735),
worst RELATIVE 3.253493e-15 at seed 2. Saying "exactly" here would be false: the
gap is real arithmetic, because RoPE reaches R(m) from cos/sin at angle
theta_f*m in one operation while VGPE reaches the same rotation by m successive
matrix products, and those are different roundings of the same number. The
paired RED (theta_0 scaled by 1 + 1e-9, VGPE side only) never reads below
3.415882e-07, a factor 5.494e+06 above the clean worst, so the tolerance is
nowhere near wide enough to swallow a perturbation even that small.

NOTHING HERE IS TRAINED. Every generator in this file is random-init or
hand-built. No sentence in it is evidence about a learned gauge.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

DTYPE = torch.float64

#: The 2x2 skew generator. so(2) is one-dimensional and this spans it.
J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE)


# --------------------------------------------------------- the group itself

def cayley(omega: torch.Tensor) -> torch.Tensor:
    """U = (I - Om)^-1 (I + Om), batched over any leading dims.

    Om skew puts U in SO(d): (I + Om)^T = I - Om, and (I - Om)^-1 commutes with
    (I + Om) because both are polynomials in Om, so U^T U = I falls out without
    an appeal to sampling. det U = +1 because Cayley's image is the connected
    component containing I (Om = 0 gives U = I).
    """
    eye = torch.eye(omega.shape[-1], dtype=omega.dtype, device=omega.device)
    return torch.linalg.solve(eye - omega, eye + omega)


def skew(m: torch.Tensor) -> torch.Tensor:
    """Project onto so(d). (M - M^T)/2, the orthogonal projection in Frobenius."""
    return (m - m.transpose(-2, -1)) / 2.0


def random_generators(n_actions: int, d: int, seed: int,
                      scale: float = 1.0) -> torch.Tensor:
    """`n_actions` random skew generators, [A, d, d]. Stands in for learning."""
    g = torch.Generator().manual_seed(seed)
    return skew(torch.randn(n_actions, d, d, generator=g, dtype=DTYPE) * scale)


def random_orthogonal(d: int, seed: int) -> torch.Tensor:
    """A fixed random frame change V, for the conjugated twin."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DTYPE))
    return q


def conjugate(us: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The gauge transform U_a -> V U_a V^T. Same connection, different frame."""
    return v @ us @ v.transpose(-2, -1)


# ------------------------------------------------------------ RoPE, exactly

def rope_thetas(d: int, base: float = 10000.0) -> torch.Tensor:
    """theta_f = base^(-2f/d), f = 0 .. d/2 - 1. The published RoPE schedule."""
    f = torch.arange(0, d // 2, dtype=DTYPE)
    return base ** (-2.0 * f / d)


def cayley_angle(t: float) -> float:
    """The convention fixer: Cayley(t J) is Rot(2 arctan t), not Rot(t)."""
    return 2.0 * math.atan(t)


def rope_generator(thetas: torch.Tensor) -> torch.Tensor:
    """Om = (+)_f tan(theta_f / 2) J, the generator whose Cayley image IS R(1).

    tan(theta_f / 2) and NOT theta_f. See the module header: Cayley(t J) rotates
    by 2 arctan t, so the parameter has to be pre-warped by the inverse of that.
    Feeding theta_f straight in is the natural mistake and it puts the recovered
    angle at 2 arctan(theta_f) -- 1.570796 rad instead of 1.0 at f = 0, an
    overshoot of 5.707963e-01 rad, and roughly a factor of two for small theta.
    """
    d = 2 * int(thetas.numel())
    om = torch.zeros(d, d, dtype=DTYPE)
    t = torch.tan(thetas / 2.0)
    f = torch.arange(0, d, 2)
    om[f, f + 1] = -t
    om[f + 1, f] = t
    return om


def rope_scores(q: torch.Tensor, k: torch.Tensor,
                thetas: torch.Tensor) -> torch.Tensor:
    """The RoPE baseline: S[i, j] = <R(i) q_i, R(j) k_j>, rotated pairwise.

    Written the way RoPE is actually implemented -- cos/sin on the interleaved
    pairs, never forming R(m) -- because the bind's claim is that VGPE
    reproduces the code RoPE runs, not a matrix rewrite of it. That choice is
    also what puts the recovery at 6.217e-14 rather than at 0: the two paths
    reach the same rotation by different arithmetic.
    """
    m = torch.arange(q.shape[0], dtype=q.dtype).unsqueeze(-1)
    ang = m * thetas.unsqueeze(0)
    c, s = torch.cos(ang), torch.sin(ang)

    def rot(x):
        xe, xo = x[:, 0::2], x[:, 1::2]
        return torch.stack([xe * c - xo * s, xe * s + xo * c], -1).flatten(1)

    return rot(q) @ rot(k).transpose(0, 1)


# ------------------------------------------------- paths and their transport

def rho(us: torch.Tensor, word) -> torch.Tensor:
    """P prod_{a in word} U_a. Signed 1-based letters; -a traverses edge a back.

    The backward step is U^T. That is the inverse BECAUSE U is orthogonal, and
    the whole relativity claim rests on it -- a non-orthogonal connection
    transported back along its own path does not return.
    """
    d = us.shape[-1]
    out = torch.eye(d, dtype=us.dtype, device=us.device)
    for lt in word:
        u = us[abs(lt) - 1]
        out = out @ (u if lt > 0 else u.transpose(-2, -1))
    return out


def reduce_word(word) -> tuple:
    """Free-GROUP reduction: cancel adjacent a, -a. Empty <=> retraced walk.

    This is the tree/loop discriminator and it is exact, not a tolerance. On a
    tree the walk i -> j -> i retraces itself letter for letter, so it reduces to
    (). A walk around a genuine cycle -- (1, 2, -1, -2), say -- has nothing to
    cancel and stays non-empty.
    """
    out: list[int] = []
    for lt in word:
        if out and out[-1] == -lt:
            out.pop()
        else:
            out.append(lt)
    return tuple(out)


def closure(us: torch.Tensor, word):
    """(reduced_word, ||rho(word) - I||_2) for a CLOSED walk.

    The caller must read both. `reduced == ()` means the walk retraces itself and
    the defect is an ERROR with no other reading. `reduced != ()` means a genuine
    loop and the defect is HOLONOMY, which is the signal, not a fault. Returning
    the defect alone would make those two indistinguishable, which is the bug
    this signature exists to prevent.
    """
    d = us.shape[-1]
    m = rho(us, word) - torch.eye(d, dtype=us.dtype, device=us.device)
    return reduce_word(word), float(torch.linalg.matrix_norm(m, ord=2))


def invert_word(word) -> tuple:
    """The return leg: reverse the walk and flip every letter."""
    return tuple(-lt for lt in reversed(word))


def path_word(length: int, n_actions: int, seed: int) -> tuple:
    """A random forward walk of `length` letters over `n_actions` edges."""
    g = torch.Generator().manual_seed(seed)
    return tuple(
        int(x) + 1 for x in torch.randint(0, n_actions, (length,), generator=g))


def chain_scores(q: torch.Tensor, k: torch.Tensor,
                 u: torch.Tensor) -> torch.Tensor:
    """VGPE scores on a CHAIN with ONE action type: S[i,j] = <q_i, rho(i->j) k_j>.

    The chain path i -> j is the letter repeated |j - i| times, forward for
    j > i and backward for j < i, so rho(i -> j) = U^(j-i) and the powers are
    built by repeated multiplication -- literally the path-ordered product, not
    a closed form substituted for it. `torch.matrix_power` would be a different
    (and cheaper) rounding and would weaken the bind into a statement about
    binary exponentiation.
    """
    s, d = q.shape
    pows = [torch.eye(d, dtype=q.dtype, device=q.device)]
    for _ in range(s - 1):
        pows.append(pows[-1] @ u)
    out = torch.zeros(s, s, dtype=q.dtype, device=q.device)
    for off in range(-(s - 1), s):
        r = pows[off] if off >= 0 else pows[-off].transpose(0, 1)
        i = torch.arange(max(0, -off), min(s, s - off))
        out[i, i + off] = (q[i] * (k[i + off] @ r.transpose(0, 1))).sum(-1)
    return out


# ------------------------------------------------------- the invariant probe

def wilson_spectrum(us: torch.Tensor, loops) -> torch.Tensor:
    """{tr rho(loop)} over a family of loops. Invariant under U -> V U V^T.

    tr(V rho V^T) = tr(rho) because conjugation passes through the whole
    path-ordered product -- the V^T V pairs between consecutive letters cancel --
    and trace is a similarity invariant. A probe that read the entries of rho
    instead would move under V and would be reporting the frame.
    """
    return torch.stack([torch.trace(rho(us, w)) for w in loops])


def commutator_norms(us: torch.Tensor) -> torch.Tensor:
    """||U_a U_b - U_b U_a||_F for every ordered pair. [A, A]. The K3 dial.

    Zero everywhere is the abelian collapse, which is RoPE's algebra. Anything
    else is the amount of structure the free monoid is actually carrying, and it
    is a MEASUREMENT of a gauge rather than an architectural claim: this file
    reports it for random generators only, and a trained gauge could read
    anywhere on it including zero.
    """
    c = us.unsqueeze(1) @ us.unsqueeze(0) - us.unsqueeze(0) @ us.unsqueeze(1)
    return c.flatten(2).norm(dim=-1)


def abelian_collapse(omegas: torch.Tensor) -> torch.Tensor:
    """Project every generator onto the standard Cartan subalgebra of so(d).

    Keeping only the block-diagonal 2x2 skew blocks leaves generators that all
    live in one maximal torus, so they commute, so their Cayley images commute,
    so the path-ordered product loses its ordering and every loop's holonomy is
    the identity. That is not "an abelian-ish version" -- it is exactly RoPE's
    algebra with a per-action frequency vector, which is what makes it the right
    ablation: the difference between it and the full gauge is the whole claim.
    """
    d = omegas.shape[-1]
    out = torch.zeros_like(omegas)
    f = torch.arange(0, d - 1, 2)
    out[..., f, f + 1] = omegas[..., f, f + 1]
    out[..., f + 1, f] = -omegas[..., f, f + 1]
    return out


# ------------------------------------------------------------- the controls

def additive_matrix(omegas: torch.Tensor) -> torch.Tensor:
    """I + Om, the MATCHED control: same generators, first-order composition.

    This is what the connection looks like without the group -- Euler's step
    instead of Cayley's. It is the same object type as U (a d x d matrix acting
    on the same q, k) so the two operator norms compare like with like. It is
    not orthogonal, ||I + Om||_2 = sqrt(1 + ||Om||_2^2) > 1, and the excess
    compounds once per step.
    """
    eye = torch.eye(omegas.shape[-1], dtype=omegas.dtype, device=omegas.device)
    return eye + omegas


def additive_code_norm(vs: torch.Tensor, word) -> float:
    """||sum_{a in word} +-v_a||, the VECTOR additive positional code.

    The other kind of control, and the one the round spec's 22.6x refers to: a
    code that ADDS a per-action embedding instead of composing a group element.
    With INDEPENDENT unit directions its norm is a random walk, E||e||^2 = depth
    exactly, so the spec's 22.6 is sqrt(512) = 22.627417 read as an RMS. The
    MEAN of ||e|| sits below that by Jensen and by the d = 16 chi spread:
    measured 19.068710 over 8 draws, range [13.915, 25.590]. Over a SMALL
    alphabet the repeated letters add coherently instead and the drift is far
    worse -- 213.535932 for a 4-letter alphabet at depth 512 -- so both are
    reported rather than whichever flatters the comparison. Note the two norms
    are of different objects, an operator norm against a vector norm; what is
    being compared is only whether depth moves the magnitude at all.
    """
    e = torch.zeros(vs.shape[-1], dtype=vs.dtype, device=vs.device)
    for lt in word:
        e = e + (vs[abs(lt) - 1] if lt > 0 else -vs[abs(lt) - 1])
    return float(e.norm())


def unit_vectors(n: int, d: int, seed: int) -> torch.Tensor:
    """n unit directions, for the additive control."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=DTYPE)
    return v / v.norm(dim=-1, keepdim=True)


# ------------------------------------------------------------ the K1 battery

#: Non-reducible closed words. Each is a genuine cycle in the causal graph, so
#: each carries holonomy; the first three are commutators, which are the
#: discrete stand-in for the curvature two-form.
LOOPS = [(1, 2, -1, -2), (1, 3, -1, -3), (2, 3, -2, -3),
         (1, 2, 3, -1, -2, -3), (1, 1, 2, -1, -1, -2)]

#: A walk out along a tree. Followed by `invert_word` of itself it free-reduces
#: to (), so that round trip MUST be the identity; no holonomy reading is
#: available for it.
TREE_OUT = (1, 3, 2, 3, 4)

DEPTH = 512

#: Fixed before the runs, and used by both this module and the test file so the
#: two cannot drift apart.
GEN_SEED, GEN_SCALE, N_ACTIONS, WALK_SEED = 20260830, 0.25, 4, 7
TWIN_SEED, OTHER_SEED = 4242, 99999


def _geometry(s: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(s, d, generator=g, dtype=DTYPE),
            torch.randn(s, d, generator=g, dtype=DTYPE))


def default_gauge(d: int):
    """(Om, U) for the standard battery gauge. One definition, two readers."""
    om = random_generators(N_ACTIONS, d, GEN_SEED, scale=GEN_SCALE)
    return om, cayley(om)


def rope_recovery(s: int, d: int, seed: int, perturb: float = 0.0):
    """(max |VGPE - RoPE| over the s x s score matrix, max |RoPE|, bitwise?).

    `perturb` scales theta_0 by (1 + perturb) on the VGPE side only. At
    perturb = 0 this is the G3-shaped bind; at perturb != 0 it is the RED that
    has to break it, drawn over the SAME instance so the two numbers are
    comparable and neither can be an artefact of the geometry.
    """
    q, k = _geometry(s, d, seed)
    th = rope_thetas(d)
    ref = rope_scores(q, k, th)
    th_v = th.clone()
    th_v[0] = th_v[0] * (1.0 + perturb)
    got = chain_scores(q, k, cayley(rope_generator(th_v)))
    return (float((got - ref).abs().max()), float(ref.abs().max()),
            bool(torch.equal(got, ref)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(4)))
    ap.add_argument("--perturb", type=float, default=1e-9)
    a = ap.parse_args()

    print("=== K1a -- RoPE RECOVERY BIND (G3 shape), float64 ===")
    print("  warp: Om = (+)_f tan(theta_f/2) J, because Cayley(tJ) = "
          "Rot(2 arctan t)")
    print(f"  {'seed':>5} {'max|VGPE-RoPE|':>16} {'max|RoPE|':>12} "
          f"{'bitwise':>9} {'RED max|dev|':>14}")
    worst, red_min, n_bitwise = 0.0, float("inf"), 0
    for sd in a.seeds:
        dev, scale, bitwise = rope_recovery(a.s, a.d, sd)
        red, _, _ = rope_recovery(a.s, a.d, sd, perturb=a.perturb)
        worst, red_min = max(worst, dev), min(red_min, red)
        n_bitwise += int(bitwise)
        print(f"  {sd:>5} {dev:>16.6e} {scale:>12.4f} {str(bitwise):>9} "
              f"{red:>14.6e}")
    print(f"  bitwise identical: {n_bitwise}/{len(a.seeds)} -- the bind is a "
          f"TOLERANCE result, not a bitwise one")
    print(f"  worst clean deviation {worst:.6e}; RED (theta_0 *= 1+{a.perturb}) "
          f"never below {red_min:.6e}  ({red_min / worst:.3e}x)")
    th0 = float(rope_thetas(a.d)[0])
    print(f"  the warp is load-bearing and NOT a small correction: feeding theta "
          f"straight in gives Rot(2 arctan theta) =")
    print(f"    {cayley_angle(th0):.6f} rad against theta_0 = {th0:.6f}, an "
          f"overshoot of {cayley_angle(th0) - th0:.6e} rad, ~2x for small theta")

    print(f"\n=== K1b -- ORTHOGONALITY AT DEPTH {DEPTH} ===")
    om, us = default_gauge(a.d)
    w = path_word(DEPTH, N_ACTIONS, WALK_SEED)
    r = rho(us, w)
    n2 = float(torch.linalg.matrix_norm(r, ord=2))
    od = float(torch.linalg.matrix_norm(
        r.transpose(0, 1) @ r - torch.eye(a.d, dtype=DTYPE), ord=2))
    add_m = float(torch.linalg.matrix_norm(rho(additive_matrix(om), w), ord=2))
    add_v = additive_code_norm(unit_vectors(N_ACTIONS, a.d, 11), w)
    # Averaged over 8 draws, because a single random walk of 512 steps has a
    # wide enough spread that one draw could not be compared with sqrt(512).
    indep = [additive_code_norm(unit_vectors(DEPTH, a.d, 12 + t),
                                tuple(range(1, DEPTH + 1))) for t in range(8)]
    add_i = sum(indep) / len(indep)
    print(f"  VGPE    ||rho||_2                         = {n2:.15f}"
          f"   (|dev from 1| = {abs(n2 - 1.0):.3e})")
    print(f"          ||rho^T rho - I||_2               = {od:.3e}")
    print(f"  CONTROL matched matrix ||prod(I+Om)||_2   = {add_m:.6e}")
    print(f"  CONTROL additive vector code, 4 letters   = {add_v:.6f}")
    print(f"  CONTROL additive vector code, indep dirs  = {add_i:.6f}"
          f"   mean of 8 draws, range [{min(indep):.3f}, {max(indep):.3f}],")
    print(f"                                              "
          f"against sqrt({DEPTH}) = {math.sqrt(DEPTH):.6f}")

    print("\n=== K1c -- TRANSPORT ROUND TRIP: tree vs loop ===")
    tree = TREE_OUT + invert_word(TREE_OUT)
    red_t, def_t = closure(us, tree)
    print(f"  TREE  {tree}")
    print(f"        reduced={red_t}  ||rho - I||_2 = {def_t:.3e}   -> "
          f"{'ERROR' if def_t > 1e-12 else 'identity, as required'}")
    for lp in LOOPS:
        rd, df = closure(us, lp)
        print(f"  LOOP  {str(lp):<22} reduced={str(rd):<24} "
              f"||rho - I||_2 = {df:.6f}  -> HOLONOMY, not an error")
    red_a, def_a = closure(additive_matrix(om), tree)
    print(f"  RED   non-orthogonal connection, SAME tree walk: reduced={red_a}"
          f"  defect = {def_a:.6e}  -> flagged ERROR")

    print("\n=== K1d -- CONJUGATED TWIN (K5), Wilson spectrum ===")
    v = random_orthogonal(a.d, TWIN_SEED)
    sp = wilson_spectrum(us, LOOPS)
    sp_tw = wilson_spectrum(conjugate(us, v), LOOPS)
    sp_ot = wilson_spectrum(
        cayley(random_generators(N_ACTIONS, a.d, OTHER_SEED, scale=GEN_SCALE)),
        LOOPS)
    print(f"  {'loop':<24} {'tr rho':>14} {'tr rho (twin)':>16} "
          f"{'|diff|':>11} {'other gauge':>14}")
    for lp, x, y, z in zip(LOOPS, sp, sp_tw, sp_ot):
        print(f"  {str(lp):<24} {float(x):>14.10f} {float(y):>16.10f} "
              f"{abs(float(x) - float(y)):>11.3e} {float(z):>14.10f}")
    print(f"  twin max |diff|        = {float((sp - sp_tw).abs().max()):.3e}"
          f"   (must be ~0: conjugation is only a frame change)")
    print(f"  other-gauge max |diff| = {float((sp - sp_ot).abs().max()):.6f}"
          f"   (must be LARGE, or the probe is vacuous)")

    print("\n=== K3 dial -- commutator norms, and the abelian collapse ===")
    cn = commutator_norms(us)
    us_ab = cayley(abelian_collapse(om))
    off = cn.numel() - cn.shape[0]
    print(f"  full gauge   max ||[U_a, U_b]||_F = {float(cn.max()):.6f}   "
          f"mean over the {off} ordered off-diagonal pairs = "
          f"{float(cn.sum()) / off:.6f}")
    print(f"  collapsed    max ||[U_a, U_b]||_F = "
          f"{float(commutator_norms(us_ab).max()):.3e}")
    print(f"  collapsed    max loop holonomy    = "
          f"{max(closure(us_ab, lp)[1] for lp in LOOPS):.3e}   "
          f"(abelian => every loop is trivial)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
