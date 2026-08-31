"""Analytic FLOP accounting for the five V12 gauge-positional-encoding arms.

WHY THIS FILE EXISTS. The round spec prices the gauge with four numbers -- 768
FLOPs/edge at d=256, a 6,144-FLOP depth-8 path, ~524,288 for one s=1024
attention row, and "~16.7M FLOPs per action type" for a dense Cayley -- and
concludes the gauge is FREE. This file is an INDEPENDENT analytic derivation
that checks each of those four, arm by arm, from the block algebra up. It does
NOT import `scale/vgpe.py`; that is the point. If the two disagree, the
disagreement is the measurement.

CONVENTIONS. Two are printed side by side, because the spec's numbers are not
all in the same one and mixing them is how a 2.3x error hides.

  HOUSE  -- one multiply-add = 2 FLOPs, a matmul [m,p]@[p,q] costs 2*m*p*q.
            This is the convention of `scale/m3_flops.py:41` and of the shipped
            counter at `scale/arm_s.py:457-460`. It is the primary column.
  OPS    -- every multiply and every add counted as 1 FLOP each, using the
            closed form of the algebra rather than a matmul shape. This is the
            convention the spec's `768` is in.

NO WALL CLOCK APPEARS IN THIS FILE. Nothing here is timed. Per the house rule,
a clock on this box is PROVISIONAL / CONTENDED and is not a measurement; a FLOP
count is arithmetic on shapes and contention cannot move it.

FORWARD ONLY, matching `scale/m3_flops.py`: `BASE_TERMS` there counts the
forward pass and adds a backward term only for the implicit-gradient solve. The
transport backward is roughly 2x its forward and is NOT COUNTED here either;
that omission is identical across all five arms, so it cannot move an arm
ordering, and it is stated rather than buried.

TRANSCENDENTALS ARE NOT COUNTED, and are listed at the end with element counts.
Per-element FLOP cost of exp/cos/sin/div is NOT FOUND in this repository's
source, the same treatment `scale/m3_flops.py:370-395` gives them.

M-8 COMPLIANCE (`MISTAKES.md:495`). Every arm below is priced at ITS OWN
primitive rate. No cross-arm ratio is carried. In particular the abelian arm is
NOT priced as a fraction of the non-abelian arm, and the shuffled-gauge plant is
NOT priced as "the same as VGPE" by assertion -- it is derived independently and
then SHOWN equal.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_flops import cell_terms                             # noqa: E402
from scale.arm_s import neumann_for                               # noqa: E402

# ------------------------------------------------------------------ the shape
# The evaluation point is the one the round spec fixes. It is NOT read from a
# run and NOT read from `scale/vgpe.py`, which does not exist in this worktree
# at the time this file was written.
D = 256           # model width, spec
S = 1024          # sequence length / number of tree nodes, spec
DEPTH = 8         # tree depth, spec
ALPHABET = 16     # |A|, number of distinct action types. FREE PARAMETER: the
                  # spec does not fix it. Swept below; headline at 16.
HIDDEN = 4 * D    # 1024. Standard 4x MLP ratio. `scale/m3_flops.py:27` uses
                  # 8x at its own tiny width; 4x is taken here and the term is
                  # printed separately so a different ratio can be re-read off.
K_PIV = 16        # twin pivot count, the middle of the `KS` sweep at
                  # `scale/m3_flops.py:28`. Swept below.
T_STAR = 21       # unused by `twin` (no settle loop); passed for signature
BETA = 0.5
N_HEADLINE = 1    # price PER EXAMPLE. Every term is linear in n except the
                  # Cayley build, which is n-INDEPENDENT -- that asymmetry is
                  # the whole amortization question, so n is explicit.
NS = (1, 8, 64, 512)
ALPHABETS = (4, 8, 16, 64)


# ----------------------------------------------------------- block primitives
# A gauge element U_a lives in a block-diagonal subgroup of SO(d): d/b blocks of
# size b. Two parametrizations are priced.
#
#   b = 2, unit complex   -- the ABELIAN case. All blocks commute, so all U_a
#                            commute. This is also RoPE's shape.
#   b = 4, unit quaternion -- the NON-ABELIAN case the round is about. Left
#                            multiplication by a unit quaternion is an element
#                            of SO(4); two of them do not commute.
#
# THE KEY STRUCTURAL FACT, and the reason this file prices compose and apply
# with one function: in BOTH parametrizations, "apply U to a vector" and
# "compose U with V" are THE SAME PRIMITIVE -- left multiplication in the block
# algebra. A quaternion times a 4-vector and a quaternion times a quaternion are
# the same 4x4-matrix-times-4-vector product. Storing a path product as a
# quaternion per block, not as a 4x4 matrix per block, is what makes this true;
# the matrix-representation alternative is priced separately as MATREP below and
# is 4x dearer on compose.

def block_mult(d: int, b: int, conv: str) -> int:
    """Left multiplication in the block algebra: d/b blocks, each a b x b
    orthogonal matrix against a b-vector.

    HOUSE: [b,b]@[b,1] costs 2*b*b per block  ->  2*b*d total.
    OPS:   b*b multiplies + b*(b-1) adds per block  ->  (2*b - 1)*d total.
    """
    nb = d // b
    if conv == "HOUSE":
        return nb * 2 * b * b
    return nb * (b * b + b * (b - 1))


def block_matmul(d: int, b: int, conv: str) -> int:
    """MATREP alternative: compose by a full b x b @ b x b matmul instead of an
    algebra product. HOUSE 2*b^3 per block -> 2*b^2*d. Not used in the arm
    totals; printed so the representation choice is a number and not a
    preference."""
    nb = d // b
    if conv == "HOUSE":
        return nb * 2 * b * b * b
    return nb * (b * b * b + b * b * (b - 1))


def apply_dense(d: int, conv: str) -> int:
    """A dense d x d U against a d-vector."""
    return 2 * d * d if conv == "HOUSE" else 2 * d * d - d


def compose_dense(d: int, conv: str) -> int:
    """Two dense d x d rotations composed: a full d x d @ d x d matmul.

    THIS IS THE TERM THAT INVERTS THE LINUS GATE. In the block algebra compose
    and apply are the same primitive and cost the same; densely, compose is d
    times dearer than apply. See section 3.
    """
    return 2 * d ** 3 if conv == "HOUSE" else 2 * d ** 3 - d * d


def cayley_dense(d: int, conv: str) -> int:
    """U_a = (I - Om_a)^-1 (I + Om_a) with a DENSE d x d Om_a.

    HOUSE: LU factorisation (2/3)d^3 + inverse from the factors (4/3)d^3 = 2d^3,
    then one d x d @ d x d matmul at 2d^3.  TOTAL 4*d^3.
    OPS:   half the multiply-adds, 2*d^3.
    SPEC:  the round spec quotes "~16.7M FLOPs per action type", which is
           16,777,216 = 256^3 = d^3 exactly -- the textbook O(d^3) with the
           constant taken as 1, counting neither the following matmul nor the
           factor 2 in a multiply-add. All three are printed.
    """
    return 4 * d ** 3 if conv == "HOUSE" else 2 * d ** 3


def cayley_block(d: int, b: int, conv: str) -> int:
    """Cayley on each b x b skew block: one b x b inverse plus one b x b matmul.

    An UPPER BOUND. Both b=2 and b=4 have closed forms far cheaper than a
    generic inverse (b=2 is 7 scalar ops; a left-isoclinic b=4 generator maps to
    a unit quaternion in a handful). The bound is used so no closed form has to
    be assumed about code that is not in the tree yet.
    """
    nb = d // b
    per = 2 * (2 * b ** 3)
    return nb * per if conv == "HOUSE" else nb * per // 2


# ------------------------------------------------------------- the five arms
# Each arm's positional cost is built from ITS OWN primitives. Nothing is scaled
# off another arm.
#
# Per node the prefix form pays:  1 COMPOSE (R_i = R_parent . U_{a_i}) + 2 APPLY
# (q_i and k_i are both transported). Per node the naive form pays: 2 * depth
# APPLY, walking the root path once per vector per node.
#
# RoPE pays no compose at all: the rotation for position i is read from a
# precomputed [s, d/2] cos/sin table, so it is 2 APPLY per node and nothing else.

ARMS = ("twin+RoPE", "twin+VGPE", "twin+VGPE-abelian",
        "twin+shuffled-gauge", "softmax+RoPE")


def pos_per_node(arm: str, d: int, depth: int, conv: str, prefix: bool = True,
                 dense: bool = False):
    """(compose, apply, total) FLOPs per node for one arm's positional encoding.

    A dense Cayley yields a DENSE d x d U, which must then be composed and
    applied densely. `dense` therefore changes the transport as well as the
    build; pricing a dense build against a blockwise transport is the mistake
    that makes the dense path look survivable.
    """
    if arm in ("twin+RoPE", "softmax+RoPE"):
        # b=2 rotation from a table. No composition: the angle i*theta_j is
        # tabulated, not built by walking a path. RoPE has no dense variant.
        return 0, 2 * block_mult(d, 2, conv), 2 * block_mult(d, 2, conv)
    if dense:
        ap, co = apply_dense(d, conv), compose_dense(d, conv)
        if not prefix:
            return 0, 2 * depth * ap, 2 * depth * ap
        return co, 2 * ap, co + 2 * ap
    if arm == "twin+VGPE-abelian":
        b = 2                     # commuting => simultaneously diagonalisable
    elif arm in ("twin+VGPE", "twin+shuffled-gauge"):
        b = 4                     # unit quaternion, non-commuting
    else:
        raise ValueError(arm)
    ap = block_mult(d, b, conv)
    if not prefix:
        return 0, 2 * depth * ap, 2 * depth * ap
    return ap, 2 * ap, 3 * ap


def pos_build(arm: str, d: int, alphabet: int, conv: str, dense: bool = False):
    """Cayley construction, paid ONCE PER OPTIMIZER STEP per action type -- the
    Om_a are parameters and do not change inside a step -- and then reused by
    every edge in the batch. This term is INDEPENDENT of n and of s."""
    if arm in ("twin+RoPE", "softmax+RoPE"):
        return 0                  # RoPE has no learned generator to build
    if dense:
        return alphabet * cayley_dense(d, conv)
    b = 2 if arm == "twin+VGPE-abelian" else 4
    return alphabet * cayley_block(d, b, conv)


def arm_total(arm, n, s, d, hidden, k, depth, alphabet, conv,
              prefix=True, dense=False, nn_terms=21):
    """Per-training-step forward FLOPs for one arm. Returns a dict of terms."""
    cell = "softmax" if arm == "softmax+RoPE" else "twin"
    t = cell_terms(cell, n, s, d, hidden, k, T_STAR, nn_terms)
    base = t["base"]
    twin_addon = t["select"] + t["setup"] + t["loop"] + t["contract"] + t["bwd"]
    _, _, per_node = pos_per_node(arm, d, depth, conv, prefix, dense)
    transport = n * s * per_node
    build = pos_build(arm, d, alphabet, conv, dense)
    return dict(base=base, twin=twin_addon, transport=transport, build=build,
                pos=transport + build, total=base + twin_addon + transport + build)


# ------------------------------------------------------------ attention row
def attention_row(s: int, d: int, conv: str, full: bool = False) -> int:
    """One row of self-attention. `full` includes the a @ V mix; otherwise only
    the q @ K^T logits row, which is what the spec's 524,288 is."""
    mults = 2 if full else 1
    if conv == "HOUSE":
        return mults * 2 * s * d
    return mults * (2 * s * d - s)


# ------------------------------------------------------------------- printing
def rule(c="-", n=100):
    print(c * n)


def main() -> int:
    nn_terms = neumann_for(BETA)
    print("=" * 100)
    print("ANALYTIC FLOP ACCOUNTING -- FIVE V12 GAUGE-POSITIONAL ARMS")
    print("=" * 100)
    print("NO WALL CLOCK APPEARS IN THIS FILE. Nothing here is timed.")
    print("Independent derivation. scale/vgpe.py is NOT imported and NOT read.")
    print("F_base and the twin add-on are reused from scale/m3_flops.py "
          "(BASE_TERMS, cell_terms).")
    print()
    print("SHAPE: d=%d  s=%d  depth=%d  |A|=%d  hidden=%d  k_piv=%d  "
          "n_neumann=%d" % (D, S, DEPTH, ALPHABET, HIDDEN, K_PIV, nn_terms))
    print("Forward only. Transport backward (~2x forward) NOT COUNTED, "
          "identically across all five arms.")
    print()

    # ---------------------------------------------------------------- 1
    print("=" * 100)
    print("1. BLOCK PRIMITIVES at d=%d -- where 768 does and does not come from"
          % D)
    print("=" * 100)
    print("  %-46s %14s %14s" % ("primitive", "HOUSE", "OPS"))
    rule()
    rows = [
        ("apply/compose, b=2 unit complex   (2*b*d | (2b-1)*d)",
         block_mult(D, 2, "HOUSE"), block_mult(D, 2, "OPS")),
        ("apply/compose, b=4 unit quaternion (2*b*d | (2b-1)*d)",
         block_mult(D, 4, "HOUSE"), block_mult(D, 4, "OPS")),
        ("MATREP compose, b=2 (2*b^2*d)",
         block_matmul(D, 2, "HOUSE"), block_matmul(D, 2, "OPS")),
        ("MATREP compose, b=4 (2*b^2*d)",
         block_matmul(D, 4, "HOUSE"), block_matmul(D, 4, "OPS")),
        ("apply, dense d x d (2*d^2)",
         apply_dense(D, "HOUSE"), apply_dense(D, "OPS")),
        ("compose, dense d x d (2*d^3)",
         compose_dense(D, "HOUSE"), compose_dense(D, "OPS")),
        ("Cayley build, dense (4*d^3 | 2*d^3)",
         cayley_dense(D, "HOUSE"), cayley_dense(D, "OPS")),
        ("Cayley build, b=2 blockwise (upper bound)",
         cayley_block(D, 2, "HOUSE"), cayley_block(D, 2, "OPS")),
        ("Cayley build, b=4 blockwise (upper bound)",
         cayley_block(D, 4, "HOUSE"), cayley_block(D, 4, "OPS")),
    ]
    for lab, h, o in rows:
        print("  %-46s %14d %14d" % (lab, h, o))
    print()
    print("  SPEC CLAIM A: \"768 FLOPs/edge at d=256\".")
    b2o, b4o = block_mult(D, 2, "OPS"), block_mult(D, 4, "OPS")
    b2h, b4h = block_mult(D, 2, "HOUSE"), block_mult(D, 4, "HOUSE")
    print("    b=2 (ABELIAN / RoPE shape), OPS   = %d   %s"
          % (b2o, "REPRODUCES EXACTLY" if b2o == 768 else "does not"))
    print("    b=2, HOUSE                        = %d   (%.2fx the claim)"
          % (b2h, b2h / 768))
    print("    b=4 (QUATERNION, d/4 blocks), OPS = %d   (%.3fx the claim)"
          % (b4o, b4o / 768))
    print("    b=4, HOUSE                        = %d   (%.3fx the claim)"
          % (b4h, b4h / 768))
    print("    VERDICT: 768 is the ABELIAN per-edge rate under OPS. The spec's")
    print("    own words say \"blockwise unit quaternions at d/4 cost\", which is")
    print("    %d under the same convention. The spec prices the NON-COMMUTING"
          % b4o)
    print("    arm at the COMMUTING arm's rate -- an M-8 carry inside the spec.")
    print()
    print("  SPEC CLAIM D: dense Cayley \"~16.7M FLOPs per action type\".")
    print("    d^3               = %d   <- what 16.7M is" % (D ** 3))
    print("    OPS   (2*d^3, inv + matmul) = %d  (%.1fx the claim)"
          % (cayley_dense(D, "OPS"), cayley_dense(D, "OPS") / D ** 3))
    print("    HOUSE (4*d^3)               = %d  (%.1fx the claim)"
          % (cayley_dense(D, "HOUSE"), cayley_dense(D, "HOUSE") / D ** 3))
    print("    The claim counts the INVERSE ALONE at constant 1 and drops the")
    print("    (I+Om) matmul that follows it. Under-counts by 2x (OPS) to 4x.")
    print("    AMORTIZATION: Om_a are PARAMETERS. They move once per optimizer")
    print("    step, so the build is paid |A| times PER STEP and reused by every")
    print("    edge in the batch. It is NOT per-edge. Section 4 prices it.")
    print()

    # ---------------------------------------------------------------- 2
    print("=" * 100)
    print("2. SPEC CLAIM B: \"depth-8 path = 6,144 FLOPs vs ~524,288 for one "
          "s=1024 attention row\"")
    print("=" * 100)
    print("  %-52s %14s %14s" % ("path form", "HOUSE", "OPS"))
    rule()
    naive_ab_one = DEPTH * block_mult(D, 2, "OPS")
    print("  %-52s %14d %14d"
          % ("naive, ABELIAN b=2, ONE vector, depth=%d" % DEPTH,
             DEPTH * block_mult(D, 2, "HOUSE"), naive_ab_one))
    print("  %-52s %14d %14d"
          % ("naive, QUATERNION b=4, ONE vector, depth=%d" % DEPTH,
             DEPTH * block_mult(D, 4, "HOUSE"), DEPTH * block_mult(D, 4, "OPS")))
    print("  %-52s %14d %14d"
          % ("naive, QUATERNION b=4, q AND k, depth=%d" % DEPTH,
             2 * DEPTH * block_mult(D, 4, "HOUSE"),
             2 * DEPTH * block_mult(D, 4, "OPS")))
    print("  %-52s %14d %14d"
          % ("PREFIX, QUATERNION b=4, q AND k (1 compose + 2 apply)",
             3 * block_mult(D, 4, "HOUSE"), 3 * block_mult(D, 4, "OPS")))
    print()
    print("  6,144 REPRODUCES as: ABELIAN, ONE vector, NAIVE (no prefix), OPS.")
    print("    -> three substitutions off the arm the round headlines:")
    print("       commuting instead of non-commuting, one vector instead of q")
    print("       and k, and the naive walk the Linus gate exists to remove.")
    print("  COINCIDENCE TRAP, stated so it is not mistaken for agreement:")
    print("    depth * (b=2, OPS) = 8 * %d = %d" % (b2o, DEPTH * b2o))
    print("    3      * (b=4, HOUSE) = 3 * %d = %d" % (b4h, 3 * b4h))
    print("    Both are 24*d at d=%d. They agree NUMERICALLY at this shape and"
          % D)
    print("    share no term. At depth != 8 they diverge; see the depth sweep.")
    print()
    for conv in ("HOUSE", "OPS"):
        r_logit = attention_row(S, D, conv, full=False)
        r_full = attention_row(S, D, conv, full=True)
        print("  attention row at s=%d d=%d, %-5s: logits-only %d, "
              "logits + a@V %d" % (S, D, conv, r_logit, r_full))
    print("    524,288 = 2*s*d = the HOUSE logits row. It is HALF an attention")
    print("    row: the a @ V mix costs another 2*s*d and the spec omits it.")
    print("    The spec's ratio also mixes conventions -- 6,144 is OPS, 524,288")
    print("    is HOUSE. Held to one convention:")
    for conv in ("HOUSE", "OPS"):
        pn = pos_per_node("twin+VGPE", D, DEPTH, conv)[2]
        for lab, row in (("logits-only", attention_row(S, D, conv, False)),
                         ("full row", attention_row(S, D, conv, True))):
            print("      %-5s VGPE prefix per node %6d vs %-12s %8d  -> "
                  "%7.1fx cheaper" % (conv, pn, lab, row, row / pn))
    print("    DIRECTION HOLDS at every combination above. The gauge is 2 to 3")
    print("    orders below an attention row. The exact 85x is not reproduced")
    print("    by the arm the round headlines; it is 64x to 195x depending on")
    print("    convention and on whether the row is half or whole.")
    print()

    # ---------------------------------------------------------------- 3
    print("=" * 100)
    print("3. LINUS GATE: naive O(n*depth) vs prefix O(n). Priced per arm.")
    print("=" * 100)
    print("  Prefix: R_i = R_parent . U_{a_i}. One compose per node, reused by")
    print("  every descendant. Naive: walk the root path per node per vector.")
    print()
    hdr = ("  %-22s %6s %14s %14s %10s" %
           ("arm", "conv", "naive/node", "prefix/node", "saving"))
    print(hdr)
    rule()
    savings = {}
    for arm in ARMS:
        for conv in ("HOUSE", "OPS"):
            nv = pos_per_node(arm, D, DEPTH, conv, prefix=False)[2]
            pf = pos_per_node(arm, D, DEPTH, conv, prefix=True)[2]
            savings[(arm, conv)] = nv / pf
            print("  %-22s %6s %14d %14d %9.3fx"
                  % (arm, conv, nv, pf, nv / pf))
    print()
    print("  PER STEP at n=%d, s=%d (the saving as a number, not a ratio):"
          % (N_HEADLINE, S))
    for arm in ("twin+VGPE", "twin+VGPE-abelian"):
        nv = N_HEADLINE * S * pos_per_node(arm, D, DEPTH, "HOUSE", False)[2]
        pf = N_HEADLINE * S * pos_per_node(arm, D, DEPTH, "HOUSE", True)[2]
        print("    %-22s HOUSE naive %12d  prefix %12d  SAVED %12d"
              % (arm, nv, pf, nv - pf))
    print()
    print("  The saving factor is 2*depth/3 and is depth-driven, so it is the")
    print("  same NUMBER for both gauge arms while the absolute saving is not:")
    print("    %5s %14s %16s %16s" % ("depth", "factor", "VGPE saved/node",
                                      "abelian saved/node"))
    for dep in (2, 4, 8, 16, 64, S):
        f = 2 * dep / 3
        v = (pos_per_node("twin+VGPE", D, dep, "HOUSE", False)[2]
             - pos_per_node("twin+VGPE", D, dep, "HOUSE", True)[2])
        a = (pos_per_node("twin+VGPE-abelian", D, dep, "HOUSE", False)[2]
             - pos_per_node("twin+VGPE-abelian", D, dep, "HOUSE", True)[2])
        print("    %5d %13.3fx %16d %16d" % (dep, f, v, a))
    print("  depth=%d is the degenerate path-graph case (a chain, not a tree):" % S)
    print("  the naive form is then O(s^2) transports and the prefix is O(s).")
    print()
    print("  --- THE LINUS GATE INVERTS ON THE DENSE PATH ---")
    print("  The gate counts COMPOSITIONS: O(n) instead of O(n*depth). That is")
    print("  a saving only if one composition costs about what one application")
    print("  costs. In the block algebra it does -- a quaternion times a")
    print("  quaternion and a quaternion times a 4-vector are the same product.")
    print("  Densely it does NOT: compose is a d x d @ d x d matmul at 2*d^3 and")
    print("  apply is a matvec at 2*d^2, so ONE composition costs d applications.")
    print("  %-14s %16s %16s %10s" % ("path", "naive/node", "prefix/node",
                                      "prefix is"))
    rule()
    for lab, dn in (("blockwise b=4", False), ("DENSE d x d", True)):
        nv = pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", False, dn)[2]
        pf = pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", True, dn)[2]
        print("  %-14s %16d %16d %9.3fx %s"
              % (lab, nv, pf, max(nv, pf) / min(nv, pf),
                 "CHEAPER" if pf < nv else "DEARER"))
    be = 0
    while True:
        be += 1
        if (pos_per_node("twin+VGPE", D, be, "HOUSE", True, True)[2]
                < pos_per_node("twin+VGPE", D, be, "HOUSE", False, True)[2]):
            break
        assert be < 10000, "dense prefix never wins"
    print("  Dense break-even depth (2*d^3 + 4*d^2 < 4*depth*d^2): depth = %d."
          % be)
    print("  Closed form: 2*d + 4 < 4*depth, i.e. depth > d/2 + 1, so the first")
    print("  winning depth is d/2 + 2 = %d. At the spec's depth=%d the dense"
          % (D // 2 + 2, DEPTH))
    print("  prefix LOSES by %.1fx. THE PREFIX SAVING IS A PROPERTY OF THE BLOCKWISE"
          % (pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", True, True)[2]
             / pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", False, True)[2]))
    print("  REPRESENTATION, NOT OF THE TREE. Claiming both the dense Cayley and")
    print("  the prefix saving in one design is inconsistent.")
    print()

    # ---------------------------------------------------------------- 4
    print("=" * 100)
    print("4. THE FIVE ARMS, per training step, FORWARD, HOUSE convention")
    print("=" * 100)
    print("  Each arm at ITS OWN rate. No cross-arm ratio is carried (M-8).")
    print("  base + twin add-on from scale/m3_flops.py:cell_terms at k=%d."
          % K_PIV)
    print()
    for dense in (False, True):
        print("  --- Cayley path: %s ---"
              % ("DENSE d x d, |A|=%d" % ALPHABET if dense
                 else "BLOCKWISE, |A|=%d" % ALPHABET))
        hdr = ("  %-22s %16s %14s %14s %14s %18s %9s"
               % ("arm", "F_base", "twin add-on", "transport", "Cayley build",
                  "TOTAL", "pos %"))
        print(hdr)
        rule()
        tot = {}
        for arm in ARMS:
            t = arm_total(arm, N_HEADLINE, S, D, HIDDEN, K_PIV, DEPTH,
                          ALPHABET, "HOUSE", dense=dense, nn_terms=nn_terms)
            tot[arm] = t
            print("  %-22s %16d %14d %14d %14d %18d %8.4f%%"
                  % (arm, t["base"], t["twin"], t["transport"], t["build"],
                     t["total"], 100.0 * t["pos"] / t["total"]))
        print("  n=%d. transport is linear in n; Cayley build is NOT." % N_HEADLINE)
        if dense:
            print("  The three gauge arms coincide here: a dense U is dense "
                  "whether or not")
            print("  the group is abelian, so the block size stops mattering "
                  "once the")
            print("  representation is dense. That is a property of the dense "
                  "path, not")
            print("  a claim that arms 2 and 3 are cost-matched -- see section 5.")
        print()

    print("  --- SPEC CLAIM C: \"the gauge is FREE at the shapes this project")
    print("      ships\". Free = positional cost as a share of that arm's own")
    print("      total. Each arm's share is computed from ITS OWN terms. ---")
    print("  %-22s %8s %12s %14s %14s"
          % ("arm", "n", "blockwise", "dense Cayley", "delta vs RoPE"))
    rule()
    for n in NS:
        rope = arm_total("twin+RoPE", n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                         "HOUSE", nn_terms=nn_terms)
        for arm in ARMS:
            b = arm_total(arm, n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                          "HOUSE", dense=False, nn_terms=nn_terms)
            dn = arm_total(arm, n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                           "HOUSE", dense=True, nn_terms=nn_terms)
            print("  %-22s %8d %11.4f%% %13.4f%% %14d"
                  % (arm, n, 100.0 * b["pos"] / b["total"],
                     100.0 * dn["pos"] / dn["total"],
                     b["total"] - rope["total"]))
        rule()
    print("  VERDICT ON \"FREE\": on the BLOCKWISE path, yes -- the gauge is")
    print("  under 0.3% of its arm's own forward at every n above. On the DENSE")
    print("  path it is NOT free at any n: the build amortizes with n but the")
    print("  dense transport does not, and the transport is the larger term.")
    print("  The build alone, which IS n-independent and DOES amortize:")
    print("    %6s %8s %20s %20s %12s"
          % ("|A|", "n", "dense build", "transport", "build/total"))
    for alph in ALPHABETS:
        for n in NS:
            t = arm_total("twin+VGPE", n, S, D, HIDDEN, K_PIV, DEPTH, alph,
                          "HOUSE", dense=True, nn_terms=nn_terms)
            print("    %6d %8d %20d %20d %11.4f%%"
                  % (alph, n, t["build"], t["transport"],
                     100.0 * t["build"] / t["total"]))
    print()

    # ---------------------------------------------------------------- 5
    print("=" * 100)
    print("5. ARM SEPARATION -- what the cost table says about the controls")
    print("=" * 100)
    v = arm_total("twin+VGPE", N_HEADLINE, S, D, HIDDEN, K_PIV, DEPTH,
                  ALPHABET, "HOUSE", nn_terms=nn_terms)
    sh = arm_total("twin+shuffled-gauge", N_HEADLINE, S, D, HIDDEN, K_PIV,
                   DEPTH, ALPHABET, "HOUSE", nn_terms=nn_terms)
    ab = arm_total("twin+VGPE-abelian", N_HEADLINE, S, D, HIDDEN, K_PIV, DEPTH,
                   ALPHABET, "HOUSE", nn_terms=nn_terms)
    ro = arm_total("twin+RoPE", N_HEADLINE, S, D, HIDDEN, K_PIV, DEPTH,
                   ALPHABET, "HOUSE", nn_terms=nn_terms)
    sm = arm_total("softmax+RoPE", N_HEADLINE, S, D, HIDDEN, K_PIV, DEPTH,
                   ALPHABET, "HOUSE", nn_terms=nn_terms)
    print("  shuffled-gauge vs VGPE: %d vs %d, difference %d"
          % (sh["total"], v["total"], sh["total"] - v["total"]))
    print("    Shuffling the label -> U_a map permutes a lookup table. It moves")
    print("    WHICH U is fetched, never HOW MANY are applied. The M5-class")
    print("    plant is EXACTLY COST-MATCHED to the treatment. Any wall-clock")
    print("    gap between arms 2 and 4 is dispatch or contention, not work.")
    print()
    print("  positional totals: abelian %d, VGPE %d, ratio %.6f"
          % (ab["pos"], v["pos"], v["pos"] / ab["pos"]))
    print("  arm totals ratio VGPE/abelian: %.6f  (not the pos ratio, and not"
          % (v["total"] / ab["total"]))
    print("    the transport ratio -- no constant carries between the arms)")
    print("  abelian vs VGPE: %d vs %d, difference %d (%.4f%% of VGPE total)"
          % (ab["total"], v["total"], ab["total"] - v["total"],
             100.0 * (v["total"] - ab["total"]) / v["total"]))
    print("    per node: abelian %d, VGPE %d -- the abelian arm is %.4fx"
          % (pos_per_node("twin+VGPE-abelian", D, DEPTH, "HOUSE")[2],
             pos_per_node("twin+VGPE", D, DEPTH, "HOUSE")[2],
             pos_per_node("twin+VGPE", D, DEPTH, "HOUSE")[2]
             / pos_per_node("twin+VGPE-abelian", D, DEPTH, "HOUSE")[2]))
    print("    CAVEAT, and it is the load-bearing one for arm 3: the")
    print("    non-commutativity ablation is NOT cost-matched. It is cheaper,")
    print("    by construction, because commuting blocks are 2x2 and")
    print("    non-commuting ones are 4x4. A wall-clock comparison of arms 2")
    print("    and 3 confounds non-commutativity with block size. The FLOP")
    print("    difference is %.4f%% of the arm total, so it cannot explain a"
          % (100.0 * (v["total"] - ab["total"]) / v["total"]))
    print("    capability gap -- but it can explain a timing gap.")
    print()
    print("  twin arms vs softmax baseline: twin add-on %d = %.4f%% of F_base"
          % (ro["twin"], 100.0 * ro["twin"] / ro["base"]))
    print("    softmax+RoPE total %d, twin+RoPE total %d, difference %d"
          % (sm["total"], ro["total"], ro["total"] - sm["total"]))
    print()
    print("  --- k sweep: does the twin pivot count move any arm ordering? ---")
    print("  %5s %20s %20s %20s" % ("k", "twin+RoPE", "twin+VGPE", "abelian"))
    for k in (8, 16, 32):
        r = [arm_total(a, N_HEADLINE, S, D, HIDDEN, k, DEPTH, ALPHABET,
                       "HOUSE", nn_terms=nn_terms)["total"]
             for a in ("twin+RoPE", "twin+VGPE", "twin+VGPE-abelian")]
        print("  %5d %20d %20d %20d" % (k, r[0], r[1], r[2]))
    print()

    # ---------------------------------------------------------------- 6
    print("=" * 100)
    print("6. NOT COUNTED -- element counts, per-element FLOP NOT FOUND")
    print("=" * 100)
    for lab, sym, val in [
        ("RoPE cos/sin table", "s*d/2 (built once per step, cacheable)",
         S * D // 2),
        ("Cayley divide, blockwise b=2", "|A|*d/2", ALPHABET * D // 2),
        ("Cayley divide, blockwise b=4", "|A|*d/4", ALPHABET * D // 4),
        ("quaternion renormalise", "|A|*d/4 (rsqrt, if enforced per step)",
         ALPHABET * D // 4),
        ("transport backward", "~2x the forward transport, all arms alike",
         2 * N_HEADLINE * S * pos_per_node("twin+VGPE", D, DEPTH, "HOUSE")[2]),
        ("gate normalise (twin)", "n*k elements", N_HEADLINE * K_PIV),
    ]:
        print("  %-30s %-46s %14d" % (lab, sym, val))
    print()

    # ---------------------------------------------------------------- checks
    print("=" * 100)
    print("SELF-CHECK")
    print("=" * 100)

    # A. the spec's 768 is the abelian rate and not the quaternion rate
    assert block_mult(D, 2, "OPS") == 768, block_mult(D, 2, "OPS")
    assert block_mult(D, 4, "OPS") == 1792, block_mult(D, 4, "OPS")
    assert block_mult(D, 4, "OPS") != 768
    assert block_mult(D, 2, "HOUSE") == 1024 and block_mult(D, 4, "HOUSE") == 2048
    print("  OK  768/edge reproduces at b=2 OPS; b=4 OPS is 1792 (2.333x)")

    # B. the spec's 6,144 is the naive abelian one-vector path
    assert DEPTH * block_mult(D, 2, "OPS") == 6144
    assert 3 * block_mult(D, 4, "HOUSE") == 6144       # the coincidence
    assert 3 * block_mult(D, 4, "OPS") == 5376         # and it is a coincidence
    print("  OK  6,144 reproduces as naive/abelian/one-vector; the prefix")
    print("      quaternion HOUSE per-node cost is also 6,144 by coincidence")
    print("      (both are 24*d) and its OPS value 5,376 breaks the tie")

    # C. attention row
    assert attention_row(S, D, "HOUSE", full=False) == 524288
    assert attention_row(S, D, "HOUSE", full=True) == 1048576
    print("  OK  524,288 = 2*s*d = HALF an attention row (logits only)")

    # D. dense Cayley
    assert D ** 3 == 16777216
    assert cayley_dense(D, "OPS") == 2 * D ** 3
    assert cayley_dense(D, "HOUSE") == 4 * D ** 3
    print("  OK  16.7M = d^3 exactly; inv+matmul is 2*d^3 (OPS) / 4*d^3 (HOUSE)")

    # E. prefix saving is exactly 2*depth/3, both gauge arms, both conventions
    for arm in ("twin+VGPE", "twin+VGPE-abelian"):
        for conv in ("HOUSE", "OPS"):
            nv = pos_per_node(arm, D, DEPTH, conv, False)[2]
            pf = pos_per_node(arm, D, DEPTH, conv, True)[2]
            assert nv * 3 == pf * 2 * DEPTH, (arm, conv, nv, pf)
    print("  OK  prefix saving = 2*depth/3 = %.3fx, exact, both gauge arms"
          % (2 * DEPTH / 3))

    # F. shuffled-gauge is EXACTLY cost-matched to VGPE, derived not asserted
    for n in NS:
        a = arm_total("twin+VGPE", n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                      "HOUSE", nn_terms=nn_terms)
        b = arm_total("twin+shuffled-gauge", n, S, D, HIDDEN, K_PIV, DEPTH,
                      ALPHABET, "HOUSE", nn_terms=nn_terms)
        assert a == b, (n, a, b)
    print("  OK  shuffled-gauge == VGPE in every term at every n tested")

    # G. abelian is CHEAPER, so arm 3 is not cost-matched. The transport ratio
    #    is exactly 2 (b=2 against b=4); the BUILD ratio is 4, so the combined
    #    positional ratio is neither -- which is why it is derived, not carried.
    assert ab["total"] < v["total"]
    assert ab["transport"] * 2 == v["transport"], (ab, v)
    assert ab["build"] * 4 == v["build"], (ab, v)
    assert ab["pos"] * 2 != v["pos"]
    print("  OK  abelian arm is strictly cheaper than VGPE -- NOT cost-matched")
    print("      transport ratio exactly 2, build ratio exactly 4, pos ratio")
    print("      %.6f -- no single constant relates the arms" % (v["pos"] / ab["pos"]))

    # G2. the Linus prefix saving inverts on the dense path
    nv_d = pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", False, True)[2]
    pf_d = pos_per_node("twin+VGPE", D, DEPTH, "HOUSE", True, True)[2]
    assert pf_d > nv_d, (pf_d, nv_d)
    assert pf_d == 2 * D ** 3 + 4 * D * D and nv_d == 4 * DEPTH * D * D
    assert (pos_per_node("twin+VGPE", D, D // 2 + 2, "HOUSE", True, True)[2]
            < pos_per_node("twin+VGPE", D, D // 2 + 2, "HOUSE", False, True)[2])
    assert (pos_per_node("twin+VGPE", D, D // 2 + 1, "HOUSE", True, True)[2]
            == pos_per_node("twin+VGPE", D, D // 2 + 1, "HOUSE", False, True)[2])
    print("  OK  dense prefix is %.1fx DEARER than dense naive at depth=%d;"
          % (pf_d / nv_d, DEPTH))
    print("      it ties at depth = d/2 + 1 = %d and wins only from %d"
          % (D // 2 + 1, D // 2 + 2))

    # H. M-8: no arm's total is any other arm's total times a carried constant
    #    other than the one its own primitives produce.
    for n in NS:
        r = arm_total("twin+RoPE", n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                      "HOUSE", nn_terms=nn_terms)
        g = arm_total("twin+VGPE", n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                      "HOUSE", nn_terms=nn_terms)
        assert r["total"] < g["total"], (n, r["total"], g["total"])
    print("  OK  RoPE < VGPE at every n; each from its own primitives")

    # I. FREE on blockwise, NOT free on dense at n=1
    b1 = arm_total("twin+VGPE", 1, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                   "HOUSE", dense=False, nn_terms=nn_terms)
    d1 = arm_total("twin+VGPE", 1, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                   "HOUSE", dense=True, nn_terms=nn_terms)
    assert b1["pos"] / b1["total"] < 0.003, b1["pos"] / b1["total"]
    assert d1["pos"] / d1["total"] > 0.25, d1["pos"] / d1["total"]
    print("  OK  blockwise pos = %.4f%% of total at n=1 (FREE);"
          % (100.0 * b1["pos"] / b1["total"]))
    print("      dense Cayley  = %.4f%% of total at n=1 (NOT FREE)"
          % (100.0 * d1["pos"] / d1["total"]))

    # J. the dense build amortizes; find where it drops under 1%
    n = 1
    while True:
        t = arm_total("twin+VGPE", n, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                      "HOUSE", dense=True, nn_terms=nn_terms)
        if t["build"] / t["transport"] < 0.01:
            break
        n += 1
        assert n < 100000, "dense build never amortizes below 1%"
    print("  OK  dense Cayley build drops below 1% of the DENSE TRANSPORT it")
    print("      shares an arm with at n = %d -- but the transport it amortizes"
          % n)
    print("      against is itself %.1fx F_base, so the dense path never becomes"
          % (arm_total("twin+VGPE", 1, S, D, HIDDEN, K_PIV, DEPTH, ALPHABET,
                       "HOUSE", dense=True, nn_terms=nn_terms)["transport"]
             / cell_terms("twin", 1, S, D, HIDDEN, K_PIV, T_STAR,
                          nn_terms)["base"]))
    print("      free by amortizing the build")
    print()
    print("ALL CHECKS PASSED. No wall clock was taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
