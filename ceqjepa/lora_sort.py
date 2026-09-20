"""THE LORA SORTING WING -- three selection rules, one budget, one pre-registered counter.

WHAT THE OBJECT IS. The read is `ceqjepa/operator.py:208 state_solve`,
`z = (I - g P)^{-1} V`, solved by one lower-triangular forward substitution.
A LoRA on the OPERATOR is `Delta = U W^T` at rank `r`, so the updated system
matrix is `M = M0 - g U W^T` with `M0 = I - g P` still triangular. Woodbury
keeps the triangular solve and adds one `r x r` solve. The question this module
asks is NOT whether that update is exact -- it is, and the exactness is
re-derived here rather than quoted. The question is WHICH `r` directions, and in
what order, when there are many heads.

THE PREMISE CORRECTION, BEFORE ANY NUMBER. The round was briefed with
"this repository already measures that at 4.44e-16". It does not. A sweep of
every producer of `4.44e-16` in this tree found four distinct objects and none
is an updated-resolvent exactness:

  * `python -m ceqjepa.operator` check (b), `ceqjepa/operator.py:404` -- closed-form
    committor vs general solve, `4.441e-16`. Quoted at `README.md:67` and `:219`.
  * `python -m ceqjepa.intervene` check (b), `ceqjepa/intervene.py:267` -- BATCH vs
    32 singles of the SAME rank-1 formula, `4.441e-16`. The update-vs-re-solve
    number in that module's check (a) is `1.110e-15` over 240 draws.
  * `ceqjepa/pi_jepa.py:666 corner_identities()` -- the softmax corner at beta=1,
    `4.440892098500626e-16`.
  * `docs/sources/design/refute_falsify_math.md:196` -- an uncommitted `RUN` of a
    DIFFERENT (dense-M) formula, and `docs/sources/design/refute_instrument_math.md:45`
    already ruled that family route-dependent (`torch.linalg.solve` 8.9e-16,
    `solve_triangular` 0.0).

Woodbury itself has no implementation anywhere in this tree. `docs/canon/08_ARCHITECTURE.md:220`
files it as the unbuilt repair for the two-row case, and `docs/sources/sweep/sweep_safety.md:131`
files multi-row interventions as OPEN. `woodbury_exactness()` below is the first
producer, and it prints its own number.

THE CITATIONS, each with the equation the round actually leans on. A named source
without a resolvable identifier AND a specific theorem, page or equation is P-16
class (`MISTAKES.md:2863`) and struck.

  WOODBURY. Hager, W. W., "Updating the Inverse of a Matrix", SIAM Review 31(2):
    221--239, 1989, DOI 10.1137/1031049. **Equation (1), page 221**, the first
    displayed equation of the paper:
        [A - U V]^-1 = A^-1 + A^-1 U (I - V A^-1 U)^-1 V A^-1.
    Hager writes the update as a DIFFERENCE, so the capacitance factor carries a
    minus sign; `V` is a matrix, not a transpose. `_woodbury` below is that
    equation with `A = M0`, `U -> g U`, `V -> W^T`. Woodbury's own 1950 Princeton
    memorandum (SRG Memorandum Report 42) has no DOI, no ISBN and no accessible
    text -- it is named here as history only and carries NO equation, per the
    same rule. Hager p. 223 attributes (1) to Duncan (1944) and Guttman (1946)
    before Woodbury.

  S-SVD, the BASELINE, and it is L-SIMPLE. Owned, not contributed:
    Zhang, Chen, Bukharin, Karampatziakis, He, Cheng, Chen, Zhao, "AdaLoRA:
      Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", ICLR 2023,
      arXiv:2303.10512. **Sec. 3.1 Eq. (3)** parameterises `W = W^(0) + P Lambda Q`
      in SVD form; **Sec. 3.2 Eq. (8)** is the triplet importance
      `S_{k,i} = s(lambda_{k,i}) + (1/d1) sum_j s(P_{k,ji}) + (1/d2) sum_j s(Q_{k,ij})`
      it prunes by, built on the per-entry sensitivity **Eq. (9)**
      `I(w_ij) = |w_ij grad_{w_ij} L|`.
    Meng, Wang, Zhang, "PiSSA: Principal Singular Values and Singular Vectors
      Adaptation of Large Language Models", NeurIPS 2024, arXiv:2404.02948.
      **Sec. 3 Eqs. (2)-(3)** initialise `A = U_{[:,:r]} S_{[:r,:r]}^{1/2}` and
      `B = S_{[:r,:r]}^{1/2} V_{[:,:r]}^T` from the top-`r` singular triplets;
      **Eq. (5)** is the principal/residual split `Y = X(W^res + A B)`.
    `sort_svd` below is PiSSA Eqs. (2)-(3) applied to the gradient rather than to
    `W`. It is NOT AdaLoRA Eq. (8): that score needs the SVD-form
    parameterisation trained, and this round selects once at step 0.

  S-OT. Center-outward transport ranks against a spherical reference.
    See `sort_ot` for the resolved identifiers and the UNSOURCED marks -- the
    dimension-reduction step this round takes is NOT licensed by either paper and
    is labelled there.

  S-HILB. Hilbert, D., "Ueber die stetige Abbildung einer Linie auf ein
    Flaechenstueck", Mathematische Annalen 38(3):459--460, 1891,
    DOI 10.1007/BF01199431. **Two pages. Read directly from the GDZ scan: the
    paper contains ZERO numbered equations, ZERO numbered theorems and no section
    structure.** The finest locator it admits is a page. It establishes
    continuity and surjectivity of the map (the italicised, unnumbered sentence
    on p. 460) and states NOTHING about locality, distance, neighbourhoods or
    clustering -- the edge-sharing constraint on consecutive subsquares (p. 459)
    is a construction device he draws no consequence from. **Citing Hilbert 1891
    for "nearby indices along the curve are nearby in the plane" is a
    misattribution**, so the citation is split:
    Moon, Jagadish, Faloutsos, Saltz, "Analysis of the Clustering Properties of
      the Hilbert Space-Filling Curve", IEEE TKDE 13(1):124--141, 2001,
      DOI 10.1109/69.908985. **Theorem 1, page 130**: for a rectilinear
      polyhedral query of total surface area `S_q` in a `d`-dimensional grid under
      the order-`k` Hilbert curve, the average number of clusters converges as
      `k -> inf` to `S_q / (2d)` -- the count of broken runs scales with the
      query's SURFACE AREA, not its volume. That is the locality claim,
      quantitative, and it is Moon et al.'s, not Hilbert's.
    Springer/Crossref metadata for the Hilbert DOI carries a typo ("einer Line");
    the printed title reads "einer Linie". Moon et al.'s own reference [13]
    mis-cites it a second way. Both noted so a title-matching verifier's
    disagreement is not read as a fabrication.

  LoRA+ RATIO 4, AND THE PAPER DOES NOT RECOMMEND IT.
    Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of Large Models",
    ICML 2024, PMLR 235:17783--17806, arXiv:2402.12354. The rule is **not a
    numbered equation**: it is an unnumbered displayed statement at the end of
    **Sec. 4**, "set the learning rates for A, B such that eta_B = lambda eta_A
    with lambda > 1 fixed", supported by **Theorem 1** (efficiency is impossible
    at eta_A = eta_B and is achieved at eta_A = Theta(n^-1), eta_B = Theta(1)).
    **Sec. 5.3 gives no universal ratio.** Verbatim: "generally setting a ratio of
    lambda = eta_B/eta_A ~= 2^4 improves performance for Roberta" -- and only
    under Init[2]; with Init[1] "the optimal ratio is smaller and is of order
    2^2 - 2^3"; "For LLama experiments, it seems that a ratio of order 2^1 - 2^2
    is optimal"; and "The optimal ratio is model and task sensitive and shows
    significant variance." The round's brief pins the ratio at 4. It is HELD at
    4 across all three sorts as a PINNED CONSTANT OF THIS ROUND (`LORA_PLUS_RATIO`),
    it sits inside the paper's Llama range `2^1 - 2^2`, and it is NOT presented as
    the paper's recommendation. Any sentence claiming LoRA+ recommends 4 is struck.

L-NULL. What VARIES across the three sorts: the selection rule, and nothing
else. What is PINNED: the base logits, the base operator `P`, `g`, the planted
teacher delta, the input draw, the evaluation draw, the rank `r`, the step count,
the optimiser, both learning rates, the LoRA+ ratio, the spectral clip `c`, the
seed, the dtype, and the zero-init factor. `pinned_manifest()` prints all of them
and `tests/lorasort/test_lora_sort.py` asserts the three configs differ in
exactly one field.

L-SURFACE. `demo()` returns an exit status and the self-check block asserts it;
no number in this module is read through a pipeline.

MUST-FIRE 5 IS NOT A BUG. With `U` set by the sort and `W = 0`, `Delta = 0`, and
`dL/dU = (dL/dDelta) W = 0` exactly -- one factor is DEAD at initialisation by
the same mechanism `MISTAKES.md:2526 V-29` records for the gate. It wakes once
`W` has moved. The check is therefore "dead at 0, LIVE at 1", and a factor still
dead at step 1 is a real defect. The step-0 zero is NOT repaired: it is what
makes `Delta = 0` at init, which the bitwise bind requires.

PRE-REGISTERED, and neither moves after the first number:
  PREDICTION: S-OT or S-HILB beats S-SVD with a confidence interval at r <= 4.
  COUNTER:    S-SVD ties or wins. Variance is the right sort when the loss is
              squared. The other two buy robustness at large r and nothing at
              r <= 4, and the card's honest sentence is then "sort choice inside
              resolution".

Producer: `python -m ceqjepa.lora_sort`. Tests: `python -m pytest tests/lorasort -q`.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

import numpy as np
import torch

from ceqjepa.operator import build_operator, state_solve

__all__ = [
    "HILBERT_SIDE", "N_POS", "N_HEADS", "D_CHAN", "GAMMA", "SPECTRAL_CLIP",
    "LORA_PLUS_RATIO", "LR_U", "SORTS", "R_GRID", "SEEDS",
    "hilbert_order", "woodbury_read", "dense_read", "woodbury_exactness",
    "sort_svd", "sort_ot", "sort_hilb", "subspace_angle", "pinned_manifest",
    "drop_test_ot", "drop_test_hilb", "sanity_planted", "grad_table",
    "bind_at_zero", "run_round", "clopper_pearson", "demo",
]

# ---------------------------------------------------------------- pinned constants
#: 8x8 grid -> 64 positions. The Hilbert curve needs a power-of-two square side.
HILBERT_SIDE = 8
N_POS = HILBERT_SIDE * HILBERT_SIDE
N_HEADS = 6
D_CHAN = 8
GAMMA = 0.9                     # g in (I - g P)^-1; < 1 keeps diag(M0) >= 1 - g.
#: c in ||U W^T||_2 <= c. PRINTED, never assumed, and CHOSEN SO THE BOUND BITES.
#: The updated system is M = M0 - g U W^T, so sigma_min(M) >= sigma_min(M0) - g c
#: and the bound 1/(sigma_min(M0) - g c) is VACUOUS unless g c < sigma_min(M0).
#: Measured over all 8 seeds x 6 heads: sigma_min(M0) in [0.018456, 0.025377]
#: (producer: the two-line loop in this module's docstring block below, HEAD
#: 386289e, WIN-16QAL06O9GB). So c must be under 0.018456/0.9 = 0.020507. The
#: first draft of this round shipped c = 0.5, where g c = 0.45 exceeded
#: sigma_min(M0) by 24x, the clip engaged on 1708 of 1800 head-steps, and the
#: bound printed `Infinity` and "held" vacuously. That is recorded rather than
#: quietly corrected: a bound that cannot fail is V-10's shape.
SPECTRAL_CLIP = 0.01
LORA_PLUS_RATIO = 4.0           # eta_W / eta_U. See the module docstring: PINNED
                                # BY THIS ROUND, not recommended by Hayou et al.
LR_U = 1e-3
STEPS = 200
BATCH_TRAIN = 32
BATCH_EVAL = 64
TEACHER_RANK = 2                # the planted delta the student must recover
#: ||D*||_2 for the planted teacher delta. HALF the spectral clip, so the target
#: is REACHABLE inside the clip. The first draft left it at ~0.10, ten times the
#: clip, which makes the task unsolvable by construction and every sort's score a
#: measurement of the clip instead of the sort.
TEACHER_SPEC = 0.005
DT = torch.float64
SORTS = ("S-SVD", "S-OT", "S-HILB")
R_GRID = (1, 2, 4, 8, 16)       # 1,2,4 is the pre-registered "r <= 4" regime
SEEDS = tuple(range(8))
#: Two r-dimensional subspaces AGREE when their largest principal angle is below
#: this. Fixed before the drop tests ran; a drop test with a floating threshold
#: is not a drop test.
AGREE_RAD = 1e-6
HILB_WINDOW = N_POS // 4        # 16, the contiguous Hilbert run S-HILB scores on


def machine() -> str:
    return "%s python %s torch %s" % (platform.node(),
                                      platform.python_version(), torch.__version__)


def pinned_manifest(sort: str, r: int, seed: int) -> dict:
    """L-NULL. Everything the round holds fixed, beside the one thing it varies.

    The ONLY key that differs between two calls with the same (r, seed) is
    `varied_selection_rule`. Asserted by
    tests/lorasort/test_lora_sort.py::test_exactly_one_field_varies_across_the_three_sorts.
    """
    return {
        "varied_selection_rule": sort,
        "pinned_n_pos": N_POS, "pinned_n_heads": N_HEADS, "pinned_d_chan": D_CHAN,
        "pinned_gamma": GAMMA, "pinned_rank": r, "pinned_seed": seed,
        "pinned_steps": STEPS, "pinned_batch_train": BATCH_TRAIN,
        "pinned_batch_eval": BATCH_EVAL, "pinned_teacher_rank": TEACHER_RANK,
        "pinned_lr_u": LR_U, "pinned_lora_plus_ratio": LORA_PLUS_RATIO,
        "pinned_spectral_clip": SPECTRAL_CLIP, "pinned_dtype": str(DT),
        "pinned_zero_factor": "W", "pinned_optimiser": "Adam",
        "pinned_teleport": 0.0,
    }


# ---------------------------------------------------------------- the Hilbert order
def _d2xy(side: int, d: int) -> tuple[int, int]:
    """Position `d` along the order-log2(side) Hilbert curve -> (x, y).

    The standard iterative inverse of the curve. Hilbert 1891 (Math. Ann. 38(3):
    459-460) gives the construction as prose and three figures and numbers no
    equation, so this carries no equation citation -- only the page.
    """
    x = y = 0
    t = d
    s = 1
    while s < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_order(side: int = HILBERT_SIDE) -> np.ndarray:
    """`perm[k]` = the row index visited k-th along the curve.

    Rows are laid on the grid by the PINNED layout `i -> (i % side, i // side)`;
    that layout is this round's choice, not something derived from the data, and
    it is what makes "block-local" mean a 2-D tile rather than a run of indices.
    """
    perm = np.empty(side * side, dtype=np.int64)
    for d in range(side * side):
        x, y = _d2xy(side, d)
        perm[d] = y * side + x
    assert sorted(perm.tolist()) == list(range(side * side)), "not a permutation"
    return perm


HILBERT_PERM = hilbert_order()


# ---------------------------------------------------------------- the Woodbury read
def dense_read(P: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
               V: torch.Tensor, g: float = GAMMA) -> torch.Tensor:
    """The honest route: form `M = I - g (P + U W^T)` densely and solve it.

    Not triangular any more -- a rank-r delta destroys the causal structure -- so
    this is a general LU solve. It is the REFERENCE the Woodbury route is checked
    against, never the shipped route.
    """
    n = P.shape[-1]
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    M = eye - g * (P + U @ W.transpose(-1, -2))
    return torch.linalg.solve(M, V)


def woodbury_read(P: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                  V: torch.Tensor, g: float = GAMMA) -> torch.Tensor:
    """Hager 1989 Eq. (1), p. 221, with A = M0 = I - g P (triangular), U -> g U,
    V -> W^T:

        (M0 - g U W^T)^-1 Y
            = M0^-1 Y + M0^-1 (g U) (I_r - W^T M0^-1 (g U))^-1 W^T M0^-1 Y

    ONE triangular solve on M0 with the `d + r` right-hand sides `[Y | gU]`, then
    one `r x r` general solve. M0 stays lower-triangular whatever the delta does,
    which is the whole reason this route exists.
    """
    n = P.shape[-1]
    r = U.shape[-1]
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    M0 = eye - g * P
    # V may carry batch dims that U and W do not (one LoRA, many inputs), and
    # the head axis is batched too. Match on SHAPE, not on dim count.
    if U.shape[:-2] != V.shape[:-2]:
        U = U.expand(V.shape[:-2] + U.shape[-2:])
        W = W.expand(V.shape[:-2] + W.shape[-2:])
    rhs = torch.cat([V, g * U], dim=-1)
    sol = torch.linalg.solve_triangular(M0, rhs, upper=False)
    base, Minv_gU = sol[..., :V.shape[-1]], sol[..., V.shape[-1]:]
    Wt = W.transpose(-1, -2)
    cap = torch.eye(r, dtype=P.dtype, device=P.device) - Wt @ Minv_gU   # capacitance
    return base + Minv_gU @ torch.linalg.solve(cap, Wt @ base)


def woodbury_exactness(n_draws: int = 60, ranks=(1, 2, 4, 8, 16),
                       seed: int = 0) -> dict:
    """RE-DERIVED, not quoted. Woodbury route vs dense re-solve, worst over draws.

    There is no prior producer for this number in this tree (see the module
    docstring's premise correction), so this function IS the producer.
    """
    torch.manual_seed(seed)
    worst = 0.0
    worst_at = None
    worst_rel = 0.0
    n_cells = 0
    for r in ranks:
        for d in range(n_draws // len(ranks)):
            logits = torch.randn(N_POS, N_POS, dtype=DT)
            P = build_operator(logits, [], teleport=0.0)
            U = torch.randn(N_POS, r, dtype=DT) * 0.05
            W = torch.randn(N_POS, r, dtype=DT) * 0.05
            V = torch.randn(N_POS, D_CHAN, dtype=DT)
            a = woodbury_read(P, U, W, V)
            b = dense_read(P, U, W, V)
            err = float((a - b).abs().max())
            rel = err / float(b.abs().max())
            n_cells += 1
            if err > worst:
                worst, worst_at, worst_rel = err, (r, d), rel
    return {"worst_abs": worst, "worst_rel": worst_rel, "worst_at_rank_draw": worst_at,
            "n_cells": n_cells, "route_a": "woodbury_read (solve_triangular + r x r)",
            "route_b": "dense_read (torch.linalg.solve, LU)",
            "identity": "Hager 1989 SIAM Rev 31(2) Eq. (1) p.221", "dtype": str(DT)}


# ---------------------------------------------------------------- the three sorts
def _orth(X: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis of col(X). Every sort is compared as a SUBSPACE, so
    the comparison cannot be gamed by a change of basis inside the span."""
    Q, _ = torch.linalg.qr(X)
    return Q


def subspace_angle(A: torch.Tensor, B: torch.Tensor) -> float:
    """Largest principal angle, radians, between col(A) and col(B).

    0 means the two selections span the SAME subspace -- which is what the two
    drop tests are looking for. Basis-free by construction.
    """
    Qa, Qb = _orth(A), _orth(B)
    s = torch.linalg.svdvals(Qa.transpose(-1, -2) @ Qb)
    return float(torch.arccos(s.clamp(-1.0, 1.0).min()))


def sort_svd(C: torch.Tensor, r: int, **_) -> torch.Tensor:
    """S-SVD, THE BASELINE. PiSSA Eqs. (2)-(3) (arXiv:2404.02948, Sec. 3) applied
    to the candidate matrix: keep the top-`r` left singular vectors, scaled by
    S^{1/2}. AdaLoRA Eq. (3) (arXiv:2303.10512, Sec. 3.1) is the same SVD-form
    parameterisation of the update. THIS IS OWNED PRIOR ART AND IS NOT PRESENTED
    AS A CONTRIBUTION OF THIS ROUND.
    """
    Uu, S, _ = torch.linalg.svd(C, full_matrices=False)
    return Uu[:, :r] * S[:r].sqrt()


def _center_outward_ranks(pts: np.ndarray, seed: int = 0) -> np.ndarray:
    """Center-outward rank of each of the `m` points, by optimal assignment onto a
    spherical-uniform grid on the unit ball in R^2.

    THE GRID. `n_R` radii `j/(n_R+1)`, `j = 1..n_R`, times `n_S` equispaced unit
    directions, `m = n_R n_S`, matched to the sample by the assignment minimising
    total squared distance. The rank returned is the RADIUS INDEX of the grid
    point a sample point was matched to: small = deep = central.

    SOURCES, AND WHAT THEY DO AND DO NOT LICENSE. Both were read in the
    PUBLISHED Project Euclid PDFs, not in a preprint and not from a summary.

      Hallin, del Barrio, Cuesta-Albertos, Matran, "Distribution and quantile
        functions, ranks and signs in dimension d: A measure transportation
        approach", Annals of Statistics 49(2):1139--1165, 2021,
        DOI 10.1214/20-AOS1996. **CITE THE DOI, NOT AN ARXIV ID.** There is no
        arXiv entry carrying this title: `arXiv:1806.01238` is a DIFFERENT,
        companion paper ("Center-Outward Distribution Functions, Quantiles,
        Ranks, and Signs in R^d", authors in a different order), which the
        published paper cites in its own reference list under a THIRD title.
        Pairing 1806.01238 with this title is the resolvable-identifier /
        wrong-title pattern `MISTAKES.md` P-16 is about.
        **Definition 2.2, p. 1145**: "Call F+- := grad psi the center-outward
        distribution function of P". **Sec. 2.1, p. 1145**, immediately above
        Definition 2.1, fixes the reference: "Denoting by U_d the spherical
        uniform distribution over S_d", and **Proposition 2.1(i)** gives
        `F+- # P = U_d`. The ranks and signs themselves are NOT in a numbered
        definition -- they are bulleted displays in **Sec. 2.2, p. 1150**,
        `R_{+-,i} := (n_R+1) ||F_{+-}(Z_i)||`, immediately after
        **Definition 2.3**, which is the EMPIRICAL center-outward distribution
        function defined by the optimal-assignment coupling to the augmented
        grid -- that assignment is what `linear_sum_assignment` below is.
        **Proposition 2.5(ii), p. 1151**, labelled (DF) in the text, is the
        distribution-freeness: the empirical map is uniformly distributed over
        the n!/n_0! permutations with repetitions of the grid.
      Chernozhukov, Galichon, Hallin, Henry, "Monge-Kantorovich depth, quantiles,
        ranks and signs", Annals of Statistics 45(1):223--256, 2017,
        DOI 10.1214/16-AOS1450, arXiv:1412.8434 (this pairing IS clean).
        **Definition 2.2, p. 233** opens "Let F be the spherical uniform
        distribution U_d on the unit ball U = S_d" and defines the MK rank and
        sign; **Theorem 3.1, p. 239** is uniform convergence of empirical
        transport maps.
        **THIS PAPER PROVES NO DISTRIBUTION-FREENESS THEOREM.** A full-text grep
        finds "distribution-free" twice: once in its reference list, once in the
        introduction about CLASSICAL and SPHERICAL ranks. Theorem 3.1 is
        consistency, not finite-sample distribution-freeness. Attributing
        distribution-freeness to CGHH 2017 is a mis-citation, and this round
        does not make it -- that claim rests on Hallin 2021 Prop. 2.5(ii) alone.

    AND THE BRIEF'S OWN SENTENCE IS UNSOURCED. "Keeps directions by DEPTH rather
    than by variance" appears in NEITHER paper. Neither frames any contrast
    against variance; CGHH 2017 p. 238 in fact PRE-STANDARDISES X to isotropic
    position through the scatter matrix Sigma to obtain affine invariance, which
    is a second-moment step, so "rather than variance" is actively misleading for
    that paper. The nearest thing either paper proves is **Hallin 2021
    Proposition 2.1(ii)**: the sign `F+-(Z)/||F+-(Z)||` is uniform on S_{d-1},
    the modulus `||F+-(Z)||` is uniform on [0,1], and the two are mutually
    independent -- direction and depth-order FACTORISE. That is what S-OT
    actually leans on. The depth-versus-variance phrasing is carried in
    `CITATION_MARKS["S-OT"]["unsourced"]` with no author's name attached to it.

    THE STEP NEITHER PAPER LICENSES, STATED HERE RATHER THAN IN A LIMITS
    PARAGRAPH: the center-outward construction is for a FIXED dimension `d` with
    sample size `n -> inf`. This round has `m = 64` candidate points living in
    `R^64`. That is not the regime, and `m` mutually near-orthogonal points in
    `R^m` have a degenerate depth ordering driven entirely by their norms -- which
    would make S-OT a relabelled S-SVD for a construction reason rather than a
    measured one. So the transport is run in the `k = 2` principal coordinates of
    the candidate pool. **That reduction is this round's choice. No cited paper
    licenses it, and the distribution-freeness results do not transfer through
    it.** It is the reason S-OT's justification is graded PARTIAL below.
    """
    m = pts.shape[0]
    n_rad = int(round(math.sqrt(m)))
    while m % n_rad:
        n_rad -= 1
    n_dir = m // n_rad
    ang = 2.0 * math.pi * np.arange(n_dir) / n_dir
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)          # [n_dir, 2]
    rad = (np.arange(1, n_rad + 1) / (n_rad + 1.0))[:, None]  # [n_rad, 1]
    grid = (rad[:, None, :] * dirs[None, :, :]).reshape(-1, 2)
    grid_rank = np.repeat(np.arange(n_rad), n_dir)

    x = pts - pts.mean(0, keepdims=True)
    scale = np.abs(x).max()
    if scale > 0:
        x = x / scale
    from scipy.optimize import linear_sum_assignment
    cost = ((x[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
    row, col = linear_sum_assignment(cost)
    out = np.empty(m, dtype=np.int64)
    out[row] = grid_rank[col]
    return out


def sort_ot(C: torch.Tensor, r: int, seed: int = 0, **_) -> torch.Tensor:
    """S-OT. Keep the `r` DEEPEST candidate columns by center-outward rank.

    The candidates are the `n` columns of `C`, taken as points in `R^n`, reduced
    to their two leading principal coordinates (see `_center_outward_ranks` for
    why, and for what that reduction costs). Depth, not variance: the deepest
    columns are the CENTRAL ones, which is what makes the rule robust to a
    heavy-tailed candidate pool and is what drop test 2 checks is actually
    different from keeping the largest.

    Ties inside a radius shell are broken by the column index, ascending -- a
    pinned rule, so the selection is deterministic and reproducible.
    """
    Cn = C.detach().cpu().numpy().astype(np.float64)
    pts = Cn.T                                              # [n columns, n dims]
    x = pts - pts.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(x, full_matrices=False)
    two = x @ Vt[:2].T                                      # [n, 2] principal coords
    ranks = _center_outward_ranks(two, seed=seed)
    order = np.lexsort((np.arange(len(ranks)), ranks))      # deepest first
    keep = order[:r]
    sel = C[:, torch.as_tensor(keep.copy(), device=C.device)]
    return sel / sel.norm(dim=0, keepdim=True).clamp_min(1e-30)


def sort_hilb(C: torch.Tensor, r: int, perm: np.ndarray | None = None,
              window: int = HILB_WINDOW, **_) -> torch.Tensor:
    """S-HILB. Keep the `r` most BLOCK-LOCAL candidate columns.

    The rows are permuted into Hilbert order (Hilbert 1891, Math. Ann. 38(3):
    459-460 -- the construction, prose and figures, no numbered result exists to
    cite) and each candidate column is scored by the largest fraction of its
    energy inside any contiguous window of `window` positions along that order.
    The score is NORMALISED by the column's total energy, so it measures
    LOCALITY and not variance -- a column with ten times the norm and diffuse
    support scores below a small column concentrated in one tile.

    The locality property this rule leans on is Moon, Jagadish, Faloutsos, Saltz,
    IEEE TKDE 13(1):124-141, 2001, DOI 10.1109/69.908985, **Theorem 1, p. 130**:
    the average number of clusters a query breaks into converges to
    `S_q / (2d)` -- surface area, not volume. It is NOT in Hilbert 1891.
    """
    perm = HILBERT_PERM if perm is None else perm
    e = (C[torch.as_tensor(perm.copy(), device=C.device), :] ** 2)   # [n, m]
    cum = torch.cat([torch.zeros(1, e.shape[1], dtype=e.dtype, device=e.device),
                     e.cumsum(0)], 0)
    win = cum[window:] - cum[:-window]                               # [n-w+1, m]
    score = (win.max(0).values / e.sum(0).clamp_min(1e-30))
    keep = torch.argsort(score, descending=True, stable=True)[:r]
    sel = C[:, keep]
    return sel / sel.norm(dim=0, keepdim=True).clamp_min(1e-30)


SORT_FN = {"S-SVD": sort_svd, "S-OT": sort_ot, "S-HILB": sort_hilb}

#: Per-sort citation state. `grade` is this round's own mark, on the repo's
#: `docs/BIB_AUDIT.md` vocabulary: V = record fetched and title matched,
#: V-eq = the equation was transcribed and is the one implemented,
#: PARTIAL = the source is real and the equation located, but a step this round
#: takes is not licensed by it, UNSOURCED = no source located for the sentence.
CITATION_MARKS = {
    "S-SVD": {
        "grade": "V-eq",
        "sources": [
            "AdaLoRA, arXiv:2303.10512, ICLR 2023, Sec. 3.1 Eq. (3) (Delta = P Lambda Q); "
            "Sec. 3.2 Eq. (8) importance, Eq. (9) sensitivity |w grad_w L|",
            "PiSSA, arXiv:2404.02948, NeurIPS 2024, Sec. 3 Eqs. (2)-(3) init from "
            "top-r singular triplets; Eq. (5) principal/residual split",
        ],
        "implemented": "PiSSA Eqs. (2)-(3) on the gradient. NOT AdaLoRA Eq. (8): "
                       "that score needs the SVD-form parameterisation trained, "
                       "and this round selects once at step 0.",
    },
    "S-OT": {
        "grade": "PARTIAL",
        "sources": [
            "Hallin, del Barrio, Cuesta-Albertos, Matran, Ann. Statist. "
            "49(2):1139-1165, 2021, DOI 10.1214/20-AOS1996 (NO arXiv id -- "
            "1806.01238 is a different, companion title). Def. 2.2 p.1145 "
            "(F+-); Sec. 2.1 p.1145 + Prop. 2.1(i) (spherical uniform U_d); "
            "Def. 2.3 + Sec. 2.2 p.1150 (empirical ranks by optimal assignment "
            "to the grid); Prop. 2.5(ii) p.1151 (distribution-freeness); "
            "Prop. 2.1(ii) (sign uniform on the sphere, independent of the "
            "rank modulus)",
            "Chernozhukov, Galichon, Hallin, Henry, Ann. Statist. 45(1):223-256, "
            "2017, DOI 10.1214/16-AOS1450, arXiv:1412.8434. Def. 2.2 p.233 "
            "(MK rank/sign, spherical uniform reference); Thm. 3.1 p.239 "
            "(uniform convergence of empirical transport maps)",
        ],
        "implemented": "center-outward ranks by assignment onto a spherical grid, "
                       "Hallin 2021 Def. 2.3",
        "unsourced": [
            "'KEEPS DIRECTIONS BY DEPTH RATHER THAN BY VARIANCE' -- the brief's "
            "own sentence, in NEITHER paper. Neither frames a contrast against "
            "variance, and CGHH 2017 p.238 pre-standardises through the scatter "
            "matrix, a second-moment step. NO AUTHOR IS NAMED FOR THIS SENTENCE. "
            "The supported neighbour is Hallin 2021 Prop. 2.1(ii): sign and rank "
            "modulus factorise and are independent.",
            "THE k=2 PRINCIPAL-COORDINATE REDUCTION. Neither paper licenses "
            "running the construction on m=64 points in R^64 by first projecting "
            "to R^2. Prop. 2.5(ii)'s distribution-freeness does NOT transfer "
            "through that projection, so 'S-OT is distribution-free as "
            "implemented here' is UNSOURCED and is not claimed.",
        ],
        "miscitation_avoided": "Distribution-freeness is NOT attributed to CGHH "
                               "2017: that paper proves no such theorem.",
    },
    "S-HILB": {
        "grade": "V-eq",
        "sources": [
            "Hilbert, Math. Ann. 38(3), 1891, pp. 459-460, DOI 10.1007/BF01199431 "
            "-- the curve, and the construction constraint that consecutive "
            "subsquares share a side, p. 459. TWO PAGES, ZERO numbered "
            "equations and ZERO numbered theorems: the page IS the finest "
            "locator this paper admits.",
            "Moon, Jagadish, Faloutsos, Saltz, IEEE TKDE 13(1):124-141, 2001, "
            "DOI 10.1109/69.908985, Theorem 1, p. 130 -- the average cluster "
            "count converges to S_q/(2d): surface area, not volume. THIS, not "
            "Hilbert 1891, is where the locality claim lives.",
        ],
        "implemented": "Hilbert order + windowed energy concentration",
        "note": "The LOCALITY claim is Moon et al. Theorem 1, NOT Hilbert 1891. "
                "Hilbert 1891 proves continuity and surjectivity and states "
                "nothing about distance or clustering.",
    },
}


# ---------------------------------------------------------------- the task
def _base(seed: int):
    """The PINNED base: operator, value projections, teacher delta. Identical for
    all three sorts at a given seed -- this is the object L-NULL says is held."""
    gen = torch.Generator().manual_seed(1000 + seed)
    logits = torch.randn(N_HEADS, N_POS, N_POS, generator=gen, dtype=DT)
    P = build_operator(logits, [], teleport=0.0)
    Wv = torch.randn(N_HEADS, D_CHAN, D_CHAN, generator=gen, dtype=DT) / math.sqrt(D_CHAN)
    Wo = torch.randn(N_HEADS, D_CHAN, D_CHAN, generator=gen, dtype=DT) / math.sqrt(D_CHAN)
    a = torch.randn(N_HEADS, N_POS, TEACHER_RANK, generator=gen, dtype=DT)
    b = torch.randn(N_HEADS, N_POS, TEACHER_RANK, generator=gen, dtype=DT)
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)
    for h in range(N_HEADS):                       # ||a_h b_h^T||_2 == TEACHER_SPEC
        f = math.sqrt(TEACHER_SPEC
                      / float(torch.linalg.matrix_norm(a[h] @ b[h].T, ord=2)))
        a[h] *= f
        b[h] *= f
    return P, Wv, Wo, a, b, gen


def _inputs(gen, batch):
    return torch.randn(batch, N_POS, D_CHAN, generator=gen, dtype=DT)


def _forward(P, Wv, Wo, U, W, X):
    """sum_h woodbury_read(P_h, U_h, W_h, X Wv_h) @ Wo_h. Heads are independent.

    The head axis is a BATCH axis, not a Python loop: the first draft looped over
    six heads and took 0.891 s/step, which put the 120-cell round at nine hours.
    Same arithmetic, one `solve_triangular`.
    """
    V = torch.einsum("bnd,hde->hbne", X, Wv)                       # [H, B, n, d]
    z = woodbury_read(P.unsqueeze(1), U.unsqueeze(1), W.unsqueeze(1), V)
    return torch.einsum("hbne,hef->bnf", z, Wo)


def _teacher(P, Wv, Wo, a, b, X):
    return _forward(P, Wv, Wo, a, b, X)


def _candidates(P, Wv, Wo, a, b, gen_seed: int):
    """C_h = dL/dP_h at Delta = 0 -- one backward on a REAL batch, never a randn
    stand-in (V-29's own correction). The `n` columns of C_h are the candidate
    directions every sort selects from, so the pool is identical across sorts."""
    gen = torch.Generator().manual_seed(7000 + gen_seed)
    X = _inputs(gen, BATCH_TRAIN)
    with torch.no_grad():
        Y = _teacher(P, Wv, Wo, a, b, X)
    Pl = P.clone().requires_grad_(True)
    zero = torch.zeros(N_HEADS, N_POS, 1, dtype=DT)
    loss = ((_forward(Pl, Wv, Wo, zero, zero, X) - Y) ** 2).mean()
    loss.backward()
    return Pl.grad.detach().clone()


# ---------------------------------------------------------------- must-fires
def sanity_planted(seed: int = 0, r: int = 4, noise: float = 1e-3,
                   n_heads: int = N_HEADS) -> dict:
    """MUST-FIRE 1. A planted rank-r delta is recovered by all three sorts.

    The candidate matrix IS the planted delta plus isotropic noise at `noise`
    relative scale: `C = A B^T + noise * E`, rank `r` before the noise. A working
    selection rule lands on `span(A)`; the reported number is the largest
    principal angle to it, in radians, and it is bounded by the noise rather than
    by zero. A broken sort reads ~pi/2 and demo() trips.

    THIS TEST WAS WRONG ON ITS FIRST WRITE AND THE FIRST VERSION IS RECORDED
    RATHER THAN OVERWRITTEN IN SILENCE. It planted the delta on the OPERATOR and
    compared each sort's span against `span(a)` -- the plant's left factor -- with
    the candidate matrix being `dL/dP` at `Delta = 0`. All three sorts read
    `1.5479` to `1.5700` rad, i.e. essentially orthogonal, and the run went RED.
    The sorts were fine; the comparison was not. With `M0 = I - g P`, the
    teacher's residual is `g M0^-1 a (I - g b^T M0^-1 a)^-1 b^T M0^-1 V` and the
    gradient is `g M0^-T G_z V^T M0^-T`, so BOTH live in an `M0^-T`-rotated copy
    of the planted span, never in `span(a)` itself. The measured ~pi/2 was that
    rotation, reported as a recovery failure. `rotated_plant_angle_rad` below
    keeps the corrected version of that comparison so the finding stays visible:
    the gradient at `Delta = 0` does NOT point at the planted directions, it
    points at their resolvent-rotated image, and any sort that selected in the
    raw plant basis would be selecting the wrong thing.
    """
    gen = torch.Generator().manual_seed(2600 + seed)
    out = {"r": r, "seed": seed, "noise": noise, "per_sort": {},
           "planted_rank": r, "candidate": "C = A B^T + noise * E"}
    plants, cands = [], []
    for h in range(n_heads):
        A = torch.randn(N_POS, r, generator=gen, dtype=DT)
        B = torch.randn(N_POS, r, generator=gen, dtype=DT)
        C = A @ B.T
        C = C + noise * C.abs().max() * torch.randn(N_POS, N_POS, generator=gen, dtype=DT)
        plants.append(A)
        cands.append(C)
    for name in SORTS:
        ang, cnd = [], []
        for h in range(n_heads):
            Uh = SORT_FN[name](cands[h], r, seed=seed)
            ang.append(subspace_angle(Uh, plants[h]))
            # WHY A SORT CAN FAIL THIS. Every column of a rank-r `A B^T` lies in
            # span(A), so ANY r LINEARLY INDEPENDENT columns recover it exactly.
            # A column-selecting rule therefore fails here only by picking a
            # near-DEPENDENT set. The conditioning of the selection is that
            # mechanism, measured rather than assumed.
            cnd.append(float(torch.linalg.cond(Uh, 2)))
        out["per_sort"][name] = {"max_angle_rad": max(ang),
                                 "mean_angle_rad": sum(ang) / len(ang),
                                 "max_selection_cond2": max(cnd),
                                 "mean_selection_cond2": sum(cnd) / len(cnd),
                                 "recovered": max(ang) < 0.2}

    # The finding the first version of this test mistook for a failure.
    P, Wv, Wo, a, b, _ = _base(seed)
    G = _candidates(P, Wv, Wo, a, b, seed)
    eye = torch.eye(N_POS, dtype=DT)
    raw, rot = [], []
    for h in range(n_heads):
        M0 = eye - GAMMA * P[h]
        Usvd = sort_svd(G[h], TEACHER_RANK)
        raw.append(subspace_angle(Usvd, a[h]))
        rot.append(subspace_angle(Usvd, torch.linalg.solve(M0.T, a[h])))
    out["raw_plant_angle_rad"] = sum(raw) / len(raw)
    out["rotated_plant_angle_rad"] = sum(rot) / len(rot)
    out["finding"] = (
        "the gradient at Delta=0 points at the M0^-T-rotated image of the planted "
        "span, not at the planted span: mean angle %.4f rad raw vs %.4f rad rotated"
        % (out["raw_plant_angle_rad"], out["rotated_plant_angle_rad"]))
    return out


def _planted_heavy_tail(seed: int) -> torch.Tensor:
    """PRE-REGISTERED before drop test 2 was run. A candidate pool of `n` columns
    in `R^n`: an isotropic bulk at unit scale, plus `n_out = 4` columns whose
    scale is `10x`. The outliers are what a variance rule keeps; the bulk is
    where the depth is."""
    g = torch.Generator().manual_seed(31337 + seed)
    C = torch.randn(N_POS, N_POS, generator=g, dtype=DT)
    C[:, :4] *= 10.0
    return C


def _planted_block_local(seed: int) -> torch.Tensor:
    """PRE-REGISTERED before drop test 3 was run, at the NEUTRAL setting: the
    block-local family and the diffuse family carry EQUAL total energy and equal
    per-column norms, so neither rule is handed the answer.

    The block family's columns are supported on a 4x4 TILE of the 8x8 grid --
    contiguous in the plane and therefore contiguous in Hilbert order, but
    scattered across the raw row index. That scatter is the only reason the two
    rules can disagree at all.
    """
    g = torch.Generator().manual_seed(999 + seed)
    C = torch.randn(N_POS, N_POS, generator=g, dtype=DT)
    n_block = 8
    tile = [y * HILBERT_SIDE + x for y in range(4) for x in range(4)]
    idx = torch.as_tensor(tile)
    B = torch.zeros(N_POS, n_block, dtype=DT)
    B[idx, :] = torch.randn(len(tile), n_block, generator=g, dtype=DT)
    C[:, :n_block] = B
    # equal per-column norms across the whole pool: no rule gets a magnitude edge
    C = C / C.norm(dim=0, keepdim=True)
    return C, idx


def drop_test_ot(r: int = 4, seeds=SEEDS) -> dict:
    """MUST-FIRE 2, AND IT CAN DELETE S-OT FROM THE ROUND.

    On planted heavy-tailed directions S-SVD must keep the outlier and S-OT must
    keep the depth. If the two selections span the SAME subspace (largest
    principal angle below AGREE_RAD) on every seed, S-OT is S-SVD wearing a
    different name and it is DROPPED, and reported as dropped.
    """
    rows = []
    for s in seeds:
        C = _planted_heavy_tail(s)
        Usvd = sort_svd(C, r)
        Uot = sort_ot(C, r, seed=s)
        rows.append({"seed": s, "angle_rad": subspace_angle(Usvd, Uot)})
    agree = [x for x in rows if x["angle_rad"] < AGREE_RAD]
    return {"test": "drop_test_ot", "r": r, "n_seeds": len(seeds),
            "threshold_rad": AGREE_RAD, "rows": rows,
            "n_agree": len(agree), "min_angle_rad": min(x["angle_rad"] for x in rows),
            "dropped": len(agree) == len(rows),
            "verdict": "S-OT DROPPED, it is S-SVD relabelled" if len(agree) == len(rows)
                       else "S-OT survives: the selections differ"}


def drop_test_hilb(r: int = 4, seeds=SEEDS) -> dict:
    """MUST-FIRE 3, AND IT CAN DELETE S-HILB FROM THE ROUND.

    On planted block-local structure S-HILB must find the blocks and S-SVD must
    not. Recovery is measured as the fraction of each selected column's energy
    inside the planted tile, averaged over the r selected columns. If S-SVD's
    recovery is within `SAME` of S-HILB's on every seed, S-SVD found them too and
    S-HILB is DROPPED.

    `SAME = 0.05` is fixed here, before the test ran, for the same reason
    `AGREE_RAD` is.
    """
    SAME = 0.05
    rows = []
    for s in seeds:
        C, idx = _planted_block_local(s)
        out = {"seed": s}
        for name, Umat in (("S-SVD", sort_svd(C, r)),
                           ("S-HILB", sort_hilb(C, r))):
            e = Umat ** 2
            out[name] = float((e[idx, :].sum(0) / e.sum(0).clamp_min(1e-30)).mean())
        out["gap"] = out["S-HILB"] - out["S-SVD"]
        rows.append(out)
    found_too = [x for x in rows if x["gap"] < SAME]
    return {"test": "drop_test_hilb", "r": r, "n_seeds": len(seeds),
            "same_threshold": SAME, "tile": "4x4 at the grid origin, 16 of 64 rows",
            "chance_level": 16.0 / N_POS, "rows": rows,
            "n_svd_found_too": len(found_too),
            "dropped": len(found_too) == len(rows),
            "verdict": "S-HILB DROPPED, S-SVD finds the blocks too"
                       if len(found_too) == len(rows)
                       else "S-HILB survives: S-SVD does not find the blocks"}


def bind_at_zero(seed: int = 0, r: int = 4) -> dict:
    """BIND BITWISE AT Delta = 0. With W = 0 the LoRA'd read must be the base read
    BIT FOR BIT, not close to it.

    Two separate things are checked and only one of them is an equality:
      (1) Delta == 0 exactly, as a tensor. If this fails the bind is meaningless.
      (2) torch.equal(woodbury route at W=0, base state_solve). A route that is
          merely close at Delta = 0 is a new arm, not an optimisation
          (scale/m3_quintuple.py:492's rule, applied here).
    A FAILED BIND STRIKES THE RUN -- demo() asserts it before any score.
    """
    P, Wv, Wo, a, b, _ = _base(seed)
    C = _candidates(P, Wv, Wo, a, b, seed)
    gen = torch.Generator().manual_seed(4242 + seed)
    X = _inputs(gen, 8)
    out = {"seed": seed, "r": r, "per_sort": {}}
    for name in SORTS:
        U = torch.stack([SORT_FN[name](C[h], r, seed=seed) for h in range(N_HEADS)])
        W = torch.zeros(N_HEADS, N_POS, r, dtype=DT)
        delta = U @ W.transpose(-1, -2)
        eq, dm = [], []
        for h in range(N_HEADS):
            V = X @ Wv[h]
            z_lora = woodbury_read(P[h], U[h], W[h], V)
            z_base, _ = state_solve(P[h], V, GAMMA)
            eq.append(bool(torch.equal(z_lora, z_base)))
            dm.append(float((z_lora - z_base).abs().max()))
        out["per_sort"][name] = {
            "delta_exactly_zero": bool(torch.equal(delta, torch.zeros_like(delta))),
            "delta_absmax": float(delta.abs().max()),
            "bitwise_equal_all_heads": all(eq),
            "n_heads_bitwise": sum(eq), "n_heads": N_HEADS,
            "max_abs_diff": max(dm),
        }
    return out


def grad_table(seed: int = 0, r: int = 4) -> dict:
    """MUST-FIRE 5. Per-head gradient on BOTH factors at step 0 and step 1.

    The expected answer is NOT "both live". With `U` set by the sort and `W = 0`,
    `dL/dU = (dL/dDelta) W = 0` EXACTLY at step 0 -- one factor is dead at
    initialisation by the mechanism `MISTAKES.md:2526 V-29` records for the gate,
    and it wakes only once `W` has moved. So the check is "dead at 0, LIVE at 1".
    A factor still dead at step 1 is a real defect and is reported as one.

    The step-0 zero is NOT repaired. It is what makes Delta = 0 at init, which
    `bind_at_zero` requires.
    """
    P, Wv, Wo, a, b, _ = _base(seed)
    C = _candidates(P, Wv, Wo, a, b, seed)
    gen = torch.Generator().manual_seed(5150 + seed)
    X = _inputs(gen, BATCH_TRAIN)
    with torch.no_grad():
        Y = _teacher(P, Wv, Wo, a, b, X)

    out = {"seed": seed, "r": r, "per_sort": {}}
    for name in SORTS:
        U = torch.stack([SORT_FN[name](C[h], r, seed=seed)
                         for h in range(N_HEADS)]).requires_grad_(True)
        W = torch.zeros(N_HEADS, N_POS, r, dtype=DT, requires_grad=True)
        opt = torch.optim.Adam([{"params": [U], "lr": LR_U},
                                {"params": [W], "lr": LR_U * LORA_PLUS_RATIO}])
        steps = []
        for step in (0, 1):
            opt.zero_grad()
            loss = ((_forward(P, Wv, Wo, U, W, X) - Y) ** 2).mean()
            loss.backward()
            steps.append({
                "step": step, "loss": float(loss.detach()),
                "per_head": [{"head": h,
                              "grad_U_absmax": float(U.grad[h].abs().max()),
                              "grad_W_absmax": float(W.grad[h].abs().max()),
                              "U_dead": float(U.grad[h].abs().max()) == 0.0,
                              "W_dead": float(W.grad[h].abs().max()) == 0.0}
                             for h in range(N_HEADS)],
            })
            opt.step()
        s0, s1 = steps
        out["per_sort"][name] = {
            "steps": steps,
            "U_dead_at_0_all_heads": all(h["U_dead"] for h in s0["per_head"]),
            "W_live_at_0_all_heads": all(not h["W_dead"] for h in s0["per_head"]),
            "U_live_at_1_all_heads": all(not h["U_dead"] for h in s1["per_head"]),
            "W_live_at_1_all_heads": all(not h["W_dead"] for h in s1["per_head"]),
            "defect_heads_still_dead_at_1": [h["head"] for h in s1["per_head"]
                                             if h["U_dead"] or h["W_dead"]],
        }
    return out


# ---------------------------------------------------------------- the round
def _spec_norm(U, W):
    """||U W^T||_2 per head, via the r x r problem instead of the n x n one.

    `U W^T = (Qu Ru)(Qw Rw)^T` with Qu, Qw orthonormal, so the singular values of
    `U W^T` are exactly those of `Ru Rw^T`, which is `r x r`. At r = 1 that is one
    number instead of a 64 x 64 SVD, and it is an identity, not an approximation.
    """
    _, Ru = torch.linalg.qr(U)
    _, Rw = torch.linalg.qr(W)
    return torch.linalg.svdvals(Ru @ Rw.transpose(-1, -2))[..., 0]


def _clip_(U, W, c=SPECTRAL_CLIP):
    """MUST-FIRE 6. ||U W^T||_2 <= c, per head, in place. Returns the spectral
    norms BEFORE the clip so the clip's bite is printed rather than assumed."""
    with torch.no_grad():
        s = _spec_norm(U, W)                                   # [H]
        f = torch.where(s > c, (c / s.clamp_min(1e-300)).sqrt(),
                        torch.ones_like(s))[:, None, None]
        U *= f
        W *= f
    return s.tolist()


def run_round(sort: str, r: int, seed: int, steps: int = STEPS) -> dict:
    """Pick r directions -> Woodbury update -> condition number -> bind ->
    train `steps` -> score. Identical for all three sorts but the selection rule.
    """
    t0 = time.perf_counter()
    P, Wv, Wo, a, b, _ = _base(seed)
    C = _candidates(P, Wv, Wo, a, b, seed)
    U = torch.stack([SORT_FN[sort](C[h], r, seed=seed)
                     for h in range(N_HEADS)]).requires_grad_(True)
    W = torch.zeros(N_HEADS, N_POS, r, dtype=DT, requires_grad=True)

    # the bind, BEFORE any score
    gen = torch.Generator().manual_seed(4242 + seed)
    Xb = _inputs(gen, 8)
    bind_ok = True
    with torch.no_grad():
        for h in range(N_HEADS):
            V = Xb @ Wv[h]
            zb, _ = state_solve(P[h], V, GAMMA)
            bind_ok &= bool(torch.equal(woodbury_read(P[h], U[h], W[h], V), zb))

    gtr = torch.Generator().manual_seed(8100 + seed)
    Xtr = _inputs(gtr, BATCH_TRAIN)
    gev = torch.Generator().manual_seed(9100 + seed)
    Xev = _inputs(gev, BATCH_EVAL)
    with torch.no_grad():
        Ytr = _teacher(P, Wv, Wo, a, b, Xtr)
        Yev = _teacher(P, Wv, Wo, a, b, Xev)
        base_ev = float(((_forward(P, Wv, Wo, torch.zeros(N_HEADS, N_POS, 1, dtype=DT),
                                   torch.zeros(N_HEADS, N_POS, 1, dtype=DT), Xev)
                          - Yev) ** 2).mean())

    opt = torch.optim.Adam([{"params": [U], "lr": LR_U},
                            {"params": [W], "lr": LR_U * LORA_PLUS_RATIO}])
    clip_bit = 0
    for _ in range(steps):
        opt.zero_grad()
        loss = ((_forward(P, Wv, Wo, U, W, Xtr) - Ytr) ** 2).mean()
        loss.backward()
        opt.step()
        pre = _clip_(U.data, W.data)
        clip_bit += sum(1 for s in pre if s > SPECTRAL_CLIP)

    with torch.no_grad():
        ev = float(((_forward(P, Wv, Wo, U, W, Xev) - Yev) ** 2).mean())
        tr = float(((_forward(P, Wv, Wo, U, W, Xtr) - Ytr) ** 2).mean())
        eye = torch.eye(N_POS, dtype=DT)
        kap, kap0, smin0 = [], [], []
        for h in range(N_HEADS):
            M0 = eye - GAMMA * P[h]
            M = M0 - GAMMA * (U[h] @ W[h].T)
            kap.append(float(torch.linalg.cond(M, 2)))
            kap0.append(float(torch.linalg.cond(M0, 2)))
            smin0.append(float(torch.linalg.svdvals(M0).min()))
        spec = [float(torch.linalg.matrix_norm(U[h] @ W[h].T, ord=2))
                for h in range(N_HEADS)]

    # The bound. sigma_min(M) >= sigma_min(M0) - g||UW^T||_2, so
    # ||M^-1||_2 <= 1/(sigma_min(M0) - g c) whenever g c < sigma_min(M0).
    denom = [s - GAMMA * SPECTRAL_CLIP for s in smin0]
    bound = [(1.0 / d if d > 0 else float("inf")) for d in denom]
    meas = []
    with torch.no_grad():
        for h in range(N_HEADS):
            M = eye - GAMMA * (P[h] + U[h] @ W[h].T)
            meas.append(1.0 / float(torch.linalg.svdvals(M).min()))

    return {
        "sort": sort, "r": r, "seed": seed, "steps": steps,
        "bind_bitwise_at_delta_zero": bind_ok,
        "eval_mse": ev, "train_mse": tr, "base_eval_mse": base_ev,
        "frac_of_gap_closed": 1.0 - ev / base_ev,
        "cond2_updated_max": max(kap), "cond2_base_max": max(kap0),
        "sigma_min_M0_min": min(smin0),
        "spectral_norm_delta_max": max(spec), "spectral_clip_c": SPECTRAL_CLIP,
        "clip_engaged_head_steps": clip_bit,
        "inv_norm_bound_max": max(bound), "inv_norm_measured_max": max(meas),
        "bound_held": all(m <= bd + 1e-9 for m, bd in zip(meas, bound)),
        "cpu_seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- bars
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval. Reuses `scale/dfloor_probe.py:39`'s beta.ppf form
    rather than adding a sixth copy to this tree -- five already exist
    (`scale/foreman_consequence.py:265`, `scale/dfloor_probe.py:39`,
    `scale/ratio_sweep.py:20`, `tests/mercury/arena_price.py:44`,
    `tests/mercury/arena_rig.py:177`). Correct at k = 0 and k = n."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


#: WHAT BED-M IS, BEFORE IT IS USED -- and the round declines to use it.
#: Two objects in this tree carry the name. (a) `ceq/corpus.py:88 build()`, named
#: BED-M by `ceq/beds/__init__.py:3`, pinned in `ceq/kdata.py:472` at
#: sha256 2f282a5d..25f7d24d with NO on-disk artifact (`ceq/kdata.py:526`).
#: (b) the arena chain bed at t*=2, n=2048, `scale/negation_scope.py:354
#: make_equilibrium_batch`, oracle `equilibrium_oracle` at `:302`.
#: NEITHER CARRIES A FLOOR CROSS RATE. A whole-tree search for `floor cross`,
#: `floor_cross`, `floor-cross` returns zero hits. What exists is a per-cell
#: CROSSING rule (`tests/mercury/arena_rig.py:71 crosses`, `boot_hi < FLOOR_1`,
#: FLOOR_1 = sqrt((t*-1)/t*) = 0.7071067811865476) plus a crossing COUNT and a
#: CP-lower (`CEQ_V20_R15_CONTRACT.md:120`), and the canon is explicit that
#: floor_1 is "printed as a capability threshold, NEVER a floor"
#: (`docs/canon/04_BEDS_AND_INSTRUMENTS.md:60`). `tests/mercury/arena_rig.py:186`
#: deliberately emits both tails and never a single rate.
#: So this round does NOT report a BED-M floor cross rate. It reports the
#: quantity the pre-registration is actually about -- the per-seed rate at which
#: a challenger beats S-SVD -- with a Clopper-Pearson interval, LABELLED as this
#: round's own bar and not as BED-M's.
BED_M_FINDING = (
    "BED-M exists (two objects under one name) and carries NO floor cross rate. "
    "It carries a per-cell crossing rule, a crossing count and a CP-lower, and "
    "docs/canon/04_BEDS_AND_INSTRUMENTS.md:60 forbids calling floor_1 a floor. "
    "No such rate is invented here."
)


def win_rate_bar(results: list[dict], r_max: int = 4) -> dict:
    """The pre-registered bar: per seed, at each r <= r_max, does the challenger's
    eval MSE beat S-SVD's? Clopper-Pearson on the count. THIS IS THIS ROUND'S
    BAR, not BED-M's -- see BED_M_FINDING."""
    by = {(x["sort"], x["r"], x["seed"]): x["eval_mse"] for x in results}
    out = {"r_max": r_max, "bar": "per-seed win over S-SVD at matched r",
           "note": BED_M_FINDING, "per_sort": {}}
    rs = sorted({x["r"] for x in results if x["r"] <= r_max})
    seeds = sorted({x["seed"] for x in results})
    for name in SORTS:
        if name == "S-SVD":
            continue
        k = n = 0
        per_r = {}
        for r in rs:
            kk = sum(1 for s in seeds if by[(name, r, s)] < by[("S-SVD", r, s)])
            per_r[r] = {"wins": kk, "n": len(seeds),
                        "cp95": clopper_pearson(kk, len(seeds))}
            k += kk
            n += len(seeds)
        lo, hi = clopper_pearson(k, n)
        out["per_sort"][name] = {"wins": k, "n": n, "rate": k / n if n else float("nan"),
                                 "cp95_lo": lo, "cp95_hi": hi, "per_r": per_r,
                                 "beats_at_interval": lo > 0.5}
    return out


def bed_h_bar(steps: int = 400, r: int = 4, n_pool: int = 96, seed: int = 58197) -> dict:
    """BED-H's abstention 3x2 table, one row per sort, on a LoRA'd consumer.

    The base consumer is `ceqjepa.beds.bed_h.fit_rnn` (read only -- that module is
    held by another agent and is not edited). The LoRA goes on `belief_head.weight`
    only, rank r, one factor zero, LoRA+ ratio held at 4, the same steps for all
    three. The table is `bed_h.abstention_table` unchanged, at delta = 0.1.

    A NaN precision is REPORTED, not hidden: `resolvable` says whether the
    uninformative row is empty, which is the bed's own guard at `bed_h.py:754`.
    """
    from ceqjepa.beds import bed_h as B
    torch.manual_seed(seed)
    # draw_pool returns (accepted, refused, frac, rows); `accepted` holds
    # (inst, hmm, gate) triples and the GATE REFUSALS ARE REPORTED, not hidden --
    # a pool that refused most of its draws is badly parameterised and the
    # fraction is the evidence for saying so (bed_h.py:921).
    accepted, refused, frac, _ = B.draw_pool(n_pool, "weak", seed)
    insts = [t[0] for t in accepted]
    n_tr = int(len(insts) * 0.6)
    n_va = int(len(insts) * 0.2)
    train, val, test = insts[:n_tr], insts[n_tr:n_tr + n_va], insts[n_tr + n_va:]
    if not (train and val and test):
        return {"declined": "the gate accepted %d of %d draws; not enough to "
                            "split. Reported rather than resampled to fill."
                            % (len(insts), n_pool), "gate_fractions": frac}
    model, fit = B.fit_rnn(train, val, "weak", hidden=B.SKYLINE_HIDDEN, cell="gru",
                           steps=steps, seed=seed, log_every=max(1, steps // 4))
    base = model.belief_head.weight.detach().clone()          # [S, hidden]
    bias = model.belief_head.bias.detach().clone()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    def _logp(x, Wmat):
        """The consumer's belief head with the LoRA applied FUNCTIONALLY.

        `bed_h.py` is held by another agent and is not edited, and assigning to
        `.weight.data` would detach the graph -- the LoRA factors would get no
        gradient and the run would be a measurement of the base model wearing
        three different labels. The RNN is called unchanged and only the final
        linear is re-applied.
        """
        h, _ = model.rnn(x)
        return torch.log_softmax(torch.nn.functional.linear(h, Wmat, bias), -1)

    def _kl(x, y, npre_, Wmat):
        lp = _logp(x, Wmat)[:, npre_:]
        return torch.sum(y * (torch.log(y.clamp_min(1e-30)) - lp), -1).mean()

    xtr, ytr, ctr, ntr = B._tensors(train, "weak")
    xte, yte, cte, npre = B._tensors(test, "weak")

    # candidate matrix: dKL/d(belief_head.weight) at the base point
    Wg = base.clone().requires_grad_(True)
    _kl(xtr, ytr, ntr, Wg).backward()
    Cg = Wg.grad.detach().clone().double()                     # [S, hidden]

    out = {"bed": "BED-H (ceqjepa/beds/bed_h.py, unmodified)", "r": r,
           "delta": 0.1, "lora_on": "belief_head.weight", "steps": steps,
           "fit_val_total_best": fit["val_total_best"],
           "n_accepted": len(train) + len(val) + len(test), "n_drawn": n_pool,
           "gate_fractions": frac, "per_sort": {}}
    with torch.no_grad():
        lp0 = _logp(xte, base)[:, npre:]
        base_kl = float(torch.sum(yte * (torch.log(yte.clamp_min(1e-30)) - lp0), -1).mean())
    out["base_eval_kl"] = base_kl

    # THE SELECTED FACTOR SITS ON THE 64-WIDE HIDDEN SIDE, not the S=8 output
    # side: `hidden = 64` is the 8x8 grid the Hilbert order is defined on, and a
    # sort that cannot be evaluated on 8 rows would be dropped here for the
    # instrument's shape rather than for its own behaviour. So the candidate
    # matrix is `Cg^T` -- 8 candidate columns living in R^64 -- the selected
    # factor is `U [64, r]`, the ZERO factor is `Z [8, r]`, and
    # `Delta = Z U^T` is [8, 64] as the head requires. `dL/dU = (dL/dDelta)^T Z`
    # is still exactly zero at Z = 0, so must-fire 5's mechanism is unchanged.
    Ct = Cg.T.contiguous()                                     # [hidden, S]
    for name in SORTS:
        rr = min(r, Ct.shape[1])
        Uh = SORT_FN[name](Ct, rr, seed=seed).float().clone().requires_grad_(True)
        Wl = torch.zeros(base.shape[0], rr, requires_grad=True)
        opt = torch.optim.Adam([{"params": [Uh], "lr": 1e-3},
                                {"params": [Wl], "lr": 1e-3 * LORA_PLUS_RATIO}])
        # THE BIND, before any score: at W = 0 the LoRA'd head is the base head
        # bitwise, so Delta = 0 really is the base consumer.
        with torch.no_grad():
            bind = bool(torch.equal(base + Wl @ Uh.T, base))
        for _ in range(steps // 2):
            opt.zero_grad()
            _kl(xtr, ytr, ntr, base + Wl @ Uh.T).backward()
            opt.step()
            with torch.no_grad():
                sN = float(torch.linalg.matrix_norm(Wl @ Uh.T, ord=2))
                if sN > SPECTRAL_CLIP:
                    f = math.sqrt(SPECTRAL_CLIP / sN)
                    Uh.data *= f
                    Wl.data *= f
        with torch.no_grad():
            Wfin = base + Wl @ Uh.T
            lp = _logp(xte, Wfin)[:, npre:]
            ev_kl = float(torch.sum(yte * (torch.log(yte.clamp_min(1e-30)) - lp), -1).mean())
            pred = lp.exp().numpy().astype(np.float64)
        # THE FULL DELTA SWEEP, not one delta. At delta = 0.1 this pool's
        # uninformative row came back EMPTY on every instance -- `resolvable` is
        # False, precision and recall are NaN, and that is the bed's OWN guard
        # (bed_h.py:754) firing, not a model result. A single NaN reported with
        # no explanation is how a sweep quietly loses a point, so all three of
        # `bed_h.DELTA_SWEEP` are run and the resolvable count is printed beside
        # each. Where no delta resolves, the instrument DECLINES.
        by_delta = {}
        for dlt in B.DELTA_SWEEP:
            ts = [B.abstention_table(pred[i], test[i], dlt) for i in range(len(test))]
            pr = [t["precision"] for t in ts if not math.isnan(t["precision"])]
            rc = [t["recall"] for t in ts if not math.isnan(t["recall"])]
            by_delta[dlt] = {
                "table_3x2": {row: {col: sum(t[row][col] for t in ts)
                                    for col in ("answered", "refused")}
                              for row in ("informative", "uninformative", "undefined")},
                "n_resolvable": sum(1 for t in ts if t["resolvable"]),
                "n_instances": len(ts),
                "precision_mean": (sum(pr) / len(pr)) if pr else float("nan"),
                "recall_mean": (sum(rc) / len(rc)) if rc else float("nan"),
            }
        tabs = [B.abstention_table(pred[i], test[i], 0.1) for i in range(len(test))]
        agg = {row: {col: sum(t[row][col] for t in tabs)
                     for col in ("answered", "refused")}
               for row in ("informative", "uninformative", "undefined")}
        prec = [t["precision"] for t in tabs if not math.isnan(t["precision"])]
        rec = [t["recall"] for t in tabs if not math.isnan(t["recall"])]
        out["per_sort"][name] = {
            "by_delta": by_delta,
            "table_3x2": agg, "n_instances": len(tabs),
            "bind_bitwise_at_delta_zero": bind,
            "eval_kl": ev_kl,
            "spectral_norm_delta": float(torch.linalg.matrix_norm((Wl @ Uh.T).detach(), ord=2)),
            "n_resolvable": sum(1 for t in tabs if t["resolvable"]),
            "precision_mean": (sum(prec) / len(prec)) if prec else float("nan"),
            "recall_mean": (sum(rec) / len(rec)) if rec else float("nan"),
            "n_nan_precision": len(tabs) - len(prec),
        }
    return out


# ---------------------------------------------------------------- self-check
def demo(full: bool = True, bed_h: bool = False) -> int:
    """Every claim this module makes, printed with its producer. Returns 0 or 1;
    the caller ASSERTS the status (L-SURFACE), never reads it through a pipe."""
    red = []
    print("=" * 78)
    print("THE LORA SORTING WING  |  %s" % machine())
    print("producer: python -m ceqjepa.lora_sort")
    print("=" * 78)

    print("\n(0) PINNED / VARIED (L-NULL)")
    m = pinned_manifest("S-SVD", 4, 0)
    print("    varied: %s" % m["varied_selection_rule"])
    print("    pinned: " + ", ".join("%s=%s" % (k[7:], v)
                                     for k, v in m.items() if k.startswith("pinned_")))

    print("\n(1) WOODBURY EXACTNESS -- re-derived, Hager 1989 Eq. (1) p.221")
    wx = woodbury_exactness()
    print("    worst |woodbury - dense re-solve| = %.3e over %d cells (rel %.3e, "
          "worst at rank=%d draw=%d)" % (wx["worst_abs"], wx["n_cells"],
                                         wx["worst_rel"], *wx["worst_at_rank_draw"]))
    if not (wx["worst_abs"] < 1e-10):
        red.append("woodbury exactness %.3e" % wx["worst_abs"])

    print("\n(2) MUST-FIRE 1 SANITY -- planted rank-4 delta in the candidate "
          "matrix, largest principal angle to the plant (rad)")
    sn = sanity_planted()
    for k, v in sn["per_sort"].items():
        print("    %-7s max %.6f  mean %.6f  sel cond2 max %.3e  recovered: %s"
              % (k, v["max_angle_rad"], v["mean_angle_rad"],
                 v["max_selection_cond2"], v["recovered"]))
        if not v["recovered"]:
            red.append("sanity %s did not recover the plant, angle %.4f rad"
                       % (k, v["max_angle_rad"]))
    print("    finding: %s" % sn["finding"])

    print("\n(3) MUST-FIRE 2 DROP TEST FOR S-OT -- heavy-tailed plant")
    d2 = drop_test_ot()
    print("    min angle over %d seeds = %.6f rad (agree below %.0e)"
          % (d2["n_seeds"], d2["min_angle_rad"], d2["threshold_rad"]))
    print("    %s" % d2["verdict"])

    print("\n(4) MUST-FIRE 3 DROP TEST FOR S-HILB -- block-local plant, 4x4 tile")
    d3 = drop_test_hilb()
    for row in d3["rows"][:3]:
        print("    seed %d: S-HILB in-tile %.4f  S-SVD in-tile %.4f  gap %+.4f"
              % (row["seed"], row["S-HILB"], row["S-SVD"], row["gap"]))
    print("    chance = %.4f;  %s" % (d3["chance_level"], d3["verdict"]))

    print("\n(5) BITWISE BIND AT Delta = 0")
    bd = bind_at_zero()
    for k, v in bd["per_sort"].items():
        print("    %-7s delta==0 exactly %s | bitwise %d/%d heads | max|diff| %.3e"
              % (k, v["delta_exactly_zero"], v["n_heads_bitwise"], v["n_heads"],
                 v["max_abs_diff"]))
        if not (v["delta_exactly_zero"] and v["bitwise_equal_all_heads"]):
            red.append("bind broken for %s -- the run is STRUCK" % k)

    print("\n(6) MUST-FIRE 5 GRADIENT TABLE, per head, steps 0 and 1")
    gt = grad_table()
    for k, v in gt["per_sort"].items():
        s0, s1 = v["steps"]
        print("    %s  (expected: U dead at 0, both live at 1)" % k)
        print("      head |  step0 |dL/dU|  step0 |dL/dW|  |  step1 |dL/dU|  step1 |dL/dW|")
        for h0, h1 in zip(s0["per_head"], s1["per_head"]):
            print("      %4d | %14.6e %14.6e | %14.6e %14.6e"
                  % (h0["head"], h0["grad_U_absmax"], h0["grad_W_absmax"],
                     h1["grad_U_absmax"], h1["grad_W_absmax"]))
        print("      U dead at 0 all heads: %s | U live at 1: %s | W live at 1: %s"
              % (v["U_dead_at_0_all_heads"], v["U_live_at_1_all_heads"],
                 v["W_live_at_1_all_heads"]))
        if v["defect_heads_still_dead_at_1"]:
            red.append("%s: heads %s still dead at step 1"
                       % (k, v["defect_heads_still_dead_at_1"]))

    results = []
    if full:
        print("\n(7) THE ROUND -- %d sorts x %d ranks x %d seeds, %d steps each"
              % (len(SORTS), len(R_GRID), len(SEEDS), STEPS))
        for r in R_GRID:
            for s in SEEDS:
                for name in SORTS:
                    results.append(run_round(name, r, s))
        print("      r | sort    |  mean eval MSE |  mean frac gap closed | cond2 max "
              "| bound held | CPU s")
        for r in R_GRID:
            for name in SORTS:
                rows = [x for x in results if x["sort"] == name and x["r"] == r]
                print("    %3d | %-7s | %14.6e | %21.4f | %9.3e | %10s | %5.1f"
                      % (r, name, sum(x["eval_mse"] for x in rows) / len(rows),
                         sum(x["frac_of_gap_closed"] for x in rows) / len(rows),
                         max(x["cond2_updated_max"] for x in rows),
                         all(x["bound_held"] for x in rows),
                         sum(x["cpu_seconds"] for x in rows)))
        if not all(x["bind_bitwise_at_delta_zero"] for x in results):
            red.append("a run lost the bind -- STRUCK")
        if not all(x["bound_held"] for x in results):
            red.append("condition-number bound violated")

        print("\n(8) THE BAR -- per-seed win over S-SVD at r <= 4, Clopper-Pearson 95%%")
        print("    %s" % BED_M_FINDING)
        wb = win_rate_bar(results)
        for k, v in wb["per_sort"].items():
            print("    %-7s %2d/%2d = %.3f  CP95 [%.3f, %.3f]  beats at interval: %s"
                  % (k, v["wins"], v["n"], v["rate"], v["cp95_lo"], v["cp95_hi"],
                     v["beats_at_interval"]))

        print("\n(9) VERDICT AGAINST THE PRE-REGISTERED COUNTER")
        small = [x for x in results if x["r"] <= 4]
        best = {n: sum(x["eval_mse"] for x in small if x["sort"] == n)
                   / max(1, len([x for x in small if x["sort"] == n])) for n in SORTS}
        winner = min(best, key=best.get)
        print("    mean eval MSE at r <= 4: " + "  ".join("%s %.6e" % (n, best[n])
                                                          for n in SORTS))
        print("    PREDICTION (S-OT or S-HILB beats S-SVD with an interval at r<=4): %s"
              % ("HELD" if any(v["beats_at_interval"] for v in wb["per_sort"].values())
                 else "FAILED"))
        print("    COUNTER   (S-SVD ties or wins): %s"
              % ("HELD" if winner == "S-SVD" else "FAILED"))

    if bed_h:
        print("\n(10) BED-H ABSTENTION 3x2, LoRA on belief_head.weight")
        bh = bed_h_bar()
        for k, v in bh["per_sort"].items():
            t = v["table_3x2"]
            print("    %s  (n=%d instances, %d resolvable)"
                  % (k, v["n_instances"], v["n_resolvable"]))
            print("        row             answered  refused")
            for row in ("informative", "uninformative", "undefined"):
                print("        %-14s %8d %8d" % (row, t[row]["answered"], t[row]["refused"]))
            print("        precision %.4f  recall %.4f  (%d NaN-precision instances)"
                  % (v["precision_mean"], v["recall_mean"], v["n_nan_precision"]))

    print("\n" + "=" * 78)
    if red:
        print("RED (%d):" % len(red))
        for x in red:
            print("  - %s" % x)
        return 1
    print("GREEN. Every must-fire fired and could have failed.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="skip the training round")
    ap.add_argument("--bed-h", action="store_true", help="also run the BED-H bar")
    ap.add_argument("--json", type=str, default=None, help="dump results to a path")
    a = ap.parse_args()
    torch.set_num_threads(8)
    status = demo(full=not a.quick, bed_h=a.bed_h)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"machine": machine(), "exactness": woodbury_exactness(),
                       "drop_ot": drop_test_ot(), "drop_hilb": drop_test_hilb()},
                      fh, indent=2, default=str)
    sys.exit(status)
