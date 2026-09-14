"""Does the PER-COORDINATE corner assignment do measurable work in pi_jepa?

Every test here asserts a claim the project needs to be TRUE for the assignment
to be load-bearing. Each was written to be RED against the code as it stands.

The four claims under attack:

  1. the ASSIGNED beta vector beats a PERMUTED one (same betas, same refusals,
     shuffled across coordinates) on the winning frozen-encoder arm;
  2. it beats a CONSTANT one (all softmax, all linear) with the same columns;
  3. the pi_jepa Encoder's OWN coordinates carry the exponents the mask claims
     for them, alpha_d = 1 - beta_d, which is the only thing that could make
     the index-to-index transfer from pi_assign legitimate;
  4. the frozen arm's score needs the encoder network and needs the read; a
     bare fixed projection, a literal truncation of the raw bed, and a purely
     LINEAR read should all be materially worse.

Weights are bitwise identical across mask variants: build(seed, beta=...) seeds
torch before constructing, and no parameter shape depends on the mask, only the
buffers do. So every difference measured here IS the mask.
"""

import copy
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

import ceqjepa.pi_assign as pa
import ceqjepa.pi_jepa as pj


N_PERMUTATIONS = 12
PERM_SEED = 20260914


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def _nrmse(beta, seed=pj.SEED, train_encoder=False, encoder=None, predictor=None):
    """Held-out NRMSE for one arm. Mirrors pj.train_run + pj._held_out_scores.

    A collapse refusal comes back as the Refusal's own text, never as NaN and
    never as a silent default, so a caller cannot average it into a mean.
    """
    model = pj.build(seed, beta=beta)
    if encoder is not None:
        model.online = encoder
        model.target = copy.deepcopy(encoder)
        for p in model.target.parameters():
            p.requires_grad_(False)
    if predictor is not None:
        model.pred = predictor
    if not train_encoder:
        for p in model.online.parameters():
            p.requires_grad_(False)
    try:
        pj.fit(model, steps=pj.N_STEPS, nu=pj.NU, seed=seed, detector=True)
    except AssertionError as exc:
        return pj.Refusal("ARM-COLLAPSED", str(exc))
    return pj._held_out_scores(model, seed + 1)["nrmse"]


def _permutations(beta, n=N_PERMUTATIONS, seed=PERM_SEED):
    """n shuffles of the WHOLE 9-vector, refusals travelling with it.

    VOID UNDER L-NULL, and the docstring below is the reason it reads as sound.

    VARIED, named: which coordinate carries which corner; AND which coordinates
    are refused, because refusals travel with the shuffle. A refused coordinate
    is dropped from the read, so every row also re-selects which 7 of the 9
    encoder coordinates the read sees at all.
    PINNED, named: the corner multiset, the count of refusals, the seed, the
    encoder weights, the bed.

    Measured at PERM_SEED=20260914, commit 391a2d0, WIN-16QAL06O9GB: the
    assigned mask refuses {6, 7}, and 12 of 12 rows refuse some other pair.
    So no row isolates corner assignment. The refusal-pinned arrangement space
    is C(7, 4) = 35 and is sampled 0 times; the 12 rows are drawn from the 1260
    distinct arrangements of the whole multiset.

    A replacement null must pin the refused coordinates at {6, 7} and permute
    only the 4 ones among the 7 assigned positions. Until it exists, the claim
    this function licensed -- that the assignment carries no information -- is
    UNBOUND, and the kill it supports rests instead on the constant-corner
    comparison and on the encoder aggregating no context, both of which are
    produced by other tests in this file.

    The identity permutation is rejected, so no row is the assigned mask again.
    """
    rng = np.random.default_rng(seed)
    out, seen = [], {tuple(str(b) for b in beta)}
    while len(out) < n:
        p = rng.permutation(len(beta))
        cand = [beta[i] for i in p]
        key = tuple(str(b) for b in cand)
        if key in seen:
            continue
        seen.add(key)
        out.append((tuple(int(i) for i in p), cand))
    return out


def _constant(beta, value):
    """All coordinates at one corner, with R1's refusals left exactly where they
    are, so the column count and the target dimension are unchanged."""
    return [b if pj.is_refusal(b) else float(value) for b in beta]


class _BareProjection(nn.Module):
    """No hidden layers, no GELU: one fixed random linear map x -> s."""

    def __init__(self, x_dim=pj.X_DIM, d=pj.D_LATENT):
        super().__init__()
        self.net = nn.Linear(x_dim, d)


    def forward(self, x):
        return self.net(x)


class _Truncate(nn.Module):
    """No network at all: the first d coordinates of the raw bed."""

    def __init__(self, d=pj.D_LATENT):
        super().__init__()
        self.d = int(d)
        self.keep = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x[..., :self.d]


class _LinearRead(nn.Module):
    """The read with the attention removed: s_hat = W s, one matrix, no beta.

    If this matches the family's read on the frozen arm then the frozen arm is a
    random-feature model with a linear head and beta is not in the circuit.
    """

    def __init__(self, d=pj.D_LATENT):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)

    def forward(self, s, beta, a=None):
        cols = tuple(i for i, b in enumerate(beta) if not pj.is_refusal(b))
        return pj.ReadOut(
            out=self.w(s)[..., cols], cols=cols, axis=pj.MASK_AXIS,
            axis_len=len(beta),
            refusals=OrderedDict((i, b) for i, b in enumerate(beta)
                                 if pj.is_refusal(b)))


class _RealForward(pj.PiJepa):
    """PiJepa whose forward does not demand a complex read, for the linear arm."""

    def forward(self, x, a=None):
        return self.pred(self.online(x), self.beta_list(), a)


# ---------------------------------------------------------------------------
# 0. the control: this file reproduces the module's own published numbers
# ---------------------------------------------------------------------------

def test_control_reproduces_the_published_frozen_and_trained_scores():
    beta = pj.beta_vector()
    frozen = _nrmse(beta, train_encoder=False)
    trained = _nrmse(beta, train_encoder=True)
    assert abs(frozen - 0.6881) < 5e-4, frozen
    assert abs(trained - 0.8624) < 5e-4, trained


# ---------------------------------------------------------------------------
# 1. assigned vs permuted
# ---------------------------------------------------------------------------

def _permutation_table(train_encoder):
    beta = pj.beta_vector()
    assigned = _nrmse(beta, train_encoder=train_encoder)
    rows = []
    for perm, cand in _permutations(beta):
        rows.append((perm, _nrmse(cand, train_encoder=train_encoder)))
    scores = [s for _, s in rows if not pj.is_refusal(s)]
    refused = [(p, s) for p, s in rows if pj.is_refusal(s)]
    return assigned, rows, scores, refused


def test_assigned_mask_beats_every_permutation_frozen_arm():
    assigned, rows, scores, refused = _permutation_table(train_encoder=False)
    better = [(p, s) for p, s in rows
              if not pj.is_refusal(s) and s <= assigned]
    assert not better, (
        "assigned NRMSE %.4f is not the best corner-to-coordinate assignment: "
        "%d of %d permutations of the SAME betas score at least as well; best "
        "permutation %.4f at %r; permutation scores %s; refused arms %r"
        % (assigned, len(better), len(rows), min(scores),
           min(rows, key=lambda r: float("inf") if pj.is_refusal(r[1]) else r[1])[0],
           ["%.4f" % s for s in sorted(scores)], refused))


def test_assigned_mask_beats_every_permutation_trained_arm():
    assigned, rows, scores, refused = _permutation_table(train_encoder=True)
    better = [(p, s) for p, s in rows
              if not pj.is_refusal(s) and s <= assigned]
    assert not better, (
        "assigned NRMSE %.4f is not the best corner-to-coordinate assignment on "
        "the trained arm: %d of %d permutations score at least as well; best "
        "%.4f; scores %s; refused arms %r"
        % (assigned, len(better), len(rows), min(scores),
           ["%.4f" % s for s in sorted(scores)], refused))


# ---------------------------------------------------------------------------
# 2. assigned vs constant
# ---------------------------------------------------------------------------

def test_assigned_mask_beats_both_constant_masks():
    beta = pj.beta_vector()
    assigned = _nrmse(beta, train_encoder=False)
    all_softmax = _nrmse(_constant(beta, 1.0), train_encoder=False)
    all_linear = _nrmse(_constant(beta, 0.0), train_encoder=False)
    assert assigned < all_softmax and assigned < all_linear, (
        "the per-coordinate assignment does not beat a single constant corner: "
        "assigned %.4f, all-softmax(beta=1) %r, all-linear(beta=0) %r"
        % (assigned, all_softmax, all_linear))


# ---------------------------------------------------------------------------
# 3. the mechanism: does the JEPA encoder carry the exponents the mask claims?
# ---------------------------------------------------------------------------

def jepa_encoder_alpha(seed=pj.SEED, n_grid=pa.N_GRID):
    """alpha_d of the pi_jepa Encoder's OWN output, by pi_assign's own fitter.

    Same instrument as R1: RMS of coordinate d over the ensemble at each context
    length n, OLS of log2 norm on log2 n. The only change is the object measured
    -- this Encoder on this bed, rather than pi_assign's synthetic statistics.
    """
    torch.manual_seed(int(seed))
    enc = pj.Encoder()
    for p in enc.parameters():
        p.requires_grad_(False)
    norms = np.zeros((len(n_grid), pj.D_LATENT))
    for i, n in enumerate(n_grid):
        x, _ = pj.draw_bed(seed, batch=pj.BATCH, s=int(n), h=pj.HORIZON)
        s = enc(x[:, :int(n)])[:, -1, :]
        norms[i] = np.sqrt(np.mean(np.square(s.numpy()), axis=0))
    return [pa.fit_power_law(n_grid, norms[:, d]).alpha
            for d in range(pj.D_LATENT)]


def test_jepa_encoder_coordinates_carry_the_exponent_the_mask_assigns():
    beta = pj.beta_vector()
    alphas = jepa_encoder_alpha()
    bad = []
    for d, b in enumerate(beta):
        if pj.is_refusal(b):
            continue
        want = 1.0 - float(b)
        if abs(alphas[d] - want) > pa.ALPHA_TOL:
            bad.append((d, alphas[d], want))
    assert not bad, (
        "the mask assigns beta_d = 1 - alpha_d, but the pi_jepa Encoder's own "
        "coordinates do not carry those alphas: %d of %d assigned coordinates "
        "are off by more than ALPHA_TOL=%.2f -- %s; measured alphas %s"
        % (len(bad), len([b for b in beta if not pj.is_refusal(b)]),
           pa.ALPHA_TOL,
           ", ".join("d=%d alpha=%+.4f wants %+.1f" % t for t in bad),
           ["%+.4f" % a for a in alphas]))


def test_the_encoder_reads_its_context_and_not_one_position():
    """WHY every alpha reads 0. Encoder's docstring claims x_{<=t} -> s_t.

    An exponent of the context length only exists if the representation is a
    function of the context. If s_t depends on x_t alone then ||s(n)|| cannot
    move with n for any encoder, at any initialisation, trained or not, and
    alpha = 0 is an identity rather than a measurement -- so INTENSIVE is the
    only class R1 could ever return here and the corner it hands out carries no
    information about the coordinate.
    """
    torch.manual_seed(pj.SEED)
    enc = pj.Encoder()
    x, _ = pj.draw_bed(pj.SEED)
    with torch.no_grad():
        before = enc(x)[:, -1, :]
        y = x.clone()
        y[:, :-1] = x[:, torch.randperm(x.shape[1] - 1)]
        after = enc(y)[:, -1, :]
    moved = float((after - before).abs().max())
    assert moved > 0.0, (
        "permuting every context position leaves the last-position "
        "representation bitwise identical (worst change %.17g): pj.Encoder is "
        "x_t -> s_t, not x_{<=t} -> s_t, so it aggregates no context and its "
        "scaling exponent in n is 0 by construction" % moved)


def test_jepa_encoder_coordinates_are_classifiable_at_all():
    """Weaker: forget the mask, does R1's rule even admit this encoder?

    If every coordinate of the encoder the mask is APPLIED to would have been
    refused by the instrument that produced the mask, the transfer is not merely
    misaligned, there was nothing to align to.
    """
    alphas = jepa_encoder_alpha()
    classifiable = [d for d, a in enumerate(alphas)
                    if abs(a) <= pa.ALPHA_TOL or abs(a - 1.0) <= pa.ALPHA_TOL]
    assert len(classifiable) == pj.D_LATENT, (
        "only %d of %d pi_jepa Encoder coordinates sit at a corner under R1's "
        "own ALPHA_TOL=%.2f; the rest are MIXED-SCALING and R1 would refuse "
        "them: alphas %s"
        % (len(classifiable), pj.D_LATENT, pa.ALPHA_TOL,
           ["%+.4f" % a for a in alphas]))


# ---------------------------------------------------------------------------
# 4. how much of 0.6881 needs the encoder, and how much needs the read
# ---------------------------------------------------------------------------

def test_frozen_arm_needs_its_encoder_network():
    beta = pj.beta_vector()
    full = _nrmse(beta, train_encoder=False)
    torch.manual_seed(pj.SEED)
    bare = _nrmse(beta, train_encoder=False, encoder=_BareProjection())
    trunc = _nrmse(beta, train_encoder=False, encoder=_Truncate())
    assert full < bare and full < trunc, (
        "the three-layer frozen encoder buys nothing over no network at all: "
        "network %.4f, bare fixed projection %r, raw-bed truncation %r"
        % (full, bare, trunc))


def test_frozen_arm_needs_the_family_read_and_not_a_linear_map():
    beta = pj.beta_vector()
    full = _nrmse(beta, train_encoder=False)
    torch.manual_seed(pj.SEED)
    model = _RealForward(beta)
    model.pred = _LinearRead()
    for p in model.online.parameters():
        p.requires_grad_(False)
    pj.fit(model, steps=pj.N_STEPS, nu=pj.NU, seed=pj.SEED, detector=True)
    linear = pj._held_out_scores(model, pj.SEED + 1)["nrmse"]
    assert full < linear, (
        "the beta-indexed family read buys nothing over one linear matrix on the "
        "same frozen features: read %.4f, plain linear read %.4f -- the frozen "
        "arm is a random-feature model with a linear head and beta is not in "
        "the circuit" % (full, linear))
