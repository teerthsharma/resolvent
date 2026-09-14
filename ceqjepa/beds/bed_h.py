"""BED-H -- the belief bed. Contract: docs/BED_H_PREREGISTRATION.md at 58c197c.

WHY THIS BED EXISTS, in one sentence: every bed this project owned was Markov in
the current state, and on a fully observed Markov chain the future depends only
on the present, so an attention operator over a context has provably nothing to
do. Measured on the old bed on 2026-09-14, aggregation COSTS 0.0810 at the
encoder output and moving the label to the horizon does not rescue it. BED-H
makes history necessary while keeping the label exact.

HOW HISTORY IS MADE NECESSARY. The chain x_t over S = 8 states is hidden; the
observation y_t sees it through an emission matrix with snr in [0.5, 0.7]. The
pair (x_t, y_t) is Markov but the OBSERVED process y_t is not, and the label is a
function of the filtered belief b_t = Pr(x_t | y_0..t), which is a function of the
whole prefix. lambda_2 in [0.90, 0.97] sets the relaxation time 1/(1 - lambda_2)
to roughly 10-33 steps, so evidence from that far back is still worth carrying.
Gate 1 below does not trust any of that: it MEASURES whether the window-L error
decays and refuses the instance if it does not.

WHAT IS EXACT. The oracle is the forward algorithm, O(T S^2), in float64, and it
is checked against exhaustive enumeration of every hidden path on a 3-state
5-step chain in tests/beds/test_bed_h.py::test_oracle_matches_brute_force. DCM-1
died of an unchecked oracle; that test is the reason this one is not repeating it.

THE TWO LABEL CLASSES, and why they are not an index.

  b_t                      INTENSIVE -- a distribution, flat in the token count.
  C_t = sum_{tau<=t} b_tau.h   EXTENSIVE -- an expected count, linear in it.

The corner rule `beta = 1 - alpha` was retired (docs/CORNER_RULE_RETIREMENT.md)
on three grounds, the first being that pi_jepa's encoder is position-wise, so its
exponent is zero by construction and the mask reached its coordinates by index
equality alone. Here the two classes are planted by the generator's physics: the
belief is a normalised object and the cumulant is a sum, and no choice of
coordinate index can make one look like the other. `label_exponents()` measures
the exponent of each so a later round can check it recovers 0 and 1. This module
measures; it assigns no corner.

THE TRAP. No consumer is handed P. `inputs_for_consumer()` returns exactly
CONSUMER_KEYS and nothing else, and `scan_for_matrix()` fails if P -- or P
transposed, affinely rescaled, or with its entries reordered -- is present in the
float content of that payload. The trap is a claim about INPUTS: the labels are
by construction a function of P, which is what "exact oracle" means, and a bed
whose targets were independent of P would be a different bed. The window-L
forward algorithm and the exact oracle DO take P and are marked ORACLE for that
reason; the EMA and the GRU do not, and
tests/beds/test_bed_h.py::test_trap_consumers_have_no_door_for_P asserts their
signatures cannot accept it.

THREE CORRECTIONS APPLIED 2026-09-14, AFTER A PRIOR-ART AUDIT AND AGAINST THE
PRE-REGISTRATION WHERE THEY CONFLICT. They are recorded here because the
pre-registration is a contract this module may not edit.

  (i) THE S-UNIT GRU IS REPORTED AS A FLOOR, NOT AS THE SKYLINE. The
      pre-registration says "a GRU with S hidden units". Two papers bear on that,
      and BOTH were checked against their own text on 2026-09-14 rather than
      taken as summarised, because each turns out narrower than its slogan:

        Svete and Cotterell, "Recurrent Neural Language Models as Probabilistic
        Finite-state Automata", arXiv:2310.05161, EMNLP 2023 (8069-8086), give
        Omega(N |Sigma|) neurons for an N-state deterministic PFSA. THAT BOUND IS
        STATED FOR HEAVISIDE ELMAN RNNs under weak equivalence, with a full-rank
        condition and |Sigma| >= |Q|; its own Limitations section declines to
        extend it to LSTMs. It does NOT prove S = 8 is too small for a GRU, and
        this module does not claim it does. It is used as a SIZING HEURISTIC only:
        S*S = 64.

        Weiss, Goldberg and Yahav, "On the Practical Computational Power of
        Finite Precision RNNs for Language Recognition", arXiv:1805.04908, ACL
        2018, find the finite-precision GRU cannot count where the LSTM and the
        ReLU-Elman can. That result is ARGUED per-dimension (the tanh and the
        coupled gates bound h to [-1, 1]) and SHOWN EMPIRICALLY; the only formal
        impossibility proof in the paper is for the SRNN, and the multi-dimensional
        GRU case is dismissed only intuitively. So "the GRU provably cannot count"
        is NOT available, and is not asserted here.

      Since neither paper settles it, the MEASUREMENT decides: the S-unit GRU is
      reported as FLOOR_HIDDEN, and a SKYLINE_HIDDEN = 64 recurrence is run as
      both a GRU and an LSTM. The extensive head exists so the counting question
      is measured rather than argued. An arm that beats a skyline too small to
      hold the answer has measured nothing.

 (ii) MUST-FIRE 1 IS A THEOREM, NOT A GATE. Le Gland and Mevel, "Exponential
      Forgetting and Geometric Ergodicity in Hidden Markov Models", Mathematics of
      Control, Signals and Systems 13(1):63-93, 2000 (DOI 10.1007/PL00009861),
      prove the PREDICTION filter forgets its initial condition almost surely
      exponentially fast. The window-L error therefore decays BY CONSTRUCTION on
      every instance this generator can emit, so gate 1 discriminates nothing. It
      is kept, it still refuses, and `gate_report()["gate1_kind"]` reads
      "sanity-check": a failure means the GENERATOR is broken, which is a more
      useful reading than "the bed is Markov". It is NOT evidence that the bed is
      non-Markov, because a check that cannot fail supplies no such evidence.

(iii) THE WINDOW-4 BAR IS DERIVED BEFORE IT IS MEASURED, AND FROM THE RIGHT
      QUANTITY. Ye, Ma and Qian, "Estimate exponential memory decay in hidden
      Markov model and its applications to inference", arXiv:1710.06078, Physica D
      460:134053 (2024), put the subsequence length for a controlled error eps at
      B ~ log(eps)/(lambda_2 - lambda_1) -- where lambda_1 and lambda_2 are the top
      two LYAPUNOV EXPONENTS of the random product M D_1 M D_2 ..., D_n being the
      diagonal of emission likelihoods. They are NOT eigenvalues of M, and the
      paper warns in as many words that the per-matrix eigenvalues "have little to
      do with this asymptotic behavior". `lyapunov_gap()` estimates that gap by QR
      on two columns, and `tests/beds/test_bed_h.py` pins the estimator against
      the one case with a closed form: at snr = 1/S every D_n is (1/S)I, the
      product collapses to (1/S)^n (M^T)^n, and exp(gap) is exactly lambda_2(M).
      `gate_report()` returns the window-4 prediction from the Lyapunov gap, the
      prediction from lambda_2 beside it as a NAMED WRONG PREDICTOR, and the
      measurement. Whatever is measured is not adopted as the bar.

 (iv) GATE 3, added. Mossel and Roch, "Learning nonsingular phylogenies and hidden
      Markov models", arXiv:cs/0502076, Annals of Applied Probability
      16(2):583-614, 2006: learning HMMs without the nonsingularity condition on
      the TRANSITION matrices is at least as hard as learning parity with noise --
      a CONDITIONAL hardness, resting on the noisy-parity conjecture, not an
      unconditional lower bound. A near-singular P therefore makes the instance
      unlearnable for arm and baseline alike, so sigma_min(P) >= SIGMA_MIN_FLOOR
      and the instance is refused otherwise.

  WEAK form:   P fixed at WEAK_P_SEED across every instance, never handed over.
  STRONG form: P redrawn per instance and reachable only through a prefix of
               PREFIX_LEN observations from the same chain -- in-context system
               identification, the real target.

PHASE 2: THE ABSTENTION IS THE CLAIM, AND THE BED IS NOT. THE BED IS OWNED.

    arXiv:2512.22471 built this substrate first -- "Bayesian wind tunnels",
    controlled environments where the true posterior is known in closed form,
    drawing a FRESH Dirichlet HMM per sequence, supervising on the filtered
    posterior with forward-recursion ground truth, and scoring architectures at
    an entropy MAE of 0.049 Transformer, 0.024 Mamba, 0.411 LSTM (marked a
    failure), 0.40 MLP. That is this module's STRONG form, already executed.
    arXiv:2605.20824 is the second owner: six HMM families x three seeds, exact
    latent states, transition matrices, belief vectors, Bayes-optimal
    predictions and forced-state counterfactual targets, against four controls.

THIS MODULE CLAIMS NEITHER. The bed, the exact oracle labels (Rabiner 1989, DOI
10.1109/5.18626), belief-as-target, in-context HMM identification, and belief
entropy as a per-position uncertainty statistic (Durand and Guedon,
arXiv:1202.6545) are all struck from the delta by
docs/BED_H_PREREGISTRATION.md at e7f7704. The bed is built here because it is
needed as an instrument, not because it is new.

WHAT IS SCORED IS THE ABSTENTION, against a posterior that is exactly known --
see `abstention_table`, `selective_risk`, `plant_refusal_rate`.

THE REFUSAL LEG, AND WHY IT IS THE WHOLE DELTA. docs/BED_H_PREREGISTRATION.md was
rewritten at e7f7704 after an audit that resolved 37 identifiers, and it struck
three quarters of what this bed would have claimed. The substrate is published
twice (arXiv:2512.22471 draws a fresh Dirichlet HMM per sequence and supervises on
the forward-recursion posterior -- this bed's strong form, already executed;
arXiv:2605.20824 ships six HMM families with exact beliefs and counterfactual
targets). Exact oracle labels are Rabiner 1989, DOI 10.1109/5.18626. Belief
entropy as the per-position uncertainty statistic is Durand and Guedon,
arXiv:1202.6545. `bottom` as an output symbol is arXiv:1907.00208. None of those
is claimed here. What survives is one sentence:

    a protocol that scores a sequence architecture's BOTTOM emission against an
    exactly-known filter posterior, with the abstain threshold ANCHORED at
    ln S - delta rather than TUNED FOR COVERAGE.

So the load-bearing property is the ANCHOR, not the refusal. `refuse_delta_for()`
is a closed form in S and snr, both fixed before a trajectory exists, and it
cannot see the data -- asserted by signature in tests/beds/test_bed_h.py. And NO
GATE IS ADDED FOR THE REFUSAL BASE RATE, deliberately: a gate that refused
instances whose abstain-class came out inconveniently sized would BE coverage
tuning wearing a gate's clothes. The base rate is reported instead, whatever it is.

PRIOR ART, owned not discovered. Shai, Marzen, Teixeira, Gietelink Oldenziel,
Riechers, "Transformers represent belief state geometry in their residual
stream", arXiv:2405.15943 (v1 24 May 2024, v3 4 Feb 2025). Fetched and read
2026-09-14: the abstract states belief states are linearly represented in the
residual stream over an edge-emitting HMM, built on the mixed-state presentation
of computational mechanics, and the paper contains no Comments or journal-ref
field, so arXiv asserts no venue for it.

An earlier draft of this docstring conceded "that representation result is theirs
and this bed does not claim it". e7f7704 records that the concession was
INVERTED: 2405.15943 scores no architecture against anything -- no baselines, no
windowed oracle, no RNN comparison -- and recovers belief by linear probe from
residual activations after next-token training. So there was no scoring ground to
concede. The concession is withdrawn, and replaced by the narrower and true one:
this bed claims neither the representation result nor the substrate, because the
substrate is separately published (arXiv:2512.22471, arXiv:2605.20824) and the
exact oracle is Rabiner 1989. What is claimed is the scoring protocol in THE
REFUSAL LEG above, and nothing else on this page.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "S", "T_LEN", "PREFIX_LEN", "BURN_IN", "HIT_SET", "ABSORB_HORIZON",
    "L_GRID", "DECAY_SLACK", "DECAY_MARGIN", "SPREAD_FLOOR", "N_GRID",
    "LAMBDA2_BAND", "SNR_BAND", "WEAK_P_SEED", "CONSUMER_KEYS",
    "HMM", "BedHInstance", "forward_belief", "tv",
    "draw_instance", "instance_from", "gate_report", "draw_pool", "prior_mean_P",
    "inputs_for_consumer", "scan_for_matrix",
    "ema_belief", "fit_ema", "RecurrentBaseline", "fit_rnn", "rnn_scores",
    "count_obs", "count_mae", "label_exponents", "lyapunov_gap",
    "LN_S", "belief_entropy", "refuse_delta_for", "refusal_label",
    "refusal_scores", "auroc", "COVERAGES",
    "DELTA_SWEEP", "DEADBAND", "PLANT_LEN", "PLANT_SCORE_FROM",
    "PLANT_SNR_UNINF", "PLANT_SNR_INF",
    "forward_belief_lik", "draw_planted", "definedness",
    "lyapunov_exponents", "entropy_rate", "dimension_upper_bound",
    "covering_floor", "RATE_FIT_MIN_L",
    "abstention_table", "selective_risk", "plant_refusal_rate",
    "SIGMA_MIN_FLOOR", "FLOOR_HIDDEN", "SKYLINE_HIDDEN", "LAMBDA_COUNT",
]

# ---- PINNED. Identical on every instance of both forms. --------------------
S = 8
T_LEN = 256
PREFIX_LEN = 128
BURN_IN = 32
HIT_SET = (0, 1, 2)
ABSORB_HORIZON = 8
L_GRID = (1, 2, 4, 8, 16, 32)
DECAY_SLACK = 0.005
DECAY_MARGIN = 0.05
SPREAD_FLOOR = 0.1
N_GRID = (32, 64, 128, 256)
LAMBDA2_BAND = (0.90, 0.97)
SNR_BAND = (0.5, 0.7)
WEAK_P_SEED = 20260914
CONSUMER_KEYS = ("obs", "prefix_obs", "hit_vector")

# Gate 3 (Mossel-Roch). DERIVED from the construction, not fitted to the draw:
# P = (1-eps)I + eps*B with eps <= 1 - min(LAMBDA2_BAND) = 0.10 gives
# sigma_min(P) >= 1 - eps*(1 + sigma_max(B)) >= 1 - 0.10*(1 + sqrt(8)) = 0.62.
# A floor of 0.10 therefore fires only if the construction itself is broken.
SIGMA_MIN_FLOOR = 0.10

# The pre-registration's width. NOT proven too small: arXiv:2310.05161's
# Omega(N|Sigma|) is stated for Heaviside Elman RNNs, not GRUs (see (i) above).
FLOOR_HIDDEN = S
# S*S = 64, taken from that bound as a SIZING HEURISTIC only.
SKYLINE_HIDDEN = S * S
LAMBDA_COUNT = 1.0      # weight on the extensive head's loss

LN_S = float(np.log(S))          # the maximum-entropy anchor
COVERAGES = (1.0, 0.9, 0.8, 0.5)  # PINNED, the risk-coverage grid

# ---- PHASE 2, pre-registered before any abstention was measured -------------
DELTA_SWEEP = (0.05, 0.1, 0.2)   # the abstain thresholds. Not added to later.
DEADBAND = 0.02                  # nats; the UNDEFINED class, see `definedness`
# PLANT GEOMETRY, DERIVED FROM THE BED'S OWN RELAXATION TIME. The first pin was
# 24 long scored from step 8 and it FAILED must-fire 1 at 0.078/0.378/0.749: with
# a flat likelihood the belief propagates as b P and relaxes to the STATIONARY
# distribution at rate lambda_2, so 8 steps is under one relaxation time and the
# entropy ceiling is H(pi), not ln S. 1/(1-lambda_2) is 10 to 33 steps over the
# pre-registered band, so 64 steps of settle is 2 to 6 relaxation times.
PLANT_LEN = 96
PLANT_SCORE_FROM = 64
PLANT_SNR_UNINF = 1.0 / S        # pure noise: the observation says nothing
PLANT_SNR_INF = 0.95             # near-deterministic emission
N_PLANTS_EACH = 1                # 1 uninformative + 1 informative per instance
RATE_FIT_MIN_L = 4               # fit the decay slope only where the law holds

DT = np.float64


def hit_vector():
    """h, the extensive label's weight. PINNED, not drawn: a coordinate index
    cannot be the thing that separates the classes here (see module docstring)."""
    h = np.zeros(S, dtype=DT)
    h[list(HIT_SET)] = 1.0
    return h


# ---------------------------------------------------------------- the chain
class HMM:
    """The generator. ORACLE SIDE. Holds P, E, pi and is never inside a payload."""

    def __init__(self, P, snr, eps=None):
        self.P = np.asarray(P, dtype=DT)
        self.snr = float(snr)
        self.eps = eps
        n = self.P.shape[0]
        E = np.full((n, n), (1.0 - self.snr) / (n - 1), dtype=DT)
        np.fill_diagonal(E, self.snr)
        self.E = E
        ev = np.sort(np.abs(np.linalg.eigvals(self.P)))
        self.lambda2 = float(ev[-2])
        self.sigma_min = float(np.linalg.svd(self.P, compute_uv=False)[-1])
        w, v = np.linalg.eig(self.P.T)
        pi = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
        self.pi = np.abs(pi) / np.abs(pi).sum()

    @classmethod
    def from_matrix(cls, P, snr):
        """Build an HMM on a GIVEN P. The door gate 3's test walks through."""
        return cls(P, snr)

    @classmethod
    def draw(cls, rng, lambda2=None, snr=None):
        """lambda_2 is BISECTED to the target and then MEASURED back, never
        assumed from the knob -- the knob sets it only approximately."""
        target = float(rng.uniform(*LAMBDA2_BAND)) if lambda2 is None else float(lambda2)
        s = float(rng.uniform(*SNR_BAND)) if snr is None else float(snr)
        B = rng.dirichlet(np.ones(S), size=S).astype(DT)   # non-symmetric a.s.
        eye = np.eye(S, dtype=DT)

        def lam2(e):
            ev = np.abs(np.linalg.eigvals((1.0 - e) * eye + e * B))
            return float(np.sort(ev)[-2])

        lo, hi = 0.0, 1.0                                   # lam2 decreases in e
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if lam2(mid) > target:
                lo = mid
            else:
                hi = mid
        eps = 0.5 * (lo + hi)
        P = (1.0 - eps) * eye + eps * B
        P /= P.sum(1, keepdims=True)
        return cls(P, s, eps)

    def sample(self, rng, T):
        x = np.empty(T, dtype=np.int64)
        y = np.empty(T, dtype=np.int64)
        x[0] = rng.choice(S, p=self.pi)
        for t in range(1, T):
            x[t] = rng.choice(S, p=self.P[x[t - 1]])
        for t in range(T):
            y[t] = rng.choice(S, p=self.E[x[t]])
        return x, y

    def forward(self, obs):
        """ORACLE. Exact filtered belief, O(T S^2)."""
        return forward_belief(self.P, self.E, self.pi, obs)

    def window_forward(self, obs, L):
        """ORACLE WITH AMNESIA. Exact inside the last L observations, and the
        stationary distribution of P as the prior outside it."""
        T = len(obs)
        out = np.empty((T, S), dtype=DT)
        for t in range(T):
            out[t] = forward_belief(self.P, self.E, self.pi, obs[max(0, t - L + 1): t + 1])[-1]
        return out

    def absorption(self, horizon=ABSORB_HORIZON):
        """a[s] = Pr(the chain enters HIT_SET within `horizon` steps | x_t = s).
        Exact backward DP on P. The consequence label is b_t . a."""
        a = np.zeros(S, dtype=DT)
        a[list(HIT_SET)] = 1.0
        inside = a.copy().astype(bool)
        for _ in range(horizon):
            nxt = self.P @ a
            a = np.where(inside, 1.0, nxt)
        return a


def forward_belief_lik(P, pi, lik, return_norm=False):
    """The forward algorithm against a PER-STEP likelihood matrix lik[t, s] =
    Pr(y_t | x_t = s). float64, renormalised each step, O(T S^2).

    The per-step form is what keeps the label EXACT when the emission changes
    inside a sequence, which is exactly what planting an uninformative stretch
    does. A planted bed whose oracle still assumed a fixed E would have an
    approximate label, and an approximate label is what killed DCM-1."""
    lik = np.asarray(lik, dtype=DT)
    n = lik.shape[0]
    out = np.empty((n, P.shape[0]), dtype=DT)
    c = np.empty(n, dtype=DT)
    b = pi * lik[0]
    c[0] = b.sum()
    out[0] = b / c[0]
    for t in range(1, n):
        b = (out[t - 1] @ P) * lik[t]
        c[t] = b.sum()
        out[t] = b / c[t]
    return (out, c) if return_norm else out


def forward_belief(P, E, pi, obs):
    """The forward algorithm with a fixed emission matrix. O(T S^2)."""
    return forward_belief_lik(P, pi, E[:, np.asarray(obs)].T)


def entropy_rate(hmm, obs, burn=32):
    """h(Y), the observed process's entropy rate, in nats, STRAIGHT OUT OF THE
    FILTER'S OWN NORMALISERS: h = -mean(log c_t), where c_t = Pr(y_t | y_{<t}) is
    what `forward_belief_lik` already divides by. Three lines, no extra pass.

    NOT this module's result. Jurgens and Crutchfield give this as the Blackwell
    integral in the PREQUEL, arXiv:2008.12886 Eq. (16)-(17) -- NOT in
    arXiv:2102.10487 Eq. (15), which is the dimension formula and is cited on
    `dimension_upper_bound` instead. Verified against both papers 2026-09-14
    after the citation handed to this module pointed at the wrong equation.
    It is computed here because it is free and because it buys the identity
    checked in `lyapunov_exponents` below."""
    obs = np.asarray(obs)
    _, c = forward_belief_lik(hmm.P, hmm.pi, hmm.E[:, obs].T, return_norm=True)
    return float(-np.log(np.maximum(c[burn:], 1e-300)).mean())


def lyapunov_exponents(hmm, obs, burn=32):
    """(lambda_1, lambda_2), the top two Lyapunov exponents of the random product
    A_n = D(y_n) M^T, by QR on two columns.

    THE IDENTITY THAT MAKES THIS SELF-CHECKING. The un-normalised forward vector
    has norm Pr(y_{1..n}), so lambda_1 = -h(Y). MEASURED here to a residual below
    1e-3 and asserted in tests/beds/test_bed_h.py; it is a free correctness check
    on BOTH the filter and this QR code, since if either is wrong the identity
    breaks and nothing else in the suite would catch it. For the identity as a
    published result cite Holliday, Goldsmith and Glynn, IEEE Trans. Inform.
    Theory 52:3509-3532 (2006) -- NOT arXiv:2102.10487, which does not state it.

    ATTRIBUTION, corrected 2026-09-14 against the papers themselves. The
    Lyapunov-exponent characterisation of filter stability is Atar and Zeitouni,
    "Lyapunov Exponents for Finite State Nonlinear Filtering", SIAM J. Control
    Optim. 35(1):36-55 (1997), DOI 10.1137/S0363012994272046 -- Corollary 2.1,
    stated as the last line of the proof of Theorem 1.1, giving the forgetting
    rate as the gap between the top two Lyapunov exponents of the Zakai
    recursion. Le Gland and Mevel attribute it there themselves, in their
    Section 1 Introduction p. 66 -- NOT in a "Remark 2.4", which does not exist
    in that paper. Nor is Birkhoff theirs in that paper: the string does not
    occur in MCSS 13(1):63-93 at all; the Birkhoff contraction machinery is the
    COMPANION paper, MCSS 13(1):41-62. And Atar and Zeitouni themselves USE the
    Birkhoff coefficient, via Peres's lemma, as their lower bound on the gap --
    so the two papers do not divide along Birkhoff-versus-Lyapunov lines.
    Le Gland and Mevel are cited in `gate_report` for exponential forgetting
    under primitivity, which IS theirs. The rate's use as a window length is
    arXiv:1710.06078."""
    n = hmm.P.shape[0]
    Q, _ = np.linalg.qr(np.eye(n, 2, dtype=DT))
    acc, kept = np.zeros(2, dtype=DT), 0
    for t, y in enumerate(np.asarray(obs)):
        Q, R = np.linalg.qr((hmm.E[:, y][:, None] * hmm.P.T) @ Q)
        if t >= burn:
            acc += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
            kept += 1
    if kept == 0:
        raise ValueError("burn=%d consumed the whole sequence" % burn)
    lam = acc / kept
    return float(lam[0]), float(lam[1])


def dimension_upper_bound(hmm, obs, burn=32):
    """d <= h / chi, an UPPER BOUND on the dimension of the attainable belief set,
    never a measured dimension. chi = lambda_1 - lambda_2 > 0 is the contraction
    rate of the filter map.

    Jurgens and Crutchfield, arXiv:2102.10487 Eq. (15), give d_mu = -h_mu/lambda_mu
    for the statistical complexity dimension, with equality under the open set
    condition and, in their words, "failing OSC, this ratio is an upper bound on
    the dimension of the measure". No OSC is checked here, so the UPPER BOUND
    reading is the only one taken.

    THE CONFOUND, reported rather than hidden: d and chi are NOT independent --
    the bound IS d <= h/chi -- so no sweep varies one while holding the other, and
    moving snr moves both, in opposite directions. Any claim resting on this
    number has to carry that."""
    lam1, lam2 = lyapunov_exponents(hmm, obs, burn)
    h = entropy_rate(hmm, obs, burn)
    chi = lam1 - lam2
    return {"h": h, "lambda_1": lam1, "lambda_2_lyap": lam2, "chi": float(chi),
            "d_ub": float(h / chi) if chi > 0 else float("inf"),
            "ambient": S - 1}


def covering_floor(d_ub, n_states):
    """The TV resolution a reader with `n_states` distinguishable internal states
    could reach IF it placed them optimally on the attainable belief set:
    eps* = n_states^(-1/d). A FLOOR, not an achievable score -- it assumes optimal
    placement, which is exactly what a windowed reader does not do."""
    return float(n_states ** (-1.0 / d_ub)) if d_ub > 0 else float("nan")


def lyapunov_gap(hmm, obs, burn=32):
    """The bed's REAL memory-decay rate: lambda_2 - lambda_1, the gap of the top
    two Lyapunov exponents of the random product A_n = D(y_n) M^T, estimated by
    QR on two columns (arXiv:1710.06078, Physica D 460:134053, 2024).

    The filter recursion p_{n+1} = p_n M D_{n+1} is, on column vectors,
    p^T_{n+1} = D_{n+1} M^T p^T_n, so A_n = D(y_n) M^T is the matrix whose product
    carries the exponents. The gap is negative; exp(gap) is the per-step factor by
    which two different initial conditions converge, which is what a window-L
    reader pays for its amnesia.

    It is NOT lambda_2 of M. Ye et al. warn that per-matrix eigenvalues "have
    little to do with this asymptotic behavior", and this bed measures exactly
    that: with informative emissions the filter forgets far faster than lambda_2.
    """
    lam1, lam2 = lyapunov_exponents(hmm, obs, burn)
    return lam2 - lam1


def tv(a, b):
    """Mean total-variation distance. THE metric of this bed, pinned."""
    return float((0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(-1)).mean())


# ------------------------------------------------------------- the refusal
def belief_entropy(b):
    """H(b) per position, nats. NOT claimed as novel: belief entropy as the
    per-position uncertainty statistic is Durand and Guedon, arXiv:1202.6545."""
    b = np.asarray(b, dtype=DT)
    return -(b * np.log(np.maximum(b, 1e-300))).sum(-1)


def refuse_delta_for(snr, s=S):
    """THE ANCHOR, and the only thing this bed still claims is its own.

    delta = ln S - H(one look): the entropy a posterior would have after exactly
    ONE observation from a uniform prior. So the abstain rule

        H(b_t) >= ln S - delta   <=>   H(b_t) >= H(one look)

    reads "the whole history has left you no sharper than a single fresh glance".
    Both arguments are fixed at generation time and neither is a trajectory, so
    this CANNOT be tuned for coverage -- which is the property the surviving claim
    rests on, and which tests/beds/test_bed_h.py asserts by signature."""
    q = np.full(s, (1.0 - snr) / (s - 1), dtype=DT)
    q[0] = snr
    return float(np.log(s) + (q * np.log(np.maximum(q, 1e-300))).sum())


def refusal_label(belief, delta):
    """The EXACT abstain target. Exact because the filter posterior is exact --
    which is the whole reason this protocol needs this bed and not a real corpus."""
    return np.asarray(belief_entropy(belief) >= np.log(belief.shape[-1]) - delta)


def auroc(score, label):
    """Rank AUROC with average ranks for ties. No sklearn, no scipy."""
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=DT)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    srt = score[order]
    ranks = np.empty(score.size, dtype=DT)
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def refusal_scores(pred_belief, inst, delta, coverages=COVERAGES):
    """Score a consumer's BOTTOM emission against the exactly-known posterior.

    The consumer abstains by its OWN predicted entropy -- it is never shown the
    exact belief, the label, or delta. Two numbers come out:

      auroc     -- can the consumer TELL when the oracle would abstain? Ranking
                   its own entropy against the exact refusal label.
      risk_at_c -- does abstaining BUY anything? Mean TV to the exact belief over
                   the c most-confident fraction of positions. `aurc` integrates
                   that curve so one number orders arms.

    Both are computed over t >= BURN_IN, exactly as every other metric here."""
    pred = np.asarray(pred_belief, dtype=DT)[BURN_IN:]
    lab = np.asarray(inst.refuse)[BURN_IN:]
    risk = 0.5 * np.abs(pred - inst.belief[BURN_IN:]).sum(-1)
    ent = belief_entropy(pred)

    order = np.argsort(ent, kind="mergesort")       # most confident first
    run = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    out = {"base_rate": float(lab.mean()), "auroc": auroc(ent, lab),
           "aurc": float(run.mean())}
    for c in coverages:
        out["risk_at_%g" % c] = float(run[max(0, int(round(c * risk.size)) - 1)])
    return out


# ------------------------------------------------------------- the instance
@dataclass
class BedHInstance:
    obs: np.ndarray                     # INPUT
    prefix_obs: np.ndarray | None       # INPUT, STRONG form only
    hit_vector: np.ndarray              # INPUT (task definition, not the chain)
    belief: np.ndarray                  # TARGET, intensive
    cum_hits: np.ndarray                # TARGET, extensive
    consequence: np.ndarray             # TARGET, intensive (absorption under b)
    refuse: np.ndarray                  # TARGET, the exact BOTTOM label
    meta: dict = field(default_factory=dict)   # REPORT only, never an input


def instance_from(hmm, rng, form):
    if form not in ("weak", "strong"):
        raise ValueError("form must be 'weak' or 'strong', got %r" % (form,))
    prefix = hmm.sample(rng, PREFIX_LEN)[1] if form == "strong" else None
    _, obs = hmm.sample(rng, T_LEN)
    b = hmm.forward(obs)
    h = hit_vector()
    return BedHInstance(
        obs=obs, prefix_obs=prefix, hit_vector=h,
        belief=b, cum_hits=np.cumsum(b @ h), consequence=b @ hmm.absorption(),
        refuse=refusal_label(b, refuse_delta_for(hmm.snr)),
        meta={"form": form, "lambda2": hmm.lambda2, "snr": hmm.snr,
              "refuse_delta": refuse_delta_for(hmm.snr),
              "refuse_base_rate": float(refusal_label(b, refuse_delta_for(hmm.snr))[BURN_IN:].mean()),
              "relax_steps": 1.0 / (1.0 - hmm.lambda2),
              "asym": float(np.abs(hmm.P - hmm.P.T).max())},
    )


def draw_instance(seed, form):
    """WEAK: P is fixed at WEAK_P_SEED, so only the trajectory varies with
    `seed`. STRONG: P is redrawn from `seed` too and reaches the consumer only
    through the prefix."""
    p_seed = WEAK_P_SEED if form == "weak" else seed
    hmm = HMM.draw(np.random.default_rng(p_seed))
    inst = instance_from(hmm, np.random.default_rng(seed + 1), form)
    inst.meta["seed"] = seed
    return inst, hmm


def prior_mean_P(matrices):
    """The STRONG form's NO-IDENTIFICATION floor, and the bar a later round must
    hold an in-context arm to.

    Scoring a strong-form arm against a random OTHER draw from the prior is too
    easy: an arm that identifies nothing can still use the prior MEAN, which is
    strictly better than any single draw. Measured 2026-09-14 on 32 instances,
    the prior mean costs 0.0802 TV against 0.1118 for another draw, so the
    headroom in-context identification can actually win is 0.0802, not 0.1118.

    This is NOT in docs/BED_H_PREREGISTRATION.md. It is added because the strong
    form is the page's stated real target and it shipped without a floor."""
    P = np.mean([np.asarray(M, dtype=DT) for M in matrices], axis=0)
    return P / P.sum(1, keepdims=True)


def _emission(snr, s=S):
    E = np.full((s, s), (1.0 - snr) / (s - 1), dtype=DT)
    np.fill_diagonal(E, snr)
    return E


def draw_planted(seed, form, n_each=N_PLANTS_EACH):
    """An instance carrying PLANTED stretches of known definedness.

    WHY PLANTING IS NECESSARY. At the pre-registered DELTA_SWEEP the natural bed
    almost never produces an uninformative belief -- measured base rate 0.0014 at
    delta = 0.2 and lower below it -- so the uninformative row of the 3x2 table
    would be empty and precision and recall would be undefined. Planting creates
    the class instead of tuning delta until the class appears, which would be the
    coverage tuning the whole claim rests on NOT doing.

    Inside an uninformative stretch the emission is pure noise (snr = 1/S), so the
    observation carries nothing and the belief decays to the propagated prior.
    Inside an informative stretch it is near-deterministic (snr = 0.95). The
    oracle uses the true per-step emission throughout, so the label stays exact.

    Stretches are scored only from PLANT_SCORE_FROM steps in, because the belief
    needs time to decay after the emission changes; the settle-in steps are
    excluded by construction rather than discovered afterwards."""
    p_seed = WEAK_P_SEED if form == "weak" else seed
    hmm = HMM.draw(np.random.default_rng(p_seed))
    rng = np.random.default_rng(seed + 1)

    kinds = ["uninformative", "informative"] * n_each
    span = PLANT_LEN + 4
    room = T_LEN - BURN_IN - len(kinds) * span
    if room < 0:
        raise ValueError("plants do not fit in T_LEN")
    starts, cur = [], BURN_IN + 4
    for i in range(len(kinds)):
        cur += int(rng.integers(0, max(1, room // len(kinds))))
        starts.append(cur)
        cur += span
    plants = [{"start": st, "end": st + PLANT_LEN, "kind": k,
               "snr": PLANT_SNR_UNINF if k == "uninformative" else PLANT_SNR_INF}
              for st, k in zip(starts, kinds)]

    prefix = hmm.sample(rng, PREFIX_LEN)[1] if form == "strong" else None
    x = np.empty(T_LEN, dtype=np.int64)
    x[0] = rng.choice(S, p=hmm.pi)
    for t in range(1, T_LEN):
        x[t] = rng.choice(S, p=hmm.P[x[t - 1]])

    snr_t = np.full(T_LEN, hmm.snr, dtype=DT)
    for p in plants:
        snr_t[p["start"]:p["end"]] = p["snr"]
    obs = np.empty(T_LEN, dtype=np.int64)
    lik = np.empty((T_LEN, S), dtype=DT)
    for t in range(T_LEN):
        E_t = _emission(float(snr_t[t]))
        obs[t] = rng.choice(S, p=E_t[x[t]])
        lik[t] = E_t[:, obs[t]]

    b = forward_belief_lik(hmm.P, hmm.pi, lik)
    h = hit_vector()
    inst = BedHInstance(
        obs=obs, prefix_obs=prefix, hit_vector=h,
        belief=b, cum_hits=np.cumsum(b @ h), consequence=b @ hmm.absorption(),
        refuse=refusal_label(b, refuse_delta_for(hmm.snr)),
        meta={"form": form, "seed": seed, "lambda2": hmm.lambda2, "snr": hmm.snr,
              "planted": True, "plants": plants,
              "relax_steps": 1.0 / (1.0 - hmm.lambda2),
              "refuse_delta": refuse_delta_for(hmm.snr),
              "refuse_base_rate": float(
                  refusal_label(b, refuse_delta_for(hmm.snr))[BURN_IN:].mean()),
              "asym": float(np.abs(hmm.P - hmm.P.T).max())})
    return inst, hmm


# ------------------------------------------------- PHASE 2: scoring the abstention
def definedness(belief, delta, band=DEADBAND):
    """The ORACLE decides what was knowable. Returns an array of
    'informative' / 'uninformative' / 'undefined' per position.

    UNDEFINED is the deadband |H(b_t) - (ln S - delta)| < band, where the
    oracle's own call sits inside its own resolution. Those rows are excluded
    from precision and recall rather than scored either way. The spec named the
    three classes and defined two; this is the third, pinned before measuring."""
    h = belief_entropy(belief)
    thr = np.log(np.asarray(belief).shape[-1]) - delta
    out = np.where(h >= thr, "uninformative", "informative").astype(object)
    out[np.abs(h - thr) < band] = "undefined"
    return out


def abstention_table(pred_belief, inst, delta, band=DEADBAND,
                     always_answer=False):
    """The 3x2 table: {informative, uninformative, undefined} x
    {answered, refused}, plus refusal precision and recall.

    The ROW is the oracle's label of definedness. The COLUMN is the consumer's
    own decision, taken by the same rule on its OWN belief -- it never sees the
    oracle's entropy, the label, or delta. `always_answer=True` disables the
    refusal channel, which is the control for must-fire 3: a model that never
    refuses scores 0 in every refused cell BY CONSTRUCTION."""
    lab = definedness(inst.belief, delta, band)[BURN_IN:]
    thr = LN_S - delta
    ref = (np.zeros(T_LEN - BURN_IN, dtype=bool) if always_answer
           else belief_entropy(np.asarray(pred_belief, dtype=DT))[BURN_IN:] >= thr)

    t = {k: {"answered": int(((lab == k) & ~ref).sum()),
             "refused": int(((lab == k) & ref).sum())}
         for k in ("informative", "uninformative", "undefined")}
    tp = t["uninformative"]["refused"]
    fp = t["informative"]["refused"]
    fn = t["uninformative"]["answered"]
    t["precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    t["recall"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    t["n_scored"] = int(lab.size)
    t["delta"] = delta
    t["anchor"] = float(LN_S - delta)
    # RESOLVABLE. A belief that has forgotten everything equals pi, so the bed's
    # entropy ceiling is H(pi). If the anchor ln S - delta sits within one
    # DEADBAND of that ceiling, every genuinely uninformative position lands in
    # the deadband and the uninformative row is empty -- the delta cannot be
    # resolved on this bed, whatever the model does. Reported, never hidden:
    # a nan precision with no explanation is how a sweep quietly loses a point.
    t["resolvable"] = bool(t["uninformative"]["answered"]
                           + t["uninformative"]["refused"] > 0)
    return t


def selective_risk(pred_belief, inst, coverages=COVERAGES, n_boot=200, seed=0):
    """The selective-risk curve against the ALWAYS-ANSWER baseline.

    Abstention earns only if the curve beats the flat always-answer line at
    MATCHED coverage. Abstaining at random leaves the expected risk unchanged, so
    the always-answer line is risk(1.0) held flat, and dominance means the
    confidence interval on risk(c) lies below it."""
    pred = np.asarray(pred_belief, dtype=DT)[BURN_IN:]
    risk = 0.5 * np.abs(pred - inst.belief[BURN_IN:]).sum(-1)
    ent = belief_entropy(pred)

    # TIES MUST BREAK AT RANDOM, NOT BY INDEX. np.argsort(kind="mergesort") is
    # STABLE, so a predictor with constant entropy -- the uniform control, which
    # is the negative control of this whole table -- had its "most confident c"
    # silently become "the first c of the timeline". On a planted instance that
    # correlates with where the stretches sit, and the control came back
    # dominating at 1.00, which is a bug in the scorer and not a result.
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(ent.size), ent))
    run = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)

    out = {"always_answer": float(risk.mean()), "n": int(risk.size)}
    for c in coverages:
        k = max(1, int(round(c * risk.size)))
        out["risk_at_%g" % c] = float(run[k - 1])
        boot = np.array([risk[order[:k]][rng.integers(0, k, k)].mean()
                         for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["ci_at_%g" % c] = (float(lo), float(hi))
        out["dominates_at_%g" % c] = bool(hi < risk.mean())
    out["aurc"] = float(run.mean())
    return out


def plant_refusal_rate(n, form, delta, seed, n_each=N_PLANTS_EACH):
    """MUST-FIRES 1 and 2. The fraction of PLANTED positions refused, split by the
    kind of stretch, using the ORACLE's own belief as the refusing consumer.

    REACHABILITY IS REPORTED, NOT QUIETLY EXCLUDED. A belief that has forgotten
    everything equals the stationary distribution pi, so its entropy cannot exceed
    H(pi). An instance with H(pi) < ln S - delta therefore CANNOT produce an
    uninformative belief at that delta however long the stretch runs -- the
    maximum-entropy anchor is above the bed's own ceiling. `reachable_frac` is the
    fraction of instances where the anchor is reachable at all, and
    `uninformative_reachable` is the refusal rate restricted to those. Both are
    returned so a later round can see which is which."""
    hits = {"uninformative": [0, 0], "informative": [0, 0]}
    reach = {"uninformative": [0, 0]}
    n_reach = 0
    hpis = []
    for k in range(n):
        inst, hmm = draw_planted(seed + 1000 * k, form, n_each)
        hpi = float(-(hmm.pi * np.log(np.maximum(hmm.pi, 1e-300))).sum())
        hpis.append(hpi)
        ok = hpi >= LN_S - delta
        n_reach += bool(ok)
        ref = belief_entropy(inst.belief) >= LN_S - delta
        for p in inst.meta["plants"]:
            seg = ref[p["start"] + PLANT_SCORE_FROM: p["end"]]
            hits[p["kind"]][0] += int(seg.sum())
            hits[p["kind"]][1] += int(seg.size)
            if ok and p["kind"] == "uninformative":
                reach["uninformative"][0] += int(seg.sum())
                reach["uninformative"][1] += int(seg.size)
    out = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in hits.items()}
    out["reachable_frac"] = n_reach / n
    out["uninformative_reachable"] = (reach["uninformative"][0]
                                      / reach["uninformative"][1]
                                      if reach["uninformative"][1] else float("nan"))
    out["H_pi_mean"] = float(np.mean(hpis))
    out["anchor"] = float(LN_S - delta)
    out["delta"] = delta
    return out


# ------------------------------------------------------------------ the gates
def gate_report(inst, hmm):
    """Three gates, with their numbers, plus the DERIVED decay this bed predicts
    for itself. An instance outside a band is REFUSED and its numbers are
    printed; it is never silently dropped.

    GATE 1 IS A SANITY CHECK, NOT DISCRIMINATION. Exponential forgetting of the
    initial condition is a theorem for HMM filters (Le Gland and Mevel, MCSS
    13(1):63-93, 2000, under primitivity), so the window-L error decays on every instance this
    generator can emit. `gate1_kind` says so. A gate-1 failure means the
    GENERATOR is broken -- which is what the two constructed beds in
    tests/beds/test_bed_h.py exercise -- and is not evidence that the bed is
    non-Markov, because no such evidence is available from a check that cannot
    fail.

    GATE 3 is the conditioning gate (Mossel and Roch, arXiv:cs/0502076): a
    near-singular P makes the instance unlearnable for every arm, so it is worth
    nothing to anybody and is refused before it is handed out."""
    exact = inst.belief[BURN_IN:]
    per_t = {L: 0.5 * np.abs(hmm.window_forward(inst.obs, L)[BURN_IN:] - exact).sum(-1)
             for L in L_GRID}
    err = {L: float(per_t[L].mean()) for L in L_GRID}            # ARITHMETIC (annealed)
    geo = {L: float(np.exp(np.log(np.maximum(per_t[L], 1e-300)).mean()))
           for L in L_GRID}                                       # GEOMETRIC (quenched)
    seq = [err[L] for L in L_GRID]
    monotone = all(seq[i + 1] <= seq[i] + DECAY_SLACK for i in range(len(seq) - 1))
    decays = (seq[0] - seq[-1]) >= DECAY_MARGIN
    gate1 = bool(monotone and decays)

    h = inst.hit_vector
    si = float(np.std(inst.consequence[BURN_IN:]))
    se = float(np.std((inst.belief @ h)[BURN_IN:]))
    gate2 = bool(min(si, se) >= SPREAD_FLOOR)

    gate3 = bool(hmm.sigma_min >= SIGMA_MIN_FLOOR)

    # A SLOPE, AND NO LEVEL. arXiv:1710.06078 puts the subsequence length for
    # error eps at log(eps)/(lambda_2 - lambda_1) -- the gap of the top two
    # LYAPUNOV EXPONENTS of the random product D(y) M^T, not eigenvalues of M.
    # But its eq. (10) carries an unestimated constant C, introduced as "some
    # constant" and conceded to be "either too loose or still difficult to find".
    # THE THEORY SUPPLIES A SLOPE AND NO LEVEL, so no derived w4 level is
    # reported: an earlier version multiplied the QUENCHED rate onto the ANNEALED
    # level err[1] and came out 2.5 to 17 times too low on every draw, which is
    # that missing constant being set silently. The slope is kept, fitted on the
    # GEOMETRIC error where the geometric law actually holds, from L >= 4.
    def _rate(d):
        u = [(L, d[L]) for L in L_GRID if L >= RATE_FIT_MIN_L and d[L] > 1e-9]
        if len(u) < 2:
            return float("nan"), tuple(L for L, _ in u)
        xs = np.array([x[0] for x in u], dtype=DT)
        ys = np.log(np.array([x[1] for x in u], dtype=DT))
        sl = float(np.linalg.lstsq(np.stack([xs, np.ones_like(xs)], 1), ys,
                                   rcond=None)[0][0])
        return float(np.exp(sl)), tuple(L for L, _ in u)

    measured_rate, fit_L = _rate(err)      # annealed slope, for the Jensen check
    quenched_rate, _ = _rate(geo)          # quenched slope, what theory predicts

    gap = lyapunov_gap(hmm, inst.obs)
    lyap_rate = float(np.exp(gap))

    refusals = (([] if gate1 else ["gate1"]) + ([] if gate2 else ["gate2"])
                + ([] if gate3 else ["gate3"]))
    return {"window_err": err,
            "gate1_pass": gate1, "gate1_monotone": monotone,
            "gate1_decay": float(seq[0] - seq[-1]), "gate1_decays": bool(decays),
            "gate1_kind": "sanity-check",
            "spread_intensive": si, "spread_extensive": se, "gate2_pass": gate2,
            "sigma_min": hmm.sigma_min, "gate3_pass": gate3,
            "geo_window_err": geo,
            "decay_per_step_measured": measured_rate,      # arithmetic / annealed
            "decay_per_step_quenched": quenched_rate,      # geometric / quenched
            "decay_per_step_predicted": hmm.lambda2,       # NAMED WRONG PREDICTOR
            "decay_per_step_lyapunov": lyap_rate,          # what theory predicts
            "lyapunov_gap": gap, "rate_fit_L": fit_L,
            "w4_measured": float(err[4]),
            "w4_measured_geo": float(geo[4]),
            "refusals": refusals, "accepted": not refusals, "meta": inst.meta}


def draw_pool(n, form, seed0):
    """Draw n, gate all n, report the refusal fractions. No resampling to fill:
    a generator that refuses most of its draws is badly parameterised and the
    fraction is the evidence for saying so."""
    accepted, refused, rows = [], [], []
    for k in range(n):
        inst, hmm = draw_instance(seed0 + 1000 * k, form)
        g = gate_report(inst, hmm)
        rows.append(g)
        (accepted if g["accepted"] else refused).append((inst, hmm, g))
    frac = {
        "n_drawn": n,
        "frac_refused_gate1": sum("gate1" in r["refusals"] for r in rows) / n,
        "frac_refused_gate2": sum("gate2" in r["refusals"] for r in rows) / n,
        "frac_refused_gate3": sum("gate3" in r["refusals"] for r in rows) / n,
        "frac_refused_any": sum(bool(r["refusals"]) for r in rows) / n,
    }
    return accepted, refused, frac, rows


# ------------------------------------------------------------------- the trap
def inputs_for_consumer(inst):
    """EVERYTHING a consumer is handed. Nothing else is reachable from here."""
    return {k: getattr(inst, k) for k in CONSUMER_KEYS}


def scan_for_matrix(payload, M, tol=1e-9):
    """Return the leaks: every window of the payload's float content that is M
    up to reordering and an affine rescale. Catches P handed over transposed,
    scaled, shuffled, or spliced into an input array.

    Sorting both sides before the affine fit is what makes it order-blind: any
    permutation of M's entries has the same sorted vector, and any monotone
    affine map acts on that vector coordinatewise."""
    tgt = np.sort(np.asarray(M, dtype=DT).ravel())
    k = tgt.size
    if float(tgt.std()) == 0.0:
        raise ValueError("cannot scan for a constant matrix")

    flat = []
    for key in sorted(payload):
        v = payload[key]
        if v is None:
            continue
        flat.append(np.asarray(v, dtype=DT).ravel())
    if not flat:
        return []
    v = np.concatenate(flat)
    if v.size < k:
        return []

    leaks = []
    for start in range(v.size - k + 1):
        w = np.sort(v[start:start + k])
        if float(w.std()) == 0.0:
            continue
        for cand in (tgt, tgt[::-1]):
            A = np.stack([w, np.ones(k)], 1)
            coef, *_ = np.linalg.lstsq(A, cand, rcond=None)
            if abs(coef[0]) < 1e-12:
                continue
            resid = float(np.sqrt(np.mean((A @ coef - cand) ** 2)))
            if resid / float(tgt.std()) < tol:
                leaks.append({"start": int(start), "scale": float(coef[0]),
                              "offset": float(coef[1]), "resid": resid})
                break
    return leaks


# --------------------------------------------------------------- baseline 2
def ema_belief(obs, gamma, s):
    """CONSUMER. An EMA of one-hot observations. No P, no E, no pi."""
    out = np.empty((len(obs), s), dtype=DT)
    b = np.full(s, 1.0 / s, dtype=DT)
    for t, y in enumerate(obs):
        one = np.zeros(s, dtype=DT)
        one[y] = 1.0
        b = (1.0 - gamma) * b + gamma * one
        out[t] = b
    return out


def fit_ema(val_insts, grid=tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))):
    """gamma chosen on the VALIDATION split, reported, never on test."""
    scores = {float(g): float(np.mean([tv(ema_belief(i.obs, g, S)[BURN_IN:],
                                          i.belief[BURN_IN:]) for i in val_insts]))
              for g in grid}
    best = min(scores, key=scores.get)
    return best, scores


# ------------------------------------------- baseline 2b, extensive corner
def count_obs(obs):
    """CONSUMER FLOOR for the EXTENSIVE label: the running count of observations
    that land in HIT_SET. The extensive corner's analogue of the EMA -- it needs
    its own baseline or the counting claim is untested."""
    return np.cumsum(np.isin(np.asarray(obs), np.asarray(HIT_SET))).astype(DT)


def count_mae(pred, true):
    """Mean absolute error in RAW COUNTS over t >= BURN_IN."""
    return float(np.abs(np.asarray(pred)[BURN_IN:] - np.asarray(true)[BURN_IN:]).mean())


# --------------------------------------------------------------- baseline 3
class RecurrentBaseline(nn.Module):
    """A recurrence with BOTH heads: the intensive belief and the extensive count.

    WIDTH. `hidden=FLOOR_HIDDEN` (= S = 8) is the pre-registration's number and is
    reported as a FLOOR, because Svete and Cotterell (arXiv:2310.05161)
    lower-bound the neurons for an N-state probabilistic FSA over an alphabet
    Sigma and S = 8 sits below that bound at S*S = 64. `hidden=SKYLINE_HIDDEN`
    (= 64) is the skyline.

    CELL. Both, and not for symmetry: Weiss, Goldberg and Yahav (arXiv:1805.04908,
    ACL 2018) find the finite-precision GRU cannot count where the LSTM can, and
    the extensive head's target IS a count. A GRU-only skyline would be a skyline
    that provably cannot do the thing the extensive corner is about.

    CONSUMER. No P, no E, no pi in the signature, asserted by
    tests/beds/test_bed_h.py::test_trap_consumers_have_no_door_for_P_after_the_corrections.
    """

    CELLS = {"gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(self, s=S, hidden=SKYLINE_HIDDEN, cell="gru"):
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError("cell must be one of %s, got %r"
                             % (sorted(self.CELLS), cell))
        self.s, self.hidden, self.cell = s, hidden, cell
        self.rnn = self.CELLS[cell](s, hidden, batch_first=True)
        self.belief_head = nn.Linear(hidden, s)
        self.count_head = nn.Linear(hidden, 1)

    def forward(self, onehot):
        h, _ = self.rnn(onehot)
        return torch.log_softmax(self.belief_head(h), -1), self.count_head(h)[..., 0]


def _tensors(insts, form):
    """One-hot inputs, belief targets, and count targets scaled by T_LEN. In the
    STRONG form the prefix is prepended and scored on nothing -- it is there only
    to identify the chain."""
    n_pre = PREFIX_LEN if form == "strong" else 0
    x = torch.zeros(len(insts), n_pre + T_LEN, S)
    y = torch.zeros(len(insts), T_LEN, S)
    c = torch.zeros(len(insts), T_LEN)
    for i, inst in enumerate(insts):
        seq = np.concatenate([inst.prefix_obs, inst.obs]) if n_pre else inst.obs
        x[i, torch.arange(len(seq)), torch.as_tensor(seq)] = 1.0
        y[i] = torch.as_tensor(inst.belief, dtype=torch.float32)
        c[i] = torch.as_tensor(inst.cum_hits / T_LEN, dtype=torch.float32)
    return x, y, c, n_pre


def _losses(model, x, y, c, n_pre):
    logp, cnt = model(x)
    logp, cnt = logp[:, n_pre:], cnt[:, n_pre:]
    kl = torch.sum(y * (torch.log(y.clamp_min(1e-30)) - logp), -1).mean()
    mse = ((cnt - c) ** 2).mean()
    return kl, mse, kl + LAMBDA_COUNT * mse


def fit_rnn(train, val, form, hidden=SKYLINE_HIDDEN, cell="gru", steps=2000,
            lr=3e-3, batch=192, seed=0, log_every=250):
    """CONSUMER. Returns the BEST-VALIDATION model and the EVIDENCE that it
    converged: the loss trace on a held-out split, the last-quarter improvement
    and the final gradient norm. Convergence is shown, never assumed.

    EARLY STOPPING IS NOT OPTIONAL HERE. The first full run's STRONG-form GRU-64
    overfit -- val_total 0.0870 at step 250 rising to 0.1072 by step 2000, a
    last-quarter gain of -0.0414 -- and an earlier version of this function
    returned the LAST checkpoint, scoring the skyline on a model it had already
    beaten. A skyline that loses because it was allowed to overfit is worth as
    little as one that was too small. The best-val state is kept and restored."""
    torch.manual_seed(seed)
    xt, yt, ct, n_pre = _tensors(train, form)
    xv, yv, cv, _ = _tensors(val, form)
    model = RecurrentBaseline(S, hidden, cell)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    trace, t0 = [], time.time()
    best_val, best_state, best_step = float("inf"), None, 0
    for step in range(steps + 1):
        idx = torch.randint(0, len(train), (min(batch, len(train)),), generator=g)
        kl, mse, loss = _losses(model, xt[idx], yt[idx], ct[idx], n_pre)
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                vkl, vmse, vtot = _losses(model, xv, yv, cv, n_pre)
            gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                      if p.grad is not None))) if step else float("nan")
            trace.append({"step": step, "train_total": float(loss.detach()),
                          "val_kl": float(vkl), "val_count_mse": float(vmse),
                          "val_total": float(vtot), "grad_norm": gn})
            if float(vtot) < best_val:
                best_val, best_step = float(vtot), step
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        opt.zero_grad()
        loss.backward()
        opt.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    vals = [r["val_total"] for r in trace]
    q = min(max(1, len(vals) // 4), len(vals) - 1) if len(vals) > 1 else 0
    return model, {"cell": cell, "hidden": hidden, "lr": lr, "steps": steps,
                   "batch": batch, "seed": seed, "trace": trace,
                   "val_total_first": vals[0], "val_total_final": vals[-1],
                   "val_total_best": min(vals), "best_step": best_step,
                   "overfit": bool(vals[-1] > min(vals) + 1e-12),
                   "val_kl_final": trace[-1]["val_kl"],
                   "val_count_mse_final": trace[-1]["val_count_mse"],
                   "last_quarter_rel_gain": ((vals[-q - 1] - vals[-1])
                                             / max(abs(vals[-q - 1]), 1e-12)) if q else float("nan"),
                   "final_grad_norm": trace[-1]["grad_norm"],
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "seconds": time.time() - t0}


def rnn_scores(model, insts, form):
    """Both corners, in the bed's own units: TV for the belief, raw counts for
    the cumulant."""
    x, y, c, n_pre = _tensors(insts, form)
    with torch.no_grad():
        logp, cnt = model(x)
    b = logp[:, n_pre:].exp().numpy().astype(DT)
    pred_c = cnt[:, n_pre:].numpy().astype(DT) * T_LEN
    return {"tv": tv(b[:, BURN_IN:], y.numpy().astype(DT)[:, BURN_IN:]),
            "count_mae": float(np.mean([count_mae(pred_c[i], inst.cum_hits)
                                        for i, inst in enumerate(insts)]))}


# ------------------------------------------------ the corner rule's referent
def label_exponents(n_inst, form, seed):
    """MEASURES the scaling exponent of each label class against the token count
    and ASSIGNS NOTHING. alpha is the slope of log RMS(label at N) on log N over
    N_GRID. Intensive should come back near 0, extensive near 1; a later round
    checks that and decides what, if anything, to do with it."""
    insts = [draw_instance(seed + 1000 * k, form)[0] for k in range(n_inst)]
    logN = np.log(np.asarray(N_GRID, dtype=DT))
    out = {"N_GRID": N_GRID, "n_inst": n_inst, "form": form}
    for name, pick in (("intensive", lambda i, n: i.consequence[n - 1]),
                       ("extensive", lambda i, n: i.cum_hits[n - 1])):
        rms = np.array([np.sqrt(np.mean([pick(i, n) ** 2 for i in insts])) for n in N_GRID])
        A = np.stack([logN, np.ones_like(logN)], 1)
        coef, *_ = np.linalg.lstsq(A, np.log(rms), rcond=None)
        pred = A @ coef
        ss = float(np.sum((np.log(rms) - np.log(rms).mean()) ** 2))
        out["alpha_" + name] = float(coef[0])
        out["r2_" + name] = float(1.0 - np.sum((np.log(rms) - pred) ** 2) / ss) if ss > 0 else 1.0
        out["rms_" + name] = [float(r) for r in rms]
    return out


# ----------------------------------------------------------------- the report
ARMS = (("gru", FLOOR_HIDDEN, "FLOOR (the pre-registration width)"),
        ("gru", SKYLINE_HIDDEN, "SKYLINE"),
        ("lstm", SKYLINE_HIDDEN, "SKYLINE (the cell that can count)"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="BED-H: build, gate, and baseline.")
    ap.add_argument("--n-pool", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=48)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--seed", type=int, default=58197)
    ap.add_argument("--forms", nargs="+", default=["weak", "strong"])
    a = ap.parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)

    for form in a.forms:
        print("\n" + "=" * 78)
        print("BED-H  form=%s  S=%d T=%d prefix=%d burn=%d seed=%d"
              % (form, S, T_LEN, PREFIX_LEN if form == "strong" else 0, BURN_IN, a.seed))
        print("=" * 78)

        acc, ref, frac, rows = draw_pool(a.n_pool, form, a.seed)
        print("\n-- GATES, %d drawn --" % a.n_pool)
        print("L_GRID=%s  DECAY_SLACK=%.3f  DECAY_MARGIN=%.2f  SPREAD_FLOOR=%.2f"
              "  SIGMA_MIN_FLOOR=%.2f" % (L_GRID, DECAY_SLACK, DECAY_MARGIN,
                                          SPREAD_FLOOR, SIGMA_MIN_FLOOR))
        print("gate1 kind = %s (a theorem, not discrimination: Le Gland-Mevel 2000)"
              % rows[0]["gate1_kind"])
        for k, v in frac.items():
            print("  %-22s %s" % (k, v))
        if form == "weak":
            print("  NOTE: the WEAK form pins P at WEAK_P_SEED=%d, so every instance in"
                  % WEAK_P_SEED)
            print("        this pool shares one chain and only the trajectory varies.")
        for r in ref:
            g = r[2]
            print("  REFUSED %s  lam2=%.4f snr=%.4f sigma_min=%.4f  err=%s  decay=%.4f"
                  "  spread=(%.4f,%.4f)"
                  % (g["refusals"], g["meta"]["lambda2"], g["meta"]["snr"], g["sigma_min"],
                     ["%.4f" % g["window_err"][L] for L in L_GRID],
                     g["gate1_decay"], g["spread_intensive"], g["spread_extensive"]))

        lam = [r["meta"]["lambda2"] for r in rows]
        rel = [r["meta"]["relax_steps"] for r in rows]
        snr = [r["meta"]["snr"] for r in rows]
        sig = [r["sigma_min"] for r in rows]
        print("  lambda2 [%.4f, %.4f]  relax_steps [%.1f, %.1f]  snr [%.4f, %.4f]"
              "  sigma_min [%.4f, %.4f]"
              % (min(lam), max(lam), min(rel), max(rel), min(snr), max(snr),
                 min(sig), max(sig)))

        print("\n-- BASELINE 1: window-L forward algorithm (ORACLE with amnesia) --")
        print("  intensive metric: mean TV to the exact belief, t >= %d" % BURN_IN)
        print("  extensive metric: mean |C_hat - C| in RAW COUNTS, t >= %d" % BURN_IN)
        print("  prior outside the window: the stationary distribution of P")
        print("  n_accepted = %d of %d drawn" % (len(acc), a.n_pool))
        for L in L_GRID:
            e = [g["window_err"][L] for _, _, g in acc]
            cm = [count_mae(np.cumsum(hm.window_forward(i.obs, L) @ i.hit_vector),
                            i.cum_hits) for i, hm, _ in acc[:16]]
            print("  L=%-3d  TV %.4f  sd %.4f  [%.4f, %.4f]   count-MAE %.3f (n=16)"
                  % (L, np.mean(e), np.std(e), np.min(e), np.max(e), np.mean(cm)))

        print("\n-- DERIVATION vs MEASUREMENT for window-4 --")
        dm = np.array([g["decay_per_step_measured"] for _, _, g in acc])
        dp = np.array([g["decay_per_step_predicted"] for _, _, g in acc])
        wm = np.array([g["w4_measured"] for _, _, g in acc])
        dl = np.array([g["decay_per_step_lyapunov"] for _, _, g in acc])
        dq = np.array([g["decay_per_step_quenched"] for _, _, g in acc])
        wg = np.array([g["w4_measured_geo"] for _, _, g in acc])
        print("  NO DERIVED LEVEL IS REPORTED. arXiv:1710.06078 eq. (10) carries an")
        print("  unestimated constant, so the theory gives a SLOPE and no level.")
        print("  per-step decay  DERIVED exp(Lyapunov gap)        %.4f" % dl.mean())
        print("  per-step decay  MEASURED quenched (geometric)    %.4f   ratio %.4f"
              % (dq.mean(), (dq / dl).mean()))
        print("  per-step decay  MEASURED annealed (arithmetic)   %.4f   ratio %.4f"
              " <- Jensen" % (dm.mean(), (dm / dl).mean()))
        print("  per-step decay  lambda_2 of P, NAMED WRONG PRED  %.4f   ratio %.4f"
              % (dp.mean(), (dq / dp).mean()))
        print("  rate fitted on L = %s" % (acc[0][2]["rate_fit_L"],))
        print("  window-4 TV     measured arithmetic %.4f   geometric %.4f"
              % (wm.mean(), wg.mean()))
        print("  pre-registered bar 0.16-0.22: measured mean %.4f, fraction inside %.4f"
              % (wm.mean(), float(((wm >= 0.16) & (wm <= 0.22)).mean())))

        def pool(n, off):
            out, k = [], 0
            while len(out) < n:
                inst, hmm = draw_instance(a.seed + off + 1000 * k, form)
                if gate_report(inst, hmm)["accepted"]:
                    out.append(inst)
                k += 1
            return out

        tr = pool(a.n_train, 5_000_000)
        va = pool(a.n_val, 7_000_000)
        te = pool(a.n_test, 9_000_000)

        print("\n-- BASELINE 2: EMA of one-hot observations, and the naive hit count --")
        gamma, scores = fit_ema(va)
        ema_tv = float(np.mean([tv(ema_belief(i.obs, gamma, S)[BURN_IN:],
                                   i.belief[BURN_IN:]) for i in te]))
        ema_cnt = float(np.mean([count_mae(np.cumsum(ema_belief(i.obs, gamma, S) @ i.hit_vector),
                                           i.cum_hits) for i in te]))
        naive_cnt = float(np.mean([count_mae(count_obs(i.obs), i.cum_hits) for i in te]))
        print("  gamma chosen on val = %.2f (val TV %.4f)" % (gamma, scores[gamma]))
        print("  TEST  EMA      TV %.4f   count-MAE %.3f   n_test=%d"
              % (ema_tv, ema_cnt, len(te)))
        print("  TEST  raw hit count (no belief at all)   count-MAE %.3f" % naive_cnt)
        print("  val curve: %s" % {k: round(v, 4) for k, v in sorted(scores.items())})

        print("\n-- BASELINE 3: recurrences. FLOOR = %d units, SKYLINE = %d units --"
              % (FLOOR_HIDDEN, SKYLINE_HIDDEN))
        print("  lr chosen by a %d-step val probe, then refit to %d steps, batch %d"
              % (a.probe_steps, a.steps, a.batch))
        trained = []
        for cell, hidden, tag in ARMS:
            probe = []
            for lr in a.lr_grid:
                _, pev = fit_rnn(tr, va, form, hidden, cell, steps=a.probe_steps, lr=lr,
                                 batch=a.batch, seed=a.seed % 1000, log_every=a.probe_steps)
                probe.append((pev["val_total_final"], lr))
            best_lr = min(probe)[1]
            m, ev = fit_rnn(tr, va, form, hidden, cell, steps=a.steps, lr=best_lr,
                            batch=a.batch, seed=a.seed % 1000)
            sc = rnn_scores(m, te, form)
            print("\n  %s-%d  %s" % (cell.upper(), hidden, tag))
            print("    probe: %s -> lr=%g   params=%d   %.0fs"
                  % (["%g:%.4f" % (lr, v) for v, lr in probe], best_lr,
                     ev["n_params"], ev["seconds"]))
            print("    CONVERGENCE  val_total %.4f -> %.4f (best %.4f)"
                  "   last-quarter gain %+.4f   |grad| %.3e"
                  % (ev["val_total_first"], ev["val_total_final"], ev["val_total_best"],
                     ev["last_quarter_rel_gain"], ev["final_grad_norm"]))
            print("    trace (step, train_total, val_KL, val_count_MSE, |grad|):")
            for r in ev["trace"]:
                print("      %6d  %.5f  %.5f  %.6f  %.3e"
                      % (r["step"], r["train_total"], r["val_kl"], r["val_count_mse"],
                         r["grad_norm"]))
            print("    TEST  TV %.4f   count-MAE %.3f   n_test=%d"
                  % (sc["tv"], sc["count_mae"], len(te)))
            trained.append(("%s-%d" % (cell.upper(), hidden), m))

        print("\n-- REFUSAL LEG: BOTTOM scored against the exact filter posterior --")
        print("  anchor: H(b) >= ln S - delta, delta = ln S - H(one look), a closed")
        print("  form in S and snr fixed at generation time. NOT tuned for coverage,")
        print("  and NO gate is applied to the base rate -- that would be tuning.")
        print("  ln S = %.4f   delta range over the pool [%.4f, %.4f]"
              % (LN_S, min(r["meta"]["refuse_delta"] for r in rows),
                 max(r["meta"]["refuse_delta"] for r in rows)))
        br = np.array([r["meta"]["refuse_base_rate"] for r in rows])
        print("  refusal base rate over %d drawn: mean %.4f  sd %.4f  [%.4f, %.4f]"
              % (len(rows), br.mean(), br.std(), br.min(), br.max()))
        print("  arm                     AUROC(bottom)  risk@1.0  risk@0.8  risk@0.5   AURC")
        ref_arms = [("exact oracle (defines the label)", lambda i, hm: i.belief),
                    ("window-4 (ORACLE, amnesia)", lambda i, hm: hm.window_forward(i.obs, 4)),
                    ("window-8 (ORACLE, amnesia)", lambda i, hm: hm.window_forward(i.obs, 8)),
                    ("EMA gamma=%.2f (CONSUMER)" % gamma,
                     lambda i, hm: ema_belief(i.obs, gamma, S))]
        te_pairs = [(i, HMM.draw(np.random.default_rng(
            WEAK_P_SEED if form == "weak" else i.meta["seed"]))) for i in te]
        for name, fn in ref_arms:
            rs = [refusal_scores(fn(i, hm), i, refuse_delta_for(hm.snr))
                  for i, hm in te_pairs]
            agg = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
            print("  %-38s %.4f    %.4f    %.4f    %.4f   %.4f"
                  % (name, agg["auroc"], agg["risk_at_1"], agg["risk_at_0.8"],
                     agg["risk_at_0.5"], agg["aurc"]))
        for name, mdl in trained:
            x, _, _, n_pre = _tensors(te, form)
            with torch.no_grad():
                pb = mdl(x)[0][:, n_pre:].exp().numpy().astype(DT)
            rs = [refusal_scores(pb[k], i, refuse_delta_for(hm.snr))
                  for k, (i, hm) in enumerate(te_pairs)]
            agg = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
            print("  %-38s %.4f    %.4f    %.4f    %.4f   %.4f"
                  % (name + " (CONSUMER)", agg["auroc"], agg["risk_at_1"],
                     agg["risk_at_0.8"], agg["risk_at_0.5"], agg["aurc"]))

        print("\n-- LABEL CLASSES (measured; no corner assigned) --")
        ex = label_exponents(24, form, a.seed + 11_000_000)
        print("  N_GRID=%s  n_inst=%d" % (ex["N_GRID"], ex["n_inst"]))
        print("  intensive (consequence)  alpha %+.4f  R2 %.4f  RMS %s"
              % (ex["alpha_intensive"], ex["r2_intensive"],
                 ["%.4f" % r for r in ex["rms_intensive"]]))
        print("  extensive (cum_hits)     alpha %+.4f  R2 %.4f  RMS %s"
              % (ex["alpha_extensive"], ex["r2_extensive"],
                 ["%.4f" % r for r in ex["rms_extensive"]]))
        print("  separation alpha_ext - alpha_int = %+.4f"
              % (ex["alpha_extensive"] - ex["alpha_intensive"]))

        print("\n-- TRAP --")
        inst, hmm = draw_instance(a.seed, form)
        pay = inputs_for_consumer(inst)
        print("  consumer keys: %s" % sorted(pay))
        print("  leaks of P in payload: %s" % scan_for_matrix(pay, hmm.P))
        print("  leaks of E in payload: %s" % scan_for_matrix(pay, hmm.E))
        planted = dict(pay)
        planted["obs"] = np.concatenate([pay["obs"].astype(DT),
                                         hmm.P.ravel()[::-1] * 3.0 + 1.0])
        print("  scanner fires on a planted P: %s"
              % (scan_for_matrix(planted, hmm.P) != []))
        seq = np.concatenate([inst.prefix_obs, inst.obs]) if form == "strong" else inst.obs
        big = np.zeros((S, S), dtype=DT)
        for u, w in zip(seq[:-1], seq[1:]):
            big[u, w] += 1.0
        big /= big.sum(1, keepdims=True)
        print("  best naive recovery from inputs (observation bigrams) vs P: "
              "Frobenius %.4f, mean row TV %.4f"
              % (float(np.linalg.norm(big - hmm.P)), tv(big, hmm.P)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
