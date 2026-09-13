"""An intervention on a context window: the CONSEQUENCE SWAP AT FIXED SURFACE.

WHAT THIS FILE IS FOR. The field equation sources curvature from a stress tensor
built "under the read's interventions". On a board an intervention is obvious --
clamp a square, force a move. On a CONTEXT WINDOW there was no definition, and
with none the source collapses into local information density, which is occupied
ground and destroys the novelty. The owner's definition, made measurable here:
an intervention is a CONSEQUENCE SWAP AT FIXED SURFACE -- replace a span with one
of the same shape whose downstream consequence differs -- and the source is
DIRECTED, because A changing B is not B changing A.

THE BED IS SYNTHETIC AND EXACT, AND THAT IS THE POINT. Seed 20260913, 12 slots,
8 outcomes. Five slots are PLANTED branch points at [2, 4, 5, 8, 10]; each may
exclude one of two outcomes and the pools [(0, 1), (1, 2), (2, 3), (3, 4),
(4, 0)] form a cycle, so adjacent branch points share an outcome and a swap can
be MASKED by its neighbours. Every other slot is DECOR: its tokens differ in
surface and exclude nothing. The futures a context admits is the INTERSECTION of
its tokens' masks, so the truth of every swap is known by construction; the
admitted set never empties (sizes [4, 5] over the first 8 stories) and the read
is defined everywhere.

THE READ IS A COUNT OF FUTURES, NOT A MODEL. q is the uniform distribution over
the admitted set, so dq is COMPUTED and not estimated. It is not a language
model, not a trained probe, not an approximation of one. Two further reads are
run beside it and both are labelled where they appear: a naive-Bayes COUNTS read
fitted from a corpus (ESTIMATED), and an ALIKENESS read that scores how similar
the context's surface is to each outcome -- which is what self-attention computes
(Vaswani et al., arXiv:1706.03762) -- carried as the PLANTED NEGATIVE.

THE SURFACE IS HELD, TOKEN BY TOKEN. Every branch token has a consequence
partner and a surface-only partner and BOTH sit at Hamming 3 from it, checked
over all 20 branch tokens individually rather than on average. Without that the
gap below would be a surface gap wearing a different name.

THE THREE SWAPS (`python -m ceqjepa.intent_do`, block (c), this box, CPU). dq is
the mean movement of the read's commitment over the other positions, and the
WORST single swap is carried beside the mean because a mean over 1000 swaps
hides a handful of corrupted ones -- the worst is bounded on its own:

    read      family         mean       CI                     worst      swaps
    oracle    consequence    0.170141   [0.165219, 0.175062]   0.240909     1000
    oracle    surface-only   0.000000   [0.000000, 0.000000]   0.000000     1000
    oracle    null           0.000000   [0.000000, 0.000000]   0.000000     2400
    counts    consequence    0.163626   [0.158706, 0.168546]   0.296877     1000
    counts    surface-only   0.011893   [0.011632, 0.012154]   0.028126     1000
    density   consequence    0.261442   [0.253217, 0.269668]   0.685914     1000
    density   surface-only   0.287905   [0.280290, 0.295520]   0.627901     1000

THE GAP IS THE CLAIM. On the exact read the difference is 0.170141 and the ratio
has no value because the surface-only control is EXACTLY 0.0 -- not in the mean
but in every one of the 1000 swaps, the same bitwise floor the null swap sits
on. On the estimated read the difference is 0.151733 at a ratio of 13.8, its
worst single surface-only swap reaches only 0.028126, and that reading is an
ESTIMATOR GAP rather than a consequence: it falls 0.094629 -> 0.032580 ->
0.011893 as the corpus goes 500 -> 5000 -> 50000.

THE CONTROL HAS TEETH, AND THE PLANTED NEGATIVE PROVES IT. Against the alikeness
read the same two matched-surface swaps read 0.261442 and 0.287905 -- a ratio of
0.9 and a difference of -0.026463, the meaning-PRESERVING swap moving the read
slightly FURTHER than the meaning-changing one. A similarity read cannot separate
consequence from surface, so the gap on the other two reads is a property of the
READ and not of the bed. This is the occupied ground being measured and failing.

IT IS NOT LOCAL INFORMATION DENSITY, AND HALF OF THAT IS AN IDENTITY WHILE THE
OTHER HALF IS THE MEASUREMENT. The generator splits the consequence swaps by
whether the NUMBER of admitted futures is preserved. On the preserved half the
exact read's entropy move is 0.0 as a max, and that is an IDENTITY, not a
finding: a uniform read's entropy is the log of the count, so equal counts force
it. THE FINDING IS THE OTHER HALF -- over those 492 swaps dq is 0.132431
[0.123717, 0.141145], nonzero exactly where every entropy, surprisal and density
is pinned at zero by that identity. On the other 508 swaps dq is 0.206663
against a mean entropy move of 0.239024. The estimated read reads 0.140814 on
the preserved half. The alikeness read cannot make the split at all: 0.259780
against 0.263053, its own entropy moving on both halves.

THE FIELD EQUATION'S OWN FORM CANNOT BE DIRECTED. T = |dq(i) - dq(j)| is
symmetric in its two indices as an ALGEBRAIC IDENTITY: measured over every read
and story the worst |T - T^T| is 0.0, and no bed can change that. The directed
source therefore needs the TWO-INDEX form T(i, j) = the movement of the read at
position j under an intervention at i, which is what this module builds.

DIRECTEDNESS, MEASURED, NOT ASSUMED. Over the 66 position pairs of the window,
|T(i,j) - T(j,i)| has median 0.159167, p90 0.173333 and max 0.177500, with
0.6818 of pairs asymmetric above tolerance. The symmetrised control -- the same
statistic on (T + T^T) / 2 -- is 0.0 at both its median and its max. The
asymmetry is therefore not a rounding artifact.

FORWARD ONLY? NO, ONCE THE MASK IS REMOVED. Under a causal prefix read the
backward block is 0.0 over all 3960 entries, which is the MASK speaking and not
a finding -- reporting it as evidence would be circular and the module says so
in a place a test can read. Under the bidirectional leave-one-out read the
backward mean is 0.074954 with 1391 of 3960 entries nonzero. Consequence on a
context window is not a forward-only quantity; the direction lives in the
asymmetry, not in the token order.

THE REFUSAL IS A VALUE. A swap whose admitted set does not move but whose span
does carry a constraint has no defined consequence direction, and the read
returns Refusal("masked-consequence", reason) -- never NaN, never a 0.0
standing in for a measurement. A swap whose span carries no constraint at all is
a different cell and is ANSWERED with exactly 0.0. Scored on the 3x2 over 1920
cases (105 undefined, 1815 answerable, 1520 of them null), with sensitivity and
specificity SEPARATE and both trivial criteria beside the real one:

    read      criterion            sensitivity   specificity
    oracle    criterion (ours)        100.00%       100.00%
    oracle    refuse everything       100.00%         0.00%
    oracle    answer everything         0.00%       100.00%
    counts    criterion (ours)          0.00%       100.00%
    density   criterion (ours)          0.00%       100.00%

WHAT THAT SECOND HALF MEANS, and it is a limit on the whole programme. At the
pinned tolerance the counts row is CELL FOR CELL the answer-everything row: an
ESTIMATED read cannot certify that nothing moved, so it cannot refuse at all.
Sweeping the threshold, 1e-12 buys nothing, 1e-02 reaches 20.95% sensitivity at
82.20% specificity (22 true and 323 false refusals), and 1e-01 reaches 100.00%
on both -- but that row was chosen AGAINST THE LABELS and is an upper bound on
what a threshold could do here, not an operating point anyone could set in
advance. Refusal on a context window currently requires an exact read.

CURVATURE, STATIC, WIRED ONLY AS FAR AS THE EVIDENCE CARRIES. The context graph
takes an edge wherever the symmetrised source is nonzero -- the whole rule, so
there is no threshold to tune. On the oracle source that is 45 edges over 12
nodes with 0 refusals and kappa 0.333333, CI [0.299831, 0.366836]. The other two
reads spread mass everywhere and give the complete graph, 66 edges at a constant
kappa 0.545455 with a zero-width CI, which is what a featureless source looks
like. Symmetrisation is what the instrument requires and it DISCARDS 0.6268 of
the oracle source's mass (0.5443 and 0.1495 for counts and density). THE FLOW IS
NOT RUN and no figure about where it would settle appears anywhere in this file,
because such a figure needs a shortcut census beside it and this module computes
none.

WHAT IS NOT CLAIMED. Nothing here is measured on natural language: the bed is
generated and its ground truth is its generator, which is what makes dq exact and
also what bounds the reading. The oracle read cannot on its own show the
surface-only control has teeth -- it reads 0.0 there by construction -- which is
why the alikeness read is carried. The 0.6268 discarded by symmetrisation is the
size of the mismatch between a directed source and an undirected instrument, and
closing it is not attempted here. No flow is run, and no claim is made about
where curvature would settle.

RUN: python -m ceqjepa.intent_do
"""

import time
from collections import namedtuple
from functools import lru_cache

import numpy as np

from ceqjepa.curvature import (Refusal, curvature_from_graph, group_stats,
                               is_refusal)

__all__ = [
    "SEED", "WINDOW", "N_OUTCOMES", "BRANCH_SLOTS", "POOLS", "SURFACE_BITS",
    "HAMMING", "TOKENS_PER_BRANCH", "N_STORIES", "N_T_STORIES", "N_CASE_STORIES",
    "CORPUS_N", "CORPUS_SWEEP", "TOL", "BUDGET_S", "READS", "CONSEQUENCE_READS",
    "PLANTED_NEGATIVE_READ", "DEFINED", "UNDEFINED", "NULL", "MASKED",
    "Token", "SwapCase", "vocabulary", "sample_stories", "read_by_name",
    "OracleRead", "AdditiveRead", "counts_read", "density_read",
    "swap_table", "gap", "surface_profile", "estimator_gap", "entropy_control",
    "t_directed", "t_field", "t_causal", "directedness", "backward_block",
    "classify", "refusal_cases", "score_3x2", "score_board",
    "tolerance_sweep", "context_graph", "context_curvature",
]

# ---------------------------------------------------------------------------
# THE PINNED CONSTANTS
# ---------------------------------------------------------------------------

#: The one seed behind every draw in this file. Stated, not implied.
SEED = 20260913

#: Slots in the context window.
WINDOW = 12

#: Outcomes the generator can end on. A constraint set is a bitmask over these.
N_OUTCOMES = 8

#: The slots that carry a real constraint. Every other slot is DECOR: its tokens
#: differ in surface and admit every outcome, which is what makes them the null
#: cell rather than a nuisance.
BRANCH_SLOTS = (2, 4, 5, 8, 10)

#: Each branch slot may exclude one of two outcomes, and the pools form a cycle
#: so that ADJACENT SLOTS SHARE A BIT. That sharing is the whole reason the
#: UNDEFINED cell is reachable: a swap whose bit is already excluded elsewhere
#: changes the admitted set not at all, and then no consequence direction exists.
POOLS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))

#: Surface codes are this many bits wide, and the two swap families move exactly
#: HAMMING of them. Holding that equal is what makes the surface-only swap a
#: control rather than a different question.
SURFACE_BITS = 8
HAMMING = 3

#: Disjoint weight-HAMMING deltas: one for the consequence axis, one for the
#: surface-only axis, so the vocabulary is a 2x2 grid and every token has both
#: partners at the same distance.
DELTA_CONSEQUENCE = 0b00000111
DELTA_SURFACE = 0b00111000

TOKENS_PER_BRANCH = 4
TOKENS_PER_DECOR = 4

#: Stories behind the three-swap table, the T matrices, and the refusal cases.
N_STORIES = 200
N_T_STORIES = 60
N_CASE_STORIES = 80

#: The count-based read's corpus. CORPUS_N is the pinned one; the sweep is there
#: because the surface-only reading of an ESTIMATED read is an estimator gap and
#: has to be seen to shrink.
CORPUS_SWEEP = (500, 5000, 50000)
CORPUS_N = CORPUS_SWEEP[-1]

#: The wall-clock bar, in one place. It was a literal typed into two prints
#: and two asserts: four chances to drift apart and no test that would see it.
BUDGET_S = 180.0

#: One tolerance, stated once. An exact read separates at machine epsilon; an
#: estimated one cannot, and that is reported rather than tuned around.
TOL = 1e-12

#: The reads. "oracle" and "counts" are consequence reads; "density" is the
#: PLANTED NEGATIVE -- it scores how ALIKE the context is to each outcome, which
#: is what self-attention computes (Vaswani et al., arXiv:1706.03762), and it has
#: no access to consequence at all.
READS = ("oracle", "counts", "density")
CONSEQUENCE_READS = ("oracle", "counts")
PLANTED_NEGATIVE_READ = "density"

#: Softmax temperature of the alikeness read. Any positive value makes it a
#: similarity read; this one is pinned so the numbers below are reproducible.
DENSITY_BETA = 6.0

DEFINED = "DEFINED"
UNDEFINED = "UNDEFINED"
NULL = "NULL"

#: The one refusal code this module can emit.
MASKED = "masked-consequence"

_FULL = (1 << N_OUTCOMES) - 1

Token = namedtuple("Token", "surface mask consequence surface_only")
SwapCase = namedtuple("SwapCase", "story slot tok_old tok_new family truth")


# ---------------------------------------------------------------------------
# (a) THE BED: a generator with PLANTED branch points, and its exact read
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def vocabulary():
    """slot -> the four tokens available there, as a 2x2 surface/consequence grid.

    THE GENERATOR, stated. A story is WINDOW slots; each slot draws one token
    uniformly. A token carries a SURFACE code (SURFACE_BITS bits, the shape a
    reader sees) and a MASK (the outcomes it still admits, the consequence). The
    futures a context admits is the INTERSECTION of its tokens' masks, so the
    ground truth of every swap is known by construction and nothing here is
    estimated.

    The grid. Token (a, s) has surface base ^ (DELTA_CONSEQUENCE if a) ^
    (DELTA_SURFACE if s), and mask M[a]. Its CONSEQUENCE partner is (1 - a, s)
    and its SURFACE-ONLY partner is (a, 1 - s), so both partners sit at Hamming
    HAMMING from it and exactly one of them moves the mask.

    At a BRANCH slot M = (FULL without outcome u, FULL without outcome v) for
    that slot's pool. At a DECOR slot both entries are FULL: the surface still
    moves, the consequence cannot. Three outcomes lie in no pool at all, so the
    admitted set is never empty and the read is defined everywhere.
    """
    rng = np.random.default_rng(SEED)
    vocab = {}
    pool_of = dict(zip(BRANCH_SLOTS, POOLS))
    for slot in range(WINDOW):
        base = int(rng.integers(0, 1 << SURFACE_BITS))
        if slot in pool_of:
            u, v = pool_of[slot]
            masks = (_FULL & ~(1 << u), _FULL & ~(1 << v))
        else:
            masks = (_FULL, _FULL)
        toks = []
        for a in (0, 1):
            for s in (0, 1):
                surf = base ^ (DELTA_CONSEQUENCE if a else 0) ^ (DELTA_SURFACE if s else 0)
                toks.append(Token(surface=surf, mask=masks[a],
                                  consequence=(1 - a) * 2 + s,
                                  surface_only=a * 2 + (1 - s)))
        vocab[slot] = tuple(toks)
    return vocab


@lru_cache(maxsize=8)
def _mask_table():
    v = vocabulary()
    return np.array([[t.mask for t in v[s]] for s in range(WINDOW)], dtype=np.int64)


@lru_cache(maxsize=8)
def _surface_table():
    v = vocabulary()
    return np.array([[t.surface for t in v[s]] for s in range(WINDOW)], dtype=np.int64)


def sample_stories(n, seed=SEED):
    """n stories, each WINDOW token indices drawn uniformly. Reproducible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, TOKENS_PER_BRANCH, size=(int(n), WINDOW))


def _bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(N_OUTCOMES)) & 1


def _uniform_over(mask):
    b = _bits(mask).astype(float)
    total = b.sum(-1, keepdims=True)
    if not (total > 0).all():
        raise ValueError("an empty admitted set reached the read: the bed is broken")
    return b / total


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def tv(p, q, axis=-1):
    """Total variation distance. The read's commitment moves by this much."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=axis)


class OracleRead:
    """The EXACT read: uniform over the futures the generator still admits.

    WHAT IT IS: a function from a context to a distribution over outcomes,
    obtained by intersecting the tokens' constraint sets and counting what
    survives. dq is therefore COMPUTED, not estimated, and the surface-only swap
    reads exactly 0.0 rather than nearly so.

    WHAT IT IS NOT: a language model, a trained probe, or an approximation of
    one. It is the generator's own bookkeeping read back, which is precisely why
    it can serve as ground truth and precisely why it cannot, on its own, prove
    that the surface-only control has teeth -- see the alikeness read for that.
    """

    name = "oracle"
    estimated = False

    def __init__(self, vocab=None):
        self.vocab = vocab or vocabulary()
        self.M = _mask_table()

    def _row(self, story):
        return self.M[np.arange(WINDOW), np.asarray(story)]

    def q_all(self, story):
        """Leave-one-out read at every position: (WINDOW, N_OUTCOMES)."""
        m = self._row(story)
        pre = np.empty(WINDOW + 1, dtype=np.int64)
        suf = np.empty(WINDOW + 1, dtype=np.int64)
        pre[0] = suf[WINDOW] = _FULL
        for j in range(WINDOW):
            pre[j + 1] = pre[j] & m[j]
        for j in range(WINDOW - 1, -1, -1):
            suf[j] = suf[j + 1] & m[j]
        return _uniform_over(pre[:WINDOW] & suf[1:])

    def q(self, story, j):
        """The leave-j-out read at one position. q_all is the whole window."""
        return self.q_all(story)[int(j)]

    def q_causal_all(self, story):
        """Prefix read at every position: the causal mask, applied honestly."""
        m = self._row(story)
        out = np.empty(WINDOW, dtype=np.int64)
        acc = _FULL
        for j in range(WINDOW):
            acc &= m[j]
            out[j] = acc
        return _uniform_over(out)

    def q_full(self, story):
        acc = _FULL
        for m in self._row(story):
            acc &= int(m)
        return _uniform_over(acc)

    def solo(self, slot, tok):
        return _uniform_over(int(self.M[slot, tok]))


class AdditiveRead:
    """A read whose log-score is a SUM over observed slots.

    Both non-oracle reads have this shape, so leave-one-out costs a subtraction
    and the causal prefix costs a cumulative sum. `estimated` says whether the
    table behind it was fitted from a finite corpus; a read with estimated=True
    cannot certify that nothing moved, which is what the refusal board measures.
    """

    def __init__(self, name, logscore, logprior, estimated):
        self.name = name
        self.LS = np.asarray(logscore, dtype=float)
        self.lp = np.asarray(logprior, dtype=float)
        self.estimated = bool(estimated)

    def _rows(self, story):
        return self.LS[np.arange(WINDOW), np.asarray(story)]

    def q_all(self, story):
        rows = self._rows(story)
        return _softmax(self.lp + rows.sum(0) - rows, axis=1)

    def q(self, story, j):
        """The leave-j-out read at one position. q_all is the whole window."""
        return self.q_all(story)[int(j)]

    def q_causal_all(self, story):
        return _softmax(self.lp + np.cumsum(self._rows(story), axis=0), axis=1)

    def q_full(self, story):
        return _softmax(self.lp + self._rows(story).sum(0))

    def solo(self, slot, tok):
        return _softmax(self.lp + self.LS[slot, tok])


@lru_cache(maxsize=8)
def counts_read(corpus_n=CORPUS_N):
    """A read that does NOT see the constraint sets: naive Bayes over a corpus.

    ESTIMATED, and labelled so. The corpus is corpus_n stories from the same
    generator, each ending on an outcome drawn uniformly from what it admits;
    the read is add-one-smoothed counts of P(outcome) and P(token | outcome).
    The factorisation is WRONG for this bed -- the truth is an intersection, not
    a product -- and that is on purpose: a read that is merely a re-spelling of
    the generator could not fail the surface-only control for any reason worth
    knowing about.
    """
    rng = np.random.default_rng(SEED + 1)
    stories = rng.integers(0, TOKENS_PER_BRANCH, size=(int(corpus_n), WINDOW))
    M = _mask_table()
    adm = np.bitwise_and.reduce(M[np.arange(WINDOW)[None, :], stories], axis=1)
    b = _bits(adm).astype(float)
    cum = np.cumsum(b, axis=1)
    pick = rng.random((len(stories), 1)) * cum[:, -1:]
    y = (cum < pick).sum(axis=1)

    flat = ((np.arange(WINDOW)[None, :] * TOKENS_PER_BRANCH + stories) * N_OUTCOMES
            + y[:, None])
    c = np.bincount(flat.ravel(),
                    minlength=WINDOW * TOKENS_PER_BRANCH * N_OUTCOMES)
    c = c.reshape(WINDOW, TOKENS_PER_BRANCH, N_OUTCOMES).astype(float)
    cy = np.bincount(y, minlength=N_OUTCOMES).astype(float)
    ls = np.log((c + 1.0) / (cy + TOKENS_PER_BRANCH)[None, None, :])
    lp = np.log((cy + 1.0) / (cy.sum() + N_OUTCOMES))
    return AdditiveRead("counts", ls, lp, estimated=True)


@lru_cache(maxsize=1)
def density_read():
    """THE PLANTED NEGATIVE: how ALIKE the context is to each outcome.

    Every outcome gets a surface prototype; the score of an outcome is the mean
    bit agreement between the context's surface codes and that prototype, at
    temperature DENSITY_BETA. This is a content-similarity read -- symmetric in
    content, exactly what self-attention scores -- and it never touches a
    constraint set. Both swap families move the surface by HAMMING bits, so this
    read must move about equally for both, and the consequence/surface-only gap
    that the real reads open must collapse here. If it did not, the gap would be
    a property of the bed rather than of the read, and would prove nothing.
    """
    rng = np.random.default_rng(SEED + 2)
    proto = rng.integers(0, 1 << SURFACE_BITS, size=N_OUTCOMES)
    S = _surface_table()
    agree = np.empty((WINDOW, TOKENS_PER_BRANCH, N_OUTCOMES))
    for slot in range(WINDOW):
        for tok in range(TOKENS_PER_BRANCH):
            for y in range(N_OUTCOMES):
                h = bin(int(S[slot, tok]) ^ int(proto[y])).count("1")
                agree[slot, tok, y] = (SURFACE_BITS - h) / SURFACE_BITS
    return AdditiveRead("density", DENSITY_BETA * agree,
                        np.zeros(N_OUTCOMES), estimated=False)


@lru_cache(maxsize=8)
def read_by_name(name):
    if name == "oracle":
        return OracleRead()
    if name == "counts":
        return counts_read()
    if name == "density":
        return density_read()
    raise ValueError("unknown read %r, expected one of %r" % (name, READS))


# ---------------------------------------------------------------------------
# (b) (c) THE THREE SWAPS
# ---------------------------------------------------------------------------

def surface_profile():
    """Hamming distance to each partner, token by token, over the branch slots.

    Reported as two lists rather than a mean: a mean would let one mismatched
    pair hide behind the rest, and the whole force of the surface-only control
    is that the two families are matched EVERYWHERE, not on average.
    """
    v = vocabulary()
    csq, syn = [], []
    for slot in BRANCH_SLOTS:
        for tok in v[slot]:
            csq.append(bin(tok.surface ^ v[slot][tok.consequence].surface).count("1"))
            syn.append(bin(tok.surface ^ v[slot][tok.surface_only].surface).count("1"))
    return dict(consequence=csq, surface_only=syn)


def _swap_dq(read, story, qa, slot, tok_new):
    """How far the read's commitment moves AT EVERY POSITION, then averaged.

    The per-position movements are the row of the directed source; this scalar
    is their mean over the positions other than the swapped one, so the three
    swap families and the tensor are the same measurement seen two ways.
    """
    x2 = np.asarray(story).copy()
    x2[slot] = tok_new
    per_pos = tv(read.q_all(x2), qa, axis=1)
    return per_pos, float(np.delete(per_pos, slot).mean())


@lru_cache(maxsize=8)
def swap_table(read):
    """The three swap families over N_STORIES stories, with their counts.

    (a) CONSEQUENCE  surface held at HAMMING, admitted set moved.
    (b) SURFACE_ONLY surface moved by the same HAMMING, admitted set held.
    (c) NULL         the span replaced by itself, at every slot.
    """
    v = vocabulary()
    stories = sample_stories(N_STORIES, SEED)
    out = dict(consequence=[], surface_only=[], null=[])
    for story in stories:
        qa = read.q_all(story)
        for slot in BRANCH_SLOTS:
            tok = v[slot][story[slot]]
            out["consequence"].append(_swap_dq(read, story, qa, slot, tok.consequence)[1])
            out["surface_only"].append(_swap_dq(read, story, qa, slot, tok.surface_only)[1])
        for slot in range(WINDOW):
            out["null"].append(_swap_dq(read, story, qa, slot, story[slot])[1])
    return {k: np.asarray(vals, dtype=float) for k, vals in out.items()}


@lru_cache(maxsize=8)
def gap(read):
    """The three families side by side, each with a CI and its swap count.

    The gap between (a) and (b) IS the claim, quoted as a difference and as a
    ratio, with the (c) floor beside them. Where (b) is exactly zero the ratio
    has no value and says so rather than printing a large number.
    """
    tab = swap_table(read)
    out = {}
    for family, vals in tab.items():
        st = group_stats(vals)
        st["n"] = st.pop("n_edges")
        out[family] = st
    a, b = out["consequence"]["mean"], out["surface_only"]["mean"]
    out["difference"] = a - b
    out["ratio"] = float("inf") if b == 0.0 else a / b
    out["ratio_text"] = ("undefined (surface-only is exactly %r)" % b
                         if b == 0.0 else "%.1f" % (a / b))
    out["read"] = read.name
    return out


def _entropy(p):
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


@lru_cache(maxsize=8)
def entropy_control(read):
    """IS THE SOURCE JUST LOCAL INFORMATION DENSITY? Split by the entropy move.

    This is the collapse the whole definition exists to avoid, so it is measured
    rather than argued. A consequence swap at a branch point switches WHICH
    outcome is excluded, so on part of the bed it leaves the NUMBER of admitted
    futures identical while changing which futures they are. That partition is
    made by the GENERATOR -- popcount(A_old) == popcount(A_new) -- and is the
    same for every read; each read then reports its own dq and its own entropy
    move on both halves. If dq is large where the entropy move is exactly zero,
    the source is not an entropy, a surprisal or a density.
    """
    v = vocabulary()
    M = _mask_table()
    stories = sample_stories(N_STORIES, SEED)
    out = {True: dict(dq=[], dh=[]), False: dict(dq=[], dh=[])}
    for story in stories:
        qa = read.q_all(story)
        rest_all = int(np.bitwise_and.reduce(M[np.arange(WINDOW), story]))
        q0 = read.q_full(story)
        h0 = _entropy(q0)
        for slot in BRANCH_SLOTS:
            tok = v[slot][story[slot]]
            x2 = np.asarray(story).copy()
            x2[slot] = tok.consequence
            a_new = int(np.bitwise_and.reduce(M[np.arange(WINDOW), x2]))
            same_size = int(_bits(rest_all).sum()) == int(_bits(a_new).sum())
            _, d = _swap_dq(read, story, qa, slot, tok.consequence)
            out[same_size]["dq"].append(d)
            out[same_size]["dh"].append(abs(_entropy(read.q_full(x2)) - h0))
    res = {}
    for same_size, label in ((True, "entropy_preserving"), (False, "entropy_moving")):
        dq = np.asarray(out[same_size]["dq"], dtype=float)
        dh = np.asarray(out[same_size]["dh"], dtype=float)
        st = group_stats(dq)
        st["n"] = st.pop("n_edges")
        st["dh_mean"] = float(dh.mean()) if dh.size else float("nan")
        st["dh_max"] = float(dh.max()) if dh.size else float("nan")
        res[label] = st
    return res


@lru_cache(maxsize=1)
def estimator_gap():
    """The counts read's surface-only reading against corpus size.

    A read fitted from a finite corpus reads a matched-surface swap as a small
    nonzero move because its two token rows were estimated from disjoint
    samples. That is an ESTIMATOR GAP, not a consequence, and the way to say so
    is to show it fall with the corpus.
    """
    return [gap(counts_read(n))["surface_only"]["mean"] for n in CORPUS_SWEEP]


# ---------------------------------------------------------------------------
# (d) DIRECTEDNESS, MEASURED
# ---------------------------------------------------------------------------

def t_directed(read, story):
    """T(i, j): how far the read at position j moves under a swap at i.

    THE TWO-INDEX FORM. The read at j is the leave-j-out read, which imposes no
    order at all, so any asymmetry found here is a property of the source and
    not of a mask. The diagonal is zero because the read at i does not look at i.
    """
    v = vocabulary()
    qa = read.q_all(story)
    T = np.zeros((WINDOW, WINDOW))
    for i in range(WINDOW):
        per_pos, _ = _swap_dq(read, story, qa, i, v[i][story[i]].consequence)
        T[i] = per_pos
        T[i, i] = 0.0
    return T


def t_field(read, story):
    """|dq(i) - dq(j)|: the field equation's own form for the source.

    SYMMETRIC AS AN IDENTITY, whatever the bed does, so it cannot carry a
    directed source. Kept and measured because that is the finding that forces
    the two-index form above.
    """
    dq = t_directed(read, story).sum(axis=1) / (WINDOW - 1)
    return np.abs(dq[:, None] - dq[None, :])


def t_causal(read, story):
    """The same swap read through a CAUSAL PREFIX read instead.

    Every entry with j < i is zero because the prefix through j does not contain
    i. That is the mask speaking, not the bed, and backward_block labels it.
    """
    v = vocabulary()
    qc = read.q_causal_all(story)
    T = np.zeros((WINDOW, WINDOW))
    for i in range(WINDOW):
        x2 = np.asarray(story).copy()
        x2[i] = v[i][story[i]].consequence
        T[i] = tv(read.q_causal_all(x2), qc, axis=1)
    return T


@lru_cache(maxsize=8)
def _mean_T(read):
    stories = sample_stories(N_T_STORIES, SEED)
    T = np.zeros((WINDOW, WINDOW))
    for story in stories:
        T += t_directed(read, story)
    return T / len(stories)


@lru_cache(maxsize=8)
def directedness(read):
    """|T(i,j) - T(j,i)| across the window, beside a symmetrised control.

    The control is the same statistic on (T + T^T) / 2, which is exactly zero
    everywhere by construction; it is there so the reported asymmetry has
    something to be distinguishable FROM.
    """
    T = _mean_T(read)
    iu = np.triu_indices(WINDOW, 1)
    d = np.abs(T - T.T)[iu]
    Ts = 0.5 * (T + T.T)
    c = np.abs(Ts - Ts.T)[iu]
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                max=float(d.max()), n_pairs=int(d.size),
                frac_nonzero=float((d > TOL).mean()),
                control_median=float(np.median(c)), control_max=float(c.max()))


#: Under a causal read the backward block is zero because the mask put it there.
#: Reporting that as evidence of directed consequence is circular, and this flag
#: is the module saying so in a place a test can read.
CAUSAL_BACKWARD_IS_TAUTOLOGICAL = True


@lru_cache(maxsize=8)
def backward_block(read):
    """Does consequence flow forward only? Measured both ways, and labelled.

    Under the CAUSAL read the backward entries are zero by the mask. Under the
    BIDIRECTIONAL leave-one-out read nothing forbids them, so whether they are
    nonzero is a measurement -- and on this bed they are, which says consequence
    on a context window is not a forward-only quantity once the mask is removed.
    """
    stories = sample_stories(N_T_STORIES, SEED)
    il = np.tril_indices(WINDOW, -1)
    caus, bidir = [], []
    for story in stories:
        caus.append(t_causal(read, story)[il])
        bidir.append(t_directed(read, story)[il])
    caus = np.concatenate(caus)
    bidir = np.concatenate(bidir)
    return dict(causal_backward_max=float(caus.max()),
                causal_backward_n=int(caus.size),
                bidir_backward_mean=float(bidir.mean()),
                bidir_backward_nonzero=int((bidir > TOL).sum()),
                bidir_backward_n=int(bidir.size),
                causal_is_tautological=CAUSAL_BACKWARD_IS_TAUTOLOGICAL)


# ---------------------------------------------------------------------------
# (e) THE REFUSAL, AND THE 3x2
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def refusal_cases():
    """Every swap the bed can make, labelled by an oracle the criterion never sees.

    THE TRUTH comes from the generator's constraint sets: NULL when the two
    tokens admit the same futures (so there is no consequence to point
    anywhere), UNDEFINED when they admit different futures but the window
    already excludes the difference (so the consequence exists and this window
    cannot see which way it runs), DEFINED otherwise.

    THE CRITERION in classify() sees only read outputs. The two are the same
    object only when the read is exact, and the board shows what happens when it
    is not.
    """
    v = vocabulary()
    M = _mask_table()
    cases = []
    for story in sample_stories(N_CASE_STORIES, SEED + 3):
        for slot in range(WINDOW):
            tok_old = int(story[slot])
            rest = _FULL
            for m in range(WINDOW):
                if m != slot:
                    rest &= int(M[m, story[m]])
            for family in ("consequence", "surface_only"):
                tok_new = getattr(v[slot][tok_old], family)
                s_old, s_new = int(M[slot, tok_old]), int(M[slot, tok_new])
                if s_old == s_new:
                    truth = NULL
                elif (rest & s_old) != (rest & s_new):
                    truth = DEFINED
                else:
                    truth = UNDEFINED
                cases.append(SwapCase(tuple(int(t) for t in story), slot,
                                      tok_old, tok_new, family, truth))
    return tuple(cases)


def classify(read, case):
    """(verdict, value) for one swap, from the READ's outputs alone.

    d_all  how far the read's commitment on the whole window moved.
    d_solo how far it moved with only this slot observed.

    d_all above TOL          -> DEFINED, and the value is the movement.
    d_all zero, d_solo zero  -> NULL: the span carries no consequence at all, so
                                the answer is a VALUE, exactly 0.0.
    d_all zero, d_solo above -> UNDEFINED: the span carries a consequence the
                                window hides, so no direction exists. REFUSED,
                                with the reason attached, never NaN and never a
                                zero standing in for a measurement.
    """
    story = np.asarray(case.story)
    x2 = story.copy()
    x2[case.slot] = case.tok_new
    d_all = float(tv(read.q_full(x2), read.q_full(story)))
    if d_all > TOL:
        return DEFINED, d_all
    d_solo = float(tv(read.solo(case.slot, case.tok_new),
                      read.solo(case.slot, case.tok_old)))
    if d_solo > TOL:
        return UNDEFINED, Refusal(
            MASKED,
            "slot %d: swapping token %d for %d changes what that span admits but "
            "the rest of the window already excludes the difference, so the "
            "consequence has no direction in this context"
            % (case.slot, case.tok_old, case.tok_new))
    return NULL, 0.0


def score_3x2(cases, verdicts):
    """Sensitivity and specificity SEPARATELY. No pooled agreement rate.

    Sensitivity is refusal on the UNDEFINED row. Specificity is answering on the
    DEFINED and NULL rows together, since both must be answered. null_zero
    counts the NULL cases answered with exactly zero, which is the cell a
    total read cannot enter and a pooled rate cannot see.
    """
    tp = fn = tn = fp = null_zero = 0
    for case, (verdict, value) in zip(cases, verdicts):
        refused = is_refusal(value)
        if case.truth == UNDEFINED:
            tp += refused
            fn += not refused
        else:
            tn += not refused
            fp += refused
        if case.truth == NULL and not refused and value == 0.0:
            null_zero += 1
    return dict(sensitivity=100.0 * tp / max(tp + fn, 1),
                specificity=100.0 * tn / max(tn + fp, 1),
                tp=tp, fn=fn, tn=tn, fp=fp, null_zero=null_zero,
                n_undefined=tp + fn, n_answerable=tn + fp)


@lru_cache(maxsize=8)
def _verdicts(read):
    return tuple(classify(read, c) for c in refusal_cases())


@lru_cache(maxsize=1)
def score_board():
    """Every read against the real criterion and both trivial ones."""
    cases = refusal_cases()
    refuse_all = tuple((UNDEFINED, Refusal(MASKED, "refuses everything")) for _ in cases)
    answer_all = tuple((DEFINED, 0.0) for _ in cases)
    rows = []
    for name in READS:
        read = read_by_name(name)
        for label, verdicts in (("criterion (ours)", _verdicts(read)),
                                ("refuse everything", refuse_all),
                                ("answer everything", answer_all)):
            row = dict(read=name, criterion=label)
            row.update(score_3x2(cases, verdicts))
            rows.append(row)
    return rows


#: Tolerances swept for the estimated read. One stated TOL is the published
#: setting; this sweep exists because an estimated read cannot certify that
#: nothing moved at machine epsilon, and the honest question is whether ANY
#: threshold recovers the refusal.
TOL_SWEEP = (1e-12, 1e-6, 1e-4, 1e-2, 1e-1)


@lru_cache(maxsize=1)
def tolerance_sweep():
    """Sensitivity and specificity of the counts read against the threshold."""
    read = counts_read()
    cases = refusal_cases()
    pre = []
    for case in cases:
        story = np.asarray(case.story)
        x2 = story.copy()
        x2[case.slot] = case.tok_new
        d_all = float(tv(read.q_full(x2), read.q_full(story)))
        d_solo = float(tv(read.solo(case.slot, case.tok_new),
                          read.solo(case.slot, case.tok_old)))
        pre.append((d_all, d_solo))
    out = []
    for t in TOL_SWEEP:
        verdicts = []
        for d_all, d_solo in pre:
            if d_all > t:
                verdicts.append((DEFINED, d_all))
            elif d_solo > t:
                verdicts.append((UNDEFINED, Refusal(MASKED, "below threshold")))
            else:
                verdicts.append((NULL, 0.0))
        row = dict(tol=t)
        row.update(score_3x2(cases, verdicts))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# (f) THE SOURCE, WIRED TO THE INSTRUMENT. STATIC ONLY.
# ---------------------------------------------------------------------------

def context_graph(T):
    """An unweighted context graph: an edge wherever the source is nonzero.

    THE RULE IS THE WHOLE RULE -- no threshold to tune, so nothing here can be
    gamed by moving one. Ollivier-Ricci is defined on an undirected graph, so
    the directed source has to be symmetrised to enter it, and what that
    discards is reported beside the reading rather than absorbed into it.
    """
    Ts = 0.5 * (np.asarray(T) + np.asarray(T).T)
    W = (Ts > TOL).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


@lru_cache(maxsize=8)
def context_curvature(read):
    """Static curvature of the context graph, with its edge and refusal counts.

    THE FLOW IS NOT RUN HERE and this function produces no figure about where it
    would settle. Another agent measured that a curvature reading can look
    settled while the underlying geometry is materially wrong unless the
    shortcut census is zero; that agent's figure is not reproduced here because
    no run in this file could bind it. This module computes no such census, so
    it publishes no such figure.
    """
    T = _mean_T(read)
    W = context_graph(T)
    res = curvature_from_graph(W)
    st = group_stats(res["kappa"])
    denom = float(np.abs(T).sum())
    lost = float(np.abs(T - 0.5 * (T + T.T)).sum() / denom) if denom > 0 else 0.0
    return dict(stats=st, n_refused=res["n_refused"], refusals=res["refusals"],
                n_nodes=WINDOW, asymmetry_lost=lost, W=W)


# ---------------------------------------------------------------------------
# THE SELF-CHECKS
# ---------------------------------------------------------------------------

def demo():
    """Every number this module publishes, printed, with its control and count."""
    t0 = time.perf_counter()
    stats = {}

    print("(a) THE BED, synthetic and exact. Generator: %d slots, %d outcomes,"
          % (WINDOW, N_OUTCOMES))
    print("    branch slots %s with outcome pools %s forming a cycle, every other"
          % (list(BRANCH_SLOTS), list(POOLS)))
    print("    slot DECOR. Seed %d. The admitted-future set is the INTERSECTION of"
          % SEED)
    print("    the tokens' masks, so the ground truth of every swap is known by")
    print("    construction. The read is the uniform distribution over that set --")
    print("    a count of futures, not a language model and not a trained probe.")
    oracle = read_by_name("oracle")
    stories = sample_stories(8, SEED)
    sizes = sorted({int(_bits(int(np.bitwise_and.reduce(
        _mask_table()[np.arange(WINDOW), s]))).sum()) for s in stories})
    print("    admitted-set sizes over the first %d stories: %s"
          % (len(stories), sizes))
    stats["admitted_sizes"] = list(sizes)
    assert min(sizes) >= 3, "the bed produced a context with too few futures"
    assert len(sizes) > 1, \
        "every story admits the same count of futures: the branch points are inert"

    print("(b) THE SURFACE IS HELD. Both partners of every branch token sit at the")
    print("    same Hamming distance, checked token by token, not on average.")
    prof = surface_profile()
    stats["hamming_consequence"] = list(prof["consequence"])
    stats["hamming_surface_only"] = list(prof["surface_only"])
    print("    %d branch tokens: consequence partners at Hamming %s, surface-only"
          % (len(prof["consequence"]), sorted(set(prof["consequence"]))))
    print("    partners at Hamming %s -- matched everywhere."
          % sorted(set(prof["surface_only"])))
    assert prof["consequence"] == prof["surface_only"] == [HAMMING] * len(prof["consequence"])

    print("(c) THE THREE SWAPS, side by side, each with its CI and its count.")
    print("    (a) consequence swap  (b) surface-only swap, the planted negative")
    print("    (c) null swap, the span replaced by itself.")
    print("    A MEAN HIDES A HANDFUL OF CORRUPTED SWAPS, so the WORST single swap")
    print("    of each family is carried beside it and bounded on its own.")
    print("    %-8s %-13s %-24s %-9s %s"
          % ("read", "family", "mean  CI", "worst", "swaps"))
    for name in READS:
        g = gap(read_by_name(name))
        for family in ("consequence", "surface_only", "null"):
            st = g[family]
            stats["%s_%s" % (name, family)] = st["mean"]
            stats["%s_%s_n" % (name, family)] = st["n"]
            stats["%s_%s_max" % (name, family)] = st["max"]
            print("    %-8s %-13s %.6f [%.6f, %.6f]  %.6f  %5d"
                  % (name, family, st["mean"], st["ci_lo"], st["ci_hi"],
                     st["max"], st["n"]))
        print("      -> gap: difference %.6f, ratio %s, floor %.6f"
              % (g["difference"], g["ratio_text"], g["null"]["mean"]))
    stats["n_consequence_swaps"] = int(gap(oracle)["consequence"]["n"])
    stats["n_surface_swaps"] = int(gap(oracle)["surface_only"]["n"])
    stats["n_null_swaps"] = int(gap(oracle)["null"]["n"])
    for name in CONSEQUENCE_READS:
        g = gap(read_by_name(name))
        assert g["surface_only"]["mean"] < 0.1 * g["consequence"]["mean"], \
            "%s: the surface-only control is not small: the read sees surface" % name
        assert g["surface_only"]["max"] < 0.25 * g["consequence"]["mean"], \
            ("%s: ONE surface-only swap reached %r against a consequence mean of %r"
             % (name, g["surface_only"]["max"], g["consequence"]["mean"]))
        assert g["null"]["max"] == 0.0, "%s: some null swap is not exactly zero" % name
    assert gap(oracle)["surface_only"]["max"] == 0.0, \
        "the exact read moved on a surface-only swap: not one of them, bitwise"

    print("(d) THE PLANTED NEGATIVE, seen to fire. An ALIKENESS read -- how alike")
    print("    the context is to each outcome, which is what self-attention scores")
    print("    (Vaswani et al., arXiv:1706.03762) -- has no access to consequence.")
    print("    Fed the same matched-surface swaps its gap must collapse, and it does:")
    gd = gap(read_by_name(PLANTED_NEGATIVE_READ))
    go = gap(oracle)
    gc = gap(read_by_name("counts"))
    print("    density ratio %s against oracle %s and counts %s"
          % (gd["ratio_text"], go["ratio_text"], gc["ratio_text"]))
    assert gd["surface_only"]["mean"] > 0.5 * gd["consequence"]["mean"], \
        "the planted negative did not fire"
    print("    FIRED: the surface-only control separates a consequence read from a")
    print("    similarity read, so the gap above is a property of the READ.")

    print("(d2) IS IT JUST INFORMATION DENSITY? The occupied ground, measured.")
    print("    The generator splits the consequence swaps into those that leave the")
    print("    NUMBER of admitted futures identical and those that do not. On the")
    print("    first half a density, a surprisal or an entropy cannot move at all.")
    print("    %-8s %-20s %6s  %-24s %s"
          % ("read", "subset", "swaps", "mean dq  CI", "mean |dH|  max |dH|"))
    for name in READS:
        ec = entropy_control(read_by_name(name))
        for label in ("entropy_preserving", "entropy_moving"):
            st = ec[label]
            stats["%s_%s_dq" % (name, label)] = st["mean"]
            stats["%s_%s_n" % (name, label)] = st["n"]
            stats["%s_%s_dh_max" % (name, label)] = st["dh_max"]
            print("    %-8s %-20s %6d  %.6f [%.6f, %.6f]  %.6f  %r"
                  % (name, label, st["n"], st["mean"], st["ci_lo"], st["ci_hi"],
                     st["dh_mean"], st["dh_max"]))
    eo = entropy_control(oracle)["entropy_preserving"]
    assert eo["dh_max"] == 0.0, \
        "the entropy-preserving subset moves the exact read's entropy: mislabelled"
    assert eo["mean"] > 0.0, "the source is zero where the entropy cannot move"
    print("    THE ENTROPY HALF IS AN IDENTITY, NOT A FINDING: the exact read is")
    print("    uniform, so its entropy is the log of the count and equal counts")
    print("    force %r. THE MEASUREMENT IS THE OTHER HALF: over those %d swaps"
          % (eo["dh_max"], eo["n"]))
    print("    dq is %.6f, nonzero where every entropy, surprisal and density is"
          % eo["mean"])
    print("    pinned at zero. That is what rules out the density reading.")

    print("(e) THE ESTIMATOR GAP. The counts read is ESTIMATED, so its")
    print("    surface-only reading is sampling noise and must fall with the corpus.")
    est = estimator_gap()
    stats["counts_surface_by_n"] = list(est)
    for n, val in zip(CORPUS_SWEEP, est):
        print("    corpus %6d -> surface-only %.6f" % (n, val))
    assert est[-1] < est[0], "the estimator gap did not shrink: it is not an estimator gap"
    assert est[-1] == stats["counts_surface_only"], \
        "the swept read at the pinned corpus is not the published read"

    print("(f) THE FIELD EQUATION'S OWN FORM CANNOT BE DIRECTED. T = |dq(i) - dq(j)|")
    print("    is symmetric as an ALGEBRAIC IDENTITY, so no bed can make it carry a")
    print("    direction. Measured over every read and story:")
    worst = 0.0
    for name in READS:
        r = read_by_name(name)
        for story in sample_stories(5, SEED):
            Tf = t_field(r, story)
            worst = max(worst, float(np.abs(Tf - Tf.T).max()))
    stats["field_form_asymmetry"] = worst
    print("    |T_field - T_field^T| worst = %r. The directed source therefore needs"
          % worst)
    print("    the TWO-INDEX form T(i,j) = movement at j under an intervention at i.")
    assert worst == 0.0

    print("(g) DIRECTEDNESS, MEASURED, against a symmetrised control.")
    a = directedness(oracle)
    for field in ("median", "p90", "max", "n_pairs", "frac_nonzero",
                  "control_median", "control_max"):
        stats["asym_" + field] = a[field]
    print("    |T(i,j)-T(j,i)| over %d pairs: median %.6f  p90 %.6f  max %.6f"
          % (a["n_pairs"], a["median"], a["p90"], a["max"]))
    print("    symmetrised control on the same pairs: median %r  max %r -- zero BY"
          % (a["control_median"], a["control_max"]))
    print("    CONSTRUCTION, since (T + T^T) / 2 is symmetric for any T. It is the")
    print("    scale the asymmetry above is read against, not an independent arm.")
    print("    %.4f of the pairs are asymmetric above tolerance." % a["frac_nonzero"])
    assert a["control_max"] == 0.0 and a["max"] > 0.0

    print("(h) FORWARD ONLY? Both reads, and the causal one labelled.")
    b = backward_block(oracle)
    for field in ("causal_backward_max", "bidir_backward_mean",
                  "bidir_backward_nonzero", "bidir_backward_n"):
        stats[field] = b[field]
    print("    causal prefix read: backward max %r over %d entries -- ZERO BY THE"
          % (b["causal_backward_max"], b["causal_backward_n"]))
    print("    MASK, a tautology, not a finding.")
    print("    bidirectional read: backward mean %.6f, %d of %d entries nonzero --"
          % (b["bidir_backward_mean"], b["bidir_backward_nonzero"],
             b["bidir_backward_n"]))
    print("    so consequence on a context window is NOT forward-only once the mask")
    print("    is removed; the direction lives in the asymmetry, not in the order.")
    assert b["causal_backward_max"] == 0.0

    print("(i) THE REFUSAL, three ways, seen to fire.")
    cases = refusal_cases()
    stats["n_cases"] = len(cases)
    print("    %d swap cases, each labelled by the GENERATOR's constraint sets,"
          % len(cases))
    print("    which the criterion in classify() never touches.")
    shown = {}
    for case, (verdict, value) in zip(cases, _verdicts(oracle)):
        shown.setdefault(case.truth, (verdict, value))
    for truth in (DEFINED, UNDEFINED, NULL):
        verdict, value = shown[truth]
        print("    [%-9s] -> %-9s %s"
              % (truth, verdict,
                 value.reason[:64] + "..." if is_refusal(value) else "value %r" % value))
    assert is_refusal(shown[UNDEFINED][1]) and shown[UNDEFINED][1].code == MASKED
    assert shown[NULL][1] == 0.0 and not is_refusal(shown[NULL][1])
    assert not is_refusal(shown[DEFINED][1])

    print("(j) THE 3x2, sensitivity and specificity SEPARATELY, both trivial")
    print("    criteria scored beside the real one. No pooled agreement rate.")
    board = score_board()
    stats["board"] = [dict(r) for r in board]
    print("    %-8s %-18s %-12s %-12s %s"
          % ("read", "criterion", "sensitivity", "specificity", "tp/fn/tn/fp  null=0"))
    for r in board:
        print("    %-8s %-18s %8.2f%%    %8.2f%%     %d/%d/%d/%d  %d"
              % (r["read"], r["criterion"], r["sensitivity"], r["specificity"],
                 r["tp"], r["fn"], r["tn"], r["fp"], r["null_zero"]))
    ours = {r["read"]: r for r in board if r["criterion"] == "criterion (ours)"}
    assert ours["oracle"]["sensitivity"] == 100.0 and ours["oracle"]["specificity"] == 100.0
    for r in board:
        if r["criterion"] == "refuse everything":
            assert r["sensitivity"] == 100.0 and r["specificity"] == 0.0

    print("(k) THE REPRICE for the estimated read. An estimated read cannot certify")
    print("    that nothing moved, so at the pinned tolerance it answers everything.")
    print("    Whether ANY threshold recovers the refusal is a measurement:")
    sweep = tolerance_sweep()
    stats["tol_sweep"] = [dict(r) for r in sweep]
    for r in sweep:
        print("    tol %.0e -> sensitivity %6.2f%%  specificity %6.2f%%  (tp %d fp %d)"
              % (r["tol"], r["sensitivity"], r["specificity"], r["tp"], r["fp"]))
    same = [r for r in board
            if r["read"] == "counts" and r["criterion"] in
            ("criterion (ours)", "answer everything")]
    assert same[0]["tp"] == same[1]["tp"] and same[0]["fp"] == same[1]["fp"],         "the estimated read's criterion is no longer the answer-everything row"
    print("    At the pinned tolerance the counts row is CELL FOR CELL the")
    print("    answer-everything row: an estimated read cannot certify that nothing")
    print("    moved, so it cannot refuse. That is a property of estimation.")
    best = max(sweep, key=lambda r: r["sensitivity"] + r["specificity"])
    stats["best_tol"] = best["tol"]
    stats["best_tol_sensitivity"] = best["sensitivity"]
    stats["best_tol_specificity"] = best["specificity"]
    print("    best joint threshold %.0e at sensitivity %.2f%% and specificity %.2f%%."
          % (best["tol"], best["sensitivity"], best["specificity"]))
    print("    THAT ROW IS CHOSEN AGAINST THE LABELS. It is an upper bound on what a")
    print("    threshold could do here, not an operating point anyone could have set")
    print("    in advance; the pinned tolerance is the published setting.")

    print("(l) THE SOURCE WIRED TO THE INSTRUMENT, STATIC ONLY. The flow is not run")
    print("    and no figure about where it would settle is produced here.")
    for name in READS:
        c = context_curvature(read_by_name(name))
        if name == "oracle":
            stats["kappa_mean"] = c["stats"]["mean"]
            stats["kappa_ci_lo"] = c["stats"]["ci_lo"]
            stats["kappa_ci_hi"] = c["stats"]["ci_hi"]
            stats["kappa_n_edges"] = c["stats"]["n_edges"]
            stats["kappa_n_refused"] = c["n_refused"]
            stats["asymmetry_lost"] = c["asymmetry_lost"]
        print("    %-8s kappa %.6f CI [%.6f, %.6f]  %3d edges, %d refused, "
              "asymmetry discarded %.4f"
              % (name, c["stats"]["mean"], c["stats"]["ci_lo"], c["stats"]["ci_hi"],
                 c["stats"]["n_edges"], c["n_refused"], c["asymmetry_lost"]))
    assert stats["kappa_n_edges"] > 0

    dt = time.perf_counter() - t0
    stats["elapsed_s"] = dt
    print("(m) elapsed %.2f s against the %.0f s bar." % (dt, BUDGET_S))
    assert dt < BUDGET_S, "demo() took %.1f s" % dt
    print("ALL SELF-CHECKS PASSED")
    return stats


if __name__ == "__main__":
    demo()
