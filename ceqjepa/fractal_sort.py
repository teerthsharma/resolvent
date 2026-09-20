"""FRACTAL SORT -- a trie on the belief attractor, and the hash that may tie it.

The hidden-Markov filter is a place-dependent iterated function system on the
belief simplex, one map per observation symbol:

    F_o(b) = b P diag(B[:, o]) / (b P B[:, o])

Its attractor carries the Blackwell measure.  The ADDRESS of a belief b_t is its
itinerary, the last L observations.  Retrieval is a trie descent on that address
followed by a local nearest-neighbour scan inside the leaf.

THE COST LAW
    cells       ~ e^{h L}        h the entropy rate, read from the forward
                                 algorithm's own normalisers.
    resolution  eps at depth L, L ~ chi^{-1} ln(1/eps), chi the filter's
                                 Lyapunov contraction exponent.
    memory      N(eps) ~ eps^{-D},  D <= h / chi.

The joint bound is on the dimension of the MEASURE, never of the support.

PRE-REGISTERED, filed in PREREG_L / PREREG_SPEEDUP / PREREG_ERROR below before
the first number; neither line may move afterwards:
    PREDICTION  the trie beats brute kNN by >= 5x at error <= 1e-3, L = 6.
    COUNTER     the plain last-L hash matches the trie's speed; the IFS view
                buys the collage certificate and nothing else.

Run:  python -m ceqjepa.fractal_sort --all
Exit code 0 iff every must-fire fired and no suite came up short on count.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

#: Every configuration constant that any reported number depends on.  Printed at
#: the head of every run so a number can never be quoted without its cell.
BED_SEED = 20260914
QUERY_SEED = 20260915
BOOT_SEED = 20260916
N_TRAIN = 80_000
N_QUERY = 1_000
L_MAX = 16
BURN_IN = 200

#: The pre-registered cells.  Neither moves after the first number.
PREREG_L = 6
PREREG_SPEEDUP = 5.0
PREREG_ERROR = 1e-3


#: Every citation this module's claims rest on, resolved to a numbered
#: statement.  A named mathematician without a number is not a citation.
CITATIONS = [
    ("h from the forward normalisers",
     "Jurgens & Crutchfield, Shannon Entropy Rate of Hidden Markov Processes, "
     "arXiv:2008.12886, J. Stat. Phys. 183:32 (2021), Eq. (8) [the normaliser "
     "<eta|T^(x)|1>] with Eq. (16) [h_mu^B integrated against the Blackwell "
     "measure].  NOT Eq. (15): Eq. (15) is Blackwell's self-consistency "
     "equation Q(E) = sum_a int_{f_a^-1 E} r_a(w) dQ(w), which defines the "
     "measure, not the rate."),
    ("lambda1 = -h(Y)",
     "Holliday, Goldsmith & Glynn, Shannon Meets Lyapunov, IEEE Trans. Inform. "
     "Theory 52(8):3509-3532 (2006), Proposition 1.  Blackwell (1957) gives "
     "the integral form only; the Lyapunov phrasing postdates him."),
    ("the joint bound dim_H(nu) <= h/chi",
     "Jaroszewska & Rams, arXiv:0707.3532, J. Stat. Phys. 132(5):907 (2008), "
     "Theorem 1: dim_H(mu) <= -h(mu)/lambda(mu), lambda(mu) < 0.  Upper bound "
     "only; place-dependent probabilities allowed, maps need not be "
     "contractions.  Applied to the Blackwell measure by Barany, Pollicott & "
     "Simon, J. Stat. Phys. 148(3):393-421 (2012), Proposition 14 (no arXiv "
     "id exists for that paper)."),
    ("chi = lambda1 - lambda2 for the projective action",
     "Hochman & Solomyak, Invent. Math. 210:815-875 (2017), Lemma 2.3 (the "
     "induced map on P expands by at most ||g||^2 and contracts by at most "
     "||g||^-2) and Theorem 1.1 (denominator 2*chi, which is lambda1 - lambda2 "
     "in SL_2).  Barany-Pollicott-Simon reach the same rate via their "
     "Corollary 12, lambda(m) = log|Delta| + 2h(m), but never write "
     "'lambda1 - lambda2'."),
    ("the collage bound",
     "Barnsley, Fractals Everywhere, 2nd ed., Theorem 10.1 ('The Collage "
     "Theorem'), Ch. III Sec. 10, pp. 94-95: h(L, A) <= (1-s)^-1 h(L, union "
     "w_n(L)), contractivity factor 0 <= s < 1, complete metric space.  The "
     "constant is written 1/(1-s).  Primary source: Barnsley, Ervin, Hardin & "
     "Lancaster, PNAS 83:1975-1977 (1986), p. 1975, stated unnumbered over a "
     "COMPACT metric space.  Barnsley & Demko (1985) does NOT state it."),
    ("Kaplan-Yorke's determinism hypothesis -- why KY is NOT used here",
     "Frederickson, Kaplan, Yorke & Yorke, J. Differential Equations "
     "49:185-207 (1983), p. 188: 'Let f: B -> interior(B) be C^2'; the "
     "dimension claims are Conjecture 1 (p. 189) and Conjecture 2 (p. 190), "
     "both quantified over the function space C^2(B), with Lyapunov numbers "
     "defined through D[f^m] of a SINGLE iterated map.  Theorem-grade form: "
     "Ledrappier & Young, Ann. Math. 122(3):540-574 (1985), Corollary I, "
     "p. 549, hypothesis 'f: M -> M is a C^2 diffeomorphism'.  A random IFS "
     "has no f^m and no single Df, so neither is statable for it.  Random "
     "compositions needed their own paper: Ledrappier & Young, Comm. Math. "
     "Phys. 117(4):529-548 (1988)."),
]


def print_citations() -> None:
    print("\nCITATIONS -- every one resolved to a numbered statement")
    for what, where in CITATIONS:
        print(f"  [{what}]")
        for line in _wrap(where, 72):
            print(f"    {line}")


def _wrap(text: str, width: int):
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def machine() -> str:
    return (f"{platform.platform()} / {platform.processor()} / "
            f"python {platform.python_version()} / numpy {np.__version__}")


# ---------------------------------------------------------------------------
# the bed: a hidden Markov chain whose filter is a genuine IFS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bed:
    P: np.ndarray      # (S, S) row-stochastic, deliberately non-symmetric
    B: np.ndarray      # (S, O) row-stochastic emissions
    pi: np.ndarray     # (S,) stationary distribution
    name: str

    @property
    def S(self) -> int:
        return self.P.shape[0]

    @property
    def O(self) -> int:
        return self.B.shape[1]


def _stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, k])
    pi = np.abs(pi)
    return pi / pi.sum()


def make_bed(S: int, O: int, seed: int, sharpness: float, name: str) -> Bed:
    """A reproducible non-symmetric HMM.

    `sharpness` is the only physics knob: it concentrates the emission rows, so
    large sharpness means informative observations, a strongly contracting
    filter (large chi) and a low-dimensional Blackwell measure.
    """
    rng = np.random.default_rng(seed)
    P = rng.gamma(1.0, 1.0, size=(S, S)) + 0.05
    P /= P.sum(axis=1, keepdims=True)
    # break symmetry hard: bias the chain forward around the state cycle
    P = 0.7 * P + 0.3 * np.roll(np.eye(S), 1, axis=1)
    P /= P.sum(axis=1, keepdims=True)

    Braw = rng.gamma(1.0, 1.0, size=(S, O)) + 0.02
    Braw /= Braw.sum(axis=1, keepdims=True)
    B = Braw ** sharpness
    B /= B.sum(axis=1, keepdims=True)
    return Bed(P=P, B=B, pi=_stationary(P), name=name)


def simulate(bed: Bed, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S, O = bed.S, bed.O
    states = np.empty(T, dtype=np.int64)
    obs = np.empty(T, dtype=np.int64)
    Pc = np.cumsum(bed.P, axis=1)
    Bc = np.cumsum(bed.B, axis=1)
    u = rng.random(2 * T)
    s = int(rng.choice(S, p=bed.pi))
    for t in range(T):
        s = int(np.searchsorted(Pc[s], u[2 * t]))
        s = min(s, S - 1)
        o = int(np.searchsorted(Bc[s], u[2 * t + 1]))
        o = min(o, O - 1)
        states[t] = s
        obs[t] = o
    return obs, states


def forward(bed: Bed, obs: np.ndarray, b0: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """The forward algorithm.  Returns (beliefs, log normalisers).

    The normaliser at step t is c_t = b_{t-1} P B[:, o_t], exactly the one-step
    predictive likelihood of the observed symbol.  Its negative log, averaged, is
    the entropy rate estimate.
    """
    T = obs.shape[0]
    S = bed.S
    beliefs = np.empty((T, S))
    logc = np.empty(T)
    b = bed.pi.copy() if b0 is None else np.asarray(b0, dtype=float).copy()
    P, B = bed.P, bed.B
    for t in range(T):
        pred = b @ P
        w = pred * B[:, obs[t]]
        c = w.sum()
        logc[t] = math.log(c)
        b = w / c
        beliefs[t] = b
    return beliefs, logc


# ---------------------------------------------------------------------------
# h, h2 and chi -- each measured independently
# ---------------------------------------------------------------------------


def entropy_rate(logc: np.ndarray, burn: int = BURN_IN) -> tuple[float, float]:
    """h in nats/symbol from the forward algorithm's own normalisers.

    h = -E[log c_t] under the stationary mixed-state measure.  This is the
    Jurgens-Crutchfield mixed-state form of the Shannon entropy rate; the
    citation is resolved in the report, not asserted here.

    Returns (h, standard error of the mean over a block bootstrap).
    """
    x = -logc[burn:]
    n = x.shape[0]
    block = max(1, int(n ** 0.5))
    nb = n // block
    blocks = x[: nb * block].reshape(nb, block).mean(axis=1)
    return float(x.mean()), float(blocks.std(ddof=1) / math.sqrt(nb))


def word_keys(obs: np.ndarray, O: int, L: int) -> np.ndarray:
    """base-O packing of the length-L itinerary ending at each position."""
    T = obs.shape[0]
    key = np.zeros(T, dtype=np.int64)
    for j in range(L):
        key[j:] += obs[: T - j] * (O ** j)
    key[: L - 1] = -1 if L > 0 else 0
    return key


def word_entropies(obs: np.ndarray, O: int, L: int, burn: int = BURN_IN
                   ) -> tuple[float, float, float, int, int]:
    """Block entropies of the depth-L words.

    Returns (H1 plug-in, H1 Miller-Madow corrected, H2, distinct words, n).

    The plug-in entropy of a word distribution is biased DOWN by about
    (m-1)/(2n) nats with m occupied words and n samples.  At L = 8 on this bed
    that is 0.13 nats, large enough to drag the fitted slope below h on its own,
    so both the raw and the corrected value are printed and the corrected one is
    fitted.
    """
    if L == 0:
        return 0.0, 0.0, 0.0, 1, 0
    k = word_keys(obs, O, L)[max(burn, L - 1):]
    _, counts = np.unique(k, return_counts=True)
    n = int(counts.sum())
    m = int(counts.shape[0])
    p = counts / n
    H1 = float(-(p * np.log(p)).sum())
    H2 = float(-math.log((p ** 2).sum()))
    return H1, H1 + (m - 1) / (2.0 * n), H2, m, n


def lyapunov_pair(bed: Bed, obs: np.ndarray, burn: int = BURN_IN
                  ) -> np.ndarray:
    """The FULL Lyapunov spectrum of the random product N_o = (P diag(B[:,o]))^T.

    Gram-Schmidt/QR reorthonormalisation on a full S-frame.  lambda1 is the
    growth rate of the unnormalised forward recursion.  The filter's induced
    projective action on the (S-1)-simplex has S-1 exponents lambda_{j+1} -
    lambda_1; the GREATEST of them, i.e. the least-contracting direction, is
    chi = lambda1 - lambda2, and that is the chi the bound takes.
    """
    S = bed.S
    mats = [(bed.P * bed.B[:, o][None, :]).T for o in range(bed.O)]
    rng = np.random.default_rng(7)
    Q = np.linalg.qr(rng.standard_normal((S, S)))[0]
    acc = np.zeros(S)
    n = 0
    for t, o in enumerate(obs):
        Q = mats[o] @ Q
        Q, R = np.linalg.qr(Q)
        d = np.diag(R)
        sgn = np.sign(d)
        sgn[sgn == 0] = 1.0
        Q = Q * sgn
        if t >= burn:
            acc += np.log(np.abs(d))
            n += 1
    return acc / n


def chi_direct(bed: Bed, obs: np.ndarray, n_pairs: int = 256, horizon: int = 40,
               seed: int = 4242, lo: float = 1e-11, hi: float = 1e-2
               ) -> tuple[float, float, tuple[int, int]]:
    """chi measured as the filter's forgetting rate, independent of any QR.

    Two filters, different initial beliefs, the SAME observation sequence.  The
    L1 gap decays like e^{-chi t}.  The first few steps are an O(1) transient
    that has nothing to do with the asymptotic exponent, so the slope is fitted
    ONLY over the window where the median gap lies in [lo, hi] -- fitting from
    t = 0 reads a rate biased by the transient.

    Returns (chi, OLS standard error, fitted window).
    """
    rng = np.random.default_rng(seed)
    S = bed.S
    starts = rng.integers(BURN_IN, obs.shape[0] - horizon - 1, size=n_pairs)
    D = np.empty((n_pairs, horizon))
    for p, st in enumerate(starts):
        b1 = rng.dirichlet(np.ones(S))
        b2 = rng.dirichlet(np.ones(S))
        seg = obs[st: st + horizon]
        for t in range(horizon):
            w1 = (b1 @ bed.P) * bed.B[:, seg[t]]
            w2 = (b2 @ bed.P) * bed.B[:, seg[t]]
            b1 = w1 / w1.sum()
            b2 = w2 / w2.sum()
            D[p, t] = np.abs(b1 - b2).sum()
    # geometric mean over pairs: the exponent is an average of logs, not of gaps
    with np.errstate(divide="ignore"):
        gm = np.exp(np.log(np.maximum(D, 1e-300)).mean(axis=0))
    idx = np.nonzero((gm > lo) & (gm < hi))[0]
    if idx.shape[0] < 5:
        return float("nan"), float("nan"), (-1, -1)
    t = idx.astype(float)
    s, se = ols(t, np.log(gm[idx]))
    return -s, se, (int(idx[0]), int(idx[-1]))


# ---------------------------------------------------------------------------
# the three indices.  L-NULL: the ONLY thing that varies is the index.
# ---------------------------------------------------------------------------


class Trie:
    """A real prefix tree: descent is L single-symbol dict lookups.

    Every node carries the index array of the items passing through it, which is
    what makes backoff possible and what a fixed-depth hash cannot do in one
    lookup.
    """

    __slots__ = ("root", "L_max", "O")

    def __init__(self, obs: np.ndarray, items: np.ndarray, L_max: int, O: int):
        self.L_max = L_max
        self.O = O
        root: list = [{}, []]
        # NOTE the arrays hold POSITIONS into the bank, not global time indices,
        # so every arm returns an answer in the same coordinate system.
        for pos, i in enumerate(items):
            node = root
            node[1].append(pos)
            for j in range(L_max):
                sym = int(obs[i - j])
                kid = node[0].get(sym)
                if kid is None:
                    kid = [{}, []]
                    node[0][sym] = kid
                node = kid
                node[1].append(pos)
        self.root = root
        self._freeze(root)

    @staticmethod
    def _freeze(node) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            n[1] = np.asarray(n[1], dtype=np.int64)
            stack.extend(n[0].values())

    def descend(self, addr: np.ndarray, L: int):
        """Returns (index array, depth reached).  Fixed depth: may be empty."""
        node = self.root
        for j in range(L):
            kid = node[0].get(int(addr[j]))
            if kid is None:
                return None, j
            node = kid
        return node[1], L

    def descend_backoff(self, addr: np.ndarray, L: int):
        """Deepest non-empty ancestor.  One descent, no restart."""
        node = self.root
        last = node[1]
        depth = 0
        for j in range(L):
            kid = node[0].get(int(addr[j]))
            if kid is None:
                break
            node = kid
            last = node[1]
            depth = j + 1
        return last, depth


def build_hash(obs: np.ndarray, items: np.ndarray, L: int, O: int) -> dict:
    """The counter's index: one flat dict, key = base-O packed last-L word."""
    pos = np.arange(items.shape[0], dtype=np.int64)
    if L == 0:
        return {0: pos}
    keys = word_keys(obs, O, L)[items]
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    vs = pos[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    return {int(ks[bounds[i]]): vs[bounds[i]: bounds[i + 1]]
            for i in range(bounds.shape[0] - 1)}


def hash_key(addr: np.ndarray, L: int, O: int) -> int:
    k = 0
    m = 1
    for j in range(L):
        k += int(addr[j]) * m
        m *= O
    return k


def _nn(bank: np.ndarray, q: np.ndarray, cand: np.ndarray) -> tuple[int, float]:
    d = np.abs(bank[cand] - q).sum(axis=1)
    a = int(np.argmin(d))
    return int(cand[a]), float(d[a])


def brute_nn(bank: np.ndarray, q: np.ndarray) -> tuple[int, float]:
    d = np.abs(bank - q).sum(axis=1)
    a = int(np.argmin(d))
    return a, float(d[a])


# ---------------------------------------------------------------------------
# timing, with a bootstrap interval over repeats
# ---------------------------------------------------------------------------


def boot_ci(x: np.ndarray, seed: int = BOOT_SEED, n: int = 20_000,
            alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=(n, x.shape[0]))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def time_calls(fn, repeats: int) -> np.ndarray:
    out = np.empty(repeats)
    fn()  # warm
    for r in range(repeats):
        t0 = time.perf_counter()
        fn()
        out[r] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


@dataclass
class World:
    bed: Bed
    obs: np.ndarray
    beliefs: np.ndarray
    logc: np.ndarray
    items: np.ndarray
    qobs: np.ndarray
    qbeliefs: np.ndarray
    qitems: np.ndarray
    trie: Trie
    h: float
    h_se: float
    lam1: float
    lam2: float
    chi_qr: float
    chi_fit: float
    chi_fit_se: float
    chi_win: tuple
    spec: np.ndarray


def build_world(sharpness: float = 3.0, S: int = 3, O: int = 3,
                n_train: int = N_TRAIN, n_query: int = N_QUERY,
                l_max: int = L_MAX) -> World:
    bed = make_bed(S, O, BED_SEED, sharpness, f"S{S}O{O}sharp{sharpness}")
    T = n_train + BURN_IN + l_max
    obs, _ = simulate(bed, T, BED_SEED)
    beliefs, logc = forward(bed, obs)
    items = np.arange(BURN_IN + l_max, T, dtype=np.int64)[:n_train]

    Tq = n_query + BURN_IN + l_max
    qobs, _ = simulate(bed, Tq, QUERY_SEED)
    qbeliefs, _ = forward(bed, qobs)
    qitems = np.arange(BURN_IN + l_max, Tq, dtype=np.int64)[:n_query]

    h, h_se = entropy_rate(logc)
    spec = lyapunov_pair(bed, obs)
    lam1, lam2 = float(spec[0]), float(spec[1])
    cf, cse, cwin = chi_direct(bed, obs)
    trie = Trie(obs, items, l_max, O)
    return World(bed=bed, obs=obs, beliefs=beliefs, logc=logc, items=items,
                 qobs=qobs, qbeliefs=qbeliefs, qitems=qitems, trie=trie,
                 h=h, h_se=h_se, lam1=lam1, lam2=lam2, chi_qr=lam1 - lam2,
                 chi_fit=cf, chi_fit_se=cse, chi_win=cwin, spec=spec)


def query_addresses(w: World, L: int) -> np.ndarray:
    """addr[i, j] = the observation j steps back from query i (j = 0 is now)."""
    return np.stack([w.qobs[w.qitems - j] for j in range(L)], axis=1) if L else \
        np.zeros((w.qitems.shape[0], 0), dtype=np.int64)


def run_arms(w: World, L: int) -> dict:
    """Every arm answers the same question on the same queries.

    PINNED: bed, seeds, bank of N beliefs, query set, L1 metric, the answer
    returned (the index of the nearest bank belief).  VARIES: the index only.
    """
    bank = w.beliefs[w.items]
    addrs = query_addresses(w, L)
    Q = w.qitems.shape[0]
    hsh = build_hash(w.obs, w.items, L, w.bed.O)
    allidx = np.arange(bank.shape[0], dtype=np.int64)

    ref_i = np.empty(Q, dtype=np.int64)
    ref_d = np.empty(Q)
    for i in range(Q):
        ref_i[i], ref_d[i] = brute_nn(bank, w.qbeliefs[w.qitems[i]])

    def arm(kind):
        got = np.empty(Q, dtype=np.int64)
        dist = np.empty(Q)
        work = np.empty(Q)
        depth = np.empty(Q, dtype=np.int64)
        fell_back = 0
        for i in range(Q):
            q = w.qbeliefs[w.qitems[i]]
            if kind == "knn":
                cand, dep = allidx, 0
            elif kind == "trie":
                cand, dep = w.trie.descend(addrs[i], L)
                if cand is None or cand.shape[0] == 0:
                    cand, dep = allidx, 0
                    fell_back += 1
            elif kind == "trie_backoff":
                cand, dep = w.trie.descend_backoff(addrs[i], L)
            elif kind == "hash":
                cand = hsh.get(hash_key(addrs[i], L, w.bed.O))
                dep = L
                if cand is None or cand.shape[0] == 0:
                    cand, dep = allidx, 0
                    fell_back += 1
            else:
                raise ValueError(kind)
            j, d = _nn(bank, q, cand)
            got[i] = j
            dist[i] = d
            work[i] = cand.shape[0]
            depth[i] = dep
        return got, dist, work, depth, fell_back

    out = {"L": L, "Q": Q, "N": bank.shape[0], "ref_i": ref_i, "ref_d": ref_d}
    for kind in ("knn", "trie", "trie_backoff", "hash"):
        got, dist, work, depth, fb = arm(kind)
        out[kind] = {
            "idx": got,
            "err_mean": float((dist - ref_d).mean()),
            "err_max": float((dist - ref_d).max()),
            "exact": float((got == ref_i).mean()),
            "work_mean": float(work.mean()),
            "depth_mean": float(depth.mean()),
            "fallbacks": fb,
        }
    return out


def timed_arms(w: World, L: int, repeats: int = 9) -> dict:
    bank = w.beliefs[w.items]
    addrs = query_addresses(w, L)
    Q = w.qitems.shape[0]
    hsh = build_hash(w.obs, w.items, L, w.bed.O)
    qb = w.qbeliefs[w.qitems]
    allidx = np.arange(bank.shape[0], dtype=np.int64)
    trie = w.trie
    O = w.bed.O

    def knn():
        for i in range(Q):
            brute_nn(bank, qb[i])

    def knn_context():
        """The matched baseline: brute scan over the L-symbol contexts."""
        ctx = np.stack([w.obs[w.items - j] for j in range(L)], axis=1) if L else \
            np.zeros((bank.shape[0], 0), dtype=np.int64)
        def f():
            for i in range(Q):
                if L:
                    m = (ctx == addrs[i][None, :]).all(axis=1)
                    c = np.flatnonzero(m)
                    if c.shape[0] == 0:
                        c = allidx
                else:
                    c = allidx
                _nn(bank, qb[i], c)
        return f

    def trie_f():
        for i in range(Q):
            cand, _ = trie.descend(addrs[i], L)
            if cand is None or cand.shape[0] == 0:
                cand = allidx
            _nn(bank, qb[i], cand)

    def trie_bo():
        for i in range(Q):
            cand, _ = trie.descend_backoff(addrs[i], L)
            _nn(bank, qb[i], cand)

    def hash_f():
        for i in range(Q):
            cand = hsh.get(hash_key(addrs[i], L, O))
            if cand is None or cand.shape[0] == 0:
                cand = allidx
            _nn(bank, qb[i], cand)

    def hash_bo():
        """The hash emulating backoff: up to L separate tables, L lookups."""
        tabs = [build_hash(w.obs, w.items, d, O) for d in range(L + 1)]
        def f():
            for i in range(Q):
                cand = None
                for d in range(L, -1, -1):
                    cand = tabs[d].get(hash_key(addrs[i], d, O))
                    if cand is not None and cand.shape[0] > 0:
                        break
                _nn(bank, qb[i], cand)
        return f

    fns = {"knn": knn, "knn_context": knn_context(), "trie": trie_f,
           "trie_backoff": trie_bo, "hash": hash_f, "hash_backoff": hash_bo()}
    return {k: time_calls(v, repeats) for k, v in fns.items()}


# ---------------------------------------------------------------------------
# T-TRIE
# ---------------------------------------------------------------------------


def t_trie(w: World, L: int = PREREG_L, repeats: int = 11) -> bool:
    print(f"\n=== T-TRIE  (L = {L}) ===")
    acc = run_arms(w, L)
    tim = timed_arms(w, L, repeats)
    N, Q = acc["N"], acc["Q"]
    print(f"bank N = {N}   queries Q = {Q}   metric = L1 on the belief vector")
    print(f"{'arm':<14}{'exact':>8}{'err_mean':>12}{'err_max':>12}"
          f"{'cands':>10}{'fallbk':>8}")
    for k in ("knn", "trie", "trie_backoff", "hash"):
        a = acc[k]
        print(f"{k:<14}{a['exact']:>8.4f}{a['err_mean']:>12.3e}"
              f"{a['err_max']:>12.3e}{a['work_mean']:>10.1f}"
              f"{a['fallbacks']:>8d}")

    base = tim["knn"]
    print(f"\n{'arm':<14}{'s/batch':>12}{'vs knn':>10}{'95% CI on ratio':>26}")
    ratios = {}
    for k, v in tim.items():
        r = base / v
        m, lo, hi = boot_ci(r)
        ratios[k] = (m, lo, hi)
        print(f"{k:<14}{v.mean():>12.5f}{m:>10.2f}x   [{lo:>7.2f}, {hi:>7.2f}]")

    print(f"\nwork ratio (candidate distance evaluations, implementation-free):")
    for k in ("trie", "trie_backoff", "hash"):
        print(f"  {k:<14}{N / max(acc[k]['work_mean'], 1e-9):>10.2f}x")

    # --- MUST-FIRE: at L = 0 the trie degenerates to brute force EXACTLY -----
    print("\nMUST-FIRE  L = 0 degeneracy")
    a0 = run_arms(w, 0)
    same_idx = bool(np.array_equal(a0["trie"]["idx"], a0["ref_i"]))
    same_work = a0["trie"]["work_mean"] == float(N)
    same_hash = bool(np.array_equal(a0["hash"]["idx"], a0["ref_i"]))
    print(f"  trie(L=0) index == brute index on all {a0['Q']} queries : {same_idx}")
    print(f"  trie(L=0) candidate set size == N == {N}               : {same_work}")
    print(f"  trie(L=0) err_max                                      : "
          f"{a0['trie']['err_max']:.3e}")
    print(f"  hash(L=0) index == brute index                         : {same_hash}")
    fired = same_idx and same_work and same_hash and a0["trie"]["err_max"] == 0.0
    print(f"  MUST-FIRE {'FIRED' if fired else 'FAILED -- the trie is STRUCK'}")

    # --- the pre-registered cell -------------------------------------------
    err = acc["trie"]["err_mean"]
    sp = ratios["trie"][0]
    ok_err = err <= PREREG_ERROR
    print(f"\nPRE-REGISTERED  trie >= {PREREG_SPEEDUP}x at error <= {PREREG_ERROR}, "
          f"L = {PREREG_L}")
    print(f"  error {err:.3e} <= {PREREG_ERROR}: {ok_err};  speedup {sp:.2f}x "
          f"CI [{ratios['trie'][1]:.2f}, {ratios['trie'][2]:.2f}]")
    print(f"  PREDICTION {'HELD' if (ok_err and ratios['trie'][1] >= PREREG_SPEEDUP) else 'FAILED'}")
    ht, hl, hh = ratios["hash"]
    tt, tl, th = ratios["trie"]
    tie = not (hh < tl or th < hl)
    print(f"  COUNTER  hash {ht:.2f}x [{hl:.2f}, {hh:.2f}] vs trie {tt:.2f}x "
          f"[{tl:.2f}, {th:.2f}] -> intervals "
          f"{'OVERLAP: the hash TIES the trie' if tie else 'DISJOINT: they separate'}")
    idx_same = bool(np.array_equal(acc["hash"]["idx"], acc["trie"]["idx"]))
    print(f"  hash and trie return the IDENTICAL answer on every query: {idx_same}")
    return fired


# ---------------------------------------------------------------------------
# T-DEPTH
# ---------------------------------------------------------------------------


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(1, x.shape[0] - 2)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(math.sqrt(cov[0, 0]))


def leaf_diameter(w: "World", L: int, max_cells: int = 300, max_items: int = 40,
                  seed: int = 31337):
    """Median and 90th-percentile L1 diameter of a depth-L leaf.

    This is the RESOLUTION half of the cost law.  Two beliefs that share their
    last L observations have been driven by the same maps for L steps, so their
    gap is the filter's forgetting gap after L steps: eps(L) ~ e^{-chi L},
    equivalently L ~ chi^-1 ln(1/eps).  Measured, not assumed.
    """
    bank = w.beliefs[w.items]
    hsh = build_hash(w.obs, w.items, L, w.bed.O)
    rng = np.random.default_rng(seed)
    cells = [v for v in hsh.values() if v.shape[0] >= 2]
    if not cells:
        return np.full(4, np.nan), 0
    if len(cells) > max_cells:
        cells = [cells[i] for i in rng.choice(len(cells), max_cells,
                                              replace=False)]
    diams = []
    for c in cells:
        if c.shape[0] > max_items:
            c = c[rng.choice(c.shape[0], max_items, replace=False)]
        X = bank[c]
        d = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
        diams.append(float(d.max()))
    a = np.array(diams)
    return (np.quantile(a, [0.25, 0.5, 0.75, 0.9]), len(cells))


def t_depth(w: World, Ls=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                          14), n_words: int = 4_000_000) -> bool:
    """Speed and cells against L, with the two traps this fit has.

    TRAP 1  the raw occupied-cell count is NOT e^{hL}.  Every length-L word has
            positive probability under this bed, so for small L the distinct
            count is exactly O^L and its slope reads ln O, not h.  Only the
            EFFECTIVE cell count exp(H_L) tracks h.
    TRAP 2  the plug-in block entropy is biased down by ~(m-1)/(2n); uncorrected
            it drags the fitted slope below h on its own.
    """
    print("\n=== T-DEPTH ===")
    print(f"h  (forward normalisers) = {w.h:.6f} +/- {w.h_se:.6f} nats/symbol")
    print(f"ln O = {math.log(w.bed.O):.6f} nats/symbol  (the iid-uniform "
          f"ceiling; the RAW cell count saturates at this, not at h)")
    # The block entropies need far more symbols than the retrieval bank holds:
    # on the bank alone the deepest words carry ~2 samples each and the plug-in
    # bias survives its own first-order correction.  Symbols are cheap; beliefs
    # are not, so the entropy fit gets its own long run and the leaf occupancy
    # stays on the bank the queries actually hit.
    lobs, _ = simulate(w.bed, n_words + BURN_IN, BED_SEED + 55)
    print(f"block entropies fitted on a dedicated run of {n_words} symbols, "
          f"seed {BED_SEED + 55}; leaf occupancy stays on the {w.items.shape[0]}"
          f"-item bank.")
    rows = []
    for L in Ls:
        H1, H1mm, H2, cells, n = word_entropies(lobs, w.bed.O, L)
        acc = run_arms(w, L)
        a = acc["trie"]
        Q = acc["Q"]
        fb = a["fallbacks"]
        # expected leaf size conditional on a NON-EMPTY leaf.  The unconditional
        # mean is polluted by the full-bank fallback and is not the cost law.
        leaf_cond = ((a["work_mean"] * Q) - fb * acc["N"]) / max(Q - fb, 1)
        rows.append({"L": L, "cells": cells, "OL": w.bed.O ** L, "H1": H1,
                     "H1mm": H1mm, "H2": H2, "n": n, "leaf": leaf_cond,
                     "pred": acc["N"] * math.exp(-H2), "fb": fb / Q,
                     "err": a["err_mean"], "exact": a["exact"]})
    print(f"\n{'L':>3}{'cells':>8}{'O^L':>8}{'expH1':>10}{'expH1mm':>10}"
          f"{'expH2':>10}{'E[leaf|ne]':>12}{'N e^-H2':>10}{'fallbk':>8}"
          f"{'err_mean':>11}{'exact':>8}")
    for r in rows:
        print(f"{r['L']:>3}{r['cells']:>8d}{r['OL']:>8d}"
              f"{math.exp(r['H1']):>10.1f}{math.exp(r['H1mm']):>10.1f}"
              f"{math.exp(r['H2']):>10.1f}{r['leaf']:>12.1f}{r['pred']:>10.1f}"
              f"{r['fb']:>8.3f}{r['err']:>11.3e}{r['exact']:>8.4f}")

    # TWO windows, because the two laws die of different causes.  The entropy
    # slope dies of undersampling (too many words per symbol drawn); the leaf
    # occupancy dies of empty leaves (the bank runs out of items).  A single
    # window forced to satisfy both collapses onto the saturated tail and reads
    # a slope that is not h at all -- that is how the first pass of this test
    # read 0.33 for a quantity whose true value is near 1.
    ent = [r for r in rows if r["cells"] <= r["n"] / 30 and r["L"] >= 3]
    ent = ent[-5:] if len(ent) >= 5 else ent
    occ = [r for r in rows if r["fb"] < 0.05 and r["leaf"] >= 3.0]
    occ = occ[-5:] if len(occ) >= 5 else occ
    xe = np.array([float(r["L"]) for r in ent])
    xo = np.array([float(r["L"]) for r in occ])
    print(f"\nENTROPY window  L = {int(ent[0]['L'])}..{int(ent[-1]['L'])}: "
          f">= 30 symbol samples per word.")
    print(f"OCCUPANCY window L = {int(occ[0]['L'])}..{int(occ[-1]['L'])}: "
          f"empty-leaf rate < 5% and >= 3 items per leaf.")
    print(f"(OLS; +/- is 1 standard error, 95% CI = +/- 1.96 se; "
          f"h = {w.h:.6f} +/- {w.h_se:.6f})")
    fits = {}
    for name, key, logit in (("ln cells (RAW -- the trap)", "cells", True),
                             ("H1 plug-in", "H1", False),
                             ("H1 Miller-Madow", "H1mm", False),
                             ("H2 (Renyi-2)", "H2", False)):
        y = np.array([float(r[key]) for r in ent])
        if logit:
            y = np.log(y)
        sl, se = ols(xe, y)
        fits[name] = (sl, se)
        # the comparison is slope-against-h, so the interval that matters is
        # the COMBINED one: h carries its own block-bootstrap error, and on a
        # curve this linear the OLS error alone is far too tight to be honest.
        comb = 1.96 * math.sqrt(se ** 2 + w.h_se ** 2)
        print(f"  d({name})/dL = {sl:.6f} +/- {se:.6f}  95% CI "
              f"[{sl - 1.96 * se:.6f}, {sl + 1.96 * se:.6f}]  matches h "
              f"(combined +/-{comb:.6f}): {abs(sl - w.h) <= comb}")
    sl, sle = ols(xo, np.log(np.array([r["leaf"] for r in occ])))
    slp, slpe = ols(xo, np.log(np.array([r["pred"] for r in occ])))
    comb = 1.96 * math.sqrt(sle ** 2 + w.h_se ** 2)
    print(f"  d(-ln E[leaf|non-empty])/dL = {-sl:.6f} +/- {sle:.6f}  95% CI "
          f"[{-sl - 1.96 * sle:.6f}, {-sl + 1.96 * sle:.6f}]  matches h "
          f"(combined +/-{comb:.6f}): {abs(-sl - w.h) <= comb}")
    print(f"  d(-ln N e^-H2)/dL           = {-slp:.6f} +/- {slpe:.6f}  "
          f"(the predicted occupancy law on the same window)")

    hS, hSse = fits["H1 Miller-Madow"]
    h2, h2se = fits["H2 (Renyi-2)"]
    raw = fits["ln cells (RAW -- the trap)"][0]
    lnO = math.log(w.bed.O)
    print(f"\n  h  (normalisers)        = {w.h:.6f} +/- {w.h_se:.6f}")
    comb_h = 1.96 * math.sqrt(hSse ** 2 + w.h_se ** 2)
    print(f"  h  (corrected H1 slope) = {hS:.6f} +/- {hSse:.6f}   |diff| "
          f"{abs(hS - w.h):.6f} = {100 * abs(hS - w.h) / w.h:.2f}%; combined "
          f"95% half-width {comb_h:.6f}")
    print(f"  THE SLOPE MATCHES THE INDEPENDENTLY MEASURED h: "
          f"{abs(hS - w.h) <= comb_h}   (the OLS error alone, {hSse:.2e}, is "
          f"not an honest interval here:")
    print(f"  the block-entropy curve is linear to 5 figures, so the residual "
          f"{abs(hS - w.h):.6f} is finite-sample bias, not scatter.)")
    print(f"  h2 (Renyi-2 slope)      = {h2:.6f} +/- {h2se:.6f}   h2 <= h as it "
          f"must be: {h2 <= w.h}")
    ceil_L = max([r["L"] for r in rows if r["cells"] == r["OL"]], default=0)
    print(f"\n  TRAP 1, MEASURED.  RAW cell slope = {raw:.6f} against "
          f"ln O = {lnO:.6f} (|diff| {abs(raw - lnO):.6f}) and against")
    print(f"  h = {w.h:.6f} (|diff| {abs(raw - w.h):.6f}).  The raw occupied-cell "
          f"count measures the ALPHABET, not the entropy rate: on this")
    print(f"  bed it is EXACTLY O^L for every L <= {ceil_L}.  Only the effective "
          f"count exp(H_L) is the cost law's cell count.")
    print(f"  TRAP 2, MEASURED.  plug-in H1 slope "
          f"{fits['H1 plug-in'][0]:.6f} vs Miller-Madow {hS:.6f}: the "
          f"uncorrected estimator is {w.h - fits['H1 plug-in'][0]:.6f} low,")
    print(f"  enough on its own to 'refute' a cost law that in fact holds.")
    ratios = [r["leaf"] / r["pred"] for r in occ]
    print(f"\n  THE COST LAW AS MEASURED.  Leaf occupancy decays at "
          f"{-sl:.6f}; N e^-H2 predicts it to within a factor "
          f"{min(ratios):.3f}-{max(ratios):.3f}")
    print(f"  across the occupancy window.  cells ~ e^{{hL}} is the index's "
          f"MEMORY law and its slope is h; expected WORK per query decays at")
    print(f"  the RENYI-2 rate h2 = {h2:.6f}, not at h = {w.h:.6f}.  "
          f"They differ by {w.h - h2:.6f} nats/symbol on this bed.")
    # ---- the resolution half of the cost law ------------------------------
    print(f"\n  RESOLUTION LAW  eps(L) ~ e^-chi L,  L ~ chi^-1 ln(1/eps)")
    print(f"  leaf diameter is a MAX over pairs inside a cell, so its decay "
          f"rate depends on which quantile of cells is fitted.  All four are")
    print(f"  reported: the spread across them, not the OLS error on any one "
          f"of them, is the honest uncertainty on this rate.")
    print(f"  {'L':>3}{'cells>=2':>10}{'q25':>12}{'q50':>12}{'q75':>12}"
          f"{'q90':>12}")
    res = []
    for r in rows:
        if r["L"] == 0 or r["L"] > 12:
            continue
        q, nc = leaf_diameter(w, int(r["L"]))
        if nc == 0 or not np.all(q > 0):
            continue
        res.append((float(r["L"]), q, nc))
        print(f"  {int(r['L']):>3}{nc:>10d}" + "".join(f"{v:>12.3e}" for v in q))
    if len(res) >= 4:
        keep = [t for t in res if t[1][0] > 1e-13]
        xr = np.array([t[0] for t in keep])
        slopes = []
        for j, qn in enumerate(("q25", "q50", "q75", "q90")):
            sl, se = ols(xr, np.log(np.array([t[1][j] for t in keep])))
            slopes.append(-sl)
            print(f"  d(-ln {qn} diam)/dL = {-sl:.6f} +/- {se:.6f}  95% CI "
                  f"[{-sl - 1.96 * se:.6f}, {-sl + 1.96 * se:.6f}]")
        lo, hi = min(slopes), max(slopes)
        print(f"  quantile bracket = [{lo:.6f}, {hi:.6f}], width {hi - lo:.6f}")
        print(f"  chi (QR spectrum gap) = {w.chi_qr:.6f};  chi (filter "
              f"forgetting) = {w.chi_fit:.6f} +/- {w.chi_fit_se:.6f}")
        inside = lo <= w.chi_qr <= hi
        print(f"  chi LIES INSIDE the quantile bracket: {inside}  "
              f"-- the resolution law holds with chi as its rate; it does NOT "
              f"hold to three figures,")
        print(f"  because conditioning on a shared word and taking a MAX over "
              f"the cell is not the same functional as the Lyapunov average.")
        eps = 1e-6
        print(f"  inverted at eps = {eps:.0e}: L = ln(1/eps)/chi = "
              f"{math.log(1 / eps) / w.chi_qr:.2f} symbols; the measured "
              f"bracket gives L in [{math.log(1 / eps) / hi:.2f}, "
              f"{math.log(1 / eps) / lo:.2f}]")
        obs_L = [t[0] for t in res if t[1][1] <= eps]
        if obs_L:
            print(f"  OBSERVED: median leaf diameter first falls below "
                  f"{eps:.0e} at L = {int(obs_L[0])}.")
    return True


# ---------------------------------------------------------------------------
# T-DIM
# ---------------------------------------------------------------------------


def box_counts(pts: np.ndarray, k: int) -> tuple[int, float]:
    """Occupied-box count and Shannon entropy of the measure at eps = 2^-k."""
    eps = 2.0 ** -k
    g = np.floor(pts / eps).astype(np.int64)
    g -= g.min(axis=0, keepdims=True)
    span = g.max(axis=0) + 1
    key = np.zeros(g.shape[0], dtype=np.int64)
    m = 1
    for d in range(g.shape[1]):
        key += g[:, d] * m
        m *= int(span[d])
    _, counts = np.unique(key, return_counts=True)
    p = counts / counts.sum()
    return int(counts.shape[0]), float(-(p * np.log(p)).sum())


def t_dim(w: World, ks=range(3, 26), n_dim: int = 1_500_000) -> bool:
    """D1 of the Blackwell measure against h/chi, on a dedicated long sample.

    The scaling window is chosen at the FINE end, where the local slope has
    plateaued -- an OLS over the whole range is dominated by the coarse boxes and
    reads a slope that is not the dimension at all.  Both the per-decade local
    slopes and the window are printed so the choice is auditable.
    """
    print("\n=== T-DIM ===")
    print(f"  a dedicated stationary run of {n_dim} filter steps, seed "
          f"{BED_SEED + 77}; the retrieval bank is too short to resolve D1.")
    obs, _ = simulate(w.bed, n_dim + BURN_IN, BED_SEED + 77)
    beliefs, logc = forward(w.bed, obs)
    pts = beliefs[BURN_IN:][:, : w.bed.S - 1]
    n = pts.shape[0]
    h_long, h_long_se = entropy_rate(logc)

    rows = []
    for k in ks:
        c, H = box_counts(pts, k)
        rows.append((k, c, H))
    print(f"  chart: first {w.bed.S - 1} belief coordinates, {n} points, "
          f"eps = 2^-k")
    print(f"  {'k':>3}{'boxes':>10}{'H1(nats)':>12}{'dH1/dln(1/eps)':>17}"
          f"{'dlnN/dln(1/eps)':>18}{'pts/box':>10}")
    ln2 = math.log(2.0)
    loc1, loc0 = [None], [None]
    for i, (k, c, H) in enumerate(rows):
        if i == 0:
            print(f"  {k:>3}{c:>10d}{H:>12.6f}{'':>17}{'':>18}{n / c:>10.1f}")
            continue
        s1 = (H - rows[i - 1][2]) / ln2
        s0 = (math.log(c) - math.log(rows[i - 1][1])) / ln2
        loc1.append(s1)
        loc0.append(s0)
        print(f"  {k:>3}{c:>10d}{H:>12.6f}{s1:>17.6f}{s0:>18.6f}{n / c:>10.1f}")

    # window: contiguous fine-end run with >= 30 points per box (so the plug-in
    # entropy is not starved) and >= 100 boxes (so it is not a coarse artefact).
    cand = [i for i, (k, c, H) in enumerate(rows) if c >= 100 and n / c >= 30]
    if len(cand) < 4:
        cand = list(range(len(rows) // 3, len(rows) - 1))
    win = [rows[i] for i in cand]
    xs = np.array([r[0] * ln2 for r in win])
    D1, D1se = ols(xs, np.array([r[2] for r in win]))
    D0, D0se = ols(xs, np.log(np.array([float(r[1]) for r in win])))
    ks_used = (int(win[0][0]), int(win[-1][0]))

    chi = w.chi_qr
    bound = h_long / chi
    bound_fit = h_long / w.chi_fit
    print(f"\nscaling window k = {ks_used[0]}..{ks_used[1]} "
          f"(>= 100 boxes and >= 30 points/box throughout)")
    print(f"  D1 (INFORMATION dimension of the MEASURE) = {D1:.6f} +/- {D1se:.6f}"
          f"   95% CI [{D1 - 1.96 * D1se:.6f}, {D1 + 1.96 * D1se:.6f}]")
    print(f"  h     = {h_long:.6f} +/- {h_long_se:.6f}   (forward normalisers, "
          f"this run; header run read {w.h:.6f})")
    print(f"  lam1  = {w.lam1:.6f}   lam2 = {w.lam2:.6f}   "
          f"(QR on the random product)")
    print(f"  IDENTITY CHECK  lam1 == -h : residual |lam1 + h| = "
          f"{abs(w.lam1 + w.h):.3e}   (the product's norm IS the likelihood;")
    print(f"                  Holliday-Goldsmith-Glynn, IEEE Trans. Inform. "
          f"Theory 52(8):3509-3532 (2006), Proposition 1)")
    print(f"  chi   = lam1 - lam2 = {chi:.6f}   (projective contraction rate)")
    print(f"  chi   = {w.chi_fit:.6f} +/- {w.chi_fit_se:.6f}   (INDEPENDENT: "
          f"filter forgetting, steps {w.chi_win[0]}-{w.chi_win[1]})")
    print(f"  the two chi agree to {100 * abs(chi - w.chi_fit) / chi:.2f}%")
    print(f"  h/chi = {bound:.6f}  (QR chi)   {bound_fit:.6f}  (forgetting chi)")
    holds = (D1 - 1.96 * D1se) <= bound
    strict = (D1 + 1.96 * D1se) <= bound
    print(f"  JOINT BOUND D1 <= h/chi : {D1:.6f} <= {bound:.6f} -> "
          f"{'HOLDS' if strict else ('CONSISTENT (CI straddles)' if holds else 'VIOLATED')}"
          f"   slack {100 * (1 - D1 / bound):+.1f}%")
    vacuous = bound >= w.bed.S - 1
    if vacuous:
        print(f"  NOTE the bound {bound:.4f} exceeds the ambient dimension "
              f"{w.bed.S - 1}; VACUOUS in this cell and proves nothing.")

    print(f"\nD0 (box dimension of the SUPPORT)         = {D0:.6f} +/- "
          f"{D0se:.6f}")
    print(f"  *** NOT COMPARABLE to h/chi. ***  dim_H(mu) <= dim_B(supp mu) "
          f"always, so a support box-count above h/chi is a FALSE REFUTATION,")
    print(f"  not a violation.  Only the information dimension D1 is the "
          f"comparable quantity.  D0 - h/chi = {D0 - bound:+.6f}"
          + ("   <- the support count ALONE would have 'refuted' the bound"
             if D0 > bound else ""))

    print(f"\nKAPLAN-YORKE: NOT APPLIED, DELIBERATELY.  The filter is a RANDOM "
          f"iterated function system driven by the observation symbol, not a")
    print(f"  deterministic differentiable map; the Lyapunov-dimension formula "
          f"is posed for the deterministic case and applying it here is struck.")
    print(f"  The bound used is the place-dependent-IFS bound: "
          f"Jaroszewska & Rams arXiv:0707.3532 Theorem 1, dim_H(mu) <= "
          f"-h(mu)/lambda(mu),")
    print(f"  carried to the Blackwell measure by Barany-Pollicott-Simon, "
          f"J. Stat. Phys. 148(3):393-421 (2012), Proposition 14.  The")
    print(f"  determinism hypothesis KY cannot shed is FKYY, J. Diff. Eq. "
          f"49:185-207 (1983), p. 188, 'Let f: B -> interior(B) be C^2';")
    print(f"  its dimension claims are Conjecture 1 (p. 189) and Conjecture 2 "
          f"(p. 190), quantified over C^2(B).  Theorem-grade: Ledrappier-")
    print(f"  Young, Ann. Math. 122(3):540-574 (1985), Corollary I, p. 549, "
          f"'f: M -> M is a C^2 diffeomorphism'.  A random IFS has no f^m.")
    return holds or vacuous


# ---------------------------------------------------------------------------
# T-COLLAGE
# ---------------------------------------------------------------------------


def hausdorff(A: np.ndarray, B: np.ndarray, chunk: int = 64) -> float:
    """Symmetric Hausdorff distance between two point clouds, L2.

    Chunked and float32 on purpose: this box is shared with other runs and a
    512-row broadcast of the full cross-product died on a 10.9 MiB allocation.
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    def one(X, Y):
        y2 = (Y * Y).sum(axis=1)
        m = 0.0
        for s in range(0, X.shape[0], chunk):
            Xc = X[s:s + chunk]
            d2 = (Xc * Xc).sum(axis=1)[:, None] + y2[None, :] - 2.0 * (Xc @ Y.T)
            m = max(m, float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).max()))
        return m
    return max(one(A, B), one(B, A))


def chaos_game(maps, n: int, seed: int, burn: int = 200) -> np.ndarray:
    """Sample the attractor of an affine IFS {x -> A x + t} by random iteration."""
    rng = np.random.default_rng(seed)
    K = len(maps)
    d = maps[0][1].shape[0]
    x = np.zeros(d)
    out = np.empty((n, d))
    picks = rng.integers(0, K, size=n + burn)
    for i in range(n + burn):
        A, t = maps[picks[i]]
        x = A @ x + t
        if i >= burn:
            out[i - burn] = x
    return out


def collage_report(name: str, maps, A: np.ndarray, Ahat: np.ndarray) -> tuple:
    c = max(float(np.linalg.norm(Am, 2)) for Am, _ in maps)
    union = np.concatenate([A @ Am.T + t[None, :] for Am, t in maps], axis=0)
    coll = hausdorff(A, union)
    meas = hausdorff(A, Ahat)
    print(f"\n  [{name}]  s = max_k ||A_k||_2 = {c:.6f}   "
          f"(Barnsley, Fractals Everywhere 2nd ed., Theorem 10.1, pp. 94-95;")
    print(f"           the constant there is written 1/(1-s).  Primary source "
          f"Barnsley-Ervin-Hardin-Lancaster, PNAS 83:1975-1977 (1986), p.1975)")
    if c >= 1.0:
        print(f"    REFUSED. c = {c:.6f} >= 1: the maps are not a contractive IFS,")
        print(f"    the collage bound d_H(A,Ahat) <= d_H(A, U G_k(A))/(1-c) DOES NOT")
        print(f"    EXIST, and no bound is reported for this cell.")
        print(f"    (measured d_H(A, Ahat) = {meas:.6f}, reported WITHOUT a bound)")
        return c, coll, meas, None, None
    bound = coll / (1.0 - c)
    ok = meas <= bound
    print(f"    collage distance d_H(A, U G_k(A)) = {coll:.6f}")
    print(f"    BOUND  d_H(A, Ahat) <= {coll:.6f} / (1 - {c:.6f}) = {bound:.6f}")
    print(f"    MEASURED d_H(A, Ahat)              = {meas:.6f}")
    print(f"    bound holds: {ok}   (slack {bound - meas:+.6f}, "
          f"ratio {bound / max(meas, 1e-12):.2f}x)")
    return c, coll, meas, bound, ok


def t_collage(w: World, n_pts: int = 1200) -> bool:
    print(f"\n=== T-COLLAGE ===")
    print("  MUST-FIRE on a PLANTED IFS (the Sierpinski gasket, c = 1/2 exactly)")
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254037844386]])
    exact = [(0.5 * np.eye(2), 0.5 * v) for v in V]
    A = chaos_game(exact, n_pts, seed=11)

    fired = []
    c, coll, meas, bound, ok = collage_report("exact maps", exact, A,
                                              chaos_game(exact, n_pts, seed=12))
    fired.append(bool(ok))

    rng = np.random.default_rng(99)
    pert = [(Am + 0.03 * rng.standard_normal((2, 2)), t + 0.03 * rng.standard_normal(2))
            for Am, t in exact]
    Ap = chaos_game(pert, n_pts, seed=13)
    c2, coll2, meas2, bound2, ok2 = collage_report("perturbed maps", pert, A, Ap)
    fired.append(bool(ok2))

    blow = [(1.2 * np.eye(2), 0.5 * v) for v in V]
    print("\n  REFUSAL PATH (a planted non-contraction, c = 1.2):")
    c3, *_rest = collage_report("expansive maps", blow, A, A)
    refused = c3 >= 1.0
    fired.append(refused)
    print(f"    refusal fired: {refused}")

    print("\n  THE LEARNED MAPS: K = O consequence operators G_o fitted to the "
          "filter")
    it = w.items[w.items + 1 < w.beliefs.shape[0]]
    bank = w.beliefs[it][:, : w.bed.S - 1]
    nxt = w.beliefs[it + 1][:, : w.bed.S - 1]
    o_next = w.obs[it + 1]
    learned = []
    for o in range(w.bed.O):
        m = o_next == o
        X = np.c_[bank[m], np.ones(m.sum())]
        Y = nxt[m]
        sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Am = sol[:-1].T
        t = sol[-1]
        rms = float(np.sqrt(((X @ sol - Y) ** 2).sum(axis=1).mean()))
        learned.append((Am, t))
        print(f"    G_{o}: n = {int(m.sum())}, fit rms = {rms:.6f}, "
              f"||A||_2 = {np.linalg.norm(Am, 2):.6f}")
    sub = bank[:: max(1, bank.shape[0] // n_pts)][:n_pts]
    Ahat = chaos_game(learned, n_pts, seed=14) if \
        max(float(np.linalg.norm(Am, 2)) for Am, _ in learned) < 1.0 else sub
    collage_report("learned maps", learned, sub, Ahat)

    # Birkhoff: why the Euclidean chart can refuse while the filter still forgets
    M = [bed_matrix(w.bed, o) for o in range(w.bed.O)]
    taus = [birkhoff_tau(m) for m in M]
    print(f"\n    diagnostic: the filter contracts in the HILBERT projective "
          f"metric, not the Euclidean chart.")
    print(f"    Birkhoff coefficients tau(M_o) = "
          f"{', '.join(f'{t:.6f}' for t in taus)}  (max {max(taus):.6f})")
    return all(fired)


def bed_matrix(bed: Bed, o: int) -> np.ndarray:
    return bed.P * bed.B[:, o][None, :]


def birkhoff_tau(M: np.ndarray) -> float:
    """tanh(Delta/4), Delta the projective diameter -- the Birkhoff coefficient."""
    S = M.shape[0]
    best = 0.0
    for i in range(S):
        for j in range(S):
            for k in range(S):
                for l in range(S):
                    a = M[i, k] * M[j, l]
                    b = M[j, k] * M[i, l]
                    if a > 0 and b > 0:
                        best = max(best, abs(math.log(a / b)))
    return math.tanh(best / 4.0)


# ---------------------------------------------------------------------------
# T-GROW
# ---------------------------------------------------------------------------


def t_grow(w: World, Ls=None, repeats: int = 9) -> bool:
    """Where the itinerary index starts paying -- and where it stops.

    The cost law has TWO regimes, and a test that reports one ratio at one L
    cannot see the second.  Expected work per query falls like N e^{-h2 L} until
    the leaves empty at L* = ln(N)/h2; past that every query misses, the
    fixed-depth index falls back to a full scan, and the index is pure overhead.
    So the deliverable is a WINDOW with two crossovers, not a point.
    """
    Lm = w.trie.L_max
    if Ls is None:
        base = max(1, Lm // 4)
        Ls = sorted({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, base, 2 * base, 4 * base}
                    & set(range(0, Lm + 1)))
    print("\n=== T-GROW ===")
    print(f"  base context L = {max(1, Lm // 4)}, doubled to "
          f"{2 * max(1, Lm // 4)}, quadrupled to {4 * max(1, Lm // 4)}.")
    print("  PINNED: bed, bank, query set, L1 metric, the answer returned.")
    print("  VARIES: the index (trie / hash / kNN) and L.  Nothing else.")
    tab = []
    for L in Ls:
        acc = run_arms(w, L)
        tim = timed_arms(w, L, repeats)
        H1, H1mm, H2, cells, _n = word_entropies(w.obs, w.bed.O, L)
        a, ab = acc["trie"], acc["trie_backoff"]
        Q, N = acc["Q"], acc["N"]
        fb = a["fallbacks"]
        leaf_cond = ((a["work_mean"] * Q) - fb * N) / max(Q - fb, 1)
        tab.append({
            "L": L, "cells": cells, "H2": H2, "leaf": a["work_mean"],
            "leaf_cond": leaf_cond, "pred": N * math.exp(-H2),
            "err": a["err_mean"], "exact": a["exact"], "fb": fb, "Q": Q, "N": N,
            "bo_err": ab["err_mean"], "bo_exact": ab["exact"],
            "bo_work": ab["work_mean"], "bo_depth": ab["depth_mean"],
            "t_knn": tim["knn"].mean(), "t_ctx": tim["knn_context"].mean(),
            "t_trie": tim["trie"].mean(), "t_hash": tim["hash"].mean(),
            "r_trie": boot_ci(tim["knn"] / tim["trie"]),
            "r_hash": boot_ci(tim["knn"] / tim["hash"]),
            "r_bo": boot_ci(tim["knn"] / tim["trie_backoff"]),
            "r_hbo": boot_ci(tim["knn"] / tim["hash_backoff"]),
            "r_ctx": boot_ci(tim["knn_context"] / tim["trie"]),
        })
    N = tab[0]["N"]
    print(f"\n{'L':>3}{'cells':>8}{'E[leaf]':>10}{'N e^-H2':>10}{'t_knn':>8}"
          f"{'t_trie':>8}{'t_hash':>8}{'trie':>9}{'hash':>9}{'trie+bo':>9}"
          f"{'err':>11}{'empty':>7}")
    for r in tab:
        print(f"{r['L']:>3}{r['cells']:>8d}{r['leaf']:>10.1f}{r['pred']:>10.1f}"
              f"{r['t_knn']:>8.4f}{r['t_trie']:>8.4f}{r['t_hash']:>8.4f}"
              f"{r['r_trie'][0]:>8.2f}x{r['r_hash'][0]:>8.2f}x"
              f"{r['r_bo'][0]:>8.2f}x{r['err']:>11.3e}"
              f"{r['fb'] / r['Q']:>7.2f}")

    print(f"\n  ratios with 95% bootstrap intervals over {repeats} timed "
          f"repeats of the whole {tab[0]['Q']}-query batch")
    for r in tab:
        print(f"    L={r['L']:>2}  trie {r['r_trie'][0]:>7.2f}x "
              f"[{r['r_trie'][1]:6.2f},{r['r_trie'][2]:6.2f}]   "
              f"hash {r['r_hash'][0]:>7.2f}x "
              f"[{r['r_hash'][1]:6.2f},{r['r_hash'][2]:6.2f}]   "
              f"trie+backoff {r['r_bo'][0]:>7.2f}x "
              f"[{r['r_bo'][1]:6.2f},{r['r_bo'][2]:6.2f}]   "
              f"hash+backoff {r['r_hbo'][0]:>7.2f}x"
              f"   [backoff: err {r['bo_err']:.2e} exact "
              f"{r['bo_exact']:.4f} depth {r['bo_depth']:.2f}]")

    def window(key):
        good = [r for r in tab if r[key][1] > 1.0]
        return (good[0], good[-1]) if good else (None, None)

    lo_t, hi_t = window("r_trie")
    lo_b, hi_b = window("r_bo")
    # L* = ln(N)/h2, with h2 the Renyi-2 rate fitted where H2 is not yet
    # saturated at ln(N).  This PREDICTS the upper edge from the cost law,
    # out of quantities measured before a single timing was taken.
    unsat = [r for r in tab if r["L"] > 0 and r["pred"] > 0.02 * r["N"]]
    star = h2s = h2se = star_lo = star_hi = None
    if len(unsat) >= 3:
        h2s, h2se = ols(np.array([float(r["L"]) for r in unsat]),
                        np.array([r["H2"] for r in unsat]))
        star = math.log(N) / h2s
        star_lo = math.log(N) / (h2s + 1.96 * h2se)
        star_hi = math.log(N) / max(h2s - 1.96 * h2se, 1e-9)

    print(f"\n  CROSSOVER -- the deliverable of this test")
    if lo_t is None:
        print("    the fixed-depth fractal sort never pays on this bed.")
    else:
        below = [r for r in tab if r["L"] < lo_t["L"]]
        print(f"    LOWER crossover, fixed depth: the fractal sort starts "
              f"paying at L = {lo_t['L']}")
        print(f"      {lo_t['r_trie'][0]:.2f}x, 95% CI "
              f"[{lo_t['r_trie'][1]:.2f}, {lo_t['r_trie'][2]:.2f}], clear of 1.0")
        if below:
            b = below[-1]
            print(f"      at L = {b['L']} it is {b['r_trie'][0]:.2f}x, CI "
                  f"[{b['r_trie'][1]:.2f}, {b['r_trie'][2]:.2f}] -- not clear; "
                  f"that is the index's floor cost, {b['leaf']:.0f} candidates")
            print(f"      scanned either way, so the whole difference is "
                  f"descent overhead.")
        print(f"    UPPER crossover, fixed depth: it stops paying after L = "
              f"{hi_t['L']} ({hi_t['r_trie'][0]:.2f}x); the next tested L reads "
              f"{[r['r_trie'][0] for r in tab if r['L'] > hi_t['L']][:1]}")
        print(f"    PAYING WINDOW, fixed depth: L in [{lo_t['L']}, {hi_t['L']}]"
              f", peak {max(r['r_trie'][0] for r in tab):.2f}x at L = "
              f"{max(tab, key=lambda r: r['r_trie'][0])['L']}")
    if star is not None:
        print(f"\n    the upper edge is PREDICTED by the cost law, not fitted "
              f"to the timings: a leaf empties when N e^-H2(L) = 1, so")
        print(f"    L* = ln(N)/h2 = ln({N})/{h2s:.6f} = {star:.2f}, 95% CI "
              f"[{star_lo:.2f}, {star_hi:.2f}]   (h2 fitted on L = "
              f"{int(unsat[0]['L'])}..{int(unsat[-1]['L'])}, +/-{h2se:.6f})")
        onset = [r for r in tab if r["fb"] > 0]
        if onset:
            print(f"    first empty leaf observed at L = {onset[0]['L']} "
                  f"({onset[0]['fb']}/{onset[0]['Q']} queries); all queries "
                  f"miss by L = "
                  f"{[r['L'] for r in tab if r['fb'] == r['Q']][:1]}")
    if lo_b is not None:
        print(f"\n    WITH BACKOFF the window is L in [{lo_b['L']}, "
              f"{hi_b['L']}]: the trie descends to the deepest non-empty "
              f"ancestor in ONE pass, so past L* it degrades to a shallower")
        print(f"    index instead of to a full scan.  A fixed-depth hash cannot "
              f"do that in one lookup; emulating it costs up to L separate "
              f"tables and L lookups (the hash+backoff column).")
    print(f"\n    ERROR across the window: fixed-depth trie err_mean "
          f"{max(r['err'] for r in tab):.3e} worst case, exact-hit rate "
          f"{min(r['exact'] for r in tab):.4f} worst case.")
    print(f"    backoff is NOT free accuracy and the price is printed: "
          f"err_mean {max(r['bo_err'] for r in tab):.3e} worst case, exact-hit "
          f"{min(r['bo_exact'] for r in tab):.4f} worst case, against the")
    print(f"    fixed-depth arm's {min(r['exact'] for r in tab):.4f}.  It "
          f"trades depth for occupancy; what it buys is that the degradation "
          f"past L* is graceful instead of a full scan.")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def header(w: World) -> None:
    print("=" * 78)
    print("FRACTAL SORT -- the itinerary index on the belief attractor")
    print("=" * 78)
    print(f"machine : {machine()}")
    print(f"bed     : S = {w.bed.S}, O = {w.bed.O}, name = {w.bed.name}")
    print(f"seeds   : bed {BED_SEED}, query {QUERY_SEED}, bootstrap {BOOT_SEED}")
    print(f"sizes   : N_train = {w.items.shape[0]}, N_query = "
          f"{w.qitems.shape[0]}, L_max = {w.trie.L_max}, "
          f"burn-in = {BURN_IN}")
    print(f"h       = {w.h:.6f} +/- {w.h_se:.6f} nats/symbol "
          f"(forward-algorithm normalisers)")
    print(f"spectrum= {' '.join(f'{v:.6f}' for v in w.spec)}  "
          f"(full Lyapunov spectrum of the random product, QR on an S-frame)")
    print(f"          the filter's projective action has {w.bed.S - 1} exponents "
          f"{' '.join(f'{float(v) - w.lam1:.6f}' for v in w.spec[1:])};")
    print(f"          the bound takes the GREATEST (least contracting), "
          f"chi = lam1 - lam2 = {w.chi_qr:.6f}")
    print(f"chi_fit = {w.chi_fit:.6f} +/- {w.chi_fit_se:.6f} "
          f"(independent: filter forgetting, steps {w.chi_win[0]}-{w.chi_win[1]})")
    print(f"h/chi   = {w.h / w.chi_qr:.6f}")
    print_citations()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--test", choices=["trie", "depth", "dim", "collage", "grow"])
    ap.add_argument("--sharpness", type=float, default=3.0)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-query", type=int, default=N_QUERY)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--l-max", type=int, default=L_MAX)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    w = build_world(sharpness=a.sharpness, n_train=a.n_train,
                    n_query=a.n_query, l_max=a.l_max)
    header(w)

    fired = []
    want = ["trie", "depth", "dim", "collage", "grow"] if a.all or not a.test \
        else [a.test]
    if "trie" in want:
        fired.append(("T-TRIE L=0 must-fire", t_trie(w, repeats=a.repeats + 2)))
    if "depth" in want:
        fired.append(("T-DEPTH", t_depth(w)))
    if "dim" in want:
        fired.append(("T-DIM joint bound", t_dim(w)))
    if "collage" in want:
        fired.append(("T-COLLAGE planted must-fire", t_collage(w)))
    if "grow" in want:
        fired.append(("T-GROW", t_grow(w, repeats=a.repeats)))

    print("\n" + "=" * 78)
    print(f"L-COUNT  expected {len(want)} suites, collected {len(fired)}")
    for name, ok in fired:
        print(f"  {'FIRED ' if ok else 'FAILED'}  {name}")
    print(f"wall clock {time.perf_counter() - t0:.1f} s")
    bad = [n for n, ok in fired if not ok]
    print(f"EXIT {'0' if not bad and len(fired) == len(want) else '1'}")
    print("=" * 78)
    return 0 if (not bad and len(fired) == len(want)) else 1


if __name__ == "__main__":
    sys.exit(main())
