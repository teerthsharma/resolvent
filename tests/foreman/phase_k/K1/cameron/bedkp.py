"""bed_k' candidates and the window families that attack them (Cameron, it.K1).
Generators map a lane label per position to (parent, depth, root) exactly as bed_k.make_bed does (parent = the lane's
previous token; optional depth cap restarts the lane as a new root), then bed_k._pack draws n distinct ids from V.
  geo(M)      bed_k itself: labels i.i.d. uniform over M lanes (geometric positional gaps, mean M).
  flat(M)     renewal lanes: each lane's arrival times have i.i.d. Uniform(0, 2) gaps (first arrival U(0, 2)); all lanes
              merged by time; labels = lane of each of the first n arrivals. Positional gaps ~ flat on [1, 2M].
Families (all chance-credited: unresolved -> most recent root at or before the final pointer):
  recency, doubling(L), window(L, W, k, mode) with mode in {anchored, centered, hpd, local}. anchored/centered = K0
  shortcut.hybrid with the centre at v1 - mu*J (mu = mean gap); hpd = Foreman's highest-density window, here from the
  EMPIRICAL J-hop offset law of the generator (calibration beds); local = window over the token's own W predecessors.
"""
import sys
import numpy as np
sys.dont_write_bytecode = True
from bed_k import _pack, _last_root, V, NULL


def labels_geo(n, M, rng):
    return rng.integers(0, M, n)


def labels_flat(n, M, rng):
    k = int(n / M * 1.3) + 64
    t = rng.uniform(0, 2, (M, k)).cumsum(1)          # arrival times per lane, mean gap 1
    lane = np.repeat(np.arange(M), k)
    order = np.argsort(t.ravel(), kind="stable")[:n]
    assert t.ravel()[order[-1]] < t[:, -1].min()      # no lane ran out of arrivals before the n-th
    return lane[order]


def tree(labels, cap=None):
    n = len(labels)
    parent = -np.ones(n, int); depth = np.zeros(n, int); root = np.arange(n); last = {}
    for i in range(n):
        c = labels[i]
        if c in last and (cap is None or depth[last[c]] < cap):
            p = last[c]; parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


GEN = {"geo": labels_geo, "flat": labels_flat}


def make(gen, n, M, rng, cap=None):
    labels = GEN[gen](n, M, rng)
    return _pack(*tree(labels, cap), rng)


def pointer0(parent):
    A = parent.copy()
    A[A < 0] = np.flatnonzero(parent < 0)
    return A


def doubling(parent, L):
    A = pointer0(parent)
    for _ in range(L):
        A = A[A]
    return A


def offset_law(beds, Jmax):
    """Empirical law of pos(x) - pos(anc_J(x)) over non-saturated x, J = 1..Jmax. Returns list of count arrays."""
    laws = [None] + [np.zeros(1, np.int64) for _ in range(Jmax)]
    for b in beds:
        p, d = b["parent"], b["depth"]
        P0 = pointer0(p); x = np.arange(len(p)); A = x.copy()
        for J in range(1, Jmax + 1):
            A = P0[A]                                  # anc_J (saturates at the root; masked below)
            ok = d >= J
            if not ok.any():
                break
            off = (x - A)[ok]
            c = np.bincount(off)
            if len(c) > len(laws[J]):
                laws[J] = np.pad(laws[J], (0, len(c) - len(laws[J])))
            laws[J][:len(c)] += c
    return laws


def hpd_table(laws, W, mu):
    """start a >= 1 maximizing P(a <= offset < a + W | J); beyond the law's support: window centred on mu*J."""
    tab = {}
    for J in range(1, len(laws)):
        f = laws[J]
        if f.sum() < 200:
            continue
        c = np.concatenate([[0], np.cumsum(f)])
        if len(c) <= W + 1:
            tab[J] = 1
            continue
        mass = c[W:] - c[:-W]
        mass[0] = -1
        tab[J] = int(np.argmax(mass))
    return tab


def window(parent, depth, L, W, k, mode, mu, hpd=None):
    A = pointer0(parent)
    idx = np.arange(len(parent))
    for _ in range(L):
        v1 = A
        cur = A[v1]
        J = depth - depth[v1]
        if mode == "anchored":
            lo, hi = v1 - W, v1
        elif mode == "centered":
            c = (v1 - mu * J).astype(int)
            lo, hi = np.maximum(c - W // 2, 0), np.minimum(c + W // 2, v1)
        elif mode == "local":
            lo, hi = idx - W, idx
        else:  # hpd: window [v1 - (a + W - 1), v1 - a], a from the empirical J-hop law
            a = np.array([hpd.get(int(j), max(1, int(mu * j) - W // 2)) for j in range(J.max() + 1)])[J]
            lo, hi = v1 - (a + W - 1), v1 - a + 1
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
        assert (A <= idx).all()
    return A


def credited(parent, root, A):
    return _last_root(parent)[A] == root


def recency(parent, root):
    return _last_root(parent) == root
