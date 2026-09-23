# Foreman K1: generic attack kit for a replacement bed. Works on any generator returning bed_k's dict
# (parent, depth, root; ids/pids optional). The window family is the empirical HPD: the offset pos(v1) - pos(v2) for a
# J-hop jump is learned from fitting beds of the SAME generator, so a changed gap law cannot hide from it.
import sys
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/foreman")
from hunt import pointer0, doubling, hybrid_space  # noqa: E402


def last_root(parent):
    n = len(parent)
    return np.maximum.accumulate(np.where(parent < 0, np.arange(n), 0))


def offset_table(beds, Jmax, W):
    """For J = 1..Jmax: the W-wide window [a, a+W) of offsets s = pos(x) - pos(J-th ancestor of x) with most mass."""
    hist = {}
    for b in beds:
        p = b["parent"]
        A = pointer0(p)
        idx = np.arange(len(p))
        anc = idx.copy()
        for J in range(1, Jmax + 1):
            anc = A[anc]
            ok = b["depth"] >= J
            s = (idx - anc)[ok]
            if s.size == 0:
                break
            hist[J] = hist.get(J, np.zeros(1, int))
            m = max(hist[J].size, s.max() + 1)
            h = np.zeros(m, int)
            h[:hist[J].size] += hist[J]
            h += np.bincount(s, minlength=m)
            hist[J] = h
    start = {0: 1}
    for J, h in hist.items():
        c = np.concatenate([[0], np.cumsum(h)])
        if h.size <= W:
            start[J] = 1
            continue
        mass = c[W:] - c[:-W]
        mass[0] = -1
        start[J] = int(np.argmax(mass))
    return start


def ehpd(parent, depth, L, W, k, start):
    A = pointer0(parent)
    for _ in range(L):
        v1 = A
        cur = A[v1]
        J = depth - depth[v1]
        a = np.array([start.get(int(j), 1) for j in range(J.max() + 1)])[J]
        lo, hi = v1 - (a + W - 1), v1 - a + 1
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


def score(beds, L, pricing, fit, mask_fn):
    """Best credited accuracy on mask_fn(bed) over the pricing configs, for doubling / eHPD / local."""
    out = {}
    tables = {W: offset_table(fit, 2 ** L * 4, W) for _, W in pricing}
    acc = {}
    for b in beds:
        p, d, r = b["parent"], b["depth"], b["root"]
        lr = last_root(p)
        m = mask_fn(b)
        acc.setdefault("doubling", []).append((lr[doubling(p, L)] == r)[m])
        for k, W in pricing:
            acc.setdefault(f"ehpd_k{k}_W{W}", []).append((lr[ehpd(p, d, L, W, k, tables[W])] == r)[m])
            acc.setdefault(f"local_k{k}_W{W}", []).append(
                (lr[hybrid_space(p, d, np.arange(len(p)), L, W, k, "anchored", anchor_self=True)[0]] == r)[m])
    mean = {key: float(np.concatenate(v).mean()) for key, v in acc.items()}
    for fam in ("ehpd", "local"):
        best = max((x for x in mean if x.startswith(fam)), key=mean.get)
        out[fam], out[fam + "_cfg"] = mean[best], best
    out["doubling"] = mean["doubling"]
    out["tokens"] = int(sum(x.size for x in acc["doubling"]))
    return out
