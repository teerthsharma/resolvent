# Foreman K1: hand-set constructions on bed_k, with and without an absolute-position oracle. Bars: test_hunt.py.
# Imports K0 code read-only (no bytecode written into the repo).
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
K0 = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0"
sys.path[:0] = [K0 + "/cameron", K0 + "/foreman"]
import bed_k                                           # noqa: E402
from shortcut import hybrid, doubling, wald_bed, pointer0  # noqa: E402

HERE = Path(__file__).parent
R0 = [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)]   # Cameron's K1.F pricing at d = 128
WIDE = [(3, W) for W in (16, 32, 64, 128)]                            # outside R0: unbarred sweep
M = 16


def hybrid_space(parent, depth, key, L, W, k, mode, anchor_self=False):
    """K0's hybrid with the window placed in an arbitrary coordinate `key` (positions = the oracle; ids = the literal
    K0 mechanism 'the token whose id is v1 - j'). anchor_self: window [t - W, t) around the token itself (ALiBi-local)."""
    A = pointer0(parent)
    idx = np.arange(len(parent))
    hits = 0
    for _ in range(L):
        v1 = A
        cur = A[v1]
        if anchor_self:
            lo, hi = key[idx] - W, key[idx]
        elif mode == "anchored":
            lo, hi = key[v1] - W, key[v1]
        else:
            c = key[v1] - M * (depth - depth[v1])
            lo, hi = np.maximum(c - W // 2, 0), np.minimum(c + W // 2, key[v1])
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((key[cur] >= lo) & (key[cur] < hi)) | (cur == v1))
            hits += int((inw & (cur != v1)).sum())
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A, hits


def bundle_stale(parent, L, W, k):
    """No oracle: v1 stores the pointers of the W positions before it, gathered one layer earlier (ALiBi-local)."""
    A = pointer0(parent)
    prev = None
    for _ in range(L):
        v1 = A
        cur = A[v1]
        if prev is not None:
            live = np.ones(len(A), bool)
            for _ in range(k - 2):
                inw = live & (cur >= v1 - W) & (cur < v1)
                cur = np.where(inw, prev[cur], cur)
                live = inw
        prev, A = A, cur
    return A


def beyond(ok, d, L):
    return float(ok[d > 2 ** L].mean())


def closure(parent, root, L, overlap):
    n = len(parent)
    H = [frozenset((i, parent[i])) if parent[i] >= 0 else frozenset((i,)) for i in range(n)]
    for _ in range(L):
        inv = {}
        if overlap:
            for u in range(n):
                for x in H[u]:
                    inv.setdefault(x, []).append(u)
        new = []
        for t in range(n):
            s = set(H[t])
            for u in H[t]:
                s |= H[u]
            if overlap:
                for x in H[t]:
                    for u in inv[x]:
                        if u <= t:
                            s |= H[u]
            new.append(frozenset(s))
        H = new
    return np.array([root[t] in H[t] for t in range(n)])


def main():
    out = {"machinery": {}, "wald": {}, "bedk": {}, "closure": {}}
    mism = 0
    # ---- pinned wald bed: ids are positions
    p, d, r = wald_bed()
    ids = np.arange(len(p))
    Aid, _ = hybrid_space(p, d, ids, 6, 26, 3, "anchored")
    Apos = hybrid(p, d, 6, 26, 3, "anchored")
    out["wald"] = {"mismatches": int((Aid != Apos).sum()), "id_addr_all": float((Aid == r).mean()),
                   "pos_all": float((Apos == r).mean())}
    # ---- bed_k test beds, Cameron's K1.F seeds
    for n in (4096, 8192, 16384):
        beds = [bed_k.make_test(np.random.default_rng([12, n, k]), n) for k in range(8)]
        out["bedk"][str(n)] = {}
        for L in (4, 7):
            acc = {}
            hit = {}
            for b in beds:
                p, d, r, ids = b["parent"], b["depth"], b["root"], b["ids"]
                pos = np.arange(n)
                lr = bed_k._last_root(p)
                acc.setdefault("dbl", []).append(beyond(lr[doubling(p, L)] == r, d, L))
                for k, W in R0 + WIDE:
                    for mode in ("anchored", "centered"):
                        Ao, _ = hybrid_space(p, d, pos, L, W, k, mode)
                        if n == 4096 and b is beds[0] and (k, W) in R0:
                            mism += int((Ao != hybrid(p, d, L, W, k, mode)).sum())
                        acc.setdefault(f"oracle_{mode}_k{k}_W{W}", []).append(beyond(lr[Ao] == r, d, L))
                        Ai, h = hybrid_space(p, d, ids, L, W, k, mode)
                        acc.setdefault(f"id_{mode}_k{k}_W{W}", []).append(beyond(lr[Ai] == r, d, L))
                        hit.setdefault(f"id_{mode}_k{k}_W{W}", []).append(h / (n * L))
                    Al, _ = hybrid_space(p, d, pos, L, W, k, "anchored", anchor_self=True)
                    acc.setdefault(f"local_k{k}_W{W}", []).append(beyond(lr[Al] == r, d, L))
                    acc.setdefault(f"stale_k{k}_W{W}", []).append(beyond(lr[bundle_stale(p, L, W, k)] == r, d, L))
            mean = {key: float(np.mean(v)) for key, v in acc.items()}
            cell = {"dbl_cred": mean["dbl"], "sweep": mean}
            for fam in ("oracle", "id", "local", "stale"):
                inb = [key for key in mean if key.startswith(fam + "_") and
                       any(key.endswith(f"_k{k}_W{W}") for k, W in R0)]
                best = max(inb, key=lambda kk: mean[kk])
                cell[f"{fam}_best"], cell[f"{fam}_best_cfg"] = mean[best], best
            cell["id_hit_rate"] = max(float(np.mean(v)) for key, v in hit.items()
                                      if any(key.endswith(f"_k{k}_W{W}") for k, W in R0))
            out["bedk"][str(n)][str(L)] = cell
            if n == 4096 and L == 7:
                out["machinery"]["cameron_cell"] = mean["oracle_centered_k3_W10"]
    out["machinery"]["space_vs_hybrid_mismatches"] = mism
    # ---- content closure
    cbeds = [bed_k.make_bed(1024, 16, np.random.default_rng([13, k])) for k in range(8)]
    for L in range(1, 6):
        mp = mo = 0
        for p, d, r in cbeds:
            want = d <= 2 ** L
            mp += int((closure(p, r, L, False) != want).sum())
            mo += int((closure(p, r, L, True) != want).sum())
        out["closure"][str(L)] = {"mismatch_plain": mp, "mismatch_overlap": mo, "tokens": 8 * 1024,
                                  "resolved_frac": float(np.mean([(d <= 2 ** L).mean() for _, d, _ in cbeds]))}
    (HERE / "hunt.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    R = main()
    print(json.dumps({"machinery": R["machinery"], "wald": R["wald"], "closure": R["closure"]}, indent=1))
    for n, row in R["bedk"].items():
        for L, c in row.items():
            s = c["sweep"]
            print(f"n={n} L={L} dbl {c['dbl_cred']:.4f} | oracle {c['oracle_best']:.4f} [{c['oracle_best_cfg']}] | "
                  f"id {c['id_best']:.4f} hit {c['id_hit_rate']:.2e} | local {c['local_best']:.4f} [{c['local_best_cfg']}] | "
                  f"stale {c['stale_best']:.4f} [{c['stale_best_cfg']}] || wide(unbarred) local W64 {s['local_k3_W64']:.4f} "
                  f"W128 {s['local_k3_W128']:.4f} stale W64 {s['stale_k3_W64']:.4f} W128 {s['stale_k3_W128']:.4f}")
