# Foreman K0: hand-set opponent constructions on the pinned wald.py bed (seed 23) and on 8 fresh beds.
# Every construction is a per-layer information-flow simulation of a causal transformer: in one layer a token
# may (a) fetch one token's pre-layer state per head at an address it knows BEFORE the layer (content lookup),
# and (b) its MLP may combine what it fetched. Width is accounted, never assumed free.
import contextlib, io, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
WALD = Path("C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/instances/wald.py")
M, N = 16, 4096
BITS = 12  # one pointer = one position in [0, 4096) = 12 bits


def wald_bed():
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):  # exec the pinned file untouched; no .pyc written
        exec(compile(WALD.read_text(), str(WALD), "exec"), ns)
    return ns["parent"], ns["depth"], ns["root"]


def fresh_bed(seed, n=N, m=M):  # same generator as wald.make_bed, own numpy Generator
    chain = np.random.default_rng(seed).integers(0, m, n)
    parent, depth, root, last = -np.ones(n, int), np.zeros(n, int), np.arange(n), {}
    for i in range(n):
        c = chain[i]
        if c in last:
            p = last[c]
            parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


def score(pred, depth, root, L):
    ok = pred == root
    b = depth > 2 ** L
    return {"all": float(ok.mean()), "beyond": float(ok[b].mean()) if b.any() else None, "n_beyond": int(b.sum())}


def pointer0(parent):
    A = parent.copy()
    A[A < 0] = np.flatnonzero(parent < 0)  # a root points to itself (saturation)
    return A


def doubling(parent, L):
    A = pointer0(parent)
    for _ in range(L):
        A = A[A]  # one head: fetch my pointer's pointer
    return A


def hybrid(parent, depth, L, W, k, mode, m=M):
    """Content fetch of v2 = A(v1) plus a positional window of W slots holding A(u) for u near v1.
    k = max hops of the current pointer per layer (2 = doubling). Chasing hop j >= 3 inside the window is a
    (j-2)-level multiplexer: MLP width ~ W^(k-2) * BITS. anchored: window [v1-W, v1); centered: window of W
    around the expected spot of v2, v1 - M*hops(t) (hop counts are additive, so a token can track them)."""
    A = pointer0(parent)
    idx = np.arange(len(parent))
    for _ in range(L):
        v1 = A
        cur = A[v1]  # hop 2: content lookup, always available
        if mode == "anchored":
            lo, hi = v1 - W, v1  # window of positions [lo, hi)
        else:
            c = v1 - m * (depth - depth[v1])  # m = chain count, the mean positional gap per hop
            lo, hi = np.maximum(c - W // 2, 0), np.minimum(c + W // 2, v1)
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
        assert (A <= idx).all()  # causal: pointers only go backwards
    return A


def probes(parent, depth, root, train_seeds):
    """Content/position-only predictors that never follow the chain to a root."""
    n = len(parent)
    out = {}
    # P1 wald's floor: most recent root
    lr, pred = 0, np.zeros(n, int)
    for i in range(n):
        if parent[i] < 0:
            lr = i
        pred[i] = lr
    out["most_recent_root"] = score(pred, depth, root, 6)
    roots = np.sort(np.unique(root))
    # P2 round-robin leak: root rank = position mod 16
    out["pos_mod_16"] = score(roots[np.arange(n) % len(roots)], depth, root, 6)

    # P3 learned local table: (pos mod 16, own gap, parent's gap, grandparent position mod 16) -> root rank,
    # counts fitted on fresh beds, argmax applied to the wald bed.
    def feats(par):
        gap = np.where(par >= 0, np.arange(n) - par, 0)
        g1 = np.minimum(gap, 63)
        gp = np.where(par >= 0, gap[np.maximum(par, 0)], 0)
        g2 = np.minimum(gp, 63)
        return (np.arange(n) % 16) * 64 * 64 + g1 * 64 + g2

    table = np.zeros((16 * 64 * 64, M))
    for s in train_seeds:
        p, d, r = fresh_bed(s)
        rk = np.searchsorted(np.sort(np.unique(r)), r)
        np.add.at(table, feats(p), np.eye(M)[rk])
    rank_pred = table[feats(parent)].argmax(1)
    out["local_table"] = score(roots[rank_pred], depth, root, 6)
    return out


def main():
    parent, depth, root = wald_bed()
    assert depth.max() == 269 and int(np.median(depth)) == 127, "not the pinned bed"
    R = {"bed": {"max_depth": int(depth.max()), "median_depth": int(np.median(depth)),
                 "n_depth_gt_64": int((depth > 64).sum())}}
    R["doubling"] = {str(L): score(doubling(parent, L), depth, root, L) for L in (4, 5, 6, 7, 8)}
    R["probes"] = probes(parent, depth, root, train_seeds=range(1000, 1040))
    R["hybrid"] = {}
    for L in (4, 6, 8):
        for k in (3, 4):
            for mode in ("anchored", "centered"):
                for W in (8, 16, 26, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 4096):
                    R["hybrid"][f"L{L}_W{W}_k{k}_{mode}"] = dict(
                        score(hybrid(parent, depth, L, W, k, mode), depth, root, L),
                        attn_dims=W * BITS, mlp_hidden=W ** (k - 2) * BITS)
    # replication of the headline configurations on 8 fresh beds
    rep = {}
    for key in ("L6_W4096_k3_anchored", "L6_W26_k3_anchored", "L6_W128_k3_centered", "L6_W64_k3_centered"):
        L, W, k, mode = int(key[1]), int(key.split("_W")[1].split("_")[0]), int(key.split("_k")[1][0]), key.split("_")[-1]
        vals = []
        for s in range(2000, 2008):
            p, d, r = fresh_bed(s)
            vals.append(score(hybrid(p, d, L, W, k, mode), d, r, L)["beyond"])
        rep[key] = {"beyond_mean": float(np.mean(vals)), "beyond_min": float(np.min(vals)), "beyond_max": float(np.max(vals))}
    R["replication_8_fresh_beds"] = rep
    (HERE / "results.json").write_text(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    R = main()
    for L, v in R["doubling"].items():
        print(f"doubling L={L}: all {v['all']:.4f} beyond {v['beyond']}")
    for k, v in R["probes"].items():
        print(f"probe {k}: all {v['all']:.4f} beyond(>64) {v['beyond']:.4f} n={v['n_beyond']}")
    for key, v in R["hybrid"].items():
        if key.startswith("L6"):
            print(f"{key}: all {v['all']:.4f} beyond(>64) {v['beyond']:.4f} attn_dims {v['attn_dims']} mlp_hidden {v['mlp_hidden']}")
    print(json.dumps(R["replication_8_fresh_beds"], indent=1))
