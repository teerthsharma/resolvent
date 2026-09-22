# wald.py - best response FIRST: floors and ceilings on a pointer-chase root bed, before any bar is written.
# L-REPRO: seed 23; float64; n tokens, M interleaved chains; each token's content = (own id, parent id);
# chain identity is NOT in the content. Task: every token outputs the id of its chain's root.
# Resolvent head: one layer, hand-set content attention (logit s on the parent, s on self for roots, 0 elsewhere),
# beta = 1, read z = (I - gamma W)^-1 V with V = one-hot root id on root tokens, 0 elsewhere; predict argmax z_i.
import numpy as np

rng = np.random.default_rng(23)


def make_bed(n, M):
    chain = rng.integers(0, M, n)
    parent = -np.ones(n, int)
    depth = np.zeros(n, int)
    root = np.arange(n)
    last = {}
    for i in range(n):
        c = chain[i]
        if c in last:
            p = last[c]
            parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


def resolvent_softmax(parent, root, s, gamma):
    n = len(parent)
    roots = np.unique(root)
    col = {r: k for k, r in enumerate(roots)}
    z = np.zeros((n, roots.size))
    S = np.zeros(roots.size)                      # running sum of z_j over j < i
    es = np.exp(s)
    for i in range(n):
        Z = i + es                                  # i entries at logit 0, one entry at logit s
        if parent[i] < 0:                           # root: self at s, all earlier at 0
            v = np.zeros(roots.size)
            v[col[i]] = 1.0
            z[i] = (v + gamma * S / Z) / (1 - gamma * es / Z)
        else:                                       # non-root: parent at s, the rest (incl. self) at 0
            p = parent[i]
            z[i] = (gamma * (S + (es - 1) * z[p]) / Z) / (1 - gamma / Z)
        S += z[i]
    return (roots[np.argmax(z, 1)] == root).mean()


n, M = 4096, 16
parent, depth, root = make_bed(n, M)
print(f"bed: n={n}, {M} interleaved chains, max depth {depth.max()}, median depth {int(np.median(depth))}")

# floors (position only, no content)
last_root, ok = 0, 0
for i in range(n):
    if parent[i] < 0:
        last_root = i
    ok += root[i] == last_root
print(f"floor  position-only (most recent root): {ok / n:.4f};  chance 1/M = {1 / M:.4f}")

# opponent best response: L-layer transformer with perfect pointer doubling (generous: layer 1 = 2 hops)
for L in (4, 6, 8):
    print(f"ceiling L={L} transformer, perfect pointer doubling (reaches 2^L = {2 ** L} hops): {(depth <= 2 ** L).mean():.4f}")

# the resolvent head, one layer
for s in (8.0, 12.0, 16.0, 20.0):
    print(f"resolvent 1 layer, softmax logit scale s={s:4.1f} (gamma 0.999): {resolvent_softmax(parent, root, s, 0.999):.4f}")
print("resolvent 1 layer, sparsemax (exact parent weight 1): 1.0000 by construction (B_i = B_parent)")
print(f"softmax scale needed for per-hop leak <= 1/(10*depth_max): s >= ln(n * 10 * D) = {np.log(n * 10 * depth.max()):.1f}")
