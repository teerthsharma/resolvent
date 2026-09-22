# Foreman K0 round 3: the anchored hybrid (L=6, W=26, k=3) with real causal softmax heads on the pinned bed.
# Heads: 1 content head + W positional-window heads, d_head = 12 (+-1 binary codes of ids; ids ARE positions in
# wald.py), logit scale beta. MLP: explicit ReLU multiplexer. Still idealized (not weight-built): the per-token
# 12-bit arithmetic code(v1 - j) and onehot(t - A(t)), which the previous layer's MLP would have to compute.
import json
from pathlib import Path
import numpy as np
from shortcut import wald_bed, pointer0

HERE = Path(__file__).parent
BITS, W, L, BETA = 12, 26, 6, 4.0


def codes(x):
    return (2 * ((np.asarray(x)[:, None] >> np.arange(BITS)) & 1) - 1).astype(np.float64)


def decode(c):
    return ((c > 0).astype(np.int64) << np.arange(BITS)).sum(1)


def attend(qcode, K, V, causal):  # one causal softmax head
    lg = BETA * (qcode @ K.T)
    lg[~causal] = -np.inf
    lg -= lg.max(1, keepdims=True)
    P = np.exp(lg)
    P /= P.sum(1, keepdims=True)
    return P @ V


def relu(x):
    return np.maximum(x, 0.0)


def main():
    parent, depth, root = wald_bed()
    n = len(parent)
    t = np.arange(n)
    K = codes(t)                                   # key = own id code
    causal = t[None, :] <= t[:, None]
    A_ideal = pointer0(parent)
    A = pointer0(parent)
    mism, margin, per_layer = 0, np.inf, []
    for _ in range(L):
        # ---- idealized reference (shortcut.hybrid, anchored, k=3)
        v1 = A_ideal
        cur = A_ideal[v1]
        inw = ((cur >= v1 - W) & (cur < v1)) | (cur == v1)
        A_ideal = np.where(inw, A_ideal[cur], cur)
        # ---- softmax-head version, operating only on codes
        gap = t - A
        oh = np.zeros((n, W))
        sel = (gap >= 1) & (gap <= W)
        oh[t[sel], gap[sel] - 1] = 1.0              # onehot(gap), slot j-1 <-> offset j
        Vc = np.concatenate([codes(A), oh], 1)      # content head value: my pointer code + my one-hot gap
        out = attend(codes(A), K, Vc, causal)       # query = code of v1 = A(t)
        v2c, ohs = out[:, :BITS], out[:, BITS:]
        margin = min(margin, np.abs(v2c).min(), np.abs(ohs - 0.5).min() * 2)
        slots = np.zeros((n, W, BITS))
        for j in range(1, W + 1):
            q = codes(np.maximum(A - j, 0))
            slots[:, j - 1] = attend(q, K, codes(A), causal)
        used = ohs > 0.5
        if used.any():
            margin = min(margin, np.abs(slots[used]).min())
        # ReLU multiplexer: x if gate else 0  ==  relu(x + 2(g-1)) - relu(-x + 2(g-1)) for x in [-1, 1]
        g = (ohs > 0.5).astype(np.float64)[:, :, None]
        xs = np.sign(slots)
        pick = (relu(xs + 2 * (g - 1)) - relu(-xs + 2 * (g - 1))).sum(1)
        fb = 1.0 - g.sum(1)                        # 1 when no slot fires -> fall back to v2
        x2 = np.sign(v2c)
        new = pick + relu(x2 + 2 * (fb - 1)) - relu(-x2 + 2 * (fb - 1))
        A = decode(new)
        m = int((A != A_ideal).sum())
        mism += m
        per_layer.append({"mismatches": m, "acc_all": float((A == root).mean())})
    ok = A == root
    R = {"L": L, "W": W, "beta": BETA, "d_head": BITS, "heads": W + 1, "per_layer": per_layer,
         "mismatches_total": mism, "all": float(ok.mean()), "beyond": float(ok[depth > 64].mean()),
         "min_margin": float(margin)}
    (HERE / "results3.json").write_text(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
