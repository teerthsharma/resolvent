# Foreman K1: the W = 14 sc-HPD window on bed_k' built with real causal softmax heads at d = 128. Bar: test_kp_w14.py.
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kp_attack as K        # noqa: E402
from hunt import pointer0    # noqa: E402

torch.set_num_threads(24)
W, L, N = 14, 7, 4096
PER = np.array([8.0, 64.0, 512.0, 4096.0])
BETA, BIG, CHUNK = 100.0, 8192.0, 512
HEADS, D_HEAD = 1 + W, 2 * len(PER)                # 15 heads x 8 = 120 <= 128
MLP_UNITS = 4 * W + 1                              # 57 <= 512


def rot(pos, dt):
    ang = 2 * np.pi * np.asarray(pos, np.float64)[..., None] / PER
    return torch.tensor(np.concatenate([np.cos(ang), np.sin(ang)], -1), dtype=dt)


def relu(x):
    return torch.clamp(x, min=0)


def realized_layer(A, depth, t2, dt):
    v1 = A
    J = depth - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo = v1 - (a + W - 1)                                             # idealized per-token arithmetic
    qpos = np.concatenate([v1[None], lo[None] + np.arange(W)[:, None]], 0)   # (15, N) addressed positions
    Q = rot(qpos, dt)                                                 # (15, N, 8)
    Kc = rot(np.arange(N), dt)                                        # (N, 8) keys: own position code
    V = torch.tensor(A, dtype=dt)                                     # value: the token's pointer (a position)
    out = torch.empty(HEADS, N, dtype=dt)
    j = torch.arange(N)
    for t0 in range(0, N, CHUNK):
        t = torch.arange(t0, t0 + CHUNK)
        lg = BETA * torch.einsum("hqd,kd->hqk", Q[:, t0:t0 + CHUNK], Kc)
        lg = lg.masked_fill((j[None, :] > t[:, None])[None], float("-inf"))
        out[:, t0:t0 + CHUNK] = torch.softmax(lg, dim=-1) @ V
    x2, s = out[0], out[1:]                                           # pos(v2); window slot values
    d = x2[None] - torch.tensor(lo, dtype=dt)[None] - torch.arange(W, dtype=dt)[:, None]
    bump = relu(d + 1) - 2 * relu(d) + relu(d - 1)                    # 3 ReLU units per slot
    sel = relu(s - BIG * (1 - bump)).sum(0)                           # 1 gate unit per slot
    new = sel + relu(x2 - BIG * bump.sum(0))                          # fallback unit
    err = float((new - torch.round(new)).abs().max())
    return torch.round(new).numpy().astype(np.int64), err


def main():
    beds = [K.bed_kp.make_test(np.random.default_rng([22, N, k]), N) for k in range(8)]
    cal = [K.bed_kp.make_test(np.random.default_rng([23, N, k]), N) for k in range(8)]
    tables = K.calibrate(cal, L, W, 0)
    ideal_acc, _ = K.run(beds, L, W, 0, tables)
    out = {"idealized_mean": ideal_acc, "heads": HEADS, "d_head": D_HEAD, "attn_dims": HEADS * D_HEAD,
           "mlp_units": MLP_UNITS, "beta": BETA, "periods": PER.tolist()}
    for name, dt in (("f64", torch.float64), ("f32", torch.float32)):
        t0 = time.time()
        mism, maxerr, accs = 0, 0.0, []
        for b in beds:
            p, dep = b["parent"], b["depth"]
            Ai = pointer0(p)
            Ar = pointer0(p)
            for t2, t3 in tables:
                Ai = K.step(Ai, dep, W, 0, t2, t3)
                Ar, e = realized_layer(Ar, dep, t2, dt)
                mism += int((Ai != Ar).sum())
                maxerr = max(maxerr, e)
            accs.append(float((K.lastroot(p)[Ar] == b["root"])[dep > 2 ** L].mean()))
        out[name] = {"mismatches": mism, "max_prerounding_err": maxerr, "mean": float(np.mean(accs)), "per_bed": accs,
                     "wall_s": round(time.time() - t0, 1)}
        print(name, out[name], flush=True)
    (HERE / "kp_w14.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
