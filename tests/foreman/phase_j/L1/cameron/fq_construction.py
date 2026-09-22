"""Certificate that the SU(2) prefix state carries the shell-bed label exactly.
Each move token gets a fixed unit quaternion (a 180-degree rotation, since every
transposition of cups is a half-turn of a rigid body whose axes are the cups):
  S3: cups = 3 axes at 120 deg in a plane; (a b) = half-turn about the third cup's axis
  S4: cups = the 4 body diagonals of a cube; (a b) = half-turn about d_a - d_b
Pi from su2.prefix_scan (the arms' contract-order scan), fp32, L=64; label read as
the cup axis nearest R(Pi) d_0. Same eval set as shell_bed.py (seed 12345, N=2000).
Writes fq_construction.json."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "../..")
import su2, shell_bed as sb


def axes(name):
    if name == "S3":
        return np.array([[np.cos(2 * np.pi * c / 3), np.sin(2 * np.pi * c / 3), 0.0] for c in range(3)])
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)


def token_quats(name, cycles, A):
    qs = []
    for cyc in cycles:
        a, b = cyc
        if name == "S3":
            u = A[3 - a - b]
        else:
            u = A[a] - A[b]
        u = u / np.linalg.norm(u)
        qs.append([0.0, *u])
    return torch.tensor(qs, dtype=torch.float32)


def rot(q, v):  # q v conj(q) on pure quaternions, batched
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return su2.qmul(su2.qmul(q, vq), su2.qconj(q))[..., 1:]


res = {}
for name in ("S3", "S4"):
    spec = sb.BEDS[name]
    rng = np.random.default_rng(sb.EVAL_SEED)
    tok = sb.gen_sequences(spec["n_tok"], sb.EVAL_N, sb.L, rng)
    lab = sb.labels_from_tokens(tok, spec["table"])
    A = axes(name)
    Q = token_quats(name, spec["cycles"], A)
    Pi = su2.prefix_scan(Q[torch.from_numpy(tok).long()])  # (N, L, 4), Pi_t = q_t ... q_0
    At = torch.tensor(A, dtype=torch.float32)
    img = rot(Pi, At[0].expand(Pi.shape[0], Pi.shape[1], 3))
    pred = (img @ At.T).abs().argmax(-1).numpy()
    acc = (pred == lab).mean(0)
    res[name] = dict(acc_all_positions_min=float(acc.min()), acc_mean_ge32=float(acc[31:].mean()),
                     acc_at={str(p): float(acc[p - 1]) for p in (8, 16, 32, 48, 64)})
    print(name, res[name])
json.dump(res, open("fq_construction.json", "w"), indent=1)
