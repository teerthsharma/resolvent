# Foreman K1: densest buildable window at d = 128 (W = 30, d_head 4) and the far-band property on bed_k''s generator.
# Bars: test_kp_next.py.
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kp_attack as K                     # noqa: E402
from hunt import pointer0, doubling, hybrid_space  # noqa: E402

torch.set_num_threads(24)


def rot(pos, per, dt):
    ang = 2 * np.pi * np.asarray(pos, np.float64)[..., None] / per
    return torch.tensor(np.concatenate([np.cos(ang), np.sin(ang)], -1), dtype=dt)


def relu(x):
    return torch.clamp(x, min=0)


def realized_layer(A, depth, t2, W, per, beta, dt, big=8192.0, chunk=256):
    n = len(A)
    v1 = A
    J = depth - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo = v1 - (a + W - 1)
    Q = rot(np.concatenate([v1[None], lo[None] + np.arange(W)[:, None]], 0), per, dt)
    Kc = rot(np.arange(n), per, dt)
    V = torch.tensor(A, dtype=dt)
    out = torch.empty(W + 1, n, dtype=dt)
    j = torch.arange(n)
    for t0 in range(0, n, chunk):
        t = torch.arange(t0, min(t0 + chunk, n))
        lg = beta * torch.einsum("hqd,kd->hqk", Q[:, t0:t0 + chunk], Kc)
        lg = lg.masked_fill((j[None, :] > t[:, None])[None], float("-inf"))
        out[:, t0:t0 + chunk] = torch.softmax(lg, dim=-1) @ V
    x2, s = out[0], out[1:]
    d = x2[None] - torch.tensor(lo, dtype=dt)[None] - torch.arange(W, dtype=dt)[:, None]
    bump = relu(d + 1) - 2 * relu(d) + relu(d - 1)
    new = relu(s - big * (1 - bump)).sum(0) + relu(x2 - big * bump.sum(0))
    return torch.round(new).numpy().astype(np.int64), float((new - torch.round(new)).abs().max())


def w30():
    W, L, N = 30, 7, 4096
    per, beta = np.array([64.0, 4096.0]), 6000.0
    beds = [K.bed_kp.make_test(np.random.default_rng([22, N, k]), N) for k in range(8)]
    cal = [K.bed_kp.make_test(np.random.default_rng([23, N, k]), N) for k in range(8)]
    tables = K.calibrate(cal, L, W, 0)
    ideal, _ = K.run(beds, L, W, 0, tables)
    mism, err, accs = 0, 0.0, []
    for b in beds:
        p, dep = b["parent"], b["depth"]
        Ai, Ar = pointer0(p), pointer0(p)
        for t2, t3 in tables:
            Ai = K.step(Ai, dep, W, 0, t2, t3)
            Ar, e = realized_layer(Ar, dep, t2, W, per, beta, torch.float32)
            mism += int((Ai != Ar).sum())
            err = max(err, e)
        accs.append(float((K.lastroot(p)[Ar] == b["root"])[dep > 2 ** L].mean()))
    return {"mismatches": mism, "mean": float(np.mean(accs)), "idealized_mean": ideal, "max_prerounding_err": err,
            "heads": W + 1, "d_head": 2 * len(per), "attn_dims": (W + 1) * 2 * len(per), "mlp_units": 4 * W + 1}


def reach_and_far(L, beds, cal):
    depth = np.concatenate([b["depth"] for b in beds])
    oks = {}
    for W in (10, 14, 30):
        tab = K.calibrate(cal, L, W, 0)
        o = []
        for b in beds:
            A = pointer0(b["parent"])
            for t2, t3 in tab:
                A = K.step(A, b["depth"], W, 0, t2, t3)
            o.append(K.lastroot(b["parent"])[A] == b["root"])
        oks[f"schpd_W{W}"] = np.concatenate(o)
    oks["local_W30"] = np.concatenate([K.lastroot(b["parent"])[hybrid_space(b["parent"], b["depth"], np.arange(len(b["parent"])),
                                                                                L, 30, 3, "anchored", anchor_self=True)[0]] == b["root"] for b in beds])
    oks["doubling"] = np.concatenate([K.lastroot(b["parent"])[doubling(b["parent"], L)] == b["root"] for b in beds])
    w = max(1, 2 ** L // 8)

    def reach(ok):
        D = 0
        for d in range(w, depth.max() + 1, w):
            m = (depth > d - w) & (depth <= d)
            if m.sum() < 50:
                continue
            if ok[m].mean() < 0.5:
                break
            D = d
        return D / 2 ** L
    rs = {k: reach(v) for k, v in oks.items()}
    c = max(v for k, v in rs.items() if k != "doubling")
    far = depth > 2 * c * 2 ** L
    fa = {k: float(v[far].mean()) for k, v in oks.items()}
    return {"c": c, "per_family": rs}, {"kappa": 2 * c, "tokens": int(far.sum()), "worst": max(fa.values()), "per_family": fa}


def main():
    out = {"w30": w30(), "reach": {}, "far": {}}
    print("w30", out["w30"], flush=True)
    beds = [K.bed_kp.make_test(np.random.default_rng([57, k]), 32768) for k in range(4)]
    cal = [K.bed_kp.make_test(np.random.default_rng([58, k]), 32768) for k in range(4)]
    for L in (4, 7):
        out["reach"][str(L)], out["far"][str(L)] = reach_and_far(L, beds, cal)
        print(L, out["reach"][str(L)], out["far"][str(L)], flush=True)
    (HERE / "kp_next.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
