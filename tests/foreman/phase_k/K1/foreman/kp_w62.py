# Foreman K1, Amendment K1-a follow-ups (1) 2-dim slots and (2) the 30-slot build at long n. Bars: test_kp_w62.py.
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kp_attack as K                               # noqa: E402
from kp_next import rot, relu                       # noqa: E402
from hunt import pointer0, doubling, hybrid_space   # noqa: E402

torch.set_num_threads(24)
F32 = torch.float32


def realize(A, depth, t2, W, per, beta, rows=None, big=65536.0, chunk=128):
    """One realized layer (real softmax heads + ReLU multiplexer) on query rows `rows` (default: all)."""
    n = len(A)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    v1 = A[rows]
    J = depth[rows] - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo = v1 - (a + W - 1)
    Q = rot(np.concatenate([v1[None], lo[None] + np.arange(W)[:, None]], 0), per, F32)   # (W+1, r, dh)
    Kc = rot(np.arange(n), per, F32)
    V = torch.tensor(A, dtype=F32)
    out = torch.empty(W + 1, len(rows), dtype=F32)
    j = torch.arange(n)
    rt = torch.tensor(rows)
    for r0 in range(0, len(rows), chunk):
        t = rt[r0:r0 + chunk]
        lg = beta * torch.einsum("hqd,kd->hqk", Q[:, r0:r0 + chunk], Kc)
        lg = lg.masked_fill((j[None, :] > t[:, None])[None], float("-inf"))
        out[:, r0:r0 + chunk] = torch.softmax(lg, dim=-1) @ V
    x2, s = out[0], out[1:]
    d = x2[None] - torch.tensor(lo, dtype=F32)[None] - torch.arange(W, dtype=F32)[:, None]
    bump = relu(d + 1) - 2 * relu(d) + relu(d - 1)
    new = relu(s - big * (1 - bump)).sum(0) + relu(x2 - big * bump.sum(0))
    return torch.round(new).numpy().astype(np.int64)


def beds(n, k_n):
    return ([K.bed_kp.make_test(np.random.default_rng([22, n, k]), n) for k in range(k_n)],
            [K.bed_kp.make_test(np.random.default_rng([23, n, k]), n) for k in range(8)])


def full_chain(bs, cal, L, W, per, beta):
    tables = K.calibrate(cal, L, W, 0)
    mism, Ars = 0, []
    for b in bs:
        p, dep = b["parent"], b["depth"]
        Ai, Ar = pointer0(p), pointer0(p)
        for t2, t3 in tables:
            Ai = K.step(Ai, dep, W, 0, t2, t3)
            Ar = realize(Ar, dep, t2, W, per, beta)
            mism += int((Ai != Ar).sum())
        Ars.append(Ar)
    return mism, Ars


def layerwise(bs, cal, L, W, per, beta, nrows=2048):
    tables = K.calibrate(cal, L, W, 0)
    bad = tot = 0
    for k, b in enumerate(bs):
        p, dep = b["parent"], b["depth"]
        A = pointer0(p)
        for layer, (t2, t3) in enumerate(tables):
            nxt = K.step(A, dep, W, 0, t2, t3)
            rows = np.random.default_rng([61, k, layer]).choice(len(p), nrows, replace=False)
            bad += int((realize(A, dep, t2, W, per, beta, rows) != nxt[rows]).sum())
            tot += nrows
            A = nxt
    return {"mismatches": bad, "rows": tot, "rate": bad / tot}


def reach_far(L):
    bs = [K.bed_kp.make_test(np.random.default_rng([57, k]), 32768) for k in range(4)]
    cal = [K.bed_kp.make_test(np.random.default_rng([58, k]), 32768) for k in range(4)]
    depth = np.concatenate([b["depth"] for b in bs])
    oks = {}
    for W in (10, 14, 30, 62):
        tab = K.calibrate(cal, L, W, 0)
        o = []
        for b in bs:
            A = pointer0(b["parent"])
            for t2, t3 in tab:
                A = K.step(A, b["depth"], W, 0, t2, t3)
            o.append(K.lastroot(b["parent"])[A] == b["root"])
        oks[f"schpd_W{W}"] = np.concatenate(o)
    oks["local_W30"] = np.concatenate([K.lastroot(b["parent"])[hybrid_space(b["parent"], b["depth"], np.arange(32768), L, 30, 3,
                                                                                "anchored", anchor_self=True)[0]] == b["root"] for b in bs])
    oks["doubling"] = np.concatenate([K.lastroot(b["parent"])[doubling(b["parent"], L)] == b["root"] for b in bs])
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
    far = depth > 2 * rs["schpd_W62"] * 2 ** L
    fa = {k: float(v[far].mean()) for k, v in oks.items()}
    return ({"c_w62": rs["schpd_W62"], "per_family": rs},
            {"kappa": 2 * rs["schpd_W62"], "tokens": int(far.sum()), "worst": max(fa.values()), "per_family": fa})


def main():
    out = {"w62_4096": {}, "reach": {}, "far": {}, "w30_long": {}}
    t0 = time.time()
    for L in (4, 7):
        out["reach"][str(L)], out["far"][str(L)] = reach_far(L)
        print("reach", L, out["reach"][str(L)], out["far"][str(L)], f"{time.time() - t0:.0f}s", flush=True)
    bs, cal = beds(4096, 8)
    for L in (4, 7):
        m, _ = full_chain(bs, cal, L, 62, np.array([4096.0]), 3e7)
        out["w62_4096"][str(L)] = {"mismatches": m, "beds": 8}
        print("w62_4096", L, m, f"{time.time() - t0:.0f}s", flush=True)
    for n, beta in ((8192, 1e8), (16384, 4e8)):
        bs, cal = beds(n, 4)
        out[f"w62_{n}"] = dict(layerwise(bs, cal, 7, 62, np.array([float(n)]), beta), beta=beta)
        print(f"w62_{n}", out[f"w62_{n}"], f"{time.time() - t0:.0f}s", flush=True)
    for n, nb in ((8192, 4), (16384, 2)):
        bs, cal = beds(n, nb)
        out["w30_long"][str(n)] = {}
        for L, kappa in ((4, 6), (7, 7)):
            m, Ars = full_chain(bs, cal, L, 30, np.array([np.sqrt(n), float(n)]), 3.0 * n)
            ok = np.concatenate([K.lastroot(b["parent"])[A] == b["root"] for b, A in zip(bs, Ars)])
            dep = np.concatenate([b["depth"] for b in bs])
            band = dep > kappa * 2 ** L
            out["w30_long"][str(n)][str(L)] = {"mismatches": m, "beds": nb, "kappa": kappa, "band_tokens": int(band.sum()),
                                               "band_acc": float(ok[band].mean()) if band.any() else None,
                                               "beyond_2L": float(ok[dep > 2 ** L].mean())}
            print("w30", n, L, out["w30_long"][str(n)][str(L)], f"{time.time() - t0:.0f}s", flush=True)
    (HERE / "kp_w62.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
