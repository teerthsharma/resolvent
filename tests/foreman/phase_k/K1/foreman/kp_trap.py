# Foreman K1: realized sc-HPD windows with a flat-top (trapezoid) ReLU multiplexer. Bars: test_kp_trap.py.
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0/wilson"]
import kp_attack as K                  # noqa: E402
from kp_next import rot, relu          # noqa: E402
from hunt import pointer0              # noqa: E402
from train_ladder import get_slopes    # noqa: E402

torch.set_num_threads(24)
F32 = torch.float32


def realize(A, depth, t2, W, per, beta, m=0.0, big=65536.0, chunk=128):
    n = len(A)
    v1 = A
    J = depth - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo = v1 - (a + W - 1)
    Q = rot(np.concatenate([v1[None], lo[None] + np.arange(W)[:, None]], 0), per, F32)
    Kc = rot(np.arange(n), per, F32)
    V = torch.tensor(A, dtype=F32)
    out = torch.empty(W + 1, n, dtype=F32)
    jf = torch.arange(n, dtype=F32)
    for t0 in range(0, n, chunk):
        t = jf[t0:t0 + chunk]
        lg = beta * torch.einsum("hqd,kd->hqk", Q[:, t0:t0 + chunk], Kc)
        if m:
            lg = lg - m * (t[:, None] - jf[None, :])[None]
        lg = lg.masked_fill((jf[None, :] > t[:, None])[None], float("-inf"))
        out[:, t0:t0 + chunk] = torch.softmax(lg, dim=-1) @ V
    x2, s = out[0], out[1:]
    d = x2[None] - torch.tensor(lo, dtype=F32)[None] - torch.arange(W, dtype=F32)[:, None]
    bump = 2 * (relu(d + 1) - relu(d + 0.5) - relu(d - 0.5) + relu(d - 1))     # flat top on |d| <= 0.5
    new = relu(s - big * (1 - bump)).sum(0) + relu(x2 - big * bump.sum(0))
    return torch.round(new).numpy().astype(np.int64)


def chain(n, nb, L, W, per, beta, m=0.0, kappa=None):
    bs = [K.bed_kp.make_test(np.random.default_rng([22, n, k]), n) for k in range(nb)]
    cal = [K.bed_kp.make_test(np.random.default_rng([23, n, k]), n) for k in range(8)]
    tables = K.calibrate(cal, L, W, 0)
    mism, ok, dep = 0, [], []
    for b in bs:
        p, d = b["parent"], b["depth"]
        Ai, Ar = pointer0(p), pointer0(p)
        for t2, t3 in tables:
            Ai = K.step(Ai, d, W, 0, t2, t3)
            Ar = realize(Ar, d, t2, W, per, beta, m)
            mism += int((Ai != Ar).sum())
        ok.append(K.lastroot(p)[Ar] == b["root"])
        dep.append(d)
    ok, dep = np.concatenate(ok), np.concatenate(dep)
    res = {"mismatches": mism, "beds": nb, "beyond_2L": float(ok[dep > 2 ** L].mean())}
    if kappa is not None:
        band = dep > kappa * 2 ** L
        res.update(kappa=kappa, band_tokens=int(band.sum()), band_acc=float(ok[band].mean()) if band.any() else None)
    return res


def main():
    out = {"w30_8192": {}, "w62_16384": {}, "slope": float(get_slopes(64)[39])}
    t0 = time.time()
    for L in (4, 7):
        out["w30_8192"][str(L)] = chain(8192, 4, L, 30, np.array([np.sqrt(8192), 8192.0]), 3.0 * 8192)
        print("w30_8192", L, out["w30_8192"][str(L)], f"{time.time() - t0:.0f}s", flush=True)
    for L, kappa in ((4, 8.5), (7, 10.0)):
        out["w62_16384"][str(L)] = chain(16384, 2, L, 62, np.array([2048.0]), 2e7, out["slope"], kappa)
        print("w62_16384", L, out["w62_16384"][str(L)], f"{time.time() - t0:.0f}s", flush=True)
    (HERE / "kp_trap.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
