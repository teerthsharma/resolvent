# Foreman K1: sc-HPD W62 with 2-dim heads at n = 8192 / 16384, rotary period 2048 + the twin's ALiBi slope 2^-5.
# Bars: test_kp_w62a.py.
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
W, P, BETA = 62, 2048.0, 2e7
M = float(get_slopes(64)[39])


def realize_rows(A, depth, t2, rows, big=65536.0, chunk=128):
    n = len(A)
    v1 = A[rows]
    J = depth[rows] - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo = v1 - (a + W - 1)
    addr = np.concatenate([v1[None], lo[None] + np.arange(W)[:, None]], 0)       # (W+1, r)
    viol = int(((addr + P) <= rows[None, :]).sum())
    Q = rot(addr, np.array([P]), F32)
    Kc = rot(np.arange(n), np.array([P]), F32)
    V = torch.tensor(A, dtype=F32)
    out = torch.empty(W + 1, len(rows), dtype=F32)
    j = torch.arange(n, dtype=F32)
    rt = torch.tensor(rows, dtype=F32)
    for r0 in range(0, len(rows), chunk):
        t = rt[r0:r0 + chunk]
        lg = BETA * torch.einsum("hqd,kd->hqk", Q[:, r0:r0 + chunk], Kc) - M * (t[:, None] - j[None, :])[None]
        lg = lg.masked_fill((j[None, :] > t[:, None])[None], float("-inf"))
        out[:, r0:r0 + chunk] = torch.softmax(lg, dim=-1) @ V
    x2, s = out[0], out[1:]
    d = x2[None] - torch.tensor(lo, dtype=F32)[None] - torch.arange(W, dtype=F32)[:, None]
    bump = relu(d + 1) - 2 * relu(d) + relu(d - 1)
    new = relu(s - big * (1 - bump)).sum(0) + relu(x2 - big * bump.sum(0))
    return torch.round(new).numpy().astype(np.int64), viol


def main():
    out = {"slope": M, "period": P, "beta": BETA}
    t0 = time.time()
    for n in (16384, 8192):
        bs = [K.bed_kp.make_test(np.random.default_rng([22, n, k]), n) for k in range(4)]
        cal = [K.bed_kp.make_test(np.random.default_rng([23, n, k]), n) for k in range(8)]
        out[str(n)] = {}
        for L in (4, 7):
            tables = K.calibrate(cal, L, W, 0)
            bad = tot = viol = 0
            for k, b in enumerate(bs):
                p, dep = b["parent"], b["depth"]
                A = pointer0(p)
                for layer, (t2, t3) in enumerate(tables):
                    nxt = K.step(A, dep, W, 0, t2, t3)
                    rows = np.random.default_rng([62, n, k, layer]).choice(n, 2048, replace=False)
                    got, v = realize_rows(A, dep, t2, rows)
                    bad += int((got != nxt[rows]).sum())
                    tot += len(rows)
                    viol += v
                    A = nxt
            out[str(n)][str(L)] = {"mismatches": bad, "rows": tot, "rate": bad / tot, "alias_violations": viol}
            print(n, L, out[str(n)][str(L)], f"{time.time() - t0:.0f}s", flush=True)
    (HERE / "kp_w62a.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
