"""Does the parity ratio hold as the model grows, or is 3.3M an artifact?

The user's standard: parity at 3.3M is FALSE unless it survives at 300M. A ratio
that holds small and widens with size is an artifact, and this project has already
seen exactly that pathology once -- the L1 operator went 1.2191x at 250 steps to
1.337x at 600.

Self-contained: builds and trains models directly rather than through
`lm.train_one`, so it does not collide with concurrent edits to `ceq/lm.py`.
"""
from __future__ import annotations
import statistics, sys, time, pathlib
import torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import lm

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "tinystories_20k.txt"
SEEDS = (0, 1, 2)
STEPS, BS, LR = 600, 32, 1e-3


def train(kind, corpus, d, layers, heads, seq, seed, device):
    torch.manual_seed(seed)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=seed).to(device)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    m.train()
    for _ in range(STEPS):
        x, y = corpus.batch("train", BS, seq, g, device)
        loss = F.cross_entropy(m(x).reshape(-1, lm.VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    vg = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        v = float(torch.stack([F.cross_entropy(m(a).reshape(-1, lm.VOCAB), b.reshape(-1))
                               for a, b in (corpus.batch("val", BS, seq, vg, device)
                                            for _ in range(20))]).mean())
    return v, m.n_params()


def main():
    corpus = lm.ByteCorpus(CORPUS.read_text(encoding="utf-8")[:4_000_000])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.RHO, lm.SGATE_LAM, lm.HOPS = 1.5, 0.10, 2
    print(f"device={dev}  lr={LR}  steps={STEPS}  seeds={SEEDS}  rho=1.5 lam=0.10 hops=2", flush=True)
    print(f"{'d':>5} {'L':>3} {'H':>3} {'params':>11} {'softmax':>9} {'sgate':>9} "
          f"{'ratio':>8} {'spread':>17} {'s/run':>7}", flush=True)
    for d, layers, heads, seq in ((128, 2, 4, 128), (256, 4, 4, 128),
                                  (384, 6, 6, 128), (512, 8, 8, 128)):
        row = {}
        for kind in ("softmax", "sgate"):
            t0 = time.perf_counter()
            out = [train(kind, corpus, d, layers, heads, seq, s, dev) for s in SEEDS]
            row[kind] = ([v for v, _ in out], out[0][1], (time.perf_counter() - t0) / len(SEEDS))
        sm, npar, _ = row["softmax"]; sg, _, dt = row["sgate"]
        rs = [a / b for a, b in zip(sg, sm)]
        print(f"{d:>5} {layers:>3} {heads:>3} {npar:>11,} {statistics.median(sm):>9.4f} "
              f"{statistics.median(sg):>9.4f} {statistics.median(rs):>8.4f}   "
              f"{min(rs):.4f}-{max(rs):.4f} {dt:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
