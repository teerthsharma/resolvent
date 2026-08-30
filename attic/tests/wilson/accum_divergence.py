"""Does gradient accumulation change the answer? Measure it; do not assert it.

The 8192-example run died for MEMORY, not time: the full-batch operator is
[8192,64,64], the hop-2 term another, autograd saves both, and the pivot arms
stack 8192 separate [64,64] tensors -- several GB per step.

MINIBATCHING IS NOT THE FIX. Minibatch SGD is a DIFFERENT OPTIMISER: it takes
`steps` noisy steps instead of `steps` exact ones, and it would change the
numbers. Under the standing policy that makes it A NEW ARM, not an optimisation,
and this project has published a new arm under an old name before.

ACCUMULATION IS THE FIX, and it is exact in exact arithmetic:

    mse(all)  =  (1/N) Σ_i r_i²  =  Σ_chunks (m/N) · mse(chunk)

so scaling each chunk's loss by `m/N` and summing the backward passes gives the
same gradient as one full-batch backward. **In FLOAT it is not the same**:
summing 8 partial reductions is a different association than one reduction over
8192 terms, and float addition is not associative.

So the equivalence claim is MEASURED here rather than asserted. That distinction
is the whole point -- "mathematically equivalent" is exactly the kind of sentence
this repository has been burned by, and a number is cheap.

READ IT THIS WAY, fixed before running:
  * divergence at or below float32 epsilon scale -> accumulation is an
    optimisation and may be used, with the measured number recorded beside it;
  * divergence materially larger -> it is a different arm and must be labelled
    one, or the chunk count must be reported as a parameter of the result.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scale.m3_capability import Arm, LR, D_MODEL          # noqa: E402
from scale.negation_scope import make_batch, nrmse        # noqa: E402


def train(kind: str, x, y, *, s: int, steps: int, seed: int, chunks: int = 1):
    """`run_arm`'s loop. `chunks=1` is the shipped full-batch path verbatim."""
    torch.manual_seed(seed)
    model = Arm(kind, s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu, sigma = float(y.mean()), float(y.std(unbiased=False)) or 1.0
    ystd = (y - mu) / sigma
    n = x.shape[0]
    size = (n + chunks - 1) // chunks
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        if chunks == 1:
            torch.nn.functional.mse_loss(model(x), ystd).backward()
        else:
            for a in range(0, n, size):
                xb, yb = x[a:a + size], ystd[a:a + size]
                # weight by the chunk's SHARE of the batch, so the accumulated
                # gradient equals the full-batch gradient in exact arithmetic
                (torch.nn.functional.mse_loss(model(xb), yb)
                 * (xb.shape[0] / n)).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x) * sigma + mu
    return model, nrmse(pred, y)


def main() -> int:
    torch.set_num_threads(2)
    s, d, n, steps, seed = 64, 24, 2048, 60, 0
    x, y, _f, _p = make_batch(n, s, d, d_model=D_MODEL, seed=seed)
    print("GRADIENT ACCUMULATION vs FULL BATCH -- same seed, same init, same data")
    print(f"s={s} d={d} n={n} steps={steps} seed={seed}\n")
    print(f"{'arm':<16} {'chunks':>7} {'train nrmse':>12} {'max |dparam|':>14} "
          f"{'max rel':>10}")
    for kind in ("softmax", "pivot_signed"):
        base, base_nrmse = train(kind, x, y, s=s, steps=steps, seed=seed, chunks=1)
        bp = torch.cat([p.detach().reshape(-1) for p in base.parameters()])
        print(f"{kind:<16} {1:>7} {base_nrmse:>12.6f} {'-':>14} {'-':>10}")
        for ch in (4, 8):
            m, nr = train(kind, x, y, s=s, steps=steps, seed=seed, chunks=ch)
            mp = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
            dab = float((mp - bp).abs().max())
            rel = float(((mp - bp).abs() / bp.abs().clamp_min(1e-12)).max())
            print(f"{kind:<16} {ch:>7} {nr:>12.6f} {dab:>14.3e} {rel:>10.3e}")
    print(f"\n  float32 eps = {torch.finfo(torch.float32).eps:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
