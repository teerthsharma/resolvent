"""A banded sgate whose band may be DILATED, plus the composed sign-flip probe.

WHY THIS FILE EXISTS. `ceq.bench._causal_mask` builds a CONTIGUOUS band,
`i - w <= j < i`. A contiguous band of width `w` composed over `L` layers reaches
`L * w`, so global reach at context `s` costs `L = s / w` layers -- LINEAR depth.
A band that keeps `w` entries but spaces them `d` apart reaches `w * d` in one
layer at the SAME row width, so a schedule `d_l = 2^l` reaches `s` at
`L = log2(s)` layers -- LOGARITHMIC depth. In both cases the row normalizer sums
over at most `w` terms and NEVER over `s`, which is the quantity
`DONE_ARCHIVE_ROUND1.md` measured as the cause of the decay.

THE BIND. `sgate_masked` is a reimplementation, and this repository's costliest
defect (instrument #17) was a correct measurement of an operator that ships
nowhere. So `dilation = 1` is asserted BITWISE against the shipped
`ceq.bench._causal_sgate_operator(..., window=w)` in
`test_composition_is_the_uncosted_route.py::test_the_dilated_arm_is_the_shipped_operator_at_dilation_one`.
Nothing here is believed until that passes.
"""
from __future__ import annotations

import math

import torch

from ceq import bench


def dilated_mask(s: int, device, window: int, dilation: int) -> torch.Tensor:
    """`j < i`, `i - j <= window * dilation`, `(i - j) % dilation == 0`.

    At `dilation = 1` this is exactly `bench._causal_mask(s, device, window)`:
    every offset is a multiple of 1, so the modulus clause is vacuous and the
    reach clause is `i - j <= window`. Asserted bitwise, not argued.
    """
    off = (torch.arange(s, device=device)[:, None]
           - torch.arange(s, device=device)[None, :])
    return (off > 0) & (off <= window * dilation) & (off % dilation == 0)


def sgate_masked(q: torch.Tensor, k: torch.Tensor, m: torch.Tensor,
                 rho: float = 1.5, lam: float = 0.10) -> torch.Tensor:
    """`bench._causal_sgate_operator` with the mask supplied instead of derived.

    Copied line for line from `ceq/bench.py::_causal_sgate_operator`; the only
    edit is that `m` is a parameter rather than a call to `_causal_mask`.
    """
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    pm = torch.softmax((-w).masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def composed_draws(*, n_draws: int, s: int, d: int, i: int, j: int, c: int,
                   dilations, window: int = 8, hops: int = 2, seed: int = 0,
                   rho: float = 1.5, lam: float = 0.10, device=None) -> list:
    """`bench.sign_flip_draws` with a PER-LAYER dilation schedule.

    The draw loop, the random-stream order (`wq`, `wk`, `wo`, then `x0`, `v0`,
    then the two `c` values), the `hops` path sum `h + A h + A^2 h`, the `w_o`
    postfactor and the `d(out_i)/d(v_j)` readout are all the same as
    `bench.sign_flip_draws` at `wrt="v"`. `depth` is `len(dilations)`, and layer
    `l` uses `dilated_mask(s, dev, window, dilations[l])`.

    `dilations = [1] * L` is the CONTIGUOUS composition, and at `L = 1` the whole
    function is `bench.sign_flip_draws(..., depth=1, window=w)` -- bound bitwise
    by `test_the_composed_probe_is_the_shipped_probe_at_depth_one`.
    """
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    depth = len(dilations)
    masks = [dilated_mask(s, dev, window, dl) for dl in dilations]

    pairs = []
    for _ in range(n_draws):
        wq = [rnd(d, d) for _ in range(depth)]
        wk = [rnd(d, d) for _ in range(depth)]
        wo = [rnd(d, d) for _ in range(depth)]
        x0, v0 = rnd(s, d), rnd(s, d)
        grads = []
        for c_val in (rnd(d), rnd(d)):
            x = x0.clone()
            x[c] = c_val
            v = v0.clone().requires_grad_(True)
            h = v
            for layer in range(depth):
                a = sgate_masked(x @ wq[layer], x @ wk[layer], masks[layer],
                                 rho=rho, lam=lam)
                acc, term = h, h
                for _ in range(hops):
                    term = a @ term
                    acc = acc + term
                h = acc @ wo[layer]
            if not h.requires_grad:
                grads.append(0.0)
                continue
            grad, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            grads.append(0.0 if grad is None else float(grad[j].sum()))
        pairs.append((grads[0], grads[1]))
    return pairs


def reach_fraction(draws) -> float:
    """Share of draws whose gradient is nonzero at BOTH values of `c`.

    A bounded receptive field that does not reach `j` returns exact `0.0` from
    `torch.autograd.grad` -- the edge is absent from the graph, not small. So
    this is the REACH measurement, and it is separate from the flip rate on
    purpose: a zero flip rate with reach 1.0 means something different from a
    zero flip rate with reach 0.0, and one number cannot say which.
    """
    if not draws:
        return 0.0
    return sum(1 for a, b in draws if a != 0.0 and b != 0.0) / len(draws)


def log_schedule(s: int, window: int = 8, hops: int = 2) -> list:
    """Doubling dilations `1, 2, 4, ...`, enough layers to reach `s`.

    Layer `l` displaces at most `hops * window * 2^l`, so `L` layers reach
    `hops * window * (2^L - 1)`. Solve for the smallest `L` covering `s`.
    """
    reach, out, dl = 0, [], 1
    while reach < s:
        out.append(dl)
        reach += hops * window * dl
        dl *= 2
    return out
