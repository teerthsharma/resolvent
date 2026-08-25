"""ARM A — the torque probe. Pure measurement. Nothing is built until this survives.

THE GEOMETRY IS TEXTBOOK AND THAT IS STATED, NOT HIDDEN. The square-root map
`φ(p) = √p` carries the simplex isometrically onto the positive orthant of the
UNIT SPHERE, and by Čencov the Fisher–Rao metric is the unique invariant choice.
Both facts are classical information geometry [CITED, G1 r5 iter 1]. They are a
REASON to use this metric, never a claim this project may make.

WHAT IS BEING TESTED is whether the angle buys anything the existing probe does
not. G1 also found the TV ablation probe OCCUPIED as interpretability practice —
"ablate a component, measure how the attention distribution changes" is standard.
So K3 is decisive: if θ cannot beat raw TV on identical draws, this round is
classical geometry pointed at an existing probe, and TV ships instead.

    X = √A(·|c)      Y = √A(·|∅)        rows are unit vectors: Σ_j A_ij = 1
    θ_i = arccos Σ_j √(A_ij^c · A_ij^∅)      the Bhattacharyya angle
    D_FR = E_i[θ_i]
    τ = ½(YᵀX − XᵀY)                          equilibrium ⇔ τ = 0
    ξ_i = (θ_i / sin θ_i)(y_i − cos θ_i · x_i)      log map
    γ_r = σ_r − σ_{r+1}                        MANDATORY printed diagnostic

THREE KILLS, pre-registered:
  K1 flip(s) slope ≤ −0.4 on X₄ AND D_FR slope ≥ −0.1 on the SAME draws.
     D_FR slope < −0.3 ⇒ displacement dies with flip ⇒ "no leap".
  K2 D_FR(causal c) vs D_FR(filler c) must separate with DISJOINT CIs. A
     statistic that cannot lose to a filler is M2-clause-2 again — the control
     that was zero by construction — and it VOIDS the table.
  K3 θ must separate causal-from-filler with a LARGER standardized effect than
     raw TV on identical draws.

CARPET DISCIPLINE, contractual: `c` and `j` are drawn UNIFORMLY AT RANDOM and
never placed on any schedule. Two rounds broke this and both produced artifacts —
a lattice-aligned `c` gave a flat reading that was placement, not physics, and a
fixed `j = s//4` made every schedule read severance 1.0000 because the offset was
unreachable.

G8: no multiplication inside any sign decision. The flip half reads through the
X₄ valuation instrument, which compares sign FIELDS.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots, clopper_pearson     # noqa: E402
from scale.valuation import valuation, v_opposite_signs          # noqa: E402

EPS = 1e-12


def rows_with_and_without(q, k, c: int, *, window: int = 0):
    """(A^c, A^∅) — the causal softmax rows with token c present and masked.

    Masking sets c's logit to −inf and RENORMALISES, which is the whole point:
    on a simplex a removed token's mass is REALLOCATED, never deleted. Both are
    returned on the full index set with A^∅[:, c] = 0, so the Bhattacharyya sum
    simply drops that term.
    """
    a_c = bench._softmax_operator(q, k, window=window)
    s = q.shape[-2]
    m = bench._causal_mask(s, q.device, window).clone()
    m[:, c] = False
    w = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    neg = torch.finfo(q.dtype).min
    a_0 = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    return a_c, a_0


def theta_rows(a_c: torch.Tensor, a_0: torch.Tensor) -> torch.Tensor:
    """Per-row Bhattacharyya angle. Rows with no mass either side give 0."""
    bc = (a_c.clamp_min(0).sqrt() * a_0.clamp_min(0).sqrt()).sum(-1)
    return torch.arccos(bc.clamp(-1.0, 1.0))


def tv_rows(a_c: torch.Tensor, a_0: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a_c - a_0).abs().sum(-1)


def torque(a_c: torch.Tensor, a_0: torch.Tensor) -> float:
    x, y = a_c.clamp_min(0).sqrt(), a_0.clamp_min(0).sqrt()
    yx = y.transpose(-2, -1) @ x
    return float((0.5 * (yx - yx.transpose(-2, -1))).norm())


def shadow(a_c: torch.Tensor, a_0: torch.Tensor):
    """(Ξ, singular values). ξ_i is the log map of y_i at x_i on the sphere."""
    x, y = a_c.clamp_min(0).sqrt(), a_0.clamp_min(0).sqrt()
    th = theta_rows(a_c, a_0)
    st = torch.sin(th).clamp_min(EPS)
    xi = (th / st).unsqueeze(-1) * (y - torch.cos(th).unsqueeze(-1) * x)
    return xi, torch.linalg.svdvals(xi)


def one_draw(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    """One draw. `c` and `j` UNIFORM AT RANDOM; `filler` picks c OUTSIDE P."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = set(int(p) for p in select_pivots(kk, min(k, s - 2), exclude=(i, j)))
    pool = sorted(piv) if not filler else [t for t in range(1, i) if t not in piv and t != j]
    if not pool:
        return None
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
    a_c, a_0 = rows_with_and_without(q, kk, c)
    th = theta_rows(a_c, a_0)
    tv = tv_rows(a_c, a_0)
    return dict(theta=float(th.mean()), tv=float(tv.mean()),
                tau=torque(a_c, a_0), i=i, j=j, c=c, a_c=a_c, a_0=a_0)


def cohen_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / max(1, na - 1)
    vb = sum((x - mb) ** 2 for x in b) / max(1, nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    return (ma - mb) / sp if sp > 0 else float("nan")


def boot_ci(v: list[float], seed: int = 0, b: int = 2000):
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(v)
    n = t.numel()
    means = torch.stack([t[torch.randint(0, n, (n,), generator=g)].mean()
                         for _ in range(b)]).sort().values
    return float(means[int(0.025 * b)]), float(means[int(0.975 * b)])
