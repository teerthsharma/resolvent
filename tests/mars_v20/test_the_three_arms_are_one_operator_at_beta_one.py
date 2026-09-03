"""MARS, CEQ v20 R15 it.1 -- the OPERATOR-level check the round's wing list
rests on, run rather than assumed.

The v17-K tournament names three arms (`results/v17k_r4_retake.jsonl` header:
`"arms": ["arm_pl", "arm_smprime", "softmax"]`), and `ceq/arm_phase.py` ships a
fourth operator module. This suite measures whether they are distinct
PRIMITIVES or one primitive under four parametrizations.

The algebra, then the measurement:

    arm_pl.operator     = softmax_j( w_ij + s_j - cumsum(g)_j )
    arm_phase.operator  = softmax_j( w_ij + s_j - cumsum(log m)_j ) * e^{i(Phi_i-Phi_j)}
    arm_smprime.operator= G_ij e^{qk w_ij} / Z_i^beta,  G = prod_{k>j}^{i} m_k e^{i th_k}

At `beta = qk = g = 1` the S-M' quotient telescopes: `exp(R_i)` cancels between
numerator and normalizer, leaving `exp(w_ij - R_j) / sum_j' exp(w_ij' - R_j')`
with `R = cumsum(log m)` -- which is ARM PHASE at `s = 0`, and ARM PL at
`s = 0, g = log m`.

TOLERANCE. `1e-12` in float64. That is ~100x ABOVE the tree's own worst float64
identity-bind residual on this journal (`9.547918011776346e-15`, the `arm_pl`
identity row of `results/v17k_r4_retake.jsonl`), so it cannot be accused of
being tuned to admit a reading. Every case below is measured, and the control
shows what a real difference reads at the same shapes.

Nothing here trains. No module is constructed; the exported operator functions
are called directly at matched settings.
"""
from __future__ import annotations

import math

import torch

from ceq import arm_phase as ph
from ceq import arm_pl as pl
from ceq import arm_smprime as smp

#: 100x the worst float64 identity-bind residual the deciding journal carries.
TOL = 1e-12
TREE_WORST_IDENTITY_RESIDUAL = 9.547918011776346e-15

SHAPES = [(8, 4, 15), (32, 6, 3)]        # two shapes: not one lucky input


def _bed(s, d, seed):
    """Strictly-positive magnitudes, free phases. `log m` finite everywhere, so
    the prefix-scan and path-product routes are both defined."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(s, d, generator=g, dtype=torch.float64)
    k = torch.randn(s, d, generator=g, dtype=torch.float64)
    m = torch.rand(s, generator=g, dtype=torch.float64) * 0.9 + 0.05
    th = (torch.rand(s, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
    return q, k, m, th


def _report(name, s, d, seed, delta):
    print(f"  [{name}] s={s} d={d} seed={seed}  max|delta| = {delta:.6e}")


# ------------------------------------------------------------------ CONTROL

def test_control_a_genuine_dial_move_is_visible_at_these_shapes():
    """Move ONE switch off its corner. If this reads at the noise floor too,
    the comparison below is blind and proves nothing."""
    worst = 0.0
    for s, d, seed in SHAPES:
        q, k, m, th = _bed(s, d, seed)
        a = smp.operator(q, k, m, th, beta=1.0, qk=1.0, g=1.0)
        b = smp.operator(q, k, m, th, beta=0.9, qk=1.0, g=1.0)
        delta = float((a - b).abs().max())
        _report("control beta 1.0 vs 0.9", s, d, seed, delta)
        worst = max(worst, delta)
    assert worst > 1e-3, worst


# ------------------------------------------------------------------- STRIKE

def test_arm_phase_is_arm_smprime_at_the_beta_one_corner():
    """ARM PHASE at `s = 0` against ARM S-M' at `beta = qk = g = 1`, BOTH of
    S-M''s routes. Distinct primitives must separate above TOL."""
    for s, d, seed in SHAPES:
        q, k, m, th = _bed(s, d, seed)
        z = torch.zeros(s, dtype=torch.float64)
        a = ph.operator(q, k, m, th, z)
        for route in smp.ROUTES:
            b = smp.operator(q, k, m, th, beta=1.0, qk=1.0, g=1.0, route=route)
            delta = float((a - b).abs().max())
            _report(f"arm_phase vs arm_smprime[{route}]", s, d, seed, delta)
            assert delta > TOL, (
                f"arm_phase(s=0) and arm_smprime(beta=1,route={route}) agree to "
                f"{delta:.6e} at s={s},d={d},seed={seed} -- below TOL={TOL:.0e}, "
                f"which is itself {TOL / TREE_WORST_IDENTITY_RESIDUAL:.0f}x the "
                f"tree's worst float64 identity residual. One operator, two names.")


def test_arm_pl_is_arm_smprime_at_the_beta_one_phase_free_corner():
    """ARM PL at `s = 0, g = log m` against ARM S-M' at `theta = 0,
    beta = qk = g = 1`. Distinct primitives must separate above TOL."""
    for s, d, seed in SHAPES:
        q, k, m, _th = _bed(s, d, seed)
        z = torch.zeros(s, dtype=torch.float64)
        a = pl.operator(q, k, torch.log(m), z)
        b = smp.operator(q, k, m, z, beta=1.0, qk=1.0, g=1.0).real
        delta = float((a - b).abs().max())
        _report("arm_pl vs arm_smprime[product,theta=0]", s, d, seed, delta)
        assert delta > TOL, (
            f"arm_pl(g=log m, s=0) and arm_smprime(theta=0, beta=1) agree to "
            f"{delta:.6e} at s={s},d={d},seed={seed} -- below TOL={TOL:.0e}. "
            f"One operator, two names.")


def test_the_two_smprime_routes_are_two_primitives_off_the_zero_set():
    """`ROUTES = ("product", "exp_scan")`, and `exp_scan` is the module's
    shipped PLANTED NEGATIVE for BIND 1. On a draw with no exactly-zero
    magnitude the two must still be distinguishable, or the negative's
    rejection region is the zero set alone."""
    for s, d, seed in SHAPES:
        q, k, m, th = _bed(s, d, seed)
        a = smp.operator(q, k, m, th, beta=1.0, qk=1.0, g=1.0, route="product")
        b = smp.operator(q, k, m, th, beta=1.0, qk=1.0, g=1.0, route="exp_scan")
        delta = float((a - b).abs().max())
        _report("smprime product vs exp_scan", s, d, seed, delta)
        assert delta > TOL, (
            f"the shipped route and its planted negative agree to {delta:.6e} "
            f"at s={s},d={d},seed={seed}; they separate only where some m_k == 0")


def test_the_exp_scan_planted_negative_has_a_nonempty_region_on_every_deciding_cell():
    """The two halves joined: the routes separate ONLY on the exactly-zero set
    (measured above), and one deciding cell reports that set is EMPTY.

    `results/v17k_r4_retake.jsonl`, `arm_smprime:t2:n2048:seed2`, manifest
    `smp_values.n_zero_gates = 0`. On that cell the forbidden route and the
    shipped route are the same map, so BIND 1's rejection region there is empty
    -- `MISTAKES.md` V-24.
    """
    import json
    import pathlib
    j = pathlib.Path(__file__).resolve().parents[2] / "results" / "v17k_r4_retake.jsonl"
    rows = [json.loads(l) for l in j.read_text().splitlines() if l.strip()]
    cells = {r["cell"]: r["manifest"]["smp_values"]["n_zero_gates"]
             for r in rows if r.get("kind") == "arm_smprime" and r.get("cell")}
    empty = {c: n for c, n in cells.items() if n == 0}
    assert not empty, (
        f"{len(empty)}/{len(cells)} deciding cells carry NO annihilating gate "
        f"{empty}; the two routes agree to 3.14e-16 off the zero set, so on "
        f"these cells the planted negative cannot fire")
