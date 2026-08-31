"""GATE-0 item G0.1 -- the four identity binds of the v17-K contract, RE-MEASURED.

    G0.1  identity binds under AD-1: oracle gates => label <= 1e-6; softmax
          corner row sums 1.000000; |a| <= 1, zero exceedances;
          deterministic-mode assertion PASS.

NOTHING IS INHERITED. `V16_ARM_SMPRIME.md` (a) reports `5.919777e-16` at
`n=64,s=64` and `7.550528e-16` at `n=512,s=64` / `n=256,s=128`. Those are a
previous run's numbers on a previous tree and are cited here only so a reader
can diff them against what this file MEASURES. Every assertion below runs the
computation again.

NO NEW CONSTRUCTIONS (the round's first law). This file builds no arm, no
oracle and no mechanism. Every object it touches is `ceq/arm_smprime.py`'s or
`scale/negation_scope.py`'s, called as shipped. The only code written here is
counters, controls and the two comparators the binds are read with.

NOTHING TRAINS (L-LEAN). The one gradient in the file is an executability probe
under `use_deterministic_algorithms(True)` on cuda; no cell, no loss and no
checkpoint is kept, and no NRMSE is computed anywhere.

THE WEAKENING SWITCH, AND WHY IT EXISTS. `G01_WEAKEN=1` replaces each bind's
bar with a vacuous one and each exceedance counter with a naive one-sided,
nan-blind form. Under it the PLANTED NEGATIVES STOP FIRING and the
non-degeneracy tests go RED. That RED is the receipt this suite owes itself: a
bind that holds on the tree as it stands produces no failing run of its own, so
the instrument is shown to be capable of failing by deliberately blunting it and
watching the negatives walk through. 14 vacuous controls have already been
struck in this repo; this is the shape of the check that would have caught them.

    $ G01_WEAKEN=1 python -m pytest tests/gate0 -q     # RED, by construction
    $ python -m pytest tests/gate0 -q -s               # the real reading
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

from ceq import arm_smprime as smp
from scale import negation_scope as ns

ROOT = pathlib.Path(__file__).resolve().parents[2]
DT = torch.float64
CT = torch.complex128

#: BED-M's measured gate support (`V16_LEAN_DOMAIN.md` section 2). Re-asserted
#: on every draw below rather than assumed -- L-DOM.
BEDM_SUPPORT = (-1.0, 0.0, 1.0)

#: THE BARS. `WEAK` is the deliberate blunting described in the module
#: docstring; it is never on in a reported run.
WEAK = os.environ.get("G01_WEAKEN") == "1"
LABEL_BAR = 1e6 if WEAK else 1e-6           # G0.1's own bar
ROW_BAR = 1.0 if WEAK else 5e-7             # "1.000000" = rounds to six places
AMAX_BAR = 1e9 if WEAK else 1.0             # AD-1's closed upper endpoint
DET_BITWISE = not WEAK                      # cross-process byte equality


def _n_outside_unit(t: torch.Tensor) -> int:
    """Count of entries NOT in the closed interval `[0, 1]`.

    Written two-sided and NAN-SAFE on purpose. `nan >= 0` is `False`, so this
    form counts a `nan`; the naive `(t > 1).sum()` does not, and neither does it
    see a negative magnitude. `test_bind_c_the_counter_fires...` exhibits both
    holes by switching to the naive form under `G01_WEAKEN`.
    """
    if WEAK:
        return int((t > AMAX_BAR).sum())     # one-sided, nan-blind: the hole
    return int((~((t >= 0.0) & (t <= 1.0))).sum())


def _bedm(n: int, s: int, d: int = 24, t_star: int = 2, seed: int = 0):
    """BED-M `e3_t2` at float64: `(gates, drives, label)`.

    The corpus is built by `scale/negation_scope.py` and cast to float64, and
    the label is RECOMPUTED by the bed's own `equilibrium_oracle` at float64 --
    so what is measured is the arm's residual and not the corpus's float32
    storage.
    """
    x, _y32, head, p = ns.make_equilibrium_batch(n, s, d, t_star=t_star,
                                                 d_model=16, seed=seed)
    x = x.double()
    return (x[:, :, ns.CH_DRIVE], x[:, :, ns.CH_FLIP],
            ns.equilibrium_oracle(x, head, p))


def _label_residual(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor,
                    mutation: str = "none") -> float:
    """The (L) setting of `ceq/arm_smprime.py`, read against the BED's label.

    `beta = 0`, QK off, `(u, theta) = (|a|, arg a)`, `V = b` UNRESCALED --
    `#5a`'s path-product corner. No `log(1-m)` and no `1/(1-m)` is instantiated,
    which is what makes `m = 1` an ordinary point.

    `mutation` is threaded through `ceq/arm_smprime.py::mutate` as shipped, so
    the planted negatives are read on THE SAME comparator as the bind and not
    on one of their own. Nothing here is a new construction: `mutate`,
    `oracle_heads` and `readout` are all the module's.
    """
    u, th = smp.oracle_heads(a)
    u, th, v, beta, route = smp.mutate(u, th, b.to(CT), mutation)
    zero = torch.zeros(a.shape[0], a.shape[-1], 1, dtype=DT, device=a.device)
    out = smp.readout(zero, zero, v.unsqueeze(-1), u, th,
                      beta=beta, qk=0.0, route=route)[..., -1, 0]
    return float((out - y.to(CT)).abs().max())


# =========================================================== BIND 1 (label)
# oracle gates in => label residual <= 1e-6, on BED-M's REAL support.

@pytest.mark.parametrize("n,s", [(64, 64), (512, 64), (256, 128)])
def test_bind_a_oracle_gates_imply_the_label(n, s):
    """THE BIND. Re-measured at the three shapes `V16_ARM_SMPRIME.md` (a)
    reports, inheriting none of its numbers."""
    a, b, y = _bedm(n, s)

    #: L-DOM: the census, printed against the theorem's hypotheses, before the
    #: residual is read. A draw off `{-1,0,+1}` is a different bind.
    support = sorted(float(v) for v in torch.unique(a))
    assert support == list(BEDM_SUPPORT), support
    n_zero = int((a == 0).sum())
    assert n_zero > 0, "no annihilating gate in the draw -- clause 4 untested"

    #: NON-DEGENERACY OF THE TARGET. A residual against an all-zero or constant
    #: label is a PASS on a degenerate input and is void. The label must carry
    #: variance.
    assert float(y.abs().max()) > 0.0
    assert float(y.std()) > 0.1, float(y.std())

    r = _label_residual(a, b, y)
    print(f"  BIND A  n={n:4d} s={s:3d}  residual = {r:.6e}   bar {LABEL_BAR:.0e}"
          f"   support {support}  zeros {n_zero}/{a.numel()}"
          f"  y.std {float(y.std()):.6f}")
    assert r <= LABEL_BAR, r


def test_bind_a_the_residual_fires_on_a_perturbed_gate():
    """NON-DEGENERACY OF THE INSTRUMENT. The same comparator, on gates moved
    off the oracle by a known amount. If a `1e-3` shift still reads inside the
    bar, the bind's PASS half proves nothing.

    The label is NOT recomputed from the perturbed gates -- the bed's own
    `equilibrium_oracle` output is held fixed, which is what makes this a
    perturbation of the ARM against a fixed truth.
    """
    a, b, y = _bedm(64, 64)
    ladder = {}
    for eps in (1e-12, 1e-9, 1e-6, 1e-3):
        ap = a.clone()
        ap[:, -1] = ap[:, -1] + eps          # the last hop, which enters t*=2
        ladder[eps] = _label_residual(ap, b, y)
    print("  BIND A  perturbation ladder (gate + eps, label held): "
          + "  ".join(f"{e:.0e}->{v:.6e}" for e, v in ladder.items()))
    assert ladder[1e-12] <= LABEL_BAR                 # below the bar, as it must be
    assert not (ladder[1e-3] <= LABEL_BAR), ladder    # the instrument fires
    assert ladder[1e-3] > ladder[1e-12]               # and it is monotone in eps


@pytest.mark.parametrize("mutation", [m for m in smp.MUTATIONS if m != "none"])
def test_bind_a_planted_negatives_fire_on_the_real_corpus(mutation):
    """PLANTED NEGATIVES, on BED-M itself and not on a toy draw.

    `ceq/arm_smprime.py::mutate` ships four, each deleting exactly one part of
    the (L) setting. `exp_scan` is the route `no_prefix_scan_represents_a_zero_gate`
    forbids and it lives in the shipped module, so BIND 1's rejection region is
    occupied by real code (`MISTAKES.md` V-24).
    """
    a, b, y = _bedm(64, 64)
    honest = _label_residual(a, b, y, "none")
    r = _label_residual(a, b, y, mutation)
    print(f"  BIND A  negative {mutation:16s} residual = {r:.6e}"
          f"   (honest {honest:.6e})")
    assert honest <= LABEL_BAR, honest
    assert not (r <= LABEL_BAR), f"{mutation} did not fire: {r}"


def test_bind_a_the_bos_slot_is_excluded_and_is_exactly_b0():
    """THE REPAIR PINNED. `ceq/arm_smprime.py::label_cell` used to score EVERY
    position against `chain_label`, whose convention is `y_0 = 0`. The
    path-product read-out at position `0` is `G_00 b_0 = b_0`, and `G_00` is the
    EMPTY product -- no gate, no `m`, no `theta`, no route, no `beta` -- so that
    term is bitwise invariant under every mutation of the arm. A term no setting
    of the arm can move is not a reading of the arm.

    `smp.bedm_draw` sets `b[0] = 0.0`, which hid it. BED-M's own
    `make_equilibrium_batch` zeroes `b[s-1]` instead, so on the REAL corpus the
    old scorer read `max |b_0|` -- an O(1) residual with HONEST settings.

    Repaired in `V17_LABEL_CELL_REPAIR.md` by excluding the BOS slot and keeping
    it as its own record key `residual_bos`. This test now pins the repair from
    both sides: the scored residual is clean, and the excluded term is still
    reported and is still exactly `max |b_0|`.
    """
    a, b, y = _bedm(8, 64)
    rec = smp.label_cell(a, b, mutation="none")
    cell = float(rec["residual"])
    bos = float(rec["residual_bos"])
    b0 = float(b[:, 0].abs().max())
    bz = b.clone()
    bz[:, 0] = 0.0
    cell_zeroed = float(smp.label_cell(a, bz, mutation="none")["residual"])
    last = _label_residual(a, b, y, "none")
    print(f"  BIND A  label_cell on the real corpus = {cell:.6e}"
          f"   residual_bos = {bos:.6e}   max|b_0| = {b0:.6e}"
          f"   with b_0 zeroed = {cell_zeroed:.6e}"
          f"   last-position bind = {last:.6e}")
    # Non-degeneracy FIRST: on the real corpus b_0 is NOT zero, so the two
    # assertions below are separating a real O(1) term from a real 1e-16 one.
    # On `bedm_draw` this test would pass while proving nothing.
    assert b0 > 1e-3, f"b_0 is degenerate on this draw ({b0}); the test is vacuous"
    assert cell <= LABEL_BAR, cell                  # the scored residual is clean
    assert abs(bos - b0) <= 1e-12, (bos, b0)        # the excluded term IS b_0
    assert cell_zeroed <= LABEL_BAR, cell_zeroed    # and nothing else is wrong
    assert last <= LABEL_BAR, last


# ======================================================= BIND 2 (row sums)
# softmax corner: beta = 1, g = 0, QK on  =>  every row sums to 1.000000.

def _corner_inputs(n: int = 64, d: int = 8, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=DT),
            torch.randn(n, d, generator=g, dtype=DT),
            torch.rand(n, generator=g, dtype=DT),
            (torch.rand(n, generator=g, dtype=DT) * 2.0 - 1.0) * math.pi)


def test_bind_b_the_softmax_corner_rows_sum_to_one():
    """THE BIND, at `#5a`'s first corner exactly."""
    q, k, u, th = _corner_inputs()
    #: the corner is a SETTING OF THE SWITCHES, checked before it is used:
    #: `g = 0` must give `m = 1` and `theta_eff = 0` exactly, or this is not
    #: `#5a`'s `g == 0` corner at all.
    m0, th0 = smp.blend(u, th, 0.0)
    assert torch.equal(m0, torch.ones_like(m0))
    assert torch.equal(th0, torch.zeros_like(th0))

    w = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=0.0)
    assert float(w.imag.abs().max()) == 0.0        # g=0 => the row is real
    rows = w.real.sum(-1)
    dev = float((rows - 1.0).abs().max())
    n_bad = int((~((rows - 1.0).abs() <= ROW_BAR)).sum())
    n_exact = int((rows == 1.0).sum())

    #: NON-DEGENERACY. A row of equal weights sums to 1 for a reason that has
    #: nothing to do with the normalizer. The last row's spread says the check
    #: is being read on a real softmax and not on a constant.
    last = w.real[-1, :]
    spread = float(last.max() / last.min())
    assert spread > 10.0, spread

    print(f"  BIND B  rows {rows.numel()}  max|sum-1| = {dev:.6e}"
          f"   bitwise-exact rows {n_exact}/{rows.numel()}"
          f"   out-of-bar {n_bad}   last-row max/min {spread:.3f}"
          f"   (|W| row sum dev {float((w.abs().sum(-1) - 1.0).abs().max()):.6e}"
          f" -- vacuous, see the g-corner test)")
    assert n_bad == 0, (n_bad, dev)
    assert dev <= ROW_BAR, dev


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.999999])
def test_bind_b_the_row_sum_fires_off_the_corner(beta):
    """NON-DEGENERACY / PLANTED NEGATIVE. `beta` is the switch the corner is a
    setting of; move it and the row sum must leave `1.000000`. `0.999999` is
    the tightest of the three and is what says the bar is `5e-7` and not `1e-2`.
    """
    q, k, u, th = _corner_inputs()
    rows = smp.operator(q, k, u, th, beta=beta, qk=1.0, g=0.0).real.sum(-1)
    dev = float((rows - 1.0).abs().max())
    print(f"  BIND B  negative beta={beta:<9} max|sum-1| = {dev:.6e}")
    assert not (dev <= ROW_BAR), f"beta={beta} did not fire: {dev}"


def test_bind_b_the_row_sum_fires_off_the_g_corner_only_on_the_real_part():
    """NON-DEGENERACY / PLANTED NEGATIVE, second switch -- AND THE STATISTIC
    THAT CANNOT DO IT.

    `V16_ARM_SMPRIME.md` (e) reads the corner as `Sigma_j |W_ij| = 1.000000` at
    `beta = 1`. That statistic is VACUOUS for the `g` switch and this test is
    what says so: `|W_ij| = R_ij e_ij / Z_i` and `Z_i = Sigma_j R_ij e_ij` BY
    CONSTRUCTION, so the absolute row sum is exactly `1` at `beta = 1` for
    EVERY setting of `g` and every gate whatsoever. It certifies the `beta`
    switch and nothing else.

    The row sum that carries the corner is the one over `Re W`, which is what a
    softmax row sum means: at `g = 1` the phases rotate the terms and the real
    row sum leaves `1`. Both numbers are measured here and the vacuous one is
    asserted VACUOUS, so no later node can read it as a corner certificate.
    """
    q, k, u, th = _corner_inputs()
    w = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=1.0)
    dev_abs = float((w.abs().sum(-1) - 1.0).abs().max())
    dev_real = float((w.real.sum(-1) - 1.0).abs().max())
    print(f"  BIND B  negative g=1.0   max|sum|W|-1| = {dev_abs:.6e}  (VACUOUS:"
          f" an identity at beta=1)   max|sum Re W -1| = {dev_real:.6e}")
    #: the vacuous statistic, pinned as vacuous rather than quietly dropped.
    assert dev_abs <= 1e-12, dev_abs
    #: the statistic that actually separates the corner.
    assert not (dev_real <= ROW_BAR), f"g=1 did not fire: {dev_real}"


# ========================================================= BIND 3 (|a| <= 1)
# AD-1's CLOSED [0,1], zero exceedances, both endpoints on the draw.

#: A sweep that is adversarial about the cap rather than a sample near it: the
#: closed endpoints exactly, one ulp either side of each, and values that put
#: the un-capped blend far outside on both sides at large `|g|`.
_U_SWEEP = (-1e12, -5.0, -1.0, -1e-16, 0.0, 1e-16, 0.5,
            1.0 - 1e-16, 1.0, 1.0 + 1e-16, 2.0, 5.0, 1e12)
_G_SWEEP = (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 1e6)


def test_bind_c_the_magnitude_never_leaves_the_closed_interval():
    """THE BIND. `m = clamp(lerp(1, u, g), 0, 1)` over the sweep, plus the hop
    modulus row it induces."""
    us = torch.tensor(_U_SWEEP, dtype=DT)
    exceed = raw_exceed = 0
    lo, hi = math.inf, -math.inf
    for gv in _G_SWEEP:
        m, _ = smp.blend(us, torch.zeros_like(us), gv)
        exceed += _n_outside_unit(m)
        raw_exceed += _n_outside_unit(torch.lerp(torch.ones_like(us), us, gv))
        lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
        #: clause 1's right-hand side: a product of things in [0,1] is in [0,1],
        #: so the HOP inherits the cap and not only the per-position magnitude.
        exceed += _n_outside_unit(smp.path_product(m))

    #: NON-DEGENERACY. The counter must be counting something that could have
    #: been non-zero: the SAME sweep, un-capped, leaves the interval.
    assert raw_exceed > 0, "the sweep never left [0,1] uncapped -- vacuous"
    #: BOTH CLOSED ENDPOINTS ARE ATTAINED VALUES, which is the whole of AD-1.
    assert lo == 0.0 and hi == 1.0, (lo, hi)

    print(f"  BIND C  sweep {len(_U_SWEEP)}x{len(_G_SWEEP)}  exceedances = {exceed}"
          f"   uncapped-control exceedances = {raw_exceed}"
          f"   m range [{lo}, {hi}]")
    assert exceed == 0, exceed
    assert hi <= AMAX_BAR


def test_bind_c_the_endpoints_are_on_bed_m_itself():
    """The two closed endpoints are not a sweep artefact: BED-M's own gates are
    `{-1, 0, +1}`, so `|a| in {0, 1}` and BOTH endpoints are on the corpus."""
    a, _b, _y = _bedm(64, 64)
    u, _th = smp.oracle_heads(a)
    n_zero, n_one = int((u == 0.0).sum()), int((u == 1.0).sum())
    print(f"  BIND C  BED-M |a|: min {float(u.min())}  max {float(u.max())}"
          f"   at 0: {n_zero}   at 1: {n_one}   exceedances "
          f"{_n_outside_unit(u)}")
    assert n_zero > 0 and n_one > 0
    assert _n_outside_unit(u) == 0
    assert float(u.max()) <= AMAX_BAR


@pytest.mark.parametrize("seed", range(8))
def test_bind_c_the_shipped_module_respects_the_cap(seed):
    """`a_hat_max` on the shipped `ArmSMPrime`, as-constructed, on random data
    -- the reading R1's three divergent seeds put at `285` with an OPEN range.
    Untrained; nothing is kept (L-LEAN)."""
    torch.manual_seed(seed)
    arm = smp.ArmSMPrime(16, d_model=8).to(DT)
    g = torch.Generator().manual_seed(seed + 100)
    x = torch.randn(4, 16, 8, generator=g, dtype=DT)
    with torch.no_grad():
        m, _ = smp.blend(*arm.heads(x), arm.g)
    print(f"  BIND C  ArmSMPrime seed {seed}  a_hat_max = {float(m.max()):.6f}"
          f"   min {float(m.min()):.6f}   exceedances {_n_outside_unit(m)}")
    assert _n_outside_unit(m) == 0
    assert float(m.max()) <= AMAX_BAR


def test_bind_c_the_counter_fires_on_the_wrong_cap_order_and_on_nan():
    """NON-DEGENERACY / PLANTED NEGATIVE, two holes at once.

    (i) `ceq/arm_smprime.py::blend`'s docstring states the cap is applied AFTER
    the blend and that the other order lets `g > 1` extrapolate to a NEGATIVE
    magnitude. The rejected order is built here and the counter must see it.

    (ii) `clamp` propagates `nan`, and `nan > 1` is `False`, so a one-sided
    counter reads a `nan` magnitude as being inside the closed interval. The
    counter must see that too.
    """
    us = torch.tensor(_U_SWEEP, dtype=DT)
    ones = torch.ones_like(us)
    bad = torch.cat([torch.lerp(ones, torch.clamp(us, 0.0, 1.0), gv)
                     for gv in _G_SWEEP])           # cap FIRST -- rejected order
    n_bad = _n_outside_unit(bad)

    nanned = torch.tensor([0.5, float("nan"), 1.0], dtype=DT)
    n_nan = _n_outside_unit(smp.magnitude(nanned))

    print(f"  BIND C  negative cap-first-blend-second: exceedances = {n_bad}"
          f"   range [{float(bad.min()):.1f}, {float(bad.max()):.1f}]"
          f" | nan magnitude: counted = {n_nan}")
    assert n_bad > 0, f"the rejected cap order did not fire: {n_bad}"
    assert n_nan == 1, f"a nan magnitude read as inside [0,1]: {n_nan}"


# ================================================ BIND 4 (deterministic mode)
# use_deterministic_algorithms(True) + cuda: does arm_smprime RUN, and is it
# bitwise reproducible?  The single most load-bearing fact in G0.1.

_DET_PROBE = r'''
import hashlib, json, os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, sys.argv[1])
import torch
torch.use_deterministic_algorithms(True)
from ceq import arm_smprime as smp

out = {"torch": torch.__version__,
       "flag_live": bool(torch.are_deterministic_algorithms_enabled()),
       "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
       "cuda_available": bool(torch.cuda.is_available())}
if not out["cuda_available"]:
    print(json.dumps(out)); raise SystemExit(0)
dev = torch.device("cuda")
out["device_name"] = torch.cuda.get_device_name(0)

S = 64
a, b = smp.bedm_draw(seed=15, s=S, device=dev)

def _readout(aa, bb):
    u, th = smp.oracle_heads(aa)
    v = bb.to(aa.dtype)
    zq = torch.zeros(aa.shape[-1], 1, dtype=torch.float64, device=dev)
    return smp.readout(zq, zq, v.unsqueeze(-1), u, th, beta=0.0, qk=0.0)[..., 0]

def _sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

try:
    r1, r2 = _readout(a, b), _readout(a, b)
    out["readout_ran"] = True
    out["within_process_bitwise"] = bool(torch.equal(r1, r2))
    out["sha"] = _sha(r1)
    out["scalar"] = round(float(r1.abs().max()), 3)
    bp = b.clone()
    inf = torch.tensor(float("inf"), dtype=bp.dtype, device=dev)
    bp[1] = torch.nextafter(bp[1], inf)
    ru = _readout(a, bp)
    out["sha_one_ulp"] = _sha(ru)
    out["scalar_one_ulp"] = round(float(ru.abs().max()), 3)
except Exception as e:
    out["readout_ran"] = False
    out["readout_error"] = "%s: %s" % (type(e).__name__, e)

try:
    out["label_cell_residual"] = smp.label_cell(a, b, seed=15)["residual"]
    out["label_cell_ran"] = True
except Exception as e:
    out["label_cell_ran"] = False
    out["label_cell_error"] = "%s: %s" % (type(e).__name__, e)

# the shipped module: FORWARD alone, then forward+backward. Split, because a
# Kaggle run needs the second and the identity binds need only the first.
torch.manual_seed(0)
arm = smp.ArmSMPrime(16, d_model=8).to(torch.float64).to(dev)
x = torch.randn(4, 16, 8, dtype=torch.float64, device=dev)
try:
    with torch.no_grad():
        f1 = arm(x)
        f2 = arm(x)
    out["module_forward_nograd"] = "OK"
    out["module_forward_bitwise"] = bool(torch.equal(f1, f2))
    out["module_forward_sha"] = _sha(f1)
except Exception as e:
    out["module_forward_nograd"] = "%s: %s" % (type(e).__name__, e)
try:
    (arm(x) ** 2).mean().backward()
    out["module_fwd_bwd"] = "OK"
    out["module_grad_finite"] = all(bool(torch.isfinite(p.grad).all())
                                    for p in arm.parameters() if p.grad is not None)
except Exception as e:
    out["module_fwd_bwd"] = "%s: %s" % (type(e).__name__, e)

# the contrast the whole item turns on.
probes = (
    ("cumsum_f64", lambda: torch.cumsum(torch.randn(S, dtype=torch.float64, device=dev), 0)),
    ("cumprod_f64", lambda: torch.cumprod(torch.rand(S, dtype=torch.float64, device=dev), 0)),
    ("cumprod_c128", lambda: torch.cumprod(torch.randn(S, dtype=torch.complex128, device=dev), 0)),
    ("cumprod_backward", lambda: torch.cumprod(
        torch.rand(S, dtype=torch.float64, device=dev, requires_grad=True), 0).sum().backward()),
    ("path_product_f64", lambda: smp.path_product(torch.rand(S, dtype=torch.float64, device=dev))),
    ("path_product_c128", lambda: smp.path_product(torch.randn(S, dtype=torch.complex128, device=dev))),
    ("hop_scan", lambda: smp.hop_scan(torch.rand(S, dtype=torch.float64, device=dev),
                                      torch.zeros(S, dtype=torch.float64, device=dev))),
)
for name, fn in probes:
    try:
        fn()
        torch.cuda.synchronize()
        out[name] = "OK"
    except Exception as e:
        out[name] = "%s: %s" % (type(e).__name__, e)

# LAST, because it mutates the process-wide flag: does `warn_only=True` let the
# backward through, and is the gradient then reproducible in-process?
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    arm2 = smp.ArmSMPrime(16, d_model=8).to(torch.float64).to(dev)
    x2 = torch.randn(4, 16, 8, dtype=torch.float64, device=dev)
    grads = []
    for _ in range(2):
        arm2.zero_grad()
        (arm2(x2) ** 2).mean().backward()
        grads.append(torch.cat([p.grad.reshape(-1).clone()
                                for p in arm2.parameters() if p.grad is not None]))
    out["warn_only_fwd_bwd"] = "OK"
    out["warn_only_grad_bitwise"] = bool(torch.equal(grads[0], grads[1]))
    out["warn_only_grad_finite"] = bool(torch.isfinite(grads[0]).all())
    out["warn_only_grad_sha"] = _sha(grads[0])
except Exception as e:
    out["warn_only_fwd_bwd"] = "%s: %s" % (type(e).__name__, e)

print(json.dumps(out))
'''


def _run_det_probe(tmp_path, tag: str) -> dict:
    p = tmp_path / f"det_probe_{tag}.py"
    p.write_text(_DET_PROBE, encoding="utf-8")
    r = subprocess.run([sys.executable, str(p), str(ROOT)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_bind_d_deterministic_mode_on_cuda(tmp_path):
    """THE BIND, and the fact G0.1 turns on.

    `V16_R1_DEVICE_READY.md` documents `cumsum_cuda_kernel` having no
    deterministic implementation in torch 2.5.1, which makes `arm_phase` and
    `arm_pl` RAISE under the flag. `arm_smprime` uses `cumprod`. This measures
    whether the same hole is there, in a SUBPROCESS -- the flag is global, four
    other agents share this tree, and `CUBLAS_WORKSPACE_CONFIG` has to be set
    before torch initialises.

    Run TWICE, in two processes, so the equality claim is cross-process and not
    a cached tensor compared with itself.
    """
    first = _run_det_probe(tmp_path, "a")
    if not first["cuda_available"]:
        pytest.fail("BLOCKED: no cuda visible to the probe; the bind is not "
                    "decidable on this box right now")
    second = _run_det_probe(tmp_path, "b")

    print("  BIND D  " + json.dumps(first, indent=2).replace("\n", "\n          "))
    print(f"  BIND D  second process sha = {second.get('sha')}")

    #: NON-DEGENERACY OF THE REGIME. If the flag were not live, every equality
    #: below would hold for a reason that has nothing to do with determinism.
    #: `cumsum` raising is the proof the regime is on.
    assert first["flag_live"] is True
    assert "does not have a deterministic implementation" in first["cumsum_f64"], \
        first["cumsum_f64"]

    assert first["readout_ran"] is True, first.get("readout_error")
    assert first["within_process_bitwise"] is True
    assert first["label_cell_ran"] is True, first.get("label_cell_error")

    #: the shipped module's FORWARD, which is what the identity binds need.
    assert first["module_forward_nograd"] == "OK", first["module_forward_nograd"]
    assert first["module_forward_bitwise"] is True
    assert first["module_forward_sha"] == second["module_forward_sha"]

    #: PINNED FINDING, not part of the bind. `cumprod`'s BACKWARD calls
    #: `cumsum_cuda_kernel`, so the arm is deterministic to EVALUATE on cuda and
    #: not deterministic to TRAIN there. Pinned by exact kernel name so a torch
    #: bump that closes the hole shows up as a failing test rather than as
    #: nobody noticing.
    assert "cumsum_cuda_kernel" in first["cumprod_backward"], first["cumprod_backward"]
    assert "cumsum_cuda_kernel" in first["module_fwd_bwd"], first["module_fwd_bwd"]

    #: and what the escape hatch buys, measured rather than assumed: with the
    #: refusal demoted to a warning the SAME arithmetic runs. Cross-process
    #: gradient equality says whether that demotion costs reproducibility.
    assert first["warn_only_fwd_bwd"] == "OK", first["warn_only_fwd_bwd"]
    assert first["warn_only_grad_bitwise"] is True
    assert first["warn_only_grad_finite"] is True
    assert first["warn_only_grad_sha"] == second["warn_only_grad_sha"], \
        (first["warn_only_grad_sha"], second["warn_only_grad_sha"])

    #: NON-DEGENERACY OF THE COMPARATOR. A byte equality that cannot fail
    #: proves nothing: ONE ULP on a single drive must change it. Under
    #: `G01_WEAKEN` the comparator is a 3-decimal scalar summary instead, which
    #: is blind to that ulp -- and that is the RED.
    key = (lambda rec, suf="": rec["sha" + suf]) if DET_BITWISE \
        else (lambda rec, suf="": rec["scalar" + suf])
    assert key(first) != key(first, "_one_ulp"), "the comparator is blind"
    assert key(first) == key(second), (key(first), key(second))
