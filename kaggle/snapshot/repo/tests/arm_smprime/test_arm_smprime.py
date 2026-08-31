"""ARM S-M' -- the section S-M' hop, built as the PATH PRODUCT and not as a
prefix scan, with three binds and a planted negative on each.

    W_ij = G_ij * exp(qk * q_i.k_j) / Z_i^beta          j <= i
    G_ij = prod_{k=j+1}^{i} m_k e^{i theta_k}           the PATH PRODUCT
    Z_i  = sum_{j<=i} (prod_{k=j+1}^{i} m_k) * exp(...)

WHY NOT `exp(C_i - C_j)`. `lean/CEQ/V16Domain.lean`:

    no_prefix_scan_represents_a_zero_gate (C : N -> C) (m theta : N -> R) {i j}
        (hz : exists k in Ico (j+1) (i+1), m k = 0) :
      Complex.exp (C i - C j) != pathProd m theta i j

`Complex.exp` is never zero and the path product is. BED-M's gate is exactly `0`
on `126,976 / 131,072` entries and `0 of 2,048` sequences satisfy `forall k,
0 < m k` -- structurally, `scale/negation_scope.py:429` zeroes the first
`head + 1` positions of every row. So NO prefix scan represents BED-M's hop,
which is why `V15_ARM_PHASE.md` (e) reads `nan` on the band.

The re-stated `#2` (`prefix_logit_mask_restated`) defines `pathProd` DIRECTLY,
with no logarithm, and proves `m = 0 => product exactly 0` as clause 4 and
`|prod a| = prod m` as clause 1. This suite is the float64 shadow of those two
clauses and of `#5a`'s three corners. It implements a PROVED STATEMENT; it does
not invent a mechanism (`CEQ_V16_CONTRACT.md` KILLS, first law).

EVERY BIND SHIPS ITS REJECTION REGION (`MISTAKES.md` V-24). The `exp_scan`
route is the arm's own shipped mutilation, so the rejection region of BIND 1 is
occupied by the forbidden construction itself and not by a test-local copy.

NOTHING HERE TRAINS (L-LEAN). The one place a gradient appears is
`test_the_arm_completes_forty_gradient_steps`, which is a FINITENESS probe on
random data -- `V16_DEVICE_CERT.md` section 5.4 found `ceq/arm_phase.py` fails
that probe at step 0 from its own initialiser, and an initialiser that produces
`-inf` is an identity failure, not a training result. No cell, no corpus label,
no optimizer state is kept.
"""
from __future__ import annotations

import functools
import math
import os

import pytest
import torch

from ceq import arm_smprime as smp
from ceq import lm
from scale import negation_scope as ns

DT = torch.float64
CT = torch.complex128
S = 8                       # positions 0..S-1
SEED = 15                   # the fork probe's seed, so numbers are comparable

#: THE DEVICE THE WHOLE SUITE RUNS ON, same convention as
#: `tests/arm_phase/conftest.py`'s suite: `ARM_SMPRIME_DEVICE=cuda python -m
#: pytest tests/arm_smprime` re-runs the IDENTICAL suite with the identical
#: test ids, so the two runs diff line for line. Every draw is made on the HOST
#: and moved afterwards (`V16_BAR_RECERT.md` 6.1's construct-then-move rule), so
#: this changes the device and not the corpus.
DEV = torch.device(os.environ.get("ARM_SMPRIME_DEVICE", "cpu"))
ON_CUDA = DEV.type == "cuda"

#: BED-M's measured gate support, `V16_LEAN_DOMAIN.md` section 2.
BEDM_SUPPORT = (-1.0, 0.0, 1.0)


def _qk(n: int = S, d: int = 4, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=DT).to(DEV),
            torch.randn(n, d, generator=g, dtype=DT).to(DEV))


def _heads(n: int = S, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g, dtype=DT)
    th = ((torch.rand(n, generator=g, dtype=DT) * 2.0) - 1.0) * math.pi
    return u.to(DEV), th.to(DEV)


def _bedm(n: int = 64, s: int = 64, d: int = 24, t_star: int = 2, seed: int = 0):
    """BED-M `e3_t2` at float64. The corpus is built by `negation_scope` and
    then cast, and the label is RECOMPUTED by the bed's own oracle at float64 --
    so the residual is the arm's, not the corpus's float32 storage."""
    x, _y32, head, p = ns.make_equilibrium_batch(n, s, d, t_star=t_star,
                                                 d_model=16, seed=seed,
                                                 device=DEV)
    x = x.double()
    return x, ns.equilibrium_oracle(x, head, p), head


# ======================================================= THE HARD CONSTRAINT
# The theorem this whole arm exists to obey, measured rather than cited.

def test_no_prefix_scan_represents_bedm_hop_and_the_product_route_does():
    """`no_prefix_scan_represents_a_zero_gate` in float64, on BED-M's own draw.

    Two readings, and both are the theorem: the `exp(C_i - C_j)` route is
    NEVER exactly zero on the causal triangle (`Complex.exp` is never `0`),
    while the true hop is zero on almost all of it; and the scan route's
    read-out is `nan`, which is the arithmetic `V15_ARM_PHASE.md` (e) reported
    as a band residual.
    """
    x, _y, _head = _bedm(n=8)
    a = x[:, :, ns.CH_DRIVE]
    m, th = a.abs(), torch.angle(a)
    s = a.shape[-1]
    tri = torch.ones(s, s, dtype=torch.bool, device=a.device).tril(0)

    prod = smp.path_product(smp.gate(m, th))
    scan = smp.hop_scan(m, th)

    n_true_zero = int((prod == 0)[:, tri].sum())
    assert n_true_zero > 0, "BED-M draw carried no annihilating hop"
    #: exp is never zero -- not approximately, not at any C.
    assert int((scan == 0)[:, tri].sum()) == 0
    #: and the difference form is not merely wrong, it is undefined.
    assert bool(torch.isnan(scan[:, tri]).any())


def test_the_scan_route_is_the_shipped_planted_negative_for_the_label():
    """PLANTED NEGATIVE, BIND 1. `route="exp_scan"` is the forbidden route
    living in the arm's own module (V-24: the rejection region is occupied by
    shipped code). On BED-M's support it must NOT return a passing residual."""
    a, b = smp.bedm_draw(seed=SEED, s=S, device=DEV)
    r = smp.label_cell(a, b, mutation="exp_scan", seed=SEED)["residual"]
    assert not (r <= 1e-6), r
    assert math.isnan(r), r


# ================================================================== BIND 1
# ORACLE GATES IN => LABEL <= 1e-6, ON BED-M'S ACTUAL SUPPORT.

def test_bind1_the_oracle_gates_are_reachable_on_bedm_support_exactly():
    """`#2` re-stated parametrizes `a = m e^{i theta}` with `m in [0,1]`
    CLOSED. Exhibit the parameters that produce each of BED-M's three drawn
    values, and read what the float64 parametrization actually returns.

    `bedM_gate_exact` proves this exact in the complex numbers. In float64
    `polar(1, pi)` carries `1.2246e-16` of imaginary part -- named here rather
    than rounded away, because it is the whole of BIND 1's residual downstream.
    """
    u = torch.tensor([1.0, 0.0, 1.0], dtype=DT, device=DEV)     # m
    th = torch.tensor([math.pi, 0.0, 0.0], dtype=DT, device=DEV)  # theta
    a = smp.gate(smp.magnitude(u), th)
    target = torch.tensor(BEDM_SUPPORT, dtype=DT, device=DEV).to(CT)
    assert torch.equal(a.real, target.real)          # -1, 0, +1 EXACTLY
    assert float(a.imag.abs().max()) <= 1.3e-16
    assert float((a - target).abs().max()) <= 1.3e-16
    #: and the magnitudes are the CLOSED endpoints, attained as values.
    assert float(smp.magnitude(u).max()) == 1.0
    assert float(smp.magnitude(torch.zeros_like(u)).min()) == 0.0


def test_bind1_the_label_holds_on_the_real_bedm_corpus():
    """THE BIND THAT HAS NEVER HELD ON BED-M's ACTUAL SUPPORT.

    Gates and drives come from `make_equilibrium_batch`; the label is the bed's
    own `equilibrium_oracle`, recomputed at float64. The arm is set to the
    path-product corner (`beta = 0`, QK off, gates = oracle), which is `#5a`'s
    third corner and `#2` re-stated's clause 1+4 evaluated directly.

    NO value rescale and NO key bias appear, so neither `log(1 - m)` nor
    `1/(1 - m)` is instantiated and the `m = 1` end has no singularity to hit.
    """
    x, y, _head = _bedm(n=64)
    a, b = x[:, :, ns.CH_DRIVE], x[:, :, ns.CH_FLIP]
    assert sorted(float(v) for v in torch.unique(a)) == list(BEDM_SUPPORT)

    u, th = smp.oracle_heads(a)
    zero = torch.zeros(a.shape[0], a.shape[-1], 1, dtype=DT, device=a.device)
    out = smp.readout(zero, zero, b.unsqueeze(-1).to(CT), u, th,
                      beta=0.0, qk=0.0)[..., -1, 0]
    residual = float((out - y.to(CT)).abs().max())
    assert residual <= 1e-6, residual
    assert residual < 1e-13, residual        # ten orders inside the bar


def test_bind1_the_label_holds_at_both_closed_endpoints():
    """`m = 0` and `m = 1` are the two values that break a logarithm, and they
    break DIFFERENT ones: `m = 0` kills `log m`, `m = 1` kills `log(1 - m)`.
    The path-product route instantiates neither, so both endpoints are ordinary
    points here. Measured on a draw that carries both."""
    a, b = smp.bedm_draw(seed=SEED, s=32, device=DEV)
    assert float(a.abs().min()) == 0.0 and float(a.abs().max()) == 1.0
    assert float(a.real.min()) == -1.0            # both signs present
    r = smp.label_cell(a, b, seed=SEED)["residual"]
    assert r <= 1e-6, r


def test_bind1_the_singular_quantities_are_absent_from_this_construction():
    """What `m = 1` costs, stated as arithmetic. `V15_ARM_PHASE.md` section 6's
    oracle needs `s_j = log(1 - m_j)` and `V_j = b_j/(1 - m_j)`; at `m = 1`
    those are `-inf` and `inf`, and its band residual is `nan`. This arm's
    (L) setting uses `V = b` unrescaled, so the two singular expressions are
    not evaluated at all -- shown by exhibiting them beside a passing cell."""
    one = torch.ones(1, dtype=DT, device=DEV)
    assert math.isinf(float(torch.log1p(-one)))          # log(1 - 1)
    assert math.isinf(float(1.0 / (1.0 - one)))          # 1/(1 - 1)
    a, b = smp.band_draw(seed=SEED, s=S, device=DEV)     # |a| = 1 everywhere
    cell = smp.label_cell(a, b, seed=SEED)
    assert cell["residual"] <= 1e-6, cell["residual"]
    assert cell["v_max"] < math.inf and math.isfinite(cell["v_max"])


@pytest.mark.parametrize("mutation", [m for m in smp.MUTATIONS if m != "none"])
def test_bind1_planted_negatives_all_fire(mutation):
    """Every mutilation of the (L) setting must break the bind, and break it at
    `O(1)` or worse -- not at `2e-6`. `drop_phase` deletes the sign carrier,
    `drop_magnitude` deletes the annihilator, `beta_one` normalizes the row the
    path product is supposed to weight, `exp_scan` takes the forbidden route."""
    a, b = smp.bedm_draw(seed=SEED, s=S, device=DEV)
    honest = smp.label_cell(a, b, seed=SEED)
    assert honest["residual"] <= 1e-6
    r = smp.label_cell(a, b, mutation=mutation, seed=SEED)["residual"]
    assert not (r <= 1e-6), f"{mutation} did not fire: {r}"
    assert math.isnan(r) or r > 0.1, f"{mutation} fired only at {r}"


def test_the_bos_gate_is_never_read():
    """Every product starts at `k = j + 1 >= 1`, so `a[0]` cannot enter any
    hop. Checked by moving it to an arbitrary value and reading the residual,
    rather than by poisoning it and inspecting for `nan` -- a poisoned BOS
    would make the `exp_scan` negative fire from the draw instead of from the
    corpus's own zeros."""
    a, b = smp.bedm_draw(seed=SEED, s=S, device=DEV)
    base = smp.label_cell(a, b, seed=SEED)["residual"]
    moved = a.clone()
    moved[0] = complex(-7.25, 3.5)
    assert smp.label_cell(moved, b, seed=SEED)["residual"] == base


def test_bind1_the_manifest_moves_under_every_mutilation():
    """A manifest that cannot tell a mutilated cell from an honest one is not
    an identity (`scale/identity_manifest.py`). The `SMP_FIELDS` block is folded
    into the base hash for exactly that reason."""
    a, b = smp.bedm_draw(seed=SEED, s=S, device=DEV)
    honest = smp.label_cell(a, b, seed=SEED)["manifest"]
    assert honest["values"]["device"] == DEV.type      # device is LIVE
    seen = {honest["hash"]}
    for mutation in smp.MUTATIONS[1:]:
        h = smp.label_cell(a, b, mutation=mutation, seed=SEED)["manifest"]["hash"]
        assert h not in seen, f"{mutation} left the hash unmoved"
        seen.add(h)


def test_bind1_clause_one_of_the_restatement_holds_in_float64():
    """`|pathProd| = prod m` -- clause 1, whose hypothesis is `0 <= m` ONLY, so
    it covers the `m = 0` draw. The real cumulative product is what the
    normalizer uses, so this is also the check that `Z` is the modulus row."""
    u, th = _heads(24, seed=11)
    m = smp.magnitude(u)
    g_c = smp.path_product(smp.gate(m, th))
    g_m = smp.path_product(m)
    assert float((g_c.abs() - g_m).abs().max()) < 1e-14
    #: clause 4: a zero anywhere on the path annihilates EXACTLY.
    m0 = m.clone()
    m0[5] = 0.0
    z = smp.path_product(smp.gate(m0, th))
    assert bool((z[10:, :5] == 0).all())


# ================================================================== BIND 2
# THE THREE CORNERS OF #5a, BITWISE ON fp64 PROBES.

def test_bind2_softmax_corner_is_bitwise_against_the_statement_of_5a():
    """`three_corners_containment` first clause: `Hop 1 g0 qk = softmaxAttn`,
    where `#5a` states `softmaxAttn qk i j = exp(qk i j) / sum_{j'<=i} exp(...)`.
    The reference is that expression, written here from the THEOREM."""
    q, k = _qk()
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    up = torch.ones(S, S, dtype=torch.bool, device=DEV).triu(1)
    e = torch.exp(w.masked_fill(up, float("-inf")))
    ref = e / e.sum(-1, keepdim=True)
    got = smp.operator(q, k, beta=1.0, qk=1.0, g=0.0)
    assert torch.equal(got.real, ref)
    assert torch.equal(got.imag, torch.zeros_like(ref))


def test_bind2_softmax_corner_against_the_repos_own_softmax_operator():
    """The same corner against `ceq/lm.py`'s `Attention("softmax_x").operator`
    -- the repo's INCLUSIVE-causal control, and deliberately not
    `ceq/bench.py`, whose `tril(-1)` would put two binds on two objects
    (V-24's shared-carrier defect).

    This one is NOT bitwise and the gap is named. `torch.softmax` is a fused
    kernel that subtracts the row maximum; `#5a`'s `softmaxAttn` does not. No
    expression-level route reproduces the fused kernel bit for bit, so a
    section S-M' operator that materializes `Z_i^beta` cannot be bitwise
    against it -- `ceq/arm_phase.py` is bitwise only because it CALLS
    `torch.softmax`. The number is sub-ulp and is reported, not rounded.
    """
    q, k = _qk()
    ref = lm.Attention("softmax_x", 4, 1).operator(q.unsqueeze(0),
                                                  k.unsqueeze(0))[0]
    got = smp.operator(q, k, beta=1.0, qk=1.0, g=0.0).real
    gap = float((got - ref).abs().max())
    assert gap < 2.3e-16, gap        # one ulp of 1.0
    assert not torch.equal(got, ref)  # and it is NOT bitwise; stated, not hidden


def test_bind2_linear_corner_is_bitwise():
    """`Hop 0 g0 qk = linearAttn qk`, `linearAttn qk i j = exp(qk i j)`, no
    normalizer. `beta = 0` must give `Z^0 = 1` for EVERY `Z`."""
    q, k = _qk()
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    up = torch.ones(S, S, dtype=torch.bool, device=DEV).triu(1)
    ref = torch.exp(w.masked_fill(up, float("-inf")))
    got = smp.operator(q, k, beta=0.0, qk=1.0, g=0.0)
    assert torch.equal(got.real, ref)
    assert torch.equal(got.imag, torch.zeros_like(ref))


def test_bind2_path_product_corner_is_bitwise(capsys):
    """`Hop 0 g (qk = 0) = the path product`. The reference is an explicit
    double loop in the association order the definition states, sharing no code
    with the vectorized route.

    BITWISE ON cpu, AND THE DEVICE SPLIT IS REPORTED RATHER THAN WIDENED AWAY.
    `torch.cumprod` is sequential on cpu and a parallel scan on cuda, so on cuda
    the vectorized route re-associates: `11` of `64` entries move, by at most
    `5.551115e-17` -- a quarter of one ulp of `1.0`. That is the same class of
    reading as `V15_ARM_PHASE.md`'s `n_exactly_one` split (`9767` cpu vs `7713`
    cuda) and it is a property of the SUMMATION ORDER, not of the construction:
    the corner's zeros stay exactly zero on both devices, which is the clause
    the label bind depends on.
    """
    q, k = _qk()
    u, th = _heads()
    got = smp.operator(q, k, u, th, beta=0.0, qk=0.0, g=1.0)
    a = smp.gate(smp.magnitude(u), th)
    ref = torch.zeros(S, S, dtype=CT, device=DEV)
    for i in range(S):
        for j in range(i + 1):
            p = torch.ones((), dtype=CT, device=DEV)
            for kk in range(i, j, -1):          # the stated order
                p = p * a[kk]
            ref[i, j] = p
    n_moved = int((got != ref).sum())
    gap = float((got - ref).abs().max())
    with capsys.disabled():
        print(f"\n  path-product corner  {DEV.type:5s}  entries moved "
              f"{n_moved}/{S * S}   max|gap| {gap:.6e}")
    assert torch.equal(got == 0, ref == 0)      # the zeros agree on both
    if ON_CUDA:
        assert n_moved == 11 and gap < 2.3e-16, (n_moved, gap)
    else:
        assert torch.equal(got, ref)


def test_bind2_the_association_order_is_named_and_its_cost_measured():
    """A complex product is order-sensitive in float64, so the definition fixes
    one order. The other order is measured rather than assumed equal -- and on
    BED-M's own support the two agree BITWISE, because `{-1, 0, +1}` multiplies
    exactly."""
    u, th = _heads(16, seed=5)
    a = smp.gate(smp.magnitude(u), th)
    got = smp.path_product(a)
    asc = torch.zeros(16, 16, dtype=CT, device=DEV)
    for i in range(16):
        for j in range(i + 1):
            p = torch.ones((), dtype=CT, device=DEV)
            for kk in range(j + 1, i + 1):
                p = p * a[kk]
            asc[i, j] = p
    assert float((got - asc).abs().max()) < 1e-15
    #: on BED-M's own support the two orders agree to the last bit of the
    #: MODULUS -- the only quantity `Z` and the annihilation reading use -- and
    #: differ only in the imaginary dust `polar(1, pi)` injects.
    ab, _ = smp.bedm_draw(seed=SEED, s=16, device=DEV)
    gb = smp.path_product(ab)
    bsc = torch.zeros(17, 17, dtype=CT, device=DEV)
    for i in range(17):
        for j in range(i + 1):
            p = torch.ones((), dtype=CT, device=DEV)
            for kk in range(j + 1, i + 1):
                p = p * ab[kk]
            bsc[i, j] = p
    assert torch.equal(gb.real, bsc.real)
    assert float((gb - bsc).abs().max()) < 1e-14


def test_bind2_the_corners_are_numerically_distinct():
    """`corners_are_distinct`: a containment whose corners coincide is
    decoration. All three pairwise gaps must be `O(1)`."""
    q, k = _qk()
    u, th = _heads()
    c1 = smp.operator(q, k, beta=1.0, qk=1.0, g=0.0)
    c2 = smp.operator(q, k, beta=0.0, qk=1.0, g=0.0)
    c3 = smp.operator(q, k, u, th, beta=0.0, qk=0.0, g=1.0)
    for name, gap in (("1-2", (c1 - c2)), ("1-3", (c1 - c3)), ("2-3", (c2 - c3))):
        assert float(gap.abs().max()) > 0.5, name


def test_bind2_planted_negative_a_wrong_beta_breaks_the_softmax_corner():
    """PLANTED NEGATIVE, BIND 2. `beta` is the switch `#5b` names, so setting
    it wrong must break the corner it decides -- at `O(1)`, not at an ulp."""
    q, k = _qk()
    ref = lm.Attention("softmax_x", 4, 1).operator(q.unsqueeze(0),
                                                  k.unsqueeze(0))[0]
    bad = smp.operator(q, k, beta=0.0, qk=1.0, g=0.0).real
    assert float((bad - ref).abs().max()) > 0.5
    #: and leaving the gate ON breaks it too, which is what makes the corner a
    #: statement about `g` as well as about `beta`.
    u, th = _heads()
    bad2 = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=1.0).real
    assert float((bad2 - ref).abs().max()) > 0.1


# ================================================================== BIND 3
# beta, QK AND g ARE PRESENT AND EFFECTIVE.

def test_bind3_beta_decides_softmax_class_membership_by_row_sum():
    """`#5b`: `beta`, not `g`, is the switch that decides membership in the
    softmax class (`softmax_row_sum_one` vs `gate_zero_beta_zero_row_not_one`).
    Read as row sums of the MODULUS row, which is the probability vector."""
    q, k = _qk()
    u, th = _heads()
    r1 = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=1.0).abs().sum(-1)
    r0 = smp.operator(q, k, u, th, beta=0.0, qk=1.0, g=1.0).abs().sum(-1)
    assert float((r1 - 1.0).abs().max()) < 1e-14, float((r1 - 1.0).abs().max())
    assert float((r0 - 1.0).abs().max()) > 0.1
    #: and at the `g = 0` corner the same holds, so the reading is about beta.
    s1 = smp.operator(q, k, beta=1.0, qk=1.0, g=0.0).real.sum(-1)
    s0 = smp.operator(q, k, beta=0.0, qk=1.0, g=0.0).real.sum(-1)
    assert float((s1 - 1.0).abs().max()) < 1e-14
    assert float((s0 - 1.0).abs().max()) > 0.1


def test_bind3_the_qk_switch_is_effective():
    """QK off must delete the content term exactly, not approximately: at
    `qk = 0` the operator must not depend on `q` or `k` at all."""
    q, k = _qk()
    q2, k2 = _qk(seed=99)
    u, th = _heads()
    off_a = smp.operator(q, k, u, th, beta=0.0, qk=0.0, g=1.0)
    off_b = smp.operator(q2, k2, u, th, beta=0.0, qk=0.0, g=1.0)
    assert torch.equal(off_a, off_b)
    on = smp.operator(q, k, u, th, beta=0.0, qk=1.0, g=1.0)
    assert float((on - off_a).abs().max()) > 0.1


def test_bind3_the_g_switch_is_effective():
    """`g = 0` must be the `g == 0` corner EXACTLY -- `m = 1`, `theta = 0`, so
    the hop is the all-ones causal mask and the operator is gate-free."""
    q, k = _qk()
    u, th = _heads()
    off = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=0.0)
    bare = smp.operator(q, k, beta=1.0, qk=1.0, g=0.0)
    assert torch.equal(off, bare)
    on = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=1.0)
    assert float((on - off).abs().max()) > 0.1
    #: the hop itself at `g = 0` is the all-ones causal mask, bitwise.
    hop, mod = smp.hop(u, th, g=0.0)
    tri = torch.ones(S, S, dtype=DT, device=DEV).tril(0)
    assert torch.equal(mod, tri)
    assert torch.equal(hop.real, tri) and torch.equal(hop.imag, tri * 0.0)


def test_bind3_the_switches_are_live_parameters_on_the_shipped_module():
    """A switch that is not a parameter of the module the harness trains is not
    a switch. All three must be `nn.Parameter`s that reach `forward`."""
    torch.manual_seed(0)
    arm = smp.ArmSMPrime(S).to(DEV).to(DT)
    names = dict(arm.named_parameters())
    for s in ("beta", "qk", "g"):
        assert s in names and names[s].requires_grad, s
    x = torch.randn(4, S, 16, dtype=DT, device=DEV)
    base = arm(x)
    for s in ("beta", "qk", "g"):
        with torch.no_grad():
            getattr(arm, s).fill_(0.0)
        moved = float((arm(x) - base).abs().max())
        with torch.no_grad():
            getattr(arm, s).fill_(1.0)
        assert moved > 1e-8, f"switch {s} changed nothing: {moved}"


def test_bind3_planted_negative_a_frozen_exponent_collapses_the_reading():
    """PLANTED NEGATIVE, BIND 3. If `Z_i^beta` were written `Z_i` -- the switch
    present in the signature but absent from the arithmetic -- the two row sums
    the bind separates would coincide. Measured against that mutilation."""
    q, k = _qk()
    u, th = _heads()
    num, mod = smp.numerator(q, k, u, th, qk=1.0, g=1.0)
    frozen = num / mod.sum(-1, keepdim=True).to(num.dtype)   # beta ignored
    r1 = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=1.0).abs().sum(-1)
    assert float((frozen.abs().sum(-1) - r1).abs().max()) < 1e-14
    #: so the mutilation is exactly "beta has no effect", and the bind's
    #: beta = 0 reading is what rejects it.
    r0 = smp.operator(q, k, u, th, beta=0.0, qk=1.0, g=1.0).abs().sum(-1)
    assert float((frozen.abs().sum(-1) - r0).abs().max()) > 0.1


# ============================================ THE INITIALISER, AS AN IDENTITY
# V16_DEVICE_CERT.md section 5.4: `ceq/arm_phase.py` fails at step 0 from its
# own initialiser because `log(clamp(u, 0, 1))` is `-inf` at the closed lower
# endpoint. This arm takes no logarithm of the magnitude at all.

@pytest.mark.parametrize("init", ["as-constructed", "identity"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_arm_completes_forty_gradient_steps(init, dtype, capsys):
    """A FINITENESS probe on random data, not a training run: 40 steps, the
    budget `V16_DEVICE_CERT.md` section 5.4 used, and the reading is the first
    step at which any gradient goes non-finite. Nothing is fitted, no cell is
    produced, no result is kept."""
    r = smp.gradient_finiteness(init=init, steps=40, n=128, s=32,
                                dtype=getattr(torch, dtype), device=DEV)
    with capsys.disabled():
        print(f"\n  40-step gradient probe  {DEV.type:5s} {dtype:8s} "
              f"{init:15s} -> first non-finite: {r['first_non_finite']}")
    assert r["first_non_finite"] is None, r


def test_the_magnitude_never_leaves_the_closed_interval_through_the_switch():
    """The `g` switch blends a raw head toward the identity gate; the CLOSED
    cap is applied AFTER the blend, so no setting of `g` can put `m` outside
    `[0, 1]`. Without that ordering `g > 1` drives `m` negative, the modulus row
    goes negative, and `Z^beta` is `nan` -- measured before it was fixed."""
    g_ = torch.Generator().manual_seed(4)
    u = (torch.rand(4096, generator=g_, dtype=DT) * 6.0 - 3.0).to(DEV)
    th = torch.zeros_like(u)
    for gsw in (-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 25.0):
        m, _ = smp.blend(u, th, gsw)
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0, gsw


# =========================================== THE DIAGNOSTIC, AND ITS CONTROLS
# L-DIAG. What it must distinguish: an arm whose hop CAN annihilate from one
# whose hop cannot. The statistic is chosen here, on data, and both controls are
# read before it is trusted -- `MISTAKES.md` M-21, and `V15_ARM_PHASE.md`'s
# gate-R^2, which reads 1.000000 on the corpus alone because its target is an
# input channel.

def test_the_annihilation_mcc_is_read_on_the_corpus_alone_and_at_zero_steps(capsys):
    """(a) the corpus alone, no arm: the bed's own gate channel drives the hop,
    which is the ceiling. (b) a zero-step `ArmSMPrime`, 8 seeds: the floor.
    (c) the must-fire: the `exp_scan` route, whose hop is never exactly zero,
    must read exactly `0.0` however good its gates are.

    The gate-`R^2` instrument is reported beside it at the same shape, because
    `V15_ARM_PHASE.md` section 8 read `1.000000` for it on the corpus alone and
    that reading has to travel with any successor.
    """
    s, d, d_model, t_star = 64, 24, 16, 2
    live = list(range(s - t_star, s))
    x, _y, _head = _bedm(n=128, s=s, d=d, t_star=t_star, seed=0)
    a = x[:, :, ns.CH_DRIVE]
    truth = smp.path_product(smp.gate(a.abs(), torch.angle(a))) == 0

    corpus = smp.annihilation_mcc(smp.path_product(a.abs()) == 0, truth)
    scan = smp.hop_scan(a.abs(), torch.angle(a))
    must_fire = smp.annihilation_mcc(scan == 0, truth)

    zero_step = []
    for seed in range(8):
        torch.manual_seed(seed)
        arm = smp.ArmSMPrime(s, d_model=d_model).to(DEV).to(DT)
        zero_step.append(smp.annihilation_mcc(arm.zero_hop_mask(x), truth)["mcc"])
    z = sum(zero_step) / len(zero_step)
    with capsys.disabled():
        print(f"\n  annihilation MCC, CORPUS ALONE (no arm)   = "
              f"{corpus['mcc']:.6f}   (positives {corpus['n_pos']}/{corpus['n']})")
        print(f"  annihilation MCC, ZERO-STEP ArmSMPrime x8 = {z:.6f}"
              f"   per seed {[round(v, 6) for v in zero_step]}")
        print(f"  annihilation MCC, exp_scan MUST-FIRE      = "
              f"{must_fire['mcc']:.6f}   (predicted positives {must_fire['n_pred']})")
    #: the ceiling is exact in the counts, and 1.0 to float64 in the ratio.
    assert corpus["fp"] == 0 and corpus["fn"] == 0, corpus
    assert corpus["mcc"] == pytest.approx(1.0, abs=1e-12), corpus
    assert must_fire["mcc"] == 0.0 and must_fire["n_pred"] == 0, must_fire
    assert z < corpus["mcc"] - 0.05, (z, corpus["mcc"])
    assert all(math.isfinite(v) for v in zero_step)
    assert live[-1] == s - 1


def _probe(f_tr, y_tr, f_ev, y_ev):
    """`scripts/v15_r1.py::probe`'s form: least squares fit on train, scored
    out of sample on eval."""
    a_tr = torch.cat([torch.ones_like(f_tr[:, :1]), f_tr.double()], 1)
    a_ev = torch.cat([torch.ones_like(f_ev[:, :1]), f_ev.double()], 1)
    w = torch.linalg.lstsq(a_tr, y_tr.double().unsqueeze(1)).solution
    pred = (a_ev @ w).squeeze(1)
    yv = y_ev.double()
    sst = float(((yv - yv.mean()) ** 2).sum())
    return (float("nan") if sst == 0.0 else
            1.0 - float(((yv - pred) ** 2).sum()) / sst)


def test_the_gate_r2_instrument_has_no_headroom_on_the_corpus_alone(capsys):
    """The reading `V15_ARM_PHASE.md` section 8 published, reproduced on this
    arm so the successor instrument is comparable rather than asserted better.
    `R^2 = 1.000000` on the corpus alone: the target IS an input channel
    (`scale/negation_scope.py:432`), so any reading short of `1.0` measures the
    parametrization and never the availability of the information."""
    s, d, d_model, t_star = 64, 24, 16, 2
    live = list(range(s - t_star, s))
    x_tr, _, _ = _bedm(n=512, s=s, d=d, t_star=t_star, seed=0)
    x_ev, _, _ = _bedm(n=512, s=s, d=d, t_star=t_star, seed=12345)
    a_tr = x_tr[:, live, ns.CH_DRIVE].reshape(-1)
    a_ev = x_ev[:, live, ns.CH_DRIVE].reshape(-1)
    corpus = _probe(x_tr[:, live, :].reshape(-1, d_model), a_tr,
                    x_ev[:, live, :].reshape(-1, d_model), a_ev)
    zs = []
    for seed in range(8):
        torch.manual_seed(seed)
        arm = smp.ArmSMPrime(s, d_model=d_model).to(DEV).to(DT)
        zs.append(_probe(arm.gate_feature(x_tr, live), a_tr,
                         arm.gate_feature(x_ev, live), a_ev))
    z = sum(zs) / len(zs)
    with capsys.disabled():
        print(f"\n  gate R^2, CORPUS ALONE (no arm)           = {corpus:.6f}")
        print(f"  gate R^2, ZERO-STEP ArmSMPrime x8         = {z:.6f}"
              f"   per seed {[round(v, 6) for v in zs]}")
    assert corpus > 0.99, corpus
    assert z < corpus


# ================================================================== HYGIENE

def test_the_operator_dispatched_by_the_module_is_the_bound_one():
    """A bind on a free function the shipped module does not dispatch through
    is a reading of a non-shipped operator."""
    torch.manual_seed(0)
    arm = smp.ArmSMPrime(S).to(DEV).to(DT)
    arm.identity_heads()
    x = torch.randn(2, S, 16, dtype=DT, device=DEV)
    u, th = arm.heads(x)
    got = arm._operator(arm.wq(x), arm.wk(x), u, th)
    ref = smp.operator(arm.wq(x), arm.wk(x), u, th,
                       beta=arm.beta, qk=arm.qk, g=arm.g)
    assert torch.equal(got, ref)


def test_identity_heads_is_the_softmax_corner_and_not_the_zero_gate():
    """`V15_ARM_PHASE.md` section 7 item 4: any harness that resets heads to
    zero puts a closed-cap arm at `m = 0`, the ANNIHILATING gate, not at the
    identity. The method is named `identity_heads` for that reason and the
    magnitude head's BIAS is the thing set to one."""
    torch.manual_seed(0)
    arm = smp.ArmSMPrime(S).to(DEV).to(DT).identity_heads()
    x = torch.randn(3, S, 16, dtype=DT, device=DEV)
    u, th = arm.heads(x)
    m, t = smp.blend(u, th, 1.0)
    assert torch.equal(m, torch.ones_like(m))
    assert torch.equal(t, torch.zeros_like(t))
    q, k = arm.wq(x), arm.wk(x)
    ref = lm.Attention("softmax_x", 4, 1).operator(q, k)
    assert float((arm._operator(q, k, u, th).real - ref).abs().max()) < 2.3e-16


def test_the_arm_is_a_drop_in_of_the_same_shape_as_the_phase_arm():
    """Same constructor shape, same `forward(x) -> [n]` reading position
    `s - 1`, so `scale/m3_capability.py`'s harness takes it unchanged. The
    parameter count is priced rather than claimed equal."""
    torch.manual_seed(0)
    arm = smp.ArmSMPrime(64, d_model=16)
    n = sum(p.numel() for p in arm.parameters())
    x = torch.randn(5, 64, 16)
    assert arm(x).shape == (5,)
    #: softmax control 4,769 (`V15_R1.md`); the 10% bar is on the ratio.
    assert abs(n - 4769) / 4769 < 0.10, n
