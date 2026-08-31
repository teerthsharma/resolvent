"""ARM S-M' -- section S-M's hop, built as the PATH PRODUCT the re-stated
Lean `#2` defines, and not as the prefix scan section S-M' writes it with.

    W_ij  =  G_ij * exp(qk * q_i.k_j) / Z_i^beta            j <= i,  0 above
    G_ij  =  prod_{k=j+1}^{i} m_k e^{i theta_k}             THE PATH PRODUCT
    R_ij  =  prod_{k=j+1}^{i} m_k                           its modulus, clause 1
    Z_i   =  sum_{j<=i} R_ij exp(qk * q_i.k_j)              the modulus row

`beta`, `qk` and `g` are the three switches `CEQ_V16_CONTRACT.md` PART I names.
`#5a`'s three corners are settings of them:

    softmax attention     beta = 1, g = 0, QK on     row sums to 1
    linear attention      beta = 0, g = 0, QK on     no normalizer
    the path product      beta = 0,        QK off    W = G

WHY THE PRODUCT AND NOT `exp(C_i - C_j)`. `lean/CEQ/V16Domain.lean`:

    no_prefix_scan_represents_a_zero_gate (C : N -> C) (m theta : N -> R) {i j}
        (hz : exists k in Ico (j+1) (i+1), m k = 0) :
      Complex.exp (C i - C j) != pathProd m theta i j

`Complex.exp` is never `0`; the path product is. BED-M's gate is exactly `0` on
`126,976 / 131,072` entries and NO sequence satisfies `forall k, 0 < m k` --
structurally, `scale/negation_scope.py:429` zeroes the first `head + 1`
positions of every row. So no prefix scan whatsoever, no repaired `log`, no
extended-real convention, represents BED-M's hop. That is a theorem, not a
numerical accident, and it is why `V15_ARM_PHASE.md` (e) reads `nan` on the
band.

THIS MODULE IMPLEMENTS THE PROVED STATEMENT, IT DOES NOT INVENT ONE.
`prefix_logit_mask_restated` defines

    pathProd m theta i j = prod_{k in Ico (j+1) (i+1)} (m k) * exp(i theta k)

with NO logarithm anywhere in the definition, and proves as clause 1 that its
modulus is `prod m` and as clause 4 that a single `m_k = 0` sends the product
to exactly `0`. `path_product` below is that definition, evaluated as a masked
cumulative product; `hop` returns the complex product and its real modulus row
as two objects, which is clauses 1 and 4 side by side. Nothing here is a
mechanism the round's first law would strike -- it is the computational shadow
of `#2` re-stated and of `#5a` corner 3. A reader should check that claim
against `V16Domain.lean` directly.

THE TWO CLOSED ENDPOINTS BREAK TWO DIFFERENT LOGARITHMS, AND THIS ARM TAKES
NEITHER. `m = 0` kills `log m` -- `V16_DEVICE_CERT.md` section 5.4 measured
`ceq/arm_phase.py` failing at step 0 from its own initialiser for exactly that
reason, because `clamp(u, 0, 1)` puts half of a randomly-initialised head on the
closed lower endpoint. `m = 1` kills `log(1 - m)` and `1/(1 - m)` -- the value
rescale, which is what `V15_ARM_PHASE.md` (e)'s `nan` and
`V15_X36_PRIOR_ART.md`'s `73,766` infinities are. The (L) setting here is the
`beta = 0`, QK-off corner, where the read-out is `O_i = sum_j G_ij V_j` with
`V = b` UNRESCALED, so neither expression is instantiated and both endpoints are
ordinary points. The `[0,1]` versus `[0,1)` question is the author's to rule on;
this module builds `[0,1]` CLOSED as `X36` specifies and reports what it costs.

THE ORDER OF THE PRODUCT IS FIXED AND STATED. A complex product is not
associative in float64. `path_product` accumulates from `k = i` down to
`k = j + 1` (the cumulative product runs right-to-left along the row). On
BED-M's own support `{-1, 0, +1}` both orders agree bitwise; off it they differ
at `1e-16`, measured in `tests/arm_smprime`.

NOTHING HERE TRAINS (L-LEAN). `gradient_finiteness` takes gradient steps on
random data and keeps nothing: it is the identity check
`V16_DEVICE_CERT.md` section 5.4 asks for, because an initialiser that produces
`-inf` is an identity failure and not a training result.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ceq.arm_phase import band_draw, chain_label
from scale import identity_manifest

NAME = "arm_smprime"
#: The hop route and the switch set, recorded so a cell measured under the
#: prefix-scan route can never be filed under this arm's name.
VARIANT = "pathprod_hop+beta_qk_g_switches"
DTYPE = torch.float64
CDTYPE = torch.complex128
BOS = 0

#: `product` is the arm. `exp_scan` is the route
#: `no_prefix_scan_represents_a_zero_gate` forbids, kept in the SHIPPED module
#: so BIND 1's rejection region is occupied by real code and not by a
#: test-local copy (`MISTAKES.md` V-24).
ROUTES = ("product", "exp_scan")

#: The (L) setting's mutilations, one per part of the construction.
MUTATIONS = ("none", "drop_phase", "drop_magnitude", "beta_one", "exp_scan")

SMP_FIELDS = ("variant", "route", "beta", "qk", "g", "m_setting",
              "theta_setting", "v_setting", "m_max", "v_max", "n_zero_gates")

#: `-inf` and not `finfo.min`: masking BEFORE the exponential means the upper
#: triangle is `exp(-inf) = 0` exactly and its backward is `grad * 0`, so no
#: `inf` is ever created there to meet a zero gradient and make a `nan`.
NEG = float("-inf")


def _ctype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else CDTYPE


# ------------------------------------------------------- the parametrization

def magnitude(u: torch.Tensor) -> torch.Tensor:
    """`m = clamp(u, 0, 1)`. The X36 cap, CLOSED: `0` and `1` are attainable
    VALUES, not limits, which is what lets the parametrization reach BED-M's
    `a in {-1, 0, +1}` at all."""
    return torch.clamp(u, 0.0, 1.0)


def blend(u: torch.Tensor, theta: torch.Tensor, g) -> tuple:
    """The `g` SWITCH. `g = 0` is `#5a`'s `g == 0` corner exactly: `m = 1` and
    `theta = 0`, so the hop is the all-ones causal mask and the operator is
    gate-free.

    THE CAP IS APPLIED AFTER THE BLEND, and the ordering is load-bearing.
    Clamping first and blending second lets `g > 1` extrapolate past the lower
    endpoint to a NEGATIVE magnitude; the modulus row then goes negative and
    `Z^beta` is `nan` for non-integer `beta`. Measured: with the blend last, a
    freely-moving `g` produced a non-finite gradient at step 32 of 40. With the
    cap last, `m in [0,1]` is a property of the arithmetic at every value of the
    switch, which is what the closed cap is for.
    """
    m = magnitude(torch.lerp(torch.ones_like(u), u, g))
    return m, theta * g


def gate(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """`gateOf m theta = m * exp(i theta)`, complex -- `V16Domain.lean`'s own
    definition, with no branch of `Real.log` in it.

    `bedM_gate_exact` proves `gateOf |a| (arg a) = a` for each of `{-1, 0, +1}`
    exactly in the complex numbers. In float64 `polar(1, pi)` carries
    `1.2246e-16` of imaginary part; the real part and the modulus are exact.
    """
    return m.to(_ctype(m.dtype)) * torch.polar(torch.ones_like(theta), theta)


def path_product(a: torch.Tensor) -> torch.Tensor:
    """`G_ij = prod_{k=j+1}^{i} a_k` for `j <= i`, `0` above the diagonal.

    `[..., S] -> [..., S, S]`. Works on the complex gate (giving `pathProd`)
    and on the real magnitude (giving `prod m`, clause 1's right-hand side),
    which is why the normalizer never has to take `abs` of a complex zero -- an
    operation whose backward meets `sgn(0) = 0` and produced `nan` gradients
    when it was tried.

    THE MECHANISM, and it is the whole repair: this is a cumulative PRODUCT, so
    a single `a_k = 0` sends every entry whose window contains `k` to exactly
    `0` (clause 4). The prefix-scan form `exp(C_i - C_j)` cannot do that for any
    `C` whatsoever. It is also why the ratio form `cumprod_i / cumprod_j` is
    not used: at a zero that ratio is `0/0`.

    `ponytail:` O(S^2) materialization, which is what the [S,S] operator is
    anyway; a segmented associative scan would be the move if S outgrew the
    bed's 64.
    """
    s = a.shape[-1]
    idx = torch.arange(s, device=a.device)
    le = idx.unsqueeze(-1) >= idx.unsqueeze(-2)          # [i, k]: k <= i
    #: entries with `k > i` are set to `1`, so the row's reverse cumulative
    #: product ignores them exactly rather than approximately.
    rows = torch.where(le, a.unsqueeze(-2), torch.ones_like(a).unsqueeze(-2))
    q = torch.flip(torch.cumprod(torch.flip(rows, [-1]), -1), [-1])
    #: `q[..., i, j] = prod_{k=j}^{i} a_k`, so the hop is its shift by one.
    g = torch.cat([q[..., 1:], torch.ones_like(q[..., :1])], -1)
    return g.masked_fill(~le, 0)


def hop_scan(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """PLANTED NEGATIVE, shipped: `exp(C_i - C_j)` with
    `C = cumsum(log m) + i cumsum(theta)` -- section S-M's hop as written.

    Kept so the theorem can be MEASURED rather than cited. On any window
    carrying a zero magnitude this returns `nan` (`-inf - (-inf)`), and it is
    never exactly `0` anywhere, which is `Complex.exp` being never zero. One
    zero gate poisons every later pair, not only the pairs that straddle it.
    """
    c = torch.complex(torch.cumsum(torch.log(m), dim=-1),
                      torch.cumsum(theta, dim=-1))
    s = m.shape[-1]
    up = torch.ones(s, s, dtype=torch.bool, device=m.device).triu(1)
    return torch.exp(c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(up, 0)


def hop(u: torch.Tensor, theta: torch.Tensor, *, g=1.0,
        route: str = "product") -> tuple:
    """`(G, R)`: the complex hop and its real modulus row, both `[..., S, S]`.

    Two objects and not one, because clause 1 (`|pathProd| = prod m`) is an
    identity in the reals that costs `1e-16` in float64. `R` is the right-hand
    side, computed on the magnitudes directly, and it is what `Z` sums -- so
    the normalizer is a statement about the theorem's own quantity.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")
    m, th = blend(u, theta, g)
    if route == "exp_scan":
        h = hop_scan(m, th)
        return h, h.abs()
    return path_product(gate(m, th)), path_product(m)


# --------------------------------------------------------------- the operator

def numerator(q: torch.Tensor, k: torch.Tensor,
              u: torch.Tensor | None = None, theta: torch.Tensor | None = None,
              *, qk=1.0, g=1.0, route: str = "product") -> tuple:
    """`(G_ij exp(qk q_i.k_j), R_ij exp(qk q_i.k_j))` -- the numerator and the
    modulus row it is normalized by, before `beta` is applied.

    `qk = 0` deletes the content term EXACTLY: `0.0 * w` is `0.0` and
    `exp(0) = 1`, so the QK-off corner is the path product itself and not the
    path product times something near one.
    """
    n = q.shape[-2]
    du, dev = q.dtype, q.device
    w = qk * ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]))
    up = torch.ones(n, n, dtype=torch.bool, device=dev).triu(1)
    #: mask BEFORE the exponential: `exp(-inf) = 0` exactly, and no `inf` is
    #: created above the diagonal to meet a zero gradient in the backward.
    e = torch.exp(w.masked_fill(up, NEG))
    uu = torch.ones(n, dtype=du, device=dev) if u is None else u
    tt = torch.zeros(n, dtype=du, device=dev) if theta is None else theta
    gh, rh = hop(uu, tt, g=g, route=route)
    return gh * e.to(gh.dtype), rh * e


def operator(q: torch.Tensor, k: torch.Tensor,
             u: torch.Tensor | None = None, theta: torch.Tensor | None = None,
             *, beta=1.0, qk=1.0, g=1.0, route: str = "product") -> torch.Tensor:
    """The `[..., S, S]` complex operator `W_ij = num_ij / Z_i^beta`.

    `Z_i = sum_{j<=i} R_ij exp(...)` is real and STRICTLY POSITIVE: the
    diagonal term is `R_ii exp(w_ii) = exp(w_ii) > 0` because the empty product
    is `1`. So `Z^beta` needs no positivity side condition and `Z^0 = 1` for
    every `Z`, which is how the `beta = 0` corners stay exact.

    The division is done on the real and imaginary parts SEPARATELY. Dividing a
    complex tensor by a real one promotes and then runs the general complex
    quotient `(ac + bd)/(c^2 + d^2)`, which is not bitwise `a/c` even at
    `d = 0`; the two corners that are bitwise here would not be.
    """
    num, mod = numerator(q, k, u, theta, qk=qk, g=g, route=route)
    zb = mod.sum(-1, keepdim=True) ** beta
    return torch.complex(num.real / zb, num.imag / zb)


def readout(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            u: torch.Tensor | None = None, theta: torch.Tensor | None = None,
            *, beta=1.0, qk=1.0, g=1.0, route: str = "product") -> torch.Tensor:
    """`O_i = sum_{j<=i} W_ij V_j`, complex.

    At `beta = 0` with QK off this is `sum_j (prod_{k>j} a_k) b_j` -- BED-M's
    label by its own definition (`scale/negation_scope.py::equilibrium_oracle`
    calls it "the signed path sum"), with `V = b` and no rescale.
    """
    a = operator(q, k, u, theta, beta=beta, qk=qk, g=g, route=route)
    return a @ v.to(a.dtype)


# ------------------------------------------------------------- the (L) setting

def oracle_heads(a: torch.Tensor):
    """`(u, theta) = (|a|, arg a)`. That is the whole oracle setting.

    There is no key bias and no value rescale to set, which is the difference
    from `ceq/arm_phase.py::oracle_heads`: those exist to make a SOFTMAX row
    reproduce a path product, and at `beta = 0` the row is the path product.
    `s_j = log(1 - m_j)` and `V_j = b_j/(1 - m_j)` are the two expressions that
    are `-inf` and `inf` at `m = 1`, and neither is written here.
    """
    return a.abs(), torch.angle(a)


def bedm_draw(seed: int = 15, s: int = 8, device=None):
    """BED-M's own gate support, `{-1, 0, +1}`, with the head zeroed the way
    `scale/negation_scope.py:428-429` zeroes it.

    `a[0] = 1` -- the BOS carries the IDENTITY gate, not a poisoned one, and
    the choice is deliberate. `b[0] = 0` because the chain's `y_0` is `0`, and
    the hop never reads `a[0]` at all because every product starts at
    `k = j + 1 >= 1` (`test_the_bos_gate_is_never_read` is the receipt). Were
    the BOS poisoned with `nan` or set to `0`, the `exp_scan` planted negative
    would fire from the BOS rather than from the corpus's own zeros, which
    would make it a check on the draw instead of on the theorem.

    Drawn on the HOST and moved afterwards, so the bytes are the same on every
    device (`V16_BAR_RECERT.md` section 6.1).
    """
    g = torch.Generator().manual_seed(seed)
    sign = torch.randint(0, 2, (s + 1,), generator=g, dtype=torch.int64) * 2 - 1
    m = sign.abs().to(DTYPE)
    #: the head zeroing, in the builder's proportion: the first third of the
    #: row carries no gate, so the draw contains BOTH closed endpoints.
    m[1:1 + max(1, s // 3)] = 0.0
    th = torch.where(sign < 0, torch.full_like(m, math.pi), torch.zeros_like(m))
    m[0], th[0] = 1.0, 0.0
    b = torch.randn(s + 1, generator=g, dtype=DTYPE)
    b[0] = 0.0
    return gate(m, th).to(device), b.to(device)


def mutate(u, theta, v, mutation: str):
    """One of `MUTATIONS` applied to the oracle setting. Each removes exactly
    one part of the construction; the two that change a SWITCH return it."""
    if mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}; expected one of {MUTATIONS}")
    if mutation == "drop_phase":
        return u, torch.zeros_like(theta), v, 0.0, "product"
    if mutation == "drop_magnitude":
        return torch.ones_like(u), theta, v, 0.0, "product"
    if mutation == "beta_one":
        return u, theta, v, 1.0, "product"
    if mutation == "exp_scan":
        return u, theta, v, 0.0, "exp_scan"
    return u, theta, v, 0.0, "product"


def label_cell(a: torch.Tensor, b: torch.Tensor, *, y: torch.Tensor | None = None,
               mutation: str = "none", seed: int = 15,
               cell: str | None = None) -> dict:
    """One measured cell of the (L) bind, with its manifest.

    `y` defaults to `ceq/arm_phase.py::chain_label`, which computes the label
    from its own recurrence, so comparing the read-out to it is a check and not
    an identity restated twice.
    """
    u, th = oracle_heads(a)
    v = b.to(a.dtype)
    u, th, v, beta, route = mutate(u, th, v, mutation)
    n = a.shape[-1]
    zero_q = torch.zeros(*a.shape[:-1], n, 1, dtype=DTYPE, device=a.device)
    out = readout(zero_q, zero_q, v.unsqueeze(-1), u, th,
                  beta=beta, qk=0.0, route=route)[..., 0]
    tgt = chain_label(a, b) if y is None else y
    live = a[..., 1:]

    record = {
        "cell": cell or f"{NAME}:{mutation}",
        "kind": NAME,
        "task": "chain_label_complex",
        "s": int(n - 1),
        "d": 1,
        "d_model": 1,
        "steps": 0,                   # L-LEAN: nothing is trained by this node
        "seed": seed,
        #: LIVE, read off the tensors the cell was measured with
        #: (`V16_R1_DEVICE_READY.md`): a literal here made a cuda cell hash
        #: identically to a cpu one.
        "device": a.device.type,
        "torch_version": torch.__version__,
        "variant": VARIANT,
        "route": route,
        "beta": float(beta),
        "qk": 0.0,
        "g": 1.0,
        "m_setting": ("1" if mutation == "drop_magnitude" else "|a_j|"),
        "theta_setting": ("0" if mutation == "drop_phase" else "arg a_j"),
        "v_setting": "b_j, UNRESCALED",
        "m_max": float(live.abs().max()),
        "v_max": float(v.abs().max()),
        "n_zero_gates": int((live.abs() == 0).sum()),
    }
    #: THE BOS SLOT IS NOT SCORED, AND IS REPORTED INSTEAD.
    #: `chain_label`'s loop starts at `i = 1` and leaves `y_0 = 0`, so slot 0 is
    #: the chain's drive-free INITIAL STATE. The read-out there is `G_00 V_0`
    #: and `G_00` is the EMPTY path product, so no gate, no `m`, no `theta`, no
    #: route and no `beta` enters it: measured, the slot-0 read-out is `V_0`
    #: bitwise under all of `MUTATIONS` (`exp_scan` excepted, where it is `nan`
    #: from the BOS's own `log m_0 = -inf`, a quantity the path product never
    #: reads). A term no setting of the arm can move is not a reading of the
    #: arm -- it is a reading of what the caller put in slot 0.
    #: This module's own draws set `b[0] = 0` and it read `0`; BED-M's
    #: `scale/negation_scope.py::make_equilibrium_batch` zeroes `b[s-1]` and
    #: NOT `b[0]`, and the residual then read `max |b_0|` = 1.440495 with
    #: honest settings, where the hops read 2.965914e-16
    #: (`V17_LABEL_CELL_REPAIR.md`).
    #: The excluded term is REPORTED and not deleted: the drive at slot 0 does
    #: reach later rows whenever `G_i0 != 0`, and that disagreement stays in
    #: `residual`. Zeroing `b_0` instead would have moved BOTH sides and hidden
    #: it. `residual_bos` is outside `SMP_FIELDS`, so no published hash moves.
    record["residual_bos"] = float((out[..., BOS] - tgt[..., BOS]).abs().max())
    record["residual"] = float((out[..., BOS + 1:] - tgt[..., BOS + 1:]).abs().max())
    record["manifest"] = cell_manifest(
        record, callables=(operator, readout, hop, path_product, oracle_heads),
        params={"u": u, "theta": th, "V": v})
    return record


def cell_manifest(record: dict, *, callables, params) -> dict:
    """`identity_manifest.manifest` plus the `SMP_FIELDS` block.

    The base manifest hashes `CONFIG_FIELDS` only, which has no slot for the
    route, the three switches or the head settings -- and a manifest that
    cannot tell a mutilated cell from an honest one is not an identity.
    `scale/identity_manifest.py` is another node's file and every published hash
    depends on `CONFIG_FIELDS`, so the block is folded in here.
    """
    m = identity_manifest.manifest(record, callables=callables, params=params)
    values = {f: record.get(f) for f in SMP_FIELDS}
    blk = identity_manifest._sha(identity_manifest._canon(values))
    m["smp"] = blk
    m["smp_values"] = values
    m["hash"] = identity_manifest._sha(m["hash"].encode(), blk.encode())
    return m


# ------------------------------------------------------------ the diagnostic

def annihilation_mcc(pred_zero: torch.Tensor, true_zero: torch.Tensor) -> dict:
    """Matthews correlation between "this hop annihilates" as the arm reads it
    and as the corpus is, over the causal pairs `j <= i`.

    WHAT IT MUST DISTINGUISH (L-DIAG states the job, not the statistic): an arm
    whose hop CAN be exactly zero from one whose hop cannot. The `exp_scan`
    route is the must-fire -- `Complex.exp` is never zero, so its predicted
    positive set is empty and this reads exactly `0.0` however good its gates
    are. The statistic was chosen because the gate `R^2` cannot separate those
    two: it reads `1.000000` on the corpus alone (`V15_ARM_PHASE.md` section 8)
    since the gate is an input channel, and it is blind to whether the arm's
    hop can annihilate at all.

    MCC and not accuracy: on BED-M `96.9%` of causal pairs annihilate, so
    accuracy is `0.969` for the constant predictor. Returns `0.0` when a margin
    is empty, which is the degenerate case rather than a passing one, and the
    counts are returned beside it so the degeneracy is visible.
    """
    s = pred_zero.shape[-1]
    tri = torch.ones(s, s, dtype=torch.bool, device=pred_zero.device).tril(0)
    p = pred_zero[..., tri].reshape(-1)
    t = true_zero[..., tri].reshape(-1)
    tp = int((p & t).sum())
    tn = int((~p & ~t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    den = math.sqrt(float(tp + fp) * float(tp + fn)
                    * float(tn + fp) * float(tn + fn))
    return {"mcc": 0.0 if den == 0.0 else (tp * tn - fp * fn) / den,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n": tp + tn + fp + fn, "n_pos": tp + fn, "n_pred": tp + fp}


# --------------------------------------------- the initialiser, as an identity

def gradient_finiteness(*, init: str = "as-constructed", steps: int = 40,
                        n: int = 512, s: int = 64, d_model: int = 16,
                        seed: int = 0, lr: float = 1e-3,
                        dtype: torch.dtype = torch.float32, device=None) -> dict:
    """First step at which any gradient over the arm's parameters goes
    non-finite, on RANDOM data, or `None` if none does in `steps`.

    `V16_DEVICE_CERT.md` section 5.4's probe, same budget and same shape class.
    It found `ceq/arm_phase.py` failing at step 0 as-constructed on both
    devices, because `log(clamp(u, 0, 1))` is `-inf` at the closed lower
    endpoint that half a randomly-initialised head lands on. This module takes
    no logarithm of the magnitude, so that mechanism is absent by construction.

    NOTHING IS KEPT (L-LEAN). No cell, no checkpoint, no loss curve; the return
    value is a finiteness verdict on the initialiser, which is an identity
    property and not a training result.
    """
    torch.manual_seed(seed)
    arm = ArmSMPrime(s, d_model=d_model).to(dtype).to(device)
    if init == "identity":
        arm.identity_heads()
    elif init != "as-constructed":
        raise ValueError(f"unknown init {init!r}")
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(n, s, d_model, generator=g).to(dtype).to(device)
    y = torch.randn(n, generator=g).to(dtype).to(device)
    opt = torch.optim.AdamW(arm.parameters(), lr=lr)
    first, frac = None, None
    for t in range(steps):
        opt.zero_grad()
        ((arm(x) - y) ** 2).mean().backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in arm.parameters()):
            first = t
            break
        opt.step()
    with torch.no_grad():
        m, _ = blend(*arm.heads(x), arm.g)
        frac = float((m == 0).to(torch.float64).mean())
    return {"first_non_finite": first, "steps": steps, "init": init,
            "dtype": str(dtype), "device": str(x.device.type),
            "frac_positions_at_m_zero": frac}


# ------------------------------------------------------------- the drop-in arm

class ArmSMPrime(nn.Module):
    """Section S-M' as `scale/m3_capability.py`'s harness wants it.

    Same constructor shape as `arm_phase.ArmPhase` and `arm_pl.ArmPL` --
    `wq`/`wk`, one shared-width MLP, a scalar read-out, `forward(x) -> [n]`
    reading position `s - 1`. TWO per-position scalar heads (`u`, `theta`)
    instead of ARM PHASE's three: there is no key bias to learn, because at
    `beta = 0` the row is not a softmax that needs compensating. Three scalar
    SWITCHES on top -- `beta`, `qk`, `g` -- which is what `#5a` requires to be
    settable, and they are `nn.Parameter`s so the harness trains them.

    `identity_heads()` and not `zero_heads()`, for `V15_ARM_PHASE.md` section 7
    item 4's reason: a harness that resets heads to zero puts a closed-cap arm
    at `m = 0`, the ANNIHILATING gate, not at the identity. The magnitude head's
    BIAS is the thing set to one.

    NOT TRAINED HERE and not by this node (L-LEAN).
    """

    def __init__(self, s: int, d_model: int = 16, hidden: int = 128):
        super().__init__()
        self.kind = NAME
        self.seq = s
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.m_head = nn.Linear(d_model, 1)
        self.theta_head = nn.Linear(d_model, 1)
        #: the three switches, at the softmax corner's settings. `beta = 1` is
        #: the only initialisation inside the softmax class (`#5b`), which is
        #: where an arm built FROM softmax should start.
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.qk = nn.Parameter(torch.tensor(1.0))
        self.g = nn.Parameter(torch.tensor(1.0))
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_model))
        self.readout = nn.Linear(d_model, 1)

    def identity_heads(self) -> "ArmSMPrime":
        """`m = 1, theta = 0` exactly, switches at `beta = 1, qk = 1, g = 1`:
        the softmax corner, reached through the gate rather than around it."""
        with torch.no_grad():
            for h in (self.m_head, self.theta_head):
                h.weight.zero_()
                h.bias.zero_()
            self.m_head.bias.fill_(1.0)
            for sw in (self.beta, self.qk, self.g):
                sw.fill_(1.0)
        return self

    def heads(self, x: torch.Tensor):
        return (self.m_head(x).squeeze(-1), self.theta_head(x).squeeze(-1))

    @torch.no_grad()
    def gate_feature(self, x: torch.Tensor, live) -> torch.Tensor:
        """`Re(a_hat_j)` at the live positions, `[n*len(live), 1]` -- the gate
        `R^2` instrument's feature, in `ceq/arm_phase.py`'s form so the two
        arms' readings are comparable."""
        self.eval()
        m, th = blend(*self.heads(x), self.g)
        return (m * torch.cos(th))[:, live].reshape(-1, 1)

    @torch.no_grad()
    def zero_hop_mask(self, x: torch.Tensor) -> torch.Tensor:
        """`[n, S, S]` bool: where the arm's own hop annihilates EXACTLY.

        Read off the modulus row, which is `prod m` -- clause 1's right-hand
        side -- so a `1e-300` product does not read as a zero it is not.
        """
        self.eval()
        u, th = self.heads(x)
        return hop(u, th, g=self.g)[1] == 0

    def _operator(self, q, k, u, theta) -> torch.Tensor:
        return operator(q, k, u, theta, beta=self.beta, qk=self.qk, g=self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, seq, _ = x.shape
        u, th = self.heads(x)
        a = self._operator(self.wq(x), self.wk(x), u, th)
        h = self.mlp(x + (a @ x.to(a.dtype)).real)
        return self.readout(h).squeeze(-1)[:, seq - 1]


#: re-exported so a probe can draw BED-M's `{-1, +1}` band without importing
#: two arm modules; the draw is `ceq/arm_phase.py`'s, unchanged.
__all__ = ["NAME", "VARIANT", "ROUTES", "MUTATIONS", "SMP_FIELDS", "magnitude",
           "blend", "gate", "path_product", "hop_scan", "hop", "numerator",
           "operator", "readout", "oracle_heads", "bedm_draw", "band_draw",
           "chain_label", "mutate", "label_cell", "cell_manifest",
           "annihilation_mcc", "gradient_finiteness", "ArmSMPrime"]
