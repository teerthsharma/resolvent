"""ARM PHASE -- ARM PL's gate made a WAVE, so that `|a| <= 1` is arithmetic
rather than luck.

    a_j   =  m_j * exp(i theta_j),   m_j = clamp(u_j, 0, 1)      CLOSED
    C_j   =  sum_{k<=j} log m_k  +  i sum_{k<=j} theta_k         prefix-PHASE
    l_ij  =  q_ij - Re C_j + s_j                       j = 0..i  (real)
    A_ij  =  softmax_j(l_i.) * exp(i (Im C_i - Im C_j))          (complex)
    O_i   =  sum_{j<=i} A_ij V_j

This is `CEQ_V15_3_DELTA.md` X36. It replaces `ceq/arm_pl.py` and is a drop-in
for the same harness: same constructor shape, same `forward(x) -> [n]` reading
position `s - 1`, same `label_cell` record.

THE MEASUREMENT IT IS AIMED AT, and it is a measurement and not a hunch.
`V15_R1.md` section 7 ran ARM PL at BED-M `t* = 2, n = 2048, N = 8`. Five seeds
crossed `floor_1` with `a_hat_max` in `[1.10, 1.51]`; three diverged with
`a_hat_max` of `20.31`, `49.66` and `285.07` -- above `1`, where the `1/(1-a)`
value rescale is undefined. There is an order of magnitude between the two
populations with nothing in it. `clamp(u, 0, 1)` is the whole delta: the same
parameter values that produced `285.07` produce `1.0` here.

WHY A HARD CAP AND NOT A SIGMOID. The delta's word is **CLOSED**: `0` and `1`
are attainable VALUES, not limits. LRU (Orvieto 2023) is `lambda =
exp(-exp nu + i theta)`, whose magnitude is OPEN -- it reaches neither endpoint
at any finite `nu`. BED-M's own coefficients are `a in {-1, 0, +1}`, i.e.
`m in {0, 1}` and `theta in {0, pi}`, both endpoints, so an open magnitude
cannot represent the corpus it is trained on. `V15_R1.md` section 1 measured
exactly that hole from ARM PL's side: `g = log a` reads `nan` at `a = -1`,
`-inf` at `a = 0` and the value rescale is `inf` at `a = +1`. The closed cap
closes the GATE half of that hole. It does not close the value half -- see
`label_cell` and BIND 5 in the tests, where the band cell is measured failing.

THE QUERY-SIDE SCAN COMES BACK, ON THE PHASE SIDE ONLY. `V15_JUPITER2_FORK.md`
section 8.4 item 3 deleted `C_i` from the logit because a causal softmax
annihilates any term constant across `j` in row `i`. That argument is about the
LOGIT. `exp(i Im C_i)` is a multiplicative factor OUTSIDE the normalizer, so it
survives, and it has to: the parity of the path `j -> i` is a function of both
endpoints. The magnitude half is still key-only.

THE ROW OF MODULI IS A PROBABILITY VECTOR; THE COMPLEX ROW IS NOT. `|A_i.|`
sums to `1` at every setting (`normalizer`), which is what keeps the object
inside the softmax class. `sum_j A_ij` is not `1`, because the twist is the
parity carrier and not a probability. Stated here so no column can quote a
complex row sum as a conservation failure.

NOTHING HERE TRAINS (L-LEAN). No optimizer, no gradient step, no fit. Every
number this module produces is a float64/complex128 identity residual or a
bitwise comparison.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from scale import identity_manifest

NAME = "arm_phase"
#: `key_only` for the MAGNITUDE, `sym` for the PHASE. The manifest records the
#: pair, so a cell measured under a key-only phase cannot be filed under this.
VARIANT = "key_only_mag+sym_phase"
DTYPE = torch.float64
CDTYPE = torch.complex128
BOS = 0

#: The (L) setting's mutilations. `drop_phase` is this arm's own -- the other
#: four are ARM PL's, kept by name so the two arms' rejection regions are
#: comparable. The BIND 1 mutilation is not here: it is `gate(..., cap=False)`,
#: which is a mutilation of the PARAMETRIZATION rather than of the setting.
MUTATIONS = ("none", "drop_key_bias", "drop_value_rescale",
             "drop_bos_sink", "half_key_bias", "drop_phase")

PHASE_FIELDS = ("variant", "m_setting", "theta_setting", "s_setting",
                "bos_value", "m_max", "v_max", "dyn_range_bound")


def _ctype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else CDTYPE


# ------------------------------------------------------- the parametrization

def magnitude(u: torch.Tensor) -> torch.Tensor:
    """`m = clamp(u, 0, 1)`. THE HARD CAP, and the whole of BIND 1.

    `clamp` and not `sigmoid`: the endpoints must be ATTAINED. `clamp` maps the
    entire half-line `u >= 1` to exactly `1.0` and `u <= 0` to exactly `0.0`,
    including `+-inf`, so `|a| <= 1` is a property of the arithmetic and not of
    the parameter values a run happens to reach.
    """
    return torch.clamp(u, 0.0, 1.0)


def gate(u: torch.Tensor, theta: torch.Tensor, *, cap: bool = True) -> torch.Tensor:
    """`a = m * exp(i theta)`, complex.

    `cap=False` is PLANTED NEGATIVE 1: the open magnitude ARM PL had, kept in
    the shipped module so the bind's rejection region is occupied by the arm's
    own code rather than by a test-local copy of it.
    """
    m = magnitude(u) if cap else u
    return m.to(_ctype(m.dtype)) * torch.polar(torch.ones_like(theta), theta)


def scan_phase(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """`C = cumsum(log m) + i cumsum(theta)`, the prefix-PHASE.

    The path product `j -> i` is `exp(C_i - C_j)`, whose modulus is
    `exp(Re C_i - Re C_j) = prod m_k <= 1` by the cap and `= 1` exactly on the
    band `m = 1`. That inequality is what `Lean #6` bounded and what `#16` turns
    into an equality.

    UNDEFINED PAST A ZERO GATE, and `m = 0` is attainable by design: once a
    magnitude is `0`, `Re C` is `-inf` at every later position and `C_i - C_j`
    is `nan` for two of them. `torch.cumprod` on `gate(...)` returns the true
    `0`. Reported rather than repaired -- ARM PL has the identical hole at
    `a = 0`, where `g = log 0 = -inf` (`V15_R1.md` section 1).
    """
    return torch.complex(torch.cumsum(torch.log(m), dim=-1),
                         torch.cumsum(theta, dim=-1))


def key_bias(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`-Re C_j + s_j`. The REAL half of the difference from a standard head,
    on the key side, which is the only side a causal softmax lets a scan into."""
    return s - torch.cumsum(torch.log(m), dim=-1)


def phase_factor(theta: torch.Tensor) -> torch.Tensor:
    """`exp(i (Phi_i - Phi_j))`, `[..., S, S]`, `Phi = cumsum(theta)`.

    The IMAGINARY half, and it is two-sided. It is a multiplicative factor
    outside the softmax, so unlike a logit term it is not annihilated by being
    constant across `j`.
    """
    phi = torch.cumsum(theta, dim=-1)
    d = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    return torch.polar(torch.ones_like(d), d)


def identity_setting(n: int, dtype: torch.dtype = DTYPE, device=None):
    """`(u, theta, s) = (1, 0, 0)`: the standard-attention identity.

    `m = clamp(1) = 1` exactly, so `log m = 0.0`, `Re C = 0.0` and the key bias
    is exactly `0.0`; `Phi = 0` so the twist is exactly `1 + 0j`. This is
    `a = 1`, the cap's upper endpoint -- the same boundary ARM PL's `g = 0` sat
    on (`section 8.4` item 5), reached from inside a closed interval instead of
    from an open one.
    """
    o = torch.ones(n, dtype=dtype, device=device)
    return o, torch.zeros_like(o), torch.zeros_like(o)


# --------------------------------------------------------------- the operator

def operator(q: torch.Tensor, k: torch.Tensor,
             u: torch.Tensor | None = None, theta: torch.Tensor | None = None,
             s: torch.Tensor | None = None, *, diagonal: int = 0) -> torch.Tensor:
    """The `[..., S, S]` complex operator: a real causal softmax row times the
    unimodular twist.

    `diagonal = 0` includes `j = i`, which the label needs (`P_ii = 1`). At the
    identity setting the bias is exactly `0.0` and the twist exactly `1 + 0j`,
    so `.real` is BITWISE `ceq.lm.Attention("softmax_x").operator` and `.imag`
    is exactly zero. That is the whole of what the (P) bind claims.
    """
    n = q.shape[-2]
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    du, dev = w.dtype, w.device
    m = magnitude(torch.ones(n, dtype=du, device=dev) if u is None else u)
    th = torch.zeros(n, dtype=du, device=dev) if theta is None else theta
    sb = torch.zeros(n, dtype=du, device=dev) if s is None else s
    w = w + key_bias(m, sb).unsqueeze(-2)
    nm = ~torch.ones(n, n, dtype=torch.bool, device=dev).tril(diagonal)
    p = torch.softmax(w.masked_fill(nm, torch.finfo(du).min),
                      -1).masked_fill(nm, 0.0)
    return p.to(_ctype(du)) * phase_factor(th)


def readout(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            u: torch.Tensor | None = None, theta: torch.Tensor | None = None,
            s: torch.Tensor | None = None, *, diagonal: int = 0) -> torch.Tensor:
    """`O_i = sum_{j<=i} A_ij V_j`, complex."""
    a = operator(q, k, u, theta, s, diagonal=diagonal)
    return a @ v.to(a.dtype)


def normalizer(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`Z_i = exp(Re C_i) * sum_{j<=i} exp(s_j - Re C_j)`.

    At the oracle setting `exp(-R_j)(1 - m_j)` telescopes to `exp(-R_i) - 1`
    plus the BOS `1`, so `Z_i = 1` EXACTLY. A statement about the MAGNITUDES
    only: the twist has modulus `1` and cannot move it.
    """
    r = torch.cumsum(torch.log(m), dim=-1)
    return torch.exp(r) * torch.cumsum(torch.exp(s - r), dim=-1)


# ------------------------------------------------------------- Z2 winding

def winding(theta: torch.Tensor):
    """`(W_i, residual)`: the per-position `Z` winding `Phi_i / pi`, rounded to
    an integer, with the worst distance to that integer beside it.

    `CEQ_V15_3_DELTA.md` X37 (a): the winding is an INTEGER, and a non-integer
    reading is an instrument defect. The residual is what says which happened.
    """
    r = torch.cumsum(theta, dim=-1) / math.pi
    w = torch.round(r)
    return w.long(), float((r - w).abs().max())


def winding_matrix(theta: torch.Tensor) -> dict:
    """`W_ij = (Phi_i - Phi_j)/pi` rounded, and `(-1)^W` -- the parity mask read
    off the phase. `parity` is exactly `+-1`, never a rounded float."""
    phi = torch.cumsum(theta, dim=-1)
    r = (phi.unsqueeze(-1) - phi.unsqueeze(-2)) / math.pi
    w = torch.round(r)
    return {"winding": w.long(), "residual": float((r - w).abs().max()),
            "parity": 1.0 - 2.0 * (w.long() % 2).to(theta.dtype)}


def parity_sign_mask(p: torch.Tensor, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """`lean/CEQ/V15.lean` `parity_sign`, in float64:
    `chi(pscan p i - pscan p j) = prod_{k=j+1}^{i} chi(p k)`, `chi` the sign
    character of `ZMod 2`. `pscan` is the inclusive prefix sum mod 2.

    The repo's convention, restated here from the theorem rather than from the
    phase construction -- comparing the phase arm to a mask derived FROM the
    phase arm would be `V-3`.
    """
    ps = torch.cumsum(p.long(), dim=-1)
    d = (ps.unsqueeze(-1) - ps.unsqueeze(-2)) % 2
    return 1.0 - 2.0 * d.to(dtype)


# ------------------------------------------------- the band modulus, BIND 2

def band_modulus(n: int = 10_000, seed: int = 15,
                 band_magnitude: float = 1.0, device=None) -> dict:
    """`[RUN: 1e4 phases -> modulus 1.000000000000]`, both routes.

    `prefix_route` is the construction: `|exp(C_n - C_0)|`. `product_route` is a
    numpy cumulative complex product, which shares no code with it -- two
    algorithms rather than one identity restated twice (`V-3`).

    `band_magnitude` is PLANTED NEGATIVE 2: at `0.9` the modulus must collapse,
    which is what makes this a statement about `m` and not about `exp(i theta)`.

    `device` moves the PREFIX route only. The product route stays on the host
    ON PURPOSE and is not a device omission: numpy has no cuda, and the whole
    point of the second route is that it shares no code with the first. It is
    fed `.cpu()` copies of the same tensors, so what it checks on a cuda run is
    `cuda prefix scan` against `host cumulative product` -- a stronger reading
    than the cpu run's, not a weaker one. The three `.numpy()` calls were bare
    until V16 and raised `TypeError` on a cuda tensor (`V16_BAR_RECERT.md` F4).
    """
    rng = np.random.default_rng(seed)
    th = torch.from_numpy(rng.uniform(-math.pi, math.pi, n)).to(device)
    m = torch.full((n,), float(band_magnitude), dtype=DTYPE, device=device)
    c = scan_phase(m, th)
    prefix = torch.exp(c).abs()
    host_m, host_th = m.cpu().numpy(), th.cpu().numpy()
    prod = np.abs(np.cumprod(host_m * np.exp(1j * host_th)))
    return {"n": n,
            "device": (m.device.type),
            "prefix_route": float(prefix[-1]),
            "product_route": float(prod[-1]),
            "max_abs_dev": float((prefix - 1.0).abs().max()),
            "max_route_gap": float(np.abs(prefix.cpu().numpy() - prod).max()),
            #: the modulus is bounded ABOVE by 1 in float64 as well as in the
            #: reals -- the one-ulp misses are all misses DOWNWARD.
            "n_exactly_one": int((prefix == 1.0).sum()),
            "n_above_one": int((prefix > 1.0).sum())}


# ------------------------------------------------------------- the (L) setting

def oracle_heads(a: torch.Tensor, b: torch.Tensor):
    """`(u, theta, s, V)` at the (L) setting, for COMPLEX `a`:
    `u_j = |a_j|`, `theta_j = arg a_j`, `s_j = log(1 - |a_j|)`,
    `V_j = b_j / (1 - |a_j|)`, and `u_0 = 1, theta_0 = s_0 = V_0 = 0` at BOS.

    `u_0 = 1` and not `0`: the BOS carries no gate, and `m_0 = 1` is what makes
    `log m_0 = 0` so the BOS logit is exactly `s_0`. `a[0]` is never read.
    """
    m = a.abs()
    u = torch.ones_like(m)
    th = torch.zeros_like(m)
    s = torch.zeros_like(m)
    v = torch.zeros_like(m)
    u[..., 1:] = m[..., 1:]
    th[..., 1:] = torch.angle(a[..., 1:])
    s[..., 1:] = torch.log1p(-m[..., 1:])
    v[..., 1:] = b[..., 1:] / (1.0 - m[..., 1:])
    return u, th, s, v


def chain_label(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`y_0 = 0`, `y_i = a_i y_{i-1} + b_i`, complex. By its own recurrence, so
    that comparing it to the head's read-out is a check and not an identity."""
    y = torch.zeros_like(a)
    bb = b.to(a.dtype)
    for i in range(1, a.shape[-1]):
        y[..., i] = a[..., i] * y[..., i - 1] + bb[..., i]
    return y


def draw(seed: int = 15, s: int = 8, lo: float = 0.15, hi: float = 0.85,
         device=None):
    """The fork probe's draw made complex: `|a|` uniform on `(lo, hi)`, `arg a`
    uniform on `(-pi, pi)`, drives standard normal, BOS at index 0 poisoned with
    `nan` so any code path that reads it says so loudly.

    `device` moves the result AFTER `default_rng` drew it, so the bytes are the
    host draw on every device (`V16_BAR_RECERT.md` §6.1's construct-then-move
    ordering). A device generator would draw different numbers and retire every
    residual `V15_ARM_PHASE.md` published.
    """
    rng = np.random.default_rng(seed)
    m = np.empty(s + 1)
    m[0] = np.nan
    m[1:] = rng.uniform(lo, hi, size=s)
    th = np.zeros(s + 1)
    th[1:] = rng.uniform(-math.pi, math.pi, size=s)
    b = np.zeros(s + 1)
    b[1:] = rng.normal(size=s)
    return (torch.from_numpy(m * np.exp(1j * th)).to(device),
            torch.from_numpy(b).to(device))


def band_draw(seed: int = 15, s: int = 8, magnitude: float = 1.0, device=None):
    """BED-M's own coefficients: `a in {-1, +1}` on the live band, i.e. `m = 1`
    and `theta in {0, pi}` -- both ENDPOINTS of the closed cap.

    `scale/negation_scope.py:428` draws exactly this. It is the setting the
    delta is aimed at and the setting the label bind cannot be evaluated at.

    `magnitude < 1` walks the same draw in from the band, so the label bind's
    divergence rate can be read as a ladder rather than as one `nan`.
    """
    rng = np.random.default_rng(seed)
    th = np.zeros(s + 1)
    th[1:] = math.pi * rng.integers(0, 2, size=s)
    m = np.full(s + 1, float(magnitude))
    m[0] = np.nan
    b = np.zeros(s + 1)
    b[1:] = rng.normal(size=s)
    return (torch.from_numpy(m * np.exp(1j * th)).to(device),
            torch.from_numpy(b).to(device))


def mutate(u, th, s, v, b, mutation: str):
    """One of `MUTATIONS`, applied to the oracle setting. Each removes exactly
    one part of the construction and nothing else."""
    if mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}; expected one of {MUTATIONS}")
    if mutation == "drop_key_bias":
        return u, th, torch.zeros_like(s), v
    if mutation == "drop_value_rescale":
        return u, th, s, b.clone().to(v.dtype)
    if mutation == "drop_bos_sink":
        v = v.clone()
        v[..., BOS] = 1.0
        return u, th, s, v
    if mutation == "half_key_bias":
        return u, th, s / 2.0, v
    if mutation == "drop_phase":
        return u, torch.zeros_like(th), s, v
    return u, th, s, v


def label_cell(a: torch.Tensor, b: torch.Tensor, *, mutation: str = "none",
               seed: int = 15, cell: str | None = None) -> dict:
    """One measured cell of the (L) bind, with its manifest and its
    dynamic-range column.

    `q = 0` at the oracle setting, inherited from ARM PL and from section S-M's
    own oracle-gate bind: at this theta the head is not doing content
    addressing.

    `dyn_range_bound = 1/(1 - m_max)` is `inf` on the band, and that is the
    cost this construction did not remove.
    """
    u, th, s, v = mutate(*oracle_heads(a, b), b, mutation)
    n = a.shape[-1]
    zero_q = torch.zeros(n, 1, dtype=DTYPE, device=a.device)
    out = readout(zero_q, zero_q, v, u, th, s)
    y = chain_label(a, b)
    m_max = float(a[..., 1:].abs().max())

    record = {
        "cell": cell or f"{NAME}:{mutation}",
        "kind": NAME,
        "task": "chain_label_complex",
        "s": int(n - 1),
        "d": 1,
        "d_model": 1,
        "steps": 0,                   # L-LEAN: nothing is trained by this node
        "seed": seed,
        #: THE DEVICE THIS CELL WAS MEASURED ON, read off the tensors it was
        #: measured with. It was the literal `"cpu"` until V16, which made a
        #: cuda run of this arm hash IDENTICALLY to a cpu one --
        #: `identity_manifest.CONFIG_FIELDS` carries `device` as a first-class
        #: field and the literal was filling it with a constant, so the
        #: manifest asserted the run happened somewhere it had not
        #: (`V16_BAR_RECERT.md` F4). This arm is the one
        #: `CEQ_V16_CONTRACT.md` names for R1', so the constant was on the
        #: deciding measurement's identity record.
        #: `.type` and not `str(...)`: `"cuda"`, not `"cuda:0"`, so the value
        #: matches `--device`'s vocabulary and the bucket
        #: `refuse_cross_device_pool` reads.
        "device": a.device.type,
        "torch_version": torch.__version__,
        "variant": VARIANT,
        "m_setting": "clamp(|a_j|, 0, 1), m_0 = 1",
        "theta_setting": ("0" if mutation == "drop_phase" else "arg a_j, theta_0 = 0"),
        "s_setting": ("0" if mutation == "drop_key_bias" else
                      "log(1 - |a_j|) / 2, s_0 = 0" if mutation == "half_key_bias"
                      else "log(1 - |a_j|), s_0 = 0"),
        "bos_value": float(v[..., BOS].real),
        "m_max": m_max,
        "v_max": float(v.abs().max()),
        "dyn_range_bound": (math.inf if m_max >= 1.0 else 1.0 / (1.0 - m_max)),
    }
    record["residual"] = float((out - y).abs().max())
    record["manifest"] = cell_manifest(
        record, callables=(operator, readout, oracle_heads, normalizer),
        params={"u": u, "theta": th, "s": s, "V": v})
    return record


def cell_manifest(record: dict, *, callables, params) -> dict:
    """`identity_manifest.manifest` plus the ARM-PHASE block `PHASE_FIELDS`.

    The base manifest hashes `CONFIG_FIELDS` only, which has no slot for the
    variant, the three head settings, the BOS value or the dynamic-range
    column -- and a manifest that cannot tell a mutilated cell from an honest
    one is not an identity. `scale/identity_manifest.py` is another node's file
    and every published hash depends on `CONFIG_FIELDS`, so the block is hashed
    here and folded in rather than added there.
    """
    m = identity_manifest.manifest(record, callables=callables, params=params)
    values = {f: record.get(f) for f in PHASE_FIELDS}
    blk = identity_manifest._sha(identity_manifest._canon(values))
    m["pl"] = blk
    m["pl_values"] = values
    m["hash"] = identity_manifest._sha(m["hash"].encode(), blk.encode())
    return m


# ------------------------------------------------------------- the drop-in arm

class ArmPhase(nn.Module):
    """ARM PHASE as `scale/m3_capability.py`'s harness wants it.

    Same shape as `m3_capability.Arm` and as `arm_pl.ArmPL` -- `wq`/`wk`, one
    shared-width MLP, a scalar readout, `forward(x) -> [n]` reading position
    `s - 1`. THREE per-position scalar heads instead of ARM PL's two: the
    magnitude `u` (capped), the phase `theta`, and the key bias `s`. That is
    `+17` parameters over ARM PL at `d_model = 16`, one `nn.Linear(16, 1)`.

    `s` is NOT tied to `m` by `s = log(1 - m)`, which would save the head: that
    tie is `-inf` at `m = 1`, which is precisely the identity setting, so the
    tie would delete the parity bind to save seventeen parameters.

    `identity_heads()` -- not `zero_heads()`, and the name is the finding: the
    identity point is `u = 1`, not `u = 0`. The magnitude head's BIAS is the
    thing set to one.

    The readout takes `Re(A @ x)`. At the identity setting `A` is real, so this
    is bitwise `A @ x` and the parity bind survives the whole forward pass.

    NOT TRAINED HERE and not by this node (L-LEAN). No optimizer, no gradient.
    """

    def __init__(self, s: int, d_model: int = 16, hidden: int = 128):
        super().__init__()
        self.kind = NAME
        self.seq = s
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.m_head = nn.Linear(d_model, 1)
        self.theta_head = nn.Linear(d_model, 1)
        self.s_head = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_model))
        self.readout = nn.Linear(d_model, 1)

    def identity_heads(self) -> "ArmPhase":
        """`m = 1, theta = 0, s = 0` exactly: the standard-attention setting.

        A CORRECTNESS POINT, NOT A TRAINING INITIALISATION -- `MISTAKES.md`
        V-29, whose closing paragraph names this method as the live half of the
        defect. The exactness is the whole value of it (`tests/arm_phase/
        test_arm_phase.py` asserts `torch.equal(magnitude(u), ones)`,
        `torch.equal(th, zeros)` and `torch.equal(s, zeros)` on what this
        returns, and `scripts/v16_device_probe.py` reads the same corner) and it
        is exactly what makes it untrainable: `m = 1` is the closed endpoint of
        `magnitude`'s clamp, whose backward is zero AT the endpoint, and
        `theta = 0` is a critical point of `Re(m e^{i theta})` for every `m`.
        FOUR of the module's fourteen tensors receive an exactly zero gradient
        from this point -- `m_head.weight`, `m_head.bias`, `theta_head.weight`,
        `theta_head.bias`. Train from `trainable_heads()` instead; nothing else
        about the operator changes.

        FOUR, NOT THE FIVE V-29 PREDICTED, and the fifth is a different animal.
        `s_head.bias` is not dead at this corner; it is dead at EVERY point,
        because `key_bias` adds `s_j` to the logit row and a causal softmax is
        invariant under a constant added across `j`. Its analytic gradient is
        zero for every parameter value, so no initialisation can wake it and a
        `|grad| > 0` guard reads its float64 rounding residue (`8.673617e-19`
        at `S=8, d_model=16` on BED-M `t*=2`) as life. `trainable_heads`
        declares it non-trainable rather than leaving it in the trainable set
        at zero, which is V-29's own rule for a parameter that must sit on a
        flat direction.
        """
        with torch.no_grad():
            for h in (self.m_head, self.theta_head, self.s_head):
                h.weight.zero_()
                h.bias.zero_()
            self.m_head.bias.fill_(1.0)
        return self

    #: How far `trainable_heads` steps each head bias off its own critical
    #: point. `ceq/arm_smprime.py:GATE_INIT_OFF` is the same number for the
    #: sibling arm and is NOT imported: `arm_smprime` imports `band_draw` and
    #: `chain_label` from this module, so the reverse import is a cycle. The
    #: two are bound to each other by assertion in
    #: `tests/arm_phase/test_the_shipped_init_has_a_live_gradient_everywhere.py`
    #: instead, which can import both.
    #:
    #: IT LIVES HERE, INSIDE THE CLASS AND BELOW `identity_heads`, AND THE
    #: PLACEMENT IS THE POINT. `MISTAKES.md` P-15 was filed because the
    #: sibling's identical repair put its constant near the TOP of its file and
    #: shifted every line-citation below it by one. Seven citations in this tree
    #: pin lines in this module -- `:60, :121, :126, :158, :179, :476, :492` --
    #: and every one of them is at or above `identity_heads`. A module-level
    #: constant would have re-fired P-15 on all six below `:60`.
    HEAD_INIT_OFF = 1e-3

    def trainable_heads(self, off: float = HEAD_INIT_OFF) -> "ArmPhase":
        """`identity_heads()`, then each bias stepped `off` off its OWN critical
        point and the gauge taken out of the trainable set -- the door to train
        this arm's gate from. THREE LINES, one per mechanism.

        `m = 1 - off` clears the clamp endpoint and `theta = off` clears the
        phase's flat point; they are two separate mechanisms, so one offset does
        not cover for the other. `s_head.bias.requires_grad_(False)` is the
        third, and it is not an offset because no offset exists: the key-bias
        head's bias is an exact softmax gauge (see `identity_heads`). The
        forward is invariant to its value -- exactly in the reals, and to
        `2.775558e-17` (one ulp of the read-out) at a unit shift in float64.
        Not "bitwise": the cancellation is exact and the rounding is not, and
        claiming bitwise here would claim more than the arithmetic gives.

        NOT LoRA INIT, AND THE REASON IS THE PARAMETRIZATION. LoRA's one-random-
        one-zero split is a statement about a PRODUCT `B A`; each head here is a
        single `nn.Linear(d_model, 1)`, i.e. `W x + b`, so there is no pair of
        factors to split. Forced -- the head output driven identically to zero,
        which is all a zero factor can buy -- it lands on `m = clamp(0, 0, 1) =
        0.0`, the clamp's LOWER endpoint, where `clamp'` is zero again, where
        `log m = -inf` puts `+inf` into the key bias, and where the loss reads
        `nan`. `scripts/v15_r1.py::identity_point` already names `m = 0` the
        drop-in trap for this arm. Measured in
        `test_lora_init_does_not_apply_and_would_make_it_worse`.

        Measured at `off = 1e-3` on a `[4, 8, 16]` BED-M `t* = 2` batch
        (`make_equilibrium_batch(4, 8, 4, t_star=2, d_model=16, seed=12345)`,
        `python -m pytest tests/arm_phase/test_the_shipped_init_has_a_live_gradient_everywhere.py -q -s`,
        `WIN-16QAL06O9GB`, python 3.11.9, torch 2.14.0+cpu): the four
        corner-dead tensors go `m_head.weight 0.000000e+00 -> 7.905095e-03`,
        `m_head.bias 0.000000e+00 -> 4.067769e-02`,
        `theta_head.weight 0.000000e+00 -> 4.709942e-05`,
        `theta_head.bias 0.000000e+00 -> 3.800616e-04`; all 13 declared-
        trainable tensors are live, and the real softmax row moves
        `4.721359e-04` off the corner -- 106 times inside the `5e-2` the
        sibling's door is held to. That price, a start NEAR the corner rather
        than ON it, is what a gradient costs here and it is not hidden.
        """
        self.identity_heads()
        with torch.no_grad():
            self.m_head.bias.fill_(1.0 - off)
            self.theta_head.bias.fill_(off)
        self.s_head.bias.requires_grad_(False)
        return self

    def heads(self, x: torch.Tensor):
        return (self.m_head(x).squeeze(-1), self.theta_head(x).squeeze(-1),
                self.s_head(x).squeeze(-1))

    @torch.no_grad()
    def gate_feature(self, x: torch.Tensor, live) -> torch.Tensor:
        """`Re(a_hat_j)` at the live positions, `[n*len(live), 1]`.

        `V15_R1.md` section 6's discriminating instrument is the gate `R^2` of
        `a_hat = exp(g)` against the corpus's `a`. `exp(g)` is ARM PL's whole
        gate; this arm's is `m exp(i theta)`, whose real part is the direct
        analogue and the only part `a in {-1, 0, +1}` can be regressed on.
        """
        self.eval()
        u, th, _ = self.heads(x)
        return (magnitude(u) * torch.cos(th))[:, live].reshape(-1, 1)

    def _operator(self, q, k, u, theta, s) -> torch.Tensor:
        return operator(q, k, u, theta, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, seq, _ = x.shape
        u, th, s = self.heads(x)
        a = self._operator(self.wq(x), self.wk(x), u, th, s)
        h = self.mlp(x + (a @ x.to(a.dtype)).real)
        return self.readout(h).squeeze(-1)[:, seq - 1]
