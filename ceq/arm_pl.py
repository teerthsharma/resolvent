"""ARM PL -- one causal softmax head that carries BOTH the parity bind and the
path-product label bind.

    l_ij  =  q_ij  -  C_j  +  s_j          j = 0..i,   C = scan(g)
    O_i   =  sum_{j<=i} softmax_j(l_i.) * V_j

`g` and `s` are two independent per-position scalar heads; position `0` is a
BOS slot. This is `V15_JUPITER2_FORK.md` §8.3, settled there and NOT redesigned
here, and it is the float64 shadow of `lean/CEQ/V15Fork.lean`:

    Asink_row_sum              the row sums to 1, with NO hypothesis on `a`
    Asink_nonneg               0 < a < 1 makes every entry non-negative
    Asink_computes_chain       on `Vsink` it reproduces the chain label exactly
    gate_zero_key_logit_identity   q + (-scan g j + s j) = q at g = s = 0
    Asink_eq_hop               the whole repair is one factor (1 - a_j) on §S-M's hop

THE QUERY-SIDE SCAN TERM IS NOT IMPLEMENTED, AND THAT IS THE SETTLED FORM.
§S-M's logit was `q_ij + (C_i - C_j) + s_j`. Under a causal softmax the `C_i`
half is annihilated identically -- it is constant across `j` in row `i` and
cancels between numerator and denominator -- so `V15_JUPITER2_FORK.md` §4
measured `max |out - out with C_i deleted| = 0` and §8.4 item 3 deletes it. The
target factor `exp(C_i)` comes back out of the RUNNING NORMALIZER, which
telescopes to `exp(-C_i)`, not out of the logit. A scan can only enter
key-side. `key_bias` is therefore `-C_j + s_j` and `VARIANT` says so.

WHAT THE (P) BIND SAYS, in the words that are true of it (MISTAKES.md V-24,
closing paragraph): **the modification enters only through the key logit,
additively, and vanishes at zero.** Not "bitwise standard attention"
unqualified -- that was §S-M's original clause and `gate_zero_not_stochastic`
refuted it. The parity point here is "both auxiliary heads zero", which is a
narrower claim than "the gate's own identity point", and the two binds sit at
opposite ends of one degeneration: (P) is `a = 1` and (L) needs `a != 1`
(§8.4 item 5). There is no theta at which both hold, and there was never going
to be.

THE DIAGNOSTIC COLUMN IS A PREDICTED FAILURE MODE, FILED BEFORE THE ARM RUNS.
§8.4 cost 6: the value dynamic range is `1/(1 - a)`, reaching `1e6` at
`a = 1 - 1e-6`. `label_cell` emits `v_max` against `dyn_range_bound` as a
per-cell column, because the prediction is that ARM PL fails first by VALUE
SATURATION on long-memory chains rather than by gate error.

NOTHING HERE TRAINS. No optimizer, no gradient step, no fit (L-LEAN). Every
number this module produces is a float64 identity residual or a bitwise
comparison. `ArmPL` exists so the next node has a drop-in for
`scale/m3_capability.py`'s harness; this node does not call it with an
optimizer and neither should any reader of these binds.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import torch.nn as nn

#: The identity manifest is `scale/identity_manifest.py`'s, not a second copy.
#: `ceq` importing `scale` is the one direction this repo did not already have;
#: the alternative was a private clone of `CONFIG_FIELDS`, and two copies of an
#: identity rule is the defect that rule exists to catch.
from scale import identity_manifest

NAME = "arm_pl"
#: `key_only` and not `sym`: the logit is `-C_j + s_j`, with no `C_i` term. The
#: manifest records this, so a cell measured under the query-side form cannot be
#: filed under this one.
VARIANT = "key_only"
DTYPE = torch.float64
BOS = 0

#: The (L) setting's four deliberate mutilations. `none` is the honest cell.
#: Every one of the other four must FAIL the label bind at O(1) -- that is what
#: makes the bind a gate rather than an assertion (MISTAKES.md rule 8, V-24).
MUTATIONS = ("none", "drop_key_bias", "drop_value_rescale",
             "drop_bos_sink", "half_key_bias")

#: What the manifest records BEYOND `identity_manifest.CONFIG_FIELDS`: the
#: operator variant, both head settings, the BOS slot, and the dynamic-range
#: column. `CONFIG_FIELDS` is not edited (it is another node's file and every
#: published hash depends on it), so these are hashed as a second block and
#: folded into the combined hash.
PL_FIELDS = ("variant", "g_setting", "s_setting", "bos_value",
             "a_max", "v_max", "dyn_range_bound")


# --------------------------------------------------------------- the operator

def scan(g: torch.Tensor) -> torch.Tensor:
    """`C = scan(g)`, the INCLUSIVE prefix sum, matching `CEQ.V15.scan`."""
    return torch.cumsum(g, dim=-1)


def key_bias(g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`-C_j + s_j`. The entire architectural difference from a standard head,
    on the KEY side, where a causal softmax is the only place a scan can go."""
    return s - scan(g)


def operator(q: torch.Tensor, k: torch.Tensor,
             g: torch.Tensor | None = None, s: torch.Tensor | None = None,
             *, diagonal: int = 0) -> torch.Tensor:
    """The [..., S, S] causal softmax row of `q_ij + key_bias_j`.

    `diagonal = 0` includes `j = i`, which the label needs (`P_ii = 1`).
    `diagonal = -1` is `ceq.bench`'s strictly-causal convention, kept as an
    argument only so the parity bind can be run against that reference too.

    At `g = s = 0` the bias is exactly `0.0` and `w + 0.0` is bitwise `w`, so
    the returned matrix is bitwise the reference operator. That is the whole of
    what the (P) bind claims.
    """
    n = q.shape[-2]
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if g is not None or s is not None:
        z = torch.zeros(n, dtype=w.dtype, device=w.device)
        w = w + key_bias(z if g is None else g, z if s is None else s).unsqueeze(-2)
    nm = ~torch.ones(n, n, dtype=torch.bool, device=w.device).tril(diagonal)
    return torch.softmax(w.masked_fill(nm, torch.finfo(w.dtype).min),
                         -1).masked_fill(nm, 0.0)


def readout(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            g: torch.Tensor | None = None, s: torch.Tensor | None = None,
            *, diagonal: int = 0) -> torch.Tensor:
    """`O_i = sum_{j<=i} A_ij V_j`. `v` may be [S] or [..., S, D]."""
    return operator(q, k, g, s, diagonal=diagonal) @ v


def normalizer(g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`Z_i = exp(C_i) * sum_{j<=i} exp(s_j - C_j)`.

    THE MECHANISM, and it is measurable rather than argued: at the oracle
    setting `exp(-C_j)(1 - a_j) = exp(-C_j) - exp(-C_{j-1})` telescopes to
    `exp(-C_i) - 1`, so `Z_i = 1` EXACTLY. The normalizer REPRODUCES the target
    factor instead of cancelling it, which is what the additive-logit form of
    `V15_JUPITER2_FORK.md` §2 could not do. `Asink_row_sum` is this with no
    hypothesis on `a` at all.
    """
    c = scan(g)
    return torch.exp(c) * torch.cumsum(torch.exp(s - c), dim=-1)


# ------------------------------------------------------------- the (L) setting

def oracle_heads(a: torch.Tensor, b: torch.Tensor):
    """`(g, s, V)` at the (L) setting: `g_j = log a_j`, `s_j = log(1 - a_j)`,
    `V_j = b_j / (1 - a_j)`, and `g_0 = s_0 = V_0 = 0` at the BOS slot.

    `a[0]` is never read -- the BOS has no gate -- and the draw poisons it with
    NaN so that any code path that does read it says so loudly.
    `log1p(-a)` rather than `log(1 - a)`: same value, one fewer cancellation.
    """
    g = torch.zeros_like(b)
    s = torch.zeros_like(b)
    v = torch.zeros_like(b)
    g[..., 1:] = torch.log(a[..., 1:])
    s[..., 1:] = torch.log1p(-a[..., 1:])
    v[..., 1:] = b[..., 1:] / (1.0 - a[..., 1:])
    return g, s, v


def chain_label(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`y_0 = 0`, `y_i = a_i y_{i-1} + b_i`. By its own recurrence, so that
    comparing it to the head's read-out is a check and not an identity."""
    y = torch.zeros_like(b)
    for i in range(1, b.shape[-1]):
        y[..., i] = a[..., i] * y[..., i - 1] + b[..., i]
    return y


def draw(seed: int = 15, s: int = 8, lo: float = 0.15, hi: float = 0.85,
         device=None):
    """The fork probe's draw, so every number here is comparable to the ones
    `V15_JUPITER2_FORK.md` §4 published: `default_rng(seed)`, gates uniform on
    `(lo, hi)`, drives standard normal, BOS slot at index 0.

    `device` moves the drawn tensors AFTER `default_rng` has produced them, so
    the bytes are the host draw on every device -- the same construct-then-move
    ordering `V16_BAR_RECERT.md` §6.1 verified byte-identical for the corpus.
    A device generator would draw different numbers and silently retire every
    published residual below.
    """
    rng = np.random.default_rng(seed)
    a = np.empty(s + 1)
    a[0] = np.nan                        # BOS has no gate; poisons any misuse
    a[1:] = rng.uniform(lo, hi, size=s)
    b = np.zeros(s + 1)
    b[1:] = rng.normal(size=s)
    return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)


def constant_gate_draw(a_const: float, s: int = 8):
    """`a = a_const`, `b = 1`: §5's row, where `max_i |y_i| = (1-a^s)/(1-a)`
    and the value scale is exactly `1/(1 - a)`."""
    a = np.full(s + 1, a_const)
    a[0] = np.nan
    b = np.ones(s + 1)
    b[0] = 0.0
    return torch.from_numpy(a), torch.from_numpy(b)


def mutate(g: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
           b: torch.Tensor, mutation: str):
    """One of `MUTATIONS`, applied to the oracle setting. Each removes exactly
    one part of the construction and nothing else."""
    if mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}; expected one of {MUTATIONS}")
    if mutation == "drop_key_bias":
        return g, torch.zeros_like(s), v
    if mutation == "drop_value_rescale":
        return g, s, b.clone()                      # V_j = b_j, no 1/(1 - a_j)
    if mutation == "drop_bos_sink":
        v = v.clone()
        v[..., BOS] = 1.0
        return g, s, v
    if mutation == "half_key_bias":
        return g, s / 2.0, v
    return g, s, v


def label_cell(a: torch.Tensor, b: torch.Tensor, *, mutation: str = "none",
               seed: int = 15, cell: str | None = None) -> dict:
    """One measured cell of the (L) bind, with its identity manifest and its
    dynamic-range column.

    `q = 0` at the oracle setting, which is inherent to the bind's design and
    is shared with §S-M's own oracle-gate bind (`V15_JUPITER2_FORK.md` LIMITS):
    at this theta the head is not doing content addressing.
    """
    g, s, v = mutate(*oracle_heads(a, b), b, mutation)
    n = b.shape[-1]
    zero_q = torch.zeros(n, 1, dtype=b.dtype, device=b.device)
    out = readout(zero_q, zero_q, v, g, s)
    y = chain_label(a, b)
    a_max = float(a[..., 1:].max())

    record = {
        "cell": cell or f"{NAME}:{mutation}",
        "kind": NAME,
        "task": "chain_label",
        "s": int(n - 1),
        "d": 1,                       # q/k width at the oracle setting (q == 0)
        "d_model": 1,                 # scalar values
        "steps": 0,                   # L-LEAN: nothing is trained by this node
        "seed": seed,
        #: THE DEVICE THIS CELL WAS MEASURED ON, read off the tensors the cell
        #: was measured with. It was the literal `"cpu"` until V16, which made a
        #: cuda run hash IDENTICALLY to a cpu one -- `identity_manifest`
        #: `CONFIG_FIELDS` carries `device` as a first-class field and the
        #: literal was filling it with a constant, so the manifest asserted the
        #: run happened somewhere it had not (`V16_BAR_RECERT.md` F4).
        #: `.type` and not `str(...)`: `"cuda"`, not `"cuda:0"`, so the value
        #: matches `--device`'s own vocabulary and the one
        #: `refuse_cross_device_pool` buckets on.
        "device": b.device.type,
        "torch_version": torch.__version__,
        "variant": VARIANT,
        "g_setting": "log a_j, g_0 = 0",
        "s_setting": ("0" if mutation == "drop_key_bias" else
                      "log(1 - a_j) / 2, s_0 = 0" if mutation == "half_key_bias"
                      else "log(1 - a_j), s_0 = 0"),
        "bos_value": float(v[..., BOS]),
        "a_max": a_max,
        "v_max": float(v.abs().max()),
        "dyn_range_bound": 1.0 / (1.0 - a_max),
    }
    record["residual"] = float((out - y).abs().max())
    record["manifest"] = cell_manifest(
        record, callables=(operator, readout, oracle_heads, normalizer),
        params={"g": g, "s": s, "V": v})
    return record


# ---------------------------------------------------------- the general DAG

def draw_dag(n: int = 9, seed: int = 15, density: float = 0.6,
             lo: float = 0.05, hi: float = 0.45) -> torch.Tensor:
    """A random DAG's edge-gate matrix: `A[i, j]` is the gate on edge `j -> i`,
    `j < i`. Index order IS a topological order, so `A` is strictly lower
    triangular and hence nilpotent -- `(I - A)^-1` terminates.

    The gate range is an argument because a residual measured only on SMALL
    path sums says little: at `(0.05, 0.45)` the resolvent's entries barely
    leave `1`, so an absolute agreement of `1e-17` is cheap. The heavy regime
    `(0.30, 0.90)` at full density drives them to `O(10)` and is where the
    figure has to be read relative to the quantity.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, (n, n)) * (rng.random((n, n)) < density)
    return torch.from_numpy(np.tril(a, -1))


def dag_resolvent(a: torch.Tensor) -> torch.Tensor:
    """`(I - A)^-1`, by a dense LU solve.

    Deliberately NOT the Neumann sum: an LU factorization knows nothing about
    paths, so comparing it to `brute_force_path_sums` is a check between two
    algorithms rather than an identity restated twice (MISTAKES.md V-3).
    """
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(eye - a, eye)


def brute_force_path_sums(a: torch.Tensor) -> torch.Tensor:
    """`R[i, j]` = the sum over EVERY directed path `j -> i` of the product of
    its edge gates, the empty path included.

    `A` is strictly lower triangular in topological order, so a path from `j`
    to `i` is exactly a strictly increasing vertex sequence `j < ... < i`, and
    the enumeration is over subsets of the open interval. No linear algebra:
    this is the definition, counted.
    """
    m = a.detach().cpu().numpy()
    n = m.shape[0]
    r = np.eye(n)
    for i in range(n):
        for j in range(i):
            total = 0.0
            middle = range(j + 1, i)
            for size in range(len(middle) + 1):
                for pick in itertools.combinations(middle, size):
                    path = (j,) + pick + (i,)
                    w = 1.0
                    for u, vtx in zip(path[:-1], path[1:]):
                        w *= m[vtx, u]
                    total += w
            r[i, j] = total
    return torch.from_numpy(r).to(a.dtype)


# ------------------------------------------------------- the identity manifest

def cell_manifest(record: dict, *, callables, params) -> dict:
    """`identity_manifest.manifest` plus the ARM-PL block `PL_FIELDS`.

    The base manifest hashes only `CONFIG_FIELDS`, which has no slot for the
    operator variant, the head settings, the BOS value or the dynamic-range
    column -- and a manifest that cannot tell a mutilated cell from an honest
    one is not an identity. `scale/identity_manifest.py` is another node's file
    and every published hash depends on `CONFIG_FIELDS`, so the block is hashed
    here and folded in rather than added there.

    The head tensors are passed as `params`, so the `rng` component is a digest
    of the actual `(g, s, V)` this cell was measured at: the head SETTING is the
    parameter vector of an untrained arm.
    """
    m = identity_manifest.manifest(record, callables=callables, params=params)
    pl_values = {f: record.get(f) for f in PL_FIELDS}
    pl = identity_manifest._sha(identity_manifest._canon(pl_values))
    m["pl"] = pl
    m["pl_values"] = pl_values
    m["hash"] = identity_manifest._sha(m["hash"].encode(), pl.encode())
    return m


# ------------------------------------------------------------- the drop-in arm

class ArmPL(nn.Module):
    """ARM PL as `scale/m3_capability.py`'s harness wants it.

    Same shape as `m3_capability.Arm` -- `wq`/`wk` projections, one shared-width
    MLP, a scalar readout, `forward(x) -> [n]` reading position `s - 1` -- so it
    drops into that harness rather than needing a parallel one. Two things
    differ, and they are the whole architecture:

      * two per-position scalar heads `g` and `s`, whose output enters the KEY
        logit additively as `-scan(g)_j + s_j`;
      * the value path, which the (L) setting rescales by `1/(1 - a_j)`.

    `_operator` takes `(q, k, g, s)` and not `(q, k)`: the extra two arguments
    are the arm. `zero_heads()` puts it at the parity point, where its operator
    is bitwise the softmax control.

    NOT TRAINED HERE, and not by this node (L-LEAN). It carries no optimizer and
    this file constructs no gradient.
    """

    def __init__(self, s: int, d_model: int = 16, hidden: int = 128):
        super().__init__()
        self.kind = NAME
        self.seq = s
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.g_head = nn.Linear(d_model, 1)
        self.s_head = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_model))
        self.readout = nn.Linear(d_model, 1)

    def zero_heads(self) -> "ArmPL":
        """The parity point `(g, s) = (0, 0)`, exactly. Returns self."""
        with torch.no_grad():
            for h in (self.g_head, self.s_head):
                h.weight.zero_()
                h.bias.zero_()
        return self

    def heads(self, x: torch.Tensor):
        return self.g_head(x).squeeze(-1), self.s_head(x).squeeze(-1)

    def _operator(self, q: torch.Tensor, k: torch.Tensor,
                  g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return operator(q, k, g, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, seq, _ = x.shape
        g, s = self.heads(x)
        a = self._operator(self.wq(x), self.wk(x), g, s)
        h = self.mlp(x + a @ x)
        return self.readout(h).squeeze(-1)[:, seq - 1]
