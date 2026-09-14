"""P1: an encoder that actually aggregates over the sequence axis, trained alone.

WHAT IS WRONG WITH THE ARM THIS REPLACES. `ceqjepa/pi_jepa.py:387` documents
`x_{<=t} -> s_t, per position` and implements `x_t -> s_t`: an `nn.Sequential`
over the LAST axis with no reduction over the sequence axis. Permuting every
context position leaves the last-position representation bitwise identical, and
zeroing the entire prefix leaves it bitwise identical too. A representation with
no dependence on its context has exponent `alpha = 0` in the token count at any
initialisation, trained or not, which is the construction that retired the
per-coordinate corner rule (`docs/CORNER_RULE_RETIREMENT.md`).

NOTHING HERE IS JOINT. No predictor, no read, no corner assignment, no probe.
One piece, one bar, one contrast. `pi_jepa.py` is NOT modified -- three agents
are reading it -- so this is built beside it and `WHAT WOULD HAVE TO CHANGE`
at the bottom of this docstring names the adoption diff.

THE BAR, AND WHY A POSITION-WISE MAP CANNOT CLEAR IT.

    y_t[d] = scale_d * sum_{j<t} lam_d^(t-1-j) * (x_j . u_d),   y_0 = 0

a bank of nine strict-prefix exponential moving averages over an i.i.d. Gaussian
bed, one decay `lam_d` per latent coordinate. ONE LINE: the sum runs over j < t
strictly and the bed is i.i.d., so `y_t` is independent of `x_t`, hence
`E[y_t | x_t] = E[y_t] = 0` and the MMSE of ANY measurable function of `x_t`
alone is exactly `Var(y_t)` -- which is `R^2 <= 0` pooled over positions. The
ceiling is information-theoretic. No width, no depth and no step count moves it.

WHY THIS STATISTIC AND NOT THE OBVIOUS ONE. A plain prefix MEAN is the obvious
running statistic and it is the wrong one here: it is permutation-invariant over
the prefix, so the permutation control and the bar would disagree -- an encoder
that cleared a prefix-mean bar perfectly SHOULD be order-blind at the last
position, and the control would then be measuring a defect as a virtue. An EMA
is recency-weighted, so order matters, and the control and the bar test the same
property. The decays are spread over `lam in [0, 0.9]` so the nine coordinates
carry nine different effective context lengths rather than one repeated nine
times; that spread is also the surviving referent for "different coordinates
aggregate over different numbers of tokens", which is the goal the retired
corner rule was built to serve.

THE PLANTED NEGATIVE IS THE SHIPPED CLASS ITSELF, imported verbatim from
`pi_jepa` rather than re-implemented, and trained on the identical bed with the
identical optimiser, loss, step count and scored positions. A bar both arms
clear measures nothing.

WHY NOT REUSE `ceq/lm.py:61`'s `Attention`. It exists, it is causal, and its
`softmax` kind is literally the same one-line torch primitive used below
(`ceq/lm.py:176`). It is not reused for three reasons: it carries a five-way
operator switch bound to the module globals `RHO`, `HOPS` and `SGATE_LAM`, whose
own docstring records them as pre-campaign values another caller overrides, so
these numbers would depend on a file this piece does not own; its `proj` is a
`d x d` layer this parameter budget has no use for; and it has NO positional
encoding -- positions enter at `ceq/lm.py:236`, in the LM, not in the block.
That last one is fatal, not cosmetic: without positions, attention over a prefix
is permutation-INVARIANT, so an encoder built on it would fail the permutation
control below for the same reason the shipped arm fails it.

NO NUMBER IS CACHED IN THIS DOCSTRING. Every figure is printed by `main()`;
`python -m ceqjepa.causal_encoder` is the producer.

WHAT WOULD HAVE TO CHANGE IN pi_jepa.py TO ADOPT THIS. Measured against the call
sites, not guessed:

  1. `pi_jepa.py:844-845` -- `self.online = Encoder(x_dim, d, hidden)` and the
     same for `self.target`. Swapping the class is the whole wiring change: the
     shape contract `(B, S, X) -> (B, S, D)` is unchanged, so every `[:, -1, :]`
     consumer (lines 620, 720, 721, 751, 753, 787, 888, 947, 988, 989, 1156,
     1163, 1377) is untouched.
  2. `pi_jepa.py:787-788` -- `sum(1 for mod in model.online.net if isinstance(mod,
     nn.Linear) ...)` reaches THROUGH the encoder into a `.net` attribute and
     iterates it. This is the one hard break. `CausalEncoder` exposes `.net` as
     an `nn.ModuleList` of its `nn.Linear` layers for exactly this reason, so the
     zero-init check keeps working; if that attribute is ever dropped, that line
     must change with it.
  3. `pi_jepa.py:474-484` -- `ema_update` walks `parameters()` AND `buffers()`.
     This encoder registers NO buffers (the sinusoid is computed in `forward`),
     so the buffer loop stays a no-op and the EMA is unaffected.
  4. `pi_jepa.py:1307-1311` -- a comment block stating "On the encoder that
     exponent is zero by construction: Encoder is a position-wise map". That
     becomes false on adoption and must be rewritten, not deleted: it is the
     premise the corner-rule retirement rests on, and re-measuring `alpha` on an
     encoder that aggregates is the reroute, not a repeal of the retirement.
  5. Parameter count moves. `pi_jepa.py:844` passes `hidden` positionally as
     `HIDDEN = 32`; at that width this class is LARGER than the arm it replaces.
     `main()` prints both counts. The matched comparison below is run at
     `hidden = 24`, where this class is SMALLER than the shipped one.
  6. `tests/curvature/test_pi_jepa_assignment_work.py:52` deepcopies the encoder;
     that works unchanged, no buffers and no non-leaf state.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ceqjepa.pi_jepa import D_LATENT, HIDDEN, S_LEN, SEED, X_DIM

__all__ = [
    "CausalEncoder", "make_bed", "ema_target", "permute_context",
    "train_arm", "r_squared", "target_constants", "main",
]

#: decays for the nine target channels. lam = 0 is "the previous observation",
#: lam = 0.9 is a long running average; nine coordinates, nine context lengths.
LAM_LO, LAM_HI = 0.0, 0.9
TARGET_SEED = 20260914          #: pins u and nothing else
TRAIN_SEQS, EVAL_SEQS = 256, 512
BATCH_FIT = 64
LR_FIT = 3e-3
STEPS_FIT = 3000
D_MODEL, N_HEAD, HID_MATCHED = 16, 4, 24


# ---------------------------------------------------------------------------
# The bed and the bar.
# ---------------------------------------------------------------------------

def make_bed(n_seq, seed, s=S_LEN, x_dim=X_DIM):
    """An i.i.d. standard-normal bed. I.I.D. IS LOAD-BEARING, not convenience.

    It is what makes `x_t` independent of `x_{<t}`, and therefore what makes the
    position-wise ceiling `R^2 <= 0` a theorem rather than a hope. On an
    autoregressive bed `x_t` carries information about its own prefix and a
    position-wise map scores above zero for free.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n_seq, s, x_dim, generator=g, dtype=torch.float64)


def target_constants(d=D_LATENT, x_dim=X_DIM):
    """`(u, lam)`, fixed forever by TARGET_SEED. Columns of u are unit norm so
    that `x_j . u_d` has unit variance on the bed above, which is what lets the
    analytic `scale_d` below put every channel on a comparable footing."""
    g = torch.Generator().manual_seed(TARGET_SEED)
    u = torch.randn(x_dim, d, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=0, keepdim=True)
    lam = torch.linspace(LAM_LO, LAM_HI, d, dtype=torch.float64)
    return u, lam


def ema_target(x):
    """`y_t[d] = sqrt(1 - lam_d^2) * sum_{j<t} lam_d^(t-1-j) * (x_j . u_d)`.

    Written as the recurrence rather than a closed form so that the strictness
    of `j < t` is visible on one line and cannot drift: `y[t]` is built from
    `f[t-1]` and `y[t-1]`, and `f[t]` is never read at step `t`.

    `sqrt(1 - lam^2)` is the asymptotic unit-variance scale for an AR(1) driven
    by unit-variance noise. It is analytic rather than fitted to the bed, so the
    target does not depend on which bed it is evaluated on.
    """
    u, lam = target_constants(D_LATENT, x.shape[-1])
    f = x @ u                                    # (B, S, D)
    scale = torch.sqrt(1.0 - lam ** 2)
    y = torch.zeros_like(f)
    for t in range(1, f.shape[1]):
        y[:, t] = lam * y[:, t - 1] + scale * f[:, t - 1]
    return y


def permute_context(x, seed):
    """Shuffle positions `0..S-2`; PIN position `S-1`.

    VARIES: the ORDER of the context positions.
    PINS: the multiset of context observations, the final observation `x_{S-1}`,
    the batch, the dtype, the shape, and every model weight.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(x.shape[1] - 1, generator=g)
    out = x.clone()
    out[:, :-1] = x[:, perm]
    return out


def r_squared(pred, y):
    """`1 - SSE / SST`, SST taken against the per-coordinate mean of `y`.

    Zero is not a convention here, it is the position-wise arm's population
    ceiling, so the sign of this number is the whole result.
    """
    sse = (pred - y).pow(2).sum()
    sst = (y - y.mean(dim=(0, 1))).pow(2).sum()
    return float(1.0 - sse / sst)


# ---------------------------------------------------------------------------
# The piece.
# ---------------------------------------------------------------------------

def _sinusoid(s, d, dtype, device):
    """Fixed sinusoidal positions. Computed, not stored: it is 16x16 numbers and
    registering it as a buffer would put it in `ema_update`'s buffer loop for no
    reason. ZERO PARAMETERS, so it does not enter the parameter match."""
    pos = torch.arange(s, dtype=dtype, device=device).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=dtype, device=device)
    ang = pos / torch.pow(torch.tensor(10000.0, dtype=dtype, device=device), i / d)
    pe = torch.zeros(s, d, dtype=dtype, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class CausalEncoder(nn.Module):
    """`x_{<=t} -> s_t`, aggregated over the sequence axis, causally.

    The claim the shipped docstring makes, implemented. One causal attention
    layer over sinusoidally-positioned inputs, then the same two-layer GELU head
    the shipped arm ends with.

    CAUSALITY IS STRUCTURAL, not a training outcome: `is_causal=True` masks the
    upper triangle with `-inf` before the softmax, so `exp(-inf) = 0` exactly and
    a future position contributes a BITWISE zero, not a small number.
    `tests/curvature/test_causal_encoder.py::test_causal_encoder_has_no_future_leak`
    asserts that per position rather than trusting the flag.

    POSITIONS ARE NOT DECORATION. Attention without them is permutation-invariant
    over the prefix, which would make this encoder order-blind at the last
    position in exactly the way the shipped one is -- passing the bar's letter
    while failing its point. The sinusoid is what makes the permutation control
    a real contrast.

    NO NORMALISATION, for the reason the shipped arm gives at `pi_jepa.py:390`:
    an RMS norm over a representation whose norm grows with `n` subtracts that
    growth from every exponent at once, and measuring `alpha` on this encoder is
    the whole point of replacing the other one.

    `.net` EXISTS FOR pi_jepa.py:788, which iterates the encoder's `.net` looking
    for zero-initialised `nn.Linear` weights. It is an `nn.ModuleList` of this
    module's Linear layers, so that collapse check keeps working on adoption.
    """

    def __init__(self, x_dim=X_DIM, d=D_LATENT, hidden=HID_MATCHED,
                 d_model=D_MODEL, n_head=N_HEAD, span="prefix"):
        super().__init__()
        assert d_model % n_head == 0, (d_model, n_head)
        assert d_model % 2 == 0, "the sinusoid needs an even width"
        assert span in ("prefix", "self"), span
        self.d_model, self.n_head, self.span = d_model, n_head, span
        self.inp = nn.Linear(x_dim, d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.mix = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, d)
        self.net = nn.ModuleList([self.inp, self.qkv, self.mix, self.out])

    def forward(self, x):
        b, s, _ = x.shape
        h = self.inp(x) + _sinusoid(s, self.d_model, x.dtype, x.device)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = lambda t: t.view(b, s, self.n_head, -1).transpose(1, 2)
        if self.span == "self":
            # THE ABLATION. Each query may attend to its own position and
            # nothing else, so the softmax row has a single unmasked entry and
            # returns v_t exactly. Every parameter, the initialisation, the
            # sinusoid, both layer widths and the whole training recipe are
            # unchanged -- the ONLY thing that varies is the set of positions a
            # query may see. That is what isolates aggregation as the cause of
            # the win, against "it had an extra layer" and "it had positions".
            eye = torch.eye(s, dtype=torch.bool, device=x.device)
            o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                               attn_mask=eye)
        else:
            o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                               is_causal=True)
        o = o.transpose(1, 2).reshape(b, s, self.d_model)
        return self.out(F.gelu(self.mix(o)))


# ---------------------------------------------------------------------------
# The measurement.
# ---------------------------------------------------------------------------

def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_arm(model, steps=STEPS_FIT, seed=SEED, lr=LR_FIT, batch=BATCH_FIT):
    """Fit one arm and score it on a HELD-OUT bed.

    EVERYTHING EXCEPT THE ENCODER CLASS IS PINNED across arms: the train bed and
    its seed, the eval bed and its seed, the target constants, Adam, the learning
    rate, the step count, the batch size, the batch order, the loss, the scored
    positions and the dtype. The encoder class is the only thing that varies.

    POSITION 0 IS EXCLUDED from both the loss and the metric. `y_0 = 0`
    identically, so both arms would predict it perfectly and both R^2 values
    would be inflated by the same free position -- which flatters the loser.
    """
    xtr = make_bed(TRAIN_SEQS, seed=seed)
    xev = make_bed(EVAL_SEQS, seed=seed + 1000)
    ytr, yev = ema_target(xtr), ema_target(xev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(int(seed) + 1)
    for _ in range(steps):
        i = torch.randint(TRAIN_SEQS, (batch,), generator=g)
        loss = (model(xtr[i])[:, 1:] - ytr[i][:, 1:]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ptr, pev = model(xtr)[:, 1:], model(xev)[:, 1:]
    _, lam = target_constants()
    per = [r_squared(pev[..., d:d + 1], yev[:, 1:, d:d + 1])
           for d in range(yev.shape[-1])]
    return {
        "params": n_params(model),
        "r2_train": r_squared(ptr, ytr[:, 1:]),
        "r2": r_squared(pev, yev[:, 1:]),
        "mse": float((pev - yev[:, 1:]).pow(2).mean()),
        "mse_constant": float((yev[:, 1:] - yev[:, 1:].mean(dim=(0, 1)))
                              .pow(2).mean()),
        "per_coord_r2": per,
        "lam": [round(float(x), 4) for x in lam],
    }


def permutation_contrast(model, seed=SEED):
    """Max |change| at the LAST position under `permute_context`.

    VARIES: the order of context positions 0..S-2.
    PINS: the multiset of context observations, `x_{S-1}`, the weights, the seed,
    the batch, the dtype. Exactly one thing moves.
    """
    x = make_bed(64, seed=seed + 5)
    xp = permute_context(x, seed=seed + 6)
    assert torch.equal(xp[:, -1], x[:, -1]), "pin broken: x_{S-1} moved"
    assert not torch.equal(xp[:, :-1], x[:, :-1]), "vary is a no-op"
    with torch.no_grad():
        a, b = model(x)[:, -1], model(xp)[:, -1]
    return {"bitwise_identical": bool(torch.equal(a, b)),
            "max_abs_change": float((a - b).abs().max())}


def leak_probe(model, seed=SEED):
    """Worst |change| at positions STRICTLY BEFORE a perturbed one, over every
    perturbable position. Anything but exactly 0.0 is a leak from the future and
    every other number in this file would be meaningless.

    VARIES: `x_k`, one position, replaced by fresh noise.
    PINS: `x_j` for every `j != k`, the weights, the seed, the dtype.
    """
    x = make_bed(16, seed=seed + 9)
    g = torch.Generator().manual_seed(int(seed) + 11)
    with torch.no_grad():
        s = model(x)
    worst_past, weakest_self = 0.0, float("inf")
    for k in range(1, x.shape[1]):
        xk = x.clone()
        xk[:, k] = torch.randn(x.shape[0], x.shape[2], generator=g, dtype=x.dtype)
        with torch.no_grad():
            sk = model(xk)
        worst_past = max(worst_past, float((s[:, :k] - sk[:, :k]).abs().max()))
        weakest_self = min(weakest_self, float((s[:, k] - sk[:, k]).abs().max()))
    return {"worst_past_change": worst_past, "weakest_own_position_change": weakest_self}


def main():
    from ceqjepa.pi_jepa import Encoder

    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
          f"dtype float64  device cpu")
    u, lam = target_constants()
    print(f"bar: strict-prefix EMA bank, lam = {[round(float(v), 3) for v in lam]}")
    print(f"     position-wise population ceiling on this bar: R^2 <= 0 exactly\n")

    rows = []
    arms = (
        ("shipped Encoder (position-wise)", Encoder),
        ("shipped Encoder, 3.2x wider", lambda: Encoder(hidden=64)),
        ("CausalEncoder, span=self (null)", lambda: CausalEncoder(span="self")),
        ("CausalEncoder (this file)", CausalEncoder),
    )
    for name, ctor in arms:
        torch.manual_seed(SEED)
        m = ctor().double()
        fit = train_arm(m)
        perm = permutation_contrast(m)
        leak = leak_probe(m)
        rows.append((name, fit, perm, leak))

    print(f"{'arm':34s} {'params':>7s} {'R2 eval':>9s} {'R2 train':>9s} "
          f"{'MSE':>9s} {'MSE const':>9s}")
    for name, fit, _, _ in rows:
        print(f"{name:34s} {fit['params']:7d} {fit['r2']:9.4f} "
              f"{fit['r2_train']:9.4f} {fit['mse']:9.4f} {fit['mse_constant']:9.4f}")

    print("\npermutation control -- VARIES: order of context positions 0..S-2;")
    print("PINS: multiset of context observations, x_{S-1}, weights, seed, dtype")
    for name, _, perm, _ in rows:
        print(f"  {name:34s} bitwise_identical={perm['bitwise_identical']!s:5s} "
              f"max_abs_change={perm['max_abs_change']:.6e}")

    print("\ncausality -- VARIES: x_k for one k; PINS: every x_j, j != k, weights")
    for name, _, _, leak in rows:
        print(f"  {name:34s} worst change at t<k = {leak['worst_past_change']:.6e}"
              f"   weakest change at t=k = {leak['weakest_own_position_change']:.6e}")

    print("\nper-coordinate R^2 on the eval bed, by decay:")
    print(f"  {'lam':>6s} " + " ".join(f"{v:>7.3f}" for v in rows[0][1]["lam"]))
    for name, fit, _, _ in rows:
        print(f"  {name[:6]:>6s} " + " ".join(f"{v:>7.3f}"
                                              for v in fit["per_coord_r2"]))

    print("\nablation -- VARIES: the set of positions a query may attend to,")
    print("prefix {j <= t} against self {t}. PINS: every parameter and its")
    print("initialisation seed, the sinusoid, both layer widths, the bed, the")
    print("target constants, Adam, the learning rate, the step count, the loss.")
    self_p = n_params(CausalEncoder(span="self").double())
    pre_p = n_params(CausalEncoder().double())
    print(f"  parameter counts identical across the two spans: "
          f"{self_p} == {pre_p} -> {self_p == pre_p}")

    big = CausalEncoder(hidden=HIDDEN).double()
    print(f"\nadoption note: at the shipped HIDDEN={HIDDEN} this class costs "
          f"{n_params(big)} params against the shipped arm's "
          f"{n_params(Encoder().double())}; the matched run above uses "
          f"hidden={HID_MATCHED}.")


if __name__ == "__main__":
    main()
