"""X29b -- the Wiener equalizer for the second hop.

THE PROBLEM THIS SOLVES. `Arm.forward` adds a second hop
`z = x + a@x + hop2@x` with `hop2 = a[:,P] @ a[P,:]`, which at `K = |P| = s`
is exactly `a@a`. That term takes the arm from LEARNS to NO READING
(t*=2, n_train=2048, steps=150, 3 seeds, threads=8: softmax 0.950252,
K=8 0.960945, K=64 1.000329, where 1.0 is predict-the-mean). The rank-1
common mode of `a@a` holds 0.7550 of its Frobenius energy at K=64 and 0.4899
at K=8, so pivot routing had been acting as an accidental common-mode filter.

DELETING THAT ONE MODE IS A SPECIAL CASE. This file weights EVERY mode of the
hop operator by its own signal-to-noise ratio,

    (1)   g_i = S_i / (S_i + N_i)

which is the MMSE gain for that mode (derivation in V13_X29B_WIENER.md). The
common-mode deflation is `g_0 ~ 0` FALLING OUT of (1); `wiener_gains` takes a
coordinate matrix and a target and nothing else -- no basis vector, no index,
no threshold -- which is what the DC must-fire below actually checks.

WHAT S_i AND N_i ARE, precisely, because this is the part most likely to be
wrong. Let `M_n` be example `n`'s hop operator, `w_n = M_n[s-1, :]` its READ
ROW (the arm returns only `out[:, s-1]`), and `b_n = x[n, :, CH_FLIP]` the
driver channel -- the only channel the label depends on. Let `V` hold the
right singular vectors of the batch-mean hop operator. Then

    (2)   c_{n,i} = (w_n . v_i) (v_i . b_n),   sum_i c_{n,i} = (M_n b_n)_{s-1}

is example `n`'s mode-`i` coordinate, and (2) is EXACT, not a truncation:
`{v_i}` is an orthonormal basis of R^s. The oracle is the bed's own

    (3)   y2 = equilibrium_hop_reading(x, t*) - equilibrium_hop_reading(x, t*-1)

-- the exact top-hop term of the label's path sum, recomputed from CH_DRIVE
and CH_FLIP by `negation_scope`, never estimated. With `<f,h>` the centred
batch inner product,

    (4)   S_i = <c_i, y2>^2 / <y2, y2>,      N_i = <c_i, c_i> - S_i

so S_i is the energy of c_i's projection onto the oracle and N_i is the
orthogonal remainder. (1) then equals the squared correlation of c_i with the
oracle -- which is why the DC gain vanishes with no special case: an all-ones
mode reads the driver MEAN, and the mean of iid drivers is uncorrelated with
a signed path product.

SVD, NOT EIGENDECOMPOSITION, AND THE REASON IS STRUCTURAL. `a` is strictly
lower triangular (`bench._causal_mask` is `tril(-1)`), so `a` and `a@a` are
NILPOTENT: every eigenvalue is exactly 0 and the eigenvectors do not span. An
eigenmode decomposition here carries no information at all, not merely
ill-conditioned information. `check_decomposition` measures both facts before the
SVD is used, so the choice is a reading and not a preference.

Run `python scripts/v13_wiener_hop.py` for the self-check (seconds, no arms).
`--arms` additionally trains softmax / plain-K / Wiener-equalized-K.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from scale.m3_capability import (Arm, D_MODEL, LR, batched_pivot_hop2,
                                 batched_select_pivots)
from scale.negation_scope import (CH_DRIVE, CH_FLIP, M3_TASKS,
                                  equilibrium_hop_reading, nrmse)

S, D, N_EVAL = 64, 24, 4096
TINY = torch.finfo(torch.float32).tiny


# ==========================================================================
# THE ESTIMATOR -- everything below this line is basis-agnostic
# ==========================================================================

def wiener_gains(c: torch.Tensor, y: torch.Tensor):
    """(1) applied per column of `c`. Returns `(g, S, N)`, each `[m]`.

    `c` `[n, m]` are mode coordinates, one column per mode; `y` `[n]` is the
    oracle target. Both are centred here, so callers need not.

    KNOWS NOTHING ABOUT DC, and that is the point: the only inputs are a
    coordinate matrix and a target. Any statement downstream about the common
    mode is therefore a consequence of (1), not of a branch.
    """
    c = c - c.mean(0, keepdim=True)
    y = y - y.mean()
    yy = float((y * y).sum())
    if yy <= 0.0:
        raise ValueError("oracle target has zero variance")
    s_energy = (c.T @ y) ** 2 / yy                     # (4), S_i
    total = (c * c).sum(0)                             # S_i + N_i
    return s_energy / total.clamp_min(TINY), s_energy, total - s_energy


def water_fill(d: torch.Tensor, budget: float):
    """Gallager's water-filling allocation: `p_i = (mu - d_i)_+`, `sum p_i = P`.

    `d_i = N_i / |h_i|^2` is the noise-to-gain ratio of parallel channel `i`.
    The allocation maximising `sum_i log(1 + p_i / d_i)` subject to
    `sum_i p_i <= P` and `p_i >= 0` is the water-pouring solution: Gallager,
    *Information Theory and Reliable Communication*, Wiley 1968, section 7.5
    (parallel Gaussian channels; the Kuhn-Tucker conditions make `p_i + d_i`
    constant on the support and require `d_i >= mu` off it). Returns
    `(p, mu)`.

    The level is found by the exact sorted scan, not by bisection: with `d`
    ascending, the level implied by the `k` cheapest channels is
    `mu_k = (P + sum_{i<k} d_i) / k`, and the answer is the largest `k` with
    `mu_k > d_k`.
    """
    ds, _ = torch.sort(d)
    k = torch.arange(1, ds.numel() + 1, dtype=ds.dtype)
    mu_k = (budget + torch.cumsum(ds, 0)) / k
    active = torch.nonzero(mu_k > ds)
    if active.numel() == 0:                      # budget below every channel
        return torch.zeros_like(d), float(ds[0])
    mu = float(mu_k[int(active.max())])
    return (mu - d).clamp_min(0.0), mu


def mode_coord(w: torch.Tensor, b: torch.Tensor, basis: torch.Tensor):
    """(2) for an arbitrary orthonormal `basis` `[s, m]`: `[n, m]`.

    `w` `[n, s]` read rows, `b` `[n, s]` drivers. Used for the SVD basis and,
    unchanged, for the single hand-built DC direction in the must-fire, so
    both go through one construction.
    """
    return (w @ basis) * (b @ basis)


def wiener_gains_block(cb: torch.Tensor, y: torch.Tensor):
    """(1) for a VECTOR-valued mode coordinate `cb` `[n, m, d]`.

    The scalar form (2) reads only CH_FLIP, but the hop-2 term delivers all
    `d_model` channels into the MLP, which is nonlinear and can multiply them.
    S_i here is the summed energy of every channel's projection onto the
    oracle, so a mode whose signal sits in a channel other than the drivers
    is still credited. Same S/(S+N) shape, so it drops into the same report
    column; the two columns differing would localise the noise model's error.
    """
    n, m, d = cb.shape
    g_flat, s_flat, n_flat = wiener_gains(cb.reshape(n, m * d), y)
    del g_flat
    s_e = s_flat.reshape(m, d).sum(1)
    total = s_e + n_flat.reshape(m, d).sum(1)
    return s_e / total.clamp_min(TINY), s_e, total - s_e


# ==========================================================================
# THE BED
# ==========================================================================

def common_mode_fraction(m: torch.Tensor) -> float:
    """Frobenius energy share of the rank-1 common mode `1 pi^T`, `pi` the
    column mean of `m`. The quantity the K sweep reported as 0.7550 / 0.4899."""
    pi = m.mean(0)
    return float(m.shape[0] * (pi * pi).sum() / (m * m).sum())


def bed(n: int, seed: int, t_star: int, k_pivots: int) -> dict:
    """One measured cell, at the arm's INITIAL operator.

    The equalizer is fitted once, before training, and then frozen: `q`/`k`
    come from a freshly seeded `Arm`, so `bed(...)` and the arm that later
    trains under the same seed hold identical weights at step 0.
    """
    batch_fn = M3_TASKS[f"e3_t{t_star}"][0]
    x, y, _f, _p = batch_fn(n, S, D, d_model=D_MODEL, seed=seed)
    torch.manual_seed(seed)
    model = Arm("pivot_unsigned", S, k_pivots=k_pivots)
    return cell_of(model, x, y, t_star, k_pivots)


def cell_of(model, x, y, t_star: int, k_pivots: int) -> dict:
    """`bed`'s body, split out so the SAME reading can be taken from a TRAINED
    operator. Nothing about (1)-(4) depends on where `model` came from."""
    with torch.no_grad():
        key = model.wk(x)
        a = model._operator(model.wq(x), key)
        hop2 = batched_pivot_hop2(a, batched_select_pivots(key, k_pivots))
        mbar = hop2.mean(0)
        _u, sing, vh = torch.linalg.svd(mbar)
        v = vh.transpose(0, 1)                                    # [s, m]
        w = hop2[:, -1, :]                                        # [n, s]
        b = x[:, :, CH_FLIP]                                      # [n, s]
        c = mode_coord(w, b, v)
        #: the same coordinate carried over EVERY channel, for the block form
        cb = (w @ v)[:, :, None] * torch.einsum("nsd,sm->nmd", x, v)
    y2 = (equilibrium_hop_reading(x, t_star)
          - equilibrium_hop_reading(x, t_star - 1))
    return dict(x=x, y=y, y2=y2, a=a, hop2=hop2, mbar=mbar, sing=sing, v=v,
                w=w, b=b, c=c, cb=cb)


class WienerArm(Arm):
    """`pivot_unsigned` with the hop-2 input equalized by `P = V diag(g) V^T`.

    `sum_i g_i c_{n,i} = w_n^T V diag(g) V^T b_n`, so applying the gains in
    mode space is exactly ONE fixed `[s, s]` matrix on the hop-2 input.
    `g = 1` therefore reproduces the plain arm (`V V^T = I`), and `g_0 = 0`
    with `v_0` the common mode IS the sibling's deflation -- neither is a
    separate code path.

    `p_eq` is a BUFFER, not a parameter: registering it consumes no RNG, so a
    seeded `WienerArm` and a seeded `Arm` hold identical initial weights and
    the 10% param-match bar is untouched.
    """

    def __init__(self, s, *, p_eq, k_pivots):
        super().__init__("pivot_unsigned", s, k_pivots=k_pivots)
        self.register_buffer("p_eq", p_eq)

    def forward(self, x):
        q, key = self.wq(x), self.wk(x)
        a = self._operator(q, key)
        z = x + a @ x
        hop2 = batched_pivot_hop2(a, batched_select_pivots(key, self.k_pivots))
        z = z + hop2 @ (self.p_eq @ x)
        #: the parent's tail. `Arm.forward` ends `out[:, s - 1]`; a previous
        #: subclass stopped at `.squeeze(-1)` and died broadcasting 64 against
        #: 2048. It is part of the arm, not formatting.
        return self.readout(self.mlp(z)).squeeze(-1)[:, x.shape[1] - 1]


def equalizer(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """`P = V diag(g) V^T`."""
    return (v * g) @ v.transpose(0, 1)


def train_eval(model, *, n_train, steps, seed, x_eval, y_eval, batch_fn) -> float:
    """`scripts/v13_hop2_gain.py::run`, unchanged, so the arm numbers here sit
    line for line beside the K sweep's."""
    x, y, _, _ = batch_fn(n_train, S, D, d_model=D_MODEL, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu = float(y.mean())
    sd = float(y.std(unbiased=False)) or 1.0
    ys = (y - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), ys).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nrmse(model(x_eval) * sd + mu, y_eval))


# ==========================================================================
# SELF-CHECK
# ==========================================================================

def _orthogonal_pair(n: int, snr: float, seed: int):
    """`c = sig + q` with `q` EXACTLY orthogonal to `sig`, both centred, and
    `||q||^2 = ||sig||^2 / snr`. Built by projection rather than by drawing
    independent noise, so the must-fire tests the estimator to machine
    precision instead of testing a sampling error."""
    gen = torch.Generator().manual_seed(seed)
    sig = torch.randn(n, generator=gen, dtype=torch.float64)
    sig -= sig.mean()
    q = torch.randn(n, generator=gen, dtype=torch.float64)
    q -= q.mean()
    q -= (q @ sig) / (sig @ sig) * sig
    q *= torch.sqrt((sig @ sig) / (snr * (q @ q)))
    return sig + q, sig


def must_fire_snr(n: int = 4096) -> None:
    """MUST-FIRE 2: a mode at SNR 10 keeps `g = 10/11 = 0.909090909...`."""
    print("\n=== MUST-FIRE 2: (1) at a built, known SNR ===")
    for snr in (0.1, 1.0, 10.0, 100.0):
        c, sig = _orthogonal_pair(n, snr, seed=17)
        g, s_e, n_e = wiener_gains(c[:, None], sig)
        want, got = snr / (snr + 1.0), float(g[0])
        print(f"  SNR={snr:>7.1f}  S={float(s_e[0]):12.6f}  N={float(n_e[0]):12.6f}"
              f"  S/N={float(s_e[0] / n_e[0]):.9f}"
              f"  g={got:.9f}  want {want:.9f}  |d|={abs(got - want):.3e}")
        assert abs(float(s_e[0] / n_e[0]) - snr) < 1e-9, "S/N is not the built SNR"
        assert abs(got - want) < 1e-9, (
            f"MUST-FIRE 2 FAILED at SNR={snr}: g={got!r} want {want!r}")
    print("  MUST-FIRE 2 PASS -- tolerance 1e-9 on g, float64, n=%d" % n)


def must_fire_dc(cell: dict, bound: float) -> tuple[float, float]:
    """MUST-FIRE 1: a pure DC mode gets `g ~ 0`, out of (1) and nothing else.

    Two readings, because they answer different questions.
      (a) an EXACTLY DC direction `1/sqrt(s)` pushed through the SAME
          coordinate construction (2) and the SAME `wiener_gains`;
      (b) the gain (1) assigns to the real operator's top singular mode --
          the near-DC one the K sweep found holding 0.7550 of the energy.
    """
    print("\n=== MUST-FIRE 1: DC gain, from (1) with no special case ===")
    n, s = cell["x"].shape[0], cell["x"].shape[1]
    dc = torch.full((s, 1), 1.0 / (s ** 0.5))
    c_dc = mode_coord(cell["w"], cell["b"], dc)
    g_dc, s_dc, n_dc = wiener_gains(c_dc, cell["y2"])
    print(f"  (a) exact DC direction 1/sqrt(s):  S={float(s_dc[0]):.6e}"
          f"  N={float(n_dc[0]):.6e}  g={float(g_dc[0]):.6e}")
    g_all, _s, _nz = wiener_gains(cell["c"], cell["y2"])
    cos_dc = float(abs(cell["v"][:, 0] @ dc[:, 0]))
    print(f"  (b) top singular mode v_0:         |cos(v_0, 1/sqrt(s))|={cos_dc:.4f}"
          f"  g_0={float(g_all[0]):.6e}")
    print(f"  null band at n={n}: E[g | rho=0] = 1/(n-2) = {1.0 / (n - 2):.6e},"
          f"  99.9% quantile 10.828/n = {10.828 / n:.6e},"
          f"  pre-registered bound {bound:.4f}")
    assert float(g_dc[0]) < bound, (
        f"MUST-FIRE 1 FAILED: exact DC gain {float(g_dc[0]):.6e} >= {bound}")
    assert float(g_all[0]) < bound, (
        f"MUST-FIRE 1 FAILED: top-mode gain {float(g_all[0]):.6e} >= {bound}")
    print(f"  MUST-FIRE 1 PASS -- both below the pre-registered {bound}")
    return float(g_dc[0]), float(g_all[0])


def check_decomposition(cell: dict) -> None:
    """(2) is exact, the SVD basis is orthonormal, and the eigendecomposition
    of the same operator is empty -- the three facts the mode choice rests on."""
    print("\n=== MODE DECOMPOSITION: why SVD and not eig ===")
    mbar, v, c = cell["mbar"], cell["v"], cell["c"]
    lam = torch.linalg.eigvals(mbar).abs()
    print(f"  batch-mean hop operator M_bar [{mbar.shape[0]}x{mbar.shape[1]}]:"
          f"  strictly lower triangular = {bool((mbar.triu(0) == 0).all())}")
    print(f"  |eig(M_bar)|  max={float(lam.max()):.3e}  mean={float(lam.mean()):.3e}"
          f"   -> NILPOTENT: the eigenbasis carries no modes to weight")
    assert float(lam.max()) < 1e-6, (
        f"M_bar is not nilpotent (max |eig| = {float(lam.max()):.3e}); the "
        f"eig-vs-svd argument in this file would need rewriting")
    orth = float((v.transpose(0, 1) @ v - torch.eye(v.shape[1])).abs().max())
    print(f"  SVD right basis V: max |V^T V - I| = {orth:.3e}  (orthonormal)")
    assert orth < 1e-4, f"V is not orthonormal to 1e-4: {orth:.3e}"
    direct = torch.einsum("nij,nj->ni", cell["hop2"], cell["b"])[:, -1]
    err = float((c.sum(1) - direct).abs().max() / direct.abs().max())
    print(f"  (2) exactness: max |sum_i c_i - (M_n b_n)_[s-1]| / max|.| = {err:.3e}")
    assert err < 1e-4, f"(2) does not reconstruct the read row: {err:.3e}"
    sing = cell["sing"]
    e = sing ** 2 / (sing ** 2).sum()
    print(f"  singular energy: sigma_0^2 share = {float(e[0]):.4f},"
          f"  top-4 = {float(e[:4].sum()):.4f},"
          f"  rank at 1e-6 = {int((sing > 1e-6 * sing[0]).sum())}")
    print(f"  rank-1 common-mode 1.pi^T energy share of M_bar = "
          f"{common_mode_fraction(mbar):.4f}")


def check_oracle(cell: dict, t_star: int) -> None:
    """(3) is the TOP-HOP path term and not something else.

    At `t* = 2` the chain reduces to `y = a_{s-1} a_{s-2} b_{s-3} +
    a_{s-1} b_{s-2}` (`make_equilibrium_batch` zeroes the sub-diagonal at and
    before `head = s-1-t*` and sets `b_{s-1} = 0`), so (3) must equal the
    first of those two products exactly. Bound rather than asserted in prose,
    because an oracle that is quietly the WRONG object would make every S_i
    below meaningless while still producing a plausible table.
    """
    if t_star != 2:
        return
    x = cell["x"]
    a, b = x[:, :, CH_DRIVE], x[:, :, CH_FLIP]
    s = x.shape[1]
    want = a[:, s - 1] * a[:, s - 2] * b[:, s - 3]
    err = float((cell["y2"] - want).abs().max())
    print(f"\n=== ORACLE BIND (3) ===")
    print(f"  max |y2 - a_[s-1] a_[s-2] b_[s-3]| = {err:.3e}"
          f"   Var(y2) = {float(cell['y2'].var(unbiased=False)):.6f}"
          f"   Var(y) = {float(cell['y'].var(unbiased=False)):.6f}")
    assert err < 1e-6, f"(3) is not the 2-hop path term: {err:.3e}"


def check_water_filling() -> None:
    """Gallager 1968 section 7.5 -- the KKT conditions, checked rather than
    quoted, plus the two facts that separate water-filling from (1)."""
    print("\n=== WATER-FILLING (Gallager 1968, sec. 7.5) ===")
    d = torch.tensor([0.05, 0.20, 0.50, 3.00, 1e9], dtype=torch.float64)
    for budget in (0.1, 1.0, 10.0):
        p, mu = water_fill(d, budget)
        act = p > 0
        lvl = (p + d)[act]
        print(f"  P={budget:>5.1f}  mu={mu:.6f}  active={int(act.sum())}/{d.numel()}"
              f"  p={[round(float(t), 4) for t in p]}")
        assert abs(float(p.sum()) - budget) < 1e-9, "budget not exhausted"
        assert float((lvl - mu).abs().max()) < 1e-9, "p_i + d_i != mu on support"
        assert bool((d[~act] >= mu - 1e-12).all()), "an inactive channel is below mu"
        assert float(p[-1]) == 0.0, "the d=1e9 (pure-DC-like) channel drew power"
    print("  KKT PASS: p_i + d_i = mu on the support, d_i >= mu off it, "
          "sum p_i = P, and the infinite-noise channel is cut at every budget.")


def report_gains(cell: dict, top: int = 8) -> torch.Tensor:
    """The measured gain spectrum against the exact oracle (3) and, as the
    deployable variant, against the raw label."""
    g2, s2, n2 = wiener_gains(cell["c"], cell["y2"])
    gy, _s, _n = wiener_gains(cell["c"], cell["y"])
    gb, _sb, _nb = wiener_gains_block(cell["cb"], cell["y2"])
    e = cell["sing"] ** 2 / (cell["sing"] ** 2).sum()
    print("\n=== MEASURED GAIN SPECTRUM ===")
    print("   i   sigma_i^2 share      S_i          N_i        g(vs y2)   "
          "g(vs y)    g(all chans)")
    order = torch.argsort(g2, descending=True)[:top]
    idx = sorted(set([0, 1, 2, 3] + [int(t) for t in order]))
    for i in idx:
        print(f"  {i:>3}      {float(e[i]):.6f}   {float(s2[i]):.4e}"
              f"   {float(n2[i]):.4e}    {float(g2[i]):.6f}   {float(gy[i]):.6f}"
              f"   {float(gb[i]):.6f}")
    null = 10.828 / cell["c"].shape[0]
    print(f"  sum_i g_i = {float(g2.sum()):.4f} over {g2.numel()} modes;"
          f"  max g = {float(g2.max()):.6f} at mode {int(g2.argmax())};"
          f"  modes with g > 0.01: {int((g2 > 0.01).sum())}")
    print(f"  99.9% zero-correlation quantile 10.828/n = {null:.6f};"
          f"  modes above it: {int((g2 > null).sum())} (scalar),"
          f" {int((gb > null).sum())} (all channels, 1 d.o.f. band is a"
          f" LOWER bound at d_model={cell['cb'].shape[2]})")
    #: the diagonal form of (1) is MMSE only if the coordinates are mutually
    #: uncorrelated. Measured, not assumed.
    cc = cell["c"] - cell["c"].mean(0, keepdim=True)
    gram = cc.T @ cc
    dg = torch.diagonal(gram)
    off = float((gram - torch.diag(dg)).abs().sum() / gram.abs().sum())
    print(f"  coordinate Gram: off-diagonal mass fraction = {off:.4f}"
          f"  (0 would make the diagonal gains exactly MMSE)")
    return g2


def gains_after_training(a, seed: int) -> None:
    """The same spectrum, re-read from the operator the plain arm TRAINS to.

    The equalizer is fitted at init, so a spectrum that stays in the null band
    after 150 steps says the hop never acquires linear signal about the oracle
    -- while a spectrum that lifts says the init-time fit, not (1), is what is
    wrong. This is the measurement that separates the two candidate causes if
    the declared kill fires.
    """
    batch_fn = M3_TASKS[f"e3_t{a.t_star}"][0]
    x_eval, y_eval, _f, _p = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    torch.manual_seed(seed)
    model = Arm("pivot_unsigned", S, k_pivots=a.k_pivots)
    ev = train_eval(model, n_train=a.n_train, steps=a.steps, seed=seed,
                    x_eval=x_eval, y_eval=y_eval, batch_fn=batch_fn)
    x, y, _f2, _p2 = batch_fn(a.n_train, S, D, d_model=D_MODEL, seed=seed)
    cell = cell_of(model, x, y, a.t_star, a.k_pivots)
    g, _s, _n = wiener_gains(cell["c"], cell["y2"])
    gb, _sb, _nb = wiener_gains_block(cell["cb"], cell["y2"])
    null = 10.828 / a.n_train
    print(f"\n=== GAIN SPECTRUM AFTER {a.steps} STEPS (seed {seed}, "
          f"plain K={a.k_pivots} arm, eval NRMSE {ev:.6f}) ===")
    print(f"  common-mode share of trained M_bar = "
          f"{common_mode_fraction(cell['mbar']):.4f}")
    print(f"  scalar : max g {float(g.max()):.6f} at mode {int(g.argmax())},"
          f"  sum {float(g.sum()):.4f},  above {null:.6f}: {int((g > null).sum())}")
    print(f"  blocked: max g {float(gb.max()):.6f} at mode {int(gb.argmax())},"
          f"  sum {float(gb.sum()):.4f},  above {null:.6f}: {int((gb > null).sum())}")
    print(f"  g_0 (top singular mode) = {float(g[0]):.6e}")


def run_arms(a) -> None:
    print("\n=== ARMS: softmax control / plain K / Wiener-equalized K ===")
    batch_fn = M3_TASKS[f"e3_t{a.t_star}"][0]
    x_eval, y_eval, _f, _p = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    kw = dict(n_train=a.n_train, steps=a.steps, x_eval=x_eval, y_eval=y_eval,
              batch_fn=batch_fn)
    rows = {}
    for name in ("softmax", "plain", "wiener"):
        t0, vals = time.time(), []
        for sd in a.seeds:
            if name == "wiener":
                cell = bed(a.n_train, sd, a.t_star, a.k_pivots)
                g, _s, _n = wiener_gains(cell["c"], cell["y2"])
                p_eq = equalizer(cell["v"], g)
                del cell
            torch.manual_seed(sd)
            model = (Arm("softmax", S) if name == "softmax" else
                     Arm("pivot_unsigned", S, k_pivots=a.k_pivots) if name == "plain"
                     else WienerArm(S, p_eq=p_eq, k_pivots=a.k_pivots))
            vals.append(train_eval(model, seed=sd, **kw))
        rows[name] = vals
        print(f"  {name:<8} mean {statistics.fmean(vals):.6f}"
              f"  sd {statistics.stdev(vals) if len(vals) > 1 else 0.0:.6f}"
              f"  seeds {[round(v, 6) for v in vals]}  {time.time() - t0:.0f}s",
              flush=True)
    print(f"  wiener - softmax = {statistics.fmean(rows['wiener']) - statistics.fmean(rows['softmax']):+.6f}")
    print(f"  wiener - plain   = {statistics.fmean(rows['wiener']) - statistics.fmean(rows['plain']):+.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-pivots", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dc-bound", type=float, default=0.02,
                    help="pre-registered ceiling on the DC gain; 41x the "
                         "zero-correlation mean 1/(n-2) at n=2048 and 3.8x "
                         "the 99.9%% null quantile 10.828/n")
    ap.add_argument("--arms", action="store_true",
                    help="also train softmax / plain-K / Wiener-K arms")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"t*={a.t_star}  n={a.n_train}  S={S}  D={D}  K={a.k_pivots}  "
          f"seeds={a.seeds}")

    must_fire_snr()
    check_water_filling()
    cell = bed(a.n_train, a.seeds[0], a.t_star, a.k_pivots)
    check_oracle(cell, a.t_star)
    check_decomposition(cell)
    must_fire_dc(cell, a.dc_bound)
    g = report_gains(cell)

    #: the equalizer is the deflation when the top mode is DC and its gain is
    #: zero -- checked as an identity on P, not asserted in prose.
    p_eq = equalizer(cell["v"], g)
    p_defl = torch.eye(S) - torch.outer(cell["v"][:, 0], cell["v"][:, 0])
    print(f"\n=== SUBSUMPTION ===")
    print(f"  ||P_wiener||_F = {float(p_eq.norm()):.6f}   vs  ||I||_F = "
          f"{S ** 0.5:.6f}  and  ||I - v_0 v_0^T||_F = {float(p_defl.norm()):.6f}")
    print(f"  ||P_wiener - (I - v_0 v_0^T)||_F / ||I - v_0 v_0^T||_F = "
          f"{float((p_eq - p_defl).norm() / p_defl.norm()):.4f}")
    print(f"  ||V diag(1) V^T - I||_max = "
          f"{float((equalizer(cell['v'], torch.ones(S)) - torch.eye(S)).abs().max()):.3e}"
          f"   (g == 1 is the plain hop)")
    assert float((equalizer(cell["v"], torch.ones(S)) - torch.eye(S)).abs().max()) < 1e-4

    #: K=8 cross-check of the common-mode share the K sweep reported (0.4899).
    c8 = bed(a.n_train, a.seeds[0], a.t_star, 8)
    print(f"  common-mode share of M_bar at K=8: "
          f"{common_mode_fraction(c8['mbar']):.4f}"
          f"   (K={a.k_pivots}: {common_mode_fraction(cell['mbar']):.4f})")
    g8, _s8, _n8 = wiener_gains(c8["c"], c8["y2"])
    print(f"  K=8 gains: max {float(g8.max()):.6f} at mode {int(g8.argmax())},"
          f"  sum {float(g8.sum()):.4f},  above 10.828/n: {int((g8 > 10.828 / a.n_train).sum())}")
    del c8

    #: water-filling on the SAME measured modes, for the report's comparison.
    #: `d_i = N_i/S_i` is the per-mode noise-to-signal ratio; the budget is
    #: `sum_i g_i`, i.e. water-filling is asked to spend the SAME total gain
    #: the per-mode Wiener solution spent, so the two are compared at equal
    #: power rather than at two different scales.
    d = ((1.0 - g).clamp_min(1e-12) / g.clamp_min(1e-12))
    p, mu = water_fill(d.double(), float(g.sum()))
    print(f"\n=== WATER-FILLING ON THE MEASURED MODES ===")
    print(f"  d_i = N_i/S_i (per-mode noise-to-signal); budget P = sum_i g_i "
          f"= {float(g.sum()):.4f}")
    print(f"  water level mu = {mu:.6f};  active modes = {int((p > 0).sum())}"
          f" of {p.numel()};  DC mode power = {float(p[0]):.6e}")
    assert float(p[0]) < 1e-6, (
        "water-filling funded the DC mode; per-mode Wiener and water-filling "
        "are supposed to agree on which modes are dead")

    if a.arms:
        gains_after_training(a, a.seeds[0])
        run_arms(a)
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
