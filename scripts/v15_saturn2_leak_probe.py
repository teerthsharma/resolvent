"""SATURN-2 / M-18 adjudication: is the proposed "add one squared feature" gate
repair a legitimate architectural fix or an oracle leak?

READ-ONLY against `scale/negation_scope.py`: every quantity below is obtained by
IMPORTING the corpus module's own public functions (`make_equilibrium_batch`,
`equilibrium_hop_reading`, `CH_DRIVE`, `CH_FLIP`), never by re-deriving the
generator or editing it. No training anywhere -- every number is a closed-form
least-squares linear probe (`numpy.linalg.lstsq`), float64 throughout, per the
node's hard constraints.

THE QUESTION THIS FILE ANSWERS, sharpened by the parallel MARS node
(`tests/mars_v15/test_skyline_gate_containment.py`, independently re-run below
and confirmed GREEN on this machine): BED-M's chain family puts the raw,
single-position gate `a_i` directly into `x[:, :, CH_DRIVE]` BY DESIGN --
documented at `scale/negation_scope.py:520-522` ("CH_DRIVE carries the
Rademacher coefficients a") and presupposed by the contract's own R1
kill-condition text ("diagnose by linear probe on log a"). That makes order-0
(single-position, no cross-position composition) access to `a_i` a disclosed,
intentional input -- not a leak. What WOULD be a leak, per MARS's attack #3, is
a COMPOSED (order>=1, multi-step) quantity -- the thing R1 actually has to
predict -- sitting in `x` as well.

So the node's job is not "is `a` in `x`" (yes, by design) but: does the proposed
fix -- one squared feature, `x_i**2` -- stay on the disclosed order-0 side of
that line, or does it reach a composed, order>=1 quantity?  Four experiments:

  A. Reproduce V15_VENUS_PREDICTIONS.md's pooled `[x, x**2] -> 1{|a|>0}` probe
     and its `x -> sign(a)` counterpart, to confirm the R^2 = 1.000000 claim and
     show exactly which half (odd/sign vs even/magnitude) needs the square.

  B. THE DISCRIMINATING TEST the node was asked to run. Shield `CH_DRIVE` --
     overwrite it with independent noise AFTER the label `y` is already computed
     from the true `a` -- so the gate is no longer recoverable from `x` by any
     fixed function, and re-run probe A. If A's R^2 = 1.0 was reading the
     disclosed channel (the LEGITIMATE story), shielding it collapses R^2 to
     chance. If R^2 survived shielding, A's number was not about `CH_DRIVE` at
     all and the LEGITIMATE story would be wrong.

  C. THE ORDER BOUNDARY. Does `[x, x**2]`, pooled per-position exactly like A,
     also recover a COMPOSED (order>=1) quantity -- MARS's own running
     `equilibrium_hop_reading` resolvent, and the actual R1 label `y` -- on the
     live band, at t*=2 (R1's cell) and t*=8 (R2's cell)? If the squared feature
     reached order>=1 the way a real leak would, this R^2 should also read near
     1.0. If it stays near the composed signal's own linear-probe null (no
     better than plain `x`), the fix never left order 0.

  D. Independent re-run of MARS's shipped containment tests (read-only,
     `pytest`, no edits) so this ruling is not taking the parallel node's GREEN
     on faith.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale.negation_scope import (                                    # noqa: E402
    CH_DRIVE, CH_FLIP, make_equilibrium_batch, equilibrium_hop_reading,
)

S, D_MODEL, N, SEED = 64, 16, 2048, 0


# =============================================================== THE PROBE
def linear_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """R^2 of the best least-squares linear fit `y ~ X @ w + b`, float64.
    Returns (R^2, SST). SST == 0 (a constant target) returns R^2 = nan rather
    than a divide-by-zero, per M-18's own finding about the degenerate probe."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan"), ss_tot
    return 1.0 - ss_res / ss_tot, ss_tot


def composed_oracle_signal(x: torch.Tensor) -> torch.Tensor:
    """The order>=1 running resolvent, identical construction to
    `tests/mars_v15/test_skyline_gate_containment.py::composed_oracle_signal`
    (reimplemented here, read-only, against the same public
    `equilibrium_hop_reading`, so this file needs no import from `tests/`)."""
    n, s, _ = x.shape
    out = torch.zeros(n, s, dtype=x.dtype)
    for i in range(1, s + 1):
        out[:, i - 1] = equilibrium_hop_reading(x[:, :i, :], k=i - 1)
    return out


def pooled_xy(x: torch.Tensor, target: torch.Tensor, positions=None):
    """Flatten (example, position) pairs to rows, matching VENUS's and MARS's
    pooled layout: features are ONE position's own channel vector (order-0,
    no cross-position mixing)."""
    n, s, d = x.shape
    pos = range(s) if positions is None else positions
    xs = torch.stack([x[:, i, :] for i in pos], dim=1).reshape(-1, d).numpy()
    ys = torch.stack([target[:, i] for i in pos], dim=1).reshape(-1).numpy()
    return xs, ys


# ============================================================ EXPERIMENT A
def experiment_A_reproduce_venus(t_star: int = 2, seed: int = SEED):
    x, y, head, p = make_equilibrium_batch(N, S, 2, t_star=t_star, d_model=D_MODEL,
                                           seed=seed)
    a = x[:, :, CH_DRIVE].clone()
    band = (a.abs() > 0).double()          # 1{|a_i| > 0}, VENUS's target

    X, Yb = pooled_xy(x, band)
    r2_x_only, sst = linear_r2(X, Yb)                       # x -> band (odd probe on an even target)
    X2 = np.concatenate([X, X ** 2], axis=1)
    r2_x_x2, _ = linear_r2(X2, Yb)                          # [x, x^2] -> band

    Xa, Ysign = pooled_xy(x, a, positions=range(head + 1, S))  # live band only
    r2_sign, sst_sign = linear_r2(Xa, Ysign)                # x -> sign(a) (odd probe on an odd target)
    return dict(head=head, sst_band=sst, r2_band_x_only=r2_x_only,
                r2_band_x_and_xsq=r2_x_x2, sst_sign=sst_sign, r2_sign_x_only=r2_sign)


# ============================================================ EXPERIMENT B
def experiment_B_shielded(t_star: int = 2, seed: int = SEED, *, mask_payload=False):
    """Same batch, same label y (computed BEFORE shielding, from the true a).
    CH_DRIVE is then overwritten with independent noise of the SAME
    distribution `make_batch` uses for every other noise channel
    (`randn * 0.1`, see scale/negation_scope.py:96), drawn from an
    INDEPENDENT generator seed -- so nothing about the true `a` survives in
    `x` by any fixed function of x.

    `mask_payload`: a second, UNRELATED channel also needs controlling for a
    clean reading. `make_batch`'s payload draw sits at a FIXED position
    `p = s - 2` (:105, `torch.randn(n)` -- unit variance) while every other
    channel elsewhere is `randn * 0.1` (:96 -- variance 0.01), and `p` happens
    to fall inside the live band for every t* >= 2 this corpus is run at
    (`p = s - 2`, live band starts at `s - t*`). `x**2` of the payload channel
    is therefore a near-perfect POSITION tell on its own, with nothing to do
    with `a` -- a geometric coincidence of this fixture, not a re-leak of the
    shielded drive channel. `mask_payload=True` also overwrites CH_PAYLOAD
    with the ambient-scale noise so this confound cannot be mistaken for
    CH_DRIVE resurfacing."""
    x, y, head, p = make_equilibrium_batch(N, S, 2, t_star=t_star, d_model=D_MODEL,
                                           seed=seed)
    a = x[:, :, CH_DRIVE].clone()
    band = (a.abs() > 0).double()

    g = torch.Generator(device="cpu").manual_seed(seed + 55555)
    x_shielded = x.clone()
    x_shielded[:, :, CH_DRIVE] = torch.randn(N, S, generator=g) * 0.1
    if mask_payload:
        g2 = torch.Generator(device="cpu").manual_seed(seed + 66666)
        x_shielded[:, :, CH_FLIP + 1] = torch.randn(N, S, generator=g2) * 0.1  # CH_PAYLOAD == CH_FLIP+1

    X, Yb = pooled_xy(x_shielded, band)
    r2_x_only, sst = linear_r2(X, Yb)
    X2 = np.concatenate([X, X ** 2], axis=1)
    r2_x_x2, _ = linear_r2(X2, Yb)

    # per-channel diagnostic: which channel of x**2 alone carries the residual?
    per_channel = {}
    for c in range(x_shielded.shape[-1]):
        xc = x_shielded[:, :, c].reshape(-1).numpy()
        r2c, _ = linear_r2(np.stack([xc, xc ** 2], axis=1), Yb)
        per_channel[c] = r2c

    return dict(head=head, sst_band=sst, r2_band_x_only=r2_x_only,
                r2_band_x_and_xsq=r2_x_x2, per_channel_r2=per_channel)


# ============================================================ EXPERIMENT C
def experiment_C_order_boundary(t_star: int, seed: int = SEED):
    """Does [x, x^2] -- pooled per-position, exactly A's feature set -- also
    recover a COMPOSED (order>=1) quantity on the live band? Two composed
    targets: MARS's running resolvent, and the actual R1/R2 label itself
    (broadcast to every live position via the same running-reading
    construction, so it is on the same per-position footing as the other
    probes)."""
    x, y, head, p = make_equilibrium_batch(N, S, 2, t_star=t_star, d_model=D_MODEL,
                                           seed=seed)
    composed = composed_oracle_signal(x)          # order>=1, running resolvent
    live = range(head + 1, S)

    Xc, Yc = pooled_xy(x, composed, positions=live)
    r2_x_only, sst = linear_r2(Xc, Yc)
    Xc2 = np.concatenate([Xc, Xc ** 2], axis=1)
    r2_x_x2, _ = linear_r2(Xc2, Yc)

    # the actual R1/R2 scalar label, at the single readout position s-1, from
    # that position's own [x, x^2] only (the strongest-case pointwise probe)
    Xlast = x[:, S - 1, :].numpy()
    Xlast2 = np.concatenate([Xlast, Xlast ** 2], axis=1)
    r2_label, sst_label = linear_r2(Xlast2, y.numpy())

    return dict(head=head, t_star=t_star, sst_composed=sst, live_n=len(list(live)),
                r2_composed_x_only=r2_x_only, r2_composed_x_and_xsq=r2_x_x2,
                sst_label=sst_label, r2_label_x_and_xsq_at_readout=r2_label)


# ============================================================ EXPERIMENT D
def experiment_D_rerun_mars_containment():
    """Independent re-run of the parallel MARS node's shipped containment
    tests, read-only, no edits to tests/mars_v15/. Returns pytest's exit
    code (0 = all passed/skipped as expected)."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(ROOT / "tests" / "mars_v15" / "test_skyline_gate_containment.py"),
         "-v"],
        cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _fmt(v):
    return "nan" if isinstance(v, float) and v != v else (
        f"{v:.6f}" if isinstance(v, float) else str(v))


def main():
    print("=" * 78)
    print("EXPERIMENT A -- reproduce VENUS's pooled probe (t*=2, n=2048, s=64)")
    print("=" * 78)
    a_res = experiment_A_reproduce_venus(t_star=2)
    for k, v in a_res.items():
        print(f"  {k:>28} = {_fmt(v)}")

    print()
    print("=" * 78)
    print("EXPERIMENT B -- SHIELDED variant: CH_DRIVE overwritten with")
    print("independent noise after y is computed from the true a (t*=2)")
    print("=" * 78)
    b_res = experiment_B_shielded(t_star=2)
    for k, v in b_res.items():
        if k == "per_channel_r2":
            print(f"  {k:>28} = " + ", ".join(
                f"c{c}:{v2:.4f}" for c, v2 in v.items()))
        else:
            print(f"  {k:>28} = {_fmt(v)}")
    print(f"  CH_DRIVE={CH_DRIVE} shielded-channel R^2 alone = "
          f"{b_res['per_channel_r2'][CH_DRIVE]:.6f}  "
          f"(vs CH_PAYLOAD={CH_FLIP + 1} R^2 = "
          f"{b_res['per_channel_r2'][CH_FLIP + 1]:.6f})")

    print()
    print("-- B2: same shield, PLUS CH_PAYLOAD masked too (isolates CH_DRIVE) --")
    b2_res = experiment_B_shielded(t_star=2, mask_payload=True)
    for k in ("r2_band_x_only", "r2_band_x_and_xsq"):
        print(f"  {k:>28} = {_fmt(b2_res[k])}")

    print()
    print("=" * 78)
    print("EXPERIMENT C -- order boundary: does [x, x^2] reach the COMPOSED")
    print("(order>=1) quantity, at R1's cell (t*=2) and R2's cell (t*=8)?")
    print("=" * 78)
    for t_star in (2, 8):
        c_res = experiment_C_order_boundary(t_star=t_star)
        print(f"  -- t*={t_star} --")
        for k, v in c_res.items():
            print(f"  {k:>28} = {_fmt(v)}")

    print()
    print("=" * 78)
    print("EXPERIMENT D -- independent re-run of MARS's containment tests")
    print("=" * 78)
    code, out = experiment_D_rerun_mars_containment()
    print(out)
    print(f"  pytest exit code = {code}")

    print()
    print("=" * 78)
    print("SUMMARY (the two numbers the node was asked for)")
    print("=" * 78)
    print(f"  R^2, [x,x^2] -> band mask, CH_DRIVE DISCLOSED (leak-hypothesis "
          f"positive control) = {a_res['r2_band_x_and_xsq']:.6f}")
    print(f"  R^2, [x,x^2] -> band mask, CH_DRIVE SHIELDED   (discriminating "
          f"test, raw)           = {b_res['r2_band_x_and_xsq']:.6f}")
    print(f"  R^2, [x,x^2] -> band mask, CH_DRIVE SHIELDED + payload masked "
          f"(confound removed) = {b2_res['r2_band_x_and_xsq']:.6f}")


if __name__ == "__main__":
    sys.exit(0 if main() is None else 1)
