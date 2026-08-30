"""X29a -- the differential (common-mode deflated) second hop.

THE FINDING THIS ANSWERS. `scale/m3_capability.py:119-167`. `Arm._operator`
returns `bench._softmax_operator(q, k)` for BOTH `softmax` and `pivot_unsigned`,
so `pivot_unsigned` is `softmax` plus a second hop and nothing else. At
`K = |P| = s` the routed term `batched_pivot_hop2(a, P) = a[:, P] @ a[P, :]` is
exactly `a @ a`, and the K sweep at `t*=2, n_train=2048, steps=150`, seeds
0/1/2, threads=8 read `1.000329` there against a `0.950252` softmax control --
a NO READING -- while `K=8` read `0.960945`. Restricting the second hop to 8 of
64 columns was acting as an accidental common-mode filter.

WHAT THIS FILE BUILDS. The second hop propagated with a DEFLATED operator,
`a_d @ a_d` in place of `a @ a`. Hop 1 is untouched: the arm stays
`z = x + a @ x + hop2 @ x`, which is softmax's forward plus a differential
second hop.

TWO PROJECTORS ARE BUILT AND BOTH ARE MEASURED, because the naive one is not
causal and there is more than one way to repair it:

  * `deflate_tril`  : `tril(a - 1 pi^T, -1)`. Re-masks the naive deflation.
    Causal, but its rows no longer sum to zero, so it does NOT reject a DC
    input exactly.
  * `deflate`       : `a - diag(a 1) M`, `M` = `pi` restricted to each row's
    admissible prefix AND renormalised to unit mass there. Causal AND exactly
    DC-nulling, which is what the must-fire asks for.

RUN. `python scripts/v13_deflated_hop2.py` is the assert-based self-check
(derivation checks, the projector table, both must-fires, CMRR, the standing
causality guard, the softmax identity). `--sweep` adds the 3-seed trained
comparison against the pre-registered number.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from scale.m3_capability import Arm, D_MODEL, LR
from scale.negation_scope import M3_TASKS, nrmse
from scale.pivot_probe import batched_select_pivots, batched_pivot_hop2

#: The sweep's cell, restated so the deflated arm is measured on the same
#: instrument the raw K sweep was (scripts/v13_kpivot_sweep.py).
S, D, N_EVAL = 64, 24, 4096
TASK = "e3_t2"

#: PRE-REGISTERED in V13_X29A_DEFLATION.md: deflated K=64 recovers to at most
#: this -- the measured raw K=8 mean. NOT to be edited by any later run.
PREDICTION = 0.960945
SOFTMAX_REF, RAW_K64_REF, SD_REF = 0.950252, 1.000329, 0.010289


# ---------------------------------------------------------------------------
# The deflation.
# ---------------------------------------------------------------------------
def mean_row(a: torch.Tensor) -> torch.Tensor:
    """`pi = mean_i a[i, :]`, the least-squares best constant-row approximation
    to `a`. `1 pi^T` is the object a naive deflation subtracts."""
    return a.mean(-2)


def deflate_naive(a: torch.Tensor) -> torch.Tensor:
    """`a - 1 pi^T`. NOT CAUSAL -- kept only so the leak can be measured."""
    return a - mean_row(a).unsqueeze(-2)


def deflate_tril(a: torch.Tensor) -> torch.Tensor:
    """`tril(a - 1 pi^T, -1)`: the naive deflation put back under the mask.

    Causal by construction. Its row masses are `c_i - sum_{j<i} pi_j`, which is
    not zero, so a constant-along-positions input still passes with a residual
    gain -- measured in `self_check`, and the reason `deflate` exists.
    """
    return torch.tril(deflate_naive(a), -1)


def common_mode(a: torch.Tensor) -> torch.Tensor:
    """`M`: the causal common mode of `a`, one row per query position.

        M[i, j] = pi_j * 1[j < i] / sum_{j' < i} pi_{j'}

    `pi` restricted to the row's admissible prefix and renormalised to unit mass
    there. Three consequences, each checked in `self_check`:
      * `M` is supported on the same strictly-lower-triangular mask as `a`;
      * every nonempty row of `M` sums to 1, so `a - diag(a 1) M` annihilates
        any input constant along positions;
      * row `s-1` -- the row the arm reads out -- has the FULL prefix, so
        `M[s-1] = pi / sum(pi)` exactly, the global common mode.

    The scale of `pi` divides out in the renormalisation, so including the
    all-zero row 0 in the mean changes nothing.
    """
    s = a.shape[-1]
    mask = torch.ones(s, s, dtype=a.dtype, device=a.device).tril(-1)
    raw = mask * mean_row(a).unsqueeze(-2)                       # [..., s, s]
    return raw / raw.sum(-1, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)


def deflate(a: torch.Tensor) -> torch.Tensor:
    """`a_d = a - diag(a 1) M(a)`: `a` with its common mode removed, causally.

    `diag(a 1)` is the per-row mass `c_i`, which is 0 at row 0 and 1 at every
    other row of a causal softmax operator (sub-stochastic, not stochastic).
    Scaling the subtraction by `c_i` rather than by 1 is what makes `a_d 1 = 0`
    hold EXACTLY on the defective row too, not only on the stochastic rows.
    """
    return a - a.sum(-1, keepdim=True) * common_mode(a)


PROJECTORS = {"renorm": deflate, "tril": deflate_tril}


def assert_causal(name: str, h: torch.Tensor) -> None:
    """STANDING GUARD, not a one-time inspection: every operator the arm
    propagates with must carry exactly zero mass on and above the main
    diagonal. A deflation that leaks future information invalidates every
    number downstream of it, however good the number looks."""
    mass = float(h.detach().triu(0).abs().sum())
    if mass != 0.0:
        raise AssertionError(
            f"CAUSALITY LEAK in {name}: |mass| on/above diagonal = {mass:.6e}")


class DeflatedArm(Arm):
    """`Arm` with the second hop propagated by `a_d` instead of `a`.

    Everything else is the parent's: same `_operator` (softmax for
    `pivot_unsigned`), same pivot selection off the KEY, same
    `batched_pivot_hop2`, same MLP and readout, same `out[:, s-1]` tail -- an
    override that drops that tail broadcasts 64 against n and dies.

    `hop2=False` is the identity control: the forward reduces to `z = x + a @ x`
    followed by the shared head, which IS the `softmax` arm's forward.
    """

    def __init__(self, kind: str = "pivot_unsigned", s: int = S, *,
                 hop2: bool = True, projector: str = "renorm", guard: bool = True,
                 **kw):
        super().__init__(kind, s, **kw)
        self.hop2 = hop2
        self.projector = projector
        self.guard = guard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, s, _ = x.shape
        q, k = self.wq(x), self.wk(x)
        a = self._operator(q, k)                                  # [n,s,s]
        if self.guard:
            assert_causal("a", a)
        z = x + a @ x
        if self.hop2:
            ad = PROJECTORS[self.projector](a)
            h2 = batched_pivot_hop2(ad, batched_select_pivots(k, self.k_pivots))
            if self.guard:
                assert_causal(f"a_d[{self.projector}]", ad)
                assert_causal(f"hop2[{self.projector}]", h2)
            z = z + h2 @ x
        h = self.mlp(z)
        out = self.readout(h).squeeze(-1)                         # [n, s]
        return out[:, s - 1]


# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------
def rms(t: torch.Tensor) -> float:
    return float(t.pow(2).mean().sqrt())


def cm_energy(h: torch.Tensor) -> float:
    """Fraction of `||h||_F^2` carried by `1 pi^T`, `pi` the mean row of `h` --
    the decomposition the diagnosis was measured with (every row equal to the
    column mean, plus a differential remainder). Averaged over the batch."""
    cm = h.mean(-2, keepdim=True).expand_as(h)
    return float((cm.pow(2).sum((-2, -1)) / h.pow(2).sum((-2, -1))).mean())


def upper(h: torch.Tensor) -> tuple[int, float, float]:
    """(nonzero count, |mass|, max |entry|) on and above the main diagonal."""
    up = h.triu(0)
    return (int(torch.count_nonzero(up)), float(up.abs().sum()),
            float(up.abs().max()))


def planted(n: int, seed: int, task: str = TASK):
    """Inputs from the batch builder the arm trains on -- never synthetic."""
    return M3_TASKS[task][0](n, S, D, d_model=D_MODEL, seed=seed)


def operator_of(x: torch.Tensor, seed: int = 0):
    """(a, k, arm) through the PRODUCTION path: the arm's own projections and
    the arm's own `_operator`."""
    torch.manual_seed(seed)
    arm = DeflatedArm("pivot_unsigned", S, k_pivots=S)
    with torch.no_grad():
        q, k = arm.wq(x), arm.wk(x)
        a = arm._operator(q, k)
    return a, k, arm


def train_eval(make_model, *, n_train, steps, seed, x_eval, y_eval, batch_fn):
    """The K sweep's `run`, with the model built by a factory instead of by
    name. Same optimiser, same lr, same y-standardisation, same eval batch."""
    x, y, _, _ = batch_fn(n_train, S, D, d_model=D_MODEL, seed=seed)
    torch.manual_seed(seed)
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mu = float(y.mean())
    sd = float(y.std(unbiased=False)) or 1.0
    ys = (y - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x), ys).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x_eval) * sd + mu
    return float(nrmse(pred, y_eval))


# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------
def projector_table(task: str, n: int, seed: int = 0) -> None:
    """The independent measurement, reproduced here rather than quoted."""
    x, _y, _f, _p = planted(n, seed, task)
    a, _k, _arm = operator_of(x, seed)
    print(f"\n[T] projector table  task={task} n={n} seed={seed} untrained")
    print(f"    {'operator':<34} {'|mass| >= diag':>15} {'max entry':>12} "
          f"{'cm frac':>9}")
    rows = (("a (shipped)", a),
            ("1 pi^T", mean_row(a).unsqueeze(-2).expand_as(a)),
            ("a - 1 pi^T (naive)", deflate_naive(a)),
            ("tril(a - 1 pi^T, -1)", deflate_tril(a)),
            ("a - diag(a1) M (renorm)", deflate(a)))
    for name, h in rows:
        cnt, mass, mx = upper(h)
        print(f"    {name:<34} {mass:15.6e} {mx:12.6e} {cm_energy(h):9.4f}")
    print(f"    {'-- second hop --':<34}")
    for name, h in (("a @ a", a @ a),
                    ("tril-deflated squared", deflate_tril(a) @ deflate_tril(a)),
                    ("renorm-deflated squared", deflate(a) @ deflate(a))):
        print(f"    {name:<34} {'':>15} {'':>12} {cm_energy(h):9.4f}")


def self_check(n: int = 16, seed: int = 0) -> None:
    x, _y, _f, _p = planted(n, seed)
    a, k, arm = operator_of(x, seed)
    m = common_mode(a)
    ad = deflate(a)
    adt = deflate_tril(a)

    print(f"=== X29a SELF-CHECK  task={TASK} s={S} d={D} n={n} seed={seed} "
          f"untrained  threads={torch.get_num_threads()} ===")

    # -- 1. what `a` is -------------------------------------------------------
    rowsum = a.sum(-1)
    cnt_a, mass_a, _ = upper(a)
    print("\n[1] the operator")
    print(f"    row sums  min={float(rowsum.min()):.6f}  "
          f"max={float(rowsum.max()):.6f}   row 0 = "
          f"{float(rowsum[:, 0].abs().max()):.6f}  -> SUB-stochastic")
    print(f"    mass on/above diagonal: {mass_a:.6e}  ({cnt_a} entries)")
    assert_causal("a", a)
    assert abs(float(rowsum[:, 1:].min()) - 1.0) < 1e-5, "live rows not stochastic"
    assert float(rowsum[:, 0].abs().max()) == 0.0, "row 0 carries mass"

    # -- 2. the naive projector breaks causality ------------------------------
    naive = deflate_naive(a)
    cnt_nv, mass_nv, mx_nv = upper(naive)
    cnt_sq, mass_sq, mx_sq = upper(naive @ naive)
    print("\n[2] naive projector  a - 1 pi^T   (NOT USED, measured to kill it)")
    #: `s(s+1)/2` entries sit on/above the diagonal; the `s` of them in column
    #: `s-1` are zero because `pi_{s-1} = 0` (no row attends to the last
    #: position), leaving `s(s+1)/2 - s = s(s-1)/2` nonzero.
    print(f"    entries on/above diagonal: {cnt_nv // n}/example  "
          f"(structural s(s-1)/2 = {S * (S - 1) // 2})")
    print(f"    |mass| there {mass_nv:.6e}   max entry {mx_nv:.6e}")
    print(f"    at hop 2, (a - 1 pi^T)^2: {cnt_sq // n}/example, "
          f"|mass| {mass_sq:.6e}, max {mx_sq:.6e}  -> the leak compounds")
    assert cnt_nv > 0, "naive projector was expected to leak and did not"

    # -- 3. the two causal projectors -----------------------------------------
    live = m.sum(-1)[:, 1:]
    pi_hat = mean_row(a) / mean_row(a).sum(-1, keepdim=True)
    d_last = float((m[:, S - 1, :] - pi_hat).abs().max())
    print("\n[3] causal projectors")
    print(f"    M rows 1..s-1 mass  min={float(live.min()):.9f}  "
          f"max={float(live.max()):.9f}")
    print(f"    max |M[s-1] - pi/sum(pi)| = {d_last:.3e}  -> the readout row "
          f"carries the GLOBAL common mode")
    print(f"    max |a_d 1|   renorm {float(ad.sum(-1).abs().max()):.3e}   "
          f"tril {float(adt.sum(-1).abs().max()):.3e}   "
          f"(renorm is an EXACT DC null; tril is not)")
    print(f"    row 0 max |a_d| = {float(ad[:, 0].abs().max()):.3e}   "
          f"row 1 max |a_d| = {float(ad[:, 1].abs().max()):.3e}   "
          f"(one admissible predecessor is pure common mode)")
    #: IDEMPOTENCE HOLDS FOR THE PROJECTOR HELD FIXED, AND ONLY THEN. With `M`
    #: fixed at `M(a)`, `P(h) = h - diag(h 1) M` satisfies `P(P(a)) = P(a)`
    #: because `c(a_d) = 0`. RECOMPUTING the projector from `a_d` instead
    #: diverges (measured 1.204e+30 at this cell): `M`'s renormalisation divides
    #: by `sum_{j<i} pi_j`, which is a sum of NON-NEGATIVE terms for the softmax
    #: operator but a signed sum for `a_d`, and it crosses zero. `common_mode`
    #: is therefore defined for a non-negative operator only -- stated here
    #: rather than guarded away, because a signed deflation of a signed operator
    #: (the `pivot_signed` arm) would need a different projector.
    repro = ad - ad.sum(-1, keepdim=True) * m
    print(f"    idempotence (M fixed)  max |P(P(a)) - P(a)| = "
          f"{float((repro - ad).abs().max()):.3e}")
    assert_causal("M", m)
    assert_causal("a_d[renorm]", ad)
    assert_causal("a_d[tril]", adt)
    assert_causal("a_d[renorm] @ a_d[renorm]", ad @ ad)
    assert_causal("a_d[tril] @ a_d[tril]", adt @ adt)
    assert float((live - 1.0).abs().max()) < 1e-5, "projector rows not unit mass"
    assert d_last < 1e-6, "readout row is not the global common mode"
    assert float(ad.sum(-1).abs().max()) < 1e-6, "renorm rows do not sum to zero"
    assert float((repro - ad).abs().max()) < 1e-5, "deflation is not a projection"

    # -- 4. the projector table, both tasks -----------------------------------
    projector_table(TASK, n, seed)
    projector_table("e3_t8", 8, seed)

    # -- 5. energy: where the common mode sits --------------------------------
    hop2_k8 = batched_pivot_hop2(a, batched_select_pivots(k, 8))
    print("\n[5] common-mode share of the second hop's energy "
          f"(task={TASK}, n={n})")
    for name, h in (("a @ a (K=64)", a @ a),
                    ("pivot_hop2 K=8 (raw)", hop2_k8),
                    ("a_d @ a_d, tril", adt @ adt),
                    ("a_d @ a_d, renorm", ad @ ad)):
        print(f"    {name:<28} {cm_energy(h):.4f}")
    assert cm_energy(ad @ ad) < cm_energy(a @ a), "deflation did not reduce it"

    # -- 6. MUST-FIRE: both directions, planted inputs ------------------------
    #: Common mode: each planted example's own position-mean, broadcast over
    #: positions -- a DC level drawn from the batch the arm trains on.
    x_dc = x.mean(-2, keepdim=True).expand_as(x)
    #: Differential: the same planted batch with its common-mode component
    #: removed under the operator's own pi_hat. `x - 1 (pi_hat^T x)` is exactly
    #: what the READOUT row sees as pure differential, because `M[s-1] = pi_hat`.
    #: No input is pure differential for EVERY row at once: `<M[i], x> = 0` for
    #: all i forces every partial sum of `pi_j x_j` to vanish, hence `x_j = 0`
    #: wherever `pi_j > 0`. So the must-fire is stated at the readout row and
    #: the all-row figure is reported beside it.
    x_diff = x - pi_hat.unsqueeze(-2) @ x
    print("\n[6] MUST-FIRE  gain = rms(deflated response) / rms(raw response), "
          "planted inputs")
    gains = {}
    for pname, pfn in PROJECTORS.items():
        p = pfn(a)
        with torch.no_grad():
            dc_raw, dc_def = a @ x_dc, p @ x_dc
            df_raw, df_def = a @ x_diff, p @ x_diff
        g_cm = rms(dc_def) / rms(dc_raw)
        g_last = rms(df_def[:, S - 1]) / rms(df_raw[:, S - 1])
        g_all = rms(df_def) / rms(df_raw)
        gains[pname] = (g_cm, g_last, g_all)
        print(f"    [{pname:>6}] common mode  (all rows)    {g_cm:.6e}"
              f"   [need ~0]")
        print(f"    [{pname:>6}] differential (readout row) {g_last:.6f}"
              f"   [need >= 0.99]")
        print(f"    [{pname:>6}] differential (all rows)    {g_all:.6f}"
              f"   (reported, not asserted -- M is prefix-truncated at early rows)")
    g_cm, g_last, _ = gains["renorm"]
    assert g_cm < 1e-5, f"common mode not rejected: gain {g_cm}"
    assert g_last >= 0.99, f"differential attenuated: gain {g_last}"

    # -- 7. CMRR, per configuration ------------------------------------------
    print("\n[7] CMRR = 20 log10( raw DC amplitude / deflated DC amplitude ), "
          "at the output")
    with torch.no_grad():
        dc_raw1 = a @ x_dc
        cfg = [("hop 1 operator", dc_raw1, {p: PROJECTORS[p](a) @ x_dc
                                            for p in PROJECTORS}),
               ("hop 2 term K=64", (a @ a) @ x_dc,
                {p: (PROJECTORS[p](a) @ PROJECTORS[p](a)) @ x_dc
                 for p in PROJECTORS}),
               ("hop 2 term K=8 routed", hop2_k8 @ x_dc,
                {p: batched_pivot_hop2(PROJECTORS[p](a),
                                       batched_select_pivots(k, 8)) @ x_dc
                 for p in PROJECTORS})]
    for name, raw_out, defs in cfg:
        r = rms(raw_out)
        line = f"    {name:<24} raw {r:.6e}"
        for p, out in defs.items():
            dd = rms(out)
            db = 20 * math.log10(r / dd) if dd > 0 else float("inf")
            line += f"   {p} {dd:.3e} ({db:6.1f} dB)"
        print(line)

    # -- 8. identity: hop 2 off IS the softmax arm ----------------------------
    torch.manual_seed(1234)
    ctrl = Arm("softmax", S)
    torch.manual_seed(1234)
    off = DeflatedArm("pivot_unsigned", S, hop2=False, k_pivots=S)
    with torch.no_grad():
        a_out, b_out = ctrl(x), off(x)
    same = torch.equal(a_out, b_out)
    print("\n[8] identity control: DeflatedArm(hop2=False) vs Arm('softmax')")
    print(f"    bitwise equal = {same}   max |diff| = "
          f"{float((a_out - b_out).abs().max()):.3e}")
    assert same, "deflated arm with hop 2 off does not reproduce softmax"

    print("\nSELF-CHECK PASSED")


# ---------------------------------------------------------------------------
def sweep(a) -> None:
    batch_fn = M3_TASKS[TASK][0]
    x_eval, y_eval, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    kw = dict(n_train=a.n_train, steps=a.steps, x_eval=x_eval, y_eval=y_eval,
              batch_fn=batch_fn)
    print(f"=== X29a SWEEP  task={TASK} s={S} d={D} n_train={a.n_train} "
          f"steps={a.steps} seeds={a.seeds} n_eval={N_EVAL} "
          f"threads={a.threads} ===")
    print(f"    pre-registered: deflated K=64 mean eval NRMSE <= "
          f"{PREDICTION:.6f}   (raw K=8 = that number; raw K=64 = "
          f"{RAW_K64_REF:.6f}; softmax = {SOFTMAX_REF:.6f})")

    configs = [("softmax (1 hop, control)", lambda: Arm("softmax", S))]
    for k in a.ks:
        configs.append((f"raw           K={k}",
                        (lambda k: lambda: Arm("pivot_unsigned", S,
                                               k_pivots=k))(k)))
        for p in ("renorm", "tril"):
            configs.append((f"deflated {p:<6} K={k}",
                            (lambda k, p: lambda: DeflatedArm(
                                "pivot_unsigned", S, k_pivots=k,
                                projector=p))(k, p)))
    means = {}
    for name, factory in configs:
        t0 = time.time()
        vals = [train_eval(factory, seed=s, **kw) for s in a.seeds]
        means[name] = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {name:<26} mean {means[name]:.6f}  sd {sd:.6f}  "
              f"{['%.6f' % v for v in vals]}  {time.time() - t0:.0f}s", flush=True)

    kmax = max(a.ks)
    band = 2 * SD_REF / math.sqrt(len(a.seeds))
    print(f"\n  PRE-REGISTERED: deflated K={kmax} <= {PREDICTION:.6f}")
    for p in ("renorm", "tril"):
        key = f"deflated {p:<6} K={kmax}"
        if key in means:
            got = means[key]
            verdict = "HOLDS" if got <= PREDICTION else "FAILS"
            if abs(got - PREDICTION) < band:
                verdict += " (INCONCLUSIVE: gap inside 2*sd/sqrt(n_seeds) = "
                verdict += f"{band:.6f})"
            print(f"  MEASURED [{p:>6}]: {got:.6f}  "
                  f"({got - PREDICTION:+.6f})   {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="run the trained comparison (minutes) after the "
                         "self-check")
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[64])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    self_check(n=a.n_probe)
    if a.sweep:
        print()
        sweep(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
