"""V15 R1 -- BED-M, t* = 2, n = 2048, N = 8 seeds. ARM PL vs softmax, on CPU.

THE DECIDING MEASUREMENT OF THE ROUND, and it is NOT what `CEQ_V15_CONTRACT.md`
PART IV registered. Four of that clause's five parts are corrected here and the
corrections are findings of this round, not conveniences:

  1. THE ARM IS THE AMENDED OPERATOR `l_ij = q_ij - C_j + s_j` with the
     value-zero BOS slot (`ceq/arm_pl.py`), not S-M's unnormalized hop, which
     `CEQ.V15.gate_zero_not_stochastic` refuted.
  2. BIND FIRST, THEN TRAIN. The label bind is re-measured at THIS CELL'S SHAPE
     (`s = 64`) before a single gradient step, and the run aborts if it does not
     hold. A trained number from an arm whose identity broke is worthless.
  3. THE KILL DIAGNOSTIC IS `sign(a_i)`, NOT `log|a|`. BED-M's coefficients are
     Rademacher (`scale/negation_scope.py:428`), so `log|a|` has `SST = 0` on
     the live band and returns the same value whatever the arm does -- it can
     neither fire nor fail to fire (MISTAKES.md M-18).
  4. `NRMSE < floor_1` AND `h_hat > 1` ARE ONE EVENT. `h_hat = t*(1 - NRMSE^2)`
     and `floor_1 = sqrt((t*-1)/t*)`, so `h_hat = 1` exactly at `NRMSE =
     floor_1`. The contract's scoreboard pays twice for one crossing.
  5. NO TOST. `N = 8` is far below the `N = 23` where the CI first fits and the
     `N = 70` where power reaches 0.80. A resolution statement only, with its
     achieved power printed beside it (MARS attack #2).

NOTHING IS REIMPLEMENTED. `Arm`, `LR`, `D_MODEL`, `bad` come from
`scale/m3_capability.py`; `nrmse`, `bootstrap_ci`, `calibrate_bar`,
`bar_verdict`, `M3_TASKS`, `chain_flipper_dependence` from
`scale/negation_scope.py`; `GATE_TOL` and `refuse_cross_device_pool` from
`scale/r10_capacity_sweep.py`; `resolution_delta` and `achieved_power_at_reference`
from `tests/mars_v15/test_resolution_statement_achieved_power.py`, which is the
instrument MARS's attack #2 filed; the arm and its binds from `ceq/arm_pl.py`.
The training loop below is `m3_capability.run_arm`'s loop with a model factory
argument -- same optimiser, same lr, same standardisation, same RED 0-step gate
-- because `r10_capacity_sweep.train_with_checkpoints` constructs `Arm(str)`
internally and cannot be handed an `ArmPL`, and both files are outside this
node's write scope.

CPU ONLY, BY DESIGN. `--device cuda` aborts: `calibrate_bar` is CPU-only and a
threshold carried across a device boundary is MISTAKES.md V-22.
`refuse_cross_device_pool` is called on the collected rows before any verdict,
so a mixed-device pool raises rather than averages.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
import importlib.util

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ceq import arm_pl
from scale import identity_manifest
from scale.m3_capability import Arm, D_MODEL, LR, bad
from scale.negation_scope import (M3_TASKS, nrmse, bootstrap_ci, calibrate_bar,
                                  bar_verdict, chain_flipper_dependence,
                                  CH_DRIVE, CH_FLIP)
from scale.r10_capacity_sweep import GATE_TOL, refuse_cross_device_pool

#: MARS attack #2's instrument, imported from the file that filed it rather
#: than re-derived. A second copy of a power formula is the defect the
#: identity-manifest module exists to catch, one domain over.
_mars = ROOT / "tests" / "mars_v15" / "test_resolution_statement_achieved_power.py"
_spec = importlib.util.spec_from_file_location("_mars_power", _mars)
_power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_power)
resolution_delta = _power.resolution_delta
achieved_power_at_reference = _power.achieved_power_at_reference
REFERENCE_EFFECT_IN_SD = _power.REFERENCE_EFFECT_IN_SD

S, D = 64, 24                  # the shape every e3 row in results/ uses
T_STAR = 2
BIND_BAR = 1e-6                # CEQ_V15_CONTRACT.md PART IV, R1
PUBLISHED_BIND = 6.6613381477509392e-16     # V15_ARM_PL.md section 2, at s = 8


# ------------------------------------------------------------------ the arms

def make_arm(kind: str, s: int):
    if kind == "arm_pl":
        return arm_pl.ArmPL(s, d_model=D_MODEL)
    return Arm(kind, s)


def train_one(kind, x_tr, y_tr, x_ev, y_ev, *, s, steps, seed):
    """`m3_capability.run_arm`'s loop, with a model factory and an eval hook.
    Same optimiser, lr, standardisation and RED gate; nothing else differs."""
    torch.manual_seed(seed)
    model = make_arm(kind, s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu = float(y_tr.mean())
    sigma = float(y_tr.std(unbiased=False)) or 1.0

    def raw_pred(x):
        model.eval()
        with torch.no_grad():
            return model(x) * sigma + mu

    r0t, r0e = nrmse(raw_pred(x_tr), y_tr), nrmse(raw_pred(x_ev), y_ev)
    red_ok = ((not bad(r0t)) and (not bad(r0e))
              and r0t >= 1.0 - GATE_TOL and r0e >= 1.0 - GATE_TOL)

    y_std = (y_tr - mu) / sigma
    t0 = time.time()
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x_tr), y_std).backward()
        opt.step()
    secs = time.time() - t0

    pe = raw_pred(x_ev)
    ev = nrmse(pe, y_ev)
    lo, hi = bootstrap_ci(pe, y_ev, seed=seed)
    return model, dict(kind=kind, seed=seed, steps=steps,
                       n_params=sum(p.numel() for p in model.parameters()),
                       nrmse0_train=r0t, nrmse0_eval=r0e, red_ok=bool(red_ok),
                       train_nrmse=nrmse(raw_pred(x_tr), y_tr), eval_nrmse=ev,
                       boot_lo=lo, boot_hi=hi, secs=round(secs, 3))


# ------------------------------------------------------- the required columns

@torch.no_grad()
def operator_columns(model, x, *, chunk=512):
    """Conservation drift and the arm's own gate, read off the SHIPPED forward
    path -- `model._operator`, the same call `forward` dispatches through, so
    this is not a second implementation of the operator
    (`tests/loop/test_m3_harness_operator_is_shipped.py`'s rule)."""
    model.eval()
    drift, drift_ex0, g_max, g_min = 0.0, 0.0, -math.inf, math.inf
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        q, k = model.wq(xb), model.wk(xb)
        if isinstance(model, arm_pl.ArmPL):
            g, s = model.heads(xb)
            a = model._operator(q, k, g, s)
            g_max = max(g_max, float(g.max()))
            g_min = min(g_min, float(g.min()))
        else:
            a = model._operator(q, k)
        e = (a.sum(-1) - 1.0).abs()
        #: ROW 0 IS REPORTED SEPARATELY, not dropped. `bench._softmax_operator`
        #: masks with `tril(-1)` -- strictly causal, diagonal EXCLUDED -- so its
        #: first row is empty and sums to 0, giving a drift of exactly 1 that is
        #: a convention and not a conservation failure. ARM PL includes the
        #: diagonal (`P_ii = 1`, which the label bind needs), so its row 0 sums
        #: to 1. Printing one number for both would call the control broken.
        drift = max(drift, float(e.max()))
        drift_ex0 = max(drift_ex0, float(e[..., 1:].max()))
    return drift, drift_ex0, g_max, g_min


@torch.no_grad()
def gate_features(model, x, live):
    """The arm's own gate output at the live positions: `(g_j, s_j)`.
    `[n*len(live), 2]`. Softmax has no gate head and returns None."""
    if not isinstance(model, arm_pl.ArmPL):
        return None
    model.eval()
    g, s = model.heads(x)
    return torch.stack([g[:, live].reshape(-1), s[:, live].reshape(-1)], dim=1)


def probe(feat_tr, y_tr, feat_ev, y_ev):
    """Least-squares linear probe FIT on train and SCORED on eval.

    M-2: a threshold calibrated on one draw is scored on another, never on
    itself. Returns (R^2 out-of-sample, sign accuracy out-of-sample).
    """
    a_tr = torch.cat([torch.ones(feat_tr.shape[0], 1, dtype=feat_tr.dtype), feat_tr], 1)
    a_ev = torch.cat([torch.ones(feat_ev.shape[0], 1, dtype=feat_ev.dtype), feat_ev], 1)
    w = torch.linalg.lstsq(a_tr.double(), y_tr.double().unsqueeze(1)).solution
    pred = (a_ev.double() @ w).squeeze(1)
    yv = y_ev.double()
    sst = float(((yv - yv.mean()) ** 2).sum())
    ssr = float(((yv - pred) ** 2).sum())
    r2 = float("nan") if sst == 0.0 else 1.0 - ssr / sst
    acc = float((torch.sign(pred) == torch.sign(yv)).double().mean())
    return r2, acc, sst


# ------------------------------------------------------------------ the binds

def bind_check(out):
    """The (L) bind, re-measured at THIS CELL'S SHAPE before any gradient step.

    Published at `s = 8`: 6.6613381477509392e-16 (`V15_ARM_PL.md` section 2).
    Re-measured here at `s = 8` (reproduction) and at `s = 64` (the cell), with
    all four planted negatives at the cell's shape so the rejection region is
    shown occupied at the shape the bind is being claimed at, not only at the
    shape it was published at.
    """
    rows = []
    for s in (8, S):
        a, b = arm_pl.draw(seed=15, s=s)
        rec = arm_pl.label_cell(a, b, seed=15, cell=f"arm_pl:none:s{s}")
        g, sh, v = arm_pl.oracle_heads(a, b)
        zq = torch.zeros(s + 1, 1, dtype=b.dtype)
        op = arm_pl.operator(zq, zq, g, sh)
        z = arm_pl.normalizer(g, sh)
        rows.append(dict(t="bind", s=s, residual=rec["residual"],
                         row_sum_drift=float((op.sum(-1) - 1.0).abs().max()),
                         min_entry=float(op[op > 0].min()),
                         normalizer_drift=float((z - 1.0).abs().max()),
                         a_max=rec["a_max"], v_max=rec["v_max"],
                         dyn_range_bound=rec["dyn_range_bound"],
                         manifest_hash=rec["manifest"]["hash"],
                         mutations={m: arm_pl.label_cell(a, b, mutation=m, seed=15)["residual"]
                                    for m in arm_pl.MUTATIONS[1:]}))
        out(rows[-1])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=["arm_pl", "softmax"])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--tag", default="v15_r1")
    a = ap.parse_args()

    if a.device == "cuda":
        print("ABORT: --device cuda cannot produce a scored verdict. "
              "calibrate_bar is CPU-only and GATE_TOL was established on CPU; "
              "carrying either across a device boundary is MISTAKES.md V-22. "
              "See V15_NEPTUNE_SYSTEMS.md condition 2.")
        return 1
    torch.set_num_threads(a.threads)

    jl = ROOT / "results" / f"{a.tag}.jsonl"
    log = open(jl, "a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec, default=float) + "\n")
        log.flush()

    t_run = time.time()
    floor1 = math.sqrt((T_STAR - 1) / T_STAR)
    task = f"e3_t{T_STAR}"
    print(f"=== V15 R1  {task}  s={S} d={D} n_train={a.n_train} n_eval={a.n_eval} "
          f"steps={a.steps} seeds={a.seeds} arms={a.arms} ===")
    print(f"  torch {torch.__version__}  torch.get_num_threads()={torch.get_num_threads()}  "
          f"device=cpu  floor_1={floor1:.10f}")
    emit(dict(t="header", tag=a.tag, task=task, t_star=T_STAR, s=S, d=D,
              n_train=a.n_train, n_eval=a.n_eval, steps=a.steps, seeds=a.seeds,
              arms=a.arms, lr=LR, d_model=D_MODEL, device="cpu",
              threads=torch.get_num_threads(), torch=torch.__version__,
              floor_1=floor1, when=time.strftime("%Y-%m-%d %H:%M:%S")))

    # ---------------------------------------------------- 1. BIND, THEN TRAIN
    print("\n=== 1. THE BIND, RE-MEASURED AT THIS CELL'S SHAPE (no gradient yet) ===")
    binds = bind_check(emit)
    for r in binds:
        print(f"  s={r['s']:>3}  (L) residual={r['residual']:.6e}  bar={BIND_BAR:.0e}  "
              f"rowsum drift={r['row_sum_drift']:.3e}  min entry={r['min_entry']:.3e}  "
              f"Z-1={r['normalizer_drift']:.3e}")
        print("        planted negatives: " +
              "  ".join(f"{k}={v:.6f}" for k, v in r["mutations"].items()))
    cell_bind = [r for r in binds if r["s"] == S][0]
    if not (cell_bind["residual"] <= BIND_BAR):
        print(f"ABORT: the label bind does NOT hold at the cell's shape s={S} "
              f"({cell_bind['residual']:.6e} > {BIND_BAR:.0e}). A trained number "
              f"from an arm whose identity broke is worthless. Nothing is trained.")
        emit(dict(t="abort", why="bind_failed_at_cell_shape",
                  residual=cell_bind["residual"], bar=BIND_BAR))
        return 1
    if min(cell_bind["mutations"].values()) < 0.1:
        print("ABORT: a planted negative did not fire at the cell's shape -- the "
              "bind's rejection region is empty here (MISTAKES.md V-24).")
        emit(dict(t="abort", why="planted_negative_silent", **cell_bind["mutations"]))
        return 1
    print(f"  BIND GREEN at s={S}: {cell_bind['residual']:.6e} <= {BIND_BAR:.0e}, "
          f"and all four planted negatives fire at O(1). Training is licensed.")

    # ------------------------------------------------------ 2. THE BAR
    batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[task]
    print("\n=== 2. BAR CALIBRATION (must pass before any arm is credited) ===")
    cal = calibrate_bar(n=a.n_eval, s=S, d=D, steps=600, lr=LR, batch_fn=batch_fn,
                        oracle_fn=oracle_fn, feature_fn=feature_fn)
    ok, why = bar_verdict(cal, flipper_dependence=chain_flipper_dependence(S, t_star=T_STAR))
    for k, v in cal.items():
        print(f"  {k:>22} {v:.6f}")
    print(f"  BAR {'CALIBRATED' if ok else 'BROKEN'} -- {why}")
    emit(dict(t="bar", ok=bool(ok), why=why, **{k: float(v) for k, v in cal.items()}))
    if not ok:
        print("ABORT: calibration bar failed; crediting nothing.")
        return 1

    # ------------------------------------------------------ 3. THE CELL
    head = S - 1 - T_STAR
    live = list(range(head + 1, S))          # the positions where a_i != 0
    x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345)
    a_ev = x_ev[:, live, CH_DRIVE].reshape(-1)
    print(f"\n=== 3. TRAIN + EVAL  (live band = positions {live}, head={head}) ===")
    print(f"  eval n={a.n_eval} seed=12345; per-seed train n={a.n_train} at the seed itself")

    rows, models = [], {}
    for kind in a.arms:
        for seed in a.seeds:
            x_tr, y_tr, _, _ = batch_fn(a.n_train, S, D, d_model=D_MODEL, seed=seed)
            model, r = train_one(kind, x_tr, y_tr, x_ev, y_ev, s=S,
                                 steps=a.steps, seed=seed)
            if not r["red_ok"]:
                print(f"INSTRUMENT BROKEN {kind} seed={seed}: 0-step "
                      f"{r['nrmse0_train']:.6f}/{r['nrmse0_eval']:.6f}")
                emit(dict(t="instrument_broken", **r))
                return 1

            drift, drift_ex0, g_max, g_min = operator_columns(model, x_ev)
            r["conservation_drift"] = drift
            r["conservation_drift_ex_row0"] = drift_ex0
            r["v_max"] = float(x_ev.abs().max())          # the value path is x
            if kind == "arm_pl":
                r["a_hat_max"] = math.exp(g_max)
                r["a_hat_min"] = math.exp(g_min)
                r["dyn_range_bound"] = (math.inf if r["a_hat_max"] >= 1.0
                                        else 1.0 / (1.0 - r["a_hat_max"]))
                a_tr = x_tr[:, live, CH_DRIVE].reshape(-1)
                ftr = gate_features(model, x_tr, live)
                fev = gate_features(model, x_ev, live)
                r["sign_r2"], r["sign_acc"], r["sign_sst"] = probe(ftr, torch.sign(a_tr),
                                                                  fev, torch.sign(a_ev))
                r["c"] = 2 * r["sign_acc"] - 1
                #: the arm's OWN recovered gate, `a_hat_j = exp(g_j)`, against
                #: the corpus's `a_j` -- VENUS's fourth probe row.
                r["gate_r2"], _, _ = probe(ftr[:, :1].exp(), a_tr, fev[:, :1].exp(), a_ev)
                #: M-18's dead diagnostic, printed so the corrected one can be
                #: read against it rather than merely asserted better.
                loga = torch.log(a_tr.abs())
                r["loga_sst"] = float(((loga - loga.mean()) ** 2).sum())
                #: THE 0-STEP CONTROL FOR THE PROBE, and it is not optional.
                #: `make_batch` fills the noise channels at `randn * 0.1` while
                #: `CH_DRIVE` carries `a` at modulus 1, so ANY random linear
                #: readout of `x` is dominated by the drive channel and a probe
                #: on an UNTRAINED gate already scores well above chance. A
                #: trained `p` reported without this control is an init property
                #: read as a learned one -- the shape of MISTAKES.md V-10. The
                #: model is rebuilt under the same seed, so its weights are
                #: bitwise the ones training started from.
                torch.manual_seed(seed)
                m0 = make_arm(kind, S)
                r["sign_acc_0step"] = probe(gate_features(m0, x_tr, live),
                                            torch.sign(a_tr),
                                            gate_features(m0, x_ev, live),
                                            torch.sign(a_ev))[1]
                r["gate_r2_0step"] = probe(gate_features(m0, x_tr, live)[:, :1].exp(),
                                           a_tr,
                                           gate_features(m0, x_ev, live)[:, :1].exp(),
                                           a_ev)[0]
            else:
                r["a_hat_max"] = 1.0        # g == 0: the parity point
                r["dyn_range_bound"] = math.inf
            r["eval_h_hat"] = T_STAR * (1.0 - r["eval_nrmse"] ** 2)
            r["dist_to_floor"] = r["eval_nrmse"] - floor1
            r.update(task=task, t_star=T_STAR, n_train=a.n_train, n_eval=a.n_eval,
                     s=S, d=D, d_model=D_MODEL, device="cpu",
                     threads=torch.get_num_threads(), torch_version=torch.__version__,
                     cell=f"{kind}:t{T_STAR}:n{a.n_train}:seed{seed}", floor_1=floor1)

            base = {k: r[k] for k in identity_manifest.CONFIG_FIELDS if k in r}
            base.update(cell=r["cell"], kind=kind, task=task, s=S, d=D,
                        d_model=D_MODEL, steps=a.steps, n_train=a.n_train,
                        n_eval=a.n_eval, seed=seed, device="cpu",
                        torch_version=torch.__version__)
            params = dict(model.named_parameters())
            if kind == "arm_pl":
                base.update(variant=arm_pl.VARIANT, g_setting="learned: g_head(x)",
                            s_setting="learned: s_head(x)", bos_value=0.0,
                            a_max=r["a_hat_max"], v_max=r["v_max"],
                            dyn_range_bound=r["dyn_range_bound"])
                man = arm_pl.cell_manifest(base, callables=(arm_pl.operator,
                                                            arm_pl.readout,
                                                            model.forward.__func__),
                                           params=params)
            else:
                man = identity_manifest.manifest(base, callables=(model._operator.__func__,
                                                                  model.forward.__func__),
                                                 params=params)
            r["manifest"] = man
            models[(kind, seed)] = model
            rows.append(r)
            emit(dict(t="cell", **r))
            extra = (f"  p={r['sign_acc']:.4f} c={r['c']:+.4f} gateR2={r['gate_r2']:+.4f}"
                     if kind == "arm_pl" else "")
            print(f"  [{kind:>7} seed={seed}] NRMSE={r['eval_nrmse']:.6f} "
                  f"h={r['eval_h_hat']:+.4f} drift={drift:.2e} "
                  f"boot[{r['boot_lo']:.4f},{r['boot_hi']:.4f}] {r['secs']:.1f}s{extra}")
            del x_tr, y_tr

    # ------------------------------------------------------ 4. THE VERDICT
    refuse_cross_device_pool(rows)          # one device or no verdict
    print("\n=== 4. SEED AGGREGATE (N=%d draws, never one seed read as the result -- M-4) ==="
          % len(a.seeds))
    tcrit = _power.stats.t.ppf(0.975, df=len(a.seeds) - 1)
    agg = {}
    for kind in a.arms:
        v = [r["eval_nrmse"] for r in rows if r["kind"] == kind]
        m, sd = statistics.fmean(v), statistics.stdev(v)
        half = tcrit * sd / math.sqrt(len(v))
        agg[kind] = dict(kind=kind, n=len(v), mean=m, sd=sd,
                         ci_lo=m - half, ci_hi=m + half,
                         h_hat=T_STAR * (1 - m ** 2), floor_1=floor1,
                         dist_to_floor=m - floor1,
                         crosses=bool(m + half < floor1),
                         achieved_power=achieved_power_at_reference(sd, len(v)),
                         resolution_delta=resolution_delta(sd, len(v)),
                         conservation_drift=max(r["conservation_drift"]
                                                for r in rows if r["kind"] == kind),
                         conservation_drift_ex_row0=max(r["conservation_drift_ex_row0"]
                                                        for r in rows if r["kind"] == kind),
                         a_hat_max=max(r["a_hat_max"] for r in rows if r["kind"] == kind),
                         v_max=max(r["v_max"] for r in rows if r["kind"] == kind),
                         device="cpu", threads=torch.get_num_threads())
        emit(dict(t="agg", **agg[kind]))
        print(f"  {kind:>7}: mean={m:.6f} sd={sd:.6f} 95%CI=[{agg[kind]['ci_lo']:.6f},"
              f"{agg[kind]['ci_hi']:.6f}] h={agg[kind]['h_hat']:+.4f} "
              f"floor_1={floor1:.6f} dist={m - floor1:+.6f} "
              f"CROSSES={agg[kind]['crosses']} power={agg[kind]['achieved_power']:.4f}")
        print(f"           drift(all rows)={agg[kind]['conservation_drift']:.3e} "
              f"drift(rows 1..s-1)={agg[kind]['conservation_drift_ex_row0']:.3e}  "
              f"max_j|V_j|={agg[kind]['v_max']:.6f}  a_hat_max={agg[kind]['a_hat_max']:.6f}  "
              f"1/(1-a_hat_max)={'inf' if agg[kind]['a_hat_max'] >= 1 else '%.6f' % (1 / (1 - agg[kind]['a_hat_max']))}")

    if "arm_pl" in agg and "softmax" in agg:
        d = [p["eval_nrmse"] - s["eval_nrmse"]
             for p, s in zip([r for r in rows if r["kind"] == "arm_pl"],
                             [r for r in rows if r["kind"] == "softmax"])]
        dm, dsd = statistics.fmean(d), statistics.stdev(d)
        delta = resolution_delta(dsd, len(d))
        pw = achieved_power_at_reference(dsd, len(d))
        con = dict(t="contrast", mean=dm, sd=dsd, n=len(d), delta=delta,
                   achieved_power=pw, reference_effect_in_sd=REFERENCE_EFFECT_IN_SD,
                   statement=(f"excludes a difference beyond Delta = "
                              f"t(.975,{len(d) - 1}).sd/sqrt({len(d)}) = {delta:.6f} "
                              f"NRMSE and nothing smaller"))
        emit(con)
        print(f"\n  PAIRED CONTRAST arm_pl - softmax: mean={dm:+.6f} sd={dsd:.6f}")
        print(f"  RESOLUTION STATEMENT (no TOST at N={len(d)}): {con['statement']}")
        print(f"  achieved power against a {REFERENCE_EFFECT_IN_SD}*sd true effect "
              f"at N={len(d)}: {pw:.4f}")

    if "arm_pl" in agg:
        def band(key):
            v = [r[key] for r in rows if r["kind"] == "arm_pl"]
            m, h = statistics.fmean(v), tcrit * statistics.stdev(v) / math.sqrt(len(v))
            return m, m - h, m + h
        pm, plo, phi = band("sign_acc")
        gm, glo, ghi = band("gate_r2")
        p0, p0lo, p0hi = band("sign_acc_0step")
        g0, g0lo, g0hi = band("gate_r2_0step")
        emit(dict(t="probe", sign_acc_mean=pm, sign_acc_ci=[plo, phi],
                  c_mean=2 * pm - 1, gate_r2_mean=gm, gate_r2_ci=[glo, ghi],
                  sign_acc_0step_mean=p0, sign_acc_0step_ci=[p0lo, p0hi],
                  gate_r2_0step_mean=g0, gate_r2_0step_ci=[g0lo, g0hi],
                  loga_sst=rows[0].get("loga_sst")))
        print(f"\n  sign(a) PROBE off the arm's gate, out of sample: p={pm:.6f} "
              f"95%CI=[{plo:.6f},{phi:.6f}]  c=2p-1={2 * pm - 1:+.6f}")
        print(f"    0-step control (same seeds, untrained gate): p={p0:.6f} "
              f"95%CI=[{p0lo:.6f},{p0hi:.6f}]  trained gain={pm - p0:+.6f}")
        print(f"  gate-vs-a probe R^2 (VENUS's row): {gm:+.6f} "
              f"95%CI=[{glo:+.6f},{ghi:+.6f}]")
        print(f"    0-step control: R^2={g0:+.6f} 95%CI=[{g0lo:+.6f},{g0hi:+.6f}]  "
              f"trained gain={gm - g0:+.6f}")
        print(f"  M-18's retired diagnostic, for contrast: SST of log|a| on the "
              f"live band = {rows[0].get('loga_sst', float('nan')):.6e}")

    wall = time.time() - t_run
    #: NEPTUNE's memory line is `torch.cuda.max_memory_allocated`, which does not
    #: exist on CPU. Process peak working set is the CPU-side analogue and is
    #: reported as such, not as the same quantity.
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak_gib = getattr(mi, "peak_wset", mi.rss) / 2 ** 30
    except Exception as exc:                                  # pragma: no cover
        peak_gib = float("nan")
        print(f"  (peak RSS unavailable: {exc})")
    per150 = statistics.fmean([r["secs"] for r in rows if r["kind"] == "arm_pl"]) \
        if any(r["kind"] == "arm_pl" for r in rows) else float("nan")
    emit(dict(t="wall", secs=wall, peak_wset_gib=peak_gib,
              arm_pl_secs_per_150_steps=per150))
    print(f"\n  peak process working set: {peak_gib:.3f} GiB "
          f"(NEPTUNE's 0.250 GiB is a CUDA allocator figure, not this quantity)")
    print(f"  ARM PL mean seconds per {a.steps} steps at n={a.n_train}: {per150:.2f} s "
          f"(NEPTUNE inherited ~18.19 s/150 steps, re-measured 12.2 s at 12 threads)")
    print(f"\n=== WALL CLOCK: {wall:.1f} s ===")
    print(f"WROTE {jl}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
