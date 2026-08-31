"""V15 R1 -- BED-M, t* = 2, n = 2048, N = 8 seeds. ARM PL vs softmax.

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

`--device` IS LIVE AND EVERY RECORD NAMES THE DEVICE IT WAS MEASURED ON. Until
V16 this file carried a `--device` flag, called `refuse_cross_device_pool` on
the collected rows, and then wrote the STRING LITERAL `device="cpu"` on its
header, cell, manifest and summary records. The guard was therefore being fed a
constant: it could not refuse a cross-device pool because every row it saw said
`cpu` whatever had run. That is MISTAKES.md V-22's exact shape on the deciding
measurement of the round, and it is worse than an absent guard because it reads
as present. The literals are now `a.device`, and the abort that was the only
thing holding the defect shut is gone -- `V16_BAR_RECERT.md` certifies
`calibrate_bar` and the 0-step RED gate on cuda at this file's own shapes, so
the abort's stated reason ("`calibrate_bar` is CPU-only") is no longer true.
`refuse_cross_device_pool` is still called on the collected rows before any
verdict, and it now has something real to read.

THE DEVICE IS THREADED, NOT STAMPED. `batch_fn`, `calibrate_bar` and the model
all take the device; the model is CONSTRUCTED then `.to(device)`'d, which is
`train_with_checkpoints`'s ordering and what keeps the init bytes identical to
the cpu run's (`V16_BAR_RECERT.md` 6.1).

`arm_smprime` IS REACHABLE, AND R1' IS THE ARM IT IS REACHABLE FOR. Until V16
`make_arm` knew `arm_pl` and `m3_capability.Arm` only, so the arm
`CEQ_V16_CONTRACT.md` PART IV names for R1' -- `ceq/arm_smprime.py` -- could not
be constructed from the runner that scores it and R1' could not run at all.
Three things had to follow the factory rather than be assumed by it:

  A. THE IDENTITY POINT IS PER ARM. ARM PL's is zeroed heads; ARM S-M''s
     magnitude head is the magnitude itself under a CLOSED `[0,1]` cap, so
     zeroing it is `m = 0` -- the ANNIHILATING gate, where the path product is
     exactly `0` on every off-diagonal window. `identity_point` resolves it per
     module and `identity_bind` asserts the resolution with `torch.equal`
     against the all-ones causal mask BEFORE anything trains, so the harness bug
     aborts instead of publishing a whole-corpus zero as the arm's number.
  B. THE GATE COLUMNS ARE THE SAME QUANTITY FOR BOTH ARMS ONLY IF THE RECOVERY
     DIFFERS. ARM PL's gate head is `log a_hat`; ARM S-M''s feature already IS
     `Re(a_hat)`. `recovered_gate` is that one difference, so `gate_r2` is one
     column and not two.
  C. THE EARLY-WARNING COLUMN IS PER STEP, NOT PER CELL. PART IV R1' asks
     whether `lambda_hat` turns positive BEFORE the loss does; that is an
     ordering, and a column read once after training cannot check one. The
     `t="trace"` record carries both series per seed from step 0.

THE DETERMINISM REGIME IS RULING 1's, AND IT IS SET HERE. `main` calls
`torch.use_deterministic_algorithms(True, warn_only=True)` before the first
tensor is constructed, and the header record journals the flag, the `warn_only`
setting and the inherited `CUBLAS_WORKSPACE_CONFIG`. Until v17-K this file set
NO flag at all -- neither of the two regimes `V17K_RULINGS.md` RULING 1
describes -- and journalled `deterministic_algorithms: false`; the paragraph
that stood here said so and declined to decide the trade.

RULING 1 decides it. STRICT mode (`warn_only=False`) is NOT EXECUTABLE on a
cell this file takes: the cell trains for `--steps` gradient steps and the
backward of `cumprod` reaches `cumsum_cuda_kernel`, which has no deterministic
implementation in torch 2.5.1 (`COSTS.md` 1.6; `V17_R4_RETAKE_PRICE.md` 2.1
measures the raise on this box). `warn_only=True` warns there instead of
raising, which is exactly why the ruling ledger's A1.2 holds training to a
MEASURED FLOOR and not to bitwise -- so this file's cells are floor cells, and
the flag they run under is the run's flag rather than a second regime nobody
declared. `CUBLAS_WORKSPACE_CONFIG` must be EXPORTED BEFORE THE PROCESS STARTS:
set from inside Python after CUDA is initialised it does not take, and strict
mode then fails the FORWARD as well (`V17_R4_RETAKE_PRICE.md` 2.1). This file
therefore READS it and journals it rather than setting it, so a run launched
without it is legible in its own header instead of silently mis-described. See
`V16_R1_DEVICE_READY.md`, `V17_R4_RETAKE.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time
import importlib.util

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ceq import arm_phase, arm_pl, arm_smprime
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

#: The arms that carry a GATE HEAD, and therefore the arms PART IV's gate
#: columns (`gate-R^2`, `a_hat_max`, `lambda_hat`, the unit-root flag) are
#: readings of. `softmax` has no gate head; its columns are stated from `g == 0`
#: rather than measured, and `gate_columns` says so at the site.
GATED_ARMS = ("arm_pl", "arm_smprime")
#: The modules those two names resolve to, so a kind string is turned into a
#: module exactly once instead of at every branch.
ARM_MODULES = {"arm_pl": arm_pl, "arm_smprime": arm_smprime}

#: V17K_RULINGS.md RULING 5 -- "Hash into the identity manifest ... Q1/Q2 cite
#: it by hash." `identity_manifest.instrument_manifest` is NEW and ADDITIVE
#: (`scale/identity_manifest.py`'s existing `_sha`/`_code_fingerprint` are
#: reused, not reimplemented). Computed ONCE at import time, the same point
#: `_power`'s cross-module import above resolves, so every record this run
#: writes cites the same digest a caller could recompute later against this
#: exact tree.
#:
#: `reaches` is every callable that participates in SCORING a cell: the
#: batch/oracle/feature triple for the task this file runs, the two gated
#: arms' forward and operator paths, the control arm, and the guard/bar
#: functions between a draw and a verdict. NOT exhaustive -- see
#: `instrument_manifest`'s own docstring for what a named reach set never
#: claims to cover (a helper one of these calls, that nobody listed here).
_hash_task = f"e3_t{T_STAR}"
_hash_batch_fn, _hash_oracle_fn, _hash_feature_fn, _ = M3_TASKS[_hash_task]
INSTRUMENT_REACHES = (
    getattr(_hash_batch_fn, "func", _hash_batch_fn), _hash_oracle_fn, _hash_feature_fn,
    arm_pl.ArmPL.forward, arm_pl.operator, arm_pl.readout,
    arm_smprime.ArmSMPrime.forward, arm_smprime.operator, arm_smprime.readout,
    Arm.forward, calibrate_bar, bar_verdict, refuse_cross_device_pool,
    nrmse, bootstrap_ci,
)
INSTRUMENT_MANIFEST = identity_manifest.instrument_manifest(__file__, reaches=INSTRUMENT_REACHES)


# ------------------------------------------------------------------ the arms

def make_arm(kind: str, s: int):
    if kind == "arm_pl":
        return arm_pl.ArmPL(s, d_model=D_MODEL)
    if kind == "arm_smprime":
        return arm_smprime.ArmSMPrime(s, d_model=D_MODEL)
    return Arm(kind, s)


def identity_point(model):
    """Put an arm at ITS OWN identity, which is not the same point for the two
    gated arms and cannot be reached by a generic "zero the heads".

    ARM PL's identity is zeroed heads: `g = 0` makes `a_hat = exp(0) = 1` and
    `zero_heads()` is the shipped method. ARM S-M''s magnitude head is the
    magnitude ITSELF under a CLOSED `[0,1]` cap, so zeroing it gives `m = 0` --
    the ANNIHILATING gate, where the path product is exactly zero on every
    window (`V16Domain.lean` clause 4) and the whole corpus reads as a single
    zero. Its identity is `m = 1, theta = 0`, which is `identity_heads()`, and
    `V15_ARM_PHASE.md` section 7 item 4 names this the drop-in trap: a harness
    that zeroes heads reports a catastrophic ARM rather than a broken HARNESS.

    `identity_heads` is tried FIRST for that reason -- a closed-cap arm that
    also happened to carry a `zero_heads` would otherwise be sent to the
    annihilating gate by attribute order.
    """
    for name in ("identity_heads", "zero_heads"):
        fn = getattr(model, name, None)
        if fn is not None:
            fn()
            return model, name
    return model, "as-constructed"


def train_one(kind, x_tr, y_tr, x_ev, y_ev, *, s, steps, seed, device=None,
              trace=None, trace_every=10):
    """`m3_capability.run_arm`'s loop, with a model factory and an eval hook.
    Same optimiser, lr, standardisation and RED gate; nothing else differs.

    `trace(step, model, train_loss)` is PART IV R1's early-warning capture and
    it fires INSIDE the loop, at step 0 before the first update and every
    `trace_every` steps after. The contract's claim is that a diverging seed
    shows `lambda_hat > 0` BEFORE the loss does; a column read once after
    training cannot check an ordering, so the ordering is journalled per step
    and the claim is left checkable rather than asserted. At `--steps 0` the
    loop body never runs and the tail below is the step-0 reading.

    CONSTRUCT, THEN MOVE. `make_arm` runs before `.to(device)`, so the weights
    are drawn from the CPU global generator on every device and the init bytes
    are the ones every cpu reading in `results/` was taken through
    (`V16_BAR_RECERT.md` 6.1). Constructing on the device instead would draw
    different numbers and silently retire the comparison.
    """
    torch.manual_seed(seed)
    model = make_arm(kind, s)
    if device is not None:
        model = model.to(device)
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
    for t in range(steps):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x_tr), y_std)
        if trace is not None and t % trace_every == 0:
            trace(t, model, float(loss))
        loss.backward()
        opt.step()
    #: the final state, and at `--steps 0` the ONLY state. The in-loop hook
    #: fires at `t < steps` only, so step `steps` is never traced there and this
    #: is unconditional -- which is also what makes `--steps 0` produce the
    #: step-0 row PART IV requires.
    if trace is not None:
        model.eval()
        with torch.no_grad():
            trace(steps, model,
                  float(torch.nn.functional.mse_loss(model(x_tr), y_std)))
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
    drift, drift_ex0 = 0.0, 0.0
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        q, k = model.wq(xb), model.wk(xb)
        #: BOTH gated arms take their heads as the third and fourth arguments of
        #: the SAME `_operator` the module dispatches `forward` through, so the
        #: branch is on whether the arm has a gate head and not on which arm it
        #: is. `model.kind` is set by all three constructors.
        a = (model._operator(q, k, *model.heads(xb)) if model.kind in GATED_ARMS
             else model._operator(q, k))
        #: `.abs()` on a complex row sum is its MODULUS, which is the right
        #: conservation reading for ARM S-M': its operator is complex and a row
        #: that sums to `-1` or to `i` is as unconserved as one that sums to `3`.
        e = (a.sum(-1) - 1.0).abs()
        #: ROW 0 IS REPORTED SEPARATELY, not dropped. `bench._softmax_operator`
        #: masks with `tril(-1)` -- strictly causal, diagonal EXCLUDED -- so its
        #: first row is empty and sums to 0, giving a drift of exactly 1 that is
        #: a convention and not a conservation failure. ARM PL includes the
        #: diagonal (`P_ii = 1`, which the label bind needs), so its row 0 sums
        #: to 1. Printing one number for both would call the control broken.
        drift = max(drift, float(e.max()))
        drift_ex0 = max(drift_ex0, float(e[..., 1:].max()))
    return drift, drift_ex0


@torch.no_grad()
def gate_features(model, x, live):
    """The arm's own gate output at the live positions. Softmax has no gate
    head and returns None.

    ARM PL: `(g_j, s_j)`, `[n*len(live), 2]`.
    ARM S-M': `Re(a_hat_j) = m_j cos(theta_j)`, `[n*len(live), 1]`, taken from
    the module's OWN `gate_feature` so the feature is the arm's and not a second
    reading of its heads assembled here.
    """
    if model.kind == "arm_pl":
        model.eval()
        g, s = model.heads(x)
        return torch.stack([g[:, live].reshape(-1), s[:, live].reshape(-1)], 1)
    if model.kind == "arm_smprime":
        return model.gate_feature(x, live)
    return None


def recovered_gate(kind: str, feat):
    """`a_hat_j` as EACH arm recovers it, from its own gate feature.

    ARM PL's gate head is `log a_hat`, so the column exponentiates it (VENUS's
    fourth probe row). ARM S-M''s feature already IS `Re(a_hat)`; exponentiating
    that would score a different quantity under the same column name.
    """
    return feat[:, :1].exp() if kind == "arm_pl" else feat[:, :1]


@torch.no_grad()
def gate_columns(kind: str, model, x, live) -> dict:
    """PART IV's per-cell gate columns: `a_hat_max`, `lambda_hat`, the unit-root
    flag, the dynamic-range bound and the `Z` winding, all read off the arm's
    own shipped head path.

    `lambda_hat` is the per-position log growth rate of the arm's recurrence,
    `mean_j log|a_hat_j|` over the live band: a hop spanning `L` positions has
    modulus `exp(L * lambda_hat)`, so `lambda_hat > 0` is an arm whose hop grows
    and `lambda_hat < 0` one whose hop decays. THE UNIT-ROOT FLAG IS
    `a_hat_max >= 1` -- the same predicate `dyn_range_bound` already turns on in
    this file, not a second tolerance invented beside it, so the two columns
    cannot disagree.
    """
    model.eval()
    theta = None
    if kind == "arm_pl":
        #: ARM PL's gate head IS `log a_hat` (`ceq/arm_pl.py`): the log is read
        #: off the head, not taken of `exp` of it.
        lg = model.heads(x)[0][:, live]
    elif kind == "arm_smprime":
        m, theta = arm_smprime.blend(*model.heads(x), model.g)
        m, theta = m[:, live], theta[:, live]
        #: `log 0 = -inf`, and for a PATH-PRODUCT arm that is the true reading
        #: rather than an instrument failure: a window containing that position
        #: has hop exactly `0` (`V16Domain.lean` clause 4). It is reported as
        #: `-inf`, with the growth rate of the surviving positions
        #: (`lambda_hat_live`) and the annihilated fraction beside it -- the
        #: zeros are never quietly dropped into a mean that would read finite.
        lg = torch.log(m)
    else:
        #: softmax: `g == 0`, the parity point. `a_hat == 1` at every position
        #: BY CONSTRUCTION, so these are statements about the arm and not
        #: filled-in defaults; line 421's `a_hat_max = 1.0` already said so.
        return dict(a_hat_max=1.0, a_hat_min=1.0, lambda_hat=0.0,
                    lambda_hat_live=0.0, frac_gate_annihilated=0.0,
                    unit_root=True, dyn_range_bound=math.inf,
                    z_winding_max=None, z_winding_residual=None)
    fin = torch.isfinite(lg)
    a_max = float(lg.max().exp())
    col = dict(a_hat_max=a_max, a_hat_min=float(lg.min().exp()),
               lambda_hat=float(lg.mean()),
               lambda_hat_live=(float(lg[fin].mean()) if bool(fin.any())
                                else float("-inf")),
               frac_gate_annihilated=float((~fin).double().mean()),
               unit_root=bool(a_max >= 1.0),
               dyn_range_bound=(math.inf if a_max >= 1.0
                                else 1.0 / (1.0 - a_max)))
    if theta is None:
        #: A REAL GATE HAS NO PHASE, so its `Z` has no winding to read. Reported
        #: ABSENT and not as `0`: a zero here would be a measurement ARM PL
        #: never made, in a column a phase arm fills with one it did.
        col.update(z_winding_max=None, z_winding_residual=None)
    else:
        #: `ceq/arm_phase.py::winding`, X37's shipped instrument, not a second
        #: copy. `W_i = round(Phi_i/pi)` with the distance to that integer
        #: beside it. BED-M's own gates are Rademacher, so `theta in {0, pi}`
        #: and the residual is `0` for an arm that has learned the bed; ARM
        #: S-M''s phase head is a FREE real, so a residual near `0.5` says the
        #: trained phase left the `Z_2` sub-lattice. That is the reading, not an
        #: instrument defect -- unlike X37's setting, where the phase was
        #: quantized by construction.
        w, res = arm_phase.winding(theta)
        col.update(z_winding_max=int(w.abs().max()), z_winding_residual=res)
    return col


def probe(feat_tr, y_tr, feat_ev, y_ev):
    """Least-squares linear probe FIT on train and SCORED on eval.

    M-2: a threshold calibrated on one draw is scored on another, never on
    itself. Returns (R^2 out-of-sample, sign accuracy out-of-sample).
    """
    #: `ones_like` on a one-column slice, not `torch.ones(...)`: the intercept
    #: column has to land on the DEVICE the features are on, and `dtype=` alone
    #: does not carry that. A bare `torch.ones` was the only site in this file
    #: that would have raised on a cuda feature tensor.
    a_tr = torch.cat([torch.ones_like(feat_tr[:, :1]), feat_tr], 1)
    a_ev = torch.cat([torch.ones_like(feat_ev[:, :1]), feat_ev], 1)
    w = torch.linalg.lstsq(a_tr.double(), y_tr.double().unsqueeze(1)).solution
    pred = (a_ev.double() @ w).squeeze(1)
    yv = y_ev.double()
    sst = float(((yv - yv.mean()) ** 2).sum())
    ssr = float(((yv - pred) ** 2).sum())
    r2 = float("nan") if sst == 0.0 else 1.0 - ssr / sst
    acc = float((torch.sign(pred) == torch.sign(yv)).double().mean())
    return r2, acc, sst


# ------------------------------------------------------------------ the binds

def fires(residual) -> bool:
    """A planted negative FIRES iff it is not inside the bind bar.

    `not (r <= bar)` and not `r > bar`, because ARM S-M''s `exp_scan` negative
    reads `nan` -- `-inf - (-inf)` at an annihilating gate -- and `nan > bar` is
    `False`, which would read the most decisive negative in the round as silent.
    """
    return not (residual <= 0.1)


def identity_bind(kind: str, s: int, device=None) -> dict:
    """The SHIPPED MODULE at its own identity point, checked before training.

    This is where the drop-in trap is caught. ARM S-M''s magnitude head sits
    under a CLOSED `[0,1]` cap, so a harness that zeroes heads puts it at
    `m = 0`: the path product is then exactly `0` on every off-diagonal window
    and the arm reads as a whole-corpus annihilator -- a catastrophic-looking
    number that is a harness bug. The invariant asserted here is the one that
    separates the two points with no tolerance in it: AT THE IDENTITY THE
    MODULUS ROW IS THE ALL-ONES CAUSAL MASK. At the annihilating gate it is the
    IDENTITY MATRIX instead, because the empty product on the diagonal is still
    `1`. `torch.equal`, both ways, so a wrong identity point cannot pass.
    """
    torch.manual_seed(15)
    model = make_arm(kind, s)
    if device is not None:
        model = model.to(device)
    model, method = identity_point(model)
    x = torch.randn(4, s, D_MODEL, device=device)
    with torch.no_grad():
        if kind == "arm_smprime":
            row = arm_smprime.hop(*model.heads(x), g=model.g)[1]
        else:
            #: ARM PL's identity is `g = 0`, where its operator is bitwise the
            #: softmax control (`tests/arm_pl/test_arm_pl.py`), and its modulus
            #: row is the row-stochastic one rather than the all-ones mask. The
            #: all-ones invariant is ARM S-M''s; asserting it here would be a
            #: check on the wrong arm.
            row = None
    ones = None if row is None else torch.tril(torch.ones_like(row))
    #: LIVE, off the tensor the check was run on -- never the `--device`
    #: string, which is the shape of defect `V16_R1_DEVICE_READY.md` repaired
    #: at four sites in this file.
    return dict(t="identity", kind=kind, s=s, method=method,
                device=x.device.type,
                all_ones_causal_mask=(None if row is None
                                      else bool(torch.equal(row, ones))),
                annihilating=(None if row is None else
                              bool(torch.equal(row, torch.eye(
                                  s, dtype=row.dtype, device=row.device)
                                  .expand_as(row)))))


def bind_check(out, kinds, device=None):
    """The (L) bind, re-measured at THIS CELL'S SHAPE before any gradient step,
    once per GATED arm in the run.

    Published at `s = 8`: 6.6613381477509392e-16 (`V15_ARM_PL.md` section 2) for
    ARM PL and 1.110223e-16 (`V16_ARM_SMPRIME.md` section 3.3) for ARM S-M'.
    Re-measured here at `s = 8` (reproduction) and at `s = 64` (the cell), with
    every planted negative at the cell's shape so the rejection region is shown
    occupied at the shape the bind is being claimed at, not only at the shape it
    was published at.

    THE TWO ARMS' BINDS ARE DIFFERENT OBJECTS and the row says which. ARM PL's
    carries a row-sum drift and a normalizer drift because its operator is
    row-stochastic; ARM S-M''s bind is claimed at `beta = 0` where the row is
    the path product and does NOT sum to one, so those two fields are absent
    rather than filled with a number that would be a conservation failure only
    under a convention that arm does not use.
    """
    rows = []
    for kind in [k for k in kinds if k in GATED_ARMS]:
        rows.append(identity_bind(kind, S, device=device))
        out(rows[-1])
        for s in (8, S):
            #: THE BIND IS RE-MEASURED ON THE RUN'S OWN DEVICE. A bind row taken
            #: on cpu in front of cells taken on cuda licenses training against
            #: an identity that was never checked where the cells were computed
            #: -- F1's defect one level down. Both arms' draws are made on the
            #: HOST and moved after, so the drawn bytes are the same on both
            #: devices.
            mod = ARM_MODULES[kind]
            if kind == "arm_pl":
                a, b = arm_pl.draw(seed=15, s=s, device=device)
            else:
                a, b = arm_smprime.bedm_draw(seed=15, s=s, device=device)
            rec = mod.label_cell(a, b, seed=15, cell=f"{kind}:none:s{s}")
            row = dict(t="bind", kind=kind, s=s, residual=rec["residual"],
                       device=rec["device"], v_max=rec["v_max"],
                       manifest_hash=rec["manifest"]["hash"],
                       mutations={m: mod.label_cell(a, b, mutation=m,
                                                    seed=15)["residual"]
                                  for m in mod.MUTATIONS[1:]})
            if kind == "arm_pl":
                g, sh, v = arm_pl.oracle_heads(a, b)
                zq = torch.zeros(s + 1, 1, dtype=b.dtype, device=b.device)
                op = arm_pl.operator(zq, zq, g, sh)
                z = arm_pl.normalizer(g, sh)
                row.update(row_sum_drift=float((op.sum(-1) - 1.0).abs().max()),
                           min_entry=float(op[op > 0].min()),
                           normalizer_drift=float((z - 1.0).abs().max()),
                           a_max=rec["a_max"],
                           dyn_range_bound=rec["dyn_range_bound"])
            else:
                row.update(a_max=rec["m_max"], n_zero_gates=rec["n_zero_gates"],
                           beta=rec["beta"], route=rec["route"])
            rows.append(row)
            out(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=["arm_pl", "softmax"])
    #: PART IV R1's early-warning cadence. Step 0 and the final step are always
    #: traced; this is the spacing in between, and it is the resolution at which
    #: "lambda_hat turned positive before the loss did" can be read.
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--tag", default="v15_r1")
    a = ap.parse_args()

    #: THE V15 ABORT IS GONE, and what replaces it is a certification rather
    #: than a lifted guard. It read "calibrate_bar is CPU-only and GATE_TOL was
    #: established on CPU"; `V16_BAR_RECERT.md` 4 and 6 measured both on cuda at
    #: THIS FILE'S shapes -- the bar's worst clause at 8.6% of its own
    #: tolerance, the 0-step gate 5,780x inside `GATE_TOL`. What the abort was
    #: really holding shut was the four `device="cpu"` literals below it, which
    #: blinded `refuse_cross_device_pool`; those are now `a.device`.
    dev = a.device
    #: RULING 1's regime, SET rather than merely read. `warn_only=True` is the
    #: whole run's flag; strict mode is not executable on a cell that trains
    #: (`cumsum_cuda_kernel` in the backward of `cumprod`), so this is the only
    #: one of the two ruled regimes an R1/R1' cell can be taken under. Set here
    #: -- before the bind, the bar calibration, the eval draw or any cell -- so
    #: one run is one regime instead of a flag that changes partway down.
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    #: READ, not set: it only takes effect if it was exported before the process
    #: started. A `None` here says the run inherited nothing and is a fact about
    #: the reading, not a knob this file may turn.
    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    print(f"  torch {torch.__version__}  torch.get_num_threads()={torch.get_num_threads()}  "
          f"device={dev}  deterministic_algorithms={det} warn_only={warn_only}  "
          f"CUBLAS_WORKSPACE_CONFIG={cublas_cfg!r}  floor_1={floor1:.10f}")
    #: F1 SITE 1 of 4 (header). `device="cpu"` was a literal here.
    #: `deterministic_algorithms` is journalled beside it because on cuda that
    #: flag names the reduction regime, and `V16_BAR_RECERT.md` 4 certified the
    #: bar with it ON while this file does not set it. A record that names the
    #: device but not the regime still under-describes the reading.
    #: RULING 5's citation, on the header too: a reader with only the run's
    #: console log or its header line (no cell reached yet) can already name
    #: the instrument that will score every cell below it.
    emit(dict(t="header", tag=a.tag, task=task, t_star=T_STAR, s=S, d=D,
              n_train=a.n_train, n_eval=a.n_eval, steps=a.steps, seeds=a.seeds,
              arms=a.arms, lr=LR, d_model=D_MODEL, device=dev,
              deterministic_algorithms=bool(det),
              deterministic_warn_only=bool(warn_only),
              cublas_workspace_config=cublas_cfg,
              threads=torch.get_num_threads(), torch=torch.__version__,
              floor_1=floor1, instrument_hash=INSTRUMENT_MANIFEST["hash"],
              when=time.strftime("%Y-%m-%d %H:%M:%S")))

    # ---------------------------------------------------- 1. BIND, THEN TRAIN
    print("\n=== 1. THE BIND, RE-MEASURED AT THIS CELL'S SHAPE (no gradient yet) ===")
    binds = bind_check(emit, a.arms, device=dev)
    for r in binds:
        if r["t"] == "identity":
            print(f"  [{r['kind']}] identity point via {r['method']}(): "
                  f"all-ones causal mask={r['all_ones_causal_mask']}  "
                  f"annihilating={r['annihilating']}  device={r['device']}")
            continue
        extra = (f"  rowsum drift={r['row_sum_drift']:.3e}"
                 f"  min entry={r['min_entry']:.3e}"
                 f"  Z-1={r['normalizer_drift']:.3e}" if "row_sum_drift" in r
                 else f"  beta={r['beta']:.1f} route={r['route']}"
                      f"  zero gates={r['n_zero_gates']}")
        print(f"  [{r['kind']:>11}] s={r['s']:>3}  (L) residual={r['residual']:.6e}  "
              f"bar={BIND_BAR:.0e}{extra}")
        print("        planted negatives: " +
              "  ".join(f"{k}={v:.6f}" for k, v in r["mutations"].items()))

    #: THE IDENTITY POINT IS A GATE, not a printout. A run whose harness put the
    #: arm at the annihilating gate would train against a whole-corpus zero and
    #: publish it as the arm's number.
    for r in [x for x in binds if x["t"] == "identity"]:
        if r["annihilating"] or r["all_ones_causal_mask"] is False:
            print(f"ABORT: {r['kind']}'s identity point via {r['method']}() is "
                  f"NOT the identity gate (annihilating={r['annihilating']}). "
                  f"A closed-cap arm reset to zeroed heads sits at m = 0, where "
                  f"the path product is exactly zero on every window -- a "
                  f"harness bug that reads as a catastrophic arm.")
            #: `**{k: v ...}` and not `**r`: the identity row already carries a
            #: `t`, and `dict(t="abort", **r)` raises `TypeError` -- which would
            #: turn the abort into a crash and lose the journal row that says
            #: WHY the run stopped.
            emit(dict(t="abort", why="identity_point_is_the_annihilating_gate",
                      **{k: v for k, v in r.items() if k != "t"}))
            return 1

    for kind in [k for k in a.arms if k in GATED_ARMS]:
        cb = [r for r in binds if r["t"] == "bind" and r["kind"] == kind
              and r["s"] == S][0]
        if not (cb["residual"] <= BIND_BAR):
            print(f"ABORT: {kind}'s label bind does NOT hold at the cell's shape "
                  f"s={S} ({cb['residual']:.6e} > {BIND_BAR:.0e}). A trained "
                  f"number from an arm whose identity broke is worthless. "
                  f"Nothing is trained.")
            emit(dict(t="abort", why="bind_failed_at_cell_shape", kind=kind,
                      residual=cb["residual"], bar=BIND_BAR))
            return 1
        silent = [m for m, v in cb["mutations"].items() if not fires(v)]
        if silent:
            print(f"ABORT: {kind}'s planted negative(s) {silent} did not fire at "
                  f"the cell's shape -- the bind's rejection region is empty "
                  f"here (MISTAKES.md V-24).")
            emit(dict(t="abort", why="planted_negative_silent", kind=kind,
                      **cb["mutations"]))
            return 1
        print(f"  BIND GREEN for {kind} at s={S}: {cb['residual']:.6e} <= "
              f"{BIND_BAR:.0e}, and all {len(cb['mutations'])} planted negatives "
              f"fire. Training is licensed.")

    # ------------------------------------------------------ 2. THE BAR
    batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[task]
    print("\n=== 2. BAR CALIBRATION (must pass before any arm is credited) ===")
    #: `device=dev` -- the parameter `V16_BAR_RECERT.md` 2 added, threaded.
    #: The bar is a PER-DEVICE measurement: reading a cuda cell against a bar
    #: calibrated on cpu is V-22 one level up, and the `t="bar"` record below
    #: is inside `refuse_cross_device_pool`'s reach precisely so it cannot be.
    cal = calibrate_bar(n=a.n_eval, s=S, d=D, steps=600, lr=LR, batch_fn=batch_fn,
                        oracle_fn=oracle_fn, feature_fn=feature_fn, device=dev)
    ok, why = bar_verdict(cal, flipper_dependence=chain_flipper_dependence(S, t_star=T_STAR))
    for k, v in cal.items():
        print(f"  {k:>22} {v:.6f}")
    print(f"  BAR {'CALIBRATED' if ok else 'BROKEN'} -- {why}")
    emit(dict(t="bar", ok=bool(ok), why=why, device=dev,
              **{k: float(v) for k, v in cal.items()}))
    if not ok:
        print("ABORT: calibration bar failed; crediting nothing.")
        return 1

    # ------------------------------------------------------ 3. THE CELL
    head = S - 1 - T_STAR
    live = list(range(head + 1, S))          # the positions where a_i != 0
    x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,
                                      device=dev)
    a_ev = x_ev[:, live, CH_DRIVE].reshape(-1)
    print(f"\n=== 3. TRAIN + EVAL  (live band = positions {live}, head={head}) ===")
    print(f"  eval n={a.n_eval} seed=12345; per-seed train n={a.n_train} at the seed itself")

    rows, models = [], {}
    for kind in a.arms:
        for seed in a.seeds:
            x_tr, y_tr, _, _ = batch_fn(a.n_train, S, D, d_model=D_MODEL,
                                        seed=seed, device=dev)
            a_tr = x_tr[:, live, CH_DRIVE].reshape(-1)

            def early_warning(step, m, loss, _kind=kind, _seed=seed,
                              _x_tr=x_tr, _a_tr=a_tr):
                """PART IV R1's early-warning row, INSIDE the loop.

                `lambda_hat` and gate-`R^2` per step per seed, so the contract's
                claim -- a diverging seed shows `lambda_hat > 0` BEFORE the loss
                does -- is an ordering readable off the journal instead of a
                sentence written after the fact. The verdict is deliberately
                NOT computed here: the row carries the two series and the reader
                compares them.
                """
                #: the gate columns are read on the EVAL batch, which is the
                #: tensor the `t="cell"` row publishes them from, so the last
                #: trace step and the cell agree to the bit and the ordering
                #: question is asked of one quantity rather than two. The loss
                #: beside them is the training loss, because that is the series
                #: the claim says `lambda_hat` moves ahead of.
                col = gate_columns(_kind, m, x_ev, live)
                ftr = gate_features(m, _x_tr, live)
                g2 = None
                if ftr is not None:
                    g2 = probe(recovered_gate(_kind, ftr), _a_tr,
                               recovered_gate(_kind, gate_features(m, x_ev, live)),
                               a_ev)[0]
                emit(dict(t="trace", kind=_kind, seed=_seed, step=step,
                          #: LIVE, off the tensor the arm was read on.
                          device=x_ev.device.type, train_loss=loss, gate_r2=g2,
                          lambda_hat=col["lambda_hat"],
                          lambda_hat_live=col["lambda_hat_live"],
                          frac_gate_annihilated=col["frac_gate_annihilated"],
                          a_hat_max=col["a_hat_max"],
                          unit_root=col["unit_root"]))

            model, r = train_one(kind, x_tr, y_tr, x_ev, y_ev, s=S,
                                 steps=a.steps, seed=seed, device=dev,
                                 trace=early_warning, trace_every=a.trace_every)
            if not r["red_ok"]:
                print(f"INSTRUMENT BROKEN {kind} seed={seed}: 0-step "
                      f"{r['nrmse0_train']:.6f}/{r['nrmse0_eval']:.6f}")
                emit(dict(t="instrument_broken", **r))
                return 1

            drift, drift_ex0 = operator_columns(model, x_ev)
            r["conservation_drift"] = drift
            r["conservation_drift_ex_row0"] = drift_ex0
            r["v_max"] = float(x_ev.abs().max())          # the value path is x
            #: PART IV's gate columns for EVERY cell, softmax included, so the
            #: control's `a_hat_max`, `lambda_hat` and unit-root flag come from
            #: the same function the arms' do rather than from two conventions.
            r.update(gate_columns(kind, model, x_ev, live))
            #: NOT AVAILABLE, and named rather than defaulted:
            #: distance-to-skyline. The v15/v16 scan-skyline module is not
            #: built -- `tests/mars_v15/test_skyline_gate_containment.py::
            #: _find_v15_skyline` searches four import paths and skips because
            #: none resolves, and R-SKY is a separate deciding measurement.
            #: Emitting a number here would be a distance to an object that does
            #: not exist.
            r["dist_to_skyline"] = None
            r["dist_to_skyline_why"] = "no v15/v16 scan-skyline module exists (R-SKY)"
            if kind in GATED_ARMS:
                ftr = gate_features(model, x_tr, live)
                fev = gate_features(model, x_ev, live)
                r["sign_r2"], r["sign_acc"], r["sign_sst"] = probe(ftr, torch.sign(a_tr),
                                                                  fev, torch.sign(a_ev))
                r["c"] = 2 * r["sign_acc"] - 1
                #: the arm's OWN recovered gate against the corpus's `a_j` --
                #: VENUS's fourth probe row. `recovered_gate` is what makes this
                #: the same QUANTITY for both arms: ARM PL's head is `log a_hat`
                #: and exponentiates, ARM S-M''s feature already is `Re(a_hat)`.
                r["gate_r2"], _, _ = probe(recovered_gate(kind, ftr), a_tr,
                                           recovered_gate(kind, fev), a_ev)
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
                #: AS CONSTRUCTED, not at the identity point. The control's job
                #: is to say what the probe reads off the weights TRAINING
                #: STARTED FROM, so `identity_point` must NOT be called here --
                #: doing so would compare a trained arm against a hand-set one
                #: and, for ARM S-M', would also move `a_hat_max` to exactly 1.
                torch.manual_seed(seed)
                m0 = make_arm(kind, S).to(dev)
                f0t, f0e = gate_features(m0, x_tr, live), gate_features(m0, x_ev, live)
                r["sign_acc_0step"] = probe(f0t, torch.sign(a_tr),
                                            f0e, torch.sign(a_ev))[1]
                r["gate_r2_0step"] = probe(recovered_gate(kind, f0t), a_tr,
                                           recovered_gate(kind, f0e), a_ev)[0]
                r.update({f"{k}_0step": v for k, v in
                          gate_columns(kind, m0, x_ev, live).items()})
            r["eval_h_hat"] = T_STAR * (1.0 - r["eval_nrmse"] ** 2)
            r["dist_to_floor"] = r["eval_nrmse"] - floor1
            #: F1 SITE 2 of 4 (cell record). This is the row
            #: `refuse_cross_device_pool` actually reads; the literal here was
            #: the whole of the defect.
            #:
            #: RULING 5's citation, WIRED HERE rather than left for a future
            #: run to remember: every `t="cell"` row this file ever writes now
            #: names the instrument that produced it, by hash, alongside the
            #: cell's OWN identity manifest below (`r["manifest"]`) -- the two
            #: are different objects. `r["manifest"]` answers "what config,
            #: code, shapes and rng produced THIS cell"; `instrument_hash`
            #: answers "which build of the SCRIPT ran it", which
            #: `identity_manifest.manifest`'s per-cell `code` component cannot
            #: -- it fingerprints the callables named at that call site, not
            #: the file that dispatches to them.
            r.update(task=task, t_star=T_STAR, n_train=a.n_train, n_eval=a.n_eval,
                     s=S, d=D, d_model=D_MODEL, device=dev,
                     threads=torch.get_num_threads(), torch_version=torch.__version__,
                     cell=f"{kind}:t{T_STAR}:n{a.n_train}:seed{seed}", floor_1=floor1,
                     instrument_hash=INSTRUMENT_MANIFEST["hash"])

            base = {k: r[k] for k in identity_manifest.CONFIG_FIELDS if k in r}
            #: F1 SITE 3 of 4 (identity manifest). `device` is a first-class
            #: `identity_manifest.CONFIG_FIELDS` entry and the literal made a
            #: cuda cell hash identically to a cpu one -- a manifest asserting
            #: the run happened somewhere it did not.
            base.update(cell=r["cell"], kind=kind, task=task, s=S, d=D,
                        d_model=D_MODEL, steps=a.steps, n_train=a.n_train,
                        n_eval=a.n_eval, seed=seed, device=dev,
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
            elif kind == "arm_smprime":
                #: `SMP_FIELDS`, filled from the TRAINED module rather than from
                #: the constructor's defaults: `beta`, `qk` and `g` are
                #: `nn.Parameter`s the harness moves, so a manifest that hashed
                #: their initial values would assert a corner the cell was not
                #: measured at. `route` is `"product"` because that is the route
                #: `ArmSMPrime._operator` dispatches; `exp_scan` is reachable
                #: only through the module's planted negative and never from a
                #: trained cell.
                base.update(variant=arm_smprime.VARIANT, route="product",
                            beta=float(model.beta), qk=float(model.qk),
                            g=float(model.g),
                            m_setting="learned: m_head(x)",
                            theta_setting="learned: theta_head(x)",
                            v_setting="x, UNRESCALED",
                            m_max=r["a_hat_max"], v_max=r["v_max"],
                            n_zero_gates=int(round(r["frac_gate_annihilated"]
                                                   * a.n_eval * len(live))))
                man = arm_smprime.cell_manifest(
                    base, callables=(arm_smprime.operator, arm_smprime.readout,
                                     arm_smprime.hop, arm_smprime.path_product,
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
                     if kind in GATED_ARMS else "")
            print(f"  [{kind:>11} seed={seed}] NRMSE={r['eval_nrmse']:.6f} "
                  f"h={r['eval_h_hat']:+.4f} drift={drift:.2e} "
                  f"boot[{r['boot_lo']:.4f},{r['boot_hi']:.4f}] {r['secs']:.1f}s{extra}")
            print(f"                 a_max={r['a_hat_max']:.6f} "
                  f"lambda={r['lambda_hat']:+.6f} (live {r['lambda_hat_live']:+.6f}, "
                  f"annihilated {r['frac_gate_annihilated']:.4f}) "
                  f"unit_root={r['unit_root']} "
                  f"Zwind={r['z_winding_max']}/{r['z_winding_residual']} "
                  f"dist_to_skyline={r['dist_to_skyline']}")
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
                         #: F1 SITE 4 of 4 (seed-aggregate summary).
                         device=dev, threads=torch.get_num_threads())
        emit(dict(t="agg", **agg[kind]))
        print(f"  {kind:>7}: mean={m:.6f} sd={sd:.6f} 95%CI=[{agg[kind]['ci_lo']:.6f},"
              f"{agg[kind]['ci_hi']:.6f}] h={agg[kind]['h_hat']:+.4f} "
              f"floor_1={floor1:.6f} dist={m - floor1:+.6f} "
              f"CROSSES={agg[kind]['crosses']} power={agg[kind]['achieved_power']:.4f}")
        print(f"           drift(all rows)={agg[kind]['conservation_drift']:.3e} "
              f"drift(rows 1..s-1)={agg[kind]['conservation_drift_ex_row0']:.3e}  "
              f"max_j|V_j|={agg[kind]['v_max']:.6f}  a_hat_max={agg[kind]['a_hat_max']:.6f}  "
              f"1/(1-a_hat_max)={'inf' if agg[kind]['a_hat_max'] >= 1 else '%.6f' % (1 / (1 - agg[kind]['a_hat_max']))}")

    #: THE PAIRED CONTRAST IS AGAINST THE GATED ARM THE RUN ACTUALLY CARRIES.
    #: R1' runs `arm_smprime` against the same control R1 ran `arm_pl` against;
    #: a contrast hard-coded to `arm_pl` would silently print nothing for it.
    gated = [k for k in a.arms if k in GATED_ARMS and k in agg]
    for gk in (gated if "softmax" in agg else []):
        d = [p["eval_nrmse"] - s["eval_nrmse"]
             for p, s in zip([r for r in rows if r["kind"] == gk],
                             [r for r in rows if r["kind"] == "softmax"])]
        dm, dsd = statistics.fmean(d), statistics.stdev(d)
        delta = resolution_delta(dsd, len(d))
        pw = achieved_power_at_reference(dsd, len(d))
        #: `device` on the DERIVED rows too. They are not readings and
        #: `refuse_cross_device_pool` is not called on them, but a consumer
        #: handing the whole journal to the guard reads a missing field as
        #: `"cpu"` (its documented rule) and would refuse a cuda journal
        #: against its own summary -- the `t="ceiling"` false positive
        #: `V16_BAR_RECERT.md` 5 pinned. These rows DO come from one device,
        #: so naming it is the honest fix rather than a filter.
        con = dict(t="contrast", device=dev, kind=gk, mean=dm, sd=dsd, n=len(d), delta=delta,
                   achieved_power=pw, reference_effect_in_sd=REFERENCE_EFFECT_IN_SD,
                   statement=(f"excludes a difference beyond Delta = "
                              f"t(.975,{len(d) - 1}).sd/sqrt({len(d)}) = {delta:.6f} "
                              f"NRMSE and nothing smaller"))
        emit(con)
        print(f"\n  PAIRED CONTRAST {gk} - softmax: mean={dm:+.6f} sd={dsd:.6f}")
        print(f"  RESOLUTION STATEMENT (no TOST at N={len(d)}): {con['statement']}")
        print(f"  achieved power against a {REFERENCE_EFFECT_IN_SD}*sd true effect "
              f"at N={len(d)}: {pw:.4f}")

    for gk in gated:
        def band(key, _gk=gk):
            v = [r[key] for r in rows if r["kind"] == _gk]
            m, h = statistics.fmean(v), tcrit * statistics.stdev(v) / math.sqrt(len(v))
            return m, m - h, m + h
        pm, plo, phi = band("sign_acc")
        gm, glo, ghi = band("gate_r2")
        p0, p0lo, p0hi = band("sign_acc_0step")
        g0, g0lo, g0hi = band("gate_r2_0step")
        lam, llo, lhi = band("lambda_hat_live")
        n_unit = sum(1 for r in rows if r["kind"] == gk and r["unit_root"])
        emit(dict(t="probe", device=dev, kind=gk,
                  sign_acc_mean=pm, sign_acc_ci=[plo, phi],
                  c_mean=2 * pm - 1, gate_r2_mean=gm, gate_r2_ci=[glo, ghi],
                  sign_acc_0step_mean=p0, sign_acc_0step_ci=[p0lo, p0hi],
                  gate_r2_0step_mean=g0, gate_r2_0step_ci=[g0lo, g0hi],
                  lambda_hat_live_mean=lam, lambda_hat_live_ci=[llo, lhi],
                  unit_root_seeds=n_unit, n_seeds=len(a.seeds),
                  loga_sst=rows[0].get("loga_sst")))
        print(f"\n  [{gk}] sign(a) PROBE off the arm's gate, out of sample: p={pm:.6f} "
              f"95%CI=[{plo:.6f},{phi:.6f}]  c=2p-1={2 * pm - 1:+.6f}")
        print(f"    0-step control (same seeds, untrained gate): p={p0:.6f} "
              f"95%CI=[{p0lo:.6f},{p0hi:.6f}]  trained gain={pm - p0:+.6f}")
        print(f"  gate-vs-a probe R^2 (VENUS's row): {gm:+.6f} "
              f"95%CI=[{glo:+.6f},{ghi:+.6f}]")
        print(f"    0-step control: R^2={g0:+.6f} 95%CI=[{g0lo:+.6f},{g0hi:+.6f}]  "
              f"trained gain={gm - g0:+.6f}")
        print(f"  lambda_hat on the surviving positions: {lam:+.6f} "
              f"95%CI=[{llo:+.6f},{lhi:+.6f}]  unit-root seeds: "
              f"{n_unit}/{len(a.seeds)}")
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
    #: PER GATED ARM, not hard-coded to `arm_pl`. An `arm_smprime` run against
    #: an `arm_pl`-only key printed `nan` and journalled it under a field named
    #: for an arm the run never constructed.
    secs_by_arm = {k: statistics.fmean([r["secs"] for r in rows
                                        if r["kind"] == k])
                   for k in a.arms if any(r["kind"] == k for r in rows)}
    #: NEPTUNE's OWN quantity, on the device it is defined on, beside the host
    #: one rather than instead of it. `vram_gate`/`r10_it8_pricing` both still
    #: assert this harness spends no VRAM (`V16_BAR_RECERT.md` F6, not this
    #: node's scope); journalling the figure is what lets that be checked.
    vram_gib = (torch.cuda.max_memory_allocated() / 2 ** 30
                if dev == "cuda" else None)
    emit(dict(t="wall", secs=wall, peak_wset_gib=peak_gib, device=dev,
              cuda_max_memory_allocated_gib=vram_gib, steps=a.steps,
              secs_per_run_by_arm=secs_by_arm))
    print(f"\n  peak process working set: {peak_gib:.3f} GiB "
          f"(NEPTUNE's 0.250 GiB is a CUDA allocator figure, not this quantity)")
    for k, v in secs_by_arm.items():
        print(f"  {k} mean seconds per {a.steps} steps at n={a.n_train}: {v:.2f} s "
              f"(NEPTUNE inherited ~18.19 s/150 steps for ARM PL, re-measured "
              f"12.2 s at 12 threads)")
    print(f"\n=== WALL CLOCK: {wall:.1f} s ===")
    print(f"WROTE {jl}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
