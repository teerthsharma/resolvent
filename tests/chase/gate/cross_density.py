"""Cross-density row: swap the confound between gate FORM and gate ZERO-DENSITY
AT INIT, to test the leap that the three-row puzzle's effect is density, not
parametrization.

REUSES tests/chase/gate/r1_gate.py wholesale -- imported, not reimplemented:
FORMS (clamp is ceq.arm_smprime.magnitude unmodified, hard_concrete is
r1_gate.magnitude_hardconcrete), gate_values (reads m through blend(), the
same place the operator itself reads it -- NOT off m_head.bias, which only
sets the pre-blend linear layer's offset and says nothing about what the
gate actually is once g and the lerp are applied), exact_zero_frac,
backward_reach_stats, spurious_zero_count, and ceq.hf.train (T) for the data
pipeline and the 800-step optimizer loop. ceq/ is never edited; only
`arm_smprime.magnitude` and `T.build` are monkeypatched, exactly as
r1_gate.py already does, and both are restored in a `finally`.

Stage order, per the task: span-length curve (no training) FIRST, then
verify the two cells' init densities off the SAME held-out batch the table
used, then -- gated on that verification -- train both cells for 800 steps.

Command: python cross_density.py
"""
import io
import json
import os
import sys
import time

import numpy as np

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "cross_density_results.jsonl")
GATE_DIR = os.path.join(REPO, "tests", "chase", "gate")
sys.path.insert(0, REPO)
sys.path.insert(0, GATE_DIR)

import r1_gate as G  # noqa: E402  -- the reused implementation, per the task
import torch  # noqa: E402
import ceq.arm_smprime as arm_smprime  # noqa: E402
import ceq.hf.train as T  # noqa: E402
from ceq.arm_smprime import GATE_INIT_OFF  # noqa: E402


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Chase",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


FORMS = {"clamp": arm_smprime.magnitude, "hard_concrete": G.magnitude_hardconcrete}

# The two cells of the cross, named exactly as the task specifies them.
CELLS = {
    "cell_a_clamp_scarce": dict(form="clamp", m_bias=3.0,
                                 predicted_density=0.0013, tol=0.01,
                                 predicted="reach>100, loss~1.139 (hard-concrete's own row)"),
    "cell_b_hardconcrete_dense": dict(form="hard_concrete", m_bias=-2.4,
                                       predicted_density=0.50, tol=0.08,
                                       predicted="reach collapses to 1, loss~1.27 (clamp's own row)"),
}
# The two ORIGINAL-table reference points, for the mechanism curve.
BASELINES = {
    "clamp_bias0": dict(form="clamp", m_bias=0.0),
    "hard_concrete_bias0": dict(form="hard_concrete", m_bias=0.0),
}

REFERENCE_TABLE = dict(
    clamp_high_bias=dict(last50_loss=1.2665, trained_exact_zero=0.3121, median_backward_reach=1),
    straight_through=dict(last50_loss=1.2999, trained_exact_zero=0.4050, median_backward_reach=1),
    hard_concrete=dict(last50_loss=1.1391, trained_exact_zero=0.0006, median_backward_reach=234),
)


def build_with_bias(m_bias: float, theta_bias: float = GATE_INIT_OFF):
    """Same shape/build path as r1_gate.build_repaired, but with the m_head
    bias the cross design needs instead of the fixed `1-off` the reference
    row used -- theta_head stays at the reference's `off`, so bias and form
    are the only two things this row ever varies."""
    def _build(*, hidden_size, n_layers, n_heads, seq, vocab_size=256, **overrides):
        model = G._ORIG_BUILD(hidden_size=hidden_size, n_layers=n_layers,
                               n_heads=n_heads, seq=seq, vocab_size=vocab_size,
                               **overrides)
        with torch.no_grad():
            for layer in model.model.layers:
                attn = layer.self_attn
                attn.m_head.bias.fill_(m_bias)
                attn.theta_head.bias.fill_(theta_bias)
        return model
    return _build


def eval_batch():
    """The SAME held-out batch r1_gate.run_variant reads: same corpus slice,
    same eval generator seed (12345), same shape."""
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    eval_gen = torch.Generator().manual_seed(12345)
    x_eval, _ = data.batch("val", G.BATCH, G.SEQ, eval_gen, G.DEVICE)
    return x_eval


def gate_at_init(name, form, m_bias, x_eval):
    """One forward pass at init: build, patch `arm_smprime.magnitude`, read
    m through `blend()` via r1_gate.gate_values (the operator's own read
    site), restore. No training."""
    arm_smprime.magnitude = FORMS[form]
    try:
        torch.manual_seed(G.SEED)
        model = build_with_bias(m_bias)(
            hidden_size=G.HIDDEN, n_layers=G.LAYERS, n_heads=G.HEADS,
            seq=G.SEQ, vocab_size=256, operator="smprime").to(G.DEVICE)
        m_init, u_init = G.gate_values(model, x_eval)
        ez = G.exact_zero_frac(m_init)
        del model
        if G.DEVICE == "cuda":
            torch.cuda.empty_cache()
        return m_init, ez
    finally:
        arm_smprime.magnitude = G._ORIG_MAGNITUDE


def span_zero_curve(m_layers, max_L=32):
    """For span length L in 1..max_L: the empirical fraction of length-L
    windows (j, j+L] that contain AT LEAST ONE exact zero, off the REAL
    per-token gate values (layer 0, same tensor r1_gate's own init reads
    use) -- not an i.i.d. assumption. flag+cumsum, same construction as
    r1_gate.spurious_zero_count's ground truth, specialised to fixed-length
    windows instead of all (i,j) pairs."""
    m = m_layers[0]  # [B, S]
    B, S = m.shape
    flag = (m == 0.0).to(torch.float64)
    cum = torch.cat([torch.zeros(B, 1, dtype=torch.float64), torch.cumsum(flag, dim=1)], dim=1)  # [B,S+1]
    curve = {}
    for L in range(1, min(max_L, S) + 1):
        starts = torch.arange(0, S - L + 1)
        ends = starts + L
        contains = (cum[:, ends] - cum[:, starts]) > 0.0  # [B, n_windows]
        curve[L] = float(contains.to(torch.float64).mean())
    return curve


def run_cell(cell_name, form, m_bias):
    """Train 800 steps at the reference config, reusing T.train / T.build's
    plumbing exactly as r1_gate.run_variant does -- only the build function
    (bias) and the patched magnitude (form) differ per cell."""
    board("start_variant", task="cross_density", cell=cell_name, form=form,
          m_bias=m_bias, shape=dict(hidden=G.HIDDEN, layers=G.LAYERS,
          heads=G.HEADS, seq=G.SEQ, batch=G.BATCH), device=G.DEVICE,
          seed=G.SEED, steps=G.STEPS)

    x_eval = eval_batch()
    m_init, ez_init = gate_at_init(cell_name, form, m_bias, x_eval)
    reach_init = G.backward_reach_stats(m_init)
    spur_init = G.spurious_zero_count(m_init[0][0])
    board("gate_at_init", cell=cell_name, form=form, m_bias=m_bias,
          exact_zero=ez_init, backward_reach=reach_init, spurious=spur_init)

    arm_smprime.magnitude = FORMS[form]
    out_dir = os.path.join(SCRATCH, "cross_density_ckpt_{}".format(cell_name))
    T.build = build_with_bias(m_bias)
    try:
        t0 = time.time()
        record = T.train(out_dir=out_dir, steps=G.STEPS, batch=G.BATCH, seq=G.SEQ,
                          hidden_size=G.HIDDEN, n_layers=G.LAYERS, n_heads=G.HEADS,
                          device=G.DEVICE, vocab_size=256, data_path=None,
                          max_bytes=64 * 1024 * 1024, lr=3e-4, seed=G.SEED,
                          grad_checkpoint=False, log_every=20,
                          save_every=G.SAVE_EVERY, operator="smprime")
        train_s = time.time() - t0
    finally:
        T.build = G._ORIG_BUILD
        arm_smprime.magnitude = G._ORIG_MAGNITUDE

    from ceq.hf.modeling_ceq import CEQForCausalLM
    model1 = CEQForCausalLM.from_pretrained(out_dir).to(G.DEVICE)
    arm_smprime.magnitude = FORMS[form]
    try:
        m_trained, u_trained = G.gate_values(model1, x_eval)
    finally:
        arm_smprime.magnitude = G._ORIG_MAGNITUDE
    ez_trained = G.exact_zero_frac(m_trained)
    reach_trained = G.backward_reach_stats(m_trained)
    spur_trained = G.spurious_zero_count(m_trained[0][0])

    losses = record["losses"]
    last50 = losses[-50:]
    last50_mean = float(np.mean(last50))
    last50_std = float(np.std(last50))

    board("gate_after_training", cell=cell_name, form=form,
          exact_zero=ez_trained, backward_reach=reach_trained, spurious=spur_trained,
          loss_last=losses[-1], last50_mean=last50_mean, last50_std=last50_std)

    out = dict(
        cell=cell_name, form=form, m_bias=m_bias, device=G.DEVICE, seed=G.SEED,
        steps=G.STEPS, command="python cross_density.py",
        exact_zero_init=ez_init, exact_zero_trained=ez_trained,
        backward_reach_init=reach_init, backward_reach_trained=reach_trained,
        spurious_zero_init=spur_init["n_spurious"], spurious_zero_trained=spur_trained["n_spurious"],
        train_seconds=train_s, loss_first=losses[0], loss_last=losses[-1],
        last50_mean=last50_mean, last50_std=last50_std,
        n_params=record["n_params"], out_dir=out_dir, loss_curve=losses,
    )
    return out


def main():
    t0 = time.time()
    board("start", task="cross_density", note="Cross design swapping form "
          "and zero-density: is the three-row effect parametrization or "
          "step-0 zero density?")

    # ---- STAGE 1: span-length curve, no training, the mechanism itself ----
    x_eval = eval_batch()
    curves = {}
    for name, cfg in {**BASELINES, **CELLS}.items():
        m_init, ez = gate_at_init(name, cfg["form"], cfg["m_bias"], x_eval)
        curve = span_zero_curve(m_init, max_L=32)
        curves[name] = dict(form=cfg["form"], m_bias=cfg["m_bias"],
                             exact_zero_init=ez, curve=curve)
        result(stage="span_curve", name=name, form=cfg["form"], m_bias=cfg["m_bias"],
               exact_zero_init=ez, curve=curve, seed=G.SEED,
               command="python cross_density.py")
        board("span_curve", name=name, form=cfg["form"], m_bias=cfg["m_bias"],
              exact_zero_init=ez, curve_L1=curve.get(1), curve_L2=curve.get(2),
              curve_L8=curve.get(8), curve_L32=curve.get(32))

    # ---- STAGE 2: verify the two cells' densities took, before training ----
    a = curves["cell_a_clamp_scarce"]
    b = curves["cell_b_hardconcrete_dense"]
    a_ok = abs(a["exact_zero_init"] - CELLS["cell_a_clamp_scarce"]["predicted_density"]) <= \
        CELLS["cell_a_clamp_scarce"]["tol"]
    b_ok = abs(b["exact_zero_init"] - CELLS["cell_b_hardconcrete_dense"]["predicted_density"]) <= \
        CELLS["cell_b_hardconcrete_dense"]["tol"]
    density_check = dict(
        cell_a_exact_zero_init=a["exact_zero_init"], cell_a_predicted=0.0013, cell_a_ok=a_ok,
        cell_b_exact_zero_init=b["exact_zero_init"], cell_b_predicted=0.50, cell_b_ok=b_ok,
        both_ok=a_ok and b_ok,
    )
    result(stage="density_verify", **density_check)
    board("density_verify", **density_check)
    if not (a_ok and b_ok):
        board("VOID", reason="cell densities did not take as predicted; the "
              "cross did not happen and nothing after it means anything",
              **density_check)
        print("DENSITY VERIFY FAILED -- stopping before training.", density_check)
        return

    # ---- STAGE 3: train both cells, 800 steps each, gated on stage 2 ----
    results = {}
    for cell_name, cfg in CELLS.items():
        res = run_cell(cell_name, cfg["form"], cfg["m_bias"])
        results[cell_name] = res
        result(stage="variant", **{k: v for k, v in res.items() if k != "loss_curve"})

    verdict_inputs = dict(
        cell_a=dict(last50_loss=results["cell_a_clamp_scarce"]["last50_mean"],
                    trained_exact_zero=results["cell_a_clamp_scarce"]["exact_zero_trained"],
                    median_backward_reach=results["cell_a_clamp_scarce"]["backward_reach_trained"]["median"],
                    max_backward_reach=results["cell_a_clamp_scarce"]["backward_reach_trained"]["max"]),
        cell_b=dict(last50_loss=results["cell_b_hardconcrete_dense"]["last50_mean"],
                    trained_exact_zero=results["cell_b_hardconcrete_dense"]["exact_zero_trained"],
                    median_backward_reach=results["cell_b_hardconcrete_dense"]["backward_reach_trained"]["median"],
                    max_backward_reach=results["cell_b_hardconcrete_dense"]["backward_reach_trained"]["max"]),
    )
    # Predictions from the task text: (a) reach>100 & loss~1.139 (hc's own row);
    # (b) reach collapses to 1 & loss~1.27 (clamp's own row).
    a_holds = (verdict_inputs["cell_a"]["median_backward_reach"] > 100 and
               abs(verdict_inputs["cell_a"]["last50_loss"] - 1.1391) < 0.10)
    b_holds = (verdict_inputs["cell_b"]["median_backward_reach"] <= 2 and
               abs(verdict_inputs["cell_b"]["last50_loss"] - 1.2665) < 0.10)
    if a_holds and b_holds:
        verdict = "BOTH_HOLD_decoy"
    elif not a_holds and b_holds:
        verdict = "A_COLLAPSES_drift_real"
    elif a_holds and not b_holds:
        verdict = "B_DOES_NOT_COLLAPSE_product_account_wrong"
    else:
        verdict = "NEITHER_HOLDS_inconclusive"

    final = dict(
        verdict=verdict, a_holds=a_holds, b_holds=b_holds,
        density_check=density_check, cross_row=verdict_inputs,
        reference_table=REFERENCE_TABLE,
        span_curves={k: v["curve"] for k, v in curves.items()},
        wall_clock_total_s=time.time() - t0,
    )
    result(stage="verdict", **final)
    board("done", **{k: v for k, v in final.items() if k not in ("span_curves",)})

    with open(os.path.join(SCRATCH, "cross_density_verdict.json"), "w") as fh:
        json.dump(final, fh, indent=2, default=str)
    print("VERDICT", json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
