"""Q3 -- the gate sweep table. Foreman phase, LAW L-REFLECTOR.

REUSED, not reimplemented: tests/chase/gate/r1_gate.py owns FORMS
(straight_through, hard_concrete), run_variant(), backward_reach_stats(),
exact_zero_frac(), gradient_frac_nonzero(), board(). This file imports that
module and:
  (1) adds "clamp" as a third form -- arm_smprime.magnitude UNPATCHED, which
      r1_gate.py already captures as _ORIG_MAGNITUDE before any monkeypatch,
      so no new gate math is written here, only a dict entry.
  (2) drives run_variant() across 5 SEEDS per form instead of r1_gate.py's
      single hardcoded SEED=0, by setting the module-global R.SEED and a
      per-(form,seed) R.OUT_DIR_TMPL before each call -- run_variant() reads
      both as globals at call time, so this needs no edit to r1_gate.py.
  (3) writes one board event and one results-jsonl line PER SEED AS IT LANDS
      (never held to the end), plus q3_gatesweep_results.jsonl / .md.

ceq/ is not touched. tests/chase/gate/r1_gate.py is not touched (imported
read-only; its own __main__ block does not run because this file drives
run_variant() directly instead of calling r1_gate.main()).

Command: python q3_gatesweep.py [n_seeds]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "q3_gatesweep_results.jsonl")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))

import torch  # noqa: E402

import r1_gate as R  # noqa: E402  -- REUSED module, not edited


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


FORMS_ORDER = ["clamp", "straight_through", "hard_concrete"]
R.FORMS["clamp"] = R._ORIG_MAGNITUDE  # (1) the third form, reused unpatched
R.board = board  # r1_gate.run_variant's internal board() calls hardcode
                 # agent="Chase"; this is Foreman's phase, so every board
                 # write -- including the ones inside the reused function --
                 # must carry agent="Foreman". Same technique as every other
                 # monkeypatch in this file: the module-global name is
                 # replaced, run_variant() still calls "board(...)" and now
                 # resolves it to this one.


def print_fixed_structure():
    """LAW L-REFLECTOR: every named row printed BEFORE any arm is scored."""
    dev = R.DEVICE
    rows = dict(
        law="L-REFLECTOR",
        table="Q3 gate sweep",
        initializer="build_repaired: m_head.bias=1-GATE_INIT_OFF, "
                     "theta_head.bias=GATE_INIT_OFF (GATE_INIT_OFF={})".format(
                         R.GATE_INIT_OFF),
        parameterization="hidden={} n_layers={} n_heads={} d_head={} seq={} "
                          "batch={} operator=smprime".format(
                              R.HIDDEN, R.LAYERS, R.HEADS, R.HIDDEN // R.HEADS,
                              R.SEQ, R.BATCH),
        corpus_regime="ceq.hf.train.ByteBatches, byte-OFFSET 90/10 split (r1_gate.py's "
                       "own split, unchanged), data/tinystories_20k.txt first 64MiB",
        scorer_functional="arm_smprime.blend/path_product readout; magnitude(u) "
                           "varies by form: clamp=torch.clamp(u,0,1) (grad zero "
                           "outside (0,1)); straight_through=forward clamp, "
                           "backward identity; hard_concrete=stretched sigmoid "
                           "(zeta={}, gamma={}) then clamp".format(R.ZETA, R.GAMMA),
        bin_scheme="none (this table has no histogram/quantile binning of scores; "
                    "backward_reach_stats reports mean/median/p95/max over the raw "
                    "per-position consecutive-nonzero run length)",
        dtype_path="float32 (train), float64 (F1/spurious-zero self-tests only)",
        torch_build="torch {} cuda_available={} device={}".format(
            torch.__version__, torch.cuda.is_available(), dev),
        eval_subsample_size="no held-out eval in this row (Q3 reports TRAIN loss "
                             "and gate statistics only, per the task's own field "
                             "list); gate statistics are read off one fixed batch "
                             "of shape [{},{}] drawn with eval_gen seed 12345".format(
                                 R.BATCH, R.SEQ),
        steps=R.STEPS, seeds=None,  # filled by caller
    )
    return rows


def run_all(n_seeds: int):
    t0 = time.time()
    seeds = list(range(n_seeds))
    fixed = print_fixed_structure()
    fixed["seeds"] = seeds
    print(json.dumps(fixed, indent=2), flush=True)
    board("q3_fixed_structure", **fixed)

    # gate-1..3 checks from r1_gate.py, run ONCE (they are form/seed-independent):
    f1 = R.reproduce_f1()
    board("q3_f1", checks=f1["checks"], all_pass=f1["all_pass"])
    if not f1["all_pass"]:
        board("VOID", reason="F1 did not reproduce", checks=f1["checks"])
        print("F1 FAILED -- stopping.", f1["checks"]); return
    ver = R.verify_ste_forward()
    board("q3_ste_forward_equality", **ver)
    if not ver["bitwise_equal"]:
        board("VOID", reason="STE forward != clamp bitwise")
        print("STE MISMATCH -- stopping."); return
    selftest = R.spurious_zero_selftest()
    board("q3_spurious_zero_selftest", **selftest)
    if not selftest["self_test_pass"]:
        board("VOID", reason="spurious-zero self-test failed")
        print("SELFTEST FAILED -- stopping."); return

    done = 0
    for form in FORMS_ORDER:
        for seed in seeds:
            R.SEED = seed
            R.OUT_DIR_TMPL = os.path.join(SCRATCH, "q3_ckpt_seed{}_{{form}}".format(seed))
            t1 = time.time()
            res = R.run_variant(form)
            res["seed"] = seed
            res.pop("loss_curve", None)
            dt = time.time() - t1
            res["run_seconds"] = dt
            result(stage="q3_variant", **res)  # res already carries form= and seed=
            board("q3_seed_done", form=form, seed=seed,
                  loss_last=res["loss_last"], last50_mean=res["last50_mean"],
                  exact_zero_trained=res["exact_zero_trained"],
                  backward_reach_trained=res["backward_reach_trained"],
                  frac_grad_nonzero_trained=res["frac_grad_nonzero_trained"],
                  run_seconds=dt)
            done += 1
            print("[Q3] done {}/{}  form={} seed={} loss_last={:.4f} "
                  "exact_zero={:.4f} reach_mean={:.3f} live_grad={:.4f} "
                  "({:.1f}s)".format(done, len(FORMS_ORDER) * len(seeds), form, seed,
                                      res["loss_last"], res["exact_zero_trained"],
                                      res["backward_reach_trained"]["mean"],
                                      res["frac_grad_nonzero_trained"], dt), flush=True)

    board("q3_all_done", n_forms=len(FORMS_ORDER), n_seeds_requested=len(seeds),
          n_seeds_completed=done // len(FORMS_ORDER) if FORMS_ORDER else 0,
          wall_clock_s=time.time() - t0)
    print("Q3 SWEEP COMPLETE in {:.1f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_all(n)
