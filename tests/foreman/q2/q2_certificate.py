"""Q2 -- the first generalization-admissible row. Foreman phase, LAW L-REFLECTOR.

REUSED, not reimplemented:
  - tests/foreman/eval/r3_eval.py owns DocByteBatches (the by-document split),
    _corpus_text, assert_head_dim, print_repetition_row/repetition_factor,
    CORPUS_TOKENS, and train_with_eval() (the eval-bearing training loop).
    This file imports it as RE and calls RE.train_with_eval() unmodified.
  - tests/chase/gate/r1_gate.py owns magnitude_hardconcrete (FORMS["hard_concrete"])
    and build_repaired (the repaired gate init: m_head.bias=1-GATE_INIT_OFF,
    theta_head.bias=GATE_INIT_OFF). Imported as R, not reimplemented.

TWO ARMS, EQUAL PARAMETERS BY numel() (measured, not assumed -- printed below
in the fixed-structure block before either arm is scored):
  (a) softmax twin  -- build(operator="sgate") for the ordinary qkv/o_proj/mlp
      skeleton (ceq's config default, no gate heads), then CEQAttention.forward
      is monkeypatched, FOR THE DURATION OF THIS ARM ONLY, to genuine causal
      softmax attention (torch.nn.functional.scaled_dot_product_attention,
      is_causal=True) instead of ceq's own sgate_operator/path_sum. This is
      the only new code in this file that is not a straight import: the repo
      has no standalone softmax operator to import, by design (ceq/hf/
      modeling_ceq.py's own docstrings contrast every shipped operator
      AGAINST softmax), so "softmax twin" is built here as the same
      embedding/qkv/o_proj/mlp/lm_head skeleton with only the mixing rule
      swapped -- same parameter count as the skeleton minus the two gate
      heads arm (f) carries.
  (f) hard-concrete gate family -- build_repaired(operator="smprime") (the
      repaired init, R.build_repaired, reused) with arm_smprime.magnitude
      monkeypatched to R.FORMS["hard_concrete"] for the duration of this arm.

Both arms: same shape (hidden=128, n_layers=3, n_heads=8, d_head=16, seq=512,
vocab=256), same steps, same by-document split (split_seed FIXED at 0 across
all 5 seeds, so only model init / training draws vary with seed), same eval
harness. 5 SEEDS each, eval loss on the by-document split, written to
q2_certificate_results.jsonl as each seed lands (never held to the end).

ceq/ is not touched. r3_eval.py and r1_gate.py are not touched (imported
read-only).

Command: python q2_certificate.py [n_seeds]
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
RESULTS_PATH = os.path.join(SCRATCH, "q2_certificate_results.jsonl")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "foreman", "eval"))
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import r3_eval as RE  # noqa: E402 -- REUSED module, not edited
import r1_gate as R  # noqa: E402 -- REUSED module, not edited
import ceq.arm_smprime as arm_smprime  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

_ORIG_BUILD = RE.build          # ceq.hf.train.build, as bound in r3_eval's namespace
_ORIG_MAGNITUDE = arm_smprime.magnitude
_ORIG_ATTN_FORWARD = CEQAttention.forward


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


HIDDEN, LAYERS, HEADS, SEQ, BATCH = 128, 3, 8, 512, 8
VOCAB = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 0  # FIXED across all seeds: only model init/training draws vary
OUT_DIR_TMPL = os.path.join(SCRATCH, "q2_ckpt_{arm}_seed{seed}")


def softmax_forward(self, x, attention_mask=None):
    """Genuine causal softmax attention over the SAME qkv/o_proj this class
    already builds for operator='sgate' -- the only piece of new math in
    this file. attention_mask support is not needed (train_with_eval never
    passes one; DocByteBatches has no padding)."""
    if attention_mask is not None:
        raise NotImplementedError("softmax twin: no padding-mask path needed "
                                   "or exercised by this row")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def measure_params():
    """READ THE COUNT FROM numel(), for BOTH arms, before any arm is scored."""
    torch.manual_seed(0)
    m_a = _ORIG_BUILD(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                      seq=SEQ, vocab_size=VOCAB, operator="sgate")
    n_a = sum(p.numel() for p in m_a.parameters())
    del m_a
    torch.manual_seed(0)
    m_f = R.build_repaired(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                           seq=SEQ, vocab_size=VOCAB, operator="smprime")
    n_f = sum(p.numel() for p in m_f.parameters())
    del m_f
    return n_a, n_f


def print_fixed_structure(n_a, n_f, steps):
    rep_a = RE.repetition_factor(n_a)
    rep_f = RE.repetition_factor(n_f)
    rows = dict(
        law="L-REFLECTOR",
        table="Q2 generalization-admissible row",
        initializer="arm (a) softmax twin: CEQForCausalLM default init "
                     "(initializer_range=0.02), no gate heads exist. "
                     "arm (f) hard-concrete: R.build_repaired -- "
                     "m_head.bias={:.3f}, theta_head.bias={:.3f} "
                     "(GATE_INIT_OFF={}), SMPRIME_CORNER beta=qk=g=1.0".format(
                         1.0 - R.GATE_INIT_OFF, R.GATE_INIT_OFF, R.GATE_INIT_OFF),
        parameterization="hidden={} n_layers={} n_heads={} d_head={} seq={} "
                          "vocab={} batch={}".format(
                              HIDDEN, LAYERS, HEADS, HIDDEN // HEADS, SEQ, VOCAB, BATCH),
        n_params_arm_a_softmax_twin=n_a,
        n_params_arm_f_hard_concrete=n_f,
        n_params_diff="{} ({:.4f}%) -- the two m_head/theta_head gate Linears "
                       "arm (f) carries and arm (a) does not".format(
                           n_f - n_a, 100.0 * (n_f - n_a) / n_a),
        corpus_regime="data/tinystories_20k.txt, {:,} bytes==tokens (byte-level "
                       "vocab). by-document split (RE.DocByteBatches), "
                       "val_frac=0.1, split_seed={} FIXED across all seeds. "
                       "repetition_factor(a)={:.4f}x repetition_factor(f)={:.4f}x "
                       "-- both <1x, neither is the memorization regime this "
                       "project's earlier LM comparisons sat in".format(
                           RE.CORPUS_TOKENS, SPLIT_SEED, rep_a, rep_f),
        scorer_functional="arm (a): torch.nn.functional.scaled_dot_product_attention, "
                           "is_causal=True, ordinary softmax(QK^T/sqrt(d))V. "
                           "arm (f): ceq.arm_smprime.blend/path_product readout "
                           "with magnitude(u)=stretched-sigmoid-then-clamp "
                           "(zeta={}, gamma={}), repaired init".format(R.ZETA, R.GAMMA),
        bin_scheme="none (this row reports a single scalar final eval loss per "
                    "seed, not a binned/quantized score)",
        dtype_path="float32, device={}".format(DEVICE),
        torch_build="torch {} cuda_available={}".format(
            torch.__version__, torch.cuda.is_available()),
        eval_subsample_size="eval_batches x batch = {} x {} = {} windows per "
                             "eval call, drawn from the held-out val split with "
                             "a FIXED eval generator seed (r3_eval.py's own "
                             "20260921+split_seed), same subsample every arm/seed".format(
                                 EVAL_BATCHES, BATCH, EVAL_BATCHES * BATCH),
        steps=steps, lr=LR, seeds=None,  # filled by caller
    )
    return rows


LR = 3e-4
EVAL_BATCHES = 8


def train_arm_a(seed, steps):
    CEQAttention.forward = softmax_forward
    try:
        return RE.train_with_eval(
            out_dir=OUT_DIR_TMPL.format(arm="softmax", seed=seed), steps=steps,
            batch=BATCH, seq=SEQ, hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
            device=DEVICE, vocab_size=VOCAB, lr=LR, seed=seed, split_seed=SPLIT_SEED,
            eval_every=steps, eval_batches=EVAL_BATCHES, log_every=0,
            operator="sgate")
    finally:
        CEQAttention.forward = _ORIG_ATTN_FORWARD


def train_arm_f(seed, steps):
    RE.build = R.build_repaired
    arm_smprime.magnitude = R.FORMS["hard_concrete"]
    try:
        return RE.train_with_eval(
            out_dir=OUT_DIR_TMPL.format(arm="hardconcrete", seed=seed), steps=steps,
            batch=BATCH, seq=SEQ, hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
            device=DEVICE, vocab_size=VOCAB, lr=LR, seed=seed, split_seed=SPLIT_SEED,
            eval_every=steps, eval_batches=EVAL_BATCHES, log_every=0,
            operator="smprime")
    finally:
        RE.build = _ORIG_BUILD
        arm_smprime.magnitude = _ORIG_MAGNITUDE


def run_all(n_seeds: int):
    t0 = time.time()
    n_a, n_f = measure_params()
    steps = max(1, round(20.0 * n_a / (BATCH * SEQ)))  # Chinchilla-style budget, arm (a)'s count, SAME for both arms
    fixed = print_fixed_structure(n_a, n_f, steps)
    seeds = list(range(n_seeds))
    fixed["seeds"] = seeds
    print(json.dumps(fixed, indent=2), flush=True)
    board("q2_fixed_structure", **fixed)

    landed = {"softmax": [], "hard_concrete": []}
    total = 2 * len(seeds)
    done = 0
    for seed in seeds:
        for arm, fn in (("softmax", train_arm_a), ("hard_concrete", train_arm_f)):
            t1 = time.time()
            rec = fn(seed, steps)
            dt = time.time() - t1
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            row = dict(arm=arm, seed=seed, n_params=rec["n_params"],
                       final_eval_loss=rec["final_eval_loss"],
                       loss_last_train=rec["losses"][-1],
                       repetition_factor=rec["repetition_factor"],
                       n_docs=rec["n_docs"], n_train_docs=rec["n_train_docs"],
                       n_val_docs=rec["n_val_docs"], steps=steps, run_seconds=dt)
            result(stage="q2_arm_seed", **row)
            board("q2_seed_done", **row)
            landed[arm].append(rec["final_eval_loss"])
            done += 1
            print("[Q2] done {}/{}  arm={} seed={} final_eval_loss={:.4f} "
                  "({:.1f}s)".format(done, total, arm, seed,
                                      rec["final_eval_loss"], dt), flush=True)

    n_pairs = min(len(landed["softmax"]), len(landed["hard_concrete"]))
    diffs = [landed["hard_concrete"][i] - landed["softmax"][i] for i in range(n_pairs)]
    mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
    n_f_better = sum(1 for d in diffs if d < -0.01)
    n_tie = sum(1 for d in diffs if abs(d) <= 0.01)
    n_a_better = sum(1 for d in diffs if d > 0.01)
    verdict = dict(
        n_seeds_completed=n_pairs, n_seeds_requested=len(seeds),
        mean_diff_f_minus_a=mean_diff, per_seed_diff=diffs,
        n_tie_within_0p01=n_tie, n_f_better_by_gt_0p01=n_f_better,
        n_a_better_by_gt_0p01=n_a_better,
        prediction="tie within 0.01 nats (equal-and-use)" if n_pairs else "n/a",
        wall_clock_s=time.time() - t0,
    )
    result(stage="q2_verdict", **verdict)
    board("q2_all_done", **verdict)
    print("Q2 CERTIFICATE COMPLETE:", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_all(n)
