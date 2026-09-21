# kg_third -- third-domain transfer row (source code), Kaggle, 1h budget

## The question

TinyStories: hard-concrete gated arm (`ceq/arm_smprime.py`, `operator="smprime"`)
beat a matched softmax twin at 5/5 seeds, mean diff -0.2473 nats, 23x the 0.01-nat
tie band, at repetition 0.7977x/0.7985x (below one). A sibling lane is running the
second domain (WikiText-103, encyclopedic English) locally -- not duplicated here.

This row is the **third domain**: source code (`codeparrot/codeparrot-clean-valid`,
streamed from Kaggle with internet on, no credentials -- verified reachable in the
PREPARE step; fallback `vikp/python_code_instructions_18k` if codeparrot is
unreachable at kernel runtime, named explicitly if used). Long-range bracket/scope
dependencies, byte distribution unlike prose or encyclopedic English.

Pre-registered branches: **TRANSFERS** (f beats a by >0.01 nats at every seed),
**DOES NOT TRANSFER** (tie or a wins), **MIXED** (inside seed spread). No tuning.

## What this notebook is, and is not

Self-contained: stdlib + torch + `datasets` only. No `ceq/`, no `data/`, no
`results/`, no scratchpad payload uploaded. The arm's six load-bearing functions
(`magnitude`, `blend`, `gate`, `path_product`, `hop`, `numerator`, `operator`,
`readout`) are transcribed verbatim from `ceq/arm_smprime.py` as read on the
authoring machine 2026-09-22, and the model/attention shape mirrors
`ceq/hf/modeling_ceq.py` (`CEQAttention`/`CEQBlock`/`CEQForCausalLM`) and
`tests/chase/gate/r1_gate.py`'s hard-concrete magnitude
(`zeta=1.1, gamma=-0.1`) + gate init (`m_head.bias=1-1e-3, theta_head.bias=1e-3`)
exactly, so the third-domain row is testing the SAME two arms the first two rows
tested, not a simplified stand-in.

## Cell map (kg_third.ipynb)

0. intro / pre-registration
1. provenance: device/accelerator read from `torch`, never the metadata request;
   `log_row` appends one JSON line to `/kaggle/working/kg_third_results.jsonl`,
   never rewritten; `BUDGET_S = 55*60`, 5 min reserved for the tail.
2. **the rebuild gate** -- `REFERENCE` (transcribed from `ceq/arm_smprime.py`) vs
   `REBUILD` (what Cell 3's model actually calls), compared bitwise
   (`torch.equal`) at `(beta, qk) in {(0,0), (1,1), (1,0)}`, `g=1.0`. Then
   `REBUILD` is deliberately broken (one flipped sign inside `path_product`'s
   cumulative product) and the SAME comparison is asserted to FAIL. Verified
   locally (CPU/GPU smoke test, `_smoke_test.py`, not shipped) before this was
   trusted to run on Kaggle GPU time: honest rebuild PASSES all three corners,
   broken rebuild REJECTED at all three corners.
3. the two arms: `SoftmaxAttention` (genuine causal SDPA, qkv/o_proj, no gate
   heads) and `SMPrimeAttention` (Cell 2's proven `readout_fn`, hard-concrete
   magnitude, gate init as above). `CEQBlockInline`/`CEQForCausalLMInline`
   match `ceq/hf/modeling_ceq.py`'s block shape (pre-LN, MLP `d->4d->d` GELU,
   tied `lm_head`). Local smoke test at shrunk shape confirmed the param-count
   excess scales as `2*(d+1)+3` per layer, matching the pre-registration's
   783 = `3*(2*129+3)` at `d=128, n_layers=3`.
4. the corpus: codeparrot streamed and capped at `MAX_BYTES=20MiB` (TinyStories'
   own order of magnitude), split BY DOCUMENT (one row = one file),
   `val_frac=0.1, split_seed=0` -- byte-identical convention to
   `tests/foreman/eval/r3_eval.py::DocByteBatches` /
   `scratchpad/q2_second.py`. Prints `bytes_total`, `tokens`, licence, and
   `repetition_factor` for BOTH arms (labelled MEMORIZATION REGIME at >=1x;
   the bigger arm, (f), always reads a slightly higher repetition).
5. `train_with_eval` -- AdamW, grad-clip 1.0, fixed eval subsample,
   `del model; torch.cuda.empty_cache()` on return, peak CUDA memory logged
   per call (C27 hygiene).
6. calibration -- times 20 real training steps, then sizes the seed x step
   grid to the remaining 1h budget: drops seeds first, then steps, and prints
   exactly what it dropped. Verified in the local smoke test with an
   artificially tiny budget: correctly reduced seeds 5->1 and steps
   5380->3963, and correctly reported "NO COMPLETED PAIRS" when the budget
   ran out before a pair finished either arm.
7. the driver -- INCREMENTAL: each (arm, seed) result is written to the JSONL
   the instant it finishes; stops cleanly (not mid-run) if `budget_left() <
   60s`, logging which pairs were skipped.
8. the verdict -- rereads the incremental log, applies the pre-registered
   branch rule, prints the per-seed table beside the repetition factors and
   the accelerator actually used.

## Local verification before any Kaggle spend

`_smoke_test.py` (scratch-only, not shipped in the push) ran every code cell
on this machine at shrunk shapes (`hidden=32, layers=2, seq=32`) with a
synthetic corpus in place of the HF download:
- gate: PASS at all 3 corners on the honest rebuild, REJECTED at all 3 on
  the broken one.
- both arms train and eval without error; `float(loss.detach())` fix removed
  a `requires_grad` conversion warning that was present in the first draft.
- param-count excess scales correctly with the formula.
- calibration/driver/verdict cells correctly drop seeds/steps under an
  artificially tiny budget and correctly report zero completed pairs rather
  than crashing when the budget runs out mid-grid.
- one real bug caught and fixed here: the verdict cell's `log_row(...,
  accelerator=ACCEL, ...)` collided with `log_row`'s own automatic
  `accelerator` field (`TypeError: dict() got multiple values for keyword
  argument 'accelerator'`) -- removed the redundant explicit arg.

## Push / run

`kg_third_push/kernel-metadata.json`: `melowdramtic/kg-third-code-domain`,
GPU on, internet on, no dataset/kernel sources (dataset is pulled at runtime
via `datasets.load_dataset`, streamed, not attached as a Kaggle Dataset).

Board events (`C:\Users\seal\Desktop\New folder (32)\house-events.jsonl`,
agent "Cameron", append-only) mark: start, gate verified locally, push,
each poll, and done/collect-later.
