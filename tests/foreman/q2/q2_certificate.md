# Q2 -- the first generalization-admissible row

LAW L-REFLECTOR: fixed structure printed before either arm is scored (see
the `q2_fixed_structure` board event and the JSON block `q2_certificate.py`
prints at the top of its own stdout).

## Reuse, not reimplementation

- `tests/foreman/eval/r3_eval.py` (module `RE`): `DocByteBatches` (the
  by-document split), `_corpus_text`, `assert_head_dim`, `repetition_factor`,
  `CORPUS_TOKENS`, and `train_with_eval()` -- the whole eval-bearing training
  loop, called unmodified for both arms.
- `tests/chase/gate/r1_gate.py` (module `R`): `build_repaired` (the repaired
  gate init) and `FORMS["hard_concrete"]` (the stretched-sigmoid magnitude).

## The one piece of new math: the softmax twin

The repository has no standalone softmax operator to import -- every shipped
operator (`sgate`, `signed`, `smprime`) is explicitly contrasted against
softmax in its own docstring, by design. "Softmax twin" is therefore built
here, not imported: `build(operator="sgate")` gives the ordinary
qkv/o_proj/mlp/embedding/lm_head skeleton (no gate heads -- that only
happens for `operator="smprime"`), and `CEQAttention.forward` is
monkeypatched, for the duration of arm (a)'s training only, to
`torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` --
genuine causal softmax attention over the same `qkv`/`o_proj`. The class
method is restored in a `finally:` block whether or not training raises.

Arm (f) is `R.build_repaired(operator="smprime")` with
`arm_smprime.magnitude` monkeypatched to `R.FORMS["hard_concrete"]`, same
technique as Q3.

## Equal parameters, measured not assumed

`measure_params()` builds both arms once before either is trained and
prints `numel()` for each -- see `q2_fixed_structure`'s
`n_params_arm_a_softmax_twin` / `n_params_arm_f_hard_concrete` /
`n_params_diff` fields. The two arms are NOT forced to exact equality by
padding the shape; the difference (the two tiny `m_head`/`theta_head`
gate `Linear(d,1)` layers arm (f) carries and arm (a) does not) is reported
as a percentage rather than hidden, per the same "print it, do not force
it" discipline the task's own SHAPE section uses for the 720,896-vs-632,496
count.

## Sizing

`steps = round(20 * n_params_arm_a / (batch * seq))` -- the Chinchilla-style
token budget, computed from the measured `numel()` of arm (a) and applied
identically to both arms (same steps for a fair comparison). `batch=8`
(not 32): a 32-batch smoke test at this shape triggered a WDDM host spill
on arm (f) (`peak_bytes > device total`, `check_host_spill` in
`r3_eval.py` — reused, not reimplemented) even at 5 steps; batch=8 (the
same batch r1_gate.py/gate_init_repair.py already use) stays at ~2.8GB
peak on this 8.6GB card with no spill.

`split_seed=0` FIXED across all 5 seeds -- only model init / training
draws vary with `seed`; the document split itself does not, so the 5-seed
spread reported below is training-stochasticity, not data-split
stochasticity.

## Prediction (pre-registered, per the task text)

(a) and (f) tie within 0.01 nats -- equal-and-use, the expected outcome.
Counter A: (f) worse by >0.01 -- the gate costs generalization, reported
not tuned away. Counter B: (f) better by >0.01 at 5 of 5 seeds -- the first
learned win this project would have, and per the task's own instruction it
is NOT written up as a claim until verified on a second corpus.

## L-REPRO

Command: `python q2_certificate.py 5` (argv[1] = n_seeds, default 5).
**Seeds actually run: 5 of 5 requested, both arms -- 10/10 runs
completed.** Ran cleanly after the (independent) `q3_gatesweep.py` bug was
fixed; no retries needed on this file. Device: CUDA (RTX 4060 Laptop GPU,
8.6GB), torch 2.14.0+cu126. Wall clock: 2931.2s for all 10 runs (softmax
~74s/run, hard_concrete ~512s/run -- the gated arm's complex-valued
path-product readout is the cost, not the shape).

n_docs=105,095 (n_train_docs=94,585, n_val_docs=10,510) at split_seed=0,
matching this project's own by-document-split reference numbers exactly.

## Results (5 seeds each; steps=3538, per the sizing above)

| arm                    | n_params | final_eval_loss (5 seeds)                          | mean    |
|-------------------------|---------:|------------------------------------------------------|--------:|
| (a) softmax twin         | 724,608  | 1.2757, 1.2775, 1.3021, 1.3013, 1.2916               | 1.2896  |
| (f) hard-concrete gates  | 725,391  | 1.0431, 1.0424, 1.0430, 1.0392, 1.0439               | 1.0423  |

Per-seed diff (f minus a): -0.2326, -0.2351, -0.2591, -0.2622, -0.2476
(mean -0.2473).

**Verdict against the pre-registered prediction**: NOT a tie -- (f) beats
(a) by more than 0.01 nats at **5 of 5** seeds, well outside the tie band
(the smallest per-seed gap is -0.2326, 23x the 0.01 threshold). Per the
task's own COUNTER clause for this exact outcome ("(f) better by more than
0.01 at 5 of 5 seeds -- the FIRST LEARNED WIN this project would have, and
it is NOT written until verified on a second corpus"): **this is reported
as a measurement, not written up as a win.** The numbers stand; the claim
does not, until a second corpus is run. That second-corpus run is outside
this row's scope (SHAPE and CORPUS were both fixed by the task) and is not
attempted here.

One honest caveat on the comparison itself, stated once, here: arm (f)'s
higher parameter count (725,391 vs 724,608, +0.11%, the two gate-head
Linears) means the two arms are close but not exactly equal by numel() --
see the fixed-structure block above for the measured counts. A 0.11%
parameter gap does not plausibly explain a 0.23-0.26 nat loss gap.
