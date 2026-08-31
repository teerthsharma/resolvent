# CEQ v-main.4 — PHASE 2: TRAINING, KERNELS, AND THE OPTIMIZER

**Status: QUEUED. Runs after v-main.3M it.36.**
Registered 2026-08-30, round 10. Not started. This file is the spec of record;
the window fetches before implementing and the fetched definition is the spec.

Carries: L-VALUE, L-SCOPE, L-BIND, L-PRIOR, L-G2, L-FIRST, L-TOL.

## North Star (immutable)

> A consequence-understanding, path-understanding, causality-understanding
> attention architecture, as good as self-attention on a calibrated bar, that
> predicts the NEXT BEST ACTION toward equilibrium — not the next best token.
> The novelty is the WHOLE ARCHITECTURE; components may be pre-existing, always
> cited.

## Entry preconditions (measured, not assumed)

Phase 2 it.1 freezes a reference module spanning S1–S3[+S4]–S5. Three of those
stages are undecided until v-main.3M lands:

| gate | decided at | consequence for Phase 2 it.1 |
|---|---|---|
| S3 trained-selection ladder | v-main.3M it.30 | reference module's selection arm |
| S4 equilibrium readout | v-main.3M it.33 | `[+S4]` is present or absent |
| S5 head | v-main.3M it.35 | action decoder shape |

Phase 2 cannot begin before those three verdicts exist. Starting it earlier
freezes a reference to an undecided architecture, and every kernel in it.4–it.6
is bound to that reference.

## Where the two missing pieces now live

### Position code (the RoPE change, honest form)

- **arm P-ROPE** — standard rotary. Control.
- **arm P-PATH** — typed-transition transport `U_a` per action edge, score
  `<q_i, rho(path i->j) k_j>`.

P-PATH is implemented as a **cited component**: PaTH (arXiv 2505.16381) + LieRE
+ GRAPE named in ARCH.md, delta = composition only.

> **Note carried forward from the record.** A full round already died on the
> claim VGPE = PaTH + LieRE + GRAPE presented as component novelty. See
> `THEORY_V12_VGPE.md`. P-PATH re-enters here in the admissible form — the same
> object, demoted from "novel component" to "cited component, novel composition".
> Any Phase 2 sentence that reverts to the component claim is refused.

**RoPE-recovery bind:** chain graph + single type implies
`|score_PATH - score_ROPE| <= 1e-12` relative, derivation printed.

Every capability table carries BOTH arms. The architecture ships whichever wins
its CI, stated.

### Oracle wiring (one table, three columns, per task)

- **(a) LABELS** — `y` = exact solve. Dirichlet `u_I = (I - P_II)^-1 P_IB g`;
  action value `M r` with `M = (I - gamma P)^-1`.
- **(b) EVAL** — score = value comparison vs oracle output. Dual-oracle gate
  `|u_abs - u_kirchhoff|_inf <= 1e-10` per instance.
- **(c) TRAINING SIGNAL (new this phase)** — oracle distillation. Auxiliary loss
  `L_attr = ||omega_hat - omega||^2` against exact harmonic-measure attribution
  `omega` (row of `(I - P_II)^-1 P_IB`), weight `lambda_attr` swept.

The oracle is a TEACHER during training and a JUDGE at eval, and **its two roles
never share a tensor**. Leak gate: eval instances' oracles computed in a separate
process, hash-disjoint from training instances.

## New fetches

Named here; the window fetches before implementing. Fetch date cited in ARCH.md.
A fetch that contradicts this contract wins and is logged.

| id | subject |
|---|---|
| F24 | AdamW / Adam — the update equations |
| F25 | Preconditioner |
| F26 | Laplacian smoothing |
| F27 | Riemannian manifold (optimization on manifolds) |
| F28 | Stiefel manifold |
| F29 | Cayley transform (re-fetch for the optimizer use) |
| F30 | Matrix exponential |
| F31 | Tikhonov regularization |

## Topological AdamW (pre-registered before any run)

AdamW baseline:

```
m_t = b1*m + (1-b1)*g
v_t = b2*v + (1-b2)*g^2
theta <- theta - eta * m_hat/(sqrt(v_hat)+eps) - eta*lambda*theta
```

Three modifications, each an arm:

**O1 GRAPH-COUPLED SECOND MOMENT.** `v` is per-coordinate — diagonal, and
therefore topology-blind. Replace `v_hat` with the Laplacian-smoothed
`v_tilde = (I + beta*L_param)^-1 v_hat` over the pivot/coupling graph. This is
the proved resolvent shape, now used as preconditioning: parameters that share
graph structure share curvature estimates. `beta` swept. **Identity bind:**
`beta = 0` recovers AdamW exactly.

**O2 RIEMANNIAN STEP FOR TRANSITION GENERATORS.** `U_a` stays exactly orthogonal
by updating `Omega_a` in skew and mapping via Cayley `(I-Omega)^-1 (I+Omega)`.
The optimizer never leaves SO(n), so the norm-1 stability of the path code is a
theorem of the update, not a hope. `||U_a^T U_a - I||_F` printed per epoch, must
be `<= 1e-6` always. A plain-AdamW arm runs beside it to show the drift it
prevents.

**O3 DECAY-TO-STRUCTURE.** Weight decay for `Omega_a` pulls toward 0, i.e.
`U_a -> I`, the RoPE-null point. Shrinkage toward the recovery point rather than
toward the zero matrix. The prior IS the control arm. `lambda` swept, including 0.

**Prior-art fetch owed before any O-claim** (L-PRIOR, equation level): Shampoo /
K-FAC (structured preconditioners), the Riemannian Adam lineage, SAM. Claim shape
is "composition for this architecture", components cited.

**KILL.** An O-arm ships only if it beats plain AdamW at matched step budget AND
matched wall-clock, on BOTH loss-AUC and final metric, N=8, CIs excluding zero.
Otherwise it is a note, not a feature.

## The 40 iterations

| # | agent | action, statistic, deciding number |
|---|---|---|
| 1 | MERCURY | PyTorch reference module (S1–S3[+S4]–S5, both position arms) frozen as `ceq/reference.py`; every later kernel binds to it, never to itself |
| 2 | SATURN | parity harness — kernel vs reference abs-delta <= 1e-6 fp32 / <= 1e-3 bf16 elementwise, tolerance derivation printed (L-TOL); must-fire = planted sign flip in one tile is caught |
| 3 | NEPTUNE | cost model on paper: `FLOPs(path sum) = sum_h nnz(A^h) * d`, bytes moved, arithmetic intensity; target MFU >= 30% named per kernel BEFORE writing it |
| 4 | MERCURY | Triton kernel #1 — strictly-causal multi-hop path sum (`tl.dot` over tiles, hop loop unrolled to shipped K); parity bind runs before any benchmark |
| 5 | MERCURY | Triton kernel #2 — hopcache decode step (K slots, K rows/token); prefill/decode parity <= 1e-12 relative |
| 6 | MERCURY | Triton kernel #3 — fused settling step (if S4 shipped) or fused trained-selection top-k + readout (if not); dispatch count printed, target = 1 launch per step |
| 7 | NEPTUNE | benchmark per bolt-bench discipline — warmup, >= 30 samples, median + bootstrap CI, baseline = `torch.compile` reference; no speedup claim whose CI includes 1.0x |
| 8 | SATURN | autotune sweep (BLOCK sizes, `num_warps`, `num_stages`) journaled; best config hash enters the identity manifest — the tuned kernel IS a different measured object (L-BIND) |
| 9 | MARS | attack #1 — the parity suite's reachable set vs production shapes (L-SCOPE census: does parity ever see the real s, d, K?); repair or record |
| 10 | ALL | kernel verdict — parity AND MFU AND speedup-CI; table softmax-kernel-first; scoreboard line |
| 11 | JUPITER | fetch F24–F31; optimizer design note with the three O-update equations and their `beta=0` / `lambda=0` identity binds |
| 12 | SATURN | build O1 — `(I + beta*L)^-1 v_hat` via exact nilpotent/Cholesky solve at parameter-graph size; identity bind `beta=0` reproduces AdamW step BITWISE on integer-seeded fp64 probe (the one place bitwise is legal: same arithmetic path) |
| 13 | SATURN | build O2 — skew parametrization + Cayley; orthogonality census Frobenius-norm of `U^T U - I` per epoch, plain-AdamW drift arm beside it |
| 14 | SATURN | build O3 — decay target swap; `lambda` sweep grid declared |
| 15 | VENUS | files the optimizer prediction (which O-arms beat AdamW, with numbers) in its own commit, BEFORE any training run |
| 16 | MERCURY | optimizer ladder at small scale (the 4769-param shape) — AdamW / O1 / O2 / O3 / O1+O2+O3, N=8, matched steps AND matched wall-clock columns both printed |
| 17 | MARS | attack #2 — O-wins are lr-tuning artifacts; control is a full lr sweep on plain AdamW (tuning-asymmetry law: the baseline gets at least the arms' total tuning budget) |
| 18 | ALL | optimizer verdict by the KILL clause; surviving O-arms frozen into the training recipe; losers become notes |
| 19 | JUPITER | Chinchilla-style budget arithmetic for the 25.7M shape: tokens ~= 20 x params => ~514M tokens; `steps = tokens/(batch*s)`; wall-clock from measured throughput — the plan is a table, not a vibe |
| 20 | SATURN | resume bind on the REAL trainer — interrupt at step k, resume, bitwise-equal to uninterrupted at k+m (model, optim incl. O-state, RNG, dataloader position); must-fire = no-op'd `load_state_dict` breaks it |
| 21 | MERCURY | throughput probe, 500 steps, target GPU (T4/P100 both priced): measured tokens/sec, activation peak vs card bytes (C-C), session-chunk table at <= 11h against the 12h cap |
| 22 | SATURN | oracle-leak gate live — training/eval oracle processes hash-disjoint; a planted leak (eval label in a training shard) must be caught by the hash census |
| 23 | VENUS | files the training prediction — final token-bar and action-bar numbers per arm, BEFORE step 0 |
| 24 | MERCURY | TRAINING RUN A — best position arm + frozen recipe, `lambda_attr = 0` (no distillation), full budget, journals Merkle-rooted, PL-ratio `grad-L-squared / (L - L_min_hat)` tracked per epoch |
| 25 | MERCURY | TRAINING RUN B — identical except `lambda_attr` swept {0.01, 0.1, 1.0} on the shortest viable budget first; the winning lambda gets the full budget |
| 26 | MERCURY | TRAINING RUN C — the OTHER position arm at the winning recipe (the P-ROPE vs P-PATH deciding pair) |
| 27 | SATURN | mid-training audit — manifests on every checkpoint, orthogonality census (O2), `v_tilde` conditioning spread (O1: the 1.5e4x row-spread lineage is the known hazard, printed) |
| 28 | MARS | attack #3 — distillation teaches the oracle's quirks, not the task; control = eval on a HELD-OUT oracle variant (different solver route, same mathematical object — the Kirchhoff twin as the OOD judge) |
| 29 | MERCURY | the trained capability table — token bar AND action bar, softmax-trained-at-matched-everything FIRST, N=8 seeds (or the honest "n seeds, sign floor `2*2^-n`" header if compute caps n), CIs, resolution statements, manifest hashes per cell |
| 30 | ALL | THE DECIDING VERDICT — "as good" = token CIs overlap AND action CI excludes 0 our way; "supersedes" = action disjoint; adjudicated against Venus's it.23 filing; scoreboard +12 |
| 31 | MERCURY | ablation table at trained scale — position arm, O-arms off, `lambda_attr` off, S4 (if shipped) off — each one column, same draws |
| 32 | SATURN | probe battery through TRAINED checkpoints (the campaign's oldest open item): attribution `omega_hat` vs exact `omega` rank correlation with CI; Dirichlet-energy trace; selection entropy per layer |
| 33 | MARS | attack #4 — the fifteenth-class hunt on the trained artifacts (which control never sees a trained tensor?) |
| 34 | NEPTUNE | inference cost table — decode bytes/token vs softmax KV baseline, hopcache occupancy, measured not modeled |
| 35 | MERCURY | New York demo re-run on trained weights; the action head's score margin printed beside the founding-note figure |
| 36 | SATURN | HF package — real weights, limits-first card, COSTS inside, every cited test resolving, claim-resolver over the card; **upload ONLY on the author's explicit say-so** |
| 37 | JUPITER | the optimizer note or feature write-up with the Shampoo / K-FAC / Riemannian-Adam deltas at equation level |
| 38 | SATURN | MISTAKES.md phase entry; L-TIME labels; journals replay check at pinned thread count |
| 39 | VENUS | prediction-scorecard chapter — every filing vs every outcome, both halves scored, the round's forecasting record |
| 40 | ALL | prognosis — scoreboard, kills' dispositions, the one claim sentence the trained tables permit, next-X list. STOP. |

## Costing law, measured in round 10 iteration 3 — binding on it.19 and it.21

Iteration 19's Chinchilla arithmetic and iteration 21's throughput probe both price
work in wall-clock. Two measurements from Phase 0 constrain how, and both were paid
for already.

**A journalled `seconds` field is not a cost measurement.** `tests/chase/scale_axes.jsonl`
records `heads x=4` and `seq x=128` as bit-identical re-runs of one config —
`sgate_all` agrees to the last digit — at **786.28 s and 1072.54 s**. MERCURY
reproduced the spread deliberately: the same three configs, same harness, ~15 minutes
apart, with the GPU at 99–100% under eleven concurrent processes, drifted **+269%,
+356%, +304%**. The same work measured uncontended in the iteration-2 rho run spread
**9.6%–11.2%**. Wall-clock on this machine is a property of the workload *times the
number of seats holding the GPU*. Any estimate anchored on a journalled second
inherits whatever load happened to be running when it was recorded.

**A flat per-point unit hides every structural term.** Iteration 2's rho-axis estimate
was `12,391 s / 8 points = 1,548.875 s` exactly — no size term, no arm term, no seed
term. Measured: **777 s**, 15.95× over, decomposing exactly as markup 2.152× × anchor
6.413× × structure 2.727× ÷ size 2.360×. The size term was the only one running the
correct direction, and the model had none.

**The method that replaces both, and it is binding here:**

> Every estimate is a count of 600-step run-equivalents read off the source that will
> execute them, multiplied by a seconds-per-RE rate re-measured on the machine at the
> moment of spend — so the count is falsifiable before the run by reading the file,
> and the rate is falsifiable in ninety seconds by running one.

Two measured rates to anchor against: **20.58 s/RE uncontended**, **48.89 s/RE at four
concurrent seats**. Quote both, or quote the one that matches the intended dispatch.

**Two structural facts a parameter-count model is blind to**, both measured at equal
parameter count (3,319,296 at H=4, H=8, H=16): sgate goes 1.000 → 1.392 → 3.076 RE
while softmax goes 0.818 → 0.834 → 0.940; and with tokens/step pinned, seq 128 → 1024
takes sgate 1.000 → 5.525 RE against softmax 0.818 → 1.167. On the two axes a 300M run
multiplies hardest, the operator costs **3.3× (heads) and 4.7× (seq) its own control in
wall-clock at equal parameters**. `MODEL_CARD.md`'s shipped 1.32× is a seq-128, H-4,
3.3M number and does not transfer to it.19's budget.

**Idempotency is not free.** `tests/chase/axes.py::axis_rho` retrains both softmax
controls *outside* `_point`'s idempotency guard, so re-running an axis with an extended
grid retrains them unconditionally — **43% of any grid extension**. Check the guard's
scope before quoting a marginal cost; a "just one more point" estimate that omits the
control retrain is low by 1.8×–4.3×.

## Standing gates for this phase

- **No upload without explicit say-so** (it.36). The HF package is built, not published.
- **Kaggle / external compute** is an open option for it.24–it.26, to be raised only if
  the room measures the local card insufficient against C-C. Not pre-authorized.
- **L-TIME**: every timing quoted from a previous run is labelled INHERITED unless
  independently re-measured in the iteration that quotes it.

---

## Measured input for it.38, from v-main.3M round 10

Item 38 calls for a *"journals replay check at pinned thread count"*. Round 10
measured what that pinning is worth, so it.38 inherits the number rather than
re-deriving it.

**The harness is deterministic given `(seed, threads)` and NOT across thread
counts.** Three cross-thread pairs, same `(t*, n, steps, seed)`, threads 6 vs 8:

| block | drift |
|---|---|
| t\*=2 | **2.345e-03** |
| t\*=8 | 4.911e-04 |
| t\*=32 | 7.311e-06 |

Use the max. Three consequences for any replay:

1. **A replay at a different thread count cannot reproduce bitwise**, and must not
   be scored as though it could. L-VALUE's 5e-7 tolerance is **4,690×** tighter
   than the measured drift.
2. **Pin per ROW, not per run.** Round 10's own grid was swept at mixed counts
   (8, 6, 6, 12, 12) and only the per-cell `threads` field made that visible.
3. **Where a verdict's margin is smaller than the drift, the verdict is
   thread-count-dependent and must say so.** Worked case: `t*=32` at n=49152
   clears the bar by 7.00e-05 against a possible 2.345e-03 shift — 33× larger. The
   *crossing* survives because both compared cells ran at threads=12 and the offset
   cancels; the *absolute* reading does not.
