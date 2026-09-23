# CHASE — Phase K, it.K1

**Verdict.** The (f_R) hook is bound and in use.
- The resolvent layer trains inside train_ladder at R0 with exact fp32 gradients, deterministically under `--deterministic`. READY was written at 04:29.
- Clause 6 holds on every R0 (f_R) checkpoint written so far: 3 checkpoints, 4 (checkpoint, S) cells. This is under the Dispatcher's v2 line.
- The original 1e-5 read line sits below the fp32 floor on 3 of the 4 cells.
- Clause 7 (BOS < 0.5) holds, with a maximum share of 0.187.
- R-RANGE is not read. The only synthetic (f_R) checkpoint did not learn its bed (acc 0.2026 at n 1024). Its "infinite range" comes from diffuse heads: the range bars as first written pass on it by construction, and the premise-gated replacement is RED with 0 qualifying checkpoints.
- On the chain bed, the (a_L) twin's heads do not have range ≈ 1/slope: 36 of 84 pilot heads fall outside a factor of 2. On LM data they all do (16 of 16).

SP = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad`.

Lane: `SP/phase_k/K1/chase/`. Setup: torch 2.14.0+cu126, RTX 4060 Laptop 8 GB, TF32 off in the bar files.

Nurses: 3.
- A haiku nurse surveyed the lanes' trainers and checkpoint formats.
- A haiku nurse audited this report: 43/43 numbers match the logs, every sha256 file passes, and the a_L count recomputes to 36/84 (7/8/6/5/8/2) and 0/16.
- A sonnet nurse did a read-only bug review of the hook: 8 items, dispositions below.

## The hook (`resolvent_hook.py`, sha256 3855288d…)

- **`FR`.** A factory with the harness signature. Layers 0..L−2 are the K0 train_ladder `AlibiAttention`, unchanged; layer L−1 is `ResolventAttention`.
  - The resolvent is placed last for two reasons. First, its q/k/v are then built by the ALiBi layers: a root must learn to point at itself (the absorber) and V must carry the root id, and both need the content mixed first. Second, no later attention can re-mix its read, so depth past the stack's 2^(L−1) is attributable to it.
- **`ResolventAttention`.** q/k/v/out projections without bias; per-head s_i = a_h ln(i+1) + b_h, with a = 1, b = 0 at init, learned and without weight decay. W = causal softmax(s_i q_i·k_j/√D) at β = 1, and x = (1−γ)(I−γW)⁻¹V at γ = 0.999.
- **Paths.**
  - Forward: fs5c, from K0.
  - Backward: K0's `backward_blocked` with x − mean(x) in dP (see Kills).
  - Both are CUDA-graphed, with keys (device, shape, γ, c).
- **Precision.** The whole layer runs in fp32, or float64 when the weights are float64, with autocast off. The solve's Function is also `custom_fwd`/`custom_bwd`.
- **Length.** S is zero-padded to a multiple of 256.
- **Rules given in READY.** No `--compile`. Bitwise comparisons need `--deterministic`.

## Bars

Every bar was run RED first (stub hook, or `CHASE_STUB=1`), and all RED lines are on the board. No test file was edited after its RED: SHA files `run4.sha256`, `ckpt_bars*.sha256`, `clause6_v2.sha256`, `bf16b.sha256`, `x64ref.sha256`, `rrange_v3.sha256`.

| bar (chase.k1.*) | RED line | measured | status |
|---|---|---|---|
| hook_parity_S1024 (R0 layer vs K0 float64; 3 inputs × autocast on/off; y ≤ 1e-5, grads ≤ 1e-4) | stub: y 2.43, da inf (`red_stub.log`); then real hook 9d0274…: da 7.5e-4, db 6.9e-4 (`run3_hook9d0274.log`) | y ≤ 3.69e-6; dx, dw ≤ 1.5e-6; da ≤ 5.21e-5; db ≤ 5.11e-5. Autocast on = off bitwise (`run4.log`) | GREEN |
| hook_gradcheck_S64_f64 (+ 3 mutants must fail) | stub: mutants pass | passes; dq/dk/dv ×1.01 all fail | GREEN |
| hook_train_ladder_R0_50 | stub: a, b do not move | 50/50 finite; mean loss 10.807 → 9.863 (first 10 → last 10 steps); eval 9.631; a moved 0.0095 | GREEN |
| hook_deterministic_R0_50 | stub: a, b do not move | 50/50 losses and state_dict bitwise equal | GREEN |
| bf16_leak_fused_read1 (leak read of 1 ≤ 1e-5 at S 1024 and 4096) | stub | 4.95e-6 / **1.39e-5** | **RED** (kept) |
| bf16_leak_read1_vs_fp32 (leak ≤ 2 × fp32 hook + 1e-6, and < 1e-3) | stub | leak 4.95e-6 / 1.39e-5 vs fp32 4.59e-6 / 1.41e-5 | GREEN |
| bf16_leak_fused_pattern (leak rel err > 1e-3) | stub | 1.10e-2 / 7.92e-3 | GREEN |
| bf16_inputs_fused_raises | stub | `triangular_solve_cuda` not implemented for BFloat16 | GREEN |
| rrange_band_y2 (band mass reproduces Y2 to 1e-4) | stub | rel ≤ 1.9e-13 | GREEN |
| rrange_band_sink (BOS sink σ = 0.2, γ′ = γ(1−σ)) | stub | rel ≤ 1.3e-15 | GREEN |
| rrange_tail_sink_fooled (K0 tail mass < 0.5 m′ in ≥ 1 cell) | stub | m_tail / m′ = 3e-10 … 4.4e-6 in 6/6 cells | GREEN |
| c6_numerics_R0_v2 (original line, read of 1 ≤ 1e-5) | stub, and RED with no checkpoints | 1.75e-5, 9.27e-5, 1.10e-5, 8.9e-6 | **RED**: 3/4 cells. The line sits below the fp32 floor: floors 1.05e-5, 6.75e-5, 1.34e-5 |
| **clause6_v2** (γ·max rowsum < 1, and read of 1 ≤ 2 × floor) | stub | γ·max rowsum 0.99900036 in all cells; ratios 1.67, 1.37, 0.82, 0.62 | GREEN |
| clause6_v2_power (fs5 > 2 × floor in ≥ 1 cell) | stub | 4/4 cells, fs5 at 12–439 × floor | GREEN |
| c6_power_R0_v2 (bf16-rounded P gives γ·rowsum ≥ 1; fs5 read > 1e-5) | stub | 4/4 cells and 4/4 cells; max γ·rowsum(bf16 P) 1.0018 | GREEN |
| x64_blocked_matches_dense (floor's reference) | stub | 2.1e-15 | GREEN |
| c7_bos_R0_v2 (every head < 0.5) | stub | max 0.187 (far_fR_s0 head 0, S 16384); LM heads ≤ 0.029 | GREEN |
| c7_route_gamma_h_v2 | stub | not run (c7 is GREEN) | SKIP |
| rrange_fR_depth_R0_v2 / rrange_fR_span_R0_v2 | stub | range inf / inf against depth 2075 and span 16,378 | GREEN, **vacuous** (see Kills) |
| **rrange_fR_dressed_R0** (premise acc ≥ 0.8; bare < depth ≤ dressed) | stub | 0 checkpoints meet the premise (far_fR_s0 acc 0.2026; bare inf / 18,247, dressed inf / inf) | **RED** |
| rrange_aL_slope_R0_v2 (every a_L head within 2× of 1/slope) | stub | Wilson a_L_s1 and s2: 16/16 in band. Cameron's six aL7 pilots: 36/84 out of band | **RED** |

The v1 checkpoint bars (`test_chase_k1_ckpt.py`: c6/c7/rrange without `_v2`) stay registered and were not run. Cameron moved R-DEPTH to bed_k′ and `rdepth.py` no longer exports the bed functions v1 calls. The `_v2` bars are the same text, read on each run's own evaluation beds.

Checkpoints read (`ckpt_rows_v2.jsonl`, sha ffe82184…; `run_ckpt.log`):

| checkpoint | sha | a / b | cells |
|---|---|---|---|
| Cameron far_fR_s0 | 9b63c827… | a 1.484 / 1.742, b 0.694 / 0.909 | S 1024 and S 16384 |
| Wilson f_R_s1 | 4fa90e60… | a 1.284 / 1.354, b 0.420 / 0.403 | S 1024 |
| Wilson f_R_s2 | a4732959… | a 1.353 / 1.353, b 0.454 / 0.486 | S 1024 |

## Kills and replacements

| killed | measured reason | replacement | state |
|---|---|---|---|
| K0 `backward_blocked` as the γ = 0.999 training backward | da/db 7.5e-4 (bar 1e-4). u is exact to 2.8e-7 but ~2e4 × gx, and g·u·xᵀ − rowsum cancels in fp32 when the reads x_j are nearly equal (`diag_dq.log`) | dP formed from x − mean_S(x). This is exact, because the softmax Jacobian ignores a shift of x shared by all keys. dq error 2.7e-4 → 1.4e-5 | GREEN (hook_parity) |
| Clause 6's 1e-5 read line as a pass/fail on trained checkpoints | fp32's own floor on the same inputs is 1.05e-5 – 6.75e-5 | clause6_v2: 2 × the measured floor (Dispatcher), with a power bar | GREEN |
| The K0 struck bf16 row-sum failure as a risk to the fused path | the leak's read of 1 equals fp32's (4.95e-6 / 1.39e-5 vs 4.59e-6 / 1.41e-5): one shared normalizer | Keep the fp32 exemption for the pattern error (1.1e-2), not for row sums. bf16 inputs cannot run silently (they raise) | GREEN |
| K0 tail-mass instrument on heads with a sink | a sink reads as range: m_tail / m′ ≤ 4.4e-6 | band mass (window sums, position 0 never inside a band, B-weighted fit) | GREEN on Y2 and sink |
| R-RANGE v2 prediction as evidence | a diffuse, untrained head passes (bare range inf / 18,247) | rrange_fR_dressed_R0: an L-TRAINED premise plus bare < depth ≤ dressed | RED: no premise checkpoint yet |
| (a_L) "range ≈ 1/slope" on the chain bed | 36/84 pilot heads out of band; content sets their range | keep it as the LM-twin statement (16/16); on the chain bed, measure the twin's range, don't assume it | registered, measured |

Nurse review dispositions:
- Item 8, `--compile`: forbidden in READY.
- Items 5 and 6: the graph key now carries the device.
- Nit 1: the layer is forced to fp32.
- Padding, init, eval/accum aliasing and FR state_dict keys: verified correct by the nurse.

## Unbarred fields (not findings)

- **Resolvent heads look diffuse.** Every head's dressed band mass grows with distance in the window (res_m < 0 on all 8 head-cells). The LM heads' bare ranges are 1155 / inf (s1) and 516 / inf (s2) tokens at S 1024.
- **bf16-rounded P rows with γ·rowsum ≥ 1:** 295/2048, 3263/32768, 647/4096 and 649/4096.
- **Struck by the Inspector (PASS4):** my forecast "Cameron's 16k checkpoints may go RED on the floor, not a bug". clause6_v2 decided it.
- **Dry run.** `dry_aL.log` / `dry_aL_rows.jsonl` came from a pre-run of rrange_aL_slope_R0_v2 with board posting off (plumbing check). Its numbers equal those of the posted run.

## OPEN

- **R-RANGE is unread.** No (f_R) checkpoint has learned its bed yet (far_fR_s0: acc 0.2026 at n 1024, loss 7.41). far_fR_s1 is fifth in Cameron's runner and due about 08:00. Default next step when its result.json exists: `python test_chase_k1_clause6_v2.py`, then `python test_chase_k1_rrange_v3.py`, both from SP/phase_k/K1/chase, under the lock and its 3-minute gap.
- **R-BIND** (absorber count vs planted roots) is not built. It needs a trained checkpoint first.
- **γ_h route.** It is registered and never triggered. The per-head γ_h hook for training is not built.
- **Floor growth with length.** The fp32 floor rises with S (1.05e-5 at 1024 → 6.75e-5 at 16384 on far_fR_s0). The per-row-centred fused pushes (K0 OPEN) have not been tried.
- **Cameron's (a_L) far runs** (far_aL4_s0, far_aL7_s0) finish after this read. rrange_aL_slope_R0_v2 counted only his pilots.
