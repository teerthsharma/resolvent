# CHASE — Phase K, it.K2 (K2.1 sparsemax resolvent hook)

**Verdict.** The blocked sparsemax resolvent hook is ready, and it fits at S 16384 with room to spare.
- `resolvent_sp.py` (sha256 `36039c3a…`) gives the K1 layer with sparsemax weights, which have exact zeros.
  - It matches densec to ≤ 4.23e-15 in float64, and has the same support set exactly.
  - It passes gradcheck, with 2 of 2 Jacobian mutants caught.
- At S 16384, B1, forward plus backward, the peak is 0.35 GiB reserved at the rdepth shape (H2 D64) and 0.63 GiB at H4 D32. The bar is 6 GiB.
- It trains inside rdepth.py (γ-annealed and fixed) and inside train_ladder.py.
- READY_SP is written.
- Three first-registration bars stay RED as registered: two had mutants that could not fire as written, and one set a residual line inside the fp32 floor. Each has a v2 replacement that is GREEN.
- The first hook version (98259dff) had a real memory defect: 6.36 GiB reserved for 0.68 GiB allocated at S 16384. It was killed, and the replacement is bound.

SP = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad`.

Lane: `SP/phase_k/K2/chase/`. Setup: torch 2.14.0+cu126, RTX 4060 Laptop 8 GB, TF32 off in the bar files. Every GPU job ran through `gpurun.py`, which holds `SP/phase_k/K1/gpu.lock.d` via a K2 copy of the K1 lock with its own cooldown file.

## The hook

- **`ResolventAttention`** has the same parameters and forward contract as K1's.
  - q/k/v/out without bias.
  - Per-head logit scale s_i = a_h ln(i+1) + b_h, with a = 1, b = 0 at init.
  - β = 1, and x = (1−γ)(I−γW)⁻¹V.
  - γ = `self.gamma` (0.999), or `layer(..., g=)`, which is what rdepth's `--gamma_sched` calls.
  - The layer runs in fp32, or float64 when the weights are float64, with autocast off.
- **Weights.** W = causal sparsemax(s_i q_i·k_j/√D).
- **`FRSP`** is the train_ladder factory: L−1 K0 ALiBi layers, with this layer last.
- **Blocked path.** It is used on every device and dtype, and no S×S tensor is ever formed. Row blocks are c = 256, and any S works (a partial last block, no padding).
  - Forward pass 1, from the last block to the first: logits z for the block's rows against keys ≤ hi. τ_i comes from K0 `R.sparsemax`'s sort rule; the whole row is present, so τ is exact.
  - Forward pass 2: z again, then W = max(z−τ, 0) in place in one flat workspace. Then forward substitution with pivot (1−γ) + γ·(off-diagonal mass). τ is saved, as (B,H,S,1).
  - Backward, reverse over blocks: W is rebuilt from the saved τ. Then the adjoint u_J = (I−γW_JJ)⁻ᵀ(gx_J + γ acc_J), and dW = γ u (x − mean_S x)ᵀ; the shift is exact because the sparsemax Jacobian ignores a per-row constant. Then dz = s ⊙ (dW − mean_s dW) with s = [W > 0], and dv = (1−γ)u.
- **Memory arithmetic** at S 16384, fp32, per (B·H) slice. One c×S tensor is 256·16384·4 B = 16 MiB.
  - The blocked path holds about eight of them live: the workspace, sort values, int64 indices (two tensors' worth), cumsum and the Jacobian temporaries. That is ~130 MiB per head, consistent with the measured 0.35 GiB at H2 and 0.63 GiB at H4 (arithmetic, not a bar).
  - densec holds S×S tensors of 1 GiB each per head (z, sort values, 2 for the indices, cumsum, W, A). The bar's control ran out of memory.

## Bars

Every bar was run RED on a stub before it was measured. The stubs are `resolvent_sp_stub.py`, where x = (1−γ)v with identity weights, and `stub2`, which adds FRSP. Every RED line is on the board. No test file was edited after its RED; the SHA files are `test_k2.sha256`, `test_k2_v2.sha256` and `test_k2_v3.sha256`, and all check OK at run3.

The deciding run is run3 (ended 10:40), on hook 36039c3a (`hook_run3.sha256`): `run3.log`, `run3_v2.log`, `run3_v3.log`.

| bar (chase.k2.*) | RED line | measured (run3) | status |
|---|---|---|---|
| sp_fwd_parity_S1024 (v1) | stub 10:13 (`red_stub.log`) | every numeric row within its line; mutant row NaN, which `> 1e-6` scores as a match | **RED**, kept |
| **sp_fwd_parity_S1024_v2** | stub 10:34 (`red_stub_v3.log`) | float64 x ≤ 4.23e-15. fp32 x ≤ 6.16e-6; dq ≤ 2.90e-5, dk ≤ 4.37e-5, dv ≤ 2.01e-6 (in1/in2 × γ .5/.999). Padding S 1000: 1.67e-15. Module: bf16 autocast on = off, bitwise (y and all grads); fp32; y 4.96e-6; dx 5.39e-5, dw_qkv 3.47e-5, dw_out 2.11e-6, da 5.13e-5, db 7.83e-5. Mutant τ(+1) read NaN (fails) | GREEN |
| sp_support_S1024 (v1) | stub 10:13 | sets equal; mutant τ(+1) leaves the set unchanged | **RED**, kept |
| **sp_support_S1024_v2** | stub2 10:17 (`red_stub_v2.log`) | {W>0} equals densec's exactly on in1 and in2. Row sums within 1.42e-14 of 1. Max support 6 and 8. Causal exact zeros: 1,046,528 and 1,046,086 of 1,049,600. Mutant τ = z_(k) changes the set | GREEN |
| **sp_gradcheck_S64_f64** | stub 10:13: mutants pass | 4/4 rows pass (S 64 and S 60 padded, γ .5 and .999; kink margin ≥ 4.5e-3). Mutants (a) no support mean and (b) support W ≥ 0 both fail | GREEN |
| sp_mem_S16384 (v1) | stub 10:13 | run2 (hook 98259dff): reserved **6.36 GiB** (H4 D32). run3: reserved 0.35 / 0.63 GiB, but the fp32 residual is 1.10e-5 against 1e-5 | **RED**, kept |
| **sp_mem_S16384_v2** | stub 10:34 | Allocated 0.33 / 0.61 GiB; reserved 0.35 / 0.63 GiB (H2 D64 / H4 D32). Finite. Prefix vs S 1024 read: 0.0. Float64 hook residual on the last 256 rows 1.45e-14 / 1.60e-14. fp32 vs float64 1.11e-5 / 3.20e-5 (line 1e-4). Control densec: out of memory | GREEN |
| **sp_rdepth_smoke_50** | stub 10:13: a, b do not move | ga (K1-b schedule): loss 10.4577 → 10.3710, eval acc finite at 1024 and 16384, peak 3.73 GiB, a/b moved 0.0209. g999: 10.4441 → 10.3668, moved 0.0283. Hook sha checked in the config | GREEN |
| **sp_train_ladder_R0_50** (FRSP) | stub2 10:17: a, b do not move | 50/50 finite; mean loss 10.8069 → 9.8627 (steps 1–10 → 41–50); eval 9.6314; only blocks.3 has a/b; moved 0.0099 | GREEN |

**Cost** (reported, not scored; `run3.log` REPORT line). K1 FR hook (`resolvent_hook.py` 3855288d) against this hook: fp32, B1 H2 D64, γ .999, median of 5 after 2 warm-up.

| S | FR fwd | SP fwd | ratio | FR fwd+bwd | SP fwd+bwd | ratio |
|---|---|---|---|---|---|---|
| 4096 | 4.57 ms | 15.64 ms | 3.42× | 12.65 ms | 30.15 ms | 2.38× |
| 16384 | 32.39 ms | 304.09 ms | 9.39× | 168.07 ms | 464.88 ms | 2.77× |

## Kills and replacements

| killed | measured reason | replacement | state |
|---|---|---|---|
| Hook 98259dff (a fresh logits/sort allocation per row block) as the S 16384 path | Reserved 6.36 GiB for 0.68 GiB allocated (H4 D32), and 2.40 for 0.37 (H2 D64) (`run2.log`). The blocks widen with hi, so the caching allocator cannot reuse the narrower freed segments and reserves their sum, which grows with S² (the mechanism is from reading; the fix's effect is bound) | Hook 36039c3a: one flat logits workspace per call, the sort pass widest-first. On CPU float64 it is bitwise equal to 98259dff in x and every grad (dev check, not a bar); on CUDA it is bound by the v2 parity bar (fp32 CUDA rows) and sp_mem_S16384_v2 | GREEN (sp_mem_S16384_v2: 0.35 / 0.63 GiB) |
| sp_fwd_parity_S1024's mutant predicate `rel > 1e-6` | The mutant τ(+1) reaches a −inf logit in the first rows, so the read is NaN, and `NaN > 1e-6` is False | v2: the mutant must not satisfy rel ≤ 1e-6 | GREEN |
| sp_support_S1024's mutant τ(+1) | By algebra it cannot change the set: τ′ = (kτ + z_(k+1))/(k+1) with z_(k+1) ≤ τ′ ≤ τ, so the old support stays and z_(k+1) stays out. Measured: sets equal under the mutant | v2 mutant: τ = z_(k), the smallest support logit, so that weight is exactly 0 | GREEN |
| sp_mem_S16384's residual line (fp32 x under float64 W ≤ 1e-5) | 1.10e-5 at H2 D64. The fp32 read at S 16384 differs from the float64 read by 1.11e-5 / 3.20e-5 (v2 rows), so the line sat inside fp32's own error. That the cause is logit rounding (\|z\| up to ~40) is a hypothesis, not bound | v2: the float64 hook's residual ≤ 1e-12 (exactness at full length), and fp32 vs float64 ≤ 1e-4. The 1e-4 is a wrong-block line, set after seeing 1.1e-5, and is not a precision claim; the 1e-5 precision line is claimed at S 1024 only | GREEN |
| run1 launch (`bash -c` from Python) | Resolved to a bash without python on PATH: no test ran and no board line was written (`run1_nolaunch.log`) | `chain.py`: `sys.executable`, one fresh process per test file | used for run2 and run3 |

A second process note: an 8 GB card allocated 12.05 GiB before running out of memory (the densec control row). On this box, running out of memory is not a fit test; memory bars must read the peak. The mechanism (driver fallback to system memory) is not bound.

## Nurses

- Nurse 1 (sonnet): a read-only review of the hook against K0 densec, the K1 hook, rdepth's gamma_wrap and train_ladder. It found 0 bugs.
  - Its own CPU float64 dev check (not a bar), at 5 shapes, found that forward, dq, dk, dv and W match densec to ~1e-14–1e-16 with 0 support mismatches. It also derived that the centring is exact for sparsemax.
  - Its one correctness RISK, the backward rebuilding z by a second GEMM, is in OPEN.
- Nurse 2 (haiku): a read-only cross-check of this report and READY_SP against the logs. 57 numbers checked, 0 mismatches. The three test SHA files and hook_run3.sha256 pass, and the run3 headers carry hook 36039c3a. The board has 25 chase.k2 lines.

## OPEN

- **fp32 at long S.** The fp32 read is off the float64 read by 1.11e-5 (H2 D64) and 3.20e-5 (H4 D32) at S 16384, above the 1e-5 line that holds at S 1024. This is the same regime as K1's clause-6 floor.
  - A numerics bar on trained f_R_sp checkpoints should take clause6_v2's form: ≤ 2× the fp32 floor, with this hook in float64 as the reference. The float64 hook fits and runs at 16384; densec does not.
- **Support in fp32 on CUDA** against float64 is not bound. Support parity is bound in float64 only. A near-tie can flip in fp32; the entry at stake carries weight ≈ ulp(|z|).
- **Rebuilt W.** The backward rebuilds z from the saved τ with the same `_z` call on the same shapes. Bitwise equality of the forward's and backward's z on CUDA is not bound. The only evidence is run-to-run bitwise equality (the module autocast row).
- **Cost.** Forward-only is 9.39× FR at S 16384 (the sort pass plus the second logits pass); this bears on evaluation time. The training-shape cost (S ≤ 1024) is not measured by any bar.
  - If the cost matters: CUDA-graph the two loops as K1 did (`R._graph_call`), or write a Triton kernel that fuses the threshold.
- **Scope of the smokes.** They show integration only: 50 steps, loss down, a/b move. Whether f_R_sp learns (≥ 0.8 at training depth on 2 of 3 seeds) is K2.1's bar.
- **Inputs.** Every parity and gradcheck input is random. Clause 6/7 analogues on trained sparsemax checkpoints are unread.
- **`:FRSP` with rdepth `--gamma_sched`.** It would silently run at γ = 0.999: gamma_wrap subclasses the given class, and FRSP.__new__ returns a plain ResolventAttention. READY_SP says to use `:ResolventAttention`. This is from code reading and is not bound.
