# WILSON — Phase K, it.K1: clause 8 (MFU at R0) and the R0 language-model ward

WD = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/wilson`

## Facts

- **MFU at R0 is 0.134 for the ALiBi twin (a_L) and 0.115 for the resolvent arm (f_R), at ctx 1024 b8.** Both are measured against a peak re-measured in the same lock hold: 32.31 TF for (a_L) and 31.85 TF for (f_R). At ctx 4096 b2 the values are 0.144 and 0.113. Against the old 26.77 TF figure, the same logs give 0.162 and 0.174 for (a_L), which reproduces the K0 prior (0.163 and 0.171). The contract assumed 0.35. Bar `wilson.k1.mfu`: RED on its stub (17/17 checks fired), then GREEN on both sessions.
- **All four R0 runs completed** (rc 0, 17,645 steps, 144,547,840 tokens each):

| arm | seed | final val loss |
|---|---|---|
| (a_L) | 1 | 4.997022 |
| (a_L) | 2 | 4.984507 |
| (f_R) | 1 | 5.016564 |
| (f_R) | 2 | 5.008180 |

- **Δ = (f_R) − (a_L).**
  - Per seed: **+0.019543** (seed 1) and **+0.023673** (seed 2); mean +0.021608; |Δ1 − Δ2| = 0.004130.
  - Seed spread within an arm: (a_L) 0.012515, (f_R) 0.008385.
  - No R-CARRY verdict: that is K3.
- **Hours per run (job wall clock).** (a_L) 0.507 and 0.516 h; (f_R) 0.631 and 0.688 h. That is 2.54–3.44× the contract's Y4 of 0.2 h, and 1.18–1.60× the K0 R0 figure of 0.43 h.
- **Ward bar `wilson.k1.ward` is RED.** Of its 17 checks, only `done_clean` fired, on 2 of the 4 rows:
  - a_L_s2 saw foreign GPU PIDs 20036 and 30392.
  - f_R_s1 saw `DeviceCensus.exe` (PID 29244) at 04:52:19.
  - The other 16 checks pass on all 4 rows and on both deltas.
- **(a_ss) and (a_loop): not run, no hook** (see §7).

## 1. Bars (written and run RED before their measurements; neither was edited after its RED)

| test | sha256 | RED on stub | measured |
|---|---|---|---|
| `wilson.k1.mfu` (`test_mfu_k1.py`) | 46c229c45b7881ec27658878c85021592f08c21001fa9725cc325684e506eecb | 02:43:22, exit 1, 22 fails, 17/17 checks fired (`red_stub_mfu.log`) | GREEN, exit 0, 0 fails: session aL (`green_mfu_aL.log`, 02:47:39) and session fR (`green_mfu_fR.log`, 04:36:35) |
| `wilson.k1.ward` (`test_ward_k1.py`) | 7c9bcf5cba9db07d4d2e74f246e5f0b8becaa9acc8bfd78e545046ae9b8b793f | 02:45:10, exit 1, 26 fails, 17/17 checks fired (`red_stub_ward.log`) | RED, exit 1, 2 fails, both `done_clean` (`final_ward.log`, 05:56:15) |

- The test files have mtimes 02:43:20 and 02:45:06, before their RED logs. Their sha256 now equals the sha256 recorded right after each RED.
- **What `wilson.k1.mfu` asserts, per session.**
  - The peak: it was measured on the 4060, over 5 reps; the median is recomputed; the session tags match; it ran between the session start and the first probe.
  - Every probe job has exactly one table row, and no row lacks a job.
  - A probe with rc 3 or an abort record is reported `no_fit` with null tok/s and MFU; a `no_fit` row is real (rc 3 plus a memory abort).
  - For fitting rows: rc 0; all steps logged; every step's tok/s equals tokens/dt; reported tok/s is within 0.5 of the logged mean over steps 11+; reported MFU is within 5e-4 of (6N + 6·L·ctx·d)·tok/s / (session median · 1e12); peak memory is below device memory; the tok/s CV is below 0.10; no foreign GPU PID was seen.
- **What `wilson.k1.ward` asserts:** the docstring of `test_ward_k1.py`, lines 5–22.

## 2. bf16 peak, re-measured in each session

Recipe (`peak_k1.py`, sha 87e61520…): 4096³ bf16 matmul, 10 warmup + 50 timed, 5 reps. Temperature and SM clock are read before and after every rep. Every peak ran under the GPU lock, as the first and last job of the session.

| session | label | TF per rep | median | °C (start → end) | SM MHz across reps |
|---|---|---|---|---|---|
| aL | pre (02:44) | 25.80 / 32.43 / 32.84 / 32.31 / 32.14 | **32.31** | 57 → 75 | 1890 before rep 1, then 2685 |
| aL | post (02:47) | 24.07 / 30.54 / 29.12 / 29.69 / 28.11 | 29.12 | 68 → 87 | 2685 → 2490 → 2280 |
| fR | pre (04:32) | 25.16 / 32.55 / 31.98 / 31.85 / 31.50 | **31.85** | 64 → 83 | 1890 before rep 1, then 2685–2700 |
| fR | post (04:36) | 26.10 / 27.78 / 28.46 / 28.61 / 26.03 | 27.78 | 71 → 88 (max 91) | 2640 → 2295 → 2430 → 2160 |

- **Thermal drift across a ~4 min sweep.** The post/pre median is 0.901 (aL) and 0.872 (fR).
- **Rep 1 is the lowest rep in 3 of the 4 measurements.** The SM clock read 1890 MHz before it in all 4.
- **Which peak the MFU uses.** MFU uses the pre median, as the test states. It was written before the sweep.

## 3. MFU table (the bar's rows)

MFU = (6N + 6·L·ctx·d)·tok/s / (session pre median), with L = 4 and d = 128.
- Probes run 300 steps on the real shards with seed 1; tok/s is the mean over steps 11–300.
- (f_R)'s MFU counts model FLOPs only: the resolvent's triangular solve and its adjoint are not counted.

| arm | ctx | batch | status | tok/s | MFU vs session peak | MFU vs 26.77 | peak mem GiB |
|---|---|---|---|---|---|---|---|
| (a_L) | 1024 | 8 | fit | 93,243 | **0.134** | 0.162 | 5.63 |
| (a_L) | 1024 | 4 | fit | 91,551 | 0.132 | 0.159 | 2.88 |
| (a_L) | 1024 | 16 | **no fit** (rc 3, abort at 11.09 GiB) | — | — | — | — |
| (a_L) | 4096 | 2 | fit | 83,424 | **0.144** | 0.174 | 5.69 |
| (a_L) | 4096 | 1 | fit | 73,133 | 0.127 | 0.153 | 2.93 |
| (a_L) | 4096 | 4 | **no fit** (rc 3, abort at 11.15 GiB) | — | — | — | — |
| (f_R) | 1024 | 8 | fit | 78,794 | **0.115** | 0.137 | 5.73 |
| (f_R) | 1024 | 4 | fit | 68,756 | 0.100 | 0.119 | 2.94 |
| (f_R) | 1024 | 16 | **no fit** (rc 3, abort at 11.18 GiB) | — | — | — | — |
| (f_R) | 4096 | 2 | fit | 64,392 | **0.113** | 0.135 | 5.79 |
| (f_R) | 4096 | 1 | fit | 54,261 | 0.095 | 0.113 | 3.00 |
| (f_R) | 4096 | 4 | **no fit** (rc 3, abort at 11.24 GiB) | — | — | — | — |

- **Sources.** `WD/mfu/aL/mfu_table.json` and `WD/mfu/fR/mfu_table.json`, built by `mfu_table_k1.py`. Job records are in `WD/mfu/*/jobs.jsonl`; logs are in `WD/mfu/*/runs/`. Probe checkpoints were deleted.
- **K0 prior (vs 26.77), R0:** 0.163 (c1024 b8), 0.157 (b4), 0.171 (c4096 b2), 0.164 (b1). K1 (a_L) vs 26.77: 0.162 / 0.159 / 0.174 / 0.153.

## 4. Runs table (R0, 4 × 128, ctx 1024)

| arm | seed | val loss | tok/s | hours | peak mem GiB | ckpt path | hook sha256 |
|---|---|---|---|---|---|---|---|
| (a_L) | 1 | 4.9970 | 87,410 | 0.507 | 5.63 | `WD/runs/a_L_s1/ckpt.pt` | 4c00beff4ce77211113faf20a949bcb7c99d1d50967d1b8c482e7110da667e03 (`train_ladder_k1.py`; `--attn alibi` is built in) |
| (a_L) | 2 | 4.9845 | 85,587 | 0.516 | 5.63 | `WD/runs/a_L_s2/ckpt.pt` | 4c00beff… (same) |
| (f_R) | 1 | 5.0166 | 70,764 | 0.631 | 5.73 | `WD/runs/f_R_s1/ckpt.pt` | 3855288d7da7acdebf54b9b1979d9b0b5afebb3dca6d8ccefd684812d33451e2 (`K1/chase/resolvent_hook.py:FR`) |
| (f_R) | 2 | 5.0082 | 65,884 | 0.688 | 5.73 | `WD/runs/f_R_s2/ckpt.pt` | 3855288d… (same) |

- **Parameters.** (a_L) 7,227,648; (f_R) 7,227,652, the extra 4 being a_h and b_h for 2 heads.
- **Checkpoints.** Each holds the model, optimizer and step 17,645.
- **Hook dependencies.** `resolvent_hook.py` imports two files that the harness does not hash:
  - `tests/foreman/phase_k/K0/chase/resolvent.py`: f850c568f3454c36224325d7b28df6a081a0e32069b860188882b6a6374ceccd
  - `tests/foreman/phase_k/K0/wilson/train_ladder.py`: d524b13005d86bd49ae1d7692d41de4e0030542f7a19a212fb562de916758d93
- **Harness.** `train_ladder_k1.py` is the K0 harness (d524b130…) plus four edits:
  - `--eval_seed`;
  - `harness_sha256` in the config;
  - `device_total_gib` in the config;
  - `wall_total_s` in the end record.
- **Common configuration.**
  - Budget `--tokens` 144,552,960 (20 × 7,227,648), giving 17,645 steps at 8,192 tokens per step.
  - Optimiser: lr 6e-4 with warmup 100 and cosine decay to 0.1×; AdamW (0.9, 0.95); weight decay 0.1; bf16 autocast (the hook runs its layer in fp32 with autocast off); `--deterministic` off.
- **Evaluation.** Every 250 steps, 40 batches × 8 × 1024 = 327,680 validation tokens. `--eval_seed 0` gives the same validation batches in all 4 runs, so every curve has 71 points (`WD/runs/ward_curves.json`).
- **Training data.** Batches come from `rng([seed, k])`, so the same seed draws the same batches in both arms.
- **Polling.** (f_R) ran with `--no_poll`, following Chase's READY. (a_L) polled at start only.
- **Learned logit scale** in (f_R)'s resolvent layer at step 17,645 (init a = 1, b = 0):
  - s1: a = [1.2838, 1.3536], b = [0.4204, 0.4027].
  - s2: a = [1.3527, 1.3534], b = [0.4542, 0.4862].
- **Board.** One landing finding per run with its path. **Chase reads these checkpoints.**

## 5. Δ (f_R) − (a_L): numbers only, no verdict

| step | Δ seed 1 | Δ seed 2 |
|---|---|---|
| 250 | +0.0098 | +0.0124 |
| 1,000 | +0.0172 | +0.0199 |
| 2,500 | +0.0172 | +0.0244 |
| 5,000 | +0.0133 | +0.0247 |
| 10,000 | +0.0212 | +0.0364 |
| 15,000 | +0.0172 | +0.0259 |
| **17,645 (final)** | **+0.019543** | **+0.023673** |

- The final mean is +0.021608; the difference between the two seeds is 0.004130.
- Within-arm seed spread: (a_L) 0.012515, (f_R) 0.008385.
- The contract's R-CARRY prediction line is |Δ| ≤ 0.01 at every rung, and its counter is > 0.02 at R2. Neither is scored here.

## 6. Hours vs Y4 and the K0 table

| run | job wall h | training steps only h (Σ dt) | ×Y4 (0.2 h) | ×K0 R0 c1024 (0.43 h) |
|---|---|---|---|---|
| a_L_s1 | 0.5073 | 0.472 | 2.54 | 1.18 |
| a_L_s2 | 0.5157 | 0.479 | 2.58 | 1.20 |
| f_R_s1 | 0.6314 | 0.583 | 3.16 | 1.47 |
| f_R_s2 | 0.6880 | 0.632 | 3.44 | 1.60 |

- **What "job wall" covers.** gpujob.py's clock around the subprocess: process start, CUDA and graph setup, 71 evaluations, and the checkpoint save.
- **Y4 recomputed.** The Y4 formula gives 0.1992 h at ctx 1024 and 0.2396 h at ctx 4096. With the measured MFU and the session peak substituted, it gives 0.431 h for (a_L) and 0.510 h for (f_R).
- **Throughput fell within every run.** Mean tok/s, first 1,000 steps vs last 1,000:

| run | first 1,000 | last 1,000 |
|---|---|---|
| a_L_s1 | 92,680 | 88,763 |
| a_L_s2 | 89,931 | 83,605 |
| f_R_s1 | 76,927 | 68,268 |
| f_R_s2 | 65,358 | 59,006 |

- **One mid-run sample of f_R_s1** (about 04:50): 84–86 °C and SM 1,875–2,670 MHz. nvidia-smi `memory.used` read 7,705 of 8,188 MiB, with one compute process, PID 26416, which is the run's own child of gpujob PID 6532.

## 7. (a_ss) and (a_loop): not run, no hook

- Board line 5660 (Cameron) names `ass4` and `aloop5` as arms of the chain-bed runner in `K1/cameron/rdepth.py`.
- `rdepth.py:4-7` and `:96-125` define them as follows:
  - `ass` is `SSMaxAlibi`, inside rdepth's `Chain` model.
  - `aloop` is one weight-shared block applied T times at the block level.
- Neither is a train_ladder `--attn` class, and no board finding names an `--attn` hook file for either.

## 8. GPU lock and board

- **Wilson's GPU jobs.** Every one ran under `gpu.lock.d` through `gpujob.py`: one fresh subprocess per job, with owner.txt and a release in `finally`. Lock holds:

| job | held |
|---|---|
| mfu_aL | 02:44:25–02:47:23 |
| a_L_s1 | 02:48:19–03:18:46 |
| a_L_s2 | 03:18:46–03:49:43 |
| mfu_fR | 04:32:15–04:36:13 |
| f_R_s1 | 04:36:14–05:14:07 |
| f_R_s2 | 05:14:08–05:55:26 |

- **Yield.** At 03:49, following the dispatcher, Wilson posted that it would take no lock until READY. READY appeared at 04:29:09.
- **Cooldown.** The chain re-acquired the lock 1 s after releasing it, at 03:18:46 and 05:14:08. The dispatcher then ruled a 3-min cooldown, which is now in `gpujob.py` (sha ac9b829a…), applied after f_R_s2 had started.
- **Foreign GPU PIDs seen during jobs.**
  - a_L_s2: 20036 and 30392. Both had exited before they could be identified; the launcher did not timestamp sightings until after that run. The run's 1,000-step block medians for steps 3,001–8,000 are 82.8–84.3k tok/s, against 87.9–92.9k (87,927–92,923) in every other block of both a_L runs.
  - f_R_s1: `DeviceCensus.exe` at 04:52:19.
  - MFU probes: none during any probe. nvidia-smi listed PID 8944 before probe_aL_c4096_b4 (a no-fit probe) started, and it was gone by 02:47.
- **Board correction.** A board finding said "15 of 17 checks pass"; the correct count is 16. A correction line is appended on the board.

## Unverified and limits

- **Contention.** The identity of PIDs 20036 and 30392 is unverified. Their effect on a_L_s2 is inferred from the throughput dip, not measured. Whether they changed a_L_s2's loss beyond default-mode run-to-run nondeterminism is unverified; K0 measured a max |dloss| of 2.22e-4 by step 400 between identical runs.
- **Spill check.** `done_spill` compares each window with the run's own median, so a run that spilled throughout would pass it. The guard used instead is ward tok/s against probe tok/s: 0.94 and 0.92 of the probe for (a_L), 0.90 and 0.84 for (f_R).
- **The peak is an attained matmul peak** on this card; no vendor spec was fetched. It fell 10–13% between pre and post within each session. MFU against the post median would be 1/0.901× the table for (a_L) and 1/0.872× for (f_R).
- **Paging.** Whether WDDM paged any memory during the ward is unverified. nvidia-smi `memory.used` was sampled once (7,705 of 8,188 MiB, during f_R_s1) and never during the (a_L) runs. PyTorch-allocated peak was 5.63–5.73 GiB.
- **Sample sizes.**
  - Two seeds per arm, so no CI. A seed changes both the initialisation and the data order.
  - The validation set is 327,680 tokens (40 random windows of 8 × 1024) from the 10.3M-token val shard; its sampling error is not measured.
  - The (a_L) and (f_R) runs and their peaks ran at different times: 02:44–03:50 and 04:32–05:55.
- **Nurses.** Three haiku nurses launched and watched the jobs. Nurse 2's final message held only its repetition count, so Wilson read every a_L number from the files directly. Nurses 1 and 3 returned output that matches the files. No sonnet nurse was used.
- **Not re-run by Wilson:** Chase's hook bars (run4, GREEN 4/4 per `K1/chase/run4.log`).
- **Disk.** Probe checkpoints were deleted. Only the 4 final checkpoints remain (about 87 MB each; `WD/runs` is 349 MB).
