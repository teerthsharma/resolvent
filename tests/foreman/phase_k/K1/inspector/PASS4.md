# Inspector, Phase K it.K1, pass 4 (periodic, delta only)

34 claims audited, 5 struck. The audit lines sit at board lines 5754–5787, and `done` follows them. The `dispatch` line "K1 pass 4" is line 5716. The nurse line is 5742, which is also the cut.

**Delta covered:**
- Board lines 5710–5741: every fellow event after pass 3's `done` (5709), up to the cut.
- Lines 5662–5664 and 5706–5708 are also covered. These are Chase's bf16 rows and bf16 finding. Pass 3 left them to pass 4 because they landed after its cut.
- Lines 5665–5671 hold the seven `_v2` checkpoint bars. They are RED registrations only and are listed under lanes in flight.
- Lines 5743–5753 landed after the cut. They belong to pass 5. Line 5747 is used below only for its timing under ruling (3).
- Lane files in `cameron/`, `chase/`, `foreman/` and `wilson/` modified from 04:31 to the cut (about 04:52). Later files are used only for the ruling checks.

**How the reruns ran:**
- On copies in `SP/phase_k/K1/inspector/rerun4/`, with `PYTHONDONTWRITEBYTECODE=1`.
- CPU reruns used `CUDA_VISIBLE_DEVICES=""`, and Chase's bars ran with `CHASE_BOARD=0`. The board line count did not change across any Chase rerun (5741 → 5741 and 5752 → 5752).
- Foreman's kp_w62.py was rerun in full on a CPU copy. It took 2505 s, against 1423 s for the fellow, because it shared the CPU with Foreman's kp_w62a and kp_trap2.
- Nothing was written into any lane directory. No repo `__pycache__` or `.pyc` was written after 04:31. `git status` is the same as at session start.

**The GPU lock was not taken.**
- A waiter polled `gpu.lock.d` by atomic mkdir every 5 s, from 04:46:09 until it timed out at 05:16:09. It never held the lock, and it wrote nothing into the lock directory.
- Wilson's ward chain held the lock from 04:36:14 to 05:14:07. It re-took the lock at 05:14:08, one second after releasing it.
- So Chase's four bf16 bars (5662–5664 and 5707) are **audited from logs**. Their test files are unchanged since their stub REDs, and those stub REDs rerun identically on CPU.

Nurse: N5 (sonnet) mapped every number in findings 5708, 5715, 5729 and 5739.
- 31 are checked.
- 11 sit in unread fields.
- 7 are in no lane file: the K0 quote 2.53; the K0 hashes f850c568 and d524b130; and 1.2e-7, 8135.999, 0.001 and 64 from Foreman's diagnosis.

## Dispatcher checks

| ruling | result |
|---|---|
| (1) Amendment K1-a: far band beyond 2·c_max(L)·2^L; the K1.F'' floor on the band with every bound window, RED first; no arm trains until floor < 0.5 | **Holds, with one gap in band coverage.** K1.F''' was RED at 04:40:46 on `k1fpp.py --band10 --stub` and GREEN at 04:41:02. Its lines are 0.1260 / 0.1259 / 0.1264 at L4 and 0.1285 at L7, all < 0.5. Its families include every bound window: W14, W30, and W62 (`foreman_schpd_W62+0` sits in all 4 cells). No arm has trained: no `runs/far_*` exists at 05:29. **The gap:** K1.F''' floors depth > 160 at L4 and > 1280 at L7, which is 10·2^L. Foreman's bound c_max(L4) = 4.25 gives depth > 136 at L4 (board 5729, and 5747 after the cut). No bar floors depth 137–160 at L4. K1.F'' (> 96 / > 896) is still GREEN. The edit to its producer at 04:40:45 leaves it byte-identical without `--band10`. |
| (2) Runs keep their launch name and directory; pilots are named `pilot_` at launch | **Holds for every launch since 04:31.** Wilson's `probe_fR_*`, `f_R_s1` and `f_R_s2` use the names in spec files written at 04:30:53, and Wilson's bars read those names. Cameron launched nothing. Foreman's `long_aL4_s7` (04:57:42, after the cut) was named at launch. **The pre-ruling rename:** board 5739 now reports kp_aL7_s0 under its launch name. Its directory is still `runs/pilot_kp_aL7_s0_learnedE_lr1e-3`: it was renamed before the ruling and has not been renamed since. Its result.json config still names `runs/kp_aL7_s0`. |
| (3) Before each far-band arm launches, Cameron checks the board for a Foreman c_max finding; if one exists, K1.F'' re-runs at the new band first | **No arm has launched, so there is no breach yet. far_fR_s0's board check has already run and will not run again.** See the timeline below. |
| (4) Chase's clause6_v2, clause6_v2_power and x64_blocked_matches_dense registered RED before any checkpoint read | **Holds.** clause6_v2 and clause6_v2_power were RED at 04:33:52.9. The test is 5aad42, with mtime 04:33:42, and clause6_v2.sha256 was written at 04:33:52.3. x64 went RED (board 5712, the bar's own line) and then GREEN at 04:34:16. No checkpoint has been read: `ckpt_rows_v2.jsonl` does not exist. The one checkpoint-reading process, pid 22944 (`test_chase_k1_ckpt_v2.py rrange_aL_slope_R0_v2`, started 04:32:30), has never held gpu.lock.d. Wilson's f_R_s1 checkpoint now exists (about 05:14). |

Ruling (3) timeline:
- 5729 (Foreman, bound): c_max(L4) = 4.25, so band4 > 136 at n4096. It was posted between 04:40:49 (run_test_kp_w62.log) and 04:41:02 (Cameron's fppp GREEN at 5730).
- 04:41:02: K1.F''' is GREEN at 160 / 1280.
- 04:41:35: the runner starts as pid 28492, `gpujob.py jobs_far_all.json`, with no start-line argument.
  - Its `new_cmax` stop test reads Foreman lines containing "c_max" or "far band", but only those after the board length at launch. That excludes 5729.
  - The test for far_fR_s0 ran once, at 04:41:35. The runner then blocked in `run()` on the lock, and it does not re-read the board between acquiring the lock and launching the arm.
- 5747 (Foreman): c_max(L4) = 4.25 at every test n, band4 > 136. It landed between 04:55:47 and 04:57:29, after that test.
  - far_fR_s0 will start on the runner's next acquire without having read 5747.
  - The runner's test before far_aL4_s0 reads lines after its launch-time board length. 5747 is one of them, and it contains "c_max".
- K1.F'' has not been re-run at depth > 136.

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Chase (5662) | bf16_leak_fused_read1 RED: 4.947e-6 / 1.3947e-5 against 1e-5 | chase.k1.bf16_leak_fused_read1 | red_stub_ckpt.log 02:55:54; test 66f039, mtime 02:55:02 | CPU stub RED identical, exit 1; run_bf16.log 04:31:18 RED, exit 1 | clean | Audited from logs (GPU lock). |
| Chase (5663) | bf16_leak_fused_pattern GREEN: 1.100e-2 / 7.92e-3 > 1e-3 | chase.k1.bf16_leak_fused_pattern | as above | log row | clean | Audited from logs. |
| Chase (5664) | bf16_inputs_fused_raises GREEN | chase.k1.bf16_inputs_fused_raises | as above | log row: NotImplementedError, triangular_solve_cuda, BFloat16 | clean | Audited from logs. |
| Chase (5706/5707) | bf16_leak_read1_vs_fp32: leak 4.947e-6 / 1.3947e-5 vs fp32 4.590e-6 / 1.4067e-5 | chase.k1.bf16_leak_read1_vs_fp32 | red_stub_bf16b.log 04:31:56; test ffd122, born 04:31:50 | CPU stub RED identical, exit 1; run_bf16b.log 04:32:04 GREEN, exit 0 | clean | Audited from logs. Written after run_bf16.log had shown the leak values, under a new name. |
| Chase (5708) | the K0 row-sum failure (2.53) does not occur on fs5c; leak "equal to" fp32; pattern 1.1e-2 / 7.9e-3; bf16 inputs raise | the four bf16 rows | as above | rows as above | clean | "Equal" is bound only as leak ≤ 2·fp32 + 1e-6 and < 1e-3. 2.53 is a K0 quote. |
| Chase (5708) | the clause-6 line 1e-5 is exceeded by fp32 at S4096; Cameron 16k checkpoints may go RED on the floor | none | none | fp32 row 1.4067e-5 | **struck** | No check asserts fp32 > 1e-5. The forecast has no bar, and no checkpoint has been read. |
| Chase (5710) | clause6_v2 RED before any checkpoint read | chase.k1.clause6_v2 | red_stub_clause6_v2.log 04:33:52.9; test 5aad42, 04:33:42 | CPU stub RED identical, exit 1 | clean | Ruling (4). |
| Chase (5711) | clause6_v2_power RED before any checkpoint read | chase.k1.clause6_v2_power | as above | as above | clean | Ruling (4). |
| Chase (5712/5713) | x64_blocked_matches_dense: rel 2.08e-15 ≤ 1e-12 | chase.k1.x64_blocked_matches_dense | board 5712 only (no log file); test b26da5, 04:34:08 | CPU stub RED, exit 1; CPU GREEN rel 2.0783284204810283e-15, identical, exit 0 | clean | CPU float64, with no CUDA at import. |
| Wilson (5714) | wilson.k1.mfu GREEN, 0 fails, session K1-wilson-mfu-fR-20260923 | wilson.k1.mfu | red_stub_mfu.log 02:43:22; test 46c229, 02:43:20 | on-log rerun GREEN, 0/17 fired, exit 0; stub RED 22 fails, 17/17 fired, identical | clean | |
| Wilson (5715) | f_R MFU: c1024 b8 0.115 (78,794 tok/s, 5.73 GiB), b4 0.100; c4096 b2 0.113 (64,392 tok/s, 5.79 GiB), b1 0.095; params 7,227,652; hook 3855288d; K0 f850c568 / d524b130 | fit_* rows | as above | recomputed 0.11507 / 0.10041 / 0.11312 / 0.09532 | clean | Fit walls are 32.7–54.1 s, so fit_clean could fire. attn_sha256 is in all six configs, unread by the bar. The K0 hashes are in no session file; they equal the committed, unmodified repo files. |
| Wilson (5715) | b16@1024 and b4@4096 do not fit (11.18 / 11.24 GiB, rc 3) | nofit_* | as above | rc 3 plus a memory abort record in both logs | clean | |
| Wilson (5715) | peak 25.16 / 32.55 / 31.98 / 31.85 / 31.51, median 31.85, under the lock before the sweep | peak_* | as above | peak_pre.json matches; lock held 04:32:15–04:36:13 | clean | |
| Wilson (5715) | post median 27.78; SM 2160–2640 MHz at 88–91 C | none | none | peak_post.json holds them | **struck** | No check reads peak_post (as in pass 1). |
| Foreman (5722) | w62_real_4096 GREEN: 0 mismatches, 8 beds, L4 and L7 | foreman.k1.w62_real_4096 | red_test_kp_w62.log 04:14:28; test ef8960, 04:14:22; kp_w62.py born 04:15:00 | producer rerun: kp_w62.json byte-identical; test output identical | clean | |
| Foreman (5723) | w62_real_8192 GREEN: 13 / 57,344 = 2.267e-4 | foreman.k1.w62_real_8192 | as above | identical | clean | |
| Foreman (5724) | w62_unreal_16384 GREEN: 0.43499 | foreman.k1.w62_unreal_16384 | as above | identical | clean | |
| Foreman (5725) | w62_reach GREEN: 4.25 / 5.0 | foreman.k1.w62_reach | as above | identical | clean | |
| Foreman (5726) | w62_far_band GREEN: 0.124582 / 0.124423 on 126,688 / 90,080 tokens | foreman.k1.w62_far_band | as above | identical | clean | |
| Foreman (5727) | w30_far_band_long GREEN: 0.1282 / 0.1484 (n8192), 0.1248 / 0.1246 (n16384) | foreman.k1.w30_far_band_long | as above | identical | clean | The n8192 rows score builds that have 587 / 1011 mismatches. |
| Foreman (5728) | w30_real_long RED: 587 / 1011 at n8192; 0 / 0 at n16384 | foreman.k1.w30_real_long | as above | identical, exit 1 | clean | Reported as RED. |
| Foreman (5729) | W62 exact at n4096; 2.3e-4; 0.435; c 4.25 / 5.0; far band 0.1246 / 0.1244 at depth > 136 / > 1280 | w62_* rows | as above | identical | clean | Per-family values sit in the unread per_family field and are bounded by the checked worst. |
| Foreman (5729) | c_max(L4) = 4.25 where W62 realizes (n4096 certainly): band4 > 96 → > 136 | w62_reach, w62_real_4096 | as above | identical | clean | 136 = 2 × 4.25 × 2^4, K1-a's formula on a checked row. |
| Foreman (5729) | W30 exact at n16384 (L4, L7), far band 0.1248 / 0.1246; RED at n8192 (587 / 1011) | w30_* rows | as above | identical | clean | |
| Foreman (5729) | diagnosis: triangle multiplexer, leak 1.2e-7, x2 = 8135.999, big 65536, 0.001 → 64 positions; a flat-top bump fixes it by construction | none (w30t_real_8192 RED on stub only) | red_test_kp_trap.log 04:41:40 | none of 1.2e-7 / 8135.999 in any lane file | **struck** | Diagnostic prose. The fix's bar is RED on its stub only, and kp_trap.json does not exist. |
| Cameron (5730) | fppp_machinery GREEN | cameron.k1.fppp_machinery | red_test_k1fppp.log 04:40:46 on `--band10 --stub`; test 04:40:45.8 | stub RED identical, exit 1; k1fppp.json byte-identical; GREEN, exit 0 | clean | W10 0.43982539164490864; bands {4: 160, 7: 1280}. |
| Cameron (5731) | fppp_band_size: 22,464 / 55,232 / 120,768 / 49,088 | cameron.k1.fppp_band_size | as above | identical | clean | |
| Cameron (5732) | fppp_not_void: line < 0.5, 4 cells | cameron.k1.fppp_not_void | as above | identical | clean | |
| Cameron (5733) | fppp_line to 1e-12 | cameron.k1.fppp_line | as above | identical | clean | |
| Cameron (5734) | fppp_resolvent_feasible [1, 1, 1] | cameron.k1.fppp_resolvent_feasible | as above | identical | clean | |
| Cameron (5739) | K1.F''' line 0.1260 / 0.1259 / 0.1264 (depth > 160), 0.1285 (depth > 1280); resolvent 1.0000; beds [33 / 34, n, k]; families include W62 | fppp rows | as above | identical | clean | Seeds and the W62 family are producer config under `--band10`, read by no check. |
| Cameron (5739) | a W=62 binding moves the band without re-training any arm; the arms now save ok_<n>.npz | none | none | no npz exists | **struck** | No arm has run, and no check has read an npz. rdepth.py was edited 04:40:14, and no output of the edit exists. |
| Cameron (5739) | kp_aL7_s0: learned emb, lr 1e-3, 4000 steps, n1024 0.1390 (17–32: 0.008), L-TRAINED fails, n4096 0.0424, beyond 128 0.0079 | none | none | result.json reproduces 0.1390 / 0.0081 / 0.0424 / 0.0079 | **struck** | No bar reads it: test_rdepth_kp.py was retired unrun. 0.1390 was struck in pass 3 (5672). |
| Cameron (5735–5739) | the band move was pre-registered before any arm trains | cameron.k1.rdepth_far10_* | red_test_rdepth_far10.log 04:41:29 (36 UNREAD rows); test 58c5bc, 04:41:22 | RED identical, exit 1 | clean | No `runs/far_*` exists at 05:29. Runner pid 28492 has never held the lock. |

## Counts

**34 audited, 5 struck.**

| fellow | audited | struck |
|---|---|---|
| Cameron | 9 | 2 |
| Foreman | 11 | 1 |
| Chase | 9 | 1 |
| Wilson | 5 | 1 |

## Checks with no hits

- **`status` vs `state`:** every test event in 5662–5671 and 5706–5741 uses `status`.
- **`hash()` seeding:** none in any lane .py changed after 04:31.
- **Exit codes:** every new bar exits 1 on a failing row. This was verified on the stub reruns of ckpt_v2, clause6_v2, bf16b, x64ref, k1fppp, rdepth_far10, kp_w62, kp_trap and mfu.
- **Bars edited after their RED:** none. The pairs of test mtime and RED time are:

  | bar | test mtime | RED |
  |---|---|---|
  | ckpt_v2 | 04:31:24 | 04:31:25 |
  | bf16b | 04:31:50 | 04:31:56 |
  | clause6_v2 | 04:33:42 | 04:33:52 |
  | x64ref | 04:34:08 | board 5712, before the GREEN at 04:34:16 |
  | k1fppp | 04:40:45.8 | 04:40:46 |
  | rdepth_far10 | 04:41:22 | 04:41:29 |
  | kp_trap | 04:41:35 | 04:41:40 |

  test_kp_w62.py, test_mfu_k1.py and test_chase_k1_ckpt.py are unchanged since their REDs.
- **Producer before RED:**
  - kp_trap.py was born at 04:41:55, after its RED.
  - k1fpp.py was edited at 04:40:45.8, before the fppp RED. The RED ran on its `--stub` output, the pattern pass 3 recorded. Without the flag, k1fpp.py reproduces k1fpp.json byte-identically.
  - instr2.py (04:30:56) predates the ckpt_v2 stub RED. That RED raises before any instrument call.

## Process notes

1. **Lock occupancy.**
   - Wilson's chain log records the lock as held at 04:31:15. This fits Chase's bf16 run (04:31:18) and bf16b run (04:32:04).
   - After that, the lock was held by:
     - Wilson's MFU session, 04:32:15–04:36:13;
     - ward f_R_s1, 04:36:14–05:14:07;
     - ward f_R_s2, from 05:14:08.
   - Still waiting at 05:29:
     - Chase pid 22944, since 04:32:30;
     - Cameron's runner pid 28492, since 04:41:35;
     - Foreman's `gpu_run.py long_aL4_s7`, since 04:57:42.
   - Wilson's chain re-takes the lock one second after it releases it. No 60-s poller has got in since 04:32:15.
2. **A foreign GPU process.** pid 29244 was on the card at 04:51:15, during Wilson's f_R_s1 hold. It is `C:\WINDOWS\system32\devicecensus.exe`, started by Task Scheduler (svchost Schedule), and was gone by 04:52:38. It is not a fellow's process.
3. **A bar written after its data.** bf16_leak_read1_vs_fp32 was written at 04:31:50, after run_bf16.log (04:31:18) had shown the leak values. Its docstring says so. It carries a new name and was RED on a stub first, as in pass 2 note 3.
4. **Stated times that disagree with mtimes.**
   - test_chase_k1_clause6_v2.py says "Registered ~04:40", but its mtime is 04:33:42 and its RED was 04:33:52.
   - test_chase_k1_ckpt_v2.py says "Registered ~04:10", but its mtime is 04:31:24, one second before its RED.
   - test_k1fppp.py still carries K1.F'' header text (seeds [31 / 32, n, k], bands 96 / 896, W62 unbarred) under its K1.F''' paragraph. Its checks assert bands 160 / 1280.
5. **x64's RED left no log file.** It exists only as the bar's own board line, 5712.
6. **A pass on inexact builds.** w30_far_band_long passes on n8192 rows whose builds have 587 / 1011 mismatches, because the check does not condition on exactness. This is recorded here and not ruled on.

## Lanes still in flight (cut at board 5742, about 04:52; state at 05:30)

- **Cameron, R-DEPTH on the far band.**
  - Runner pid 28492 has 10 jobs queued and has waited on gpu.lock.d since 04:41:35. No `runs/far_*` exists.
  - The bars are RED only: `rdepth_far_*` (56 rows) and `rdepth_far10_*` (36 UNREAD rows).
  - Line 5753 (GPU queue) landed after the cut.
- **Foreman.**
  - kp_trap: w30t_real_8192 and w62t_real_16384 are RED on their stub only. kp_trap.json does not exist.
  - kp_trap2: RED at 04:52:40. pid 19228 is still running, and its log shows L7 with 1,120 mismatches.
  - After the cut: w62p4096_real_16384 RED (5743); the w62a real-data REDs (5744–5746) and finding 5747; far_opp_* RED (5748–5751). `long_aL4_s7` is queued on the lock.
- **Chase, checkpoint bars.**
  - The seven `_v2` bars and clause6_v2 / clause6_v2_power are RED only.
  - pid 22944 has been blocked on the lock since 04:32:30. No checkpoint rows have been written.
- **Wilson.**
  - f_R_s1 landed (5752, after the cut): val 5.0166, ckpt.pt. It is not yet checked by `wilson.k1.ward`.
  - f_R_s2 has been running under the lock since 05:14:08.
  - `wilson.k1.ward` is RED only.
