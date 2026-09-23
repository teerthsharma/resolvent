# Inspector, Phase K it.K1, pass 5 (periodic, delta only)

47 claims audited, 6 struck. The audit lines are board lines 5819–5865, and `done` follows them. The `dispatch` line "K1 pass 5" is line 5789. The nurse lines are 5794 (N6) and 5795 (N7). The cut is line 5817, Chase's `done` at about 06:24.

**Delta covered:**
- Board lines 5743–5753. Pass 4 left them to this pass because they landed after its cut. They include Wilson's f_R_s1 finding (5752, val 5.0166).
- Every fellow event from 5790 to 5817.
- Lane files in `cameron/`, `chase/`, `foreman/` and `wilson/` modified after 05:16, up to the cut. Later files are used only for the ruling checks.
- Line 5818 (Foreman, "GPU priority race, mine") landed after the cut. It is used only under ruling (b) and belongs to pass 6.

**How the reruns ran:**
- All reruns ran on copies in `SP/phase_k/K1/inspector/rerun5/`, with `PYTHONDONTWRITEBYTECODE=1`. CPU reruns used `CUDA_VISIBLE_DEVICES=""`.
- **Chase.** The reruns ran with `CHASE_BOARD=0`. In the copy only, `gpulock.py` was replaced by a no-op context manager, because the Inspector held the real lock around the process. The bars and `instr2.py` are byte-identical to the lane. The board line count stayed at 5817 across every Chase rerun.
- **Foreman, kp_w62a.py.** Two copies of this rerun ran at once. The first launch (05:38:46) was missed in a process listing and relaunched at 05:42:11. Both wrote the same kp_w62a.json, byte-identical to the lane's. The first took 36m13s.
- **Foreman, kp_trap2.py.** The rerun took 67m57s, against 38m50s for the fellow. It shared the CPU with Foreman's two kp_trap.py processes and the Inspector's two kp_w62a.py processes.
- **Wilson's ward.** The producer (ward_table_k1.py) and the bar ran on a copy of `runs/` that includes the four ckpt.pt files.
- **Lane and repo writes.** Nothing was written into any lane directory. `git status` is the same as at session start. The only repo file modified since 05:33 is house-events.jsonl, and no repo `.pyc` was written.

**The GPU lock was taken once.**
- The Inspector held it from 06:34:53 to 06:35:31 (38 s), with owner.txt written and the directory removed in `finally`, for the GPU rerun of Chase's checkpoint bars.
- The waiter polled by atomic mkdir every 5 s from 06:28:02. It acquired 1 s after Cameron's release at 06:34:52, inside Cameron's own mandatory 180-s wait.
- The Inspector had not released the lock before this, so the 3-minute rule did not apply to it.

**Nurses:**
- **N6 (sonnet)** mapped every number in 5747, 5752 and 5753. It found:
  - 782k where the row says 783,878;
  - 05:13 where the chain log says 05:14:07;
  - no source for "since 04:48".
- **N7 (sonnet)** mapped 5790, 5792 and 5793. It found:
  - Wilson's last_release.txt is absent;
  - Foreman's earlier runner was cancelled only by absence: no kill record exists.

  N7 also reported two things the Inspector checked and does not take:
  - N7 said no W62 L4 n8192 realization row exists. The row exists: `w62a_real_8192 L=4` PASS.
  - N7 matched "4x an arm" only against the test header, not against the queued arms.

## Dispatcher checks

| ruling | result |
|---|---|
| (a) Cameron scores far-band arms only on far10 (depth > 160 at L4, > 1280 at L7), never on 137–160; his runner re-checks the board for a newer c_max after acquiring the lock and before each arm | **Holds. No arm has been scored.** The runner code was fixed before any arm launched. See the details below the table. |
| (b) GPU fairness: no lane re-acquires within 3 min of its own release; priority is Cameron's far arms → Chase's checkpoint reader → Foreman's opponent and Wilson's ward | **The 3-minute rule holds. The priority order was inverted twice.** See the lock timeline below. |
| Runs keep launch names; pilots are named `pilot_` at launch | **Holds.** far_fR_s0 and far_aL4_s0 ran under their jobs-file names and directories. `pilot_fR_ganneal_s0` is named as a pilot in jobs_far_all2.json and has not launched. Foreman's long_aL4_s7 ran under the name its bar reads. Wilson's f_R_s2 ran under its spec name. |
| RED before measurement | **Holds.** The details are below the table. |
| K1-a band rule | **No breach.** No Cameron band score has been posted. Foreman's far_opp_quiet and far_opp_chance use 160 / 1280. Process note 8 records a band question in far_opp_plateau. |

Ruling (a) details:
- **The runner.** gpujob.py was edited at 05:39:51. `run()` now calls `new_cmax(start)` after `os.mkdir` and before owner.txt and the arm. It sleeps 180 s after each release.
- **The old runner.** pid 28492 used the 04:40:14 code, which has no re-check after acquiring. It was gone by 05:41 and never held the lock.
- **The new runners.**
  - Runner pid 30616 started at 05:40:02 with start line 5795. It ran far_fR_s0 from 05:57:03 to 06:16:38 and far_aL4_s0 from 06:19:39 to 06:34:52. No Foreman finding lay after 5795 at either acquire.
  - Runner pid 30732 started at 06:38:02 with `jobs_far_all2.json` and start line 5817. It runs the same gpujob.py.
- **Band scoring.** test_rdepth_far10.py uses 160 / 320 / 1280. The band rows of test_rdepth_far.py (96 / 224 / 896) were retired at 5796, but only by that board line. The file is unchanged since 04:15:25, and rdepth.py's result.json still carries band4 > 96 and band7 > 896 fields.
- **Scores so far.** far_fR_s0 (acc at n1024 0.2026) and far_aL4_s0 have landed with no Cameron claim, and test_rdepth_far10.py has not run.

Ruling (b) details:
- **Inversion 1, at 05:55:32.** Chase's reader, pid 22944, took the lock 5.6 s after Wilson's release, while Cameron's runner pid 30616 had been polling since 05:40:02. That process loaded gpulock.py at 04:32:30 (pyc 02:46:44), and that version has no yield. The gpulock.py edit at 05:33:50 adds a 180-s gap after the process's own release. It adds no yield to Cameron.
- **Inversion 2, at 06:36:19.** Foreman's gpu_run2 took the lock for long_aL4_s7. Cameron had stopped runner 30616 after its 06:34:52 release and restarted it at 06:38:02, so no `jobs_far` process existed at that moment. The 45-min timeout means the run ends by 07:21. Foreman self-reports this at 5818, after the cut.
- **The 3-minute rule.** No lane re-acquired within 3 min of its own release:
  - Wilson's chain ended at WARD_FR_DONE.
  - Chase re-acquired 20 min 17 s after its own release.
  - Cameron re-acquired 180.1 s after its release, at 06:19:39.

RED-before-measurement details:
- The seven `_v2` checkpoint bars were RED at 04:31:25. clause6_v2 and clause6_v2_power were RED at 04:33:52.9. The first checkpoint read was at 05:55:32.
- w62p4096 was RED at 04:52:40, and kp_trap2.py was born at 04:52:41.
- w62a was RED at 04:19:50, and kp_w62a.py was born at 04:20:05.
- far_opp_* was RED at 04:57:28, and no long_* run existed before 06:36:19.
- rrange_fR_dressed_R0 was RED on its stub at 06:18:39.9, before its real run at 06:18:40.9. It was written after its data; see process note 3.

Lock timeline since the ruling (lockwatch5.log and lockwatch5b.log, 1-s poll, read-only):

| time | owner.txt |
|---|---|
| 05:14:08 – 05:55:26.997 | Wilson pid 1320, ward_f_R_s2 |
| 05:55:32.595 – 05:56:39.141 | Chase pid 22944, ckpt bars v2 |
| 05:57:04.019 – 06:16:38.992 | Cameron pid 30616, far_fR_s0 |
| 06:16:56.285 – 06:17:41.573 | Chase pid 29548, clause6_v2 + ckpt v2 |
| 06:19:39.069 – 06:34:52.468 | Cameron pid 30616, far_aL4_s0 (the directory existed 1 s before owner.txt: re-check window) |
| 06:34:54.534 – 06:35:31.312 | Inspector pid 32532, pass 5 rerun (copy) |
| 06:36:20.016 – (held at 06:50) | Foreman pid 33648, long_aL4_s7 |

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Foreman (5743) | w62p4096_real_16384 registered RED | foreman.k1.w62p4096_real_16384 | red_test_kp_trap2.log 04:52:38; test f5ed8e, mtime 04:52:28 | stub RED identical, exit 1 | clean | kp_trap2.py was born at 04:52:41 and launched at 04:56:03, both after the RED. |
| Foreman (5744) | w62a_real_16384 RED: L4 0, L7 0.1052 | foreman.k1.w62a_real_16384 | red_test_kp_w62a.log 04:19:50 (board 5652); test 7f4611, 04:19:45 | kp_w62a.json byte-identical; test output identical, exit 1 | clean | Reported as RED. |
| Foreman (5745) | w62a_real_8192 RED: L4 0, L7 0.0917 | foreman.k1.w62a_real_8192 | as above | as above | clean | |
| Foreman (5746) | w62a_no_alias RED: L7 783,878 / 701,938 | foreman.k1.w62a_no_alias | as above | as above | clean | |
| Foreman (5747) | W62 realizes at L4 at n8192 and n16384: 0 mismatches, 0 alias violations | w62a L4 rows | as above | PASS rows identical | clean | Slope and period are producer config. |
| Foreman (5747) | c_max(L4) = 4.25 at every test n; band4 > 136 | w62_reach, w62_real_4096, w62a L4 | red_test_kp_w62.log 04:14:28 and above | identical | clean | 136 = 2 × 4.25 × 2^4. |
| Foreman (5747) | L7 period-2048 premise fails: 782k / 702k, 0.105 / 0.092, RED | w62a L7 rows | as above | 783,878 / 701,938; 0.10521 / 0.09169 | **struck** | 782k matches no row. The row prints 783,878, which is 784k. |
| Foreman (5747) | period 4096 registered and running; band7 stays > 896 (c_max 3.5) | w62p4096; w62_reach base; kp_next | 04:52:38 | stub RED identical | clean | 896 = 2 × 3.5 × 2^7. |
| Foreman (5748–5751) | far_opp_gate / quiet / chance / plateau registered RED (4 lines) | foreman.k1.far_opp_* | red_test_far_opponent.log 04:57:28 on runs_stub; test ab020d, 04:57:20 | 12 failing rows identical, exit 1 | clean ×4 | No long_* run existed before 06:36:19. |
| Foreman (5791) | w62p4096_real_16384 RED: L4 0 (30,576 band tokens, 0.1248); L7 1,120 | foreman.k1.w62p4096_real_16384 | 04:52:38 | kp_trap2.json byte-identical (67m57s rerun); test output identical, exit 1 | clean | Reported as RED. |
| Foreman (5792) | L7 does not realize at n16384 with either code (0.105; 1120 full chain, 2 beds; beta 3e7) | w62a L7, w62p4096 L7 | as above | identical | clean | |
| Foreman (5792) | L4 period-4096 exact at n16384; depth > 136 scores 0.1248 on 30,576 | w62p4096 L4 | as above | identical | clean | |
| Foreman (5792) | standing: c_max(L4) 4.25 at 4096 / 8192 / 16384; c_max(L7) 3.5; W62 (5.0) unrealized at 16384; far10 stands | w62_*, w62a, w62p4096, w30_real_long, fppp_machinery | as above | identical | clean | 160 > 136 and 1280 > 896. |
| Foreman (5792) | long_aL4_s7 re-queued; earlier runner cancelled without acquiring | none needed (process record) | — | no gpu_run.py process; queue.log 0 bytes; no Foreman owner before 06:36:19 | clean | |
| Foreman (5793) | gpu_run2 mechanics: 60-s poll, priority gate, 3-min gap, one subprocess, 45-min timeout | code | — | gpu_run2.py 05:35:34 | clean | The gate also matches Wilson's LM runs, whose --attn path contains `chase`. |
| Foreman (5793) | recipe; 131M tokens | process argv | — | pid 33648 argv verbatim; 16000 × 8192 | clean | |
| Foreman (5793) | the budget is 4x an arm's | none | none | the queued arms run 8000 steps × 8192 tokens = 65.5M | **struck** | No check reads a budget. Against the queued arms, 131M is 2x. |
| Foreman (5793) | aL7 long run not queued; its rows stay unmeasured | — | — | no process or directory | clean | |
| Wilson (5752) | f_R_s1 landed: sha 3855288d, seed 1, eval_seed 0, 144,547,840 tokens, 17,645 steps, rc 0, val 5.0166, ckpt | wilson.k1.ward f_R_s1 row | red_stub_ward.log 02:45:10; test 7c9bcf, 02:45:06 | ward rerun identical; ckpt step 17645 on CPU load; sha recomputed | clean | Bound after posting by the 05:56 ward run. "Same data order" is config only. The row fails done_clean (DeviceCensus.exe), which the landing line does not mention. |
| Wilson (5790) | 180-s re-acquire gap in gpujob.py; f_R_s2 launched before the edit; nothing more queued; no a_ss / a_loop hook | code and process record | — | gpujob.py 05:34:58; pid 1320 started 05:14:07; chain ended 05:55:26; ATTENTION = {alibi} | clean | last_release.txt does not exist: the pre-edit process writes none. |
| Wilson (5797) | f_R_s2 landed: val 5.0082, rc 0, 144,547,840 tokens | ward f_R_s2 row | as above | all 17 checks pass; identical | clean | |
| Wilson (5798/5799) | wilson.k1.ward RED, done_clean on a_L_s2 (20036, 30392) and f_R_s1 (29244) | wilson.k1.ward | as above | stub RED identical (26 fails, 17/17 fired); table identical modulo paths; bar output identical, exit 1 | clean | |
| Wilson (5799) | 15 of 17 checks pass on all rows | wilson.k1.ward | as above | 1/17 fired | **struck** | Self-corrected at 5800. |
| Wilson (5800) | correction: 1 of 17 fired; 16 pass on all 4 rows | wilson.k1.ward | as above | identical | clean | |
| Wilson (5799) | val 4.997022 / 4.984507 / 5.016564 / 5.008180; delta +0.019543 / +0.023673; tok/s; hours | done_val, delta, done_tok_s, done_hours | as above | identical | clean | The 6-decimal digits are the eval records that done_val binds at 5e-5. |
| Wilson (5799) | mean delta 0.021608, abs(d1 − d2) 0.004130, seed spreads 0.012515 / 0.008385 | none | none | arithmetic reproduces | **struck** | No check computes them. |
| Wilson (5799/5801) | a_L_s2 foreign PIDs exited, unidentified; dip 82.8–84.3k vs 88–91k, corrected to 87.9–92.9k | none | none | block medians reproduce 82,842–84,286 and 87,927–92,923 | **struck** | No check reads block rates. done_spill did not fire. |
| Wilson (5799) | a_ss / a_loop not run (no hook); no R-CARRY verdict | code | — | ATTENTION = {alibi}; SSMaxAlibi only in rdepth.py | clean | |
| Chase (5803) | c6_numerics_R0_v2 RED 3/4 | chase.k1.c6_numerics_R0_v2 | red_stub_ckpt_v2.log 04:31:25; test 9ff0f2, 04:31:24 | GPU rerun identical, value for value | clean | |
| Chase (5804) | c6_power_R0_v2 GREEN: 1.0018 max, 4/4 | chase.k1.c6_power_R0_v2 | as above | identical | clean | |
| Chase (5805) | c7_bos_R0_v2 GREEN | chase.k1.c7_bos_R0_v2 | as above | identical | clean | |
| Chase (5806) | rrange_fR_depth_R0_v2 GREEN (range inf, need 2075) | chase.k1.rrange_fR_depth_R0_v2 | as above | identical | clean | |
| Chase (5807) | rrange_fR_span_R0_v2 GREEN (range inf, need 16378) | chase.k1.rrange_fR_span_R0_v2 | as above | identical | clean | |
| Chase (5808) | rrange_aL_slope_R0_v2 RED on 8 checkpoints | chase.k1.rrange_aL_slope_R0_v2 | as above | 8 common checkpoints identical, plus far_aL4_s0 as a 9th; RED | clean | |
| Chase (5809) | clause6_v2 GREEN 4/4 | chase.k1.clause6_v2 | red_stub_clause6_v2.log 04:33:52.9; test 5aad42 | GPU rerun identical | clean | |
| Chase (5810) | clause6_v2_power GREEN 4/4 | chase.k1.clause6_v2_power | as above | identical | clean | |
| Chase (5811/5812) | rrange_fR_dressed_R0 RED on stub, then RED (0 checkpoints meet the premise) | chase.k1.rrange_fR_dressed_R0 | red_stub_rrange_v3.log 06:18:39.9; test 61156a, born 06:18:32 | stub RED identical; CPU rerun on both row sets identical, exit 1 | clean | Written after its data; see process note 3. |
| Chase (5813) | clause 6: 0.99900036; ratios 1.67 / 1.37 / 0.82 / 0.62; fs5 at 12–439x; bf16 at 1.0018; the 1e-5 line RED 3/4 with read / floor pairs | clause6_v2, _power, c6_* rows | as above | identical | clean | |
| Chase (5814) | clause 7: BOS < 0.5, max 0.187; LM heads ≤ 0.029; route not triggered | c7_bos_R0_v2, c7_route (SKIP) | as above | identical | clean | |
| Chase (5815) | depth / span GREEN but vacuous: acc 0.2026; ranges inf / 18,247 and inf / inf; dressed RED with 0 premise | rrange_v3 row | as above | identical | clean | |
| Chase (5815) | aL slope RED: Wilson 16/16 within 2x; Cameron pilots 36/84 out of band | rrange_aL_slope row | as above | recount: 16/16 and 48/84 in band | clean | |
| Chase (5815) | content sets the pilots' range | none | none | — | **struck** | No check tests what sets a head's range. |
| Cameron (5753) | runner has not acquired the lock; Wilson re-took it at 05:14:08; no arm started; 10 jobs | chain log, jobs file | — | as stated | clean | "Since 04:48" and "05:13" differ from the runner's start (04:41:35) and the release (05:14:07). "12–18 min" is an estimate: the arms ran 1175 s and 913 s. |
| Cameron (5796) | PASS4 rules applied: far10 only; test_rdepth_far band rows retired; re-check after acquire; 180-s wait | code and lock record | — | gpujob.py 05:39:51; acquires at 05:57:03 and 06:19:39 | clean | The retirement is recorded on the board only; the file is unchanged. |

## Counts

**47 audited, 6 struck.**

| fellow | audited | struck |
|---|---|---|
| Foreman | 21 | 2 |
| Chase | 14 | 1 |
| Wilson | 10 | 3 |
| Cameron | 2 | 0 |

## Checks with no hits

- **`status` vs `state`:** every test event in 5743–5817 uses `status`.
- **`hash()` seeding:** none in any lane .py changed after 05:16:
  - cameron: diag_fr.py, gpujob.py, rdepth.py
  - chase: gpulock.py, test_chase_k1_rrange_v3.py
  - foreman: gpu_run2.py
  - wilson: gpujob.py
- **Exit codes:** every new bar exits 1 on a failing row. This was checked for test_kp_w62a, test_kp_trap2, test_far_opponent, test_ward_k1, test_chase_k1_rrange_v3, test_chase_k1_clause6_v2 and test_chase_k1_ckpt_v2.
- **Bars edited after their RED:** none. The pairs of test mtime and RED time are:

  | bar | test mtime | RED |
  |---|---|---|
  | kp_w62a | 04:19:45 | 04:19:50 |
  | kp_trap2 | 04:52:28 | 04:52:38 |
  | far_opponent | 04:57:20 | 04:57:28 |
  | rrange_v3 | 06:18:32 | 06:18:39.9 |

  test_ward_k1.py (02:45:06), test_chase_k1_ckpt_v2.py (04:31:24) and test_chase_k1_clause6_v2.py (04:33:42) are unchanged since their REDs.
- **Producer before RED:** none. kp_trap2.py was born at 04:52:41 and kp_w62a.py at 04:20:05, both after their REDs. ward_table_k1.py (02:45:47) is unchanged, and its RED was at 02:45:10 on a built stub.

## Process notes

1. **Lock occupancy.** See the timeline above. The lock went to the first poller after each release. Only Foreman's gpu_run2 yields in code, and its gate looks for a live process, not a queue.
2. **A producer edited between the arms of one bar.**
   - Cameron's rdepth.py was edited at 06:18:18. far_fR_s0 ran on the 04:40:14 file. far_aL4_s0 (06:19:39) and Foreman's long_aL4_s7 (06:36:19) run on the edited file.
   - The diff against the pass-4 copy adds an opt-in `--gamma_sched` path and sets `HOOK["mod"]` in `load_hook`. It is inert without the flag. This was established by reading the diff, not by a rerun.
3. **A bar written after its data.**
   - test_chase_k1_rrange_v3.py was born at 06:18:32, 53 s after ckpt_rows_v2.jsonl (06:17:39) held the far_fR_s0 rows it reads. It carries a new name and was RED on a stub first.
   - Its header says "Registered ~06:25". It also says it was written before res_hop_m was looked at, which cannot be checked.
4. **Stated times that disagree with the record.**
   - rrange_v3's "~06:25" (see note 3).
   - Cameron 5753 says "since 04:48"; the runner started at 04:41:35. It says "04:36–05:13"; the release was at 05:14:07.
5. **Two copies of one producer.** Two Foreman kp_trap.py processes run at once: pid 16444, from the 04:56:03 chain (started 05:34:54), and pid 27516, started 05:35:13. Both write kp_trap.log and kp_trap.json in the lane. kp_trap.json does not exist at 06:50.
6. **Cooldown files.** Wilson's last_release.txt does not exist, because the pre-edit process writes none. Chase's .gpu_last_release exists (06:17:40) and was written by pid 29548.
7. **A result read before its owner posted it.** Chase read Cameron's far_fR_s0 checkpoint and result.json (acc at n1024 0.2026) 18 s after it landed. Cameron has posted nothing on that arm.
8. **A band read inside K1-a's band.** Foreman's far_opp_plateau reads the probe's band4 (depth > 96) and band7 (depth > 896). Its header calls 96 "beyond every bound construction at L = 4", but after 5729 K1-a's band4 is depth > 136. This is recorded here and not ruled on.
9. **The Inspector's own duplicate rerun.** Two kp_w62a.py reruns of the Inspector's ran on CPU from 05:42 to 06:18 alongside the lanes. Both produced byte-identical output.

## Lanes still in flight (cut at board 5817, about 06:24; state at 06:50)

- **Cameron.**
  - Runner pid 30732 (06:38:02, `jobs_far_all2.json`, start line 5817) waits behind Foreman.
  - far_fR_s0 (rc 0, 1175 s) and far_aL4_s0 (rc 0, 913 s) have landed, each with model.pt, result.json and ok_<n>.npz. Cameron has made no board claim on either, and test_rdepth_far10.py has not run.
  - `pilot_fR_ganneal_s0` is queued sixth.
  - CAMERON_REPORT.md is still a draft.
- **Foreman.**
  - long_aL4_s7 has held gpu.lock.d since 06:36:19, with a 45-min timeout.
  - far_opp_* is RED only.
  - Two kp_trap.py processes are running. w30t_real_8192 and w62t_real_16384 are RED on their stub only, and kp_trap.json does not exist.
  - Line 5818 landed after the cut.
- **Chase.** Done at 5817. rrange_fR_dressed_R0 and rrange_aL_slope_R0_v2 are RED, and c7_route_gamma_h_v2 was skipped.
- **Wilson.** Done at 5802. wilson.k1.ward is RED on done_clean.
