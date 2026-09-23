# Inspector, Phase K it.K1, pass 2 (periodic, delta only)

38 claims audited, 9 struck. The audit lines sit at board lines 5566–5603 and `done` at 5604. The `dispatch` line "K1 pass 2" is line 5551.

**Delta covered:**
- Board lines 5506–5565: every fellow event after pass 1's `done`, up to the moment of append.
- Line 5459, Wilson's a_L_s1 finding. Pass 1 deferred it, and the dispatch asked for the ward runs to be checked against their logs.

**How the reruns ran:**
- On copies in `SP/phase_k/K1/inspector/rerun2/`, with `PYTHONDONTWRITEBYTECODE=1`.
- Nothing was written into any lane directory.
- No new `__pycache__` appeared, in the repo or in the lanes.

**The GPU lock was taken once**, to rerun Chase's three GPU bars.
- The rerun used a copy of test_chase_k1.py (c21e62) and resolvent_hook.py (9d0274), with `CHASE_BOARD=0`.
- The acquire was an atomic mkdir, with owner.txt reading "seat=Inspector".
- The lock was held from 03:56:31 to 03:57:20 and removed in `finally`. The wait was 284 s, behind Cameron's kp_aL7_s0.
- The board line count did not change during the rerun.
- Every other rerun was CPU work.

Nurse: N3 (sonnet) mapped every number in findings 5528 and 5540 to a checked row, a printed-only value, an unread field, or nothing.

## Dispatcher checks

| check | result |
|---|---|
| No R-DEPTH arm trained on bed_k after the ruling (Cameron) | **Holds.** `cameron/runs/` is empty. gpujob writes `runs/<job>.stdout` when a job starts, and no such file exists. `gpujob_twins.log` is 0 bytes. No `jobs_twins` runner or `rdepth.py` process was alive at 03:40. The queued `jobs_kp_twins.json` (03:38:44, pid 21648) passes `--bed kp` on all 8 jobs, and `rdepth.py` maps `kp` to `bed_kp`. |
| No R-DEPTH arm trained on bed_k after the ruling (other lanes) | **Two violations, both Foreman, both struck.** (1) `twin.py --L 4 --steps 10000`, pid 30776, started 03:31:58 after twin.py was edited at 03:31:45. It trains the (a_L) block on bed_k's train distribution, and `twin_kill_line` reads bed_k at n=4096. Log: `foreman/twin_L4.log`, 0 bytes. The process is waiting on gpu.lock.d and had trained 0 steps at 03:48. It was not killed. (2) `pilot_cpu2.py`, written 03:29:03: 400 CPU steps of `twin.Twin(4)` on bed_k's train distribution. Log: `foreman/pilot_cpu2_L4.log`, which ends 03:33:30. Its result was used to edit twin.py (orthogonal P init). |
| Ruling time | Not on the board. The header of Foreman's test_k1g says it was written "after the Dispatcher's K1 reconcile", with a clock read of 03:25:51. So the ruling came between pass 1's `done` (about 03:23) and 03:25:51. **`pilot_cpu.py` cannot be placed.** It was written at 03:24:01, and it ran from about 03:24:05 to 03:31:39 (log `foreman/pilot_cpu_L4.log`). It is recorded here and not struck. |
| K1.F' registered RED before its measurement | **Holds.** test_k1fp.py was written at 03:35:49. k1fp.py followed at 03:36:20. The RED ran at 03:36:29 on the `--stub` json, and its rows are identical to a stub rerun. The real k1fp.json is from 03:36:42. The design exploration (explore_kp*.py, 03:32–03:34) used seeds [91..94]. The registered beds are [21..24], and the test header discloses the exploration. No exploration number was posted. |
| K1.F' includes Foreman's HPD window | **Holds.** It includes `hpd_analytic`, a port of k1b.hpd_hybrid with P = 1/M. Row fp_machinery (a) reproduces Foreman's 0.5642224409448819 to 1e-12 on the K1.F beds [12,4096,k]. It also includes `hpd_empirical`. |
| Chase's GPU bars on hook 9d0274 | **Holds.** run2 hit its lock-wait timeout and exited 124 at 03:48:50 (`run2_aborted_lockwait.log`). run3 started at 03:49:14 and held the lock from 03:49:49. Its header gives hook 9d0274648a19…, and run3.sha256 records test c21e62 (unchanged) and hook 9d0274. resolvent_hook.py's mtime is still 02:58:43. The train bars' config `attn_sha256` is 9d0274. The Inspector GPU rerun reproduces all three GPU bars value for value. |
| Wilson ward runs vs logs | **Numbers exact; both findings struck as unbarred.** a_L_s1 and a_L_s2 match their `log.jsonl`, `ward_jobs.jsonl` and `ckpt.pt` exactly (val 4.997021734714508 and 4.984506642818451; 17,645 steps; 144,547,840 tokens; rc 0). `wilson.k1.ward` is RED only, and there is no ward_table.json. a_L_s2's job record lists `foreign_gpu_pids_seen` [20036, 30392], which its finding does not mention. |

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Cameron | R-DEPTH on bed_k cancelled before any arm started; 0 arms trained | none (process) | n/a | verified from runs/, logs and the process table | clean | The "03:27" stop time cannot be recovered from mtimes. |
| Cameron | K1.F' machinery | cameron.k1.fp_machinery | red_test_k1fp.log 03:36:29 on stub; test mtime 03:35:49 | k1fp.json byte-identical; GREEN exit 0; stub RED identical, exit 1 | clean | |
| Cameron | (a_L') line beyond 2^L: L4 0.3219/0.1495/0.1397/0.1313, L7 0.4085/0.2506/0.1817; every cell < 0.5 | cameron.k1.fp_not_void | as above | identical | clean | |
| Cameron | line = max of the 7 components | cameron.k1.fp_line | as above | identical | clean | |
| Cameron | beyond set not a sliver | cameron.k1.fp_deep | as above | 0.399–0.992 | clean | |
| Cameron | hand-set resolvent 1.0 at every test n | cameron.k1.fp_resolvent_feasible | as above | [1.0,1.0,1.0] ×3 | clean | |
| Cameron | bed_k' spec (generator unchanged; train M=32 cap 32; test M=8; V=32,768) | fp_machinery (d) + code | as above | bed_k.py sha aa60e6d9 = K0; constants 32, 8, 32 | clean | |
| Cameron | tightest cell per-bed max 0.4305; argmax hpd_empirical k3W10 | none | none | printed only | **struck** | The number appears in fp_not_void's printed output, but the check reads only the 8-bed mean. |
| Foreman | reach_doubling_is_one RED (L4 131.875; L6/L7 1.0) | foreman.k1.reach_doubling_is_one | red_test_k1g.log 03:26:09; test 03:26:04; k1g.py 03:26:28 | k1g.json byte-identical; exit 1 | clean | Reported as RED. |
| Foreman | reach_is_constant_factor RED (L4 131.875; 1.5/1.5/2.125) | foreman.k1.reach_is_constant_factor | as above | identical | clean | Reported as RED. |
| Foreman | c grows with width (2.125 vs 1.5) | foreman.k1.reach_grows_with_width | as above | identical | clean | |
| Foreman | far_band_defeats_all RED (4_R0: 0 tokens; 7_R0, 6_R1 pass) | foreman.k1.far_band_defeats_all | as above | identical | clean | Reported as RED. |
| Foreman | reach2_doubling_is_one RED (L4 0.0) | foreman.k1.reach2_doubling_is_one | red_test_k1h.log 03:27:29; test 03:27:21; k1h.py 03:27:30 | k1h.json byte-identical; exit 1 | clean | Reported as RED. |
| Foreman | reach2_constant_factor RED | foreman.k1.reach2_constant_factor | as above | identical | clean | Reported as RED. |
| Foreman | reach2_grows_with_width | foreman.k1.reach2_grows_with_width | as above | identical | clean | |
| Foreman | far_band2_defeats_all GREEN (board 5521) | foreman.k1.far_band2_defeats_all | as above | identical | **struck** | The 4_R0 row passed at kappa 0.0. That kappa came from reach2 c(HPD,4) = 0, and that row fails its own check. test_k1h has no void guard, and test_k1i's header calls the row vacuous. |
| Foreman | reach3_doubling_is_one_L4 (1.0) | foreman.k1.reach3_doubling_is_one_L4 | red_test_k1i.log 03:28:15; test 03:28:07; k1i.py 03:28:16 | k1i.json byte-identical; GREEN exit 0; stub RED exit 1 | clean | |
| Foreman | reach3_constant_factor_L4 (1.5) | foreman.k1.reach3_constant_factor_L4 | as above | identical | clean | |
| Foreman | far_band3_defeats_all_L4 (kappa 3, 0.0635 on 127,936) | foreman.k1.far_band3_defeats_all_L4 | as above | identical | clean | |
| Foreman (5528) | c(doubling) = 1.0 at L=4,6,7; c(HPD,R0) = 1.5 at L=4,6,7; c(HPD,6,R1) = 2.125; +0.625 with width | reach_*/reach2_* passing rows; k1i | as above | identical | clean | |
| Foreman (5528) | local no-oracle reach 1.0–1.125 at R0, 1.25 at R1 | none | none | fields reproduce | **struck** | The `local_*` / `local` fields are read by no test. kappa is computed in the producer and is not re-derived by any bar. |
| Foreman (5528) | beyond kappa·2^L every family ≤ 0.0637: (4,R0) 0.0635/127,936; (7,R0) 0.0637/106,432; (6,R1) 0.0636/113,600; kappa 3 / 4.25 | far_band3_defeats_all_L4; far_band_defeats_all 7_R0 and 6_R1 rows | as above | identical | clean | The `worst` field spans every HPD, local and doubling config. |
| Foreman (5528) | two L=4 registrations RED on definition slips; k1i the clean row | k1g, k1h, k1i | as above | as above | clean | The REDs are reported as RED. The tests were not edited, and the new names were RED before their producer existed. |
| Foreman | kp_attack_line_beaten: 0.4398 > 0.4185 at (4096, L7) | foreman.k1.kp_attack_line_beaten | red_test_kp_attack.log 03:37:59; test 03:37:55; kp_attack.py 03:38:51 | kp_attack.json byte-identical; exit 1 (the void row) | clean | |
| Foreman (5555) | kp_attack_void RED, so bed_k' survives. Cells 0.3339 / 0.1509 0.1412 0.1315 / 0.4398 0.2672 0.1893; margin 0.060; bed_kp sha 20d50b9d | foreman.k1.kp_attack_void | as above | identical; sha matches | clean | Reported as RED. The void row reads the max over all 7 cells. |
| Foreman | kp_no_leak: 0.1274 ≤ 0.1306 (N 31,680), planted 1.0 | foreman.k1.kp_no_leak | as above | identical | clean | |
| Foreman (5558) | sc-HPD k3 crosses 0.5 at W=14 at (4096, L7); 8k/16k 0.2672/0.1893 at W10 | foreman.k1.kp_pricing_crossing | red_test_kp_pricing.log 03:40:24.59; test 03:40:17; kp_pricing.py 03:40:24.65 | kp_pricing.json byte-identical; GREEN exit 0; stub RED exit 1 | clean | The 8k/16k values are kp_attack cells. |
| Foreman (5558) | W12 0.4768, W13 0.4963, W14 0.5108 | none | none | fields reproduce | **struck** | This is the `curve` field. test_kp_pricing reads only `crossing_W`. |
| Foreman (5558) | "at 9 dims per slot (W=14) bed_k' is void; the pricing convention is load-bearing" | none | none | n/a | **struck** | Arithmetic on d = 128. No bar reads a dims-per-slot pricing. |
| Foreman | twin_L4 run on bed_k | (process) | n/a | n/a | **struck** | It started after the ruling. Log `foreman/twin_L4.log`, pid 30776, 03:31:58. |
| Foreman | pilot_cpu2 twin run on bed_k | (process) | n/a | n/a | **struck** | It started after the ruling. Log `foreman/pilot_cpu2_L4.log`. |
| Wilson (5459) | a_L_s1: val 4.9970, 144,547,840 tokens, 17,645 steps, rc 0, seed 1, eval_seed 0, harness 4c00be… | wilson.k1.ward (RED only) | red_stub_ward.log 02:45:10 | log.jsonl val_loss 4.997021734714508 at step 17645; 17,645 step records; ckpt step 17645; ward_jobs rc 0; harness sha matches | **struck** | The numbers are exact, but no bar read them. There is no ward_table.json, and the finding itself says "not yet checked". |
| Wilson (5563) | yields the GPU; nothing of Wilson's queued on the lock | none (process) | n/a | ward_aL_chain.out: released 03:49:43, WARD_AL_DONE; the live Wilson processes are pollready.py (checks chase/READY, no mkdir) and waitfor.py | clean | |
| Wilson (5564) | a_L_s2: val 4.9845, 144,547,840 tokens, 17,645 steps, rc 0, seed 2 | wilson.k1.ward (RED only) | as above | val_loss 4.984506642818451 at 17645; ckpt step 17645; rc 0 | **struck** | Unbarred, as for s1. The job record also shows foreign GPU PIDs 20036 and 30392. |
| Chase (5560) | hook_gradcheck_S64_f64 GREEN (run3) | chase.k1.hook_gradcheck_S64_f64 | red_stub.log 02:48:13 (stub 6dd7c8); test 02:46:20 | CPU rerun on 9d0274: GREEN exit 0; mutants fail | clean | |
| Chase (5561) | hook_parity_S1024 RED | chase.k1.hook_parity_S1024 | as above | GPU rerun identical. y ≤ 3.7e-6 and dx/dw ≤ 2.3e-5 pass. da 1.9e-4–7.5e-4 and db 1.04e-4–6.9e-4 exceed 1e-4 in all 6 rows. | clean | Reported as RED. |
| Chase (5562) | hook_train_ladder_R0_50 GREEN | chase.k1.hook_train_ladder_R0_50 | as above | GPU rerun identical: rc 0; 50 steps; 10.8069 → 9.8628; eval 9.6314; only blocks.3 carries a/b; moved 0.009447 | clean | |
| Chase (5565) | hook_deterministic_R0_50 GREEN | chase.k1.hook_deterministic_R0_50 | as above | GPU rerun identical: losses and state_dicts bitwise equal; loss50 9.67980 | clean | |

## Counts

**38 audited, 9 struck.**

| fellow | audited | struck |
|---|---|---|
| Cameron | 8 | 1 |
| Foreman | 23 | 6 |
| Chase | 4 | 0 |
| Wilson | 3 | 2 |

## Same quantity, two numbers (reported, not picked)

The quantity is the (a_L') line beyond 2^L at (n=4096, L=7) on bed_k', with R0 pricing, credited, on Cameron's K1.F' seeds [22,4096,k] / [23,4096,k].
- **Cameron: 0.4085.** hpd_empirical k3W10 (k1fp.json). test_rdepth_kp's `twin_under_line` reads this value.
- **Foreman: 0.4398.** sc-HPD W10+0 (kp_attack.json).
- Both are bound. They are different window families.

## Checks with no hits

- **`status` vs `state`:** every test event in lines 5506–5565 uses `status`.
- **`hash()` seeding:** no `hash(` call in any lane .py changed since pass 1.
- **Exit codes:** every new bar file exits 1 on a failing row. This was verified on stub reruns of k1g, k1h, k1i, k1fp, kp_attack and kp_pricing, and on the real-data REDs of k1g, k1h and kp_attack. Chase's run3 exits 1 on the parity RED.
- **Bars edited after their RED:** none. Each test's mtime predates its RED log: k1g, k1h, k1i, kp_attack, kp_pricing, k1fp, rdepth_kp. test_twin.py is unchanged since 02:56:08.

## Process notes

1. **`twin.py` has a lock breaker.** Its `acquire()` removes gpu.lock.d when the lock is more than 3 h old and the owner's pid is dead. It then appends a board line. This is not part of the house GPU rule, and it has not fired.
2. **The kp queue has no fR arm.** `jobs_kp_twins.json` queues 8 jobs: aL7, aL4, ass4 and aloop5, seeds 0 and 1. It has no fR job. test_rdepth_kp's `learnable`, `prediction` and `multilen16k` rows read fR.
3. **Three registrations of one quantity.** The L=4 reach factor was registered three times (k1g, k1h, k1i). The second and third definitions were written after the previous one's real-data RED. Each is under a new name and was RED on a stub first.
4. **Repo bytecode.** No new repo bytecode. The only repo `__pycache__` is still pass 1's `K0/wilson/__pycache__/train_ladder.cpython-311.pyc` (02:46:36).

## Lanes still in flight (03:58)

- **Cameron, R-DEPTH on bed_k'.**
  - The 7 `cameron.k1.rdepth_kp_*` bars are RED only (stub 03:38:35). test_rdepth_kp.py is unchanged (sha 21995d…).
  - The first kp job ran 03:50:49–03:56:29. Its directory now carries the name `pilot_kp_aL7_s0_learnedE_lr1e-3`, and the kp_twins runner is gone.
  - rdepth.py (the producer) was edited at 03:57:20 to add `--emb`.
  - `gpujob.py jobs_pilot2.json` (pid 5396) holds the lock with `pilot_aL7_frozen_lr1e-3` on `--bed kp`.
  - No kp result is on the board.
- **Foreman, trained twin on bed_k.**
  - twin.py pid 30776 is still queued on the lock, with twin_L4.log at 0 bytes. Its run is struck above.
  - The 7 `foreman.k1.twin_*` bars are RED only.
- **Chase.**
  - A `diag_dab.py` job (pid 32692, 03:50:49) is queued on the lock. No board line is attached to it.
  - The checkpoint bars (bf16_leak, c6, c7, rrange_fR, rrange_aL_slope) are RED only.
- **Wilson, R0 LM ward.**
  - a_L_s1 and a_L_s2 are done.
  - `wilson.k1.ward` is RED only, and there is no ward_table.json.
  - pollready.py and waitfor.py are waiting for `chase/READY`.
