# Inspector, Phase K it.K1, pass 3 (periodic, delta only)

34 claims audited, 12 struck. The audit lines sit at board lines 5672–5705, and `done` is line 5709. The `dispatch` line "K1 pass 3" is line 5616. The nurse line is 5650.

**Delta covered:**
- Board lines 5605–5660: every fellow event after pass 2's `done` (5604), up to the cut.
- Lines 5661–5671 (Wilson nurse 3, Chase's checkpoint bars bf16_leak / c6 / c7 / rrange `_v2`) landed after the cut. They belong to pass 4.
- Lane files in `cameron/`, `chase/`, `foreman/` and `wilson/` modified after 03:58. This includes `FOREMAN_REPORT.md` (04:10:40).

**How the reruns ran:**
- On copies in `SP/phase_k/K1/inspector/rerun3/`, with `PYTHONDONTWRITEBYTECODE=1`.
- CPU reruns used `CUDA_VISIBLE_DEVICES=""`, and Chase's bars ran with `CHASE_BOARD=0`.
- Nothing was written into any lane directory. No new repo `__pycache__` appeared.

**The GPU lock was taken once**, to rerun Chase's three GPU bars on hook 3855288d.
- The acquire was an atomic mkdir, with owner.txt reading "seat=Inspector".
- The lock was held from 04:28:45 to 04:30:17 and removed in `finally`.
- The wait was 0 s. The rerun started only after Chase's own run4 had printed SUMMARY (04:28:39), so the Inspector's GREENs never came ahead of the fellow's.
- The rerun appended nothing to the board. The one board line that arrived during the hold is Chase's own READY finding.

Nurse: N4 (sonnet) mapped every number in findings 5612 and 5620 to a checked row, a build constant, an unread field, or nothing.

## Dispatcher checks

| ruling | result |
|---|---|
| (1a) Foreman cancels twin pid 30776 on bed_k; opponents only on bed_k' | **Holds.** pid 30776 is gone. `twin_L4.log` is 0 bytes, unchanged since 03:31:58. No twin file and no training process of Foreman's appears after 03:58. Foreman trained no opponent in the delta: kp_w14, kp_next, kp_w62 and kp_w62a are hand-set constructions. The lock was free by 04:00:53, when Cameron's next job stdout was born. The hold Foreman reports (04:00:04 to about 04:00:40) cannot be recovered from files. |
| (1b) Foreman binds, RED first, W = 14 realizable as real softmax heads at d = 128 on bed_k' | **Holds.** test_kp_w14.py (04:02:47) was RED at 04:02:51 on the stub (board 5608/5609). kp_w14.py was born at 04:03:25. GREEN came at 04:05:59 (5610/5611). The rerun reproduces everything: stub RED, 0 mismatches in f64 and f32, and 0.5107702349869452 == idealized. The json is equal except `wall_s`. **The width (15 heads × d_head 8 = 120, 57 ReLU units) is the build's tensor shape in code. No check row reads heads, d_head, attn_dims or mlp_units, and ≤ 128 is not asserted.** |
| (2a) Cameron starts no R-DEPTH arm until the W14 test binds | **Holds for registered arms.** No `runs/kp_*` or `runs/far_*` directory exists at the cut. `jobs_far_twins.json` (04:27:10, superseded) and `jobs_far_all.json` (04:30:11) queue `--bed kpf` → `runs/far_*`. Both came after the W14 GREEN (04:05:59) and after test_rdepth_far's RED (04:15:30). Unregistered aL7 **pilots on bed_kp** ran under the lock before the bind: 03:57:37–03:59:38, 04:00:53–04:02:53 and 04:03:26–04:06:08. More ran after it: 04:07:08–04:07:28 (OOM), 04:08:23–04:11:12 and 04:11:57–04:26:41. |
| (2b) Cameron registers a NEW name reading the max of both floor tables, without editing the RED'd one | **Holds.** test_rdepth_kp_line2.py (04:00:19, sha 8f5837) was RED at 04:00:22 (board 5606) with 12 UNREAD rows. Its line is max(k1fp `line.beyond`, kp_attack `best`), which is 0.4398 at (4096, L7). test_rdepth_kp.py is unchanged (21995d, 03:38:35). Both were later retired unrun, replaced by test_rdepth_far.py under Amendment K1-a (repo commit 49f5a2e). |
| (3) Chase's new hook: sha, bars GREEN on it with the test unchanged since its RED, READY only after | **Holds.** See the four items below. |

Ruling (3) in detail:
- **The hook.** resolvent_hook.py is sha 3855288d7da7…, mtime 04:01:32. Lines 45 and 63 form dP from `xc = x - x.mean(-2)`.
- **The test.** test_chase_k1.py is c21e62, mtime 02:46:20. It is unchanged since its stub RED at 02:48:13, and run4.sha256 (04:06:53) records both hashes.
- **The bars.** run4 went GREEN on 3855288d in this order:
  - gradcheck (CPU, before 04:10:11, board 5621);
  - then, under the lock from 04:27:11: parity, train_ladder and deterministic (5656–5658).
  - SUMMARY printed at 04:28:39.
  - The Inspector reran gradcheck on CPU (GREEN, stub RED) and the three GPU bars (identical value for value).
- **READY.** READY was born at 04:29:08.9, after SUMMARY, and it names 3855288d.

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Cameron (5605) | pilots on bed_k': kp run n=1024 acc 0.139, L-TRAINED fails; frozen lr1e-3 0.098 | none | n/a | stdout 0.1390, 0.0976 | **struck** | Self-declared unbarred. No bar has read a pilot run. |
| Cameron (5605) | pilot_kp_aL7_s0_learnedE_lr1e-3 is "not an R-DEPTH arm" | none (process) | n/a | gpujob_kp_twins.log "kp_aL7_s0 rc=0 340s"; result.json config out = runs/kp_aL7_s0 | **struck** | It launched as registered job kp_aL7_s0, into the path test_rdepth_kp.py reads. The directory and the jobs file were renamed after it finished. It ran 03:50:49–03:56:29, before ruling (2). test_rdepth_kp.py is now retired unrun. |
| Cameron (5606) | rdepth_kp_twin_under_maxline RED | cameron.k1.rdepth_kp_twin_under_maxline | red_test_rdepth_kp_line2.log 04:00:22; test 04:00:19 | 12 UNREAD, line 0.4398 at (4096, L7) | clean | Ruling (2b). |
| Cameron (5622) | lock breach: CPU pilots opened CUDA contexts 04:06:40–04:10:00 unlocked; locked pilot OOM 04:07:28 | none (process) | n/a | stdout ends "CUDA error: out of memory"; gpujob_pilot2.log rc=1 20 s; PIDs gone; 04:11:04 CPU pilots absent from nvidia-smi 04:11:31 | clean | The 04:06:40–04:10:00 contexts cannot be observed after the fact. |
| Cameron (5624) | frozen+orth lr3e-3: depth-2 0.06 → 0.92 (steps 500 → 750), depth 3–4 0.405 at 1500; learned-emb never learned depth 2 | none | n/a | by_depth_fine 0.0605 / 0.9194 / 0.4054 | **struck** | Unbarred. The learned-embedding run logs no depth-2 field. |
| Cameron | K1.F'' machinery | cameron.k1.fpp_machinery | red_test_k1fpp.log 04:13:56 on `--stub`; test 04:13:27; producer born 04:13:51; real json 04:14:35 | k1fpp.json byte-identical; GREEN exit 0; stub RED identical, exit 1 | clean | |
| Cameron | band ≥ 500 tokens | cameron.k1.fpp_band_size | as above | 26,560 / 59,328 / 124,864 (L4); 8,128 / 73,664 (L7) | clean | |
| Cameron | far-band line < 0.5 | cameron.k1.fpp_not_void | as above | 0.1270 / 0.1230 / 0.1239; 0.1104 / 0.1231 (argmax recency) | clean | |
| Cameron | line = max of families | cameron.k1.fpp_line | as above | to 1e-12, 5 cells | clean | |
| Cameron | hand-set resolvent ≥ 0.99 | cameron.k1.fpp_resolvent_feasible | as above | [1.0, 1.0, 1.0] × 5 | clean | |
| Cameron (5642) | K1.F'' values; reproduces 0.43982539164490864 | fpp_* rows | as above | identical | clean | |
| Cameron (5642) | sc-HPD W=62 scores 0.1220–0.1270 on the band | none | none | field `unbarred` reproduces | **struck** | No check reads it. |
| Cameron (5655) | recipe pilot 8k: n=1024 0.9949 (17–32 0.991); n=4096 1.000 / 0.169 / 0.026 | none | none | result.json reproduces | **struck** | Self-declared unbarred. |
| Cameron (5660) | CPU smoke of fR passes; params 4,999,556, non-emb 805,124 vs 805,120 | none | none | in no lane file | **struck** | No output file, no bar. |
| Foreman (5607) | twin_L4 cancelled before step 1; own lock dir removed | none (process) | n/a | see ruling (1a) | clean | |
| Foreman | kp_w14_heads_exact_f64 | foreman.k1.kp_w14_heads_exact_f64 | red_test_kp_w14.log 04:02:51; test 04:02:47; kp_w14.py born 04:03:25 | 0 mismatches, 0.5107702349869452; GREEN exit 0; stub RED exit 1 | clean | |
| Foreman | kp_w14_heads_exact_f32 | foreman.k1.kp_w14_heads_exact_f32 | as above | same in float32 | clean | |
| Foreman (5612) | bed_k' void at (4096, L7): W=14 as 15 × 8 = 120 dims + 57 ReLU, exact f64/f32, 0.5108 > 0.5; a slot costs 8 dims | kp_w14 rows | as above | identical | clean | The width is build shape, read by no check. |
| Foreman | w30_heads_exact_f32 | foreman.k1.w30_heads_exact_f32 | red_test_kp_next.log 04:06:41; test 04:06:37; kp_next.py born 04:07:03 | kp_next.json byte-identical; 0 mismatches; 0.8095626631853785; GREEN exit 0; stub RED exit 1 | clean | |
| Foreman | next_reach_bounded | foreman.k1.next_reach_bounded | as above | c 3.0 / 3.5 | clean | |
| Foreman | next_far_band | foreman.k1.next_far_band | as above | 0.12459 / 0.12448 on 127,968 / 102,368 | clean | |
| Foreman (5620) | W=30 realized: 31 × 4 = 124 dims + 121 units, fp32, exact, 0.8096 | w30 row | as above | identical | clean | 124 and 121 are build shapes, read by no check. |
| Foreman (5620) | c = 3.0 / 3.5; with κ = 2c every construction scores 0.1246 on 127,968 / 102,368 tokens | next_* rows | as above | identical | clean | The L7 value is 0.12448 (0.1245). κ = 2c is producer code. |
| Foreman (5620) | per-family c: W10 2.0 / 2.125, W14 2.25 / 2.375, local W30 1.625, doubling 1.0 | none | none | fields reproduce | **struck** | `per_family` is read by no check. |
| Foreman (5620, report) | with 8 lanes the L=7 arm needs n ≥ 16384 | none | none | n/a | **struck** | Arithmetic; no bar. |
| Foreman (5651) | W=62: exact at (4096, L4); c(W62) 4.25 / 5.0; far band 0.1246 / 0.1244 on 126,688 / 90,080 | foreman.k1.w62_* (RED on stub only) | red_test_kp_w62.log 04:14:28 | printed in kp_w62.log only; no kp_w62.json | **struck** | Not bound when posted. |
| Foreman (report) | bed_k' "does not survive at 8k / 16k under real-head pricing" | none | none | n/a | **struck** | Real-head builds are bound at n = 4096 only, and the report itself lists W = 30 at 8k/16k as open. |
| Chase (5621) | hook_gradcheck_S64_f64 GREEN on 3855288d | chase.k1.hook_gradcheck_S64_f64 | red_stub.log 02:48:13 (stub 6dd7c8); test 02:46:20 | CPU rerun GREEN exit 0; mutants fail; stub RED exit 1 | clean | |
| Chase (5656) | hook_parity_S1024 GREEN on 3855288d | chase.k1.hook_parity_S1024 | as above; also RED on 9d0274 (run3) | y ≤ 3.69e-6, dx/dw ≤ 1.50e-6, da ≤ 5.21e-5, db ≤ 5.11e-5; GPU rerun identical | clean | |
| Chase (5657) | hook_train_ladder_R0_50 GREEN on 3855288d | chase.k1.hook_train_ladder_R0_50 | as above | rc 0; 10.8069 → 9.8628; eval 9.6314; attn_sha256 3855288d; moved 0.009454; identical | clean | |
| Chase (5658) | hook_deterministic_R0_50 GREEN on 3855288d | chase.k1.hook_deterministic_R0_50 | as above | bitwise equal; loss50 9.67982; identical | clean | |
| Chase (5659) | READY values; K0 backward misses da/db at 7.5e-4 | run4 rows; run3 RED | as above | identical | clean | Ruling (3). |
| Chase (5659) | parity bitwise identical with autocast on and off | none | none | amp0 = amp1 as printed | **struck** | No check compares the two rows. |
| Chase (5659) | cause: u ~ 2e4 gx cancels in fp32; centring takes dq 2.7e-4 → 1.4e-5 | none | none | diag_dq_v6.log 2.71e-4 → 1.37e-5 (seed 1) | **struck** | These come from diagnostics, not a bar. |

## Counts

**34 audited, 12 struck.**

| fellow | audited | struck |
|---|---|---|
| Cameron | 14 | 6 |
| Foreman | 13 | 4 |
| Chase | 7 | 2 |
| Wilson | 0 | 0 |

## Checks with no hits

- **`status` vs `state`:** all 39 test events in 5605–5660 use `status`.
- **`hash()` seeding:** none in any lane .py changed after 03:58.
- **Exit codes:** every new bar file exits 1 on a failing row. This was verified on the stub reruns of kp_w14, kp_next, k1fpp and Chase's gradcheck. For line2, rdepth_far, kp_w62 and kp_w62a it was checked in code.
- **Bars edited after their RED:** none. Each test's mtime precedes its RED log: kp_w14, kp_next, kp_w62, kp_w62a, k1fpp, rdepth_kp_line2 and rdepth_far. test_rdepth_kp.py (21995d) and test_chase_k1.py (c21e62) are unchanged.
- **Producer before RED:** none. Every new producer's birth time comes after its test file. The one producer born before its RED log is k1fpp.py, which wrote the `--stub` json that the RED ran on; the real json followed.

## Process notes

1. **Pilots choose the recipe on the gate's own beds.** The pilots evaluate on the held-out n=1024 beds `default_rng([21, k])`. These are the same beds the L-TRAINED gate of test_rdepth_far reads: rdepth.py maps `kpf` to held-out seed 21. The recipe was fixed on them (5655): frozen codes, orthogonal init, lr 3e-3, 8000 steps.
2. **Chase's diagnostics hold no lock of their own.** diag_dab.py and diag_dq.py run on CUDA and contain no lock call. Whether a wrapper held gpu.lock.d cannot be established from files. diag_dq.log (V1–V5) is stamped 04:00:45, just after the twin hold Foreman reports ended (about 04:00:40). The V6 run (04:06:34–04:06:37) falls in a gap between Cameron's jobs (04:06:08–04:07:08).
3. **Lane bytecode, not repo.** `chase/__pycache__/resolvent_hook.cpython-311.pyc` was written at 04:06:34. No new repo bytecode: K0/chase and K0/wilson `__pycache__` are unchanged since 02:56 / 02:46.
4. **Lock occupancy.** From 03:57:37 to 04:26:41, Cameron's pilot runners held the lock for all but a few gaps. Chase's run4 waited on the lock from about 04:10 until it took it at 04:27:11.
5. **Posting after `done`.** Foreman posted `done` at 5623, then registered and posted again (5630–5636, 5651–5654).
6. **An unread field that runs backwards.** In kp_w14.json, f32's `max_prerounding_err` (3.2e-10) is smaller than f64's (1.4e-5). It is recorded here and not ruled on.

## Lanes still in flight (cut at board 5660, 04:31)

- **Cameron, R-DEPTH on the far band.**
  - The 7 `cameron.k1.rdepth_far_*` bars are RED only (04:15:30, 56 rows).
  - `jobs_far_all.json` queues 10 arms on `--bed kpf`: fR and the aL4 / aL7 / ass4 / aloop5 arms, seeds 0 and 1.
  - No `runs/far_*` exists yet.
- **Foreman, W=62.**
  - `foreman.k1.w62_*` (7 rows) and `w62a_*` (3 rows) are RED only.
  - kp_w62.py (pid 4592) is still running at 04:33, and no kp_w62.json has been written.
  - Its log shows w62_8192 at mismatch rate 2.3e-4, and w30 at (8192, L7) with 1,011 mismatches.
- **Chase, checkpoint bars.** Chase holds gpu.lock.d from 04:31:08 ("ckpt bars"). Lines 5662–5671 (bf16_leak, c6, c7 and rrange `_v2`) landed after the cut.
- **Wilson.**
  - pollready read READY and exited (POLL_EXIT=0).
  - `wilson.k1.ward` is RED only, and there is no ward_table.json.
  - A nurse was dispatched at line 5661.
