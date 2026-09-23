# Inspector, Phase K it.K2, pass 1

76 claims audited this pass, 13 struck: Cameron 40 / 9, Chase 30 / 3, RECORD_K 6 / 1.

Scope: board lines 6054–6174 (after the K1 prognosis at 6053), the lanes `K2/cameron/` and `K2/chase/`, the stray `K2/verify_sp.py`, and RECORD_K's "K2, as registered here" and "K2.0 result, and K2.1 as registered" (commit 8e1bd52). Board: `dispatch` 6159, nurse lines 6171–6172 (N1, N2, sonnet), audit lines 6175–6250, `done` 6251.

## Priority items

| item | result |
|---|---|
| Cameron K2.0 bars | **All reproduce byte-identical** on CPU copies. test_k2_leak.py (sha 0bc8ea17): RED 1 37 fails, RED 2 25, measurement 22, exit 1 each. test_k2_walk.py (sha af19b1b6): 6 / 7 / 5, exit 1. RED precedes measurement in both: leak test 10:09:13, RED 10:09:28 / 10:09:42, producer 10:10:40, first cell 10:10:48; walk test 10:13:40, RED 10:13:52 / 10:13:53, producer 10:14:12, measurement 10:14:16. |
| test_k2_walk.py vs K2.0 | **The file does not re-grade K2.0.** It scores no leak_* or gamma_* row, and no K2.0 status changed on the board, in the report or in RECORD_K. The re-grade is in prose. "Leakage is most of the 4k wall" reads K2.0's RED prediction (0.8750 < 0.9) together with the walk ceiling as a partial pass. No row reads "most". It is struck in RECORD_K, in the commit title of 8e1bd52, and in the report's verdict. |
| K2.1 registrations | **RED before any run; shas unchanged.** pilot 9996d651 (10:21:12), grid a622cd95 (10:21:52), lanes 916ffa6e (10:22:26). RED 1 ran at 10:22:42, RED 2 at 10:23:12, and the pilot launched at 10:42:32. The CPU rerun of RED 1 is identical for all four (13, 133, 133 and 4 fails). RED 2 is identical for lanes (3 fails). It differs for the pilot and grid stubs only where they point at the live `chase/resolvent_sp.py`; see Records. |
| test_k2_lanes.py | **GREEN reproduces byte-identical** (r_pos 1.271, r_dep 6.325, exit 0). Its M8 half was already on disk: `runs/lanes/ptr_M8.npz` equals `runs/walk/ptr_16384.npz` array for array (N1), from 10:14:19, eight minutes before the lanes test was written (10:22:26). The M32 half is new. |
| pilot_pipeline.py | **Did not launch before READY_SP; launched only the hook READY_SP names.** It polls every 60 s. READY_SP was written at 10:41:49 and read at 10:42:32, which is also when the pilot launched. READY_SP names one hook, `resolvent_sp.py`, and the launched config's hook_sha256 is 36039c3a, equal to READY_SP's sha and the sha on disk. The config equals test_k2_pilot.py's REG on every key, and the lock owner.txt names Cameron and pilot_fR_sp_s0 at 10:42:32. Two gaps did not bite this time. (a) It takes the first `.py` named in READY_SP that is a non-stub file defining ResolventAttention, and the lane now holds two such files (`resolvent_sp.py`, `resolvent_sp_98259dff.py`). (b) It does not pin the sha READY_SP names, and pilot_config accepts a basename match, so a hook edited after READY_SP would launch and pass the config gate. |
| Chase bars | **RED on a stub before every measurement; no GREEN is reached by a check that cannot fail.** v1 (58151198): stub RED 10:13:29. v2 (198a1752): 10:17:32. The first real-hook run was run2 at 10:28–10:30; run1 at 10:20 never launched. v3 (58fbdf8c): stub RED 10:34:57, run3 10:35:03. Every GREEN bar has a clause that failed on its stub: mutants for gradcheck and support_v2, a/b movement for the smoke and train_ladder bars, finite and residual for mem_v2, and every row for parity_v2. The CPU rerun of support, gradcheck and support_v2 on both hook shas and both stubs is identical to run2, run3 and red_stub (N2). Every Chase test and hook sha equals its `.sha256` record (N2). Two v1 bars cannot pass on any hook as written: the support mutant cannot change the set, and the parity mutant reads NaN, which `> 1e-6` scores as a match. Chase declares both. |
| Standard checks | Every test line uses `status`, with no `state` key after 6053. Every test exits 1 on a failing row (chain.py ORs the exit codes). No `hash()` seeding in either lane. No test sha changed after its RED. |

## Cameron: 40 audited, 9 struck

Board 6056–6169: 14 audited, 1 struck.
- 6103: the err_profile values (n = 4096: 257–512 0.8418, "513–565" 0.6058 (N 591); n = 16384: 0.7843 / 0.2275 / 0.0702). err_profile passes iff the depth buckets partition the tokens, so it cannot fail on any of these values. The row's own label is 513–1024, not 513–565.

CAMERON_REPORT.md (10:26:43): 26 audited, 8 struck.
- Verdict: "so leakage is most of the 4k wall". The registered prediction for leakage fails (0.8750 < 0.9), and no row reads "most".
- Verdict: "At 16k the wall is the pointer itself". walk_explains, the row that tests whether the pointer accounts for the soft model's far-band errors, is RED at 16k (0.7995 < 0.9). Of the band, 0.1973 is right under the hard walk and wrong in the soft read.
- The error-profile table and "This answers the K1 report's open '513+ at 0.55–0.69'", including "max depth 565". err_profile cannot fail on the values. The value 565 is right on the saved arrays (N1), but no row or log prints it.
- "On 8 beds the 513+ bucket reads 0.6058. The γ horizon does not explain it: +0.0751 against 0.10". 0.6058 is the n = 4096 bucket. gamma_explains_513 reads bucket 513–1024 at n = 16384 (0.2275 → 0.3026), and no row reads γ 0.9999 on the 4k bucket.
- "0.0615 … are off-chain argmax links, an identity error; 0.0636 … the soft residual, which exact zeros would remove". The walk rule also fails a token that lands in its own chain below depth 8, and no row splits off-chain links from deep absorption. No exact-zero model was run.
- Unbarred diagnostics: "K1's struck ×3 value of 0.9113 was a 2-bed draw of that spread". This re-argues a struck item (K1 pass 7) with a diagnostic. The beds 0–1 value is 0.9094, not 0.9113. diag_perbed.log reproduces digit for digit from `runs/k3` and `runs/k1` (N1), but no lane script produces it.
- Kill 1 replacement: "the remaining 4k error is about half off-chain pointer identity and half soft residual". The same unsplit decomposition as above.
- Kill 3: "Sparsemax … is the measured repair for the leak part: +0.59 at 4k via κ, and 0.1973 of the 16k band is recoverable". No sparsemax arm was measured. +0.59 is κ scaling of the softmax checkpoint, and 0.1973 is a ceiling-side field, not a measured recovery.

Clean in the report:
- the verdict numbers and the bars paragraph (shas, RED counts, measured counts);
- the whole bars table;
- walk_explains at 4k (0.9326) and the 16k split fields (0.1973 / 0.5471) as a description of that row's RED;
- kills 2 and 4, and kill 3's ceiling kill;
- the K2.1 section as of 10:26:43;
- Limits, OPEN, and standing state ("leakage is real but not the whole wall", "the pointer caps … 0.4498 / 0.2292").
- The setup paragraph: model.pt sha e354c013 (N1), hook 3855288d, the three lock windows and 184 s (lock_releases.log), and no nurses (no Cameron nurse line).

The per-bed spread and the mid-position rate 0.00146 are labelled unbarred, "not findings". They are not counted as findings.

## Chase: 30 audited, 3 struck

Board 6071–6174: 12 audited, 1 struck.
- 6173: "(row blocks widen, the allocator cannot reuse them)", given as the cause of the 98259dff memory kill. No check isolates it. Its numbers (6.36 / 0.68 GiB, then 0.63) are bound by v1 sp_mem's RED on run2 and by mem_v2. CHASE_REPORT itself labels the mechanism "from reading".

READY_SP (10:41:49): 7 audited, 0 struck. Hook and sha, memory 0.33 / 0.35 and 0.61 / 0.63 GiB, precision at S 1024 (x ≤ 6.16e-6, grads ≤ 7.83e-5) and at S 16384 (1.11e-5 / 3.20e-5), the GREEN list and the RED-kept list all match run3*.log. Cost is labelled "reported, not scored".

CHASE_REPORT.md (10:45:41): 11 audited, 2 struck.
- Kills table, row 1: "On CPU float64 it is bitwise equal to 98259dff in x and every grad (dev check, not a bar)". No file in the lane records this check.
- OPEN: "Forward-only is 9.39× FR at S 16384 (the sort pass plus the second logits pass)". This is a REPORT line that no bar reads, with a cause that no measurement splits.

## RECORD_K (Dispatcher, 8e1bd52): 6 audited, 1 struck

- "Leakage is most of the 4k wall" (also the commit title "leakage is most of the 4k wall"). K2.0's registered prediction fails and its counter is quiet, which leaves the middle, and no row reads "most".

"K2, as registered here" was committed in 1554a74 at 10:03:37, before the first K2 file (10:09:13). 8e1bd52 only appends to it. Clean in RECORD_K:
- the κ grid and the 16k best;
- "γ is not the wall", which gamma_counter_quiet binds by firing;
- the walk ceiling;
- "at 16k the pointer itself links to the wrong chain", as existence: lanes_events counts at least 30 off-chain links in each of the four rates;
- the lanes numbers;
- the K2.1 shas, recipe, gates and branch table, all equal to the files.

## Process notes

- **verify_sp.py** (K2 root, 10:22:43) was written by Chase's nurse N1. That nurse is a sonnet subagent spawned by Chase and dispatched as a "read-only bug review" (board 6107).
  - It ran once on CPU at 10:22:46: ALL OK, worst 1.51e-14, 0 support mismatches, on hook 98259dff. The Inspector rerun on the 98259dff copy is identical.
  - It sits outside any lane, and it exits 0 even when it prints MISMATCH.
  - No bar or board line cites it. CHASE_REPORT's Nurses section cites its output as a "dev check (not a bar)".
- **Cameron posted after his `done`.** Cameron's `done` is at 6105, and lines 6109–6149 and 6169 follow it with no second Cameron dispatch line.
- **Headers that overstate "before any number".**
  - test_k2_walk.py reads `runs/k3/ok_*.npz` (10:10:59–10:11:01), which predate the file (10:13:40). Its counter line (agreement ≤ 0.7) was set after K2.0 gave the soft 16k accuracy of 0.2555, and the null stub already passes that counter at 0.7445. Cameron's Limits says so.
  - test_k2_lanes.py: "before any number it reads". `ptr_M8.npz` (10:24:01) equals `runs/walk/ptr_16384.npz` (10:14:19), so the M8 half of r_pos and r_dep was on disk before the file (10:22:26). Walk's head-0 position rates, including 0.00146, were printed at 10:14:24.
- **Chase's v3 lines were set after run2's reading, and Chase says so.**
  - parity_v2 changes only the mutant predicate. Its numeric rows are bit-identical to run2's.
  - mem_v2's fp32-vs-float64 line of 1e-4 was chosen after the 1.10e-5 residual was seen.
  - The v1 rows stay RED on the board.

## Records

- **Reruns.** All ran in `K2/inspector/rerun1/` on byte-identical copies (shas checked), CPU only (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`). They read the lanes' saved outputs, and the comparisons ignore CR line endings. Chase's bars ran with `CHASE_BOARD=0` through a driver that calls only the CPU bar functions (N2).
- **RED-2 differences.** The pilot and grid stubs name the live `chase/resolvent_sp.py` with the sha it had at 10:23 (98259dff). Chase replaced it with 36039c3a at 10:33:53, and READY_SP appeared at 10:41:49.
  - The pilot stub now reads config "bad: hook sha" instead of "READY_SP missing". Its fail count is unchanged at 10.
  - The id-grid stub now fails 115 rows, not 111. The +4 are all fR_sp rows: config s0 and s1, trained s0 and learnable s0; fR_sp's s0 prediction rows turn UNREAD.
- **GPU.** None was used by the Inspector. The lock has been held by Cameron's pilot since 10:42:32, and about 26 min of training remains at 2000 steps per 128 s. Chase's GPU rows were audited from run2*.log and run3*.log only: parity fp32 rows, the module row, mem, smoke and train_ladder.
- **Repo.** No repo writes except board appends. No git writes, and nothing written into any lane.

## Lanes in flight

- Cameron: pilot_fR_sp_s0 is training (24,000 steps, seed 0, hook 36039c3a; step 8000 at 10:55). pilot_pipeline.py then runs `k2_ptr.py walk` and `test_k2_pilot.py`, and posts rows to the board. Nothing of the pilot is audited here.
- Chase: `done` at 6174. Nothing in flight.
