import json, sys
C, S = "clean", "struck"
RR = "Inspector CPU rerun on a byte-identical copy in K2/inspector/rerun1"
A = [
# ---- Cameron board
("Cameron", C, "K2.0 registration: test_k2_leak.py sha 0bc8ea17 before any K2 cell; RED 1 37 rows, RED 2 null stub 25 rows, prediction FAIL and both counters firing (board 6056-6070)", f"test 10:09:13, sha 10:09:28, RED 10:09:28 / 10:09:42, producer k2_leak.py 10:10:40, first cell 10:10:48; sha unchanged; {RR}: RED 1 and RED 2 identical to red_test_k2_leak.log (37 / 25, exit 1); stub leak_counter_quiet FAIL 0.2855, gamma_counter_quiet FAIL +0.0000"),
("Cameron", C, "K2.0 status lines: 8 families green, 6 red (board 6082-6095)", f"{RR} on runs/: identical to run_test_k2_leak.log, 22 fails, exit 1; every family's status matches its rows"),
("Cameron", C, "K2.0 finding: 0.2855/0.7941/0.8689/0.8750/0.8546, max 0.8750 < 0.9, > 0.5; kappa=1 control 0.2855/0.0784/0.0322, gate 0.9963; best kappa gate 0.9962; 16k max 0.2621 / 0.0810 (board 6102)", "leak_cell / leak_prediction / leak_counter_quiet / leak_control_k1 / leak_gate_at_best rows, rerun identical"),
("Cameron", C, "gamma rows: +0.0166 on depth>1280 at 16k (counter fires), +0.0751 on 513-1024 at 16k (bar 0.10), numerics 5.9e-5 / 8.0e-5 (board 6103)", "gamma_counter_quiet / gamma_explains_513 / gamma_numerics rows, rerun identical"),
("Cameron", S, "err_profile values: n=4096 257-512 0.8418, 513-565 0.6058 (N 591); n=16384 257-512 0.7843, 513-1024 0.2275, 1025+ 0.0702 (board 6103)", "err_profile PASSES iff the depth buckets partition the tokens, so it cannot fail on any of these values; the row's label is 513-1024, not 513-565"),
("Cameron", C, "walk registration: test_k2_walk.py (sha af19b1b6) registered after K2.0, RED before its measurement; 6 RED lines (board 6076-6081, 6104)", f"test 10:13:40, RED 10:13:52 / 10:13:53, producer 10:14:12, measurement 10:14:16; sha unchanged; {RR}: RED 1 / RED 2 identical (6 / 7, exit 1); scores no K2.0 row"),
("Cameron", C, "walk status lines: machinery, sane, explains_counter_quiet green; explains, ceiling, link_grows red (board 6096-6101)", f"{RR} on runs/: identical to run_test_k2_walk.log, 5 fails, exit 1"),
("Cameron", C, "walk finding: ceiling 0.9347/0.4498/0.2292; explains 0.9326 (4k) / 0.7995 (16k), 0.1973 walk-right soft-wrong; link rate 0.00598 vs 0.00840 = 1.40x; exact-zero ceiling reading (board 6104)", "walk_ceiling / walk_explains / walk_link_grows rows, rerun identical; the ceiling reading is the file's registered header, fixed before its RED"),
("Cameron", C, "K2.1 registration (1): test_k2_pilot.py sha 9996d651, RED before pilot and READY_SP; 13 rows missing, 10 on null stub, ceiling 0.9347/0.4498/0.2292, BRANCH NONE; run config, gates, branch rule (board 6109-6117)", f"test 10:21:12, RED 10:22:42 / 10:23:12, READY_SP 10:41:49, pilot 10:42:32; sha unchanged; {RR}: RED 1 identical (13); RED 2 10 fails, config text now 'hook sha' (stub names live chase/resolvent_sp.py, edited 10:33:53)"),
("Cameron", C, "K2.1 registration (2): test_k2_grid.py sha a622cd95, both grids before the pilot; 133 rows each with nothing run, 111 on the id null stub (board 6118-6138)", f"test 10:21:52; sha unchanged; {RR}: RED 1 sp and id identical (133 / 133); RED 2 id now 115, +4 all fR_sp rows from the same hook edit (98259dff -> 36039c3a)"),
("Cameron", C, "K2.1 registration (3): test_k2_lanes.py sha 916ffa6e, RED before its producer, 4 rows missing, 3 on null-pointer stub; r_pos / r_dep, prediction and counter (board 6139-6143)", f"test 10:22:26, RED 10:22:42 / 10:23:12, producer k2_ptr.py 10:23:55; sha unchanged; {RR}: both REDs identical (4 / 3); note: ptr_M8.npz equals runs/walk/ptr_16384.npz array for array (10:14:19), so the M8 half was on disk before the file (N1)"),
("Cameron", C, "lanes GREEN: 0.00733 vs 0.00932 (r_pos 1.271), 0.00146 (72 links) vs 0.00927 (r_dep 6.325); links follow position, not depth (board 6144-6148)", f"{RR} on runs/: identical to run_test_k2_lanes.log, 0 fails, exit 0; the reading is lanes_position_driven's registered prediction"),
("Cameron", C, "pilot armed, not launched: pilot_pipeline.py polls READY_SP up to 6 h, takes the non-stub ResolventAttention hook it names, runs the registered pilot under the lock, then walk and test (board 6149)", "code read; process 13068 started 10:26:22; no launch until READY_SP (10:41:49) was read at 10:42:32; gaps (first-named wins, no sha pin) did not bite: READY_SP names one hook"),
("Cameron", C, "pilot launched 10:42:32, hook resolvent_sp.py sha 36039c3a = READY_SP, config as registered, schedule 0.5:6000,0.9:12000,0.99:18000,0.999 (board 6169)", "runs/pilot_fR_sp_s0/log.jsonl config equals test_k2_pilot.py REG on every key, hook_sha256 36039c3a; owner.txt Cameron pilot_fR_sp_s0 10:42:32"),
# ---- CAMERON_REPORT.md
("Cameron", C, "verdict: K2.0 fails its prediction, counter quiet (CAMERON_REPORT.md 10:26:43)", "leak_prediction RED, leak_counter_quiet GREEN, rerun identical"),
("Cameron", C, "verdict: 0.2855 -> 0.8750 at kappa 3, short of 0.9, above 0.5 (CAMERON_REPORT.md 10:26:43)", "leak_cell / leak_prediction rows (N1 class A)"),
("Cameron", S, "verdict: 'so leakage is most of the 4k wall, but not all of it' (CAMERON_REPORT.md 10:26:43)", "the registered leakage prediction fails (0.8750 < 0.9) and its counter is quiet; no row reads 'most'"),
("Cameron", C, "verdict: at 16k no kappa passes 0.2621 / 0.0810 (CAMERON_REPORT.md 10:26:43)", "leak_cell kappa 5 rows"),
("Cameron", S, "verdict: 'At 16k the wall is the pointer itself' (CAMERON_REPORT.md 10:26:43)", "walk_explains, the row testing whether the pointer accounts for the soft model's far-band errors, is RED at 16k (0.7995 < 0.9); 0.1973 of the band is walk-right, soft-wrong"),
("Cameron", C, "verdict: hard-walk ceiling 0.4498 / 0.2292, so sparsemax on this pointer cannot clear the 16k band (CAMERON_REPORT.md 10:26:43)", "walk_ceiling RED, read as registered in test_k2_walk.py's header before its RED"),
("Cameron", C, "setup: model.pt sha e354c013, hook 3855288d, jobs 10:10:48-10:11:08 / 10:14:12-10:14:19 (184 s) / 10:23:56-10:24:03, no nurses (CAMERON_REPORT.md 10:26:43)", "N1: sha recomputed; k2_*.log and lock_releases.log; no Cameron nurse line on the board"),
("Cameron", C, "bars paragraph: shas 0bc8ea17 / af19b1b6; RED 37 / 25, measured 22; RED 6 / 7, measured 5; written before producers; not edited after RED (CAMERON_REPORT.md 10:26:43)", "file mtimes and .sha256 files; reruns identical"),
("Cameron", C, "bars table: every leak, gamma, walk and lanes value and status (CAMERON_REPORT.md 10:26:43)", "N1 maps each to its row, no mismatch; reruns identical"),
("Cameron", S, "error-profile table and 'This answers the K1 report's open 513+ at 0.55-0.69', incl. 'max depth 565' (CAMERON_REPORT.md 10:26:43)", "err_profile cannot fail on the values (passes iff buckets partition the tokens); 565 is right on the arrays (N1) but no row or log prints it"),
("Cameron", S, "'On 8 beds the 513+ bucket reads 0.6058. The gamma horizon does not explain it: +0.0751 against 0.10' (CAMERON_REPORT.md 10:26:43)", "0.6058 is the n=4096 bucket; gamma_explains_513 reads bucket 513-1024 at n=16384 (0.2275 -> 0.3026); no row reads gamma 0.9999 on the 4k bucket"),
("Cameron", C, "at 4k the hard walk explains the soft model's errors, agreement 0.9326 (CAMERON_REPORT.md 10:26:43)", "walk_explains n=4096 PASS, rerun identical"),
("Cameron", S, "'0.0615 ... are off-chain argmax links, an identity error; 0.0636 ... the soft residual, which exact zeros would remove' (CAMERON_REPORT.md 10:26:43)", "the walk also fails a token landing in its own chain below depth 8; no row splits off-chain links from deep absorption; no exact-zero model was run"),
("Cameron", C, "at 16k 0.1973 walk-right soft-wrong and 0.5471 wrong in both (CAMERON_REPORT.md 10:26:43)", "printed fields of walk_explains n=16384, describing that row's RED; rerun identical"),
("Cameron", C, "unbarred diagnostics, labelled not findings: per-bed 0.7678-0.9142, beds 0-1 0.9094; mid-position rate 0.00146, late 5.8x (CAMERON_REPORT.md 10:26:43)", "labelled; diag_perbed.log reproduces digit for digit from runs/k3 and runs/k1 (N1), but no lane script produces it; not counted as findings"),
("Cameron", S, "'K1's struck x3 value of 0.9113 was a 2-bed draw of that spread' (CAMERON_REPORT.md 10:26:43)", "re-argues an item struck at K1 pass 7 with an unbarred diagnostic; beds 0-1 give 0.9094, not 0.9113; 0.9113 is in no lane file (N1)"),
("Cameron", C, "kill 1: prediction fails at 0.8750; counter quiet; kappa worth +0.59 at 4k (CAMERON_REPORT.md 10:26:43)", "leak rows; 0.8750 - 0.2855 = 0.5895"),
("Cameron", S, "kill 1 replacement: 'the remaining 4k error is about half off-chain pointer identity and half soft residual' (CAMERON_REPORT.md 10:26:43)", "the same unsplit decomposition: walk failures are not split into off-chain links vs deep absorption by any row"),
("Cameron", C, "kill 2: gamma-horizon retired, counter fires +0.0166, numerics GREEN (CAMERON_REPORT.md 10:26:43)", "gamma_counter_quiet RED and gamma_numerics GREEN as registered in test_k2_leak.py"),
("Cameron", C, "kill 3: exact-zero ceiling 0.9347 / 0.4498 / 0.2292 kills 'exact zeros remove the far-band wall' for this pointer (CAMERON_REPORT.md 10:26:43)", "walk_ceiling RED x3, rerun identical"),
("Cameron", S, "kill 3: 'Sparsemax ... is the measured repair for the leak part: +0.59 at 4k via kappa, and 0.1973 of the 16k band is recoverable' (CAMERON_REPORT.md 10:26:43)", "no sparsemax arm was measured; +0.59 is kappa scaling of the softmax checkpoint; 0.1973 is a ceiling-side field, not a measured recovery"),
("Cameron", C, "kill 4: walk_link_grows fails at 1.40x; lanes r_pos 1.271, r_dep 6.325; links follow position, not depth (CAMERON_REPORT.md 10:26:43)", "walk_link_grows RED and lanes rows, reruns identical"),
("Cameron", C, "K2.1 section: pilot / grid / lanes shas, recipe, gates, branch rule, RED counts 13 / 10 / 133 / 111, no pilot run, pipeline armed (CAMERON_REPORT.md 10:26:43)", "files and red_test_k21.log; true at 10:26:43"),
("Cameron", C, "Limits (CAMERON_REPORT.md 10:26:43)", "SE caveat; null-stub 0.7445 (red_test_k2_walk.log); per-head 0.3010 / 0.2971; 72 links; pilot stub cannot reach IDENTITY / GRID: all match"),
("Cameron", C, "OPEN (CAMERON_REPORT.md 10:26:43)", "questions only; 'position, not depth' is lanes_position_driven"),
("Cameron", C, "standing state: leakage real but not the whole wall; gamma not the wall; pointer caps 0.4498 / 0.2292; wrong links follow position; ceiling < 0.95 sends the identity grid (CAMERON_REPORT.md 10:26:43)", "leak_counter_quiet GREEN + leak_prediction RED; gamma_counter_quiet RED; walk_ceiling; lanes_position_driven; test_k2_pilot.py branch code"),
# ---- Chase board
("Chase", C, "v1 bars RED on stub: sp_fwd_parity_S1024, sp_support_S1024, sp_gradcheck_S64_f64, sp_mem_S16384, sp_rdepth_smoke_50 (board 6071-6075)", "test_chase_k2.py 10:12:02, sha 58151198 unchanged (N2); stub RED 10:13:29 (red_stub.log); first real-hook run is run2 10:28:42+ (run1 10:20:37 never launched); N2 CPU rerun of stub support / gradcheck identical"),
("Chase", C, "v2 bars RED on stub2: sp_support_S1024_v2, sp_train_ladder_R0_50 (board 6106, 6108)", "test_chase_k2_v2.py 10:17:13, sha 198a1752 unchanged (N2); red_stub_v2.log 10:17:32 before run2; N2 CPU rerun of stub2 support_v2 identical"),
("Chase", C, "run2 statuses on hook 98259dff: parity R, support R, gradcheck G, mem R, smoke G, support_v2 G, train_ladder G (board 6150-6156)", "run2.log / run2_v2.log; N2 CPU rerun of support, gradcheck, support_v2 on the 98259dff copy byte-identical; GPU rows from logs only (lock held by the pilot)"),
("Chase", C, "v3 bars RED on stub: sp_fwd_parity_S1024_v2, sp_mem_S16384_v2 (board 6157-6158)", "test_chase_k2_v3.py 10:34:41, sha 58fbdf8c unchanged (N2); red_stub_v3.log 10:34:57 before run3 (hook_run3.sha256 10:35:03); v3 lines set after run2's reading, self-declared"),
("Chase", C, "run3 statuses on hook 36039c3a: v1 parity R, support R, gradcheck G, mem R, smoke G; support_v2 G, train_ladder G; parity_v2 G, mem_v2 G (board 6160-6168)", "run3*.log; N2 CPU rerun of support, gradcheck, support_v2 on a 36039c3a copy byte-identical; GPU rows from logs only; v1 support cannot pass (mutant sets_equal true on both hooks, N2) and v1 parity's NaN mutant cannot pass, both declared"),
("Chase", C, "finding: READY_SP hook resolvent_sp.py sha 36039c3a; rdepth :ResolventAttention with --gamma_sched, train_ladder :FRSP (board 6173)", "READY_SP, hook_run3.sha256, disk sha, run3 headers"),
("Chase", C, "finding: GREEN values: parity v2 f64 <= 4.23e-15, fp32 x <= 6.16e-6, grads <= 7.83e-5, autocast bitwise, mutant caught; support v2 exact, mutant caught; gradcheck 2/2 mutants; mem v2 0.35 / 0.63 GiB, f64 residual <= 1.6e-14, fp32 vs f64 1.11e-5 / 3.20e-5, densec OOM; smokes and train_ladder (board 6173)", "run3.log / run3_v2.log / run3_v3.log rows; CPU rows rerun identical (N2)"),
("Chase", C, "finding: kill of hook 98259dff, reserved 6.36 GiB for 0.68 allocated at S16384 H4 D32, replacement 0.63 GiB (board 6173)", "v1 sp_mem RED on run2 (6.359 GiB reserved > 6); sp_mem_S16384_v2 on 36039c3a (0.629)"),
("Chase", S, "finding: '(row blocks widen, the allocator cannot reuse them)' as the cause of the 98259dff memory kill (board 6173)", "no check isolates the cause; the before / after differ by the whole hook edit; CHASE_REPORT itself labels it 'from reading'"),
("Chase", C, "finding: three v1 bars stay RED as registered, with v2 replacements (board 6173)", "run3.log; replacements RED on stub before run3"),
("Chase", C, "finding: cost (not scored) fwd+bwd 2.38x at S4096, 2.77x at S16384 (board 6173)", "labelled not scored; run3.log REPORT line 2.3829 / 2.7660; not counted as a finding"),
("Chase", C, "finding: fp32 precision line 1e-5 holds at S1024 only (board 6173)", "parity rows <= 1e-5 at S1024; v1 mem residual 1.10e-5 at S16384 RED"),
# ---- READY_SP
("Chase", C, "READY_SP: hook path and sha 36039c3a (READY_SP 10:41:49)", "disk sha, hook_run3.sha256"),
("Chase", C, "READY_SP: memory 0.33 / 0.35 GiB (H2 D64), 0.61 / 0.63 GiB (H4 D32) (READY_SP 10:41:49)", "sp_mem_S16384_v2 rows, run3_v3.log (0.3285 / 0.3477, 0.6099 / 0.6289)"),
("Chase", C, "READY_SP: fp32 vs f64 at S1024 x <= 6.16e-6, grads <= 7.83e-5 (READY_SP 10:41:49)", "sp_fwd_parity_S1024_v2 rows, run3_v3.log"),
("Chase", C, "READY_SP: at S16384 x 1.11e-5 (H2 D64), 3.20e-5 (H4 D32); 1e-5 line claimed at S1024 only (READY_SP 10:41:49)", "sp_mem_S16384_v2 f32_vs_f64 field, whose 1e-4 line was set after run2 (self-declared)"),
("Chase", C, "READY_SP: cost 2.38x / 2.77x fwd+bwd, 3.42x / 9.39x fwd-only, reported not scored (READY_SP 10:41:49)", "labelled; run3.log REPORT line; not counted as a finding"),
("Chase", C, "READY_SP: GREEN list of six bars on hook 36039c3a (READY_SP 10:41:49)", "run3*.log statuses"),
("Chase", C, "READY_SP: RED-kept list with reasons (mutant NaN, support+1 cannot change the set, 1.10e-5 vs 1e-5) (READY_SP 10:41:49)", "run3.log; N2 mutant_tau+1_sets_equal true"),
# ---- CHASE_REPORT.md
("Chase", C, "verdict: hook ready; <= 4.23e-15 in f64, same support; gradcheck 2/2 mutants; 0.35 / 0.63 GiB vs 6 GiB; trains in rdepth and train_ladder (50-step smokes) (CHASE_REPORT.md 10:45:41)", "run3*.log rows; CPU rows rerun identical (N2); smoke scope stated in OPEN"),
("Chase", C, "verdict: three v1 RED kept with GREEN v2 replacements; 98259dff defect 6.36 / 0.68 GiB, replacement bound (CHASE_REPORT.md 10:45:41)", "run2.log / run3*.log"),
("Chase", C, "bars section: every bar RED on a stub before measurement; no test edited after RED; run3 on 36039c3a decides (CHASE_REPORT.md 10:45:41)", "mtimes and shas (N2)"),
("Chase", C, "bars table values: parity_v2, support_v2 (1,046,528 / 1,046,086 of 1,049,600), gradcheck margin >= 4.5e-3, mem v1 / v2, smoke, train_ladder (CHASE_REPORT.md 10:45:41)", "run2*.log / run3*.log; 1,049,600 = 2 heads x 524,800 causal entries"),
("Chase", C, "memory arithmetic, ~130 MiB per head, labelled 'arithmetic, not a bar' (CHASE_REPORT.md 10:45:41)", "labelled; not counted as a finding"),
("Chase", C, "cost table, labelled 'reported, not scored' (CHASE_REPORT.md 10:45:41)", "run3.log REPORT line, all 12 values match"),
("Chase", C, "kills rows: 98259dff numbers 6.36 / 0.68 and 2.40 / 0.37 GiB (mechanism labelled 'from reading'); parity NaN mutant; support+1 algebra; residual line set after 1.1e-5, not a precision claim; run1 never launched (CHASE_REPORT.md 10:45:41)", "run2.log, run3*.log, N2, run1_nolaunch.log"),
("Chase", S, "kills row 1: 'On CPU float64 it is bitwise equal to 98259dff in x and every grad (dev check, not a bar)' (CHASE_REPORT.md 10:45:41)", "no file in the lane records this check"),
("Chase", C, "process note 12.05 GiB before OOM; Nurses: N1 dev check ~1e-14-1e-16, 0 support mismatches; N2 25 chase.k2 board lines (CHASE_REPORT.md 10:45:41)", "control row field; verify_sp.py rerun on the 98259dff copy identical to N1's transcript (worst 1.51e-14); 25 lines counted"),
("Chase", S, "OPEN: 'Forward-only is 9.39x FR at S 16384 (the sort pass plus the second logits pass)' (CHASE_REPORT.md 10:45:41)", "a REPORT line no bar reads, stated in OPEN with a cause no measurement splits"),
("Chase", C, "OPEN rest: fp32 1.11e-5 / 3.20e-5 at S16384; support in fp32 unbound; rebuilt W unbound; :FRSP note from code reading (CHASE_REPORT.md 10:45:41)", "mem_v2 field; the rest labelled unbound"),
# ---- RECORD_K
("Dispatcher", C, "RECORD_K 'K2, as registered here' before any K2 cell runs", "committed in 1554a74 at 10:03:37, before the first K2 file (10:09:13); 8e1bd52 only appends"),
("Dispatcher", C, "RECORD_K K2.0 result: RED twice; kappa grid; 16k best 0.2621 / 0.0810; gamma +0.0166 so gamma is not the wall; hard-pointer ceiling 0.9347 / 0.4498 / 0.2292 (8e1bd52)", "leak and walk rows, reruns identical"),
("Dispatcher", S, "RECORD_K 'Leakage is most of the 4k wall' (8e1bd52; also its commit title)", "K2.0's registered prediction fails (0.8750 < 0.9) and its counter is quiet; no row reads 'most'"),
("Dispatcher", C, "RECORD_K 'at 16k the pointer itself links to the wrong chain' (8e1bd52)", "as existence: lanes_events counts >= 30 off-chain links in each of the four rates"),
("Dispatcher", C, "RECORD_K lanes: 0.00932 vs 0.00733, 0.00927 vs 0.00146, r_pos 1.271, r_dep 6.325 (8e1bd52)", "lanes rows, rerun identical"),
("Dispatcher", C, "RECORD_K K2.1: pilot 9996d651, grid a622cd95, recipe, gates, branch table (8e1bd52)", "equal to test_k2_pilot.py / test_k2_grid.py; RED before any run"),
]
counts = {}
for cites, v, name, why in A:
    c = counts.setdefault(cites, [0, 0]); c[0] += 1; c[1] += v == S
print(len(A), sum(v == S for _, v, _, _ in A), counts, file=sys.stderr)
if len(sys.argv) > 1:
    for cites, v, name, why in A:
        line = json.dumps({"t": "audit", "agent": "Inspector", "cites": cites, "verdict": v, "name": name, "why": why})
        assert "\n" not in line
        with open(sys.argv[1], "a", encoding="utf-8") as f:
            f.write(line + "\n")
