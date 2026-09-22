# Inspector pass 1 (30-minute mark)

Board: house-events.jsonl lines 4719-4804 (this run). 41 audit events appended after line 4804. Re-runs: CPU only (CUDA_VISIBLE_DEVICES=""), Foreman scripts run from a copy in SPJ/L1/inspector/runs/fm so no fellow file was written. Raw outputs: SPJ/L1/inspector/runs/*.txt.

Totals: 41 checks, 8 struck, 31 clean, 2 of the clean are not re-run: GPU (Cameron costb tests) and 1 is not re-run: cost (Foreman 512/400 anchor runs), all 3 audited on order and saved output.

## Chase (SPJ/L1/chase, report CHASE_REPORT.md)

| # | Claim | Check run | Actual output (exact line) | Verdict | Why |
|---|---|---|---|---|---|
| 1 | 12/12 RED | pytest test_chase_L1_red.py | `12 failed` ... `E AssertionError: max \|record - rerun\| per-seed train loss: {'a_softmax_twin': 0.3150067925453186, 'a2_fox_twin': 0.4442369043827057, 'fq_quat_twin': 0.0011675930581986904}` | clean | every assertion line identical to final_run.txt and the 12 board RED names |
| 2 | 9/9 GREEN | pytest test_chase_L1_green.py | `FQ errs {(64, 'torch.float64'): 4.85722573273506e-16, (64, 'torch.float32'): 3.338733826074858e-07, (512, 'torch.float32'): 3.338733826074858e-07}` / `FOLD errs {'torch.float64': 7.771561172376096e-16, 'torch.float32': 1.0461785424631387e-06}` / 9 PASSED | clean | numbers equal report |
| 3 | 1/1 GREEN a5seed | pytest test_chase_L1_a5seed.py | `PASSED test_chase_L1_a5seed.py::test_a5_stable_seed_reproduces` | clean | reads saved beds_fixed/r1.json, r2.json; bed not re-trained |
| 4 | SPJ test_horn.py::test_planted_rotation_no_noise still fails | pytest on SPJ/test_horn.py | `E AssertionError: angle error too large: 4.7121609153872417e-08` | clean | as stated |
| 5 | finding A5-RERUN-REPRODUCES-RECORD | board order | RED event, then finding (logs KeyError and corrected RED line), then GREEN A5-STABLE-SEED-REPRODUCES | clean | corrected RED line on the board before any GREEN; re-run gap 0.4442369043827057 |
| 6 | finding: Q1, FOLD, FQ-ARM reproduced by independent code | board order | no RED event for Q1, FOLD or FQ-ARM in this run | struck | a finding event with no preceding RED is unbound (Chase files it Open) |
| 7 | finding Q2 kill: stated group infinite | board order + Wilson | RED Q2-CONTRACT-GENS-ICOSAHEDRAL, then GREEN Q2-STATED-GROUP-INFINITE, then finding; `rotation angle of ab = 167.2983619145489 deg, not an icosahedral angle` | clean | rerun 5000/3393, 120/60 equals Wilson [V] |
| 8 | finding Q3 not an attention read | board order + Wilson | REDs Q3-CONTROL-COMPOSES, Q3-FOX-READ-ORDER-BLIND (`diff 0.14336512334745194`), then GREEN, then finding | clean | 4.5530 equals Wilson [V] 4.552964986550147 |
| 9 | finding PF-GAMMA lattice stall | board order + Wilson | `record P_min 2.09832 vs DP global min 0.029946095698810307` | clean | REDs precede GREEN; 2.09832 equals Wilson [V] |
| 10 | finding DRIFT-SYNTH is arccos floor | board order + Wilson | `run_horn case_a noiseless angle error 4.712160915387242e-08 rad > 1e-9` | clean | RED precedes GREEN; equals Wilson [V] |
| 11 | finding leap does not survive it.L0 | board order + Wilson | `standalone SDPA 5.861 ms vs in-B' SDPA stage 4.376 ms; ratio on the stage = 2.878` | clean | REDs Q2 and COSTB-SDPA precede; 2.1486 and 12.5924/4.3761 equal Wilson [V] |
| 12 | finding: construction survives (hand-set f_Q A5 500/500 L64, 50/50 L4096; pair closes 60/120) | board order | no test event for route_handset_a5.py; GREEN Q2-REROUTE-INDEPENDENT has no RED | struck | unbound |
| 13 | finding published-number checks | board order + Wilson | `record says 'MATH', producer JSON says 'EFFICIENT_ATTENTION'`; `above 1/60: {'a2_fox_twin': 0.03125}`; `901 feasible (a=a'', f_Q) points, e.g. a=a''=0.9, f_Q=1.0; floor 0.996844` | clean | REDs precede; EFFICIENT_ATTENTION, 0.03125, 0.996844 equal Wilson [V] |
| 14 | report: FQ-ARM 36 params true only at d 8, 260 at d 64 | board order | no test event | struck | unbound |
| 15 | Chase claims vs Wilson [V] | cross-read | no contradicting line | clean | COST-B', BOARD-BED, L1-A5, DRIFT, Q2/Q3, PF, su2 order agree |

## Foreman (SPJ/L1/foreman, report FOREMAN_REPORT.md)

| # | Claim | Check run | Actual output (exact line) | Verdict | Why |
|---|---|---|---|---|---|
| 16 | RED a5_repro_test | python a5_repro_test.py (copy) | `AssertionError: RED: gen_split('train',0) differs by PYTHONHASHSEED: [3, 1, 1, 0, 1, 0, 0, 1, 3, 0, 2, 1, 0, 3, 1, 1] vs [2, 1, 0, 0, 3, 1, 2, 3, 0, 0, 0, 1, 3, 2, 1, 3]` exit 1 | clean | identical |
| 17 | GREEN a5_handset T2a | python a5_handset.py (copy) | `T2a_anchor_forced {'all': 1.0, '16': 1.0, '32': 1.0, '64': 1.0}` | clean | identical |
| 18 | RED a5_handset T2b, T2c | same run | `RED T2b: as-built SDPA, same hand-set lift, acc@64=0.0312 all=0.0574 (<0.9) \| RED T2c: trained layer-0 generators are no A5 lift: inverse_pair_err=1.176 order3_err=1.351 (bar <0.1)` exit 1 | clean | RED status reproduces; numbers differ (row 25) |
| 19 | RED a5_key0_test | python a5_key0_test.py (copy) | `RED: optimised as-built layer-1 attention reaches key 0 with p=0.0137 at query 63 (bar 0.9)` exit 1 | clean | identical |
| 20 | RED iaut_bar_test | original dir (read-only) and copy | `RED: best finite-SU(2) Bayes ceiling per seed {'0': 0.67375, '1': 0.64, '2': 0.66625}, mean 0.6600 < L1-IAUT pass bar 0.8120` exit 1 | clean | identical on both JSONs |
| 21 | GREEN iaut_su2_ceiling | python iaut_su2_ceiling.py (copy), 42 s | `GREEN` exit 0; best_2I 0.67375 / 0.64 / 0.66625 | clean | identical |
| 22 | GREEN iaut_linear_readout | python iaut_linear_readout.py (copy), 58 s | `GREEN 0.2904166666666667` exit 0 | clean | identical |
| 23 | anchored at bed budget still memorises, 0.0156 @64 | a5_anchor_train.py bed 64 150 {0,1} (copy) | bos0 `"64": 0.015625` loss 0.00438; bos1 `"64": 0.0` loss 0.00324 | clean | both memorise; number drifts (row 25) |
| 24 | GREEN a5_anchor_train 512/400: f_Q 1.0, unanchored 0.109, softmax 0.0312, FoX 0.0156 | saved a5_anchor_train.jsonl | `"bos": 1, ... "64": 1.0`; `"bos": 0, ... "64": 0.109375`; `"variant": "a" ... "64": 0.03125`; `"variant": "a2" ... "64": 0.015625` | clean, not re-run: cost | 282-453 s each; saved output equals board; REDs T2b, key0 precede |
| 25 | numbers T2b acc@64 0.0000, T2c 1.183/1.078 eval 0.059, anchored-bed 0.0156 | rows 18, 23 | 0.0312 / 0.0574; 1.176 / 1.351, eval 0.069580078125; 0.0 | struck | split seeded by salted hash(); each number is one unrepeatable draw; RED status holds |
| 26 | finding F1 A5 fold exact, no anchor | board order | RED T2b, then GREEN T2a, then finding | clean | bound |
| 27 | finding F2 f_Q loss 0.006 is memorisation | board order + Wilson | RED T2c, then finding | clean | loss matches Wilson [V] 0.005676-0.005922 |
| 28 | finding F3 L1-IAUT bar 0.8120 unreachable by any finite SU(2) image | board order + mtimes | GREEN iaut_su2_ceiling (0.674/0.640/0.666) logged before RED iaut_bar_test; iaut_su2_ceiling.json 23:16:36, iaut_bar_test.py 23:18:56 | struck | post-hoc RED re-reading a known number; the test covers 2I and 2O pairs only, not every finite SU(2) image (binary dihedral uncovered, Foreman's own Open) |
| 29 | finding F4 L1-A5 not reproducible | board order | RED a5_repro_test, then finding | clean | re-run identical |
| 30 | finding A5 root cause two-part | board order | REDs T2b, T2c, a5_key0_test, then GREEN a5_anchor_train, then finding | clean | bound |
| 31 | finding I-AUT: key 0 reachable p 0.9999; linear readout 0.2904 | board order | iaut_key0_check re-run `mean p 0.9999, min p 0.9996`, but no test event; GREEN iaut_linear_readout has no RED | struck | unbound; its representational half is F3, struck |
| 32 | finding rerouted L1-A5 passes bar on seed 0 | board order | REDs T2b, key0, then GREEN controls, then finding | clean | bound; one seed, f_Q and controls on different splits (Foreman's own Open) |
| 33 | Foreman claims vs Wilson [V] | cross-read | no contradicting line | clean | su2.py 17:07:03 newest-left, f_Q loss, DIAG 0.2860 / 0.3110 agree |

## Cameron (SPJ/L1/cameron): pass 1, incomplete

CAMERON_REPORT.md appeared at 23:33 and `done` is on the board; pass 1 audits board events only. The report is for the final pass.

| # | Claim | Check run | Actual output (exact line) | Verdict | Why |
|---|---|---|---|---|---|
| 34 | test_costb_fused RED(author) then GREEN(fused) | saved JSON (GPU) | author `"fold_max_abs_err_vs_contract_f64": 3.7753911924862122`, `"ratio_fwd": 2.486609616450163`, t 23:07:26; fused `2.2134439208176815e-06`, `"ratio_fwd": 1.0407126587706812`, t 23:07:56 | clean, not re-run: GPU | order holds; costb_fused.py was edited at 23:11, after the forward GREEN |
| 35 | test_costb_fwdbwd RED then GREEN | saved JSON (GPU) | `"grad_max_rel_err_vs_contract_f64": 1.634022908536543`, `"ratio_fwd_bwd": 2.113440460866904`; fused `9.867353587451907e-07`, `1.1213057703781357` | clean, not re-run: GPU | equals board lines |
| 36 | finding COST leg survives 1.041x / 1.121x | board order + Wilson | 2 REDs, then 2 GREENs, then finding; stages `"scan": 0.018807466824849448` | clean | EFFICIENT_ATTENTION agrees with Wilson [V]; Wilson's build flag comes from flash_sdp_enabled(), so no contradiction |
| 37 | finding costb_impl Pi order, fold err 3.775 | board order + Wilson | RED line `fold: max \|out - contract G_ij fold\| = 3.775e+00` precedes | clean | Wilson [V] costb_impl oldest-left |
| 38 | RED test_bed_admissible x3 | python test_bed_admissible.py on 3 inputs | `INADMISSIBLE board ply>=40: A1 ... \| A2 ... \| A4: no SU(2) construction on the bed \| A3: best floor rule_free_tracker = 0.9968, +0.10 = 1.0968 > 1.0`; shell: `INADMISSIBLE S3 ... short-prefix acc 0.7945 < 0.90`; long: `INADMISSIBLE S3 ... A2: twin loss did not plateau`; exit 1 x3 | clean | RED holds; current script adds A4 clauses and an A5 line missing from the logged lines |
| 39 | finding board bed retired (A3) | board order + Wilson | RED on board_bed_result precedes | clean | 0.996844 equals Wilson [V] |
| 40 | finding exact SU(2) construction on shell bed 1.000/1.000 | board order | no test event for fq_construction.py | struck | unbound |
| 41 | finding torch.compile Mode B' 1.069x; fused ratio vs S 1.132/1.065/1.035/0.978 | board order + saved JSON | costb_compile.json `"ratio_compiled": 1.0690213813268676`; no RED or GREEN event for costb_compile.py or costb_extra.py | struck | unbound; the saved number matches the claim |

## Chase rivals (SPJ/L1/chase_rivals): noted, not audited
Events from line 4782: dispatch "dissect open Chinese model mechanisms", 8 haiku nurse fetches (qwen-gated-attn, qwen3-next, kimi-linear, deepseek-v3-v32, nsa, moba-k2, minimax, glm-qwen3), 2 RED + 1 GREEN test events (they use the field "state", not "status"), 4 findings (a2 = Qwen G1 gate, GDN decay vs FoX, KDA/GDN non-commuting, MiniMax = ALiBi/FoX), done. Audit at the final pass.

## Notes for the prognosis (these are not verdicts)
- Chase says the cost leg fails (2.149x, the author costb_impl). Cameron says it survives (1.041x, a new fused Triton impl on the same bar). Both are bound, and they measure different implementations. The inspector does not reconcile them.
- Chase reads the board bar per square as satisfiable when (a), (a'') <= 0.90. Cameron's A3 reads it as needing best floor + 0.10 <= 1.0. These are two readings of one bar, and the inspector does not decide between them.
