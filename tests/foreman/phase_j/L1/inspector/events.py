import json
B = r"C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
E = [
# Chase: re-runs
("Chase","clean","CH/test_chase_L1_red.py (12 RED)","CPU re-run: 12 failed, every assertion line identical to final_run.txt and the 12 board RED names (e.g. gap 0.4442369043827057, 901 feasible, 5.861 vs 4.376 ms)"),
("Chase","clean","CH/test_chase_L1_green.py (9 GREEN)","CPU re-run: 9 passed; FQ errs 4.857e-16 / 3.339e-7 / 3.339e-7, FOLD 7.77e-16 / 1.046e-6, equal to report"),
("Chase","clean","CH/test_chase_L1_a5seed.py (1 GREEN)","CPU re-run: 1 passed on saved beds_fixed/r1.json vs r2.json (bed not re-trained)"),
("Chase","clean","claim: SPJ test_horn.py::test_planted_rotation_no_noise still fails","re-run: FAILED, angle error 4.7121609153872417e-08 > 1e-9"),
# Chase: binding
("Chase","clean","finding A5-RERUN-REPRODUCES-RECORD","RED event precedes finding and GREEN A5-STABLE-SEED; first RED run was a test KeyError, corrected RED line logged in the finding before the GREEN; re-run gap 0.4442369043827057"),
("Chase","struck","finding: Q1, FOLD, FQ-ARM reproduced by independent code","logged as a finding with no RED event on the board for Q1, FOLD or FQ-ARM; unbound (Chase files it Open)"),
("Chase","clean","finding Q2 kill: stated group infinite","RED Q2-CONTRACT-GENS-ICOSAHEDRAL precedes GREEN Q2-STATED-GROUP-INFINITE and finding; rerun 5000/3393, 120/60 equals Wilson [V]"),
("Chase","clean","finding Q3 is not an attention read","REDs Q3-CONTROL-COMPOSES and Q3-FOX-READ-ORDER-BLIND precede GREEN Q3-FIXED-PATTERN-READ and finding; 4.5530 equals Wilson [V] 4.552964986550147"),
("Chase","clean","finding PF-GAMMA unit lattice is an L-BFGS-B stall","REDs PF-RECORD-EPS003-IS-MIN, PF-REPRICE-ON-TOKEN-LATTICE precede GREEN and finding; record 2.09832 equals Wilson [V]"),
("Chase","clean","finding DRIFT-SYNTH 4.71e-8 is the arccos readout floor","RED DRIFT-NOISELESS-1E9 precedes GREEN DRIFT-STABLE-READOUT-1E9 and finding; 4.712e-8 equals Wilson [V]"),
("Chase","clean","finding leap does not survive it.L0 (Q2 exactness, cost 2.149x / 2.878x)","REDs Q2-CONTRACT-GENS and COSTB-SDPA-TIMED-CONSISTENTLY precede; 2.1486 and 12.5924/4.3761 equal Wilson [V]"),
("Chase","struck","finding: construction survives (hand-set f_Q A5 500/500 at L64, 50/50 at L4096; pair closes 60/120)","route_handset_a5.py has no RED or GREEN event on the board, and GREEN Q2-REROUTE-INDEPENDENT has no RED; unbound"),
("Chase","clean","finding published-number checks (BOARD-BED, COST-B' backend, L1-A5 hash seed, FoX 0.03125)","REDs BOARD-BAR, COSTB-BACKEND, A5-RERUN, A5-COMMIT precede; EFFICIENT_ATTENTION, 0.996844/0.843942, 0.03125 equal Wilson [V]"),
("Chase","struck","report claim: FQ-ARM 36 extra params true only at d 8, 260 at d 64","no RED or test event on the board for the parameter count; unbound"),
("Chase","clean","Chase claims vs Wilson [V]","no contradiction found: COST-B', BOARD-BED, L1-A5, DRIFT, Q2/Q3, PF, su2 order all agree with WILSON_REPORT.md [V] lines"),
# Foreman: re-runs
("Foreman","clean","SPJ/L1/foreman/a5_repro_test.py (RED)","CPU re-run: exit 1, lists [3, 1, 1, 0, ...] vs [2, 1, 0, 0, ...], identical to report"),
("Foreman","clean","SPJ/L1/foreman/a5_handset.py T2a (GREEN control)","CPU re-run: T2a all/16/32/64 = 1.0, equal to report"),
("Foreman","clean","SPJ/L1/foreman/a5_handset.py T2b, T2c (RED)","CPU re-run: exit 1, both RED (acc@64 0.0312 < 0.9; inverse 1.176, order-3 1.351 > 0.1); numbers differ, see unpinned-numbers event"),
("Foreman","clean","SPJ/L1/foreman/a5_key0_test.py (RED)","CPU re-run: exit 1, p=0.0137 at query 63, identical to report"),
("Foreman","clean","SPJ/L1/foreman/iaut_bar_test.py (RED)","CPU re-run: exit 1, mean 0.6600 (0.67375/0.64/0.66625) on original and regenerated JSON"),
("Foreman","clean","SPJ/L1/foreman/iaut_su2_ceiling.py (GREEN)","CPU re-run: exit 0, best_2I 0.67375/0.64/0.66625, equal to log"),
("Foreman","clean","SPJ/L1/foreman/iaut_linear_readout.py (GREEN)","CPU re-run: exit 0, GREEN 0.2904166666666667, equal to log"),
("Foreman","clean","SPJ/L1/foreman/a5_anchor_train.py bed budget 64/150 (bos 0, 1)","CPU re-run: both memorise (train loss 0.0044/0.0032), held-out @64 0.0156 / 0.0000 vs claimed 0.0156 / 0.0156"),
("Foreman","clean","SPJ/L1/foreman/a5_anchor_train.py 512/400 runs (fq bos1, fq bos0, a, a2)","not re-run: cost (282-453 s each); saved jsonl gives 1.0, 0.109375, 0.03125, 0.015625 at 64, equal to board lines; REDs T2b and key0 precede"),
("Foreman","struck","unpinned numbers: T2b acc@64 0.0000, T2c 1.183/1.078 eval 0.059, anchored-bed 0.0156","re-run gives 0.0312, 1.176/1.351 eval 0.0696, 0.0000: split seeded by salted hash(), each number is one unrepeatable draw; RED status holds"),
# Foreman: binding
("Foreman","clean","finding F1: A5 fold exact, bed has no anchor","RED T2b precedes GREEN T2a and the finding"),
("Foreman","clean","finding F2: recorded f_Q loss 0.006 is memorisation","RED T2c precedes the finding; loss matches Wilson [V] 0.005676-0.005922"),
("Foreman","struck","finding F3: L1-IAUT bar 0.8120 unreachable by any finite SU(2) image","GREEN iaut_su2_ceiling carried 0.674/0.640/0.666 before RED iaut_bar_test, written after the JSON (23:16:36 to 23:18:56) and only re-reading it; and 2I/2O pairs do not cover every finite SU(2) image"),
("Foreman","clean","finding F4: L1-A5 record not reproducible (salted hash)","RED a5_repro_test precedes the finding; re-run identical"),
("Foreman","clean","finding: A5 root cause two-part (anchor, then budget)","REDs T2b, T2c, a5_key0_test precede GREEN a5_anchor_train and the finding"),
("Foreman","struck","finding: I-AUT key 0 reachable (p 0.9999), linear readout 0.2904","iaut_key0_check.py has no test event (self-labelled not a finding test); GREEN iaut_linear_readout has no RED; representational part is F3, struck"),
("Foreman","clean","finding: rerouted L1-A5 passes registered bar on seed 0","REDs T2b and key0 precede GREEN controls and the finding; one seed, and f_Q and controls drew different splits (Foreman's own Open)"),
("Foreman","clean","Foreman claims vs Wilson [V]","no contradiction: su2.py mtime 17:07:03 and newest-left order, f_Q loss, DIAG 0.2860 / ceiling 0.3110 agree with WILSON_REPORT.md"),
# Cameron: pass 1, incomplete
("Cameron","clean","SPJ/L1/cameron/test_costb_fused.py RED(author) then GREEN(fused)","pass 1, incomplete; not re-run: GPU; saved JSON equals lines (3.775, 2.487x; 2.21e-6, 1.041x), RED 23:07:26 before GREEN 23:07:56; costb_fused.py edited 23:11 after the fwd GREEN"),
("Cameron","clean","SPJ/L1/cameron/test_costb_fwdbwd.py RED(author) then GREEN(fused)","pass 1, incomplete; not re-run: GPU; saved JSON equals lines (1.634, 2.113x; 9.87e-7, 1.121x)"),
("Cameron","clean","finding COST leg survives 1.041x fwd, 1.121x fwd+bwd","pass 1, incomplete; both REDs precede both GREENs and finding; EFFICIENT_ATTENTION agrees with Wilson [V]; build flag in Wilson is flash_sdp_enabled(), no contradiction"),
("Cameron","clean","finding costb_impl builds Pi_i = q_0...q_i, fold err 3.775","pass 1, incomplete; RED test_costb_fused line precedes; order agrees with Wilson [V] costb_impl oldest-left"),
("Cameron","clean","SPJ/L1/cameron/test_bed_admissible.py RED on board_bed_result, shell_bed_floors, shell_bed_floors_long","pass 1, incomplete; CPU re-run: exit 1 on all three; current script adds A4 clauses and an A5 line not in the logged lines, RED unchanged"),
("Cameron","clean","finding: board bed retired (A3 0.9968 + 0.10 > 1.0)","pass 1, incomplete; RED test_bed_admissible on board_bed_result precedes; 0.996844 equals Wilson [V]"),
("Cameron","struck","finding: exact SU(2) construction on shell bed 1.000/1.000 (fq_construction.json)","pass 1, incomplete; fq_construction.py has no RED or GREEN event on the board; unbound"),
("Cameron","struck","finding: torch.compile Mode B' 1.069x, fused ratio vs S 1.132/1.065/1.035/0.978","pass 1, incomplete; no RED or GREEN event for costb_compile.py or costb_extra.py; saved JSON matches but unbound"),
]
if __name__ == "__main__":
    with open(B, "a", encoding="utf-8") as f:
        for c, v, n, w in E:
            f.write(json.dumps({"t": "audit", "agent": "Inspector", "cites": c, "verdict": v, "name": n, "why": w}) + "\n")
    print(len(E), sum(v == "struck" for _, v, _, _ in E))
