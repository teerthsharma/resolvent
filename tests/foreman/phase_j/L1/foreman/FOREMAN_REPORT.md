# Foreman — final report (received ~23:30 local), claims as filed

All tests and outputs in this directory (SPJ/L1/foreman). Foreman states every event is on the board with each RED before its GREEN. No nurses, no GPU used.

Verdict as filed: the arm did not cause either order bed to fail; the SU(2) leap survives it.L1. Fold computes what FQ-ARM certified. A5 bed fails for want of an anchor plus budget; I-AUT tests S5, which no SU(2) gate can hold. Anchored A5 with more data: (f_Q) 1.0 at position 64 vs FoX 0.0156, one seed.

Ruled out: generalised arm breaks the fold (hand-set as-built a5_bed.Layer, 4 heads, d 64, attention forced to key 0 reads 1.0 at 16/32/64 — a5_handset.json, T2a); quaternion convention differed (live su2.py byte-identical to repo, mtime 17:07:03 before runs; a 2I lift valid on all 60 elements exists under su2.qmul and the bed's order).

F1 A5 has no anchor. Fold G_ij = Pi_i conj(Pi_j) covers j+1..i, so the running product needs attention to key 0; a5_bed.py has no BOS, token-only embedding, single-frequency _rope (ang = pos*0.5, period 12.57): offsets i and i-25 differ by 0.066 rad.
- RED a5_handset.py: "RED T2b: as-built SDPA, same hand-set lift, acc@64=0.0000 all=0.0491 (<0.9)"
- RED a5_key0_test.py: "RED: optimised as-built layer-1 attention reaches key 0 with p=0.0137 at query 63 (bar 0.9)" (800 steps on log p(i->0), stayed uniform)
- GREEN by reroute a5_anchor_train.py (BOS, 512 train seqs, 400 steps, seed 0): anchored 1.0 at positions 1-64; unanchored decays 0.95@4, 0.83@8, 0.56@16, 0.39@32, 0.109@64; controls on anchored bed softmax 0.0312, FoX 0.0156 @64. Source a5_anchor_train.jsonl.
- Kills L1-A5 FAIL as evidence against (f_Q). Reroute L1-A5': a5_bed + BOS (Embedding(5,d)), 512 train seqs, 400 steps, loss on original 64 positions, stable seeding, 3 seeds; arms (a),(a''),(f_Q); bar unchanged (f_Q >= 0.90, a'' <= 0.20). Seat Foreman.

F2 budget too small; recorded (f_Q) loss 0.006 is memorisation. At 64 seqs/150 steps even the anchored arm reads 0.0156 held-out @64.
- RED a5_handset.py: "RED T2c: trained layer-0 generators are no A5 lift: inverse_pair_err=1.183 order3_err=1.078 (bar <0.1)". Retrained seed 0 train acc 1.0, eval 0.0588.
- Reprice: every order row reports held-out accuracy plus a group-lift check (inverse-pair and order-3 errors) beside train loss. Seat Wilson's checkers.

F3 I-AUT: no SU(2) gate can reach the row's bar. Group is S5; the only involution in SU(2) is -1 (central), so every hom S5 -> SU(2) factors through sign. Key 0 reachable (p 0.9999 mean, 0.9996 min, iaut_key0_check.py). As-built linear readout of exact anchored product 0.2625/0.2862/0.3225 (mean 0.2904); quadratic features 0.40-0.44 (iaut_linear_readout.json).
- RED iaut_bar_test.py: "RED: best finite-SU(2) Bayes ceiling per seed {'0': 0.67375, '1': 0.64, '2': 0.66625}, mean 0.6600 < L1-IAUT pass bar 0.8120" (all 2I and 2O pairs, iaut_su2_ceiling.json).
- Kills L1-IAUT as a test of (f_Q). Reroute I-AUT-A5: same integer-only construction (nearest-anchor labels, 3000/800, 900 steps), automaton over A5 from generators (0 1 2),(2 3 4); arms operator (c), DIAG, (f_Q) at matched numel, same one-hidden-layer readout, 5 seeds. Seat Foreman.

F4 L1-A5 record not reproducible: gen_split seeds numpy from hash((split, seed)), salted per interpreter.
- RED a5_repro_test.py: "RED: gen_split('train',0) differs by PYTHONHASHSEED: [3, 1, 1, 0, ...] vs [2, 1, 0, 0, ...]"
- Not fixed. Reroute: random.Random(f"a5|{split}|{seed}") as iaut_race.gen_words does. Foreman's own anchored (f_Q) runs were unpinned; controls ran PYTHONHASHSEED=0.

Open (as filed): own hypothesis (f_Q capped at 0.3110 on I-AUT) died — iaut_su2_ceiling.py GREEN on first run, best 2I pair 0.674/0.640/0.666, no RED so not a finding. L1-A5' one seed per arm, fq and control runs drew different data. Self-anchoring arm not built. I-AUT bound partial (2I and 2O pairs only; not binary dihedral, infinite-image, or product plus RoPE path). L1-IAUT compared d 64 arms (13,258-13,518 params) against 404-param certified DIAG; DIAG not rerun at d 64; (a) 0.2333 vs certified 0.2285. Contract §1 writes two-sided Pi_i (conj(Pi_j) v_j Pi_j) while arm_fq.py and its oracle are left-only; FQ-ARM's check cannot tell them apart; unresolved which the contract means.
