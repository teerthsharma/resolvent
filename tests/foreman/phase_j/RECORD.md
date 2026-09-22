# Phase J — process record

Opened 2026-09-22 05:27 local. Contract: `../phase_j_contract.md` (author's, v2).
Every row appends ONE JSON line to `record.jsonl` in this directory, with the
fields below. This file is the head of the record; Dr House writes its close.

## Fields (Dr House, section D — every row)

- `row`, `arms`, `seeds`, `split_seed`, `machine` ("RTX 4060" or "CPU"),
  `seconds_per_cell` (wall clock), `tokens_seen`, `params_numel` (per arm).
- `bar` (as pre-registered), `measured`, `verdict` (PASS / FAIL / NEITHER),
  `control` (the control's number beside it).
- `quintile_profile` — d_t by NLL_a quintile, absolute and relative (paired LM rows only).
- `producer` (command) and `output` (path); `repro_class` ("operator bitwise" / "SDPA tolerance" / "exact numeric").
- `killed` — one sentence: what the row killed or failed to kill, no mechanism words.

## Dr House ruling, 05:25 — APPROVE WITH AMENDMENTS

**Abduction.** F: stratified by the twin's own NLL, the gap is −0.0049 on the
easiest fifth and +0.6763 on the hardest. H: the gate is a data-dependent
temperature. The twin's row sums are FIXED at 1; in (f) the TRAINABLE gates set
|ΣW| < 1, shrinking the attention output on exactly the tokens the twin finds
hard. A temperature cannot gain at the NLL floor and gains most where the twin is
confidently wrong. His instance: p_true 0.95 → 0.05 under global T = 1.5 gives
d = −0.080 easy / +0.901 hard; T = 1.2 gives −0.031 / +0.460 — the observed easy
fifth (−0.0049) is too small for any global T, so the temperature is adaptive.
Kill row: (a_T), the twin plus one Linear(d, 1) per-token temperature on the
output logits; C(a_T) ≥ ½·C_win kills "consequence" in favour of "adaptive
calibration."

**Amendments.**
1. §0: abstention test ran, verdict NEITHER, control failed; quintile profile recorded.
2. Every paired row reports the NLL_a-quintile profile of d_t and d_t/NLL_a.
3. Add arm (a_T); row R-TEMP: C_temp = (a) − (a_T), prediction ≥ ½·C_win.
4. Strike "632,496 params, 0.696×"; measured shape a 724,608 / f 725,391 / a2 727,704 at 0.798 epochs.
5. Strike MDP-CAL (does not exist); the calibration bed is the J2 planted-chain generator.
6. K1 becomes rel err ≤ 1e−6 fwd / 1e−5 bwd against the SDPA math backend in fp32; bitwise only for the operator path. No flash_attn on this box.
7. R-CURV exact transport via scipy.optimize.linprog (POT absent).
8. R-COMP: 6.72× steps = 97.4M tokens = 5.36 epochs — a repetition confound. Primary form = parameter-grown twin at matched wall clock, same steps.
9. Reproducibility split by arm: operator bitwise, SDPA arm to 3.4e−6.
10. it.J6 and Kaggle are out of this hour.

**Dispatcher correction to the slice:** row 1 pointed at `results/*sd0*.pt`, which
lanes may not touch; the matched pair is `../d45_ckpt_{a,f}_ss0_pair/model.pt`.

## One-hour slice (House's order)

1. Wilson — re-stratify on the saved pair: NLL_a quintile means, d/NLL_a, Spearman(a_t, twin predictive entropy). GPU eval only.
2. Foreman — (a_T) seed 0. 3. Foreman — (a″) FoX seed 0.
4. Foreman — (a_T), (a″) seeds 1, 2 only if seed 0 is within 0.05 of C_win.
5. Foreman — (a_c) parameter-grown at matched wall clock: probe, then one cell.
CPU in parallel: Chase — J1, J2, T1, C11, R-LIGHT. Cameron — ORC via linprog.
Wilson — three checkers on every number.

## Dr House close

**1. Verdicts.**
- R-STRAT — SURVIVES: ρ(a_t, NLL_a) 0.0125, ρ(a_t, entropy) 0.0036; top-minus-bottom a_t decile +0.030/+0.112/+0.233/+0.300/+0.732 in every NLL_a quintile.
- R-TEMP — FAIL: C_temp −0.00595 vs bar +0.1163.
- R-FoX — PASS ×3: C_forget 0.2330/0.2398/0.2904 vs C_win 0.2326/0.2390/0.2619 (100.2/100.3/110.9%), 727,704 params, 131.9 s vs (f) 514.8 s.
- R-COMP — PASS (prediction): (a_c) 9,976,320 params, 226 s, loss 0.8759 vs (f) 1.0431; C_win inverts to −0.1672 at 44% of (f)'s clock.
- J2 — PASS: Sherman–Morrison 1.7e−16; bag error 0.0629 vs contract 0.020.
- T1 — PASS: EP 0.90417, curl carries all, DB chain 6e−33.
- R-LIGHT — PASS: 4.0e−14, K=142 at ρ 0.8.
- J1 — PASS direction only: 0.2856/0.0280, not 0.2537/0.0143; convention unstated.
- C11 — PARTIAL: 11/34/130 each reachable; no single (C, ρ, ε).
- R-CURV — NEITHER: −0.7738 on the pinned bed vs −0.732, an unpinned bar.
- Integrity: 0 rows struck; CRASH, duplicate J1/T1, self-kill all recorded.

**2. Dead this hour.**
- My abduction. "The gate is a data-dependent temperature" — my own kill row fired: a per-token temperature gains −0.006 against ≥ +0.116. Retire.
- "Consequence" as the cause of C_win. A FoX forget gate at equal params takes 100% of the win on 3/3 seeds at 1/3.9 the clock, same quintile profile (−0.0065/0.042/0.147/0.298/0.684 vs (f) −0.0049/0.043/0.155/0.294/0.676). Reprice: (f) is a forget gate at 3.9× the clock.
- C_win as a compute-fair number. Grown toward (f)'s budget, the plain twin wins by 0.167 short of matched clock. Reprice: every grid C_win is a mismatched-clock figure.
- Three contract numbers: J2 bag 0.020, J1 0.2537/0.0143, ORC −0.732. Reprice each to the measured value or state the convention.
- C11's examples. Reroute: state the triple or drop them.

Abstention is alive but orphaned: it survives difficulty, yet FoX has no row-sum deficit and matches (f) quintile for quintile.

**3. Contract changes.**
1. Control is (a″), the FoX-gate twin, not softmax (a); every bar against (a) is void.
2. Headline contrast is wall-clock-matched: (f) −0.1672 against (a_c); C_win 0.2326 is a footnote carrying "3.9× clock".
3. §0 "consequence" prediction withdrawn; R-TEMP retired; abstention rows live only until §4 runs.
4. J2 bag 0.0629; J1 convention stated or 0.2856/0.0280; ORC "orientation figure, α=0.5 lazy walk, −0.774 on the pinned bed"; C11 triple stated.
5. R-COMP retitled: capacity, not repetition, explains the gap.

**4. Next row: R-XFER.** On the ss0 pair plus ckpt_a2F_ss0, rank the 32,704 sites by (f)'s a_t and read d_FoX = NLL_a − NLL_FoX. Bar: top-minus-bottom decile of d_FoX ≥ ½ × (+0.030/+0.112/+0.233/+0.300/+0.732) in all five NLL_a quintiles → abstention marks sites any forget gate wins, retire the abstention rows; otherwise abstention is (f)'s own. Cost 2 s GPU.
