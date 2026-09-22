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

## Hour two — Dr House opening (15:11)

Condition 2 stays closed unless a capability a forget gate lacks reopens it. This hour buys that one test, prices the kernel, settles abstention. Hard stop 15:56.

**1. Slice (GPU queue in order; CPU rows beside it).**

1. **R-XFER** — Wilson, GPU 1 min. ss0 pair + ckpt_a2F_ss0; 32,704 sites ranked by (f)'s a_t; d_FoX = NLL_a − NLL_FoX. Bar: top-minus-bottom decile of d_FoX ≥ +0.015/+0.056/+0.117/+0.150/+0.366 in all five NLL_a quintiles. PASS retires the abstention rows; FAIL keeps abstention as (f)'s own until §4.
2. **R-K5** — Cameron, GPU 3 min. Same shape and steps, 50 fwd+bwd each, three paths: (a) is_causal (flash); (a) explicit boolean causal mask (math path, no gate); (a″) float mask. Bar: (a″)/(a-explicit) ≤ 1.30. PASS → the 1.72× is the path; K5 becomes a fused-kernel question priced at (a-explicit)/(a-flash). FAIL → the gate itself costs; (a″) misses K5 with (f).
3. **R-DO-LM** — Foreman, GPU 10–12 min. J2 bed (n=8, Dirichlet(0.5), basins 6,7) as token streams; arms (a), (a″), (f) at matched numel, d 64, 2 layers, S 128; seeds 0–2, then 3–4 only if an arm is within 0.01 of the bar. Read = (J)-style: linear probe of c_i from the last hidden state, fit on the observational chain, frozen; 50 do(i→row) worlds scored against `intervene.committor_do` (1.7e−16). Bar: (f) do-error < 0.0629 (the bag) AND (a″) ≥ 0.0629 → condition 2 reopened. (a″) within 0.01 of (f) → closed for good, "consequence" struck, Phase J is a kernel phase. Neither beats the bag → NEITHER; the read is the floor.
4. **R-DPI** — Chase, CPU 3 min, beside row 3. Same 50 worlds; input-only controls: visit bag (0.0629), k-gram read (k = 1..3). Bar: any PASS in row 3 beats the best control by ≥ 0.02, else the DO win is a histogram and row 3 is void.

Wilson checks every number before the close.

**2. Struck this hour.**
- R-JEPA: no predictor, no Mode B; built in 45 min it is a bed failing at its floor.
- R-CURV re-run: −0.732 unpinned; nothing runs until the author pins it or takes −0.774.
- (a_T) seeds 1, 2, any R-TEMP variant: dead.
- K1 bitwise vs flash_attn: absent; stays 1e−6 fp32 (amendment 6).
- R-FACE, R-HOLE, R-VERB-EP, R-CLOCK, R-PHASE, R-KNN: no trained c-space until row 3 makes one.
- it.J6, Kaggle: out.

**3. Fields.** All stand. Add `clock_ratio` (arm s / (a) s at equal steps, every GPU row); `read` ("probe"/"exact"/"bag"/"kgram", rows 3–4); `queue_pos`, `started`, `finished` (local time, every row).

## Hour two — Dr House close (15:56)

**1. Verdicts.**
- R-XFER PASS: +0.0305/+0.0994/+0.2759/+0.2730/+0.7555 over +0.015/+0.056/+0.117/+0.150/+0.366; control d_f reproduces. Abstention marks sites any forget gate wins.
- R-K5 FAIL: 1.986 > 1.30. And no flash in this torch build; is_causal ran mem_efficient.
- R-WORLDS PASS: 4.4e−16, 50 worlds.
- R-DPI: floor 0.009702 (1-gram). Bag here 0.01504, not the 0.0629 my spec carried.
- R-DO-LM VOID, not FAIL; "closed for good, consequence struck" does not stand. a 0.07219 / a″ 0.07225 / f 0.07229 on a no-update null of 0.07226: every arm sits on the null to four decimals, so the row measured the bed. Both faults mine: "J2 bed as token streams" is one P_obs, so no context varied and no arm was trained to read one; and a 0.0629 bar over a 0.0097 floor made "beat the control by 0.02" unsatisfiable before a step ran. A check that could not pass joins the 16 that could not fail. Condition 2 stays closed as at 15:11, pending a bed that can pass.

**2. Dead this hour.**
- Abstention rows: retire (R-XFER).
- R-DO-LM as run: reroute → R-DO-IC.
- 0.0629 bag bar: reprice → §4 floors.
- "1.72× is the path": retire (1.99× at explicit mask).
- Every "flash" cell in the ledger: reprice → relabel mem_efficient.

**3. Kernel.** Float mask 1.99× boolean, 2.84× is_causal, all inside mem_efficient. Hour one's 3.9× is (f) over the twin's 76.6 s, also mem_efficient; a flash denominator only widens it, so 3.9× is a floor on the gap. (a″) alone is 2.84× of it; the family's residual over FoX is ~1.4×. (a″) misses K5 with (f) per the row's rule, but K5 as written, "within X of fused SDPA", cannot be scored on this box by any arm, twin included. Verdict deferred to a flash build (a flash wheel needs no yes; Kaggle does).

**4. Next row: R-DO-IC.** Foreman, GPU 12 min; floors first (CPU). Bed: 512 training streams, each from its own fresh Dirichlet(0.5) chain (n=8, basins 6,7), S 128, target its own chain's committor; probe fit across the 512 chains, frozen. Eval: 50 do() worlds, each read from a stream drawn from P_do. Floors before any arm trains: (i) no-update, mean training committor; (ii) 1-gram MLE, 0.009702; (iii) Dirichlet(0.5) posterior-mean plug-in from the same 128 tokens, the strongest input-only reader and the real floor. Bar: (f) ≤ 0.8×(iii) on 5/5 seeds AND (a″) > (iii) → condition 2 reopened. (a″) ≤ 0.8×(iii) too → closed for good. Nobody clears (iii) → DO retires: a three-line estimator beats every arm.

## R-DO-IC — Dr House ruling

**1. DO RETIRES.** Nobody clears floor (iii). Plug-in 0.07535; a 0.10536 / a″ 0.09544 / f 0.09855, 5/5 seeds. The best arm misses by 0.020; (a) sits on the no-update null; 1-gram counting (0.09221) beats all three. The stream deviation is recorded, not a strike: on the arms' exact streams the floors move DOWN (plug-in 0.07412, 1-gram 0.09032) and the margin widens to 0.021.

**2. What retired, narrowly.** This read: a frozen linear probe on the last hidden state, ~109k params, S 128, one stream, fixed steps. At that read a three-line Bayesian count of the same 128 tokens beats every arm. NOT licensed: striking "consequence", or closing condition 2 for good. That kill needs (a″) within 0.01 of (f) on a bed some arm beats; here nobody beats anything. Condition 2 stays closed as at 15:11, by absence, not by a passed kill. In-distribution probe error 0.11–0.14 exceeds the do-error: the probe is the floor before the arm is.

**3. Ordering.** a″ < f < a. Forget gate best, family second, every gap inside a band a 1-gram count beats. It says one thing: the family shows no DO capability a forget gate lacks.

**4. Standing state.**
- Condition 2 closed; consequence unstruck; DO-as-probe retired; the family's live claim is the operator frame at 1.084×.
- Kernel: FoX 2.84× the twin, family ~1.4× over FoX, 3.9× a floor; K5 deferred to a flash build.
- Abstention retired; R-TEMP dead.

Next hour: R-DO-E2E. Same bed and floors; the committor head trained end-to-end on the 512 chains, not frozen; bar (f) ≤ 0.8 × 0.07412 on 5/5 AND (a″) > 0.07412. Miss 0.07412 with a trained head and DO is dead at every read; Phase J closes as a kernel phase.

## R-DO-E2E — Dr House ruling

**1. Verdict.** "DO dead at every read" overclaims. Both DO rows read arms at 1.944–1.972 nats against an online count at 1.720; an arm that never learned the chain has no chain in its state, so probe and head returned the same nothing. The rows measured the step budget, and the budget was my silence: the spec copied R-DO-IC's arms and named no training floor. Corrected verdict: **DO untested at every read.** Phase J still closes as a kernel phase: the forget gate is the only survivor and no DO row had an admissible arm.

**2. Gate.** No DO read counts unless the arm's held-out chain NLL ≤ 1.720 (online Dirichlet(0.5) count), logged beside every do-error, and the read beats the count's plug-in 0.07412. Cost: today 150 steps, 10 s per cell. Recovering 0.36 nats of in-context estimation is a 20–100x budget: 3k–15k steps, 3–17 min per cell, 15 cells ≈ 1–4 h on the 4060. A plateau above 1.720 means the gate is unmeetable at this width.

**3. Standing state.**
- Condition 2: untested, not dead; every DO read so far was of an arm above the count.
- Phase J: closed as a kernel phase; forget gate stands; DO rows voided as step-budget measurements.
- Bar for any DO row: NLL 1.720, plug-in 0.07412.

**4. Next row.** R-DO-STEPS, outside Phase J: arm f, one seed, train to NLL plateau, logging every 100 steps. Clears 1.720: rerun the R-DO-IC probe on that checkpoint. Plateaus above: condition 2 unaskable at this width; closure final.

## Dr House leap (after the close) — kernel schedule, OPEN

Prior art handed in: triton-lang/kernels#22 (author's, merged 2026-07-28), a
forward-only Triton online-softmax kernel over causal CSR block schedules whose
top-k persistence schedule is approximate (max abs error 0.0009).

**F.** Every accuracy arm ties FoX, and the one property FoX provably lacks — an
exact zero — was only ever run on dense kernels, where by Lemma E2 a closed gate
changes nothing but cost. **H.** The family is a schedule generator for #22: the
continuous part of m is FoX's per-key bias; the exact-zero part is a learned
segment boundary the CSR kernel consumes with error exactly 0. Its claim is
condition 3 (cost), not loss. FIXED: the merged kernel, block 64, CSR, causal
line (theta = 0 by W5). TRAINABLE: one segment flag. Phase, mass, temperature,
sinks and persistence salience deleted.

**Instance, dispatcher-recomputed (S 4096, block 64):** dense causal 2080 blocks;
fixed 512-token segments 288; fixed 256-token 160 — all three exact as House
stated. Random flags at the measured density 0.0024/token visit **478 on average
(5–95%: 313–824, 200 draws)**; House's single draw was 463. Correction to his text:
that is MORE than the persistence schedule's 398, not fewer — at the trained
density, learned flags would visit more blocks than #22 already does, though at
exact rather than 0.0009 error.

**Kill, by House:** the existing trained densities (0.0/0.0024/0.0/0.0 by layer;
a separate checkpoint truncating at ~1 token) make H dead at both, unless an
intermediate density is learnable [U]. **Row R-SEG:** FoX numerics + one learned
flag with an L0 target of one flag per 512 tokens at S 4096, 3 seeds; control:
fixed block-aligned 512-token segments (288 blocks). Bar: (a) learned flags beat
the fixed control by more than seed spread at equal visited blocks, AND (b) stay
within seed spread of dense FoX. Fail (a): the family is a block-diagonal mask the
kernel already ships. Fail (b): exact skipping bought nothing a window does not.
Engineering note: #22 has no backward, so training runs dense with the mask and
cost is priced forward through #22.

Status: hypothesis, unbound. Enters Open until a fellow binds it with a RED row.
