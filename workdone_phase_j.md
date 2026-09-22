# workdone — Phase J

**Every number carries its source file or commit. Where a claim was struck by an
audit it is marked STRUCK; where no instance exists it is marked NOT MEASURED.**

Span: 2026-09-21 to 2026-09-23. Machine: one RTX 4060 laptop GPU, torch 2.14.0+cu126
(this build has no flash attention; every "fused SDPA" figure is the
EFFICIENT_ATTENTION kernel). Records: `tests/foreman/phase_j/` — `RECORD.md`
(hours one and two), `RECORD_L.md` (Addendum L), `RECORD_M.md` (Addendum M),
`RECORD_N.md` (Addendum N), each with its producers and audit files.

---

## 0. The one-line answer

**The family's language-model win is a relative-position effect, not a property of
its operator.** Given a zero-parameter rotary position embedding, the plain softmax
twin recovers 96.3 / 99.7 / 101.4% of the family's win; given ALiBi it recovers
108.8 / 106.7 / 108.8% and beats the family outright at all three split seeds, at
about a fifth of its wall clock. Nothing else the operator adds has survived a proper
control, and on the trained model its equilibrium read sits past a pole.

---

## 1. The chain of results, in order

| step | result | source |
|---|---|---|
| Paired 4×5 grid (CRN, 5 split seeds) | C_win = (a) − (f) = **+0.2470**, 95% CI [0.2271, 0.2668]. Both named channels signed against themselves: C_phase −0.0124 (p_BH 0.018), C_mass −0.0376 (p_BH 0.020); C_resid 120.2% of C_win | `tests/foreman/design4x5/`, commit 01c4e03 |
| Wall clock | the family costs **6.72×** the twin at equal steps (514.8 s vs 76.6 s per cell, n = 5 each) | per-cell logs, 01c4e03 |
| Phase J hour one | a Forgetting Transformer gate (FoX, arXiv 2503.02130) on the twin recovers **100.2 / 100.3 / 110.9%** of C_win at 3.9× less clock; a per-token temperature recovers −0.006 (the abduction behind it died); a twin grown to 9.98M parameters beats the family by 0.167 at 44% of its clock | `RECORD.md`, 9e641d4 |
| Phase J hour two | abstention marks the tokens any forget gate wins (retired); the FoX float mask costs 1.99× an explicit-mask twin; three DO rows are **void** — one trained on a single chain (every arm on the null 0.07226), two read arms that never learned the task (150 steps, 1.94–1.97 nats vs an in-context count's 1.720) | `RECORD.md`, 6240946 → d6eecb9 |
| Addendum L round | filed **attempted at it.L0**: the contract's Q2 generators generate an infinite group (proved: the axes are 54.74° apart, an octahedral angle). The icosahedral pair closes at 60/120. The SU(2) value fold is exact and costs **1.03–1.09× forward, 1.08–1.09× forward+backward** with a fused scan (Inspector GPU re-run). Its order beds failed for a missing anchor and a memorising budget, not the arm | `RECORD_L.md`, 492ccff |
| Addendum M | the instrument recovers a planted drift (0.496 vs 0.5); the registered null could never pass; a tested replacement null scores 1.000 against a surrogate 99th percentile of 0.057 | `RECORD_M.md`, 63bda2a |
| Addendum N round N1 | the diagonal theory's mathematics reproduces at β = 1 on rebuilt instances; on the trained family it does not describe the operator (learned β 1.061 / 0.930 / 0.900, no exact gate zeros). H-SINK **struck** (P1 enrichment 0.0, P4 ρ −0.020). An exact causal-resolvent read costs **1.76–1.80×** one SDPA pass; the current degree-130 Chebyshev hop diverges on the real operator (relative error 4.46e4). Exact NEVER is a **certificate only**: SSMax with a threshold ties it at every length | `RECORD_N.md` |
| House's D2 audit | D2 holds for the complex operator at any β (≤ 4e−16); the first counterexample is the head's own real-part readout, under which phase is a signed attenuation, not a change of basis | `RECORD_N.md`, `N1/house/` |
| **The leap, bound (R-POS)** | RoPE twin **96.3 / 99.7 / 101.4%** of C_win (mean 99.2%); ALiBi twin **108.8 / 106.7 / 108.8%** (mean 108.1%), beating the family by 0.0206 / 0.0160 / 0.0231 nats; 724,608 parameters in every arm; CRN digests match the grid | `N2/foreman/`, bar registered in `RECORD_N.md` before the run |
| R-DIAG′ | the Mode B read at γ = 0.99 is **past a pole** in 1,432 of 1,536 (window, layer, head) pairs (93.2%); the worst self-weight is 2.068 (pole at γ* = 0.484). Pinning β = 1 makes it well-posed but costs +0.0222 nats at evaluation | `N2/chase/` |

---

## 2. What the tables permit

**One claim sentence.** An exact value-side SU(2) fold on stock SDPA (1.03–1.09×) and
an exact causal-resolvent read (1.76× one SDPA pass) exist as engineering. No
capability of the operator family has survived a control, and its language-model win
is a relative-position effect that ALiBi reproduces at about a fifth of the cost.

**What is owned by prior art** (each checked at its primary source by Wilson):
- the forget-gate win: FoX; the same gate law ships in Qwen3-Next's Gated DeltaNet;
- the output mass gate (arm a2): Qiu et al. 2025, gated attention, a NeurIPS 2025 Best
  Paper;
- non-commuting, content-dependent order at attention cost, including A₅ with two
  reflections per token: DeltaProduct (NeurIPS 2025), Kimi Linear's KDA, PaTH on the
  q·k side;
- the relative-position effect itself: RoPE and ALiBi.

---

## 3. North star, rescored

`docs/STATUS.md` counts named gates. R-POS moves condition 2's first gate: against the
strongest zero-parameter control (the ALiBi twin), the family has **no win**; it loses
at 3 of 3 seeds. The gates that were passed against the plain twin (more than one
domain, varied splits) must be re-measured against the ALiBi twin before they count.

| condition | before Phase J | now |
|---|---|---|
| 1 understands causality | 2.5 / 5 | 2.5 / 5 |
| 2 beats anything before it | 5 / 9 | **4 / 9** (gate 1 failed) |
| 3 carries the weight of attention or JEPA | 1.5 / 5 | 1.5 / 5 |
| headline, by count | 9 / 19 = 47% | **8 / 19 = 42%** |
| the goal as stated (weakest leg) | 30% | **30%** |

---

## 4. Failure taxonomy — mechanisms added in Phase J

- **A baseline audited for its trainable parts and never for its fixed structure.**
  The grid's softmax twins, (a) and (a2), carried position only through an absolute
  table, and the zero-parameter relative prior recovers the whole of C_win (the
  L-REFLECTOR law, broken by the project that wrote it). That C_win measured nothing
  but this absence is not test-bound (struck in N2); it is the reading R-POS invites.
- **Checks that could not pass** (now 4, beside the 16 that could not fail): a DO bed
  trained on one chain; two DO reads of arms that never learned the task; Addendum M's
  "coherence below 0.2 everywhere" null.
- **Bars written after their results** (post-hoc REDs): struck by the Inspector in every
  round — 18 claims in round L1, 13 in N1.
- **Records and citations:** a nurse's `cat >` redirect truncated the Addendum L record,
  losing 11 finished rows (restored from the journal snapshot); four citation rows
  marked "fetched" were built from a hard-coded dictionary; four sentences in pushed
  commit bodies were false (corrected in `RECORD_L.md`).
- **Numerics and harness:** data splits seeded with Python's salted `hash()`; an arccos
  angle readout with a 1e−8 floor; a shared helper overwritten by three concurrent
  lanes; test events logged with `state` instead of `status`; a test file that exits 0
  when every row fails; and, observed but not yet test-bound, `poll_until_free`
  counting the caller's own CUDA context as busy.
- **Contract text:** wrong generators (Q2), a tautological order test (Q3), the ½
  convention in the phase-field constant (1/3, not √2/6), "no custom kernel" (a Triton
  scan is used), and instance scripts cited but absent (`diag_leap.py`, `crossties.py`,
  `kr_density.py`, `pf_regime.py`).

---

## 5. Audit ledger

| round | fellows | Inspector | struck |
|---|---|---|---|
| L1 (Addendum L) | Foreman, Chase, Cameron, Wilson, Chase-rivals | 2 passes, 68 audited | 18 |
| N1 (Addendum N) | Foreman, Chase, Cameron, Wilson | 1 pass, 58 audited | 13 |
| N2 (the leap) | Foreman, Chase | 1 pass, 20 audited | 5 (neither headline) |

Dr House's fable run was disabled by the author this phase; the dispatcher made the
leap, and a fellow bound it with a test that was RED first.

---

## 6. What is owed

1. Re-measure every contrast against the ALiBi twin (a_alibi), the new zero-parameter
   control: the three domains, the abstention rows, and the grid itself.
2. Any claim for the operator needs a bed that a fixed recency prior cannot solve
   (long-range retrieval or copy), with the ALiBi twin run as its floor before a bar
   is written.
3. Mode B: read it with each head scaled to its largest self-weight (well-posed by
   construction), or retrain with β fixed at 1 and price the cost.
4. The guards from Addendum N §8a: every record write through one append-only helper
   with a test that fails if a record shrinks; every "fetched" citation backed by a
   stored artifact.
5. Fix `poll_until_free` to ignore the caller's own context.
