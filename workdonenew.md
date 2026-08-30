# workdonenew — where CEQ actually stands

**Written at v-main.3M iteration 20, branch `feat/r9-causal-consequence`, pinned at
`bf2a769`.** Every number here carries its source. Where a number is a bound, a
prediction, or a single draw, it says so. Where a claim was withdrawn, it is listed
under withdrawn and not quietly dropped.

---

## 0. THE ONE-LINE ANSWER

**The engineering is strong and the architectural claim has not been made.**

`done7.md` scored the product at **37%** — engineering ≈ 80%, scientific claim
≈ 5% — and named the reason in one sentence: *the deciding measurement had been
taken zero times*.

**Round 10 took it.** That is the change. The scientific claim moves from *never
measured* to *measured, bounded, and with its boundary located*. It does **not**
yet move to *CEQ beats softmax*, because no CEQ arm has been run on the corpus that
now exists.

What you have is a **calibrated instrument and a mapped playing field**. What you
do not have is a **result on it**.

---

## 1. WHAT THE THING IS TRYING TO BE

Not the best token predictor — the **next-equilibrium predictor**. Attention that
understands causality and consequence on Turing-grade problems at the smallest
scale, with the whole attention module built to survive equal to self-attention or
supersede it. Trainable on free tiers, with a capability table.

The scientific content sits in one word: **equilibrium**. The oracle is an
absorbing-chain solve, `N = (I − Q)⁻¹`, `B = NR`, the fixed point of
`z ← Qz + R` — an equilibrium in the exact sense the goal names, and provably not
a closed form of any bounded neighbourhood.

---

## 2. THE REPO AS IT NOW STANDS

| directory | contents |
|---|---|
| `ceq/` | 26 modules — the architecture itself: `attention.py`, `arms.py`, `hankel.py`, `multizoom.py`, `nash.py`, `rips.py`, `eviction.py`, `hopcache.py`, `lm.py` |
| `scale/` | 104 modules — harnesses, gates, oracles, pricing. This is where the measurement lives |
| `tests/` | 200 files, of which `tests/loop/` holds **29 guards** written to catch instrument defects |
| `lean/` | 6 CEQ proof files, machine-checked |
| `results/` | journals — every published number's provenance |
| `attic/` | retired code, kept resolvable: guards look here before reporting a file missing |

**Ledger:** 10,809 events, 425 findings, 141 adjudications. Parse it with
`scale/ledger.py`; **never grep it** — `json.dumps` writes a space after each
colon and a literal-string grep silently returns zero.

**Suite:** 499 passing in `tests/loop`, **15 failing** — and every failure is a
bound finding with a stated route, not breakage.

---

## 3. MATHEMATICS PROVED — machine-checked in Lean

`lean/CEQ/` carries six files. These are theorems, not measurements.

| file | theorems |
|---|---|
| `Nilpotent.lean` | `pow_entry_zero`, `pow_card_eq_zero`, **`occupancy_is_exact_inverse`**, `one_not_nilpotent` |
| `Occupancy.lean` | `occupancy_telescope`, `occupancy_telescope'`, **`occupancy_eq_inverse_of_nilpotent`** |
| `Contraction.lean` | **`weighted_contraction`**, `weighted_contraction_iterate`, `rowStochastic_perron`, `expander_rowStochastic`, `expander_expands_l2` |
| `OrbitBound.lean` | `card_agree_add_card_errors`, `injOn_agree`, `card_agree_le_orbits`, **`orbit_error_bound`**, `orbit_error_bound_attained` |
| `Refcount.lean` | `mem_fibre`, `one_le_refcount`, `floor_add_orbits`, `floor_eq_sum_refcount_pred`, `card_survivors_add_refcount` |
| `OracleSeparation.lean` | `one_nonneg`, `mul_nonneg'`, `pow_nonneg'`, `diag_sq_le`, `diag_pos_double` |

**The load-bearing one is the resolvent.** `occupancy_eq_inverse_of_nilpotent` and
`occupancy_is_exact_inverse` establish that the successor representation
`M = (I − γP)⁻¹ = Σ γᵗPᵗ` **is** the nilpotent resolvent, exact in `n` terms. That
is the mathematical spine: the architecture's claim to compute equilibrium rests on
an identity that is proved, not fitted.

`weighted_contraction` is cited by the corpus's dose theorem — and see §6, because
its **norm** turned out to matter.

---

## 4. MATHEMATICS MEASURED — numbers with provenance

### 4.1 The deciding measurement, Round 10

**Question:** can softmax learn the `t* > hop` region at all? If not, no CEQ result
in that region means anything.

Pre-registered rule (v-main.3M it.11): *region learnable iff ≥1 cell < 1.0 with an
**N=8 seed CI** excluding 1.0, bootstrap B=10⁴*.

| t\* | 1-hop ceiling `√((t*−1)/t*)` | cell | seed CI (N=8) | verdict |
|---|---|---|---|---|
| 2 | 0.7071 | n=2048 | [0.9462, 0.9591] | **LEARNABLE** |
| 8 | 0.9354 | n=32768 | [0.9739, 0.9768] | **LEARNABLE** |
| 32 | 0.9843 | n=32768 | [1.000851, 1.005679] | NOT LEARNABLE |
| 32 | 0.9843 | **n=49152** | **[0.997535, 0.999930]** | **LEARNABLE** |

**Every block crosses. The wall is DATA, not architecture.** The crossing point
rises with `t*` — n=2048, n≈13–16k, n≈40k — as the ceiling headroom falls: 0.29,
0.065, **0.0157**.

**Softmax's hop budget is 1** — `scale/e_ladder.py:55`, `HOP_BUDGET = {"softmax": 1,
"glance": 1, "settled": 2, "twin": 2, …}`, an enforced constant consumed at `:202`
to compute the very ceilings above. (Not `m3_capability.py:271`, which is a print
statement; that citation was struck.)

### 4.2 What the shape of the data says

- **Steps hurt, data helps, monotonically.** `dNRMSE/d(steps) > 0` in **12 of 12**
  adjacent pairs across 6 eligible cells; smallest +0.087107, largest +0.646806.
  **All 15 sub-bar cells sit at the shortest rung (150 steps).** A budget spent on
  steps rather than examples would have concluded the region unlearnable and been
  wrong.
- **Seed variance collapses faster than sampling noise.** sd 0.003759 → 0.001804
  between n=32768 and n=49152, a factor **2.08** where `1/√n` predicts 1.22.
  Consistent with part of the spread being under-training that data removes.
- **A natural theory of the crossing over-predicts by ~6×.** If the crossing were
  set by *resolving* the shrinking margin (`m ~ 1/2t*`, `n ~ 1/m²`), then `t*` ×4
  would cost `n*` ×16. Measured: **~2.5×**. **Softmax's difficulty at large `t*` is
  not primarily margin-resolution, and the round did not identify what it is.**

### 4.3 Arms, from Round 3 (a different task — negation-scope, not the chain)

At `n_train=8192`, all params matched at **4769**:

| arm | eval NRMSE | 95% CI |
|---|---|---|
| softmax | 0.877168 | [0.830455, 0.924226] |
| pivot_unsigned | 0.747528 | [0.696849, 0.797716] |
| pivot_signed | 0.673762 | [0.632559, 0.715564] |

**Routing beat softmax with disjoint CIs.** This is the strongest pro-CEQ number in
the repo — **and it is on `negation_scope`, not on the equilibrium chain**, so it
does not transfer to the claim in §1 without being re-run there.

`ARMS = ("softmax", "pivot_signed", "pivot_unsigned", "windowed_signed")`,
`scale/m3_capability.py:66`.

### 4.4 The corpus (Phase 1b)

Harmonic extension on graphs: `u(v) = mean of neighbours` in the interior,
`u|_B = g`; oracle `u_I = (I − P_II)⁻¹ P_IB g`.

| measurement | value |
|---|---|
| dual-oracle max disagreement | **9.99e-16** (it.14), **1.554e-15** (it.15) against an imported 1e-10 |
| planted `deg+1` rejection | 29/29 absorbing-side, 17/17 Kirchhoff-side |
| **shared plant rejection** | **0/17, max gap 3.331e-16** |
| λ₂ stratification yield | naive **16/200 = 8.0%** into `[0.90, 0.95]` |
| rung coverage | all 12 in **[0.9435, 0.9499]** — the **top fifth** of the declared band |
| k-hop arm, untrained | `khop_128` reads **0.005092**; crossing of 1.0 between k=32 and k=64 |

---

## 5. CLAIMS DISPROVED, WITHDRAWN, OR STRUCK

This is the section that matters most, because these are the ones that would
otherwise still be believed.

| claim | status | why |
|---|---|---|
| **"Pivot routing beats dense because of SIGN"** (G4) | **WITHDRAWN, not overturned** | at the harness geometry, `_causal_sgate_operator(lam=0.10)` is entrywise **non-negative**, min entry exactly 0.0. The paired test compared two non-negative operators. **`pivot_signed` was `pivot_unsigned` wearing a name.** λ is a threshold at 1.0, not a dial |
| **"No arm has ever passed the bar"** | **RE-SCOPED** | it was a **data-budget** fact, not an operator fact. Every prior reading was at `n_train=128`, where the harness cannot produce a pass. At 8192 all three pass |
| **"The label is `N(0, t*+1)`"** | **CORRECTED to `N(0, t*)`** | the operator's nilpotency index is `t*+1`, but `b[s-1] = 0` kills the `m=0` term. Measured: mean Var(y) = 2.002719 / 7.984942 / 32.069818, excluding `t*+1` by **9.8 half-widths**. Five sites said `t*`; one said `t*+1`; five files were corrected |
| **"`err(t) ≤ λ₂ᵗ`"** (the dose theorem) | **FALSE AS WRITTEN** | violated **6 of 12** in `l_∞`, worst ratio 1.290631. On `g ≡ 1` it fails by construction. **True and tight in the degree-weighted 2-norm** (12/12, worst slack 7.00e-04). That norm is the amendment |
| **"The dual oracle certifies the corpus"** | **BOUNDED** | `I − Q = D⁻¹L_T` is diagonal, so both routes share adjacency and degrees. A defect in the graph data is **invisible to both** |
| **"C-F fails 12/12"** | **STRUCK to 1/12** | `δ = 0.5` was imported from the chain task's clause. `fd_max × |B| ≈ 3.741` (harmonic measure), so `fd ≥ 0.5` needs `|B| ≤ 7.5` while the corpus runs 4–80. Ten of twelve cannot reach it **by construction** |
| **"`R ∩ D = ∅` identifies scope defects"** | **NECESSARY, NOT SUFFICIENT** | fired 19 of 31, and **17 were not defects** — a refusal guard is *supposed* to be exercised on inputs production cannot emit |
| **"`4·n·s·d·heads` prices peak activation"** | **WRONG TENSOR, 64× low** | the peak is the `[n,s,s]` operator, not the projection. At batch 8192, s=1024: 512 MiB claimed, **32,768 MiB real** — four times the card |
| **JUPITER's "the curve asymptotes at 1.00185 and never crosses"** | **REFUTED** | measured CI at n=49152 lies entirely below 1.0 |

---

## 6. WHAT ROUND 10 ADDED THAT WAS NOT THERE

**A calibrated field.** Before: one task, one budget, no seed discipline. Now: the
learnable region is mapped in `(t*, n, steps)` with N=8 seed CIs, and the boundary
is located to within an octave.

**A corpus, unregistered.** Built, dual-checked to 1e-15, with its dose theorem
corrected and its band coverage honestly reported as the **top fifth** rather than
the whole. **NOT REGISTERED** — four of six admissibility clauses fail.

**Instruments that refuse.** `it11_verdict.py` (refuses below N=8, counts distinct
seeds not rows, fixed thread count), `idle_gate.py` (UNKNOWN ≠ idle, level-triggered),
`vram_gate.py` (prices host RSS, reports UNKNOWN rather than GREEN).

**A defect taxonomy.** `R10_MECHANISM.md` — one mechanism, 23 instances, split
instrument-side from reader-side. `MISTAKES.md` gained **V-14a, V-15, V-16, V-17**
and went from seven checks to ten.

---

## 7. HOW FAR FROM A NOVEL ATTENTION MECHANISM

Honest accounting.

**DONE:**
1. The mathematical spine is **proved** — the resolvent identity, machine-checked.
2. The playing field is **measured** — softmax's reachable region, with its
   boundary, at N=8.
3. An oracle exists that is an equilibrium in the exact sense, with an executable
   second route.
4. The engineering — harness, gates, journals, pricing, provenance — is genuinely
   strong. This is the 80%.

**NOT DONE:**
1. **No CEQ arm has been run on the equilibrium corpus.** The corpus is built and
   not registered; the arms exist and have not met it. This is the whole gap.
2. **The strongest pro-CEQ number is on the wrong task.** Routing beat softmax with
   disjoint CIs on `negation_scope`, not on the chain.
3. **The one signed-vs-unsigned result was withdrawn.** What "routing" buys is
   currently unexplained: the signed arm was the unsigned arm renamed.
4. **S3/S4/S5 are unbuilt.** Trained selection, equilibrium readout, action head —
   Phase 2 is gated behind v-main.3M it.36.

**The distance, in one sentence:** the field is measured and the instrument is
calibrated, so a CEQ result on the chain corpus is now a **runnable experiment**
rather than an unanswerable question — and until it is run, the scientific claim
stays where `done7.md` put it.

---

## 8. WHERE DEVELOPMENT REMAINS — ranked

1. **Declare the tensorisation.** *Does the arm see boundary values?* On geometric
   features the corpus is admissible **28/28**; adding `gbar` makes it **25/28
   inadmissible**, because `gbar` is literally the first Neumann term of
   `u = P_IB g + P_II u`. **This decides whether the corpus is a test at all.**
   The round proceeds assuming it does **not** — if wrong, the corpus is rebuilt,
   not re-declared, and downstream S1/S2 results are void.
2. **Declare the train/eval split.** By arm gives **0** shared graphs; by file gives
   12 and 43.66% leaked. Free to fix.
3. **Run a CEQ arm on the chain corpus.** The gap in §7. Everything else is
   preparation for this.
4. **Fix `admissible`** — one clause, the mirror of the one there: every boundary
   vertex needs an interior neighbour. One instance of twelve currently has a
   `do()`-bit with **zero** effect.
5. **Score the bar's control held-out.** Measured harmless (all three blocks pass
   held-out by ~38×), but it is the *only* one of `bar_verdict`'s five clauses that
   tests learnability, and it is in-sample.
6. **Explain what routing buys**, now that sign is withdrawn.
7. **Explain the 6× scaling discrepancy** — what sets the crossing, if not margin.

---

## 9. QUESTIONS THE ROUND COULD NOT ANSWER

- **What limits softmax at large `t*`?** Not margin-resolution — that theory
  over-predicts the data requirement by ~6×.
- **Does the crossing scale predictably with `t*`?** Two usable points and a bound.
  Not enough for a law.
- **Is the corpus's top-fifth band coverage enough?** Widening is priced at **64%
  more second-oracle work per rung**, and has not been taken.
- **How do you mechanize a reader-side check?** Instances 19, 23 and 24 have **no
  plant** — nothing is wrong with the instrument to plant against. They were caught
  only by asking: *is this claim verified, or only the report it arrived in?*,
  *what did this guard actually scan?*, *what did we correctly file as harmless,
  back when it was?*
- **Can the corpus provenance be reconstructed?** `data/README.md` describes a
  20,000-line head; the file has **211,765 lines** and no countable unit equals
  20,000. All 11 corpora verify byte-identical, so what exists is intact and
  cannot be recreated from its written instructions.

---

## 10. HOW TO READ THIS REPO

- **Numbers:** `results/*.jsonl` — every published figure has a row.
- **Ledger:** `scale/ledger.py`, never grep.
- **Defect taxonomy:** `R10_MECHANISM.md` diagnoses; `MISTAKES.md` prescribes.
- **Round records:** `R10_ITERATION_*.md`, one per iteration, 14 of them.
- **The rule that governs everything:** a finding is not real until a test was RED
  before the repair, and an instrument that cannot fail proves nothing.
