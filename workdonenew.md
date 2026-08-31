# workdonenew — where CEQ actually stands

The architecture-and-state document. `ARCH.md` does not exist and must not be
created; this file is it. Every number below carries its file, its cell, its seed
count and its thread count, or it is not written.

Rewritten at the close of the V13 round, 2026-08-31. The prior version is kept at
`attic/workdonenew.pre-v13.md`.

---

## 0. THE ONE-LINE ANSWER

**A calibrated instrument with a proved mathematical spine, a complete negative
result on its own capability claim, and no reading yet taken against its
registered protocol.**

The north star is one sentence:

> attention that is TOST-equal to self-attention on its own ground, built FROM
> softmax and AdamW, and capable on ground they cannot occupy — predicting the
> next STATE toward equilibrium (which basin, whether at a decision point, which
> transition), not the next token.

Distance to it, in three facts. The parity half is **unreachable at the
registered seed count** and needs N=70 rather than N=8. The capability half has a
**deciding measurement that came back negative**, 4/4 against a prediction filed
before the data existed. And the arm that was to carry the capability shares its
operator with softmax **bit-identically**, so the whole architectural difference
is one hop-2 term that five independent measurements agree carries nothing.

---

## 1. THE CLAIM LADDER — the twenty-second read

The shipped sentence requires `C-PAR ∧ (C-CAP ∨ C-TS)`.

| bar | what it demands | status | the measurement that decides it |
|---|---|---|---|
| **C-PAR** | parity with softmax at matched params | **NO INSTRUMENT — both routes closed** | *TOST route:* with `Δ_eq = 0.5σ` the 90% CI half-width is `t(.95, 2N−2)·√(2/N)` = `0.8807σ` at N=8 against a `0.5σ` margin. Two *bit-identical* arms return NO VERDICT. The CI first fits at N=23; power ≥ 0.80 first at **N=70**, and at N=23 achieved power is **0.0669** (`V15_CONTRACT_ARITHMETIC_AUDIT.md` A-1). *Identity-bind route:* **REFUTED** — see the note below |
| **C-CAP** | a seed CI below the 1-hop floor `√((t*−1)/t*)` | **NEVER ACHIEVED — 0 of 9 cells** | `cap_verdict` over every N=8 cell at fixed threads: zero crossings, `ĥ` peaks at `0.389` |
| **C-TS** | transition-state / exit accuracy | **NOT BUILT** | BED-1 is blocked on Round 11, which has not run |

**Scoreboard: 0 of 39.** Itemised in §8. `41` is the sum of the contract's line
items; X₂₆ was struck, leaving `39` live.

### The C-PAR identity bind is refuted, and both routes to parity are now closed

Recorded 2026-08-31, R11 it.5. The v15 contract retired TOST for the parity half
and substituted an identity bind: *"PARITY WITH SELF-ATTENTION is by IDENTITY
BIND, not TOST. `g == 0` gives bitwise standard attention."* That substitution
does not hold for the operator the same contract specifies.

§S-M specifies "ONE **unnormalized** causal hop `W_ij = exp(C_i − C_j)`". At
`g ≡ 0` every `C_i = 0`, so every causally-masked entry is `1` and row `i` sums
to `i + 1`. Every row of every softmax attention matrix sums to `1`, for every
query, key and weight matrix. Machine-checked in `lean/CEQ/V15.lean`:

```
gate_zero_row_sum        : ∑ j in range (i+1), Wc g i j = ↑i + 1
gate_zero_not_stochastic : 1 ≤ i → ∑ j in range (i+1), Wc g i j ≠ 1
```

Smallest witness is `i = 1`: the row is `(1, 1)`, sum `2`. Normalizing does not
recover the claim — the normalized row is uniform `1/(i+1)`, which coincides
with attention only for a head with constant QK logits, and §S-M's hop carries
no QK term. The failure is mathematical, not proof engineering: the refuting
statement is three lines and green, and the two objects differ by a per-row
factor of `i + 1` because the hop **replaces** the attention logits instead of
**adding to** them.

What is true is the mask half, and it is proved: `gate_zero_is_attention` gives
`Wc g i j = if j ≤ i then 1 else 0`, which is standard attention's *mask*. The
mask matching is what let the identity claim stand unexamined.

**Consequence for the ladder.** C-PAR now has no working instrument. TOST is
unreachable at the registered N=8 and carries 0.0669 power at the N=23 the v15
contract licenses it from; the identity bind that was to replace TOST is false.
A route exists — `gate_zero_logit_identity : q i j + (scan g i − scan g j) =
q i j`, the ADDITIVE-logit reading, where the bind is genuine — but label
reproduction (`prefix_logit_computes_chain`) is proved for the MULTIPLICATIVE
hop, and no single operator is yet known to carry both. Whether one can is open
and is the round's load-bearing question.

**This is the strongest fact the round has produced and it is a negative one.**
It was reached by proof before any gradient step, which is what putting the Lean
train-gate in front of training was for.

---

## 2. WHAT THE REPO HOLDS

| directory | contents |
|---|---|
| `ceq/` | 26 modules — the architecture: `attention.py`, `arms.py`, `hankel.py`, `multizoom.py`, `nash.py`, `rips.py`, `eviction.py`, `hopcache.py`, `lm.py`, `sizing.py` |
| `scale/` | 104 modules — harnesses, gates, oracles, pricing, verdict machinery |
| `scripts/` | 9 `v13_*` instruments built this round |
| `tests/` | 200 files; `tests/loop/` holds 29 guards written to catch instrument defects |
| `lean/` | 6 CEQ proof files, machine-checked |
| `results/` | journals — every published number's provenance |
| `attic/` | retired code, kept resolvable |

**Ledger:** 10,809 events, 425 findings, 141 adjudications. Parse with
`scale/ledger.py`; **never grep it** — `json.dumps` writes a space after each
colon and a literal-string grep silently returns zero.

**Failure record:** `MISTAKES.md` carries **54 entries**, each a failure mechanism
with a check. Ten were added this round.

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

**The load-bearing one is the resolvent.** `occupancy_eq_inverse_of_nilpotent`
and `occupancy_is_exact_inverse` establish that `M = (I − γP)⁻¹ = Σ γᵗPᵗ` **is**
the nilpotent resolvent, exact in `n` terms. The architecture's claim to compute
equilibrium rests on an identity that is proved, not fitted.

This is the strongest asset in the repository and nothing below touches it. A
negative capability result does not weaken a theorem.

---

## 4. EVERY N=8 CELL IN THE REPOSITORY

Nine cells carry N=8 distinct seeds at a fixed thread count. This is the complete
census; there are no others.

| arm | `t*` | `n` | thr | mean | sd | floor_1 | `ĥ` | verdict |
|---|---|---|---|---|---|---|---|---|
| softmax | 2 | 2048 | 6 | 0.952349 | 0.010101 | 0.707107 | 0.186 | consistent with 1 hop |
| pivot_unsigned | 2 | 2048 | 6 | 0.956525 | 0.013133 | 0.707107 | 0.170 | consistent with 1 hop |
| softmax | 8 | 2048 | 6 | 1.124057 | 0.012204 | 0.935414 | n/a | NO READING |
| softmax | 8 | 16384 | 12 | 0.983154 | 0.004732 | 0.935414 | 0.267 | consistent with 1 hop |
| softmax | 8 | 32768 | 12 | 0.975371 | 0.002360 | 0.935414 | 0.389 | consistent with 1 hop |
| **pivot_unsigned** | **8** | **32768** | **12** | **0.976488** | **0.004039** | 0.935414 | **0.372** | **consistent with 1 hop** |
| softmax | 32 | 2048 | 6 | 1.153720 | 0.041984 | 0.984251 | n/a | NO READING |
| softmax | 32 | 32768 | 12 | 1.003371 | 0.003759 | 0.984251 | n/a | NO READING |
| softmax | 32 | 49152 | 12 | 0.998841 | 0.001873 | 0.984251 | 0.074 | consistent with 1 hop |

**Zero crossings of the 1-hop floor.** `ĥ = t*(1 − NRMSE²)` peaks at `0.389`, so
no arm at any cell reaches even one hop. The theorem-grade verdict "a cell below
`floor_1` with CI is PROVEN multi-hop" has never had a candidate to adjudicate.

---

## 5. THE DECIDING MEASUREMENT — filed blind, adjudicated by script

`V13_PREDICTION_HOP2.md` was written while the target journal held **zero** cell
rows, hashed at filing, and never edited — including when an intermediate finding
made its failure look likely and this repository recorded that it did.
`scripts/v13_adjudicate_hop2.py` was written at 4 of 8 seeds and refuses below
N=8. It refused at 6/8, and refused again when the completing seeds arrived in
the wrong thread lane.

`pivot_unsigned`, `t*=8, n=32768, steps=150, threads=12`, N=8 in one lane:

```
0.973990  0.977050  0.974743  0.978135  0.982110  0.981606  0.970375  0.973892
mean 0.976488   sd 0.004039   CI95 [0.973932, 0.979036]
```

| clause | verdict | value |
|---|---|---|
| P1 mean in `[0.960, 0.990]` | **HOLDS** | 0.976488 |
| P2 CI does not clear `floor_1` | **HOLDS** | `proven_hops = 0` |
| P3 at least `0.09` above `floor_2` | **HOLDS** | 0.110462 |
| P4 `ĥ` below 1.0 | **HOLDS** | 0.372 |

**4/4 held.** Against softmax at the identical cell (`0.975371`) the contrast is
**`+0.001117`** — indistinguishable.

---

## 6. THE FIVE MEASUREMENTS THAT AGREE

`pivot_unsigned` shares softmax's operator **bit-identically** —
`Arm._operator` returns `bench._softmax_operator` for both, `max|difference| =
0.000e+00`. The entire architectural difference is one hop-2 term. Five
independent routes measured it:

| route | result |
|---|---|
| **K sweep**, `K ∈ {8,16,32,64}` at `t*=2, n=2048`, 3 seeds | `0.960945 / 0.968518 / 0.965750 / 1.000329` against softmax `0.950252`. At `K=64`, which is the full `a@a`, it stops reading |
| **Gain sweep**, `γ ∈ {0,.05,.10,.25,.50,1.0}`, branches pre-registered | `+0.000000 / +0.008261 / +0.006960 / +0.002384 / +0.031754 / +0.048981`. No `γ` beats zero, so the pre-registered **CONTENT** branch holds. `γ=1.0` reproduces the K sweep's `1.000329` to `0.000254`, so the two harnesses agree |
| **Deflation, X₂₉a** | renorm `K=64` = `0.951400`, 3/3 seeds below the pre-registered `≤ 0.960945`, `p = 0.0399`. Recovers `0.048929` of the `0.050077` destroyed and lands `+0.001148` from the 1-hop control at Welch `p = 0.9286`. **Stops the damage; does not make the hop work** |
| **Wiener/MMSE, X₂₉b** | **all 64 modes inside the zero-correlation null band**, max `g = 0.004041` against the 99.9% quantile `0.005287`, at `K=64` and `K=8`, against the exact oracle. `‖P_wiener‖_F = 0.008173` against `‖I‖_F = 8` — the optimal treatment attenuates the hop about **979×**. Still zero modes above the band on the **trained** operator |
| **Gated multiplicative hop** | `0.966692` against softmax `0.952203` and the additive hop `0.957855` — worse than both |

**Why, structurally.** `scale/negation_scope.py:307-331` — the label is
`z_i = a_i·z_{i-1} + b_i`, so its `t`-hop term is a **path product**
`a_{s-1}···a_{s-t}·b_{s-1-t}`. Attention supplies weighted **sums**, and composing
hops supplies sums of sums. The term is the wrong *shape*, which is why no `K`,
no gain and no deflation moved it.

**And the bed's own limitation.** A recurrence `z ← a·z + b` with input-dependent
`a` is the defining form of a selective state-space model. This corpus asks for
what a gated linear scan computes natively. The softmax control is therefore not
a strong baseline here — it is the wrong primitive too, which is why it sits at
`0.950252` rather than near the floor. A C-CAP sentence from this bed would
measure which arm approximates a scan less badly, not whether either occupies
ground softmax cannot.

---

## 7. INSTRUMENTS BUILT THIS ROUND, AND WHAT EACH REFUSES

| instrument | what it does | what it refuses |
|---|---|---|
| `it11_verdict.tost` | two one-sided tests, `Δ_eq = 0.5σ`, achieved power | a margin below twice the `2.345e-3` reduction-order floor; a difference test read as parity; either sample below N=8 |
| `it11_verdict.cap_verdict` | one-sample CI against `√((t*−h)/t*)`, `proven_hops` | below N=8; returns **no** hop count above the bar rather than a negative one |
| `scripts/v13_adjudicate_hop2.py` | reads the filed prediction's four clauses off the journal | below N=8, and across thread lanes |
| `r10_capacity_sweep --arm` | four arms on the chain corpus | default `softmax` reproduces published cells at `delta = 0.000e+00` |
| `r10_it8_table.py` | per-arm table with floors and `ĥ` | `ĥ` undefined above the bar |
| 0-step gate, `GATE_TOL = 1e-3` | derived from a measured 16-seed null | a genuine `0.99997` reading still aborts the run |
| `scripts/v13_derivation_check.py` | discrete Kramers, fan-out null | passes; tolerances were not loosened |
| `scripts/v13_b9_4060_probe.py` | card memory and throughput | fp32 predictions matched to an fp32 loop |

All pass a regression run taken after every subsequent edit.

---

## 8. SCOREBOARD, ITEMISED

| item | pts | state |
|---|---|---|
| R11 verdict as priced (V1 +12 / V2 +6 / V3 +2) | 12 | not achieved — no reading has run |
| corpus registration | 3 | not achieved — the λ₂ band covers 12.9% of its declared width |
| α-instrument past both must-fires | 2 | not built; the instrument is published (Ly & Gong, arXiv:2510.05606) |
| S5′ beats argmax and approaches ceiling | 8 | not built; X₂₇b's ceiling STRUCK |
| CK-test admissibility GREEN | 2 | not built |
| X₂₅ separation | 3 | not run; its deciding test needs a third arm |
| X₂₆ alarm-precedes-transition | 2 | **STRUCK** — lead-time CI `[−13.88, +1.50]` includes zero |
| TOST parity with power ≥ 0.8 | 5 | unreachable at N=8; needs N=70 |
| local-trained HF package | 4 | not done |

`41 − 2 = 39` live, **0 earned**.

---

## 9. WHAT WOULD HAVE TO BE TRUE TO SHIP THE SENTENCE

Measurable conditions, not tasks.

1. **A cell exists with N ≥ 70 seeds per arm at one thread count.** Below that
   `C-PAR` has no passing branch — two bit-identical arms return NO VERDICT.
   Priced against the measured cost curve, the reading's three points cost about
   `78 h` per arm at N=70 against `8.9 h` at N=8.
2. **Some arm's N=8 seed CI lies entirely below `√((t*−1)/t*)`.** Nine cells,
   zero crossings, `ĥ ≤ 0.389`. Nothing in the repository is near this.
3. **A hop construction exists whose term is the right shape.** Four have been
   measured and refuted. The label needs a path product; every candidate supplied
   sums.
4. **Round 11 runs.** It gates everything under D-4 and has not started.
5. **The bed can distinguish the claim.** If the corpus is a selective SSM in
   disguise, a win on it is a statement about scans rather than about this
   operator.

---

## 10. HOW TO READ THIS REPO

- **Numbers:** `results/*.jsonl` — every published figure has a row. Read the
  journal, never the task stdout: stdout buffers, and a cell can exist for an
  hour before it appears there.
- **Ledger:** `scale/ledger.py`, never grep.
- **Defect taxonomy:** `R10_MECHANISM.md` diagnoses; `MISTAKES.md` prescribes.
- **This round's working record:** `V13_DAG_TASKLIST.md`, carrying every
  derivation and every correction.
- **Independent audit:** `V13_CLAIM_AUDIT.md` — 44 CONFIRMED, 9 DISCREPANT, 1
  UNVERIFIABLE, all nine corrected.
- **Thread count is part of a cell's identity.** The harness is deterministic
  given a thread count and returns a different number across counts, drift
  `2.345e-3`. `it11_verdict.by_seed` refuses to pool, and so does the
  adjudicator.
- **The rule that governs everything:** a finding is not real until a test was
  RED before the repair, and an instrument that cannot fail proves nothing.

---

## 11. CLAIMS DISPROVED, WITHDRAWN, OR STRUCK

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
| **X₂₆ "a CUSUM alarm on the loss trajectory precedes the transition"** | **STRUCK, kill clause fired** | lead-time bootstrap CI `[−13.88, +1.50]` includes zero, mean `−6.00`, 3/8 seeds positive, so the N=8 sign floor `0.0078125` is unreachable. At matched ARL₀ the integrating detector does not beat a memoryless comparator. Dependence inflates the CUSUM interval `4.49×` (8.833 → 39.683) against Shewhart's `1.08×` (4.078 → 4.420), consuming the early-warning margin the item existed to sell |
| **"ARL₀ from the closed form is the false-alarm rate"** | **BOUNDED to i.i.d. nulls** | Siegmund matches simulation to 0.71% on the independent null and fails by `24.1×` under the measured autocorrelation `φ̂ = 0.709` — realized ARL₀ `41.5` against nominal `1000`. End-to-end on the shipped pipeline, `1459` against nominal `22000`, a 53.0% chance of at least one false alarm per run |
| **"`a*` is the true lowest-barrier exit"** (v-main.7 §2, v-main.8 Part II) | **FALSE ON LEGAL INSTANCES** | with one saddle at `E=2.00` to B and four parallel saddles at `E=2.20` to C, series–parallel reduction gives `q_C/q_B = 4e^{−0.2/T}` exact to `2.351e-15`. At `T=0.50` the higher-barrier channel is `2.68×` faster (`q_B=0.271644632`, `q_C=0.728355368`). Crossover `T*=0.144269504088897` vs predicted `0.144269504088896`. Label by splitting probability; rankings agree only where `ΔΔE‡ > T·ln(prefactor ratio)`. **Independently reproduced** this session: `ΔΔE‡/ln m` gives `0.144269504088896` against the reported `…897`, agreement `6.384e-16`; `q_B` and `q_C` at `T=0.50` match to all printed digits; and the ranking flips back at `T=0.10` (`q_B = 0.648786 > q_C = 0.351214`), confirming a genuine crossover rather than a monotone offset |
| **"the seed spread is the noise model"** | **ALREADY GUARDED — one new consequence** | the thread floor `2.345e-3` was found and fixed by this round: `it11_verdict.by_seed` builds intervals at fixed thread count and `r10_inspector_phase1a.md:25` verified the three cross-thread pairs. New only: `Δ_eq = 0.5 σ_seed` appears nowhere in the tree, and at `t*=2, n=2048` (`sd 0.010101`) it is `0.005051`, so the floor is `0.464` of it — any equivalence margin below ~`4.7e-3` is indefensible |
| **X₂₇b "no predictor can exceed exit accuracy `1 − c·f(ε)`"** | **STRUCK — unproved and false as written** | no such theorem exists in dynamics or learning theory and no source fixes `c`. Under Grebogi–Ott–Yorke's definition of `f(ε)` as the ε-uncertain phase-space fraction the inequality runs the other way, `err ≤ f(ε)(1 − 1/N_A)`, because an ε-ball counted uncertain may be arbitrarily imbalanced. Redefining `f` as pairwise disagreement inside the ball, with binary exits, makes it `c = 1/2` by three lines of Bayes-error algebra — i.e. the already-published `Π = 1 − R_B` (Xu et al., *EPL* **141**, 61003) |
| **X₂₇a "the uncertainty exponent is unused in machine learning"** | **FALSE** | Ly & Gong, arXiv:2510.05606, run the identical protocol — same `f(ε) ∝ ε^φ` law, same perturbation scheme, same log-log fit, same `√(f(1−f)/n)` error bars — on network training, reporting `φ = 0.0126 ± 0.0002` on a minimal network and `φ = 0.000 ± 0.002` on VGG-12. The instrument is not novel. The expectation holds only for language modelling specifically |
| **X₂₇c "`H` and `α̂` are two coordinates"** | **COLLAPSED to one** | basin entropy `S_b = Σ (n_k/ñ)·ε^{α_k}·log m_k` (Daza et al., *Sci. Rep.* **6**, 31416, Eq. 6) makes the entropy and the exponent one object and its scaling exponent, not two independent axes |
| **X₂₇d "`κ̂` is a second, independent rate theory"** | **NOT INDEPENDENT** | `α = κ/λ` follows from Tél's `D₁ = 1 − κ/λ` with `α = D − d`, so the escape rate is a function of the exponent X₂₇a already measures. Citation owed (`[U]`) |
| **"the corpus covers the declared λ₂ band `[0.90, 0.95]`"** | **BOUNDED to the top eighth** | the 12 accepted rungs read `0.9435 … 0.9499` (`results/r10_it15_dual_oracle.jsonl`, `per_rung`), a span of `0.0064` against a declared width of `0.050` — **12.9%** of the band, all of it at the top. 29 rungs were attempted for 12 accepted. `workdonenew.md` previously said "top fifth"; the measured figure is narrower |
| **"attention activation is `2·n·s²·heads` bytes bf16"** (v-main.6 §3, v-main.8 Part IV) | **6.63× LOW** | the leading `2` is bytes-per-element, so the line assumes ONE retained `[n,heads,s,s]` tensor per layer. `ceq/sizing.py`, calibrated against the CUDA allocator on this same RTX 4060 Laptop, records `C_OPERATOR = 3.9` retained tensors at an effective `3.4` bytes/element under bf16 autocast. `3.9 × (3.4/2.0) = 6.63`. At `n=32, s=512, h=4` the line says `0.0671 GB/layer`; the calibrated model says `0.4449 GB/layer` |
| **"the constraint is THROUGHPUT, not memory"** | **FALSE for the signed arm** | at the contract's own shape and `L=32`, softmax totals `5.806 GB` and fits the 8 GB card while signed totals `20.044 GB`, 2.5× over. Gradient checkpointing rescues the activation term to `0.645 GB`, a comparable total of `1.102 GB`. Measured on the card at `L=16, bs=8, seq=512, fp32`: signed peaks at `2875.8 MiB` against softmax's `1153.5 MiB`, **2.49×** |
| **"matched params is a matched comparison"** | **THREE DIFFERENT EXPERIMENTS** | measured on the 4060 at `L=16, bs=8, seq=512, fp32`: softmax `54,837 tok/s`, signed `20,194 tok/s` (**2.72× slower**), signed+checkpoint `15,090 tok/s` (**3.63× slower**). Matched params, matched memory and matched wall-clock select different budgets; Part IV requires matched params for C-PAR and matched wall-clock for the optimizer ladder, and at matched wall-clock softmax takes 2.72× the steps |
| **C-PAR "TOST-parity with softmax at N=8"** (v-main.6 §1, v-main.8 Part IV) | **UNREACHABLE AS REGISTERED** | with `Δ_eq = 0.5σ`, the 90% CI half-width of a two-sample contrast is `t₍.₉₅,2N−2₎·√(2/N)` in units of `σ` — `0.8807` at N=8 against a margin of `0.5`. Two **bit-identical** arms return NO VERDICT at N=8: the interval cannot fit inside the window. The CI first fits at **N=23**; TOST power at a true difference of zero first clears 0.80 at **N=70**, which is **8.8×** the registered seed count. Asserted in `scale/it11_verdict.py::demo_tost` |
| **first CEQ arm on the chain corpus** | **RUN, UNDECIDABLE** | `pivot_unsigned`, `t*=2, n=2048, steps=150, threads=6`, N=8, all eight seeds LEARNS. mean `0.956525` sd `0.013133` against softmax's mean `0.952349` sd `0.010101`. Contrast `+0.004176`, 90% CI `[−0.006189, +0.014542]`, `Δ_eq = 0.005051` → **NO VERDICT**, power `0.000`. The arms are not shown equal and not shown different; at N=8 no design could have shown either |
| **the hop-2 prediction, `V13_PREDICTION_HOP2.md`** | **4/4 HELD, adjudicated by script** | `pivot_unsigned` at `t*=8, n=32768, steps=150, threads=12`, N=8 distinct seeds in one lane: mean **`0.976488`**, sd `0.004039`, CI95 `[0.973932, 0.979036]`. Against softmax's `0.975371` at the same cell the contrast is **`+0.001117`** — indistinguishable. `floor_1 = 0.935414` is not cleared (`proven_hops = 0`), the reading sits `0.110462` above its own `floor_2 = 0.866025`, and `ĥ = 0.372`. The prediction was filed when the journal held **zero** cells and adjudicated by `scripts/v13_adjudicate_hop2.py`, written at 4/8 seeds. The second hop does not lift the arm off softmax, and no increase in `n` from this arm produces a floor crossing |
| **C-CAP "a cell below floor_1 with CI is PROVEN multi-hop"** | **NEVER ACHIEVED — complete census** | `cap_verdict` run over all 8 cells in the tree carrying N=8 distinct seeds returns zero crossings. Five read "consistent with 1 hop" (`ĥ` = 0.186, 0.170, 0.267, 0.389, 0.074) and three are NO READING above the bar. `ĥ` never exceeds `0.389`, so no arm at any cell has reached even one hop, and the theorem-grade verdict has still never had a candidate to adjudicate |
| **THE READING's "four arms"** | **THREE OPERATORS, NONE SIGNED** | measured at the harness geometry (`S=64, D=24, d_model=16`, untrained): `max\|softmax − pivot_unsigned\| = 0.000e+00` — bit-identical, since `Arm._operator` returns `bench._softmax_operator` for both (`m3_capability.py:131`). `pivot_signed` and `windowed_signed` are genuinely distinct — both `2.2727e-01` from `pivot_unsigned`, while `1.8323e-01` is the `pivot_signed`↔`windowed_signed` distance, mislabelled in an earlier draft, so the "wearing a name" finding is about signedness, not numeric identity. Every one of the four has `min entry = +0.000000` and a negative-entry fraction of `0.0000`: the round runs four arms over three operators and **zero signed operators**. What separates `softmax` from `pivot_unsigned` is not the operator at all but the added hop-2 term |
| **"pivot routing supplies the second hop"** (the M2/M3 mechanism) | **FALSIFIED AT THIS GEOMETRY** | measured on `e3_t8`, `s=64`, `K_PIVOTS=8`: the full second hop `a@a` has median rank **62**; the routed `pivot_hop2 = a[:,P] @ a[P,:]` has median rank **8**, exactly `\|P\|` as `scale/pivot_probe.py:98` states by construction. It retains `cos = 0.7398` of the full hop's direction but only `0.1932` of its magnitude, discarding 54 of 62 dimensions. `pivot_unsigned` shares softmax's operator bit-identically, so this term is the *entire* architectural difference between the arms — and at `t*=2`, where a working second hop reaches `floor_2 = 0.000000`, the arm reads `0.956525`, `0.004176` worse than softmax alone |
| **"the rank-`\|P\|` bottleneck is why hop 2 shows nothing"** | **INVERTED BY MEASUREMENT** | K sweep at `t*=2, n=2048`, 3 seeds, `threads=8`: softmax `0.950252`; `K=8` `0.960945`; `K=16` `0.968518`; `K=32` `0.965750`; **`K=64` `1.000329`, NO READING**. At `K=64` the pivot set is every position so `a[:,P] @ a[P,:]` *is* `a@a`, the full second hop. Widening the routing does not recover a starved hop — it removes a restriction that was keeping the arm learnable. Rank `≤ \|P\|`, described at `pivot_probe.py:98` as "the mechanism", measures as damage control |
| **why the full second hop destroys the arm** | **~~COMMON-MODE SWAMPING~~ — WITHDRAWN, the figure was the null** | `a` is strictly lower triangular and sub-stochastic (row sums `0.000000`–`1.000000`, row 0 zero), so `a@a` is Perron-dominated by a rank-1 carrier. Decomposed at `t*=2`, untrained: the common mode holds **0.7550** of `a@a`'s energy and **0.4899** of `pivot_hop2`'s at `K=8`. Measured against a sharpness-matched null of `0.7550 ± 0.0004` at 20 seeds, the `0.7549` figure has `z = −0.30`: it is the null, not a finding. The pivot routing looked like an accidental common-mode filter — but that is a full-matrix statement, and `Arm.forward` returns only row `s-1`. On that row `K=8` carries `0.8646` against `K=64`'s `0.8343`, so it is not a filter there and the explanation is **withdrawn**. What scales with `K` at the readout row is magnitude: hop 2 is `0.0215` of `‖z‖` at `K=8` and `0.1701` at `K=64` |
| **X₂₇a's box-counting `d` feeding `κ = λ(1−d)`** | **WRONG DIMENSION** | every fetched statement of the Kantz–Grassberger relation uses the **information** dimension `D₁` of the natural measure on the saddle, not box-counting `D₀`. Since `D₁ ≤ D₀` always, `λ(1−D₀) ≤ λ(1−D₁) = κ`: the substitution under-predicts the escape rate with a known sign. On an exactly solvable two-branch linear repeller the shortfall is `0.49%` at slope ratio 1.5, `4.83%` at 3, `9.56%` at 5 and `24.5%` at 500, with equality only where `\|f′\|` is constant on the saddle. X₂₇a measures box counting, so it cannot supply `d` for this relation |
| **the second hop carries usable signal at any mode** | **REFUTED, trained and untrained** | MMSE per-mode gains `g_i = S_i/(S_i+N_i)` over an SVD basis (`a@a` is nilpotent — `max\|eig\| = 0.000e+00`, all 64 eigenvalues zero, eigenvectors do not span, so no eigendecomposition exists to use). **All 64 modes fall inside the zero-correlation null band**: max `g = 0.004041` against the 99.9% quantile `0.005287`, at `K=64` and `K=8`, against the exact oracle and the raw label. `‖P_wiener‖_F = 0.008173` against `‖I‖_F = 8` — the optimal filter attenuates the hop `~979×`. Re-measured on the **trained** operator after 150 steps: still zero modes above the band |
| **"the second hop is too large, not wrong"** (MAGNITUDE) | **REFUTED by a pre-registered sweep** | scalar gain `γ` on the hop-2 term at `K=64`, `t*=2, n=2048`, 3 seeds: `0.951602 / 0.959863 / 0.958562 / 0.953986 / 0.983356 / 1.000583` at `γ = 0 / 0.05 / 0.10 / 0.25 / 0.50 / 1.00`. No `γ` beats `γ=0`; the best non-zero point is inside one control sd of zero. `γ=1.00` reproduces the K sweep's independent `1.000329` to `0.000254`, so the harnesses agree. The pre-registered CONTENT branch holds: no constant repairs the term, its construction has to change |
| **"a multiplicative hop recovers what the additive one cannot"** | **REFUTED** | the label's `t`-hop term is a path product `a_{s-1}···a_{s-t}·b`, which additive attention hops cannot form, so a gated hop `A(g ⊙ Ax)` with `g = σ(Wx)` was built and measured at `t*=2, n=2048`, 3 seeds: **`0.966692`** (sd `0.005926`) against softmax's `0.952203` and the additive hop's `0.957855`. Worse than both. Four hop constructions have now been measured — routed, scaled, deflated, gated — and none reaches the 1-hop baseline |
| **X₃₂ "`ρ_P = √2` for a planted antisymmetric A"** (§3 must-fire) | **INVERTED** | `J` antisymmetric gives `Jᵀ = −J`, so `ρ_P = 2` **exactly** — measured `2.000000` at n = 4, 16, 64, 256, and again on the contract's own RPS matrix. `√2` is the **random-matrix** value (`E‖J−Jᵀ‖²/E‖J‖² = 2n(n−1)/n² → 2`), measured `1.411720` at n=256. A gate set at `√2` passes a random fitness Jacobian and fails a genuine RPS one |
| **X₃₂ "Poincaré–Hopf gives Σ index = 1 for the replicator"** (§4) | **NEEDS μ > 0, AND §7 CAN BREAK IT** | at μ=0 the field is tangent to every face — measured `dz₀ = −0.000000e+00` — so the transversality hypothesis fails and the index sum is not pinned to χ. Mutation restores it (`+3.333e-03` at μ=0.01). But §7's injection `μ(W_b x_t)` is a learned linear map free to go negative, which points the flow outward and voids the gate. Fix: `μ·softplus(W_b x_t)` |
| **X₃₂ "Fisher gives `d f̄/dt = Var_z(f)`"** (§3) | **CORRECT THEOREM, WRONG QUANTITY** | measured ratio `1.0000` for constant fitness but **`2.0000`** for symmetric `A`. Resolved: `P = ½zᵀAz` and `f̄ = zᵀAz`, so `f̄ = 2P` and `dP/dt = Var(f)` exactly (`1.0000`). The `= Var(f)` form belongs to the potential; mean fitness carries the factor 2. Conclusion unaffected — both are ≥ 0 |
| **X₃₂ "the scan deletes normalization"** (§0) | **NO-OP WITHOUT INJECTION** | per-step vs post-hoc normalisation differ by `1.665e-16` at μ=0 — a multiplicative recurrence commutes with rescaling. With μ=0.15 they differ by `2.862e-01` and select a **different argmax**. The mechanism is normalisation × injection, not normalisation; an S3-R arm with weak injection is softmax with extra steps |
| **X₃₂ §2's three-word verdict** | **WORKS, and answers "stable" not "good"** | over 4,003 deviations: coordination vertices **ESS**, interior mix **UNSTABLE** (gain `+0.333333`), RPS interior **NEUTRALLY STABLE** — Nash but failing the second-order condition, which is why it cycles. Caveat: coordination's *worst* vertex (payoff 1) is also an ESS, so the verdict certifies local stability, never that the basin reached is the right one |
| **X₃₁'s horizon arithmetic** (§X₃₁a/b) | **CONFIRMED** | `T_guard = (1/λ)ln(margin/δ)` gives `20.022` and `10.011` against the contract's 20 and 10; steering `1e-6·e^{0.69·20} = 0.9846` → O(1). Prediction and steering horizons are the same number. Bound inherited: `T_guard` is logarithmic in δ, so 1000× better resolution buys **10 steps**, not 1000× horizon |
| **Wiener-equalized hop vs softmax** | **NOT RESOLVED at N=3** | softmax `0.950252`, plain `K=64` `1.000329`, Wiener `K=64` **`0.958816`** (sd `0.005158`). Against plain: `−0.041513`, 3.5 SE, resolved. Against softmax: `+0.008563`, 0.74 SE, **not** resolved. The equalizer repairs the damage the raw hop does and does not produce a gain over one hop |
| **the `0.4899` K=8 common-mode share** | **DOES NOT REPRODUCE** | three definitions return `0.7535 / 0.6517 / 0.6532`; `K=64`'s `0.7550` reproduces to `1e-4`. Two agents reached the non-reproduction independently. The figure was load-bearing for the withdrawn "K=8 filters the common mode" claim and is void on its own terms |
| **X₂₈c "three instruments certifying each other"** | **NOT CIRCULAR, BUT NOT PHYSICS** | independent estimators for `κ`, `λ` and `D₁` exist and have been run together on a 1-D map (Drótos et al.: `λ = 0.54`, `κ = 0.075`, `D₁ = 0.86` against `1 − κ/λ = 0.8611`), so the triangle is not an algebraic identity checked against itself. But `κ = λ(1 − D₁)` is a **theorem** for hyperbolic systems, so agreement certifies three estimators against each other and nothing about the bed. The same source records choosing estimator definitions to minimise deviation from the relation, and substituting `1 − κ/λ` for `D₁` "since it can be computed more precisely" — both of which do collapse the check into an identity |
| **the contract's `λ=0.69, d=0.8 ⇒ κ=0.138`** | **EXACT AND EMPTY** | `0.69 × (1 − 0.8) = 0.138` to the digit, but it follows from the formula alone, and the relation's ordering constraint `λ > κ` holds automatically for any `d ∈ (0,1)`. It tests arithmetic, not the system |
| **the selector under the capability claim** | **DELIBERATELY CRUDE, BY ITS OWN DOCSTRING** | `scale/pivot_probe.py:83-86`: pivots are ranked by key-norm, "*Deliberately crude: a cleverer selector is a confound at this stage, and the point of M2 is whether ROUTING fixes the path count, not whether a selector is good. S2/C4 (nucleolus vs top-k salience) is where selection is compared.*" The capability claim is therefore being adjudicated against a selector the module never offered as one, and a negative C-CAP result licenses "routing at K=8 with a key-norm selector does not deliver hop 2", never "multi-hop routing does not work" |

---

