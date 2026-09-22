# PHASE J — ADDENDUM N (THE DIAGONAL, AND WHAT TIES THE LEAPS) — v2

[Dispatcher note, 2026-09-23 00:01: this is the author's second version, received while round N1 was running on v1 (ADDENDUM_N_CONTRACT.md). It adds §8a (errata after 492ccff), strikes X3 and removes R-KR, re-reads X3a/X3b with atan2, and adds the instances kr_density.py and pf_regime.py. None of diag_leap.py, crossties.py, kr_density.py or pf_regime.py exists on this machine (searched Downloads, Desktop, Documents); every number in §9 is the author's claim until an instance is rebuilt from this text and re-run.]

**In a causal head, equilibrium lives on the diagonal. The eigenvalues are the self-weights W_ii. Phase and rotation gates only change the basis (a similarity transform). Magnitude is the only thing that moves the spectrum, and a magnitude zero creates an absorbing token. The leaps tie together as follows. Magnitude and rotation factor exactly, so Addendum L's SU(2) gate and this addendum's absorbers compose without interfering. Each gated head is then one stage of a reset-then-group cascade. Leray's sheaves say where a rotation can carry content that no change of basis can remove: on the (position × layer) and (position × step) plaquettes of Addendum M's volume. It can never do so along the sequence.**

Author: Seal (Teerth Sharma). Contract date: 2026-09-22. Amends PHASE_J_ADDENDUM_L_LEAP.md (Q2, arms, novelty sentence), PHASE_J_ADDENDUM_M_VOLUME.md (DRIFT rows) and PHASE_J_HYPERSPACE_v2.md §6 (R-PHASE, Mode B cost). Instances: `diag_leap.py` (T1–T10), `crossties.py` (X1–X4).

---

## 0. Standing

- Addendum L stands with three corrections (§4): Q2's generators, a missing scan-only arm, and PaTH as a named owner. Addendum M stands; its DPI caution is unchanged and applies here.
- Unchanged: L-EQ, L-REPRO, L-REFLECTOR, L-WIENER, L-SHA, security lines, and the bed precondition. **No mechanism sentences before the contrasts land:** H-SINK (§5) is a bet with a kill, not an explanation.
- Conventions for every statement below: causal means lower-triangular including the diagonal; β = 1; Z_i sums |G_ij| e^{s_ij}; residue r_γ = (1−γ)(I−γW)⁻¹V; path gate G_ij = g_i g_{i−1} ⋯ g_{j+1} with G_ii = 1.

---

## 1. Pointer resolved: Jean Leray

- **Fetched:** French, 1906–1998. From 1940 to 1945 he was a prisoner of war at Edelbach, Austria. There he concealed his expertise in differential equations, fearing it would be used for the German war effort, and did topology instead: sheaves and spectral sequences. He proved the Leray–Schauder theorem with Juliusz Schauder in 1934, an existence result for solutions without uniqueness. In 1934 he also founded the theory of weak solutions of Navier–Stokes. Wolf Prize 1979.
- **Why he is the cross-tie:** a sheaf is exactly the mathematics of gluing local pieces (segments, layers, steps) along overlaps into one global object. Its first cohomology H⁰ counts the globally consistent sections, and a drop in H⁰ is the obstruction: curvature. §3 X4 turns this into a row.
- **Page lines only, no rows (no instance yet):** Leray–Schauder for the nonlinear equilibrium read, where the linear resolvent is the first-order case; spectral sequences for the depth-wise composition of heads.
- If the pointer meant someone else (Hausdorff for the BCH commutator terms, Cartan for connections), the name on the page changes. The instances and rows below stand either way.

---

## 2. The diagonal (theorems, each with its instance)

- **D1: the spectrum is the diagonal.** det(W − λI) = ∏(W_ii − λ). W_ii = 1 exactly when token i attends only to itself, i.e. it is absorbing. The number of closed classes is the number of unit diagonals: an O(n) read with no eigendecomposition. *T1: max |eig − diag| = 0.0; the only unit diagonal is BOS.*
- **D2: path gates are similarity transforms.** Any gate with G_ij = Π_iΠ_j⁻¹ that leaves Z unchanged gives W_G = D W D⁻¹, for any group and any representation. The spectrum, the resolvent's poles and the closed classes are all unchanged. *T3 (U(1)): 1.6e−16. T4 (SU(2) on 4-blocks): 3.3e−16. Eigenvalues unchanged exactly. T3b (real-part readout): diagonal unchanged; residue stays on the v₀ line (4.3e−4, O(1−γ)).*
- **D3: magnitude is the only spectral actor.** λ_i = 1/(1 + m_i ρ_i), with ρ_i = Σ_{j<i} G_{i−1,j} e^{s_ij − s_ii}. m_i = 0 exactly when λ_i = 1, i.e. token i becomes an absorber. *T5: 4.4e−16; unit diagonals at exactly the planted zeros [0, 16, 40].*
- **D4: softmax's equilibrium is the attention sink.** Only W₀₀ = 1, so r_γ → 1·v₀ᵀ whatever the scores and tokens 1..n−1 are. *T2: 1.2e−1, 1.2e−2, 1.2e−3 at γ = 0.99 / 0.999 / 0.9999. New scores plus new tokens move the residue by 6.6e−4 while the inputs move up to 4.62. σ₂/σ₁ = 4.1e−5.*
- **Corollary: three classes of equilibrium.**
  - Softmax: rank 1, v₀ only, blind to the input.
  - Path-product gates with exact zeros (FoX with zeros, barcode, U(1), SU(2)): rank = number of segments; each row goes to its own segment start, deterministically. *T5: σ = 18.9, 17.1, 10.5, then 0.014.*
  - Non-separable exact zeros (sparsemax/entmax): rank = number of anchors; absorption is mixed, which gives a root-cause map. B_ia = 0 exactly when anchor a is not an ancestor of token i: an exact NEVER. *T6: 44/58 and 855/929 rows mixed. T8: zero pattern vs ancestry, 0/348 and 0/88,255 mismatches.*
- **D5: the cost is the causal depth.** Remove self-loops (jump chain) and the remainder is strictly lower-triangular, so the read is exact in D hops, where D is the causal depth. *T7: after D−1 hops the error is 4.9e−4 (D = 12) and 1.9e−6 (D = 21); after D hops it is 0.0.* The causal resolvent at any γ < 1 is one forward substitution. *T9: agrees with the dense solve to 1.2e−15.*
- **The difference only shows at length** (same logits, Δ = 6). Softmax leaks (i+1−p)/(p·e^Δ + i+1−p) per hop. *T8: 0.071 at n = 64 and 0.559 at n = 1024, equal to the formula.* Even with the roots made absorbing by an oracle and the best of 241 thresholds:
  - n = 64: 0/348 mismatches. **This is a tie, and it stays on the page.**
  - n = 1024: 703/88,255 mismatches, 0.110 of the mass on non-ancestors.
  - Sparsemax: 0 on both, at both lengths.

---

## 3. The cross-ties

### X1 Polar split: the two leaps factor
- The full gate is a nonzero quaternion g_k = m_k u_k ∈ ℍ* = ℝ₊ × SU(2). Quaternion norms multiply, so G_ij = M_ij U_ij and W_G = D (W_m ⊗ I₄) Dᵀ.
- Magnitude owns the spectrum, the absorbers and NEVER. Rotation owns order. Neither leaks into the other's job.
- *X1: 2.0e−15; eigenvalues = diag(W_m) exactly; absorbers at [0, 16, 40].*

### X2 Reset-then-group comes for free
- G_ij only involves tokens j+1..i, so a magnitude zero isolates its segment exactly. The rotation state readable at i is the product since the last absorber: a reset followed by a group, which is one stage of a Krohn–Rhodes cascade. This is a theorem. Its bed was struck in X3, because recency does the reset just as well.
- *X2a: replace every rotation u₀..u₄₀, every value v₀..v₃₉ and every score to keys 0..39; the outputs at i ≥ 40 change by 8.9e−16.*
- **Author prediction struck by the author's own instance:** a segmented scan (reset at absorbers) was predicted to improve fp32 accuracy. *X2b: n = 4096 with absorbers every 64 gives 4.0e−7 for the global prefix vs 1.3e−6 for the segmented one.* The global scan is kept; there is no numerics argument for segmenting.

### X3 A hard reset vs recency: no separation (STRUCK)
- The A₅ word problem with reset tokens R (corrected generators; §4 C1). The (m,u) head with m = 0 at R and v = 1 only at R is exact. *X3a (atan2 readout): n = 4096, 212 resets; max error 2.2e−13°.*
- The opponent is the same SU(2) gate without zeros, attending over the R tokens with a recency slope.
  - *X3b (atan2): 0.4565 / 0.8857 / 1.0000 at slopes 0.05 / 0.2 / 1.0.*
  - *`kr_density.py`, seed 11: at reset densities 0.05 / 0.2 / 0.5, every slope ≥ 1 gives 1.0000. Slope 0.5 gives 1.0000 / 0.9719 / 0.8501.*
- **Struck.** The earlier 0.9692 at slope 1 was the arccos floor, the same bug the rooms found in DRIFT. A rotation-only head with a learnable recency slope already solves the reset bed exactly. The magnitude leg buys nothing on order-with-reset beds.
- **Replacement route:** the magnitude leg's only unique job is exact NEVER at length (R-NEVER-LEN). R-KR is removed, and its compute moves there.

### X4 Leray's tie: where drift can be real
- Put the transports on a cellular sheaf: stalks are the value blocks, restriction maps are the rotations, and H⁰ = ker of the sheaf Laplacian counts the global sections.
- Along the sequence, D2 says the connection is always flat, so drift along the sequence is a change of basis, not content.
- On a (position × layer) or (position × step) plaquette, curvature is possible:
  - *X4, pure gauge: H⁰ = 4 (R⁴ left action) / 3 (R³ rotation), gap 2.000.*
  - *X4, 50° holonomy: H⁰ = 0 / 1 (the axis), gap 0.012 / 0.047.*
- This is the row DRIFT was missing (R-HOLONOMY, §5).

---

## 4. Corrections to Addendum L

- **C1 (Q2 is wrong as written).** 72° about z and 120° about (1,1,1)/√3 are 54.74° apart. That is not one of the icosahedral 5-fold/3-fold axis angles (37.38°, 79.19°, 100.81°, 142.62°), so the pair generates an infinite group. *T10: the closure passed 5,000 elements and never stopped.*
  - Replace with 72° about (0, 1, φ)/‖·‖ and 120° about (1,1,1)/√3. *T10: 120 unit quaternions (the binary icosahedral group 2I, contains −1) and 60 rotations (A₅).*
  - The Q2 kill line becomes: "closure ≠ 120 quaternions / 60 rotations → no leap."
- **C2 (missing arm).** By D2 the SU(2) gate never changes what the head mixes, only the basis. The A₅ certificate belongs to the scan Π_i, read out through BOS. Add arm (s_Q): the same quaternion scan with no attention mixing. If (s_Q) ties (f_Q) on A₅ and on board state, the leap is a quaternion RNN with attention on top, and it is filed that way.
- **C3 (owner, fetched).** PaTH (Yang et al. 2025) already does data-dependent accumulated Householder products on the q·k side. Addendum L's novelty sentence narrows to "value-side," and PaTH becomes arm (a‴) on every order bed. Its state-tracking claims were not in the abstract [U].
- **C4 (cost, conditional).** If Mode B computes the causal resolvent by Neumann/Chebyshev hops (130 at γ = 0.99), it is paying roughly 130× for a triangular solve (D5, T9). R-COST-N decides it.

---

## 5. Rows

### R-DIAG
- Replace every eigenvalue count in R-PHASE with the diagonal read on the causal operator.
- Check: same m, different θ gives bitwise identical counts.
- Counter: if the counts differ, the operator is not causal-triangular, and the row names which operator it actually was.

### R-SINK (H-SINK)
- Hypothesis: "the gate closes" means the optimizer is creating absorbers, the same job attention sinks do in softmax (D4).
- Runs on the existing (a)/(f) checkpoints, with no training:
  - **P1:** closure positions (m < 0.05) are at least 2× enriched at delimiters (newline, full stop, end of sequence). AUC ≥ 0.7 against a within-document position shuffle.
  - **P2:** tokens after a closure put ≥ 0.3 of their attention on the closure token.
  - **P3:** value norm at closure tokens is ≤ 0.5× the median.
  - **P4:** Spearman ≥ 0.3 between (f)'s per-position closure rate and the twin's per-position non-BOS received attention. Shuffle null ≤ 0.05.
  - **P5 (optional, training):** a Qiu et al. output gate on the twin recovers ≥ 0.5·C_win means the win is sink relief; < 0.2 means it is not.
- **Kill:** P1 and P4 at null means H-SINK is struck, and Landau's double-well stays the only account of closure.

### R-NEVER-LEN
- Bed: root-cause attribution on planted causal chains at n ∈ {256, 1024, 4096}, with fixed causal depth.
- Arms: softmax twin (a), SSMax (a_s), FoX (a″), oracle threshold on (a), and a sparsemax/entmax head with the D-hop absorption read (f_N).
- Metrics: exact ancestor-set accuracy, and false-influence mass (mass on non-ancestors).
- Prediction: (f_N) false mass = 0 at every n; (a) grows as the leak law says; (f_N) beats (a_s) on exact-set accuracy at n = 4096 by ≥ 0.10.
- Counter: (a_s) plus a threshold comes within 0.02, in which case exactness is a certificate only. The n = 64 tie from T8 is pinned above the table.

### R-KR (removed before running)
- Its counter already holds at hand-set weights (§3 X3). Its compute moves to R-NEVER-LEN.

### R-HOLONOMY (amends DRIFT in L and M)
- For the fitted drift rotations on V_layer and V_train, compute the holonomy around every (segment, layer) and (segment, step) plaquette, plus H⁰ of the drift sheaf.
- Instrument check: along the sequence the holonomy must be the identity within its CI (D2). If it is not, the instrument is broken, not the model.
- Null: shuffled segment order. Pure-gauge control: a synthetic change of frame.
- Prediction: non-trivial holonomy in at least one layer pair.
- Counter: flat everywhere. Then every DRIFT table reports "gauge," and drift leaves the list of phase evidence.

### R-COST-N
- Measure the Mode B causal read as a blocked forward substitution vs the current hop scheme, wall-clock at S = 4096 in fp32.
- Prediction: ≤ 1.5× one SDPA pass.
- Counter: > 3×, meaning the sequential block steps dominate, and the cost sentence is struck.

---

## 6. Kills

- D1–D5 or X1/X2a fail on the rooms' re-run (any error > 1e−12 in f64) → the addendum is void until the discrepancy is explained.
- R-SINK: P1 and P4 at null → H-SINK is struck.
- R-NEVER-LEN counter holds → NEVER is a certificate only; condition 2 is not met by it.
- R-KR: already killed (X3). X2 stays a theorem without a bed.
- R-HOLONOMY flat everywhere → drift is gauge; DRIFT rows are renamed and leave the phase evidence.
- R-COST-N > 3× → the Mode B cost claim is struck.

---

## 7. Room, fetch orders, iteration

- **Chase:** re-run `diag_leap.py` and `crossties.py` (SHA pinned); R-DIAG. Lean targets `causal_spectrum_eq_diag` (from Mathlib's `Matrix.det_of_lowerTriangular` [U]) and `gauge_similarity_charpoly` (the characteristic polynomial is invariant under conjugation).
- **Cameron:** R-NEVER-LEN, R-HOLONOMY, R-COST-N.
- **Foreman:** R-SINK P1–P4 on the existing checkpoints first (no training); then P5; then the L1-A5′ arms from §8a.
- **Wilson:** arbitration and scoring. The struck X2b prediction and the T8 tie are pinned above their tables.
- **House:** adversarial audit of D2. Find where it breaks: β ≠ 1, a Z that does not sum |G|, gates that are not path products. Report the first counterexample.
- **Fetch before any table (L-EQ):**
  - Already fetched or found: Leray (fetched); Qiu et al. 2025, gated attention (fetched; NeurIPS 2025 Best Paper, confirmed); Yang et al. 2025, PaTH (fetched); Nakanishi 2025, SSMax (fetched); Barbero et al. 2025, "Why do LLMs attend to the first token?" (found); Erel et al. 2025, "Attention (as Discrete-Time Markov) Chains" (found).
  - Still to fetch: Hansen & Ghrist 2019, sheaf Laplacians [U]; Bodnar et al. 2022, neural sheaf diffusion [U]; Krohn & Rhodes 1965 [U]; Liu et al. 2023, flip-flop language [U]; Kemeny & Snell 1960, absorbing chains [U]; Abnar & Zuidema 2020, rollout [U]; Xiao et al. 2023, StreamingLLM [U]; Sun et al. 2024, massive activations [U]; Correia et al. 2019, adaptively sparse transformers [U]; Blelloch 1990, segmented scan [U].
- **Iteration lines:**
  - it.N0: re-run the instances.
  - it.N1: R-DIAG and R-SINK on the existing checkpoints.
  - it.N2: C1–C3 folded into Addendum L's it.L0.
  - it.N3: L1-A5′ with the §8a arms (s_Q, DeltaProduct n_h = 2, PaTH, GDN) and a length sweep.
  - it.N4: R-NEVER-LEN.
  - it.N5: R-HOLONOMY.
  - it.N6: R-COST-N.
  - it.N7: tables; the one-line leap kept or struck.

---

## 8. Report format

For each row: prediction, author counter, measured value with a 95% CI (5 seeds unless stated), verdict (CERTIFIED / STRUCK / VOID), and the SHA of the code that produced it.

---

## 8a. Errata after the it.L0 report (commit 492ccff)

All seven errors below are the author's, in Addenda L and M.

- **Q2 generators.** The pair makes an infinite group; 54.74° is the cube's 4-fold/3-fold angle. Confirmed three ways (rooms, Inspector, T10). Fixed in §4 C1.
- **Q3 was a tautology.** It held the attention pattern fixed. By D2, order can also enter through the scores, so FoX telling AB from BA at 0.143 is legitimate. Q3 is struck; the order question moves to the scan-only control (C2).
- **PHASE-FIELD constant and regime.**
  - For P_ε as written, the interface constant is 2∫₀¹√W = **1/3**. The √2/6 on the page was the ½-convention.
  - The Γ-limit only appears when ε is large compared with one token; for ε ≪ 1 the lattice cost is ε per boundary. *`pf_regime.py`: 0.0299 at ε = 0.03 (the rooms' "true lattice minimum"), 0.0980, 0.2509, 0.3305, 0.3330, then 0.3333 from ε = 10.*
  - The Γ-check grid becomes ε ∈ {1, 3, 10, 30}. At small ε, the penalty ÷ ε counts the barcode boundaries directly.
- **M1 null was unpassable.** "Coherence < 0.2 everywhere" takes a maximum over thousands of points of a statistic that is biased upward at low SNR (Wilson's Rician finding). The rooms' replacement null is accepted. Pin its definition in Addendum M as a permutation quantile with the noise bias subtracted.
- **"No custom kernel" is false.** The pass uses a Triton scan. Replace with: "one Triton scan plus stock SDPA, 1.03–1.09× forward, 1.08–1.09× forward+backward (GPU re-run by the Inspector)."
- **Two-sided fold in the text, left-only in the arm.** D2 holds for either representation. Keep the arm: left action on R⁴, where the group is 2I with 120 elements. A₅ labels then need a sign-invariant readout (q ~ −q). Fix the §1 text to match.
- **Board bed chosen without a floor run.** A rule-free tracker scores 0.9968, so the bed is retired, as the floor rule requires.

**Owners the novelty sentence must read before it.L1 writes any claim.** All fetched; §4 C3 is extended.
- Grazzi et al. 2024: eigenvalues in [0,1] cannot solve parity; extending to [−1,1] fixes Mamba and DeltaNet.
- DeltaProduct (Siems et al., NeurIPS 2025): products of n_h generalized Householders per token, with negative eigenvalues. Two reflections compose to a rotation, so n_h = 2 already reaches unit-circle complex eigenvalues. That is the rooms' "only new thing left." This is linear algebra, not a quoted claim; their exact groups were not in the abstract [U].
- PaTH (Yang et al. 2025): the same thing inside softmax attention, on the q·k side.
- The residue that could still be claimed is an engineering composition: a value-side fold on stock SDPA at 1.03–1.09×, plus an exact reset from magnitude zeros. It is not a new capability.

**Arms added to it.L1 (L1-A5′).**
- (s_Q) scan only. f_Q 1.0 vs FoX 0.016 (≈ 1/60, chance) certifies the scan. Foreman's BOS finding is D2's readout path.
- DeltaProduct n_h = 2 and PaTH, beside the GDN floor.
- A length sweep, not position 64 alone.

**Guards for the two ops failures Wilson found.**
- Record files are append-only: every write goes through one helper, and a pre-commit test fails if any record file shrinks.
- A citation marked "fetched" must have a stored artifact (URL, date, content hash). A test fails on any "fetched" without one. That is what the hard-coded dict violated.

---

## 9. Reference instances (L-REPRO)

`diag_leap.py`: numpy 2.4.4, float64/complex128. Seed 0 for T1–T5 and T9, seed 1 for T6–T8. `crossties.py`: seed 7; X2b in fp32.

| id | quantity | value |
|---|---|---|
| T1 | max \|eig(W) − diag(W)\|, n = 64 | 0.0 |
| T2 | max \|r_i − v₀\| at γ = .99 / .999 / .9999 | 1.2e−1 / 1.2e−2 / 1.2e−3 |
| T2b | residue change under new scores + tokens; σ₂/σ₁ | 6.6e−4; 4.1e−5 |
| T3 / T3b | U(1) similarity residual; off-v₀ residue | 1.6e−16; 4.3e−4 |
| T4 | SU(2) similarity residual | 3.3e−16 |
| T5 | λ law residual; unit diagonals | 4.4e−16; [0, 16, 40] |
| T6 | mixed rows, n = 64 / 1024 | 44/58; 855/929 |
| T7 | error after D−1 hops / after D hops | 4.9e−4 / 0.0 (D = 12); 1.9e−6 / 0.0 (D = 21) |
| T8 | sparsemax false mass & mismatches | 0 & 0/348; 0 & 0/88,255 |
| T8 | softmax false mass & best-threshold mismatches | 0.002 & 0/348; 0.110 & 703/88,255 |
| T8 | one-hop leak = law | 0.071 (n = 64); 0.559 (n = 1024) |
| T9 | forward substitution vs dense solve | 1.2e−15 |
| T10 | Q2 as written; corrected | > 5,000, not closed; 120 quaternions / 60 rotations |
| X1 | polar-split similarity residual | 2.0e−15 |
| X2a | output change after replacing everything before the boundary | 8.9e−16 |
| X2b | fp32 error, global vs segmented prefix | 4.0e−7 vs 1.3e−6 (prediction struck) |
| X3a | reset+group head, max rotation error (atan2) | 2.2e−13° |
| X3b | recency opponent at slope .05 / .2 / 1 (atan2) | 0.4565 / 0.8857 / 1.0000 |
| KR | recency opponent at slope ≥ 1, densities .05 / .2 / .5 (`kr_density.py`, seed 11) | 1.0000 at every density (X3 struck) |
| PF | discrete interface cost at ε = .03 / .1 / .3 / 1 / 3 / 10 / 30 (`pf_regime.py`) | .0299 / .0980 / .2509 / .3305 / .3330 / .3333 / .3333 |
| X4 | H⁰, pure gauge vs 50° holonomy (R⁴ / R³) | 4/3 vs 0/1; gaps 2.000 vs 0.012/0.047 |
