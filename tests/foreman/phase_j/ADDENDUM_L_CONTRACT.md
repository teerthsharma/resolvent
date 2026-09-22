# PHASE J — ADDENDUM L (LEAP)

**The trigger passed on all three counts: the theory is dead, it died for want of an idea, and it is still wanted. This addendum supplies the idea in one sentence and the three people who make it precise: the gate is a piecewise-constant non-commutative connection — segments from a phase-field limit (De Giorgi, Landau), rotations from the smallest non-abelian group (Hamilton, Galois), drift read as Euler-pole rotation between segments (Wegener, Bullard). A forget gate is the scalar shadow of this object and provably cannot compose order.**

Author: Seal (Teerth Sharma). Contract date: 2026-09-22. Amends PHASE_J_HYPERSPACE_v2.md. Runs beside House's leap; if both survive it.L0 they are one row each; if only one does, that one is the leap.

---

## 0. What died, stated once

- Phase (U(1), abelian) lost twice. Row mass lost. Temperature lost. Abstention shared with FoX. Kernel 1.99×–3.9×. Nothing the scalar family has that softmax lacks has held up, and the scalar family is FoX up to endpoints.
- The one certified structural capability that survived every round is **order**: I-AUT, operator 0.8620 vs the commuting control 0.2860 saturated at its multiset ceiling 0.3110, generator handed over as bare integers. It never reached the language rows because the r×r operator gate had no cheap path. This addendum gives it one.

---

## 1. The leap: the SU(2) gate

- **Object.** Per token a unit quaternion q_k ∈ SU(2) from content (a `Linear(d,4)` head, normalized). Path product G_ij = q_i q_{i−1} ⋯ q_{j+1}. Values carry a 4-block (or several) on which the product acts.
- **Why it is cheap.** Quaternion products are associative and invertible (inverse = conjugate), so the prefix products Π_i = q_i ⋯ q_1 come from one **associative parallel scan** (log-depth, O(n) work), and G_ij = Π_i · conj(Π_j). The head then reads Σ_j p_ij G_ij v_j = Π_i · Σ_j p_ij (conj(Π_j) v_j Π_j)⋯ — a rotation of values before stock SDPA and a rotation of the output after. **RoPE-class overhead; the attention pattern p_ij is untouched; no custom kernel.** This is the Phase G reduction (I4: G_ij = Π_i Π_j⁻¹ for invertible gates) applied to a non-commutative group.
- **Why it does what a forget gate provably cannot.** Any commutative gate — scalar decay (FoX), diagonal decay (Mamba-class), abelian phase — makes G_ij a function of the multiset of gates between j and i, hence order-blind; measured as the DIAG control at the multiset ceiling; stated as a theorem for diagonal state-space models and transformers [U: Merrill, Petty, Sabharwal 2024, "The Illusion of State"; Grazzi et al. 2024, negative eigenvalues]. SU(2) contains the icosahedral group A₅ (60 elements), the smallest non-solvable group; the A₅ word problem is NC¹-complete by Barrington's construction, whose 5-cycles are even permutations; Galois' theorem (A₅ is simple, S₅ not solvable) is the reason. **The gate is order-capable by theorem and cheap by construction.**
- **What it is not.** It is not a new attention pattern, not a new kernel, and not a prediction of lower LM loss at 725k parameters — state tracking is not the bottleneck there, and the page says so before the row runs. It is a certified capability at RoPE cost, which is exactly what condition 3 lacked.
- **Instances to run in it.L0** (the author's script was halted before execution; no numbers are claimed):
  - Q1: parallel scan vs sequential prefix product, n = 4096, f64 — expected ≤ 1e−12; G_ij by direct path product vs Π_i·conj(Π_j) — expected ≤ 1e−12; fp32 chain of 4096 with renormalization every 256 — expected ≤ 1e−4 against f64 (the S-linear rounding law of I.1 makes bf16 forbidden for the scan).
  - Q2: two generator quaternions (72° about z, 120° about (1,1,1)/√3) generate a closure of **60** elements; a length-10,000 random word composed by quaternion products equals the group-table truth; |ab − ba| > 0 while any commutative gate gives identical products for both orders.
  - Q3: attention read with a fixed value vector distinguishes AB from BA (|out(AB) − out(BA)| > 0); a scalar or phase gate gives exactly 0.
- **Prior art, conceded before the row:** unitary and quaternion RNNs [U: Arjovsky 2016; Parcollet 2018]; DeltaNet/RWKV-7 state tracking via non-diagonal transitions [U]; LieRE Lie-group position encodings [U: Ostmeier 2024], which are commutative in position. What is claimed: the composition — content-dependent SU(2) gate, folded into values by the prefix-product trick on top of stock SDPA, with the A₅ word problem as its certificate.

---

## 2. De Giorgi and Landau: the barcode is a sharp-interface limit

- **Who.** Ennio De Giorgi (fetched): Hilbert's 19th problem 1956–57 *"in parallel with John Nash"*, De Giorgi's published first; Caccioppoli Prize 1960 *for sets of finite perimeter*; Wolf Prize 1990; Γ-convergence (1975). The Russian half: the Ginzburg–Landau free energy (Landau 1937; Ginzburg–Landau 1950 [U]); the bridge between them is the Modica–Mortola theorem (1977 [U]): the phase-field functional Γ-converges to a perimeter functional as ε → 0. (Ladyzhenskaya and Pontryagin are the other Russians the pointer could mean; named on the page, not used.)
- **Object.** Train the magnitude gate under the phase-field penalty P_ε(m) = Σ_k [ ε (m_{k+1} − m_k)² + W(m_k)/ε ], W(m) = m²(1−m)². Γ-limit as ε → 0: m ∈ {0, 1} almost everywhere and P → c_W · (number of segment boundaries), c_W = ∫₀¹ √(2W) ds = √2/6. **The barcode is the sharp-interface limit of a diffuse gate, and the number of segments is its perimeter.** Exact zeros become a thermodynamic outcome, not a parameterization trick.
- **What it explains, testably.** "The gate closes and nobody knows why" is, under Landau, the m = 0 well winning a double-well competition. Row PHASE-FIELD: train (f) with P_ε at ε ∈ {1, 0.3, 0.1, 0.03}, 5 seeds; report the fraction of gates within 0.05 of {0, 1}, the segment count, and the penalty value; prediction: penalty ÷ segment count → c_W within 20% at ε = 0.03 (the Γ-limit check), loss vs segment count traces a curve with a knee (the phase diagram), and the effective well asymmetry W_eff(0) − W_eff(1), read from the loss along the m-axis of the trained model, has the sign that predicts the closure. Counter: no knee and no asymmetry — the closure is not a phase transition, and De Giorgi's page line stays a line.
- **Constant on record:** c_W = √2/6 ≈ 0.2357 for W = m²(1−m)².

---

## 3. Wegener and Bullard: drift is a rotation

- **Who.** Alfred Wegener (German; continental drift 1912; served in the war and did not emigrate; rejected by German geology in his lifetime; died on the Greenland ice in 1930; vindicated by plate tectonics in the 1960s [U]). The mechanism came from others; the *fit* was made rigorous by Bullard, Everett and Smith (1965 [U]): the best rotation of one continental shelf onto another about an Euler pole. Euler's rotation theorem (1776) says plate motion is a rotation; plate motions compose non-commutatively. Wahba's problem (1965) and Horn's quaternion solution to absolute orientation (1987) [U] are the same fit in the SU(2) coordinates of §1.
- **Object.** Row DRIFT: for consecutive segments of the barcode, take the c-space clouds (or the value-block clouds) and fit the best rotation between them by Horn's quaternion method (closed form, an eigenvector of a 4×4 matrix). Drift = the rotation angle. A phase boundary is a jump in the fitted rotation. Null: split one segment in half and fit — angle must be near zero. Prediction: drift angle peaks at the barcode's boundaries and at the near-1-eigenvalue changes of R-PHASE, agreeing on ≥ 80% of boundaries; counter: no agreement — one instrument is noise, and the Tutte-delta diagnostic decides which.
- **Note's own words honoured:** *"equilibrium breaks when we detect drift"* — the drift is now a number with a null.

---

## 4. How the three combine, in one sentence

The gate is a **piecewise-constant SU(2) connection along the sequence**: segment boundaries from the Γ-limit (magnitude, De Giorgi–Landau), a rotation per token inside segments (order, Hamilton–Galois), and the rotation between segments read as drift (Wegener–Bullard–Horn). The forget gate is the U(1)-scalar shadow of this object and cannot compose order by theorem. Mode B′ = this connection folded into values on top of stock SDPA; Mode B (the resolvent read) is unchanged and sits on top.

---

## 5. Make-or-break for the leap (three counts, pre-registered)

1. **Exactness:** Q1–Q3 as stated; any failure → no leap, and the addendum is filed as "attempted".
2. **Cost:** Mode B′ at ≤ 1.3× SDPA wall-clock at S = 4096 in fp32 (scan + two rotations); > 2× → the capability exists but the cost sentence is struck.
3. **A bed where language needs it:** chess move-sequence → board-state prediction (Toshniwal et al. 2022 [U], where transformers track state imperfectly); Dyck-k / code scope depth with types; the A₅ word problem itself. Arms: (a) twin, (a″) FoX, (f_Q) SU(2) gate. Prediction: (f_Q) beats (a) and (a″) on board-state accuracy at ply ≥ 40 by ≥ 0.10 absolute and ties on LM loss within 0.01 nats. Counter: (f_Q) ties (a″) on board state — then the transformer's own depth already tracks the state at this length, and the leap is a theorem without a bed; filed, not spun.

---

## 6. Kills

- Q2 closure ≠ 60 or the word problem fails → the gate is not the group it claims; no leap.
- Cost > 2× → capability without the cost leg; condition 3 unchanged.
- Board-state tie with (a″) at ply 40 and 80 → no bed; the leap is a certificate only.
- PHASE-FIELD without knee or asymmetry → the barcode stays a parameterization, De Giorgi stays a line.
- DRIFT agreeing with the barcode on < 50% of boundaries → one of the two instruments is struck by the delta diagnostic.

---

## 7. Room, fetch orders, iteration

- Chase — Q1–Q3, the scan, the value-fold, Lean target `a5_word_problem_by_su2` (A₅ ⊂ SO(3), non-solvable; cite Mathlib's `Equiv.Perm` and the icosahedral group if present [U]). Foreman — PHASE-FIELD training, (f_Q) arm. Cameron — DRIFT instrument (Horn), the board-state bed, cost row. Wilson — the "capability, not loss" sentence pinned above the row; majority-of-three checkers; House's leap and this one scored on the same table.
- Fetch before any table (L-EQ): Modica–Mortola 1977; Ginzburg–Landau 1950; Bullard–Everett–Smith 1965; Wahba 1965; Horn 1987; Merrill–Petty–Sabharwal 2024; Grazzi et al. 2024; Toshniwal et al. 2022; Arjovsky 2016; Parcollet 2018; Ostmeier 2024; Wegener 1912/1915 and the 1960s reception; Hamilton 1843.
- it.L0 — Q1–Q3; closure count; cost microbenchmark. it.L1 — (f_Q) on the A₅ bed and I-AUT. it.L2 — board-state bed, three arms. it.L3 — PHASE-FIELD sweep. it.L4 — DRIFT with its null. it.L5 — tables; the one-sentence leap kept or struck.

---

## 8. Reference instances

None claimed. The author's instance script for Q1–Q3 was halted before it ran; every number above marked "expected" is a prediction, and it.L0 replaces it with a measurement or a strike.

---

*Open leap question (author):* this buys a certified capability at RoPE cost, not lower LM loss at 725k; the bed where language needs it is chess move-sequence → board state (Toshniwal 2022) and code scope. If (f_Q) ties FoX there, the leap is a theorem without a bed, and we file it that way.
