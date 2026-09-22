# Chase: Addendum N rebuilt and broken

**Verdict:** the diagonal theory holds on its own instances. Every D1–D5, X1 and X2a residual is below 1e-12 in f64, so N's kill line does not fire. It does not hold as written on our operator at the trained β. At β ≠ 1:
- the unit-diagonal count finds 0 of 3 absorbers;
- an absorber's eigenvalue can exceed 1, which puts a resolvent pole inside γ < 1;
- D3's λ law is wrong for any phase-carrying gate.

X2b is not reproduced: in this rebuild, segmenting is 5.9× more accurate in fp32.

**Code:** `scratchpad/phase_j/N1/chase/`
- `n_impl.py`, sha256 `05dfb0aa21f2c542…`
- `test_n.py`, sha256 `19efc9b39a4e83d5…`
- stub `n_impl_stub.py`, sha256 `f3d0f089c2b5dca8…`
- logs: `run1.log`, `run2.log` (final), `b3_gamma.log`
- results: `results_n_impl.json`

The test file exits 1 on any failed assertion.

**How the RED runs were recorded:** every test first ran as `NIMPL=n_impl_stub python test_n.py`, before `n_impl.py` existed. Each such run printed `RED   <name> NotImplementedError: instance not built yet`. On the board, those events carry the name `scratchpad/phase_j/N1/chase/test_n.py::<name> [n_impl_stub]`: the same test function, with the module tag appended. KR and PF got their own stub RED run after the v2 amendment and before they were implemented.

**Conventions (stated, not tuned):**
- Scores s ~ N(0,1), V ~ N(0,1) with d = 8, n = 64, γ = 0.99.
- Gates: m ~ U(0.3, 1) with exact zeros at 16 and 40; θ ~ U(−π, π).
- Unit quaternions are normalised Gaussian 4-vectors.
- Path products are computed directly (g_i⋯g_{j+1}), so D2 is actually tested.
- DAG: token 0 is a root; each later token is a root with probability 0.1, otherwise it has min(2, i) uniform earlier parents. Logits are Δ = 6 on parents (on self for a root) and 0 elsewhere.
- Sparsemax is Martins & Astudillo.
- Plaquette: a 4-cycle with pure-gauge restriction maps g_v g_wᵀ; holonomy is put on one edge.
- Recency decode: nearest of the 120 elements of 2I, sign-invariant.
- PF: unit lattice, m = 0 → 1, L-BFGS-B minimisation.

## Instances

| id | author | mine | RED → final | verdict |
|---|---|---|---|---|
| T1 | 0.0; unit [0] | 0.0; [0] | stub RED → GREEN | CERTIFIED |
| T2 | 1.2e−1 / 1.2e−2 / 1.2e−3 | 1.20e−1 / 1.23e−2 / 1.23e−3 | stub RED → GREEN | CERTIFIED |
| T2b | 6.6e−4 vs inputs 4.62; σ₂/σ₁ 4.1e−5 | 5.9e−4 vs 4.68; 4.12e−5 | stub RED → GREEN | CERTIFIED |
| T3 | 1.6e−16 | 7.95e−16; eigenvalue shift 1.1e−16 | stub RED → GREEN | CERTIFIED |
| T3b | diagonal unchanged; off-v₀ 4.3e−4 | diagonal equal to within 1e−16 but **not bitwise**; off-v₀ 4.8e−2 / 4.8e−3 / 4.9e−4 at γ = .99 / .999 / .9999 | stub RED → run1 RED (`re_diag_bitwise_equal: False`) → GREEN | CERTIFIED |
| T4 | 3.3e−16 | 2.2e−16 | stub RED → GREEN | CERTIFIED |
| T5 | 4.4e−16; [0, 16, 40]; σ 18.9 / 17.1 / 10.5 / 0.014 | 1.7e−16; [0, 16, 40]; σ 18.90 / 17.11 / 10.49 / 0.0022 | stub RED → GREEN | CERTIFIED (4th σ differs; rank 3 holds) |
| T6 | 44/58; 855/929 | 56/56; 881/917 (different DAG) | stub RED → GREEN | CERTIFIED as structure; counts not comparable |
| T7 | 4.9e−4 / 0.0 (D = 12); 1.9e−6 / 0.0 (D = 21) | 8.0e−3 / 8.9e−16 (D = 9); 4.3e−8 / 8.9e−16 (D = 26); N^{D+1} ≡ 0 bitwise | stub RED → GREEN | CERTIFIED |
| T8 sparsemax | 0 & 0/348; 0 & 0/88,255 | 0 & 0/448; 0 & 0/98,119 | stub RED → GREEN | CERTIFIED |
| T8 softmax | 0.002 & 0/348; 0.110 & 703/88,255 | 0.0083 & 0/448; 0.120 & 697/98,119 | stub RED → recorded | CERTIFIED pattern (the n = 64 tie is reproduced) |
| T8 leak law | 0.071; 0.559 | 0.0713581 = law (5e−17); 0.5588188 = law | stub RED → GREEN | CERTIFIED |
| T9 | 1.2e−15 | 2.7e−15 | stub RED → GREEN | CERTIFIED |
| T10 | > 5,000, not closed; 120 / 60 | 5,762, not closed; 120 / 60, contains −1; axis angles 54.7356° / 37.3774° | stub RED → GREEN | CERTIFIED |
| X1 | 2.0e−15; eigenvalues = diag; [0, 16, 40] | 1.9e−15; eigenvalue − diag = 0.0; [0, 16, 40] | stub RED → GREEN | CERTIFIED |
| X2a | 8.9e−16 | direct 0.0 (bitwise); scan form 8.0e−15 | stub RED → GREEN | CERTIFIED |
| X2b | global 4.0e−7 vs segmented 1.3e−6 | global **3.9e−6** vs segmented **6.6e−7** (means 1.5e−6 vs 1.8e−7); global norm drift 2.1e−6 | stub RED → still RED (`'global': 3.899e-06, 'segmented': 6.571e-07`) | **VOID** (not reproduced; the ordering is reversed) |
| X3a (v2, atan2) | 2.2e−13° | 4.3e−13° (194 resets) | stub RED → GREEN | CERTIFIED |
| X3b (v2) | 0.4565 / 0.8857 / 1.0000 | 0.4802 / 0.8604 / 1.0000 | stub RED → GREEN | CERTIFIED; the X3 strike stands |
| KR (v2) | slope ≥ 1: 1.0000 at every density; slope .5: 1.0000 / 0.9719 / 0.8501 | 1.0000 at every density; 1.0000 / 0.9800 / 0.8547 | stub RED → GREEN | CERTIFIED |
| PF (v2) | .0299 / .0980 / .2509 / .3305 / .3330 / .3333 / .3333 | .02995 / .09802 / .25091 / .33045 / .33302 / .33331 / .33333 | stub RED → GREEN | CERTIFIED (constant 1/3) |
| X4 | H⁰ 4/3 vs 0/1; gaps 2.000 vs 0.012 / 0.047 | 4/3 vs 0/1; 2.000 / 2.000 vs 0.01189 / 0.04741 | stub RED → GREEN | CERTIFIED |

## Breaks

All RED lines below are from `run2.log`.

| break | measured | RED → final | verdict |
|---|---|---|---|
| D1 with row mass < 1 at β = 1 (arm_smprime; Re row mass 0.367, modulus mass 1.000) | unit diagonals 3 = structural absorbers 3 | stub RED → GREEN `B1_D1_unit_count_beta1` | CERTIFIED: a phase deficit does not break D1 |
| D1 at trained β 1.09 / 0.96 / 0.99 | 0 unit diagonals vs 3 absorbers (0/60 over 20 draws). Absorber W_ii = e^{(1−β)s_ii}: 1.068 / 1.063 / 1.026 at β = 1.09 | still RED: `('1.09', {'unit': 0, 'structural': 3, ...})` | **STRUCK** for β ≠ 1 |
| Resolvent pole inside γ < 1 | at β = 1.09, max W_ii = 1.0677, so the pole is at γ* = 0.937 < 0.99. Draws with a pole below 0.99: 16/20 at β = 1.09, 16/20 at 0.96, 10/20 at 0.99 | still RED `B5_no_pole_inside_unit_gamma` | **STRUCK**: "at any γ < 1" (D4/D5) requires β = 1 |
| D3 as written (ρ uses complex G_{i−1,j}) | residual 1.33; with \|G_{i−1,j}\|, 2.2e−16 | still RED `B2_D3_as_written_complex_gate`; `B2r` GREEN | **STRUCK** as written; the modulus form is CERTIFIED |
| D5 depth for dense causal softmax | D = n − 1 (63 / 1023). The jump chain reaches 1e−12 in 22 / 30 hops, the same at γ = .99 / .999 / .9999; plain Neumann takes 2,749 | stub RED → GREEN `B3_D5_dense_depth` | theorem CERTIFIED; **"the cost is the causal depth" STRUCK for dense heads** |
| R-DIAG, gate route | diagonal bitwise identical across θ; count 3 = 3 | stub RED → GREEN | CERTIFIED at β = 1 |
| R-DIAG, logit/both route | count 3 = 3, but the diagonal moves by up to **0.830** with θ | still RED `B4_RDIAG_logit_route_diag` | **STRUCK**: the count check is blind to score-side phase |
| R-DIAG at trained β | count 0 under both θ, so the check "passes" | still RED `B4_RDIAG_can_fire_at_trained_beta` | **VOID**: the check cannot fire |
| Rounding-manufactured unit diagonals (m = 0.9 everywhere, peaked self-scores, structural count 1) | 2 unit diagonals in f64, 13 in f32 | recorded inside `b4` | finding: the float W_ii == 1 read overcounts |

**Lean (read-only; Mathlib rev a45ae637):**
- `Matrix.det_of_lowerTriangular` is at `lean/.lake/packages/mathlib/Mathlib/LinearAlgebra/Matrix/Block.lean:265`, with hypothesis `M.BlockTriangular toDual`. The [U] on it can be lifted.
- There is no charpoly-under-conjugation lemma. The nearest are:
  - `Matrix.det_units_conj` (`LinearAlgebra/Matrix/Determinant.lean:215`)
  - `LinearMap.charpoly_toMatrix` (`LinearAlgebra/Charpoly/ToMatrix.lean:46`)
  - `Matrix.charpoly_reindex` (`Charpoly/Basic.lean:107`, permutations only)
- So `gauge_similarity_charpoly` needs a proof, not a citation.

## Replacement routes

**D1 / R-DIAG count at β ≠ 1**
- *What it was for:* an O(n) count of closed classes with no eigendecomposition.
- *Reroute:* count structural absorbers as {i : m_i == 0 exactly} ∪ {0}. This is O(n), independent of β, and reads the gate rather than a float diagonal. Report each absorber's eigenvalue e^{(1−β)s_ii} separately.
- *Measurement:* per layer on the (f) checkpoints, the exact-zero m count vs the W_ii == 1 count.
- *Seat:* Chase, R-DIAG.
- *Why sharper:* it cannot miss 3 of 3 absorbers at β = 1.09, and it cannot count the 12 rows that fp32 rounding manufactures.

**R-DIAG check**
- *Reprice:* require a bitwise-identical **diagonal** across θ, not identical counts. The counter fires when θ enters Z (logit/both routes), which the count check misses by 0.830.

**Pole (D4/D5 "at any γ < 1")**
- *Reprice:* every Mode B read checks γ < γ* = 1 / max_i W_ii. This is an O(n) read off the diagonal.
- *Measurement:* max W_ii and γ* per layer on the checkpoints.
- *Seat:* Cameron, R-COST-N / Mode B.
- *Alternative:* retire β ≠ 1 on any head whose equilibrium is read.

**D3 text**
- *Reroute:* ρ_i = Σ_{j<i} |G_{i−1,j}| e^{s_ij − s_ii}, which is arm_smprime's R_{i−1,j}. Extend T5 to a phase-carrying gate (measured 2.2e−16).

**C4 cost**
- The saving comes from forward substitution (T9), not from depth; dense heads have D = n − 1.
- *Reroute:* add the jump-chain hop scheme as a third arm in R-COST-N (22 / 30 hops to 1e−12, independent of γ, vs 130 Chebyshev hops), measured on the checkpoint operators at S = 4096.
- *Seat:* Cameron.

**X2b**
- The author's strike of the segmented-scan prediction does not reproduce.
- *Reroute:* pin the inverse (conjugate vs true inverse) and the renormalisation convention, then re-run with the author's script once it exists.
- *Seat:* Cameron. Until then, the segmented scan is the fp32 default: it is 5.9× better on the max error and 8.6× on the mean.

## OPEN

- None of diag_leap.py, crossties.py, kr_density.py or pf_regime.py exists. Seed-dependent counts (T6, T7's D, T8's cells) are my DAG, not the author's, and cannot be compared one to one.
- The pole frequency (16/20, 10/20) comes from random q and k. The trained s_ii distribution decides whether the checkpoints actually sit past a pole. That is the one checkpoint read still owed.
- The jump-chain hop counts are on diffuse random-score softmax. Peaked chains of weights near 1 converge slower, up to D.
- House owns the D2 audit at β ≠ 1. This rebuild shows D1 and D3 break there; it did not test D2.
- **Default next move:** run the structural-count and γ* reads on the four trained layers of the existing (f) checkpoint, CPU, no training, as Chase's R-DIAG.
