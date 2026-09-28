# SUN PLAN A — THE SHELVED CANON

**A new topology for consequence, built by hand from mathematics that history shelved.**

The frontier labs are running a size race on a single topology: flat Euclidean embeddings, fixed depth, softmax mixing. Three facts measured for this plan say size is the wrong lever for *consequence*.

1. **Consequences branch, and branching wants negative curvature.** A 255-node consequence tree embeds in **2 hyperbolic dimensions with worst-case distortion 1.216**. Classical Euclidean MDS needs **128 dimensions** to reach 1.815 (instance S1).
2. **Consequences are chaotic, and chaos taxes size logarithmically.** On Lorenz-63 (λ = 0.9059), every 1000× gain in initial precision buys only **about 7.6 time units** of forecast (S2). Past the Lyapunov horizon, the only honest output is a calibrated distribution. No parameter count changes that.
3. **Language is self-similar** (Hurst parameter ≈ 0.7). Fractal parameters predict downstream performance beyond bits-per-byte (Alabdulmohsin, Tran, Dehghani 2024, fetched).

Canon builds each structure into the architecture from sidelined mathematics. It goes pillar by pillar, and each pillar is gated against its strongest owner on a bed whose truth is exact.

Author: Seal (Teerth Sharma). Date: 2026-09-27. Instance scripts: `sun_maths.py` (S1, S2) and `sun_budget.py`. They ship with this file and go into the repo with a pinned SHA before any row runs.

---

## 0. Laws

- **All earlier laws apply:** L-EQ, L-REPRO, L-REFLECTOR, L-SHA, L-AUDIT, L-BR, L-SHORTCUT, L-LEARN, L-TRAINED, L-LOSS, L-SCALE.
- **L-SUN:** no sentence claims AGI. The plan climbs **waypoints** (§5), each measurable and each with a kill.
- **L-OWNER:** every pillar names its strongest published owner, and runs that owner as an arm before any claim.
- **Security unchanged:** uploads only on the author's word; tokens and keys stay local; no GPU process is killed by name.

---

## 1. The shelved canon: what each source contributes

Everything below was fetched unless marked [U].

| Source | Its fate | What Canon takes | Strongest owner (must be an arm) |
|---|---|---|---|
| **Lobachevsky and Bolyai**, hyperbolic geometry [U] | Sidelined in their lifetimes | A hyperbolic latent for consequence trees and DAGs (S1) | Poincaré embeddings (Nickel & Kiela 2017, fetched); hyperbolic networks and transformers [U] |
| **Hermann Grassmann**, *Ausdehnungslehre* (1844, 1862) | Ignored. Kummer's verdict ended his academic hopes. About 600 copies were pulped as waste paper in 1864. He taught at a gymnasium in Stettin. Clifford extended the work in 1878 into geometric algebra, and vector spaces caught up around 1920. | The **geometric product**: one algebra carrying order and orientation, non-commutative by construction. This is the program's one surviving result (order, 0.8620 vs 0.2860) in its oldest form. | Geometric Algebra Transformer, Clifford layers [U]; HAE for group actions (fetched) |
| **Bernard Koopman** (1931) | A linear Hilbert-space operator on the observables of a nonlinear system. It inspired von Neumann's ergodic theorem (1932, fetched). Modern revival (Mezić, DMD) [U]. | A **linear lift** of nonlinear dynamics, with our resolvent/Volterra machinery acting on the lift | DMD/EDMD, deep Koopman [U]; TD-JEPA (successor features, fetched) |
| **Poincaré** (1890) and **Lorenz** (1963) [U] | Sensitivity to initial conditions: the butterfly | A **Lyapunov-aware horizon**. Outputs carry a spread that grows as e^{λt}, and the model must be calibrated past T* (S2). | Reservoir computing (next-generation RC [U]); transformer forecasters |
| **Hausdorff** [U] and **Mandelbrot** [U], with DeepMind's fractal-language paper (fetched) | Fractals sat at the edge of mainstream practice for decades | **Self-similar mixing:** one rule shared across scales 2^k, so parameters are not repeated per scale | Multi-scale and hierarchical transformers [U] |
| **Alexey Ivakhnenko**, GMDH (Kyiv, 1968; eight-layer networks by 1971, per Schmidhuber; fetched) | A Soviet deep-learning method that got limited adoption in the West | **Topology grown layer by layer**, each layer kept only if it improves held-out data. "New topology" taken literally. | Neural architecture search [U] |
| **Setun** (Moscow State University, 1958; Brusentsov and Sobolev; 50 built; stopped in 1965 because it sold too cheaply; fetched) | Balanced ternary, shelved | **Ternary weights {−1, 0, +1}.** BitNet b1.58 (fetched) claims to match FP16 transformers at the same size and tokens. "Never the size", in bits. | BitNet b1.58 |
| **D-JEPA** (Liu et al., arXiv 2609.24749, 21 Sep 2026; full HTML fetched 2026-09-27) | Current, not shelved: the owner of "JEPA + consequences + bounded operator" | Nothing is taken. It is the owner Canon must beat or concede to on consequence. Its operator is two 4-head Transformer layers with no positional encoding, so it is permutation-equivariant over candidate futures. It emits a rank-8 correction bounded by ε·tanh with ε = 0.2, trained with listwise success mass plus a pairwise softplus ranking loss on executed outcomes. Its "decision-local prediction gap" is empirical: within the top-4 candidates nearest the goal, Spearman falls from 0.90 to 0.11 for LeWM. The words chaos, Lyapunov, Koopman and hyperbolic do not appear in the paper. "Calibration" appears only as a tuning subset, and "horizon" only as rollout length (audited 2026-09-27). | D-JEPA itself: an arm in P2 (ranking of candidate actions), and the reference for every consequence claim |
| **ACPC** (An et al., arXiv 2608.12939, 13 Aug 2026; abstract fetched) | Current | A diagnostic only: action-conditioned predictive consistency, bisimulation-style (Invariance Radius, Separation Rate) | Not an arm; a read on P2 beds |
| **Tesla** | Tesla *won* the war of currents. What is sold as "Tesla's lost science" (scalar waves, free energy) is not mathematics. | **Nothing enters.** | — |

---

## 2. Measured for this plan (L-EQ)

- **S1 (negative curvature).** Binary tree, depth 7, 255 nodes. Sarkar's construction in the Poincaré disk, 2 dimensions:
  - τ = 1: worst-case distortion 5.762, mean relative error 0.0865.
  - **τ = 2: worst-case distortion 1.216, mean relative error 0.0146.**
  - τ = 4: 1.541 / 0.1286 (corrected 2026-09-27 from 1.554; rerun of `sun_maths.py`), where float64 loses precision near the boundary (max |z| = 1 − 7e−12). That is why Canon uses the **Lorentz model**, not the disk.
  - Euclidean classical MDS: 2 dimensions collapse distinct nodes (distortion ∞, mean error 0.3442; corrected 2026-09-27 from 0.3362); 10 and 50 dimensions still near-collapse some pairs (worst-case 6.75e15 and 2.28e14, mean error 0.1066 and 0.0506); **128 dimensions reach 1.815 / 0.0475.**
  - MDS is a baseline, not the optimal Euclidean embedding. The theoretical lower bound for Euclidean tree embeddings (Bourgain) is [U].
- **S2 (the butterfly).** Lorenz-63 (σ 10, ρ 28, β 8/3), RK4 with dt 0.01, Benettin's method over 200,000 steps: **λ = 0.9059** (the literature value is about 0.906).
  - Median time for an initial error δ to grow to 1, over 24 starts, against the law (1/λ)·ln(1/δ):

    | δ | 1e−3 | 1e−6 | 1e−9 | 1e−12 |
    |---|---|---|---|---|
    | measured | 6.20 | 14.00 | 22.38 | 29.59 |
    | law | 7.63 | 15.25 | 22.88 | 30.50 |

  - Each 1000× gain in precision buys **7.63** time units.
- **Budget** (`sun_budget.py`; every price is an **assumption**: H100 dense bf16 989 TF, MFU 0.40, $2.50 per GPU-hour, 20 tokens per parameter, 4k context, 5 arms × 3 seeds per rung):

  | Rung | Per run | Full grid | Cumulative |
  |---|---|---|---|
  | 125M | 0.9 GPU-h ($2) | $33 | $33 |
  | 350M | 10.2 GPU-h ($26) | $384 | $417 |
  | 1.3B | 143 GPU-h ($359) | $5,379 | $5,796 |
  | 2.7B | 605 GPU-h ($1,512) | $22,680 | $28,476 |
  | 7B | 3,789 GPU-h ($9,472) | $142,075 | $170,551 |

  - **A full five-arm ladder to 2.7B costs about $28k.** The constraint is the idea and the people, not the chips. Replace the assumed prices with a real quote.

---

## 3. The Canon block (each part is a switch, like CEQ's β / g / qk)

- **State:** points on a hyperbolic manifold (Lorentz model) carrying Clifford multivector features.
- **Mixing:** attention scored by −d_H (hyperbolic distance), inside self-similar windows at scales 2^k with one shared rule.
- **Composition:** the geometric product for ordered actions (non-commutative).
- **Prediction:** a Koopman lift with a linear operator K, whose growth rate is tied to a learned Lyapunov estimate λ̂. The output is a mean plus a spread that grows as e^{λ̂·t}, scored for calibration past T*.
- **Weights:** ternary. **Depth:** grown GMDH-style, where a layer stays only if held-out loss improves.
- **Everything off:** the ALiBi twin. That is the floor, per Phase J.

---

## 4. Pillars (each gated alone, before any integration)

For each pillar the rooms run floors, a shortcut hunt, and trained best responses before the bar.

**P1 Hyperbolic consequence.**
- Bed: tree- and DAG-structured causal worlds (root-cause and consequence-set queries, exact truth; the far band past the twin's reach, per Phase K), plus the WordNet noun hierarchy.
- Arms: Euclidean twin at equal parameters and at 8× width; Poincaré embeddings; a hyperbolic transformer [U].
- **Prediction:** hyperbolic at width d matches Euclidean at 8d on consequence-set F1.
- **Counter:** Euclidean at 2d matches.

**P2 Butterfly: Koopman lift plus horizon.**
- Beds: Lorenz-63; Kuramoto–Sivashinsky [U]; A5-World's policy-conditioned consequence queries (Phase L, L5).
- Metrics: skill before T*; Murphy reliability REL after T*.
- Arms: transformer forecaster, next-generation reservoir computing (Gauthier et al. 2021, fetched), EDMD (Williams et al. 2015, fetched), deep Koopman (Lusch et al. 2018, fetched), and **D-JEPA** (arXiv 2609.24749) on the candidate-ranking read.
- **D-JEPA sub-bed (P2-DJ):** do candidate-action rankings outlive state prediction past T*? The Bayes ceiling is measured from a perfect model with a Monte Carlo posterior (`p2/BAR_DJ.md`). Any Canon claim touching consequence must say what it does that D-JEPA does not. The open ground is the horizon: D-JEPA scores rankings with no notion of when a ranking stops being predictable.
- **Prediction:** equal skill before T*; after T*, REL ≤ 0.01 for Canon against ≥ 0.05 for the transformer.
- **Counter:** the transformer reaches REL ≤ 0.02, meaning calibration isn't architectural.

**P3 Grassmann order.**
- Bed: A5-World's C1 row. Arms: HAE (owner), DeltaProduct, PaTH.
- **Prediction:** a tie with HAE. **No novelty is claimed.** The pillar exists because Canon needs order.

**P4 Self-similar mixing.**
- Bed: FineWeb-Edu language modelling at R1–R2, with the ALiBi twin as floor.
- **Prediction:** equal loss (±0.01) at **half the unique parameters**. The model's own outputs should carry a Hurst exponent within 0.05 of the data's.
- **Counter:** worse by > 0.02.

**P5 Ternary plus growth.**
- Reproduce BitNet b1.58 at R1 first, then apply it to Canon.
- **Prediction:** equal loss at ≤ ¼ of the weight memory.
- **Counter:** BitNet's claim fails at our scale.

**Integration gate:** the Canon block is assembled only after **≥ 3 of the 5 pillars** pass.

---

## 5. Sun waypoints (measurable; no AGI sentence)

1. **Consequence beyond depth.** Exact root-cause and consequence-set accuracy on the far band at 1.3B, against Transformer++ and Gated DeltaNet trained on the same tokens.
2. **Honest past the horizon.** REL ≤ 0.01 beyond T*, where the matched twin is overconfident.
3. **Never the size.** Equal language-model loss at ≤ ½ the unique parameters with 1.58-bit weights.
4. **A named public task.** RULER multi-hop tracing and BABILong at 1.3B against matched baselines [U fetch first].

---

## 6. Kills

- Fewer than 3 pillars pass → no integration; the passing pillars are published alone.
- P1's counter holds → hyperbolic leaves Canon.
- P2's counter holds → the horizon is a training trick, not architecture.
- Waypoint 1 lost at 1.3B → Canon is filed as small-scale only.
- Any instance fails to reproduce → the plan is void until explained.

---

## 7. Room and timeline (12 weeks)

- **Chase:** P1 and P3.
- **Cameron:** P2 and the chaotic beds.
- **Foreman:** P4, P5, and the ladder.
- **Wilson:** the owners library, fetches, and the stored-artifact citations.
- **Inspector:** every iteration.
- **House:** the shortcut hunt on each bed.
- **Schedule:**
  - Weeks 1–2: instances, beds, floors.
  - Weeks 3–6: pillar gates at R0–R1.
  - Weeks 7–9: integration at R1–R2.
  - Weeks 10–12: 1.3B waypoint runs, the Zenodo record, and the paper.
- **Funding tiers:**
  - Tier 1 (≤ $125k): everything through 2.7B.
  - Tier 2: the 7B confirmation and the team.

---

## 8. Reference instances (L-REPRO)

`sun_maths.py`, seed 51, float64.

| id | value |
|---|---|
| S1 hyperbolic, 2 dims, τ = 1 / 2 / 4 | worst-case 5.762 / **1.216** / 1.541; mean error 0.0865 / **0.0146** / 0.1286 |
| S1 Euclidean MDS, 2 / 10 / 50 / 128 dims | ∞ / 6.75e15 / 2.28e14 / **1.815**; mean error 0.3442 / 0.1066 / 0.0506 / 0.0475 (six cells corrected 2026-09-27: 1.554, ≈5e15, ≈4e14, 0.3362, 0.1058, 0.0505 were stale) |
| S2 Lyapunov exponent | **0.9059** |
| S2 horizon at δ = 1e−3 / 1e−6 / 1e−9 / 1e−12 | 6.20 / 14.00 / 22.38 / 29.59 (law 7.63 / 15.25 / 22.88 / 30.50) |
| Budget to 2.7B / to 7B (assumed prices) | $28,476 / $170,551 |
