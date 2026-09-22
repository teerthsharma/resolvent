# PHASE K — PROPAGATOR (MASS IS A POLE, AND WE GO BIG)

[Dispatcher note, 2026-09-23 01:36: the contract says its instance scripts `wald.py` and `yukawa.py` are "shipped with this file". Neither exists on this machine (searched C:/Users/seal to depth 4, including Downloads, Desktop and Documents). Until the author's files arrive, every number in §3 and §9 is the author's claim; any instance run before then is a rebuild from this text and is labelled as such, not as the pinned instance.]

**Phase J ended with a clean negative result. The family's language-model win was relative position (ALiBi: 108.1% of it, at about a fifth of the clock), and every capability claim died against a proper control. Phase K changes two things.**

**First, the axis.** Physics says a wave's mass is the pole of its propagator, and that pole sets how far its influence reaches (Yukawa). Attention's resolvent is a propagator. Relative-position biases give it mass, meaning short range, which is what language modelling rewards. Summing all hops takes that mass away, meaning long range, which is what following a chain of causes needs, and a bounded-depth transformer can't do it. Phase K tests the family only where its structure is the task (chains of consequence), with the resolvent trained in the loop.

**Second, the scale.** No headline from a single 725k-parameter point. Every headline is a curve over at least three rungs (7M → 24M → 51M, with one 124M run), plus a public benchmark and a Zenodo-grade record.

Author: Seal (Teerth Sharma). Contract date: 2026-09-23. Supersedes the Phase J plan. Its records stay read-only (L-AUDIT). Instance scripts shipped with this file: `wald.py`, `yukawa.py`. **Foreman copies them into `tests/foreman/phase_k/instances/` and pins their SHA before it.K0.** Phase J's contract text cited scripts the repo did not have. That does not happen twice.

---

## 0. Laws

- **Unchanged:** L-EQ, L-REPRO, L-REFLECTOR, L-WIENER, L-SHA, L-AUDIT, the floor rule, the security lines, and the no-mechanism-before-contrasts rule.
- **New (from the Phase J pattern: I tested on the planes that came back):**
  - **L-BR:** no prediction is written until the opponent's best response has run on the author's own instance.
  - **L-TRAINED:** a theorem describes the family only after its premises (β, exact zeros, readout) are checked on the trained checkpoint.
  - **L-LOSS:** a bed counts only if the operator's structure is inside the training loss and the target needs it.
  - **L-SCALE:** no headline from one rung. A claim is a curve over ≥ 3 rungs with CIs, or it is a note.
- **Security:**
  - Zenodo and Hugging Face uploads happen **only on the author's explicit say-so**.
  - The Kaggle token stays local.
  - No lane kills a GPU process by name.
  - Deciding numbers come from the certified local 4060; Kaggle reproduces.

---

## 1. Pointers resolved (fetched unless marked [U])

- **"CERN's golden equation":** the Standard Model Lagrangian. Phase K uses only three of its ideas, each with an instance:
  - **Mass is the pole of the propagator and the decay rate of its influence.** For a massive lattice wave, (−Δ + m²)G = δ decays at arccosh(1 + m²/2). *Y1: measured = predicted to 6 digits at m = 0.1, 0.5, 1.0.*
  - **Range ≈ 1/mass (Yukawa, February 1935, fetched).** A massless mediator gives the Coulomb 1/r and infinite range. The Yukawa form is the Klein–Gordon propagator.
  - **Summing all hops dresses the mass (Dyson resummation [U]).** See §2 and Y2.
  - The Higgs mechanism (mass acquired from coupling to a field [U]) is a **page line and an analogy only**: FoX's content-dependent forget gate acquires its "mass" from the content; ALiBi's is bare. No row rests on it.
- **"How does mass propagate, and what is binding?"** A massive wave's influence dies as e^{−m·r}. The mass *is* that rate.
  - Binding is a pole where the wave stops leaking: a bound state. In our operator, that is a closed class, i.e. a unit self-weight (D1) with its projector (C10).
- **Werner Heisenberg** enters **only** through his S-matrix (1943 [U]) and the uncertainty relation.
  - Poles of the scattering matrix are the bound states and resonances. Range × mass ≈ 1 is the uncertainty relation's reading of Yukawa.
  - The Lippmann–Schwinger/Born-series instance already on file is his object in our coordinates. Nothing else from his biography enters the contract.
- **The young European probabilist: Hugo Duminil-Copin** (fetched). French, born 1985, IHES permanent professor since 2016, University of Geneva before that. Fields Medal 2022 "for solving longstanding problems in the probabilistic theory of phase transitions in statistical physics, especially in dimensions three and four."
  - With Raoufi and Tassion (arXiv 1705.03104, fetched) he proved sharp phase transitions: exponential decay of correlations below the critical point, mean-field behaviour above it. The proof uses an inequality on decision trees generalising the OSSS inequality from randomized algorithms. That is the "computer probability style."
  - If the pointer meant Martin Hairer instead, the name changes and the rows do not.
- **Topological data analysis: prior art conceded.** Kushnareva et al., EMNLP 2021 (fetched), compute topological features of attention maps (for artificial-text detection). TDA on attention is not ours. Only its use as a phase instrument on a trained resolvent (R-PERC) is.

---

## 2. The object: the attention propagator

- **Propagator:** G = (I − γW)⁻¹ for a causal head W at β = 1 (L-TRAINED: β pinned; Phase J's β = 1.061 / 0.930 / 0.900 made the read run past a pole).
- **Poles:** at γ* = 1/W_ii (D1).
- **Bound states:** heads with W_ii = 1.
- **Mass of a head:** m = −lim (1/r) ln |G(i, i−r)|. **Range:** 1/m.
- **Mass renormalization (Y2, exact).** A causal head with bare mass λ (kernel ∝ e^{−λr}, the ALiBi shape) has a resolvent with dressed mass **m = ln(γ + (1−γ)e^λ)**, which is massless at γ = 1.
  - *Measured = predicted to 5 digits at λ ∈ {0.5, 1.0}, γ ∈ {0.9, 0.99, 0.999}.*
  - The range grows from 2 tokens (bare) to **1,542 tokens** at γ = 0.999.
  - So one resolvent read of a local head is a long-range channel. The Phase J win came from giving attention mass. Phase K's axis needs the massless channel.
- **Phases (instrument language, borrowed from Duminil-Copin, not his theorem):**
  - Subcritical (massive): connection probability decays exponentially. This is ALiBi.
  - Critical (massless along bindings): the range covers the chain depth.
  - Supercritical (everything flows to one absorber): the softmax sink of D4, where everything collapses into BOS.

---

## 3. Best responses already run (L-BR)

*`wald.py`, seed 23.* Bed: pointer chasing, n = 4096, 16 interleaved chains, depth up to 269. Each token's content is only its own id and its parent's id.

| Arm | Accuracy |
|---|---|
| Position/recency floor (the ALiBi prior) | 0.0645 (chance 0.0625) |
| L = 4 / 6 / 8 transformer, *perfect* pointer doubling (its best known response) | ceilings 0.0664 / **0.2539** / 0.9856 |
| One resolvent layer, softmax logit scale s = 8 / 12 / 16 / 20 | 0.0869 / 0.9954 / **1.0000** / 1.0000 |
| One resolvent layer, sparsemax | exact by construction |

*`yukawa.py` Y3 (author's law struck before running).* The author predicted s_50 = ln(n · D_median) + c. Measured s_50 = 8.92 / 10.73 / 13.00 at n = 1k / 4k / 16k. The offset drifted from −1.48 to −2.44 to −2.94, so the law fails. The transition **broadens** with n (width 1.81 → 2.61 → 3.82): no sharp threshold appears in the toy.

What survives: the softmax resolvent's logit scale must grow with ln n (slope about 1.3–1.6 per ln n in this range). So the resolvent arm adopts SSMax-style scaling s = a·ln n + b with a learned a, or uses sparsemax.

**Missing planes, stated before the rows.**
- The weights above are hand-set.
- The 2^L ceiling is the best known opponent strategy; its matching lower bound is unverified (Sanford–Hsu–Telgarsky 2024, fetched: the abstract states only that log depth is sufficient).
- Owners are still unread: Diffuser's multi-hop attention diffusion (fetched; whether it truncates or is causal is unread), looped transformers [U], DEQ [U], and PPR propagation in graph nets (APPNP) [U]. Each is read before any novelty line, and each is either an arm or a stated difference.

---

## 4. The scale ladder

*`yukawa.py` Y4.* Kaplan approximation C ≈ (6N + 6·L·n_ctx·d)·D. Tokens = 20 per parameter. Throughput = the 26.77 TF measured bf16 peak × **assumed** MFU 0.35. Rung 0 measures the real MFU and replaces the assumption.

| Rung | Layers × width | Parameters | Tokens | Hours per run | Weights + AdamW |
|---|---|---|---|---|---|
| R0 | 4 × 128 | 7.2M | 0.14B | 0.2 | 0.11 GiB |
| R1 | 6 × 320 | 23.5M | 0.47B | 2.6 | 0.35 GiB |
| R2 | 8 × 512 | 50.9M | 1.02B | 12.3 | 0.76 GiB |
| R3 | 12 × 768 | 123.6M | 2.47B | 70.9 | 1.84 GiB |

- Parameter counts include a 50,304-entry embedding, which dominates R0.
- **Plan:** R0–R2 for every arm, 2 seeds. R3 once for the best resolvent arm and once for the ALiBi twin. Synthetic chain beds are cheap at every rung. Language-model runs are the budget.
- **Data:** the current 18.17M-token corpus cannot feed R1+ without heavy repetition. Row DATA fetches a ≥ 2.5B-token educational web subset (e.g. FineWeb-Edu [U]), stored as uint16 shards (about 5 GB) with a hash manifest.
- **Kaggle** reproduces R0–R1. R2–R3 are local only.

---

## 5. Rows

Every row carries: prediction, author counter, the opponent's best response (L-BR), the premise check (L-TRAINED), the in-loss check (L-LOSS), measured value, CI, verdict, SHA.

### R-DEPTH (headline)
- Train the chain bed with the resolvent in the loop (β = 1, fixed γ, logit scale ∝ ln n or sparsemax).
- Train at depth ≤ 32 (n = 1024). Test at n ∈ {4k, 8k, 16k}, depth up to about 1,000.
- Arms:
  - (f_R): L−1 standard ALiBi layers plus 1 resolvent layer.
  - (a_L): the ALiBi twin at L and at L+3 layers (parameter-matched).
  - (a_loop): a looped ALiBi twin, FLOP-matched to (f_R).
  - (a_ss): the twin with SSMax.
  - Diffuser, if runnable.
- **Prediction:** (f_R) ≥ 0.95 beyond depth 2^L at every rung; (a_L) stays under its doubling ceiling (0.254 at L = 6, depth to 269).
- **Counter:** (a_L) > 0.5 beyond depth 2^L, meaning a shortcut exists that the author did not find.

### R-SCALE (the Zenodo headline)
- Two curves over R0–R2 (R3 at one point):
  - Accuracy vs depth: the twin's critical depth D_c(L) against the resolvent's flat line.
  - Language-model loss vs compute for (f_R) and (a_L): power-law fits with CIs.
- **Prediction:** D_c(L) tracks 2^L (fit exponent within 0.8–1.2 of doubling), and (f_R) shows no critical depth up to the longest context.
- **Counter:** the twin's D_c grows faster than 2^L with rung. The edge is then a small-model artifact.

### R-CARRY (condition 3)
- At R0–R2: (f_R) vs (a_L) language-model loss at matched parameters, wall clock priced.
- **Prediction:** |Δ| ≤ 0.01 nats at every rung.
- **Counter:** (f_R) worse by > 0.02 at R2. The resolvent layer then does not carry the weight.

### R-BENCH (named task)
- RULER multi-hop tracing (variable tracking; RULER fetched: 13 tasks, "multi-hop tracing" is a category) at 4k / 8k / 16k, and BABILong multi-hop QA [U, fetch first]. R2 and R3.
- Opponent: the ALiBi twin at the same rung.
- **Prediction:** (f_R) − (a_L) ≥ +0.15 at 16k on variable tracking.
- **Counter:** < +0.05, in which case the synthetic edge doesn't transfer.

### R-RANGE (mass of every trained head)
- Fit m for every head from the decay of its resolvent kernel. The instrument must reproduce Y2 on a synthetic Toeplitz head to 1e−4 before use.
- **Prediction:** at least one head in (f_R) has range ≥ the bed's depth; (a_L) heads have range ≈ 1/slope.
- **Counter:** no long-range head in (f_R). The depth edge then isn't coming from the resolvent's range.

### R-BIND (Heisenberg's poles)
- On the trained (f_R), count the unit and near-unit self-weights (D1, O(n)) per head. Compare them with the planted roots.
- **Prediction:** the absorbers match the roots (Jaccard ≥ 0.8) in the resolvent layer.
- **Counter:** ≤ 0.3. Binding then is not what the layer learned.

### R-PERC (Duminil-Copin's instrument)
- Threshold each trained head's attention graph at t. Sweep t and record the H0 persistence (union-find) and the connection probability vs distance.
- Null: row-shuffled attention.
- **Prediction:** below a critical t_c, exponential decay; above it, a component spanning each chain's bindings.
- **Counter:** no t_c distinguishable from the null. The phase language then leaves the page.

### R-COST
- Resolvent layer forward+backward vs one SDPA pass at 4k / 8k / 16k (1.76× measured at 4k in Phase J).
- **Prediction:** ≤ 2.0× at 16k.
- **Counter:** > 2.5×. The cost leg then fails.

### DATA
- Fetch, hash and shard the corpus.
- No language-model rung starts until the manifest exists.

---

## 6. Kills

- **R-DEPTH:**
  - The counter holds → the bed is void, and the author owes the shortcut.
  - (f_R) < 0.8 at the *training* depth → a learnability kill.
- **R-SCALE:** the counter holds → the depth edge is filed as small-scale only.
- **R-CARRY:** worse by > 0.02 at R2 → condition 3 is not met.
- **R-BENCH:** < +0.05 → no public-benchmark claim.
- **R-RANGE / R-BIND / R-PERC:** each counter strikes its own sentence, and nothing else.
- Any instance (Y1, Y2, wald.py) fails to reproduce at 1e−8 (or at the stated digits) → this contract is void until explained.

---

## 7. Room, fetch orders, iteration

- **Foreman:** DATA; the ladder runs; copying the instance scripts and pinning their SHA (first task).
- **Chase:** the resolvent layer in training (β = 1, the logit-scale law, gradients by adjoint triangular solve); R-RANGE; R-BIND.
- **Cameron:** R-DEPTH arms and beds; R-BENCH; R-COST at 16k.
- **Wilson:** L-BR audit on every row before its bar; R-PERC; the Zenodo manifest; scoring.
- **Inspector:** one pass per iteration. Post-hoc bars are struck on sight.
- **House:** optional adversarial pass on R-DEPTH, to find the shortcut before the run does.
- **Fetch before any table:**
  - Fetched: Duminil-Copin, Raoufi–Tassion 1705.03104, Yukawa, RULER, Kushnareva 2021, Sanford–Hsu–Telgarsky 2024, Diffuser.
  - Still [U]: BABILong, looped transformers, DEQ, APPNP, Heisenberg S-matrix 1943, Dyson 1949, Higgs 1964, FineWeb-Edu, Kaplan 2020, Chinchilla 2022.
- **Iterations:**
  - it.K0: scripts in the repo, SHA pinned; instances re-run; DATA.
  - it.K1: R0 for every arm; the real MFU replaces the assumption; R-DEPTH at R0.
  - it.K2: R1; R-RANGE, R-BIND, R-PERC on R0/R1 checkpoints.
  - it.K3: R2; R-CARRY; R-BENCH at R2.
  - it.K4: R3 (two runs); R-SCALE curves; R-COST at 16k.
  - it.K5: tables and the Zenodo package, held for the author's word.

---

## 8. The Zenodo record (what "heavy" means)

- Code at a pinned SHA, with configs and seeds for every run.
- The raw logs; every RECORD file with its audit files, including Phase J's strikes (L1: 18, N1: 13, N2).
- The instance scripts; checkpoints for R0–R2 (all arms) and R3 (both runs); evaluation outputs.
- The data manifest with hashes.
- A negative-results section (Phase J's relative-position finding) that stands on its own.
- License and DOI. **Uploaded only on the author's explicit word.**

---

## 9. Reference instances (L-REPRO)

`yukawa.py`: seed 31, float64. `wald.py`: seed 23, float64.

| id | quantity | value |
|---|---|---|
| Y1 | lattice Klein–Gordon decay vs arccosh(1+m²/2), m = .1 / .5 / 1 | .099958 / .494933 / .962424 (both columns) |
| Y2 | dressed mass, λ = .5, γ = .9 / .99 / .999 | .06285 / .00647 / .00065 (= formula); range 16 / 155 / 1,542 tokens |
| Y2 | dressed mass, λ = 1, γ = .9 / .99 / .999 | .15857 / .01704 / .00172 (= formula); range 6 / 59 / 582 tokens |
| Y3 | s_50 at n = 1k / 4k / 16k; width | 8.92 / 10.73 / 13.00; 1.81 / 2.61 / 3.82 (author's law struck) |
| Y4 | hours per run R0–R3 at the assumed MFU 0.35 | 0.2 / 2.6 / 12.3 / 70.9 |
| W | floor / L = 6 ceiling / resolvent s = 16 | 0.0645 / 0.2539 / 1.0000 |
