# SUN PLAN B — DAEDALUS, THE DISCOVERY ENGINE

**Let AI make the maths. Make the verifier the moat.**

In 2026, machines find real mathematics. From Wikipedia's list of AI discoveries (fetched 27 Sep 2026):
- **Lean-formalized:**
  - Sendov's conjecture (examined by Terence Tao);
  - an Erdős–Rankin improvement on prime gaps (Ben Green confirmed the approach);
  - the dying-percolation conjecture;
  - the Kozma–Nitzan conjectures;
  - a disproof of the Köthe conjecture (Lean 4 + Mathlib, verified by Epoch AI);
  - a **13-million-line Lean proof of Fermat's Last Theorem.**
- **Many others are listed as not independently verified.** The list says so itself.

Two engines, both fetched:
- **AlphaEvolve** (Google, June 2025) found a way to multiply 4×4 complex matrices with 48 scalar multiplications. That is the first improvement over Strassen in 56 years in that setting.
- **The Darwin Gödel Machine** (Zhang, Hu, Lu, Lange, Clune) raised SWE-bench from 20.0% to 50.0% by evolving its own code.

**The pattern:** proposals are cheap, and verification is scarce. Our record is a dataset of how AI research agents fail:
- 127 claims struck across three audit rounds (18 + 13 + 96);
- bars written after their results;
- a hard-coded dictionary passed off as fetched citations;
- beds with window shortcuts;
- a twin missing its relative position (L-REFLECTOR);
- arms that never learned the task.

A 2026 position paper names the same disease: AI scientists defending general claims that their own data contradicts (arXiv 2606.23175, found).

**Daedalus is an evolutionary search over operators and objectives whose fitness function is our verification stack.** The search cannot win by any failure mode we have already caught. The gene pool is Plan A's shelved canon plus everything that survived CEQ. Daedalus built his wings from what was inside the prison.

Author: Seal (Teerth Sharma). Date: 2026-09-27.

---

## 0. Laws and safety

- **All earlier laws apply, and the engine enforces them mechanically:** L-EQ, L-REPRO, L-REFLECTOR, L-SHA, L-AUDIT, L-BR, L-SHORTCUT, L-LEARN, L-TRAINED, L-LOSS, L-SCALE, L-OWNER, L-SUN.
- **L-FIXED-BAR:** each bed's bar is registered *before* the engine sees the bed. Candidates can't write bars, so post-hoc bars become impossible by construction.
- **L-SEALED-TRUTH:** candidates run in a sandbox with no read access to truth tables, evaluator code, or held-out beds. Any output hash matching a truth file is an automatic reject.
- **Safety** (the Darwin Gödel Machine practice, plus ours):
  - sandboxed execution;
  - human sign-off on every promotion;
  - the engine cannot modify the verifier;
  - network restricted to the fetch allowlist;
  - uploads only on the author's word;
  - keys stay local;
  - no GPU process is killed by name.

---

## 1. Owners (the engine is not new; its verifier is the claim)

- **Fetched:** AlphaEvolve; the Darwin Gödel Machine; the 2026 list of AI mathematical discoveries; the AI-scientist failure paper (2606.23175).
- **[U]:** FunSearch (2023), The AI Scientist (Sakana), the neural architecture search literature.
- **The distinct cell claimed:** a verifier built from a documented, audited record of research-agent failures, used as the fitness function for architecture discovery, with exact-truth beds and Lean-checked invariants. The claim is checked against the owners before any sentence (Wilson's novelty stage).

---

## 2. The machine

**Proposer.** One or more LLMs propose program diffs to an operator library. The library covers mixing, predictor, objective, optimizer, and quantizer slots. It is seeded with the gene pool:
- quaternion and Clifford products (Hamilton, Grassmann);
- hyperbolic distances (Lorentz model);
- Koopman and Volterra kernels;
- IFS contractions (fractal);
- ternary quantizers (Setun → BitNet);
- GMDH growth;
- reservoir dynamics;
- the CEQ switches (β / g / qk), the causal-resolvent read, the SU(2) fold, and the D1 pole check.

**Verifier, the moat.** Stages run in order, and any stage can reject:
- **V0 static:** it compiles, shapes are right, there is no leakage (L-SEALED-TRUTH), and no output matches a truth-table hash.
- **V1 exactness:** Lean-stated invariants for any operator that claims one (causal mask, row sums, similarity, pole-freedom via D1), plus property tests generated from those specs.
- **V2 floors and shortcuts:**
  - Automatic floors: position-only, windowed, commutative ceiling, random encoder, raw pixels.
  - A library of the hand-built shortcuts the rooms have found so far (Phase K's window constructions, pointer doubling), run against every bed.
- **V3 trained best responses:** ALiBi twin, Gated DeltaNet, DeltaProduct, PaTH (flash-linear-attention kernels) at matched parameters and FLOPs, trained under the same budget.
- **V4 learnability and seeds:** ≥ 0.9 in-distribution, then ≥ 3 seeds before any comparative number.
- **V5 scale:** promotion ladder R0 → R1 → R2. The margin must survive each rung (L-SCALE).
- **V6 audit:** an Inspector agent plus human review of every promoted candidate.

**Beds (exact truth; cents per evaluation at R0):**
- A5-World (Phase L);
- pointer chase with the far band (`wald.py`, as corrected by K1-a);
- Lorenz and other chaotic systems, with calibration past T* (Plan A's S2);
- consequence trees (Plan A's S1);
- language modelling at R0 on the 2.99B FineWeb-Edu shards.

**Five held-out beds** are never shown to the engine and are used only at V5–V6.

**Fitness:** a Pareto front, not one scalar. The axes are:
- margin over the trained best response;
- cost (× SDPA);
- unique parameters;
- calibration reliability (REL).

A single scalar is what gets gamed (Goodhart).

---

## 3. Milestones

- **M0 (week 2): the engine must rediscover our known truths before it is trusted.** Five known verdicts must reproduce automatically from the record:
  1. ALiBi beats the CEQ family;
  2. FoX recovers C_win;
  3. the corrected Q2 pair closes at 120 quaternions / 60 rotations;
  4. the X3 reset bed is solved by recency;
  5. the Phase K beds fall to window shortcuts.
  - **Also at M0, House plants 20 cheating candidates:** leaked truth, hard-coded outputs, post-hoc bars, shortcut exploits. The verifier must reject **20 of 20**.
- **M1 (week 6):** at least one candidate passes V0–V4 on at least 2 beds, with the pre-registered margin over trained best responses.
- **M2 (week 10):** that candidate keeps its margin at R2 (51M), passes V6, and has its claimed invariant proved in Lean.
- **M3 (weeks 12–16):**
  - a 1.3B confirmation (about $360 per run at the assumed prices; see `sun_budget.py`);
  - RULER multi-hop and BABILong against Transformer++ and Gated DeltaNet at matched tokens [U fetch first].
- **Sun waypoint:** an operator that no human on the team designed, verified exact, beating trained best responses at 1.3B on a named task, with its full failure record public.

---

## 4. Budget (all prices are assumptions to be replaced with quotes)

- **Search:** 20,000 candidates × about 2 GPU-minutes on R0 beds ≈ 670 H100-hours ≈ $1.7k at $2.50 per hour.
- **Proposals:** 20,000 × about 8k tokens ≈ 160M tokens. That is $0.5k–2.4k at an assumed $3–15 per million.
- **Promotions:** about 1% go to R1, and a handful to R2 and 1.3B (see Plan A's table: the 1.3B five-arm grid costs about $5.4k).
- **So Daedalus fits comfortably inside Tier 1 (≤ $125k).** The expensive input is people. Tier 2 buys a larger search, more held-out beds, and the 7B confirmation.

---

## 5. Kills

- **M0 fails** (fewer than 4 of 5 known verdicts reproduced, or any planted cheat passes) → the engine is not trusted. Fix the engine; do not publish a single candidate.
- **M1: no candidate in 6 weeks** → report whether the gene pool or the verifier was the wall. That report is itself a result.
- **M2: the margin vanishes at R2** → L-SCALE strike; the candidate is filed as small-scale only.
- **Novelty check finds an owner** → the candidate is published as an independent rediscovery, with no novelty sentence.

---

## 6. Room and timeline (16 weeks)

- **Foreman:** engine infrastructure, sandbox, ladder.
- **Chase:** V1 (Lean specs and property tests) and V2 (the shortcut library).
- **Cameron:** beds and V3 opponents.
- **Wilson:** owners library, the novelty stage, stored-artifact citations.
- **Inspector:** V6.
- **House:** red-teams the verifier, plants the M0 cheats, and runs a monthly attack.
- **Human:** signs every promotion.
- **Schedule:**
  - Weeks 1–2: M0.
  - Weeks 3–6: search to M1.
  - Weeks 7–10: M2.
  - Weeks 11–16: M3, the Zenodo record, and the paper.
- **Two papers either way:**
  - the engine and its verifier, whose value holds even if no candidate lands;
  - the candidate, if one does.

---

## 7. Sources used (fetched 27 Sep 2026)

AlphaEvolve (arXiv 2506.13131); Darwin Gödel Machine (arXiv 2505.22954); the Wikipedia list of mathematical discoveries by AI; the position paper on AI scientists (arXiv 2606.23175). Plan A's sources supply the gene pool.
