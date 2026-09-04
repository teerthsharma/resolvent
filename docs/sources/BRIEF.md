# BRIEF — the shape paper, shared context for every planet

Read all of this before doing anything. Every path below is absolute. The repository is
`C:\Users\seal\Desktop\New folder (32)` (branch `v17k-gate0`, HEAD `207e7b9`). Your
scratch directory is
`C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\12493ed6-8e0b-4168-994a-075dc6a1800a\scratchpad`
(call it `$SCRATCH` below). Write your output where your prompt tells you, and nowhere else.

## 0. THE MISSION, IN THE AUTHOR'S OWN WORDS (2026-09-03)

Message 1:

> i want you to look at the repo the north star aka the main desgin is to create a new
> generation of attention mechanism go deep see all maths i feel the repo is failing not
> dueto less engineering but becuase of smarts thats why i have chosen your model I want
> you to rethink the shape use what we proved just make a huge plan a better paper then
> whatever here one research paper /research /topology find skills you require to do it
> expected output is inside docs where i will find a research paper in pdf and tex and md
> where it will decribe finally for once whats our main shape with all citations

Message 2 (the north star, refined):

> the main attention mechanism i want is causality for ai to learn conseqeunces for it to
> predict safest move under multiple constraint something self attention is bad at and
> anyt other benefit comming from next transient state phase equillbirum predcitor based
> attention which well anything near it

Message 3:

> do not write any code onlyu docs

The standing north star in the repository (`CEQ_V20_R15_CONTRACT.md:42-45`, immutable
across rounds) reads: *attention EQUAL to self-attention on its own ground, built FROM
softmax and AdamW, capable on ground they cannot occupy — predicting the NEXT STATE toward
equilibrium, not the next token.* The author's message 2 sharpens "next state toward
equilibrium" into three concrete capabilities: **(a) consequences of an intervention**,
**(b) the safest move under several constraints at once**, **(c) the next transient state /
phase / equilibrium**. The paper must be built around those three.

The author licensed a new primitive on 2026-09-03 ("rethink the shape"). The round-15
contract's rule *no new primitive that is not a repair* is therefore superseded by the
author for this document. Say so where it matters.

## 1. THE THESIS — what the shape is

**The record's verdict on itself (`MATHEMATICS.md:16-45`):** every bed in the repository
asks for a *scalar* label at one position (`equilibrium_oracle` returns `z*_{s-1}`,
`scale/negation_scope.py:286-304`), which is the single-location regression task on which
one attention layer is (asymptotically, at `L = o(d)`) Bayes-optimal
(Marion–Berthier–Biau–Boyer, arXiv:2410.01537; caveats at `MATHEMATICS.md` §17.5). The
campaign therefore raced softmax at its own proven optimum — `MISTAKES.md` D-1, the
largest mechanism in the taxonomy. The capability the author wants is not a scalar at one
position. It is a *jointly determined configuration* (the fixed point the context settles
into), its *displacement under an intervention*, and a *decision* over candidate moves
scored against several constraint sets. None of those is a per-row mixture.

**The shape (working definition — the design panel refines it, but this is the spine):**

```
  P      = the causal, row-stochastic operator of the record's three-corner family
           (β = 1 corner: softmax; ceq/arm_smprime.py, lean/CEQ/V16Domain.lean)
  γ      ∈ [0, 1)   a learnable switch
  𝒜_k    absorbing (constraint) position sets, k = 1..K; rows of P on 𝒜_k are identity
  z      = (I − γP)^{-1} V                     the jointly determined state ("the shape")
  O      = P z  =  P (I − γP)^{-1} V           the read
  q^{(k)} = (I − Q)^{-1} R_k 1                  the committor / absorption probability
                                               into constraint set k (Q = transient block)
  do(a)  : intervene on the context (a candidate move / a token), re-solve; the
           displacement Δz = z(do a) − z is the CONSEQUENCE
  safest move = argmin_a  max_k q^{(k)}(do a)  (or the lexicographic / weighted form)
```

**Five identities, four of them RUN by the coordinator on 2026-09-03 against the repo's
own code (`$SCRATCH/shape_identities.py`, float64, torch 2.5.1, this box):**

| id | statement | class | number |
|---|---|---|---|
| I1 | at `γ = 0`, `P (I − γP)^{-1} V` is **bitwise** `P V` (softmax parity by construction, `torch.equal`) | RUN | `True`; rejection region at `γ = 0.5`: max abs `2.3003` |
| I2 | the record's **corner 3** — the path product `G_ij = Π_{k=j+1}^{i} a_k` (`ceq/arm_smprime.py:144 path_product`) — **is** the resolvent `(I − A)^{-1}` of the sub-diagonal chain, entrywise | RUN | max abs `0.0` (exact); last row vs `equilibrium_oracle`: `6.2e-15`; `A^s = 0` exactly |
| I3 | the committor of an absorbing chain is the resolvent read with absorbing rows, `q = (I − Q)^{-1} R 1_B` | READ + prior RUN | `ceq/beds/bed_1.py:188-198 committor` solves exactly that Dirichlet problem; `harmonic_residual` reads `0.000000e+00` (`workdonenewseal.md` §5.2); `MATHEMATICS.md` §7, §11 (Kirchhoff cross-check to `1e-10`) |
| I4 | Neumann truncation certificate for row-stochastic `P`, `γ < 1`: `‖(I − γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞ ≤ γ^{K+1}/(1 − γ)` — **no sum over `s` anywhere** | RUN | holds at K = 1, 2, 4, 8, 16 (err = bound to `1e-15`); **planted negative**: a non-stochastic `P` (rows sum 1.5) reads err `119.37` against bound `1.143` — the certificate can fail, so it is a certificate |
| I5 | the causal resolvent is a **triangular solve** (`solve_triangular` vs dense inverse) | RUN | `1.8e-15`; forward substitution ≈ `s²d` multiply-adds against `2s²d` for `QKᵀ` |

I4 is the resolution of `IMPOSSIBLE.md` I1 (M6 vs M2): the record could not find a
denominator-free magnitude certificate for a *signed* multi-hop operator because the
absence of the normalizer was the whole mechanism of its sign capability. With a
row-stochastic `P` the normalizer pays for the certificate, and the sign capability is
replaced by **boundary conditions** (absorbing constraint sets), which is where "a third
token can veto" now lives. State this as a proposition with the two halves.

**Why the shape is "built from softmax":** at `γ = 0` and `β = 1` it is softmax bitwise
(I1, and Lean `three_corners_containment`). Everything the record proved about the
three-corner family transfers unchanged. Corner 3 of the record was a resolvent all along
(I2); the shape makes `γ` and `P` general and adds boundary conditions.

**Why softmax cannot occupy the ground (each must be cited, with exactly what the theorem
licenses and no more):**

1. *Per-row independence.* One softmax step is an independent convex mixture per query;
   the value at position `i` never constrains the value at `j`. A fixed point is not that
   (`MATHEMATICS.md` §0.3). The resolvent's `z` is jointly determined.
2. *Depth law.* `k`-hop composition needs depth `⌊log₂ k⌋ + 2` (Sanford–Hsu–Telgarsky
   ICML 2024, arXiv:2402.09268, Thm 4.2; the `Ω(log k)` lower bound is **conditional on
   the 1-vs-2-cycle conjecture**, Cor. 4.3). Graph connectivity / reachability needs
   logarithmic depth (Sanford et al., NeurIPS 2024, "Understanding Transformer Reasoning
   Capabilities via Graph Algorithms"). The resolvent computes all `s` hops in one
   operator (nilpotency: `lean/CEQ/Nilpotent.lean`, `Occupancy.lean`).
3. *Function composition.* A single attention layer cannot compose two functions when the
   domain is large (Peng–Narayanan–Papadimitriou 2024, arXiv:2402.08164 — VERIFY the
   identifier and statement; [U] until fetched). A consequence *is* a composition.
4. *Non-negativity.* A softmax mixture cannot let a third token reduce another token's
   contribution (`ceq/attention.py` header: influence Jacobian min entry exactly `0` in
   `0/200` draws vs `107/200` signed). The shape does **not** revive signed operators —
   that programme died (`RESEARCH.md`, `PROGNOSIS.md`) — it moves the veto into absorbing
   boundary conditions.

**The honest control the paper must carry:** a *deeper* softmax stack (depth
`⌊log₂ t*⌋ + 2`) can compute the same hops. So the claim against softmax is at **matched
depth and parameters**, and the deeper stack is the *skyline*; the shape's separate
advantages are exactness (a certificate with a printed δ), one-read joint consistency,
and the boundary-condition mechanism. "Beats softmax" sentences are banned on beds where
softmax is a fellow approximator (`CEQ_V16_CONTRACT.md`, R-SKY).

## 2. WHAT THE RECORD PROVED (machine-checked; quote by declaration name and file:line)

`lean/CEQ/` — 12 files, 169 `theorem`/`lemma` declarations (`grep -cE '^(theorem|lemma) '`,
2026-09-03), `lake build` exit 0, zero `sorry`, axioms `[propext, Classical.choice,
Quot.sound]` only (`README.md` §2). The load-bearing ones:

- `Nilpotent.pow_card_eq_zero`, `Occupancy.occupancy_is_exact_inverse`: strictly-causal
  `A` is nilpotent; the finite path sum is exactly `(I − A)^{-1}`, no sign or magnitude
  hypothesis.
- `V15Fork.Asink_row_sum / Asink_nonneg / Asink_computes_chain /
  no_row_stochastic_with_drive_values`: one softmax head with a key-side scan bias and a
  value-zero BOS sink reproduces the chain label exactly (the telescope); the normalizer
  is a change of units, not an obstruction (`workdonenew.md` §1).
- `V16Domain.three_corners_containment`, `corners_are_distinct`,
  `gate_zero_beta_zero_is_linear_attention`: softmax, linear attention and the path
  product are three settings of one family; the corners are distinct.
- `V16Domain.pathProd_eq_zero_iff`, `no_prefix_scan_represents_a_zero_gate`,
  `prefix_logit_mask_restated`, `bedM_gate_exact`: a zero gate annihilates the path
  product exactly and no exponential prefix scan can carry it — the exact sparsity /
  segmentation certificate (F0).
- `V15Kernel.first_order_cannot_delay`, `delay_forces_state_injective`,
  `linear_first_order_cannot_delay_beyond_state_dim`, `first_order_cannot_powerlaw`: a
  first-order recurrence cannot delay; delay `d` needs state dimension `> d`; the
  power-law kernel is not geometric — BED-K is scan-blind by theorem.
- `V15Source.source_is_first_order_difference`, `two_sources_recovered`: source recovery
  through `(I − A)` is `O(nnz)`.
- `V15Phase.*`, `V16Domain.six_misses_every_bedM_value`: closed-magnitude phase gates;
  the domain census that condemned #6 as decoration on BED-M.
- `V15.gate_zero_row_sum / gate_zero_not_stochastic`: the refutation of the original
  parity clause (row sums `i + 1`).
- `OracleSeparation.oracle_ne_resolvent`, `truncation_never_exact`: the absorbing-chain
  oracle is never nilpotent; the arm's operator is; every rung leaves a residual.

## 3. WHAT THE RECORD REFUTED (the paper carries these as first-class negatives)

- The signed strictly-causal path sum (`sgate`, pivot routing): sign-flip capability
  decays ≈ `1/s` (slope `−1.298` vs bar `−0.3`), pivots make it worse, COGS `0/512` vs
  softmax `15/512` (`RESEARCH.md`, `PROGNOSIS.md`, `MODEL_CARD.md`). Dead.
- Settling the *mixture weights* over pivots: a reparameterisation of the twin's family,
  `settled − twin = −0.002959`, CI covers zero (`MATHEMATICS.md` §5).
- The hop-2 term on the softmax operator carries nothing: five independent routes
  (`workdonenew.md` §6, `V13_PREDICTION_HOP2.md`, `ARCHITECTURE.md` §2).
- VGPE (path-ordered non-abelian transport as PE): occupied by PaTH (arXiv:2505.16381
  §2.1); Hankel rank is `d` regardless of alphabet (`THEORY_V12_VGPE.md`, `PRIOR_ART.md` §5).
- The parity clause "g ≡ 0 gives bitwise standard attention": false three ways; repaired
  by the telescope (`workdonenew.md` §1).
- Softmax's normalizer as the obstruction to path products: false (`γ_j = 1/(1−a_j)`).
- X₃₅ hidden-cause inference: Basseville & Nikiforov 1993 §7.2.4 in closed form.
- Closed-magnitude gates: S4D's ReLU variant, `|Ā| = 1` on 32.93% of a sample.
- The deciding measurement R1 (BED-M, `t* = 2`, N = 8): 5/8 crossed `floor₁`, 3 diverged
  (`â_max` to 285), CI `[0.617, 1.041]` straddles `0.7071`. Not earned.
- `floor₁ = √((t*−1)/t*)` is a one-hop capability *threshold*, not an information floor;
  13 of 40 cells violate it (`V20_R15_THEORY_TABLE.md` §0.2).
- The arms return one real per draw; no state axis exists for a state metric (Q6 F4).
- `MISTAKES.md`: 65 mechanisms in four classes (V vacuous controls, P provenance, M
  measurement, D design). The paper's plan must name, for each bind/gate/bed it proposes,
  which mechanism it is designed against.

## 4. THE REPOSITORY MAP (read these; do not guess their content)

| what | where |
|---|---|
| the theory of record | `MATHEMATICS.md` (thesis §0; equilibrium family §1; depth law §2; confound §4; consequence fidelity §6; absorbing chain §7; Kirchhoff §11; Page's L §12; RIP §13; Welch §14; STE §17) |
| architecture-and-state | `workdonenew.md`, `workdonenewseal.md` (status report, every number sourced), `ARCHITECTURE.md` (diagrams) |
| the standing contract & laws | `CONTRACT.md` (§0 D-1..D-4), `CEQ_V16_CONTRACT.md`, `CEQ_V20_R15_CONTRACT.md` (L-FLOOR, L-CERT, annex M1–M16) |
| round 15 | `V20_R15_JOURNAL.md` (corrections index C1–C40), `V20_R15_THEORY_TABLE.md`, `V20_R15_LEAP_LEDGER.md`, `V20_R15_WING_MANIFEST.md`, `V20_R15_IT35_JUPITER.md`, `V20_R15_IT35_MARS.md` |
| the failure taxonomy | `MISTAKES.md` (65 entries; headings `### V-n / P-n / M-n / D-n`), `R10_MECHANISM.md` (the surface-proxy defect), `STRUCK.md` |
| prior art of record | `PRIOR_ART.md` (Hankel program §1–3; equilibrium labels §4; VGPE §5), `RESEARCH.md` (signed attention table), `V13_TIER6_PRIOR_ART.md`, `V13_X25_G1_PRIOR_ART.md`, `V13_X27_G1_PRIOR_ART.md` |
| negatives & prognosis | `PROGNOSIS.md`, `RESEARCH.md`, `D1.md`, `IMPOSSIBLE.md`, `THEORY2.md` |
| the arms | `ceq/arm_smprime.py` (three corners; `path_product` at :144), `ceq/arm_pl.py` (telescope head), `ceq/arm_phase.py`, `ceq/attention.py` (the dead signed operator; header has the tier table), `ceq/lm.py`, `ceq/hf/` |
| the beds | `scale/negation_scope.py` (BED-M, `equilibrium_oracle` :286), `ceq/corpus.py`, `ceq/beds/bed_k.py` (delay/power-law), `ceq/beds/bed_1.py` (committor, guards, CK, Morse), `scale/e4_harmonic.py`, `scale/kirchhoff.py` |
| topology in the tree | `ceq/rips.py` (S² Rips corpus from mujoco#3396), `ceq/certs/topological.py` (Z winding, persistent β₁, Euler–Poincaré), `ceq/multizoom.py` (fine-near/coarse-far with a bound), `scale/hyperbolic.py` (Gromov δ), `tests/foreman/test_topology_washout.py` |
| cost & device | `COSTS.md`, `ceq/sizing.py`, `ceq/mz_kernel.py`, `scale/m3_flops.py`, `V17_R4_RETAKE_PRICE.md`, `scripts/k_cert.py` |
| Lean | `lean/CEQ/*.lean` (12 files) |
| journals | `results/*.jsonl` — parse with `scale/ledger.py`, never grep `house-events.jsonl` |
| maths survey | `results/r9_maths_survey.md` (stage A/B, differentiable top-k, DPP, SCM surgery, sheaves, submodularity) |
| THEORY v1 (the original programme) | `THEORY.md` (resolvent `(I−γP)^{-1}` as successor operator, CSR schedule, kernels#22, sheaf gate) |

## 5. RULES — every planet, no exceptions

1. **NO CODE.** The author's instruction is "do not write any code, only docs". You write
   `.md`, `.tex`, `.bib` only. You may *run existing* repository code and tests to read a
   number (`python -m pytest tests/x -q`, `python -c "..."` one-liners that only read),
   but you create no `.py`, `.lean`, `.sh` file anywhere. If you need a number that only
   new code could produce, write `NOT MEASURED — needs <what>` and move on.
2. **NO GIT WRITES.** No `git add/commit/checkout/stash/branch`. Read-only `git log`,
   `git show`, `git ls-files` are fine. (`CONTRACT.md` D-1.)
3. **Evidence classes on every load-bearing claim** (`super-investigator`): `RUN` (you
   executed it this session), `READ` (`path:line`, quoted), `CITED` (resolved primary
   source, title matched; mark `[V]` if the abs/DOI page was fetched this session, `[U]`
   if reached through a search index or memory), `DERIVED` (steps written out). A `GUESS`
   is labelled or absent. **No number without provenance. No citation without an
   identifier you fetched.** A fabricated citation is the one defect that ends the paper.
4. **Quote sparingly.** At most one short quotation (under 15 words) per external source;
   paraphrase everything else. Never reproduce a paper's abstract.
5. **House style** (`ship-like-teerth`, `research-readme`): extensive, not lean; academic
   framing (Abstract, Keywords, numbered sections, displayed equations); negatives are a
   first-class section; every claim names its control; Limits collected once at the end;
   attribution *Invented by Teerth Sharma* with the repo link. No superlatives without a
   number. Third person; the paper does things, no "I".
6. **Design against the taxonomy.** Every bind, gate, bed, control and prediction you
   propose names the `MISTAKES.md` mechanism it is designed against (e.g. V-24 empty
   rejection region ⇒ planted mutilations; V-25 hypothesis no draw satisfies ⇒ domain
   census; D-1 racing a proven optimum ⇒ the label class; D-2 oracle equal to the arm's own
   resolvent ⇒ the oracle runs on the latent environment chain, the arm never sees its
   transition matrix; M-2 refitted threshold ⇒ thresholds fixed before data; L-SIGN ⇒ a
   counter-prediction beside every prediction; L-FLOOR ⇒ an information floor beside
   every capability number; L-CERT ⇒ a certificate with printed δ beside every mask).
7. **Occupied components are cited before they are named.** If a component exists in the
   literature, say who owns it. The delta is stated *narrowly*. Absence is recorded as
   `NOT FOUND`, never as "novel".
8. **Do not restate the north star as a slogan.** Every sentence about capability names
   the bed, the label, the floor, the control, the metric.
9. **Return value.** Your final message is data for the coordinator, not prose for the
   author: a short JSON-ish summary (file written, claims count by class, open gaps).

## 6. THE PAPER'S SHAPE (sections; the assembler follows this)

0. Front matter: title, author, abstract (120–200 words), keywords.
1. Introduction — the north star in the author's words; what self-attention is bad at,
   with the four cited obstructions; the one-sentence thesis; the contributions list.
2. What the record proved — the three-corner family, the telescope, the segmentation
   certificate, the scan-blindness theorems, the resolvent identities; every theorem by
   name with its measured instance.
3. What the record refuted — the negatives, the mechanism taxonomy by class, and the
   headline defect (D-1) that explains why the campaign could not win on its own beds.
4. The shape — definition; the five identities; propositions (parity; corner 3 is a
   resolvent; committor is a resolvent read with boundaries; Neumann certificate;
   joint determination; segmentation ⇒ block structure); what it computes (consequence,
   committor, safest move, next transient state); the interventional channel; the
   topological layer (segmentation, β₀ persistence of the influence graph, isocommittor
   surfaces, Morse/Euler–Poincaré certificates, Mapper cover as the long-context
   candidate builder); the cost law and the kernel path.
5. Why softmax cannot occupy this ground — the four obstructions, exactly what each
   theorem licenses, and the deeper-softmax skyline as the honest control.
6. Beds, labels, floors, metrics — BED-M (contained), BED-K, BED-1 (committor), BED-S
   (new: safest move under K constraints, with candidate moves in the context), the
   vector-valued label, the state metric, the information floors (Fano on the argmin,
   the exact oracle at 0), the pricing rule (N ≥ 8, MDE, TOST N = 70).
7. Prior art — the occupancy table (each lineage, owner, what it takes, what is left),
   the delta stated narrowly, NOT FOUND list.
8. The programme — the plan as a DAG with a critical path: Lean targets, instruments,
   binds with planted negatives, beds, the arena, predictions and counters, prices on the
   certified RTX 4060, kills, the leap dossier discipline, what dies if what.
9. Limits — collected once.
10. References.

Deliverables (the coordinator assembles them; you write sections to `$SCRATCH`):
`docs/CEQ_SHAPE.md`, `docs/CEQ_SHAPE.tex`, `docs/CEQ_SHAPE.pdf`, `docs/references.bib`,
`docs/PLAN.md`.
