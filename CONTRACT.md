# THE CONTRACT

## THE CHALLENGE

Produce an attention mechanism with at least one **CAPABILITY** that standard
attention measurably or provably cannot have, novel against directly-fetched
prior art, surviving pre-registered kills.

**Taxonomy cells count for nothing. Statistics are not capabilities.**

**Terminal deliverable: a module trainable in Google Colab and uploaded to
HuggingFace.** A route GREEN that never becomes a trained checkpoint is a
statistic; a checkpoint with no capability table is an artifact nobody can
evaluate. Both halves ship or neither does.

**Current design bid:** pivot-routed signed multi-hop — all hop ≥ 2 propagation
through k content-selected pivots P, signed denominator-free base matrix,
numerical-radius guard, hopcache decode.

---

## §0 — DISPATCH AND ORDER LAW

Four rules governing how work is scheduled, added 2026-08-31. They bind every
round and are not round-specific.

**D-1 DEPENDENCY LAW.** Work is a DAG. Parallel dispatch is legal only on nodes
with no shared repository state — prior-art fetches, document reads, paper
derivations. Everything else is a chain and runs as one agent. Note that the
shared state which actually collides is *git*, not the file set: agents writing
disjoint files still contend for the index and the branch. Dispatched agents are
therefore forbidden from `git add`, `git commit`, and every other git write; the
coordinator reviews and commits serially.

**D-2 SKILLS ARE MODES.** Investigator, caveman, and tda-tdd are registers a
single agent wears, not workers. Planet names label responsibilities inside
documents; they do not name concurrent processes. A contract line reading
"MERCURY does X" is an assignment of accountability, not a dispatch instruction.

**D-3 LOOP GATE.** No autonomous loop mounts until the deactivation commit,
`MISTAKES.md`, and the standing loop failures have been read, and the reason
the previous loop died is stated in one sentence in the mount request. The
iteration count comes from the DAG's critical path and is never chosen by an
agent.

*First application, 2026-08-31.* The previous loop did not terminate; it was
removed by hand. `max_iterations: 0` disabled the only ceiling
(`stop-hook.sh:61`), and the `completion_promise` demanded two forty-iteration
ranges whose second could not begin until the first reached it.35
(`PHASE2_CONTRACT_V_MAIN_4.md:22-30`), so no exit condition could fire. The
state file was renamed out of the hard-coded path `.claude/ralph-loop.local.md`,
whose existence is the only thing `stop-hook.sh:13-18` tests; the `active: false`
field credited by `ecbedf7` is never parsed. Filed as V-20.

**D-4 ORDER.** Round 11 it.1–22 — registration and the reading — precede every
line of v-main.7. Contracts that schedule work behind an unreached round are
staged, not started.

**Naming correction.** Contract text through v-main.7 directs delta sentences
and the architecture statement to `ARCH.md`. No such file exists in this
repository and none ever has. The standing architecture-and-state document is
`workdonenew.md` ("where CEQ actually stands"), and it is the target those
instructions mean. Two further contract references resolve to nothing: rule
D-3 had no text in the tree before this section, and "the fifteen standing loop
failures" is not an enumerated set but the raw pytest summary at
`results/r10_loop_suite4.txt:21-35`, whose membership varies across snapshots
(15 / 15 / 14 / 15, three different sets) while its count happens to stay near
fifteen.

---

## CHECKLIST (work-stopping clause)

Item states: UNTESTED / GREEN / RED. Any MANDATORY item RED after its test runs
⇒ all build work stops; only instrument repair, prior-art search, or write-up
continues. **Items are immutable once their test has run.**

**M1. NEGATIVE INFLUENCE ON VALUE PATH.** Kill: min influence ≥ 0, or the most
negative entry is a frozen zero-gradient cell. (Precondition, not novelty.)

**M2′. SIGNED INFLUENCE CONTEXT CANNOT DILUTE — route-dependent.**
Naive flatness of a flip PROBABILITY at global reach is impossible
(Littlewood–Offord lower bounds). Four escapes, each breaking one hypothesis of
the impossibility; **ANY ONE GREEN suffices.** Test order 2 → 4 → 1 → 3:

- **R2 SIGN-DETERMINACY** (breaks "magnitude statistic"): enforce a
  sign-nonsingular pattern on the k×k pivot block; test = magnitude-
  randomization invariance, 10⁴ resamples, sign pattern bitwise constant.
  Kill: determined fraction ~0 in trained blocks, or constraint destroys
  training.
- **R4 HIERARCHICAL CRITICALITY** (breaks "flat aggregation"): Dyson-tree
  aggregation, level coupling 2^(−θl); slope in s is a function of θ.
  Kill: no zero crossing of slope(θ) in range, or θ* unstable over 3 seeds.
- **R1 CERTIFIED SELECTION** (breaks "generic token"): group-testing /
  d-disjunct decoder recovers the k causal tokens with combinatorial guarantee
  at polylog overhead; post-recovery background has k terms.
  Kill: recall < 1−ε at s=2048, or overhead exceeds stated polylog envelope.
- **R3 NON-ARCHIMEDEAN ROUTING** (breaks "Archimedean sum"): symmetrized
  max-plus / valuation aggregation — minimal-valuation term dominates
  independent of s; Newton-polygon certificate of dominance.
  Kill: annealed training > 1.10 of softmax, or balanced-ambiguity decisions
  > 10%.

**M3. LONG-RANGE SIGN CAPABILITY, ABSOLUTE BAR.** Negation-scope flip at
distance d, executable oracle, 5 seeds, matched params and lr sweep; softmax
failure distance recorded FIRST. Kill: NRMSE > 1.0 (predict-the-mean), or CIs
overlap softmax at all d ≥ 256, or the unsigned-pivot ablation matches (then
routing is the contribution — rewrite claim, re-enter at M3).

**M4. EXACT EVICTION.** Evict-before-read window: settled-state change ≤ 1e-12;
W9 non-retroactivity stated in every claim. Kill: leakage, or scope dishonesty.

**M5. CERTIFIED FINITE COMPUTATION.** Lean theorems' hypotheses satisfied by
the tensor that ships (grep-bound); `lake build` exit 0, zero `sorry`,
calibrated sorry-detector. Kill: any hypothesis unmet by shipped code.

**M6. DENOMINATOR-FREE MAGNITUDE CERTIFICATE.** Numerical-radius guard
w(A) ≤ ρ ⇒ ‖A^h‖ ≤ 2ρ^h — one scalar, no sum over s. Kill: guard reintroduces
M2′ failure, or training diverges at every lr.

**SUPPORTING** (not work-stopping): **S1** hopcache decode parity ≤ 1e-12
relative. **S2** ablation table pivots+unsigned / pivots+signed / dense+signed —
mandatory before any claim sentence. **S3** Carnot ε^h hop grading vs learned
scalars at matched params.

**GLOBAL STOPS:** **G1** prior-art sweep (direct fetch: Nyströmformer,
landmark, NSA, MoBA, ParaFormer, DeltaNet, SDA, SignGT, Cog, and per-route:
Brualdi–Shader applications, tropical attention, hierarchical/RG attention,
group-testing attention) finds the cell occupied. **G2** any published
`bench.py` number moves. **G3** any arm found reporting another arm's number
(the ParaFormer bug class). **G4** all GREEN but S2 attributes the capability to
the unsigned ablation.

---

## MATHEMATICAL ARSENAL

Each item: hook + kill. **[V]** = verified this project, **[U]** = fetch before
use.

### TIER 1 — load-bearing

- **A1 Littlewood–Offord / anti-concentration + inverse-LO** [U; Tikhomirov
  2020, CJMS ~2023] → M2′ exponent accounting, R1/R2 background counts.
  Kill: background small-ball mass fails to factor the measured exponent
  (−1.389 vs −3/2 / −4/3 candidates).
- **A2 Numerical range**: Berger power inequality, Crouzeix–Palencia (1+√2) [U]
  → M6. Spectrum {0} makes eigenvalue guards vacuous; W(A) is the non-vacuous
  control. Kill: per M6.
- **A3 LGV / total positivity / Grassmannian sign variation** [U; Karp,
  Galashin–Karp–Lam] → M1's theorem-form (variation diminishing = softmax's
  prison) and the open question: which sign strata are reachable by causal
  signed path sums.
- **A4 Directed polymers / KPZ** [U; Dauvergne–Ortmann–Virág] → path
  localization O(s^{2/3}) as the physics behind pivot routing; sign flips as
  partition-function zeros. Kill: participation ratio flat in s ⇒ pivots stand
  on A1 alone.

### TIER 2 — structural

- **B1 Carnot groups / sub-Riemannian** [V: nilpotent Lie groups, Hausdorff >
  topological dim] → S3; graded dimension Σ h·rank(hop-h) as checkpoint
  measurable.
- **B2 Discrete Morse theory** [V] → M4/M5; free-face eviction IS an elementary
  collapse; Refcount floor = critical-cell count; Lean target.
- **B3 Möbius inversion on incidence algebras** [U; Rota] → W9
  retroactive-eviction repair. Kill: correction costs Ω(N).
- **B4 Filled Julia sets / escape-time certificates** [U] → depth-stacking
  stability of composed hop polynomials on W(A_l). Kill: boundary never binds ⇒
  demote to diagnostics.
- **B5 Magnitude homology** [U; Leinster] → selector probes, judged ONLY by
  `dfloor_probe2` (flat 0.250 chance, cosine control). Kill: loses to cosine —
  the ΔFloor death, no appeal.

### TIER 3 — reincarnated with cause

- **C1 Sign-solvability / SNS patterns** [U; Brualdi–Shader, Cambridge Tracts in
  Mathematics 116, CUP 1995 — VERIFIED by direct fetch] → **IS route R2.** Only
  math where sign survival is magnitude-independent; computable at k×k where it
  was hopeless at s×s.
  **Standing caveat:** sign-determinacy is insensitivity to magnitude; pushed to
  its limit it drives the flip rate to ZERO, which is softmax's number. The
  salvageable target is **path coherence** — the j→i paths through c share a
  sign so c's contribution adds instead of cancelling.
- **C2 Frustration index / switching classes** [U; Zaslavsky] → integrity check:
  frustration ~0 = signed-in-name-only; probe must be switching-gauge
  invariant. **This item is itself a kill.**
  **Already run [RUN]:** tgate 0.5056 unbalanced triangles, deltanet 0.5641,
  sgate 0.2785, **softmax 0.0000** (instrument validates itself). The kill does
  NOT fire. Gauge invariance exact: `max|(I−DAD)⁻¹ − D(I−A)⁻¹D| = 0.000e+00`.
  Remaining: re-run on trained checkpoints.
- **C3 Blackwell approachability** [U] → training the M1–M6 VECTOR into its
  target set without scalarization. Kill: weighted sum reaches the same GREEN
  set.
- **C4 Nucleolus pivot selection** [U; Schmeidler] → coalition-stable P at small
  k via LPs. Kill: loses to top-k salience in S2.

### DEAD — do not revive without new evidence

DEQ/Hopfield settling · multigrid/spectral decimation/mixed curvature ·
sheaf-Hodge contradiction energy · Nash/QRE/ESS/MFG · Lyapunov-spectrum shaping
(migrated into A2) · IFS-collage compression · free-probability hop scaling ·
Lorentzian polynomials.

---

## STANDING DOCTRINE

- **Calibrate every instrument against a case where it must fire and one where
  it must not, before believing it.**
- **Stream isolation:** new arms may not move any published number.
- **Softmax baseline runs first and appears in every table.**
- **Honest cost accounting** (wall-clock, memory, tuning asymmetry) beside every
  quality number.
- **Negative results and deletions are recorded in Limits, above results, and
  pinned by tests.**
- **Deletion, not defense, when a kill fires.**
- **The idea is the patient.** When a theory dies you owe a REPLACEMENT, not
  just an autopsy — with a pre-registered kill that can structurally fire,
  softmax's number in the same table, and an honest cost.
- **Nurses are inference engineers: cheat hard, declare every cheat.** An
  undeclared shortcut is a fabricated result.

**This document is the contract.** `CHECKLIST.md` carries the live item states
and outranks it operationally; `LOOP_PROMPT.md` is the per-iteration procedure.
