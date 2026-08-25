# Prognosis — consequence-equilibrium attention

Room: Wilson (facts), Foreman (root cause), Chase (failure modes), Cameron (better path),
Dr House (fable leap), Health Inspector (log audit). 83 claims audited, 29 audit events.
All 9 green claims re-ran green; zero strikes under the re-run check.

---

## Prognosis

**All four named decisions are dead as formulated, and the core concept is not.** The
equilibrium was never the problem — every named *implementation* of it was. `sigmoid`'s
`(I − A)z = b` is a direct linear solve with no iteration, so decision 1 never existed
as a DEQ; a real DEQ silently loses its fixed point at step 47 while the loss keeps
falling. Decision 2's row-stochastic constraint confines the output to a scaled convex
hull of `b` (NRMSE 0.590 against an unconstrained fit, 8 of 32 target coordinates
unreachable at any `(P, γ)`). Decision 3's hierarchical schedule loses 0/15 to a
one-line sliding-window+sinks mask, costs 53× the attention it schedules, and receives
no gradient at all. Decision 4's harmonic space is `d mod 2` on any graph with a cycle —
exactly zero at every even stalk width tested.

**What survives is a single untested path**, and it is the one Dr House named: the
equilibrium taken in the **(max, +) semiring** rather than `(+, ×)`. The reduction that
killed the linear version is precisely what makes this non-trivial — APPNP *is* the
Kleene star in `(+, ×)`, which is why `torch.linalg.solve` eats the DEQ and why the whole
stage reproduces Gasteiger et al. 2019 to 2.22e-16. In `(max, +)` the Kleene star is
Bellman optimality, which has no linear closed form, so the solver does real work and the
fixed point is *literally* a value function rather than SR-flavoured. It is a hypothesis
with no RED test behind it and it stays in Open until a fellow binds it.

**The deliverable order inverts.** The interventional domain is executed code, not
physics — physics pretraining measured identical to a sham prior with its action column
permuted (0.3667 vs 0.3654, against 0.3654 for matched-λ scratch). The corpus of
`(program, mutation, executed-output-before, executed-output-after)` does not exist
publicly, needs no GPU, no Triton, and no `trust_remote_code`. Ship that first.

---

## Chart

**Wilson** — 8 claims verified, 0 fabrications, 2 content failures.
- Ramsauer: "softmax = one Hopfield update" is exact; "one iteration reaches the fixed
  point" is a stronger paraphrase — error is exponentially small in pattern separation
  `Δᵢ`, not zero. **Kept, and it argues *for* the design**: small `Δᵢ` is caustic's orbit
  collapse, so the equilibrium buys most exactly where errors live.
- Dayan 1993 prints `(I − Q)⁻¹` with **no γ**, `Q` sub-stochastic from a finite absorbing
  chain. **Kept** — THEORY.md attributed later notation to the primary source.
- The ~30-architecture inversion claim: **no such study exists.** Nearest is Tay et al.
  (arXiv:2207.10551) — ten architectures, 2.9B dense, no intercept-vs-slope
  decomposition. **Kept.** It cannot appear on a model card.
- `ker Δ_F` trivialization requires **at least one cycle** (Hansen). **Kept**, and it is
  the qualifier that makes Foreman's parity law actionable.

**Foreman** — 15 RED / 26 GREEN. Root cause: the two constraints of §1 pin `P`'s top
eigenvector to `1`, and every downstream consequence is that one fact booked four times.
- `σ_max(γP)` reaches 1.9863; row-stochasticity bounds the ∞-norm, not the 2-norm
  `sigmoid` reports. **Kept** — independently reproduced by the orchestrator (√2 witness)
  and now machine-checked in Lean.
- The resolvent of a stochastic kernel cannot amplify any mode above the mean: best ratio
  1.000000 over 1,920 kernels. Horizon and contrast are reciprocals. **Kept.**
- Decision 1 vs decision 2 is a dichotomy, not a risk: affine → DEQ residual 1.05e-15
  (outer solver computes nothing); nonlinear → 57.4% from `(I−γP)⁻¹b` (SR reading void).
  **Kept, and it is the finding Dr House's leap answers.**
- §6's resolvent stage is APPNP bit-for-bit, 2.22e-16. **Kept.**
- "Topology is decorative — the barcode is never an input to the resolvent."
  **OVERRULED twice.** First by the Inspector: UNBOUND, its only evidence is a *green*
  test and a passing test is not a falsification. Second, and more decisively, on
  aim — see *The barcode charge is misdirected* below. It is a true statement about
  `THEORY.md` §6 and a false one about the code that already exists.
- "sigmoid's README contains no `rho_max`, no NRMSE, no `(I−A)z=b`."
  **STRUCK.** All four strings are live at byte offsets 17865 / 18243 / 18448 / 18564 /
  18747. The nurse's fetch failed and its silence was logged as a finding; it also
  contradicted Foreman's own correct `σ_max` finding.

**Chase** — 32 RED / 10 GREEN, on real hardware (RTX 4060, torch 2.5.1+cu121).
- **`BlockMask.from_kv_blocks` computes fully dense attention while `to_dense()` reports
  the sparse pattern.** Output is `0.000e+00` from unmasked dense — bitwise identical —
  and causality is dropped too. `create_block_mask(mask_mod=...)` is exact. **Kept, and
  it is the highest-blast-radius finding in the room:** any sparsity or speedup number
  taken through that path is dense attention wearing a sparse label.
- Corrupt CSR: out-of-range offsets → CUDA illegal memory access that poisons the process
  context (contaminated 13 subsequent tests); negative indices → no exception, finite
  output, wrong by 3.27 absolute. The empty-row guard lives only inside
  `build_topology_block_schedule`, so **any new builder does not inherit it**. **Kept.**
  Rollback: four lines of validation in `scheduled_attention()`.
- DEQ loses its fixed point at step 47, trains 922 more steps, ρ(J_f) 0.4942 → 1.0086,
  residual degrading 45,230×, loss falling monotonically throughout. **Kept, verified to
  the digit by the Inspector.** No rollback from a loss-only monitor. No DEQ at ~1B scale
  exists in the literature, successful or failed.
- **Theorem 1 as a training objective is blind to correctness.** `floor(20 correct) = 0`
  equals `floor(20 wrong) = 0`; `select_by_floor` prefers pure noise (floor 0) to a
  100%-correct candidate carrying 2 observable collisions (floor 2). Descending a
  differentiable relaxation improved the surrogate 18.7× while the certified floor never
  moved and accuracy never moved. And `verify_injective` takes `gold: dict[str, str]` —
  **the guard needs the answer key the objective claims to avoid.** **Kept.** No rollback
  once trained on it: the weights are the artifact.
- Schedule rebuild costs 53× the dense attention it replaces (42.12 ms vs 0.792 ms at
  seq 8192). **Kept** — and it explains Cameron's 0/15 mechanically.
- The one genuinely new component receives no gradient: `d(loss)/d(salience)` is `None`,
  top-k is an argsort. Rebuilt between forward and backward: 9.4% mask difference,
  gradient relative error 0.433, nothing raises. **Kept.** §3 is a hyperparameter to
  select, not a component to train.
- Row-stochastic: `(I−γP)⁻¹b` is confined to `[min(b), max(b)]/(1−γ)` in 600/600 draws;
  simplex projection destroys 82.1% of gradient norm in one step; dense solve is 5.9× the
  attention beside it (8.09 ms vs 1.37 ms at d=2048, 129 ms per forward over 16 layers).
  **Kept.** Rollback: truncated Neumann, K matvecs, 68× cheaper at K=20.
- HF shipping: `trust_remote_code=False` refuses; CPU has no fallback path; **Triton
  ships Linux-only wheels and needs compute capability 8.0+, so a free-Colab T4 (sm_75)
  cannot run it at all**; Serverless Inference has no `trust_remote_code`, so the model
  page gets no working widget; `AttentionInterface.register` does not register into
  `AttentionMaskInterface`. **Kept.** All four have rollbacks.

**Cameron** — 9 RED / 21 GREEN / 1 xfail.
- **Physics→language transfer is zero.** Physics prior 0.3667, sham prior with permuted
  action column 0.3654, matched-λ scratch 0.3654. The apparent gain was regularization
  strength; her own first single-seed run passed, and the null control caught it.
  **Kept — this is the finding that redirects the whole programme.**
- Hierarchical schedule 0.39 vs window+sinks 0.975, beats window+sinks 0/15, below the
  random null in exactly 2 cells. Beats the 0D-salience control 6/15. **Kept**, verified
  by the Inspector on real SmolLM2-135M attention across 15 cells.
- An H-matrix far-field is **low-rank, not sparse**; CSR can only keep or drop. **Kept.**
  This is the structural reason §3 could not have worked as specified.
- Row-stochastic cannot represent negative literal→outcome coupling: fits +0.0003 where
  ≤ −0.05 is needed, because a convex combination has no negative entry. **Kept.**
- "flex backward matches dense to atol 1e-9", "executed Python is a working interventional
  domain", "Lean 4.33.1 + `repl` is a `do()` with a kernel oracle", "the constraint costs
  only 1.04×" — **all UNBOUND.** Supporting tests pass but were never logged, and the Lean
  clause has **no test anywhere in `tests/`**.
- Log integrity: 6 of 7 finding lines are truncated JSON and do not parse; zero green
  events logged against 21 passing tests. **STRUCK.**

**Dr House** (fable, 5 min) — returned **(a) THE LEAP**, not "no leap".
- Replace `(I − γP)⁻¹` with the **max-plus Kleene star**: `z* = maxₐ(rₐ(x) + γAₐz*)`,
  Bellman optimality. Escapes the APPNP reduction because APPNP *is* the Kleene star in
  `(+, ×)`; in `(max, +)` there is no linear closed form. Contraction via the Perron
  certificate rather than row-stochasticity, so the eigenvector is not pinned. `do()`
  becomes row surgery on `Aₐ` that propagates because the output is re-solved.
- **HYPOTHESIS, UNBOUND.** Exempt from the RED-first rule by construction, therefore
  barred from the verdict. It re-enters the differential as a candidate and stays in Open
  until a fellow binds it with a failing test.
- His own caution, kept: one-step tropical attention exists (2025). The ownable claim is
  the **star** — equilibrium plus certificate — not the semiring swap.

**Inspector** — 83 claims audited, 4 struck, 5 unbound, 3 contradictions resolved.
- Struck: Foreman's sigmoid-README claim; Cameron's `test_domain.py` red-label; Cameron's
  6 truncated log lines; Cameron's zero-green-coverage.
- Resolved: the two NRMSE figures are **not** in conflict — Foreman measured in-sample
  one-step on lifted Lorenz with the *whole* operator constrained against a 7.08e-12
  baseline; Cameron measured held-out OOD on executed code with only the state→state
  block constrained against an already-28%-wrong baseline. Different system, scope,
  sample, and swept axis.
- Scope limit: events appended after the Inspector's `done` line are unaudited.

---

## What replaces what

| dead | replacement | status |
|---|---|---|
| DEQ fixed-point iteration | direct solve, or max-plus value iteration where no closed form exists | max-plus unbound |
| row-stochastic `P` + `γ` | **Perron certificate**: `A ≥ 0`, `∃ w > 0` with `Aw ≤ ρw` ⇒ `ρ`-contraction in `‖·‖_w` | **proved in Lean**, runtime guard required |
| dense resolvent solve | truncated Neumann `∑_{k<K} γᵏPᵏb`, K matvecs, error `γᴷ/(1−γ)` | **proved in Lean** (`occupancy_telescope`) |
| hierarchical H-matrix CSR schedule | sliding-window + attention sinks as a `mask_mod` predicate | measured 0.975 vs 0.39 |
| raw-CSR `BlockMask.from_kv_blocks` | `create_block_mask(mask_mod=…)` | exact, `0.000e+00` |
| MuJoCo / physics interventions | executed code; Lean REPL as a kernel-oracle `do()` | code measured, Lean **unbound** |
| Theorem 1 as a training objective | Theorem 1 as a **discrete selector** over enumerated candidates, `verify_injective` against a held-out labelled set | Chase's constraint |
| sheaf harmonic gate | buried | `d mod 2`, exactly 0 at even `d` |

Two Lean theorems now do engineering work rather than decoration:
`occupancy_telescope` **is** the truncation-error certificate for Chase's 68×-cheaper
Neumann rollback, and `weighted_contraction` is what licenses dropping row-stochasticity
without losing Banach.

---

## The barcode charge is misdirected

Foreman's most damaging unbound claim was that topology is decorative because the barcode
never reaches the operator. That is true of the pipeline in `THEORY.md` §6, which routed
the barcode into a *schedule* that then fed a graph resolvent — the barcode genuinely
never arrives there. It is false of `sigmoid`, which already solved this.

`sigmoid` §3.3, the **Hilbert-series embedding**:

> "A barcode has no fixed length, so no linear operator can act on it. The numerator of
> its Hilbert series supplies one — births contribute positively, deaths negatively.
> **This step is what makes a *linear* operator on topology possible at all.**"

```
N(s) = Σᵢ s^{bᵢ} − Σᵢ s^{dᵢ}      c_k = #{births in bin k} − #{deaths in bin k}
```

And §3.4 makes those coefficients half the state: `z = [ψ ; u]`, `ψ` = Hilbert
coefficients plus Betti curves, `u` = whitened PCA of the standardized activation. The
operator acts on `z`. **The barcode is inside the operator, not beside it.**

Foreman's own green test cuts the other way once aimed correctly. Showing that
`d → d^1.5` moves the barcode while leaving the kNN graph bit-identical proves the
barcode carries information the graph *cannot* express. That is an argument for wiring it
in, not for calling it decoration.

**Three specification traps carried forward from `sigmoid` §3.1 and §3.5**, each already
measured, each capable of silently reducing the topology channel to noise:

1. **Standardize by median and 1.4826·MAD, not mean and σ.** Measured on distilgpt2, the
   largest per-dimension standard deviation was **22.1** against a median of **0.308** — a
   **72× ratio concentrated in about five channels**. Without robust standardization the
   barcode is a function of those five outliers and the "topological" feature measures
   activation magnitude and nothing else.
2. **Which cloud — temporal or spatial — decides the result before any model is fitted.**
   On an entity corpus whose ground truth is H₀ of the entity cloud, the temporal encoder
   scored **0.386 against a 0.487 majority baseline**: actively worse than guessing.
3. **Which scale.** Dividing filtration values by cloud diameter buys scale invariance and
   discards scale. When the quantity of interest lives at a fixed physical distance — a
   contact threshold, a constraint radius — normalizing destroys exactly the signal.

Reusable API already shipped in `topological-ml-toolkit`, sklearn-shaped:
`PHFeaturizer`, `BettiCurve`, `PersistenceImage`, `point_cloud_signature`,
`activation_signature`, `persistence_similarity_trajectory`. `activation_signature` is the
direct entry point for a residual stream.

And H₀ needs no simplicial complex: for a Vietoris–Rips filtration the H₀ death times
**are** the MST edge weights, exact at `O(W²D)`. That identity is what makes the merged
`kernels#22` schedule run 13× faster than its reference.

---

## Open

- **The max-plus star.** Dr House's leap, unbound. Falsifier he specified: synthetic
  token-MDPs, train on dynamics A, edge-deletion interventions at test, evaluate return
  prediction on B; three arms (standard attention / APPNP / max-plus star) at matched
  parameters. **Dead if arm 3 matches arm 2 post-intervention.** Plus a collapse probe:
  residual of the best affine fit to the learned `f` at `z*` — if the maxes anneal
  inactive it quietly became APPNP.
- **The Perron certificate's runtime hypothesis.** Chase found the failure the proof does
  not cover: on a **reducible** non-negative `A`, power iteration returns `w` with exact
  zero entries (`w = [0.768, 0.640, 0.0, 0.0]`, min 6.6e-14), so `‖v‖_w` divides by zero
  and the norm is undefined — and a learned `A` becomes reducible the moment a block is
  driven to zero. On an **imprimitive** `A` there is no dominant eigenvalue at all. The
  Lean proof is not the exposure; it is machine-checked and holds under `w_pos`. The
  exposure is the runtime check, and it must **fail closed**: refuse the step, keep the
  last certified `w`, alarm. Requires an irreducibility precondition and `min(w)` logged
  every step.
- **Physics→language at scale.** Cameron's null shows the *mechanism* of failure at
  6-dim / 135M / 1024 tokens. It does not establish failure at 1B.
- **Attention-mass recall is a proxy.** 97.5% of oracle mass is not end-task accuracy.
  The schedule ablation should be re-run against downstream accuracy before §3 is
  formally cancelled.
- **Lean REPL throughput.** No source gives tactics/sec. Snapshotting work
  (arXiv:2605.25556) reports a few ms to 500 ms per branch, 95th percentile 289 ms —
  enough to suggest a corpus is buildable, not enough to size it.
- **Weighted-norm transient growth.** Chase measured `κ = max(w)/min(w)` up to 2904.7 and
  the inequality permits κ-fold L2 transient under a valid `ρ = 0.9` certificate, but did
  not construct an adversarial `A` that exhibits it. Statement stands, demonstration does
  not.
- **`ker Δ_F`** itself remains untested directly; the trivial-fixed-point collapse
  (`‖z*‖ = 5.8e-12`) was the outer DEQ, not the sheaf Laplacian.
- **PR #22 test-count discrepancy**: the body reports "17 passed" for a file containing 18
  test functions. Unreconciled.
