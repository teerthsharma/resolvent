# Consequence-Equilibrium Attention — Theory v1

Status: first theory. Written to be attacked. Nothing here has survived the room yet.

---

## 0. The reframe

The original proposal reads as four new hard things built at once: a DEQ, a
successor-representation kernel, an H-matrix multiresolution solver, and a sheaf
Laplacian equilibrium. Four independent research risks multiply, and that is
normally how a programme dies.

The scan says three of the four are already built and merged, under different
names, in this author's own repositories. The architecture is therefore not four
new things. It is **one new thing plus a reinterpretation**, sitting on two
upstream-merged kernels.

| Decision | Claimed as new | Actual status | Asset |
|---|---|---|---|
| 1. Equilibrium, not one step | new | **exists** | `sigmoid` §5 solves `(I − A)z = b` with a measured Banach certificate |
| 2. Successor operator `(I − γP)⁻¹` | new | **exists, mislabelled** | `sigmoid` §4 fits an action-conditioned `T₀z + Σₖ aₖ Tₖ z + Ba + c`; §5's resolvent solve is the same object |
| 3. Fractal / multiresolution partition | new | **executor merged, schedule new** | `triton-lang/kernels#22` consumes a causal CSR block schedule; `topoml` supplies `metric_cover` / `nerve_graph` / `mapper_graph` |
| 4. Sheaf topology | new | **exists** | `sigmoid/sheaf.py` restriction-map gate; `topoml.sheaf_consistency_residual` |

**The single genuinely new thing is decision 3's schedule**: replacing the 0D-persistence
salience schedule with a hierarchical near-field/far-field block partition, and making
that partition switch per input at inference time.

---

## 1. The central identification

`sigmoid` learns a linear operator by ridge regression on a bilinear action lift:

```
φ(z, a) = [z ; a ⊗ z ; a ; 1]
W·φ     = T₀z + Σₖ aₖ Tₖ z + Ba + c
```

and obtains its fixed point by solving

```
(I − A)z = b       where A is the state→state block, ρ = σ_max(A)
```

That solve returns `z = (I − A)⁻¹ b`. The successor representation is
`M = (I − γP)⁻¹`. These are the same algebraic object. `A` is an unconstrained
ridge-fit matrix; `γP` is a discount times a transition kernel.

**Claim.** Converting `sigmoid`'s coupling operator into a successor operator is
two constraints on an existing, working, tested solve — not a new architecture:

1. **Factor the scale out.** Write `A = γP` with `γ ∈ (0,1)` a free scalar and `P`
   normalized. Today `γ` is entangled inside `A`, which is why `ρ` is reported as a
   diagnostic rather than used as a control.
2. **Make `P` a transition kernel.** Row-stochastic, non-negative. Today `A` is
   unconstrained, so `(I − A)⁻¹` has no occupancy interpretation and its entries can
   be negative — which is exactly why nobody read it as expected future occupancy.

Under those two constraints `(I − γP)⁻¹ b` *is* discounted expected future
occupancy, and the action-conditioned `Tₖ` blocks make it `p(· | do(aₖ))` rather than
`p(· | aₖ)` — provided the training data contains real interventions.

**Consequence for the contraction certificate.** With `P` row-stochastic,
`ρ(γP) = γ < 1` holds *by construction*, not by measurement. The Banach certificate
stops being something `sigmoid` hopes for and becomes a structural guarantee. This
also disposes of the `rho_max` problem the sigmoid README already documents and
rejects: clipping `ρ` to 0.995 degraded Lorenz one-step NRMSE from 0.067 to 0.317,
because clipping misreports chaotic dynamics. Constraining `P` to be stochastic does
not misreport anything — it changes what is being modelled from "the dynamics" to
"the occupancy of the dynamics", and occupancy is contractive whenever `γ < 1` even
when the underlying system is chaotic and `ρ_dynamics > 1`.

That last sentence is the load-bearing one and is the first thing the room should
attack.

---

## 2. The interventional domain problem, and why MuJoCo

The proposal already concedes that observational text cannot identify `p(y | do(x))`.
So `P` must be fitted where interventions are real, ground truth is available, and
outcomes are measured.

MuJoCo is that domain, and this author already has merged code in it:
`google-deepmind/mujoco#3396`, merged 2026-07-20, replaces an `ntree × ntree` dense
scratch in `mj_island` with a disjoint-set H0 partition (1,281.6× peak-memory
reduction at `ntree = 4096`). `sigmoid` was already validated against an
S²–Vietoris–Rips corpus derived from that PR, and reproduces the MuJoCo disjoint-set
island partition exactly.

A physics simulator supplies exactly what text cannot: you can set state, apply a
force, and measure the outcome — a literal `do()` operator with a measured response.
`aₖ` in the bilinear lift is the intervention. `Tₖ` is its effect.

**The gap this creates, stated plainly.** The win condition is stated in terms of
Turing-style, Lean, and AGI-style benchmarks, which are language tasks. `P` is fitted
in physics. Nothing in this design yet establishes that an occupancy operator learned
on MuJoCo interventions transfers to language reasoning. That is the crux of the
whole programme and it is currently an assumption, not a result.

---

## 3. The merged Triton kernel is the base point, and it is the right one

`triton-lang/kernels#22`, merged 2026-07-28, 5 files, +804/−1. A forward-only Triton
scheduled-attention kernel that consumes a **causal CSR block schedule**, with
topology-derived schedule construction from sink blocks, local-window blocks, and a
0D-persistence salience over key-block centroids. Tested for dense-CSR parity, block
selection, schedule validation, dtype preservation, 2D and batched/headed correctness.

Why this is the correct substrate and not merely a convenient one:

- **The attention pattern is already data, not code.** The kernel takes the sparsity
  structure as a CSR argument. Changing the geometry of attention requires building a
  different schedule, not writing a different kernel.
- **An H-matrix is a hierarchical block-sparsity pattern.** Near-field blocks dense,
  far-field blocks low-rank or dropped. That is expressible as a CSR block schedule.
  Decision 3 therefore needs a *schedule builder*, not a new kernel.
- **A schedule can be rebuilt per input.** This is the mechanism for "internal switch
  of phases or geometry in real time". The switchable geometry is the CSR schedule.
  Nothing else in the stack has to move.

The existing 0D-persistence salience builder is the control to beat. It is already
merged, already benchmarked, and any hierarchical builder that cannot beat it is not
worth the complexity.

---

## 4. Real-time phase detection, from an existing kernel module

`Epsilon-Hollow`'s `stratum` classifies a trajectory's regime — `Underfit` /
`WellFit` / `Overfit` / `Collapsing` — from the cycle rank of the Vietoris–Rips
1-skeleton over an arc-length-reparameterised Takens delay embedding, in **4,792
bytes per stream, no allocator, O(1) observation**.

That is a real-time regime detector that already runs in a kernel. Its intended use
here: gate schedule rebuilds. Recomputing a hierarchical partition every token is
wasteful; recomputing it when the regime changes is not. `stratum` is the trigger.

Reuse note: `stratum` is `topoml.time_delay_embedding → persistent_homology →
betti_at(...).beta1` rewritten under a fixed-memory budget. `topoml` is its
differential-test oracle and is currently not used as one.

---

## 5. Training without ground truth

`caustic` Theorem 1: for an injective ground relation on `n` entities producing `m`
distinct values, `err ≥ n − m`. The bound consults no answer key, which is what turns
it from a diagnostic into an objective — `select_by_floor` runs candidates in
competition scored by that floor, with the do-nothing candidate always entered.

Intended use here: **schedule selection**. Candidate CSR schedules compete; each is
scored by the certified floor on a held-out relation; the do-nothing schedule (the
merged 0D-salience builder) is always entered, so declining to switch is a
first-class outcome.

**The precondition, and it is not optional.** Theorem 1 requires the ground relation
to be injective. `caustic` documents its own violation of this — `Asmara`/`Asuncion`
share first token 1634 — and ships `verify_injective` as the guard. Applying Theorem 1
to schedule selection requires stating what the entities and the ground relation *are*
in that setting, and running `verify_injective` on them. Until that is stated, §5 is
an aspiration, not a mechanism. This is the second thing the room should attack.

---

## 6. What is actually being claimed

The architecture, end to end:

```
input window
  → barcode (topoml / sigmoid PH)                        [exists]
  → Hilbert-series embedding                             [exists, sigmoid §3]
  → hierarchical partition of the embedding space        [NEW — the only new part]
  → CSR block schedule                                   [format exists, kernels#22]
  → scheduled attention                                  [MERGED, kernels#22]
  → resolvent solve (I − γP)⁻¹ b, γ explicit, P stochastic  [exists as (I−A)z=b, sigmoid §5]
  → sheaf consistency gate ‖R·û − ψ̂‖                     [exists, sigmoid §6]
  → stratum regime check → rebuild schedule or don't     [exists, Epsilon-Hollow]
```

One new component. Everything else is merged, tested, or shipped.

## 7. Win condition

Primary: **intervention generalization.** Train on distribution A, intervene, test on
B. Report the capability gap, not perplexity. The stated reason is correct and should
be preserved: roughly 30 efficient-attention architectures won at 100M–1B and inverted
at 7B+ because they moved the scaling-law intercept and not the slope, and slope
claims require ≥3 scales at iso-FLOP with matched tuning budget.

*(That 30-architecture claim is the author's and is currently unverified. It is
load-bearing for the entire evaluation design and is on the verification list.)*

Secondary, and explicitly not primary: 1B-scale results on Turing-style, Lean, and
AGI-style benchmarks sufficient for the architecture to matter.

---

## 8. Named risks, stated before the room sees them

1. **Nested fixed points.** `(I − γP)⁻¹` is itself a Bellman fixed point, and the DEQ
   solves a second fixed point on top of it. No proof yet that the composition is a
   contraction. It may be redundant, or it may not converge.
2. **`ker Δ_F` may be trivial.** For a connected sheaf with generic restriction maps,
   the harmonic space is frequently `{0}`. An equilibrium defined as `ker Δ_F` that
   turns out to be zero carries no signal. Unverified.
3. **Physics→language transfer is assumed.** §2. This is the crux.
4. **Theorem 1's precondition is unstated in this setting.** §5.
5. **Stochastic `P` may destroy expressiveness.** Row-stochastic non-negative `P` is a
   much smaller hypothesis class than an unconstrained ridge-fit `A`. `sigmoid`'s
   measured numbers were obtained with the unconstrained version.
6. **Forward-only kernel.** `kernels#22` is forward-only. Training through it requires
   a backward pass that does not exist yet.

---

## 9. Verification list (nothing below is confirmed)

- Ramsauer et al. 2020 — softmax attention is one iteration of modern-Hopfield energy minimization
- Bai / Kolter / Koltun 2019 — DEQ, implicit differentiation through `(I − J_f)⁻¹`, O(1) memory
- Hansen & Gebhart — sheaf neural networks
- Bodnar et al. — neural sheaf diffusion, and whether `ker Δ_F` is generically trivial
- Dayan 1993 — successor representation `M = (I − γP)⁻¹`
- The ~30-architecture intercept-not-slope claim
