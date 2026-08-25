# The module — six behavioral requirements

This supersedes `THEORY.md`. That document specified an architecture by naming four
formulas; the room killed all four. This one specifies *behavior* and lets the mechanism
be whatever survives its own falsification test.

## Standing rules

**Inspiration, not dependency.** Prior repositories on `github.com/teerthsharma` are read
for ideas and for measured failure modes worth not repeating. The module imports none of
them. It is a standalone artifact with its own kernel.

**A real kernel.** Not a schedule builder feeding somebody else's kernel. The mechanism is
implemented in the kernel.

**Every test runs on CPU and GPU.** Parametrize over device. A test that only runs on CUDA
cannot be reproduced by a reader without an A100, and a CPU reference implementation is
the only honest parity oracle for a GPU kernel anyway.

**C1 — isolation, cheapest first.** Each requirement is tested alone, before any
end-to-end benchmark. No requirement is justified by the performance of the whole.

**C2 — a failed test deletes its requirement.** Not defends it, not weakens it, not
"needs more tuning". Six mechanisms is the ceiling and fewer is better.

**C3 — perplexity is not the win condition.** Intercept wins invert at scale. Win = the
capability tests below, plus scaling-law slope at ≥3 sizes, iso-FLOP, matched tuning
budget. *(Note: the "~30 architectures inverted at 7B" claim is unverified — Wilson found
no such study. The nearest real one is Tay et al., arXiv:2207.10551, ten architectures at
2.9B dense, no intercept-vs-slope decomposition. Cite that or say nothing.)*

**C4 — best-of-N must not match it.** An equal-compute sampling baseline that reaches the
same gains makes the module unnecessary.

**C5 — ≥30% accelerator utilization.** An elegant module at 5% is dead.

**C6 — honest cost.** Wall-clock and memory against a *tuned* attention baseline, not
quality deltas alone.

---

## R1 — Settle, don't glance

The module must not emit its first reading of the context. It revises until the
interpretation stops changing, and outputs the stable reading.

**Falsifier:** iterate on held input. Output must change meaningfully across early
iterations and stop changing at convergence. **If pass 1 already equals the settled state,
R1 is dead weight — delete it.**

*Known hazard, measured:* a fixed-point solver can lose its fixed point silently. In this
room a DEQ blew its 30-NFE budget at step 47 and trained 922 further steps with loss
falling monotonically while ρ(J) went 0.4942 → 1.0086. A loss curve is not an alarm.
Log the residual and ρ per step, checkpoint on residual.

*Known hazard, measured:* if the update is affine, the fixed point has a closed form and
the iteration computes nothing (residual 1.05e-15). R1 is only alive if the update is
genuinely non-affine. Include the collapse probe: best affine fit to the learned update at
the settled state.

## R2 — Weigh by consequence, not similarity

Influence must reflect what a token causes downstream, not what it resembles. Small,
dissimilar, high-consequence tokens — `not`, a bound, a flag — must outrank large,
similar, inert ones.

**Falsifier:** train where ground-truth consequences exist — interventions with measured
outcomes. Must generalize under intervention where a similarity baseline fails. **If the
learned weighting collapses back to similarity, it is a reparameterization — delete it.**

*Domain, measured:* executed code is an interventional domain with a free oracle. An
interventional corpus recovers `dy/da = −1` where the confounded observational corpus
gives `+0.63` — a sign flip. Physics simulation is **not** a usable source: a physics
prior measured 0.3667 against a sham prior with its action column permuted at 0.3654 and
matched-λ scratch at 0.3654. Indistinguishable from noise.

*Control that must be beaten:* APPNP / personalized PageRank, `Z = α(I − (1−α)Â)⁻¹H`. A
linear resolvent stage reproduces it to 2.22e-16, so beating vanilla attention proves
nothing.

## R3 — Amplify few, crush many

A few tokens must be able to flip the output. Filler, paraphrase, and typo noise must be
actively forgotten — perturbing them leaves the settled state unchanged.

**Falsifier:** perturbation ratio = (Δoutput from editing a causal token) / (Δoutput from
editing a filler token). Must be large, and a standard-attention baseline must be flatter.
**Global sensitivity fails. Global deadness fails.**

*Known hazard, measured:* a row-stochastic operator cannot do this. Its resolvent confines
output to `[min(b), max(b)]/(1−γ)` in 600/600 draws and cannot amplify any mode above the
mean — best non-constant/constant gain ratio 1.000000 over 1,920 kernels. Horizon and
contrast become reciprocals: one dial. Whatever operator is used, it must be able to
amplify a chosen mode while still contracting.

## R4 — Compress by forgetting

Per-token memory shrinks because the module *structurally cannot* retain what does not
matter — not because a pruning heuristic was bolted on.

**Falsifier:** effective size of the settled state across many contexts. **Near full width
means no compression story.** Task-relevant reconstruction must survive; reconstruction of
crushed noise must not.

*Draw inspiration from three measured places, import none:*
- **caustic's stochastic resonance** — added noise *improved* a certified floor by +0.333.
  Noise as a resource rather than a nuisance is the relevant idea.
- **Epsilon-Hollow's foliation** — memory as a quotient by an equivalence relation, where
  a block's refcount *is* the cardinality of a fibre and eviction is elementary collapse
  of a free face. Forgetting as a structural property of the data structure, not a policy
  on top of it.
- **NeMo-Relay** — execution-scope lifecycle as the model for what a runtime is entitled
  to drop and when.

## R5 — Multi-zoom reading

Fine resolution nearby, coarse summaries far away, in one self-similar scheme. Cost grows
near-linearly with context, with a **stated accuracy bound** for the coarsening.

**Falsifier:** long-context recall at matched quality with measured near-linear scaling.
**Quadratic cost fails. Unbounded coarsening error fails.**

*Known hazard, measured:* the obvious construction loses. A hierarchical dyadic schedule
scored 0.39 on the random→oracle axis against 0.975 for plain sliding-window + attention
sinks, and beat it in 0 of 15 cells. The builder cost 53× the attention it scheduled
(42.12 ms vs 0.792 ms at seq 8192) and received no gradient at all — top-k is an argsort.
The structural reason: an H-matrix far-field is **low-rank, not sparse**, and a block
schedule can only keep or drop. Any multi-zoom scheme that only drops has already lost;
the coarse level must *summarize*, not omit.

*Known trap, measured:* `BlockMask.from_kv_blocks` with raw CSR arrays computes fully
dense attention while `to_dense()` reports the sparse pattern — output was `0.000e+00`
from unmasked, bitwise identical, causality dropped too. Sparsity must be verified by
output parity against an independent reference, never by inspecting the mask.

## R6 — Local readings, global agreement

Each region is interpreted on its own terms; all local readings are then forced to
reconcile. Cross-region contradictions surface as reconciliation failure rather than being
averaged away.

**Falsifier:** plant a contradiction between two regions. The module must flag or resist it
measurably more than an averaging baseline.

*Known hazard, measured:* the sheaf-Laplacian version of this is dead. `dim ker Δ_F =
d mod 2` on a connected graph with a cycle and generic SO(d) restriction maps — exactly 0
in all 80 draws at d = 2, 4, 6, 8. It carries signal only on an acyclic base, i.e. only
where there is nothing to reconcile. If R6 survives it survives by a different mechanism.

---

## Test order (C1: cheapest first)

| order | requirement | cost | needs training? |
|---|---|---|---|
| 1 | R1 collapse probe | minutes, CPU | no |
| 2 | R3 perturbation ratio | minutes, CPU | no |
| 3 | R6 planted contradiction | minutes, CPU | no |
| 4 | R4 effective-rank sweep | minutes, CPU | no |
| 5 | R5 scaling + parity | hours, CPU+GPU | no |
| 6 | R2 intervention generalization | ~1 day, CPU | yes |

R2 is last because it is the only one needing a training run, not because it is least
important — it is the most important, and everything before it is a cheap way to avoid
paying for it on a mechanism already dead.
