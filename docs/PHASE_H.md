# Phase H — bind

Phase H asks whether binding a word to an **operator**, so that composition is a
product rather than a sum, buys anything a trained vector model cannot reach.

This page opens with what the phase does **not** invent, because the fetch row
ran before any table was drawn and found that the central move is published.

---

## 1. Prior art, stated before any result

**Binding a verb to an operator is not new.** Coecke, Sadrzadeh and Clark (2010)
already do it, and compose by contraction rather than by addition. From §4.1 of
their paper:

> the meaning vector of each transitive verb can be thought of as a function
> that inputs a subject from V and an object from W and outputs a sentence in S

Adjectives-as-matrices is Baroni and Zamparelli (2010), whose abstract
"represents nouns as vectors and adjectives as data-induced (linear) functions
(encoded as matrices) over nominal vectors". That paper is **adjectives only** —
its §1 states it "addresses instead the combination of nouns and attributive
adjectives", and verbs appear only as future work. The extension to verbs as
matrices and transitive verbs as rank-3 tensors is **Grefenstette et al. (2013)**,
three years later. An earlier draft of this page's source notes attached the verb
claim to the 2010 name; that was wrong and is corrected here.

On the sequence-model side the gate landscape is likewise occupied. RetNet
carries a scalar decay, GLA and Mamba-2 a per-token decay, and DeltaNet a matrix
gate `I − β_t k_t k_tᵀ` — a non-commuting rank-1 reflection, which is to say
DeltaNet already sits *inside* the family a matrix-valued carry would define
rather than outside it as a rival.

**What is left.** Not the binding. The claim this phase can still test is about
**reachability**: whether an exact-zero gate yields a second absorbing state
where a softmax row cannot, and whether that property is decidable rather than
merely observed. DisCoCat and Baroni–Zamparelli say nothing about reachability.
Everything below is scoped to that, and any sentence here that reads as a novelty
claim about binding is a defect in this page.

---

## 2. What the fetch row struck

Five sources were checked against the sentences this phase wanted to write with
them. Three contradicted the intended claim, and the loudest unconditional claim
turned out to be in this project's own notes rather than in any paper.

**"Constant-depth attention computes iterated *commutative* products only" does
not follow from TC⁰ containment.** Containment of log-precision transformers in
uniform TC⁰ is an *upper bound on capability*; it places no restriction to
commutative operations, and a search for "commutat" in the source returns
nothing. Any statement of the containment must also carry the **log-precision**
qualifier, which the drafted sentence dropped.

**The "above TC⁰ via NC¹" leg is unsourced, not merely unconditional.** No
source fetched in this row mentions NC¹. The background fact is that TC⁰ ⊆ NC¹
and that whether the inclusion is strict is an open problem in circuit
complexity. No capability separation is claimed on this page.

**"SDPA cannot express a non-commuting recurrence at all" is struck.** It was
written into this project's own dispatch notes, and a source fetched in the same
row refutes it — Liu et al. (2022) §4 report

> positive results (> 99% in-distribution accuracy) for every finite-state
> semiautomaton we considered

including semiautomata generating the non-solvable groups A₅ and S₅. Transformers
do simulate non-commuting recurrences **in distribution**. The failure they
exhibit is **out of length**, and that — not a complexity separation — is the
unconditional, measurable claim this phase tests.

**Liu et al. is more positive than the drafted sentence implied.** Its primary
result is that shortcuts *are* learnable; the out-of-length brittleness is a
documented secondary limitation with proposed mitigations, not the paper's
headline.

**Delétang et al. (2022) does not support the prediction it was cited for.** It
measures training at length 40 and testing at 41–500, not L against 4L; it
reports categorical failure near chance rather than a 0.2 absolute drop; it
attributes failure to **positional encodings going out of distribution**, an
architectural cause, not to shortcut learning; and it contains counter-examples
of strong transformer generalization on permutation-invariant tasks. The paper
does not use the word "shortcut".

---

## 3. Fetch ledger

| source | reached | verdict on the drafted sentence |
|---|---|---|
| Coecke–Sadrzadeh–Clark 2010 | yes, §4.1 | binding is theirs |
| Baroni–Zamparelli 2010 | yes, PDF | adjectives only; verb claim misattributed |
| Grefenstette et al. 2013 | named | the actual source of verbs-as-tensors |
| Liu et al. 2022 | yes | supports, but tone-inverted in the draft |
| Delétang et al. 2022 | yes | contradicts the L-to-4L prediction |
| Merrill–Sabharwal 2023 | arXiv only | contradicts the commutativity inference |
| Barrington 1989; Krohn–Rhodes 1965 | already in tree, DOIs verified | carried |
| DeltaNet 2406.06484 §2.2; GLA 2312.06635 §4.1 | verified 2026-09-20 | carried forward, not refetched |

**Not reached, and nothing may be cited to them.** The Merrill–Sabharwal TACL
version (ACL server error — the arXiv version covers the content, but TACL page
numbers do not exist in this row). Mamba-2 (2405.21060) was never fetched at
all; the per-token-decay form quoted for it came from a *different* paper
describing it, a secondary standing in for a primary. Four quotes attributed to
arXiv:2210.02671 are lane-only and unverified. The Pāṇini kāraka row is
auth-gated and rests on nothing, so the claim that kāraka is the older form of
thematic roles is not made on this page.

---

## 4. What this phase measures

Scoped to what survives §1 and §2:

- **Order.** Vectors add and addition commutes, so "A then B" and "B then A"
  coincide to machine epsilon. Operators in different planes do not. The question
  is how much of that untrained separation survives training, since a trained
  vector model can allocate subspaces, use position, and memorise.
- **Length.** A scan of operators composes exactly at any length because
  composition is associative. That is arithmetic and needs no open conjecture. It
  is the phase's one unconditional claim.
- **Forgetting.** Two gate primitives and only two: invertible gates (rotations,
  permutations), which a prefix scan separates by composing with an inverse; and
  projectors, which are idempotent, rank-deficient and have no inverse. The
  intended claim was that an invertible-only family cannot forget, and that the
  separation is structural rather than a matter of optimisation. **It is not, and
  §6 records why** — an invertible near-projector solves the task exactly, and
  the separating property is contraction rather than singularity.
- **Decidability.** Restricted to a finite semigroup, "does it ever reach this
  state" is answered by eigenvalue-1 multiplicity and a resolvent residue rather
  than left undecidable. The claim is that finiteness escapes Rice's
  preconditions, not that Rice is wrong.

Every model-scoring row carries the standing precondition: **a bed whose null
scores nothing is void before the model runs.**


---

## 5. Three beds, three floors, and what that cost

Nothing in §4 has been measured yet, because every bed built to measure it
failed first — and all three failed the same way. The floor each used was chosen
for being easy to compute rather than for being the thing a passing arm must
beat.

**A kernel-carry bed was floored against chance.** Its floor was position-blind
and sat at chance, 0.125. But a model that *sees the context and cannot bind*
scores **0.3460** on that generator, by guessing uniformly among the values
present. The arms read 0.401–0.414, so they cleared the real floor by 0.055–0.068
and recovered about 10% of the 0.346→1.0 headroom. The reported 0.125→0.41
separation was almost entirely "this model mixes tokens at all". Tripling the
training budget moved nothing outside one seed spread, and a two-layer variant
landed in the same place: the plateau was the task, not the budget.

**The Krohn–Rhodes bed shipped no null at all.** Its `floors_and_null()`
contains no shuffle, and its own structured return reports `null_scores=false`.
Under the standing precondition that is a halt, so the bed is void before any
model runs.

**The planted-operator text bed cannot see order, and the reason is geometric.**

| baseline | L2 error |
|---|---|
| bag-of-context | 4.9938 |
| last-operator-only | 5.0922 |
| order-free composition | **5.5367** |
| chance | 5.7160 |

Applying the correct operators in the **wrong order** scores worse than a bag of
context and within 3.1% of chance. The planted operators are norm-preserving
SO(8) rotations, so a permuted composition lands uniformly on the sphere however
wrong the permutation is; the true state lies on that same sphere; and the
expected L2 distance between two uniform points on a sphere is a constant.
Order-free therefore scores at chance **by geometry**, not by failing to know the
order. Bag-of-context wins only because averaging shrinks a vector toward the
origin, a norm artefact unrelated to composition.

Both trained arms on that bed scored **worse than chance** — 6.3206 and 6.7240
against 5.7160 — which is the signature of a metric that cannot see the property
under test, not a result about binding.

**Consequence for the kill list.** The rule "(a) ties (d) on this bed →
operator binding is dead" presupposes the bed measures something. It does not, so
that kill does not fire, and no number from those twins carries information about
binding either way.

**What order-sensitivity does have.** The Barrington bed is certified: its oracle
was rebuilt by independent array-index permutation composition over 2,000 pairs,
both matrix conventions confirmed as exact homomorphisms, and a BFS closure over
the two generators reaches exactly 120 elements, so the scan has real depth
rather than being a lookup. A scan of 5×5 permutation operators is bitwise exact
at length 10,000 in float64, float32 **and bfloat16** — every surviving entry of
a permutation product is exactly 0 or 1, so no dtype has anything to round. Of
the C(120,2) = 7,140 unordered generator pairs, 6,780 do not commute.

S₅ is non-abelian and its word problem is order-defined by construction, so order
is demonstrable there. What remains open is whether a *text-shaped* bed can ever
show it, which is a question about the corpus rather than about the operator.

---

## 6. Three claims that did not survive their own audit

**The gate class is keyed on the wrong quantity.** The intended division is
*group* (invertible) against *reset* (projector, rank-deficient, `det = 0`). An
invertible near-projector `R_ε = (1−ε)P + εI` solves parity-since-reset with
**exactly zero label error for every ε < 1/2**, measured down to `ε = 1e−15`
(`det = 1e−15`) at length 50,000 with about 5,000 resets. The worst state
entering a reset is `(0,1)`; `R_ε` sends it to `(1−ε, ε)`; an argmax decode is
correct whenever `1−ε > ε`, and the residual is re-multiplied by ε at every reset
so it never accumulates. Meanwhile an orthogonal substitute with `|det| = 1`
stays at 0.4150 error.

So `det = 0` is **sufficient but not necessary**, and the separating property is
contraction rather than singularity. A load-time gate check that rejects on
"non-idempotent and non-invertible" would admit the wrong objects and reject the
right ones. The earlier structural claim came from measuring an L2 norm on
isolated prefixes and never decoding a task label.

**The line gate is a length filter, not a noise filter.** It does separate real
English from random letters — the fraction of random samples above the threshold
is 0.0000 at every length from 50 to 150,000 characters, against 1.63–1.72 for
real English. But its reject set is monotone in line length and nothing else:
41.72% of lines under 40 letters are rejected, 4.71% at 80–100, 0.12% at 150–250,
and **0.00% above 250**. Median reject 51 letters, median keep 126.

The index of coincidence is a function of the letter multiset alone, so the gate
is blind to word order, spelling and grammar by construction. Shuffling the
letters within each of the first 1,000 lines produces **75 rejects against 75**
for the unshuffled lines — identical, not approximately. A control of real corpus
words in random order passes the gate 93.6% of the time. A filter whose stated
job is to reject noise admits word salad nine times in ten and rejects two in
five genuine short sentences.

Separately, the two published index-of-coincidence figures, 1.46 and 1.73, are
the same convention — strip to 26 letters, fold case, normalise by 26 — at two
sample lengths, and that is the only convention whose random baseline is 1.00.
Three alternative conventions were tested on two corpora and all read *higher*,
never lower, so "a different convention reads lower" is ruled out.

**A recall bar that measures nothing.** Verb-centre recall against a parser
oracle reads 1.0000 with a 95% Wilson interval of [0.9922, 1.0] at n = 489,
because the system under test and the oracle are the same parser call on the same
string, run twice and compared. It is an identity wearing error bars. The
replacement is measured and available: agreement against an independent
part-of-speech tagger that never sees the parser's output reads **0.8262, 95% CI
[0.7901, 0.8572]**, and a threshold belongs on that distribution.

---

## 7. What the Lean corpus gained

Three results landed in `lean/CEQ/V16Domain.lean`, and the accounting is stated
rather than implied: the section adds **19 declarations and 3 citable new
facts**, and two of the three named targets carry none of their own.

**A scalar gate composed over a finite ordered list is invariant under permuting
that list.** This is Mathlib's `List.Perm.prod_eq` specialised to a commutative
monoid; it carries no new mathematics and is present only as the contrast term.

**A matrix gate is not**, witnessed concretely by two nilpotent 2×2 matrices
whose products differ at entry (0,0). That is the pair's only new fact, and the
pair is the claim — commutativity alone is a library lemma, while commutativity
set against non-commutativity is the architectural statement.

**Reachability in a finite semigroup is decidable**, and now as a `def` rather
than a `theorem`, so the `Decidable` instance carries executable code. The
distinction is not cosmetic: declared as a theorem on a Type-valued goal it
proved a decision procedure *exists* while being unable to *run* one. The two
evaluations print `true` and `false` during both `lake env lean` and
`lake build`, and an instance backed by classical choice would not reduce to
literals at all. Finiteness is load-bearing — the reachability predicate carries
no `Fintype` binder of its own, and the pigeonhole step is the single place the
cardinality is consumed.

The List-shaped commutation statements are declared as **separate combinatorial
facts that do not connect to `pathProd`**, and the reason is worth recording:
`pathProd` is a `Finset.prod`, and `Finset.prod` requires a `CommMonoid` by
definition. It cannot typecheck over a non-commuting codomain at all, and
permutation-invariance over a `Finset` is vacuous because there is nothing to
permute. The operator as currently defined in the corpus therefore cannot express
a matrix gate; that is a fact about the definition, not about the mathematics.

---

## 8. H.1 — the row that measured something

A bed rebuilt to demand order finally separated the arms, and the separation
survived a matched budget. Discrete-label accuracy over eight fixed anchors,
5 seeds, same instances for every arm.

| arm | params | accuracy | std | headroom over tightest floor |
|---|---|---|---|---|
| (a) vectors + softmax + RoPE | 620 | 0.2238 | 0.0115 | **−0.1328** |
| (b) scalar gate `m·e^{iθ}` | 620 | 0.2628 | 0.0148 | **−0.0938** |
| (c) operator gate, invertible | 552 | **0.9570** | 0.0048 | **+0.6004** |
| (d) operator gate + projector | 654 | 0.9470 | 0.0059 | +0.5904 |

Tightest floor is `last_two_ops = 0.3566` on that eval split. Every figure in this table is
produced by the chain in `tests/foreman/h1/`, which runs with the six commands
listed in its README; until that directory existed these numbers appeared in
this page and nowhere else, and a reader could not re-run them. `(c) − (a) = 0.7332`
against a pooled per-seed standard deviation of `0.01250` — **59 σ**, with every
(c) seed beating every (a) seed by at least 0.708. Complete separation at n = 5.

**The arms share instances, not merely a distribution.** Both lanes use the same
population offsets, and the per-seed floors recomputed independently in each lane
agree bitwise — `0.3900 / 0.3600 / 0.3190 / 0.3640 / 0.3500`. That agreement is
what proves the split is identical. The parameter residue (620 against 552 and
654) favours the *losing* arms, so it cannot explain the gap.

Ratios are taken on excess over the majority-label baseline of 0.1640, the only
oracle-free floor: (c)/(a) = **13.3×**, (c)/(b) = **8.0×**. Excess over the
tightest floor admits no ratio at all, because (c) is +0.6004 and (a) is −0.1328 —
opposite signs. That is the honest phrasing, and it is not the same as saying the
vector side has no signal.

The 653× figure recorded for the untrained comparison must never appear beside
these. It is in consequence error, an L2 distance; these are accuracies. And that
figure was inflated in any case: its vector leg computed a constant mean
displacement over zero-mean Gaussian input, which is approximately zero for a
rotation, so `4.0717` is `√(2d)` and the leg is the always-identity null. It
compared a least-squares fit against a do-nothing predictor.

### What the gap is actually attributable to

The four arms do **not** differ only in the composition primitive. (a) and (b) are
a single attention-pooling layer with no running state; (c) and (d) are a
three-step recurrence. Matching the budget did not touch that, and an earlier
draft of this row claimed otherwise.

That confound was measured rather than conceded. A vector arm was given the
identical structural gift — same three-step recurrence, same additive injection,
same readout — with the per-verb matrix replaced by a per-verb **diagonal** gate,
which commutes by construction and therefore cannot represent order. A live
parameter search put it at **554**, a tighter match to (c)'s 552 than the 620 the
(a)/(b) lane could reach. Same bed, offsets, seeds, split and optimiser.

It scores **0.3080** at 400 steps and **0.3342 ± 0.0161** at 4000.

So the recurrence, handed free to a vector arm at matched parameters, buys
**+0.1104 — 15.1% of the 0.7332 gap** — and still lands *below* the tightest
floor. The remaining **84.9% requires the non-commutative matrix.**

### The limit this bed cannot remove

The operator arms' forward pass is `state = ops[verb] @ (state + emb[patient])`,
which is character-for-character the bed's own generative recursion. So 0.9570
measures that the generator's own form fits its own generator. What this bed
cannot separate is *non-commutative operator* from *the generator's exact operator
family*, because on this bed they are the same thing by construction.

An earlier draft of this row called the floors "trivial closed-form" baselines.
That is wrong and understates them: `last_op_only`, `last_two_ops`, the
order-free replay and the bag predictor are all handed the true entities and the
true operators. They are oracle-assisted, which makes clearing them harder rather
than easier.

---

## 9. H.1 — what the census extension turned out to be

The stability zone reproduces exactly, and it says nothing checkable about the
head.

**The zone.** The recursion `c ← e^{ac}` has fixed point `c = −W₀(−a)/a` and is
attracting exactly when `|W₀(−a)| < 1`. The boundary of the attracting region in
the parameter plane is the analytic curve `a(t) = −e^{it}·exp(e^{it})`, with
`|a(t)| = exp(cos t)` and `arg a(t) = t + sin t + π`. Its ray radii are

| φ | 0 | π/4 | π/2 | 3π/4 | π |
|---|---|---|---|---|---|
| r(φ) | 0.3678794411714423 | 1.214461 | 1.961309 | 2.513997 | 2.718281828459045 |

with both endpoints exact to machine precision at `1/e` and `e`. An earlier
empirical ray scan read `0.3669` and `2.7178`; the gaps of `0.000979` and
`0.000482` are both at or under that scan's own grid step of `0.0009970`, so the
discrepancy is resolution and not a convention error. The substitution `z = a·c`
turns the recursion into `z ← a·e^z` with a maximum floating-point difference of
**exactly 0.0** over twenty trajectories, so the census map *is* the exponential
family, bit for bit.

Dimension 1. Not fractal. A prediction of a non-integer box dimension was
withdrawn before measurement because it contradicts a theorem: the Julia set of
`λe^z` has Hausdorff dimension 2 for every `λ`. That set lives in the *dynamical*
plane; the census lives in the *parameter* plane, and the two had been conflated.
Basin boundaries of periodic attractors can indeed have dimension between 1 and 2,
which is a different object again.

**One stated behaviour of the zone is false off the real axis.** "Converges at
0.95r and diverges at 1.05r on every ray" holds only at φ = 0. At π/4, π/2, 3π/4
and π the fixed point *is* locally unstable at 1.05r exactly as theory predicts,
with multiplier ≈ 1.0245 on every ray — but the orbit is then captured by a
bounded attracting cycle of period 7, 16, 15 and 2 respectively, verified to
200,000 iterations without escape. Any test asserting literal divergence at 1.05r
will misfire on non-real phases.

**And the reduction to the head fails, established twice by different routes.**

The first route found there is no quantity to reduce. Under the gate route the
modulus row is bitwise phase-independent — the gate phase reaches the complex gate
and never the real score, which extends an earlier finding from a constant phase
to a positional ramp. Under the logit route the row does move with phase, but the
whole operator output is then exactly real, with maximum imaginary part `0.0`
across all twenty cells, so no complex quantity exists there at all. And the
normalizer is computed before `β` is ever applied, so the `(1−β)` factor in the
mapping has no counterpart in the per-position dynamics. By Cauchy–Schwarz the
normalizer is bounded by `(i+1)·e^{w_max}` — linear in position, never
exponential in itself.

The second route measured the closest candidate directly. The row-sum tracks
`c* = −W₀(−a)/a` within 5% only at `a = 0.10`, and misses by +7.6% to +8.8% at
0.20, +19.6% to +22.4% at 0.30, and +37.6% to +49.1% at 0.36 — 15 of 20 cells
fail. The cause is derived rather than observed: a single static application of the
operator has **no self-referential feedback loop**, so its row-sum tracks `e^a`,
smooth and singularity-free, while `c*` has an essential branch point at `a = 1/e`
*only because* the census recursion feeds `c` back into itself. The two agree to
first order in `a` — both approximately `1 + a` — which is exactly why the error is
small at 0.10 and diverges toward the fold.

So the Lambert bifurcation is exactly right about the scalar map and unconnected
to this operator. What survives is the real-axis statement: the fold line restated
as `|W(−a)| < 1` is exact and is a genuine sharpening, whether or not the complex
part reaches anything.

---

## 10. H.1 — what the transformer autopsy returned

**Identification first.** "JEV" does not resolve to a JEPA derivative or to a
decision-making module. It resolves to a separate commercial product launched in
September 2026: a non-autoregressive, schema-constrained classifier emitting one
structured output and a calibrated confidence per query. It contains no
state-transition model, no reward, no rollout and no value function, so it is not
prior art for the resolvent read, and it should not be named beside JEPA.

Nothing in it is liftable here. Its schema-constrained decoding solves invalid
tool-call output, a problem this project does not have, and is a weaker relative
of the exact-zero gate already killed as a masking trick.

**But it names the sharpest absence in the project.** Calibration as an
objective — training predicted confidence so that `P(correct | confidence = c) ≈ c`
against the real hit rate — is a property this project has no mechanism to produce
and has never checked. Exactness and calibration are orthogonal: being bit-identical
to a reference says nothing about whether a stated confidence tracks an empirical
rate. Only the former is covered anywhere here. It is not attachable today, because
there is no labelled decision stream to train against, and that is precisely why it
is worth recording.

**JEPA, organ by organ.** One organ is already resident: the EMA target encoder
with a stop-gradient is implemented and tested in `ceqjepa/pi_jepa.py`, at
`tau = 0.99`, and costs nothing to adopt because it is adopted. The spatial
multi-block masking is not applicable — it solves occlusion of 2D and 3D regions,
while these beds are 1D sequences whose context/target split is a temporal horizon
shift.

Prediction in representation space is a **gap for the kernel**, and this
repository's own code is the proof: `pi_jepa.py` already implements that objective
around a plain three-layer MLP encoder, never around the operator, which appears
only as the predictor. A kernel cannot contain an objective, and no setting of
β, g or qk reaches it.

**One correction to this project's own objection.** The anisotropy argument — that
clustering *is* anisotropy, so an effective-rank read prefers a frozen-random
encoder at every width — was aimed at JEPA's collapse machinery. Nothing in the
fetched papers shows them validating non-collapse by a bare post-hoc geometric
read, so the objection targets a validation method they may not use. The argument
stands against the instrument this project built; its reach into that literature is
unestablished.

### The encoder route is probably not where this belongs

`pi_jepa.py`'s encoder is a pointwise MLP with **no sequence mixing at all**,
despite its own docstring describing otherwise. A prior round already built the
repair and benchmarked it — a causal attention block lifts R² from `−0.19` to
`0.986` on this project's own bar — but it uses stock scaled dot-product attention
and never the operator.

Two structural obstacles stand between the operator and an encoder slot. Its
causal mask is not a setting but a structure: no bidirectional path exists, and a
JEPA encoder is bidirectional. And its positional mechanism is content-conditioned
rather than index-conditioned, so it is unverified to break permutation symmetry
the way a sinusoid does.

The sceptical finding is the one that matters, and it is stated rather than buried:
**none of the three surviving certificates — exact refusal, decidable NEVER, a
certified resolvent — is a property a representation-learning loss or a downstream
probe can query.** If that holds, the encoder slot is not where this contribution
belongs, and knowing that is worth more than an interface nobody would use.

---

## 11. H.1 — three repairs, one of which is progress

Three rows struck earlier in the phase were rebuilt. One became a real row; two
became struck rows with better prose, and are recorded as that.

### The Krohn–Rhodes bed is alive

It had shipped no null at all. It now carries two, both scored on the task's own
argmax label metric rather than on finiteness, at n = 2000 per seed:

| null | per-seed error | correct construction |
|---|---|---|
| shuffled operator assignment | 0.5045 / 0.4970 / 0.5175 / 0.5170 / 0.5160 | 0.0000 |
| orthogonal reset substitute, `det = 1` | 0.5445 / 0.5465 / 0.5445 / 0.5420 / 0.4645 | 0.0000 |

The second null is the one that matters, because the property under test turned
out to be contraction rather than singularity, and a *near-singular* invertible
matrix would pass the task and prove nothing. To rule out a lucky draw the entire
2×2 orthogonal family was swept — 721 angles, both determinant signs, singular
values exactly `[1, 1]` — and the best member still errs between **0.393 and
0.544** on every seed.

The contraction reading is confirmed from the other side too: `R_ε = (1−ε)P + εI`
scores **exactly zero label error at every ε tested**, from 0.4999 down to
`1e−15`. Against the same sequences, an orthogonal substitute errs 0.4175–0.572
while the contracting one errs 0.0.

One sentence of this project's own correction was itself incomplete. The failing
pairs under the pseudo-inverse prefix form require **both indices past the
projector**, not one — a pseudo-inverse cannot undo a rank-one projection at
either end.

### The line gate is still a length filter

The replacement is a two-level model — a character trigram and a word bigram,
each with its own placebo — and it passes both placebo tests that the original
failed. Letter-shuffle passes the character level **0.00%** of the time;
word-salad passes the word level **2.40%**, both under the 5% void line.

It also reproduces the expected division of labour at real corpus size: the
character level kills letter-shuffle but is **68.5% blind to salad**, while the
word level kills salad and kills letter-shuffle as well through unknown tokens.

But the confound that struck the original is **worse**, not better:

| line length | rejected |
|---|---|
| under 40 chars | **60.0%** |
| 40–100 | 18.78% |
| 100–250 | 2.7% |
| over 250 | **0.0%** |

A spread of 0.60 against the struck gate's 0.4172. The mechanism is genuinely
different — this gate is order-sensitive where an index of coincidence is a
letter-multiset statistic — but the symptom is identical and larger. It is a
better-behaved length filter. The within-bucket control that would settle whether
anything else is being measured was not run.

### The recall bar clears its floors and still measures the wrong thing

Cross-tagger agreement reproduces exactly: **0.8262** (404/489), Wilson 95%
interval **[0.7901, 0.8572]**, between a dependency parser's root verb and an
independent part-of-speech tagger's first verb-tagged token.

It clears four no-parsing floors without interval overlap — the strongest being a
first `-ing`/`-ed` heuristic at 0.3579 [0.3166, 0.4013], then longest word at
0.0716, first alphabetic token at 0.0102, sentence-final token at 0.0041. The
system beats the strongest by 0.4683, and 0.4013 < 0.7901.

The old bar of ≥ 0.95 is unreachable against an independent tagger and was only
ever met by the tautology it replaced. A defensible bar is the measurement's own
lower bound, 0.7901.

What it cannot do is the thing a bar is for. Two imperfect taggers disagree on
**17.4%** of lines when nothing is broken at all, driven by participles used as
adjectives, copulas and fronted adjuncts. So a below-bar reading cannot separate
an extraction bug from ordinary tagger ambiguity: it measures concordance, not
correctness. It is an honest number and a weak bar.

---

## 12. H.1 — two certificates, and the end of the exactness claim

Both certificates pass, and neither is a result. They are theorems with checks,
and the page says so rather than counting them.

**The dilation identity holds with four orders of slack.** Across r ∈ {2, 4, 8} at
five seeds, with σ = 0 and σ = 1 included deliberately rather than left to a
random draw, the doubled-space operator is orthogonal to a maximum of
`2.220446049250313e-16` against a bar of `1e−12`, its top-left block equals Σ
**bitwise** in every run, and the reconstruction error is `0.0` exactly.

Two things the hunt for a failure turned up. An earlier draft claimed the
angle route and the direct-product route diverge at σ = 1; they diverge at
**σ = 0**, because `arccos(0) = π/2` and `cos(π/2) = 6.123233995736766e-17`, not
zero — σ = 1 round-trips exactly. The implementation this repository ships is the
direct-product route, which never calls either function and is therefore exact at
both corners. And the naive `√(1−σ²)` loses against the stable
`√((1−σ)(1+σ))` by a relative `2.756e−10` at σ = 0.99999999, but orthogonality
never degrades past `1.1e−16` there: the precision loss lives in a scalar erasure
formula, not in the defining identity.

The attribution is Halmos 1950 for the block form and Sz.-Nagy 1953 for power
dilations. Stinespring 1955 concerns completely positive maps on C\*-algebras and
is not the citation. The parallel with a groups-and-flip-flops decomposition over
finite semigroups is a resemblance between different mathematical settings and is
not claimed as a theorem.

**And the last named structural difference is dead on the bed built to show it.**

The resolvent read is the successor representation applied to a reward, which is
prior art, and it agrees with the temporal-difference fixed point to
`8.882e−15` at worst across five seeds against a `1e−12` bar, with successor-
representation row sums exactly `10.000000`.

The only difference this project still claimed for it was exactness at absorbing
states — a gate reaching exactly zero where a sigmoid only approaches it. On a
six-state chain with one absorbing state:

| route | value at the absorbing state |
|---|---|
| resolvent read | `0.0` exact |
| TD(0) after 20,000 sweeps | `2.470e−323` |
| sigmoid gate after 20,000 steps | `0.03817905137506084` |

The gap between the exact read and TD(0) is a **subnormal float64 value**. No
tolerance anyone would set can register it.

The sigmoid's `0.038` is not a rescue, and the measurement says why: it comes from
gradient starvation as the argument approaches zero, which is a fact about
gradient descent on a sigmoid rather than about value estimation. It is not the
mechanism being contrasted.

So exactness at absorbing states buys nothing measurable against the method it was
supposed to beat. That is consistent with this project's own earlier concession
that the closed-versus-open distinction is a parameterisation default rather than
a novelty; it is now measured rather than conceded.
---

## Limits

No result tables appear on this page yet; the rows that would fill them were
running when it was written, and it exists so that the prior-art concession is
fixed before any number can tempt an adjustment to it. The unreached sources in
§3 are unreached, and their absence is not evidence either way. The reachability
claim that §1 leaves standing carries this project's own existing concession that
the exact zero is proved for a *constructed* gate and remains untested for a
*trained* one.
