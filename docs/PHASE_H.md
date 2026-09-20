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

## Limits

No result tables appear on this page yet; the rows that would fill them were
running when it was written, and it exists so that the prior-art concession is
fixed before any number can tempt an adjustment to it. The unreached sources in
§3 are unreached, and their absence is not evidence either way. The reachability
claim that §1 leaves standing carries this project's own existing concession that
the exact zero is proved for a *constructed* gate and remains untested for a
*trained* one.
