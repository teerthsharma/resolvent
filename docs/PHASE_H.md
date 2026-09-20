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
  projectors, which are idempotent, rank-deficient and have no inverse. An
  invertible-only family cannot forget, and the separation is structural rather
  than a matter of optimisation.
- **Decidability.** Restricted to a finite semigroup, "does it ever reach this
  state" is answered by eigenvalue-1 multiplicity and a resolvent residue rather
  than left undecidable. The claim is that finiteness escapes Rice's
  preconditions, not that Rice is wrong.

Every model-scoring row carries the standing precondition: **a bed whose null
scores nothing is void before the model runs.**

---

## Limits

No result tables appear on this page yet; the rows that would fill them were
running when it was written, and it exists so that the prior-art concession is
fixed before any number can tempt an adjustment to it. The unreached sources in
§3 are unreached, and their absence is not evidence either way. The reachability
claim that §1 leaves standing carries this project's own existing concession that
the exact zero is proved for a *constructed* gate and remains untested for a
*trained* one.
