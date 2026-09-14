# BED-H — the belief bed. Pre-registered 2026-09-14, before any code exists.

Hidden intents, observed statements, exact labels. This document is written
**before** the bed is built and before any arm is run, so that the prediction and
the counter-prediction below cannot move afterwards. Every later result on this
bed is scored against this page.

## Why this bed exists

Three independent measurements on 2026-09-14 killed the same thing from three
directions, and none of them was a bug:

- The committor at ply 15 is a function of the ply-15 **position**, so the walk
  is Markov and the prefix is conditionally independent of the label.
  Aggregation **costs** 0.0810 at the encoder output. Moving the label to ply 19
  does not rescue it.
- K+Q-vs-K is a fully observed MDP, so conditioning on a move and forcing it are
  the same operation and the interventional–observational divergence is
  identically zero. That is the diagnosis of DCM-1's flat move-permutation
  ablation, not a fault in it.
- `CausalEncoder` PASSED on an i.i.d. EMA bed and its sign **flipped** on the
  real target.

On a fully observed Markov chain the future depends only on the present, so an
attention operator over a context has provably nothing to do. Every bed this
project owned was Markov in the current state. The method also depends on an
**exact oracle** — the absence of one is what killed DCM-1, whose bed carried
negative headroom nobody measured first. BED-H is the first bed that makes
history necessary *and* keeps the label exact.

## The bed

Hidden chain `x_t` over `S = 8` states with a **non-symmetric** `P`. Emissions
`y_t` at `snr ∈ [0.5, 0.7]`. Second eigenvalue `λ₂ ∈ [0.90, 0.97]`, giving a
memory of roughly 10–30 steps — deliberately 2–3× the baseline's window, so a
windowed reader is beaten by the physics rather than by tuning.

**Oracle:** the forward algorithm. Exact, `O(T·S²)`.

**Labels.** `b_t`, the belief, is **intensive** — the softmax corner.
Cumulative expected hits `Σ_τ b_τ·h` is **extensive** — the linear corner.
Consequences are absorption probabilities under the belief.

**This is the first bed on which the corner rule has a referent.** The rule
`beta = 1 − alpha` was retired on 2026-09-14 (`docs/CORNER_RULE_RETIREMENT.md`)
on three grounds, the first of which was that `pi_jepa`'s encoder is
position-wise, so its scaling exponent is zero by construction and the mask
reached its coordinates by index equality alone. Here the intensive and
extensive classes are planted by the generator's physics, not by an integer.

## Must-fires — gates, not decoration

Printed per bed instance, and an instance outside the band is **REFUSED**:

1. **History must be necessary.** Window-`L` oracle error must **decay** with
   `L`. A bed where it does not is a Markov bed wearing a hidden state, and it
   cannot reward the thing under test.
2. **The label must be alive.** Label spread `≥ 0.1`.

## The trap

**Never hand the model `P`.**

- *Weak form:* `P` fixed and unknown, learned implicitly.
- *Strong form:* `P` drawn per instance and encoded in the prefix — in-context
  system identification. That is the real target.

## L-SIMPLE baselines, in order

1. **Window-`L` forward algorithm** — the oracle with amnesia, exact inside its
   window. The arm must beat **window-4's 0.16–0.22**.
2. **An EMA of one-hot observations.**
3. **A GRU with `S` hidden units** — the native primitive. This is the
   **SKYLINE, not a rival**: a belief update is a first-order recurrence in
   belief space, so a recurrence is the right tool and beating it is not the
   claim.

## Prior art — REWRITTEN 2026-09-14 after an audit, before any arm ran

The section this replaces named two references and conceded the wrong half of
one of them. An audit resolved 37 identifiers and found that **three of the four
things this bed would claim are already occupied, with numbers**. The
corrections are recorded here rather than in a later card, because the whole
point of a pre-registration is that it is fixed before the result.

### The substrate is published, twice

- **arXiv:2512.22471** — *The Bayesian Geometry of Transformer Attention*.
  "Bayesian wind tunnels — controlled environments where the true posterior is
  known in closed form." Its HMM task draws a **fresh Dirichlet HMM per
  sequence**, supervises on `p(s_t | o_1:t)` with forward-recursion ground
  truth, and scores architectures: **Transformer 0.049, Mamba 0.024 ± .009,
  LSTM 0.411 ± .003 (marked fail), MLP 0.40** in entropy MAE. That is this
  bed's strong form, already executed.
- **arXiv:2605.20824** — *Markovian Circuit Tracing for Transformer State
  Dynamic*. Calls itself "a controlled benchmark and evaluation framework":
  six HMM families × three seeds, exact latent states, transition matrices,
  Bayesian belief vectors, Bayes-optimal predictions **and forced-state
  counterfactual targets**, against four controls.

### The concession was inverted

**arXiv:2405.15943** (Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers,
NeurIPS 2024) is real and says what its title says. But it **scores no
architecture against anything** — no baselines, no windowed oracle, no RNN
comparison; belief is a *probing target* recovered by linear regression from
residual activations after next-token training. So conceding "the bed does not
claim the representation result" gave away ground that paper never held.

And the delta this page kept — **"exact oracle labels"** — is that paper's own
machinery, and is 35 years older than it: Rabiner 1989,
`10.1109/5.18626`. **It is struck from the delta.**

Also named, and absent from the original section: **arXiv:2502.01954**
(Piotrowski, Riechers, Filan, Shai), *Constrained belief updates explain
geometric structures in transformer representations* — attention **as** the
constrained Bayesian update, predicted analytically. That is closer to this
project's operator claim than 2405.15943 is.

**Frühwirth-Schnatter was miscategorised.** Kalman is the continuous twin;
Frühwirth-Schnatter (2006) is Markov *switching* — the same discrete-latent
family in a Bayesian idiom, not a continuous analogue.

### Occupied elsewhere, with owners

| claim | owner |
|---|---|
| in-context identification of an HMM | 2512.22471, 2506.07298, 2410.16546 |
| ⊥ as an output symbol | 1907.00208; per-position in a structured output, `garcia18a` ICML 2018 |
| belief entropy as the uncertainty statistic, decomposed per position | 1202.6545 (Durand & Guédon, 2012) |
| the `Σ w·v / (Σ w)^α` family | **arXiv:2607.22781**, *What Softmax Throws Away: Mass-Aware Attention for Evidence Accumulation* (Yu & Ha, 24 Jul 2026) — the extensivity argument for attention normalisation, arrived at independently, with a global `p` |
| softmax normalisation destroys the extensive read | **arXiv:2310.08661**, with the ablation, 2023 |
| extensive-vs-intensive selects the aggregator | arXiv:2207.13779 — sum for extensive, mean for intensive, chosen by physics |

### What survives, and it is one sentence

> A protocol that scores a sequence architecture's **⊥ emission** against an
> exactly-known filter posterior, with the abstain threshold anchored at
> `ln S − δ` rather than tuned for coverage — on a bed carrying an extensive
> label alongside the intensive one, against a windowed-forward-algorithm
> baseline whose headroom is **derived from `λ₂` in advance**.

Every *mechanism* in that sentence is borrowed and cited. What is new is the
**evaluation protocol** and the **maximum-entropy anchor**. Nobody scores ⊥
against a filter posterior — four phrasings found nothing. It is a much smaller
claim than this page originally made.

### The corner rule adds nothing here, and the reason is internal

Its conjunction — an exponent chosen **per output coordinate** by dimensional
analysis of what that coordinate reads — is genuinely unoccupied: every
published exponent is indexed by model, head, layer or query, never by output
coordinate. But `docs/CORNER_RULE_RETIREMENT.md` retired the rule on the same
day this bed was pre-registered, on three measured grounds, and closed with
"No third reroute is proposed." **This bed would be a third reroute.** The
external gap does not reopen an internally closed question.

## Gate correction: must-fire 1 is a theorem, not a gate

"Window-`L` oracle error must decay with `L`" is guaranteed by filter stability
— exponential forgetting of the initial condition, Le Gland & Mevel,
*Math. Control Signals Systems* 13:63–93, 2000 — with the rate in this bed's own
`λ₂` given by arXiv:1710.06078 as `B = log(ε)/(λ₂ − λ₁)`. It **fires by
construction** and discriminates nothing. It is retained as a sanity check on
the generator: if it ever fails, the generator is broken, which is the only
information it can carry.

It follows that **the window-4 band must be derived, not discovered.** `0.16–0.22`
was written as a bar; it is a function of `S`, `λ₂` and `snr` fixed at
generation time and must be computed before it is measured, with any
disagreement reported as the finding.

## Skyline correction: the GRU as specified is below a published lower bound

`S` hidden units for `S = 8` states sits under the `Ω(N·|Σ|)` bound of
arXiv:2310.05161, and arXiv:1805.04908 shows the GRU is strictly weaker than
LSTM and ReLU-Elman in finite precision **because it cannot count** — while the
extensive label `Σ_τ b_τ·h` *is* a count. The `S`-unit GRU is therefore a
**floor**, not a skyline; a properly sized recurrent arm and an LSTM must run
beside it.

## The counter-prediction is itself a published result, and its sign flips

"The GRU skyline wins outright" is arXiv:2406.04089 (June 2024): transformers
consistently underperform RNNs across all tested HMMs — in the **fixed**-HMM
regime. In the **strong** form this bed names as its real target, 2512.22471
reports the opposite with numbers, LSTM failing at 0.411 against Transformer
0.049. Whichever way this bed lands it replicates one published result and
contradicts another, and the axis that decides which is exactly the weak/strong
form of the trap.

## One more gate, from the audit

A near-singular `P` makes the bed unlearnable for every arm — learning HMMs
without nonsingularity is at least as hard as noisy parity (`cs/0502076`). A
conditioning gate is required, with its refusal fraction reported beside the
other two.

## Prediction, on the record

**The author's:** the read beats window-4 with a confidence interval, and
matches the GRU skyline within resolution at horizon `≥ 16`.

**The counter, point estimate:** the GRU skyline **wins outright**. A belief
update *is* a first-order recurrence in belief space, so the native primitive is
a recurrence and attention only approximates it. On that reading the read's
honest claim here is **parity-with-skyline plus refusal** — not a win.

Both are recorded before the bed exists. If the read wins outright, the counter
was wrong and says so; if the GRU wins, the prediction was wrong and the claim
drops to parity-plus-refusal. Neither may be revised after the first number.

## What would make this bed void

Stated now so it cannot be decided later: if the window-`L` error does not decay
with `L`, the bed is Markov and every result on it is uninterpretable. If label
spread falls under 0.1, the bed is dead. If any arm is handed `P`, the trap is
broken and the run measures nothing. Any of those three and the instance is
refused rather than reported.


## Corrections from a theory audit, 2026-09-14 — still before any arm ran

**A citation on this page carried the wrong title.** `arXiv:2607.22781` was
written as *"Mass-Aware Attention"*, which is the mechanism name from the
subtitle. The actual title is *What Softmax Throws Away: Mass-Aware Attention
for Evidence Accumulation*. Real identifier, wrong title field — the exact
pattern a citation audit exists to catch, committed here by the same session
that runs the audits. Corrected above.

**arXiv:1612.02526 does not transfer to this bed at all**, and the earlier note
that it "bounds next-observation prediction, not belief estimation" understated
it. Full-text: *belief* appears **once**, non-technically; *posterior* appears
only inside the proof of Lemma 1 as a submartingale potential, never as an
output. Everything is **doubly averaged** — over a uniformly random `t` and over
the process — and the authors themselves call Proposition 1 "a kind of negative
result—that average error is not a good metric". This bed's label **is** the
belief and its metric is per-position TV. `log n/ε` transfers to nothing here.
For a short-memory claim the citation is **Atar & Zeitouni, SICON 35(1):36–55,
Corollary 2.1** for the rate, and Ye–Ma–Qian eq. (10) for the window length with
its constant `C` flagged as never estimated by its own authors.

**Le Gland & Mevel do not own the Lyapunov-gap characterisation.** They
attribute it to Atar & Zeitouni in their Introduction, p. 66. They remain the
correct citation for forgetting.

> ~~Their own Remark 2.4 attributes it to Atar & Zeitouni; their contribution is
> a Birkhoff contraction coefficient under primitivity.~~ **SUPERSEDED.** This
> page invented a Remark 2.4 that does not exist in the paper, and attributed a
> Birkhoff contraction coefficient to authors whose text contains **zero**
> occurrences of *Birkhoff* — it is Atar & Zeitouni who use Birkhoff. Real
> paper, real attribution, fabricated locator and fabricated contribution: the
> `P-16` shape, committed on the page that files `P-16`. The attribution itself
> survives at the corrected location.

**The concession to arXiv:2405.15943 can be sharpened.** Full-text grep of v3
finds **zero** occurrences of *fractal dimension*, *box-count*, *Hausdorff*,
*correlation dimension*, *Lyapunov* or *contraction*. It **visualises** the
fractal and never measures it; its reported R² values are correlations of
pairwise distances, on one generator, one trained configuration, no architecture
comparison. Measuring the attractor's information dimension concedes nothing to
that paper.

**The fractal link is an upper bound with no matching lower bound.**
`dim_H(ν) ≤ h/χ` holds unconditionally for place-dependent IFS
(Jaroszewska–Rams, arXiv:0707.3532 Thm 1), applied to the Blackwell measure by
Bárány–Pollicott–Simon (J. Stat. Phys. 148:393–421, Prop. 14). Every **equality**
theorem assumes *constant* probabilities; the filter's weights are
place-dependent by construction, which is exactly the gap. Measured here: the
bound holds 4/4 with 11–45% slack, and is **vacuous by 23×** in the overlap
regime, where the attractor collapses to a point while `h/χ = 23.276`.

**A trap for anyone reproducing this.** Box-counting the *support* gives
`d_box = 2.57–3.31`, which appears to violate `h/χ`. It does not: the theorem
bounds the dimension of the **measure**, and `dim_H(μ) ≤ dim_B(supp μ)` always.
Only the information dimension `D₁` is the comparable quantity. Box-counting the
support here produces a **false refutation**.

**One exact identity, worth asserting as a free correctness check.** The top
Lyapunov exponent of the filter's random product equals minus the entropy rate
of the observation process, `λ₁ = −h(Y)`, identically — because the product's
norm *is* the sequence likelihood. Measured residual `1.26e-05` on the weak
draw and `1.41e-13` on a closed-form control.


## A second correction pass, 2026-09-14 — after the bed ran, before any arm is published

**Three citations on this page were misaddressed, and all three are corrected
above or here.** They were written by the same session that runs the citation
audits, which is the point of recording them.

1. **Le Gland & Mevel "Remark 2.4" does not exist.** Corrected in place above.
2. **arXiv:2102.10487 Eq. (15) is the dimension formula, not the normaliser
   identity.** The identity this page needs is the prequel, **arXiv:2008.12886
   Eq. (16)**. Cited to the wrong paper in the same series — an off-by-one-paper
   error, which resolves cleanly and is therefore invisible to an identifier
   check.
3. **Ye-Ma-Qian eq. (10) gives a slope and no level.** Its constant `C` is never
   estimated by its own authors, so no absolute window length can be derived
   from it. The derived-level figures this page carried were produced by
   multiplying a **quenched** (geometric) rate onto an **annealed**
   (arithmetic) level, and read **2.5-17x too low on every draw**. Both derived
   levels are dropped. The slope is kept and verified: fitted on `L >= 4`
   against the geometric error, quenched slope over `exp(Lyapunov gap)` reads
   **1.0073** weak and **1.0142** strong (1.65% / 2.75% mean error), while the
   annealed form misses by **40.6% / 52.5%**. Jensen holds **48/48** in both
   forms. `lambda_2` of `P` over-predicts the decay factor by **2.1x**.

**The window-4 band was pre-registered wrong.** Measured **0.1423** against the
pre-registered **0.16-0.22**, with 26.6% of draws inside. The replacement is
measured rather than preferred: the admissible region is a diagonal ridge,
`snr = 1.181 * lambda_2 - 0.575`, not an interval in either coordinate alone.

**The free identity holds.** `lambda_1 = -h(Y)` verified to **2.2e-04** on the
live draw — a correctness check on both the filter and the QR code, costing
nothing.
