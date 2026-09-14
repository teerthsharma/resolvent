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

## Prior art, owned before it is named

- **Shai et al., arXiv:2405.15943** — transformers represent HMM belief-state
  geometry. **Occupied.** The bed does not claim the representation result.
- **Kalman / Frühwirth-Schnatter** — the continuous twin.

**The delta this project claims over that prior art**, and nothing beyond it:
exact oracle labels; a **refusal** when the belief is uninformative
(`entropy ≥ ln S − δ → ⊥`); and the two corners **assigned** by
intensive/extensive rather than chosen.

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
