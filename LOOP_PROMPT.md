# CEQ v10 — ROUND 8: THE EQUILIBRIUM-CORPUS ROUND

**GOAL UNCHANGED.** Not the best token predictor — the next-equilibrium predictor.
Attention that understands causality and consequences on Turing-grade problems at the
smallest scale, and the whole attention module built to survive equal to
self-attention or supersede it. Trainable on free tiers, on HuggingFace, with a
capability table.

**THE ROUND EXISTS TO MOVE COMPONENT 6 AND NOTHING ELSE.** Component 6 is
*"understands causality and consequence"*, and `done7.md` scores it at **5 %** against
an engineering half at **80 %**, for a weighted total of **37 %**. Every item below is
subordinate to raising that one number. Work that raises any other component and not
that one is out of scope this round.

**THE ROUND HAS ITS INNOVATION ALREADY. IT NEEDS EXECUTION.** Dr House is **idle** and
is not released on a schedule; he is released only on a NEW missing-innovation death.

---

## AMENDMENT — WIND-DOWN, AND THE ARCHITECTURAL POSITION REVERSES

**HOUSE MODE IS OFF.** No further parallel dispatches. Agents already in flight finish
or do not; nothing new is spawned. The remaining effort goes to **documentation and
mathematics**, which is where the round's value now is.

**AND THE POSITION ON SOFTMAX REVERSES, ON EVIDENCE.**

Softmax is **not the opponent — it is the base case, and the module must contain it.**
Three independent results force this:

* **Softmax is Bayes-optimal on the single-location task shape** — one softmax layer
  attains Bayes risk where linear attention provably cannot (`arXiv:2410.01537`, ICLR
  2025). The shallow rung of this project's own ladder **is** that shape, so attacking
  softmax there attacks it at its proven optimum.
* **Softmax's single step is already a converged fixed-point step** — the attention
  update is a modern Hopfield update that *"converges with one update"* with
  exponentially small retrieval error (`arXiv:2008.02217`, ICLR 2021). **The one step is
  not pre-equilibrium; it is at equilibrium.** That is the answer to why a one-step
  normalised mixture resists improvement.
* **Iterating the normalisation is published and buys little** — Sinkformer, 3–5
  iterations, small gains (`arXiv:2110.11773`, AISTATS 2022).

**THE DESIGN CONSEQUENCE IS EXACT.** If softmax lies **inside** the module's function
class, the module is bounded below by softmax and every equilibrium mechanism is a
**strict addition**. If it lies **outside** — which is what was built, see the pivot
exclusion — the comparison measures the exclusion rather than the idea.

**So the goal is unchanged and its route is corrected: build equilibrium, consequence
and causality ON TOP of softmax, so that they work better BECAUSE of it rather than
instead of it.** A module that contains softmax and adds a settled correction can only
tie or win; a module that excludes it has to win a fight it was never going to win.

**And where iteration is known to pay, the iterated object is the REPRESENTATION, not
the mixture weights** (`arXiv:2311.12424`, ICLR 2024) — which is the one live direction
that has not yet been measured here.

**The full derivations, evidence classes and provenance are in `MATHEMATICS.md`.**

---

## 0. SCOREBOARD

| item | points |
|---|---|
| **X₁₇** e3-harmonic ladder built on **E4′ Rips graphs**, oracle = **absorbing-chain solve**, `λ₂` engineered to **0.90–0.95**, decoder gate + truncation gate + planted controls **all seen firing** | **+3** |
| **X₁₈** Per-draw **Ville e-process** replaces the seed-capped one, calibrated both directions, **ceiling arithmetic printed pre-run** | **+2** |
| **THE READING** — `settled` vs `twin` across `t* ∈ {1, 2, 8, 32}`, trained weights, 5 seeds, fidelity column filled **with the Identity ablation**: | |
| &nbsp;&nbsp;→ curve **grows with `t*`** per the `λ₂` prediction | **+15** |
| &nbsp;&nbsp;→ **flat at every rung including 32** ⇒ settling **RETIRES**, twin ships as pivot-routed mixture attention (the earned fallback) | **+6** |
| **X₁₉** E4′ registered in `M3_TASKS` (a build, not a measurement) | **+1** |
| **X₂₀** Kaggle segment on the certificate + HF upload **with real weights** (author's say-so gate stands) | **+4** |

Carried: **23**.

---

## 1. THE MATHEMATICS OF THE ROUND

### 1.1 THE ORACLE IS AN ABSORBING-CHAIN SOLVE

The label is a coordinate of the absorption probability of a Markov chain on the
graph. Partition the transition matrix into transient and absorbing blocks; the
fundamental matrix is `N = (I − Q)^{-1}` and the absorption probabilities are
`B = N R`. **This is an equilibrium in the exact sense the goal names**: it is the
fixed point of `z ← Q z + R`, it is not a closed-form function of any bounded
neighbourhood of a node, and computing it requires iterating to convergence or
inverting.

**Why this is not the previous corpus.** Every task registered before this round had
an oracle that was a closed-form expression of its input — a product of two entries,
or a sum squared. An arm that iterates had nothing to compute, which is why the
deciding contrast came in at `−0.002959` with an interval covering zero. **That
contrast was correct and uninformative.** An absorbing-chain solve has no such
shortcut.

**Why this is not `e1_anchor`.** E1's label is the signed path sum, which the ceq
resolvent already computes, so an arm built on that resolvent reproduces its own
forward. That is a declared rigged demo and `e3_t*` currently inherits it. **The
absorbing-chain oracle on a Rips graph is not the resolvent of the arm's own
operator**, and Foreman owns stating that separation precisely — see §2.

### 1.2 `λ₂` IS THE DIAL, AND IT MAKES THE CURVE A PREDICTION

The second-largest eigenvalue modulus of the chain sets the relaxation time
`t_rel = 1/(1 − λ₂)`. **Engineering `λ₂` into `0.90–0.95` puts `t_rel` in `10–20`**,
which is the band where the `t*` ladder `{1, 2, 8, 32}` straddles the transition:
truncating at `t* = 1` or `2` must be far from the fixed point, and `t* = 32` must be
close to it.

**This is what turns the dose-response curve from a description into a falsifiable
prediction.** The curve is not merely expected to grow — it is expected to grow *at a
rate set by a spectral quantity measured independently on the graph*. Cheeger's
inequality bounds `λ₂` by the conductance from both sides, so the engineering target
is reachable by construction rather than by search.

**The kill this creates, and it is the sharpest one in the round:** if the curve is
non-monotone in `t*` with disjoint intervals, **the `λ₂` theory is wrong about this
arm**, and that is reported as a finding rather than buried.

### 1.3 THE E-PROCESS GOES PER-DRAW

The seed-capped process could not decide anything at five seeds and this was provable
before the first seed: bounded increments and a betting fraction capped at `½` force
every factor `≤ 1.5`, giving `MIN_T_MIXTURE = 13` against a threshold of `40`, with a
maximum attainable evidence at five seeds of `3.80169140625`. **The ceiling was a
property of the schedule, not of the data.**

**A per-draw process removes that ceiling** by taking the evaluation draw, not the
training seed, as the unit — there are `n_eval = 512` of them per cell rather than 5.
Ville's inequality still gives `P(sup_t E_t ≥ τ) ≤ 1/τ`, and with two live directions
the reported event is their union, so `τ ≥ 40` for a family rate of `0.05`.

**The ceiling arithmetic is printed BEFORE the run, every run.** A process that cannot
cross must say so in the same breath as its first number.

---

## 2. THE ROOM

**CAMERON — X₁₇: the corpus and its gates.** Build the e3-harmonic ladder on the E4′
graphs. The truncation gate and the decoder gate both bind, and **every planted
control must be seen firing** — including the PASS half, which at iteration 11 was
found to be degenerate on a one-component graph and could never have controlled
anything.

**CHASE — the reading, X₁₈, X₂₀.** The `--task` port and per-cell weight saving, the
per-draw e-process, then **the reading itself**. Then the Kaggle segment and the HF
upload, which does not proceed without the author's explicit say-so.

**FOREMAN — the `λ₂`/Cheeger engineering, the Lean statement, and the G1 fetch.**
Engineer the graphs so `λ₂` lands in `0.90–0.95` and measure it rather than assuming
it. State `oracle = resolvent` — or its negation — as a Lean statement, since the
whole creditability of the ladder rests on the oracle not being the arm's own forward.

**AND THE FETCH IS LOAD-BEARING, SO STATE THE DELTA IN WRITING.** Graph neural
networks solving Dirichlet problems is established prior art. **The delta claimed here
is `equilibrium-labels-as-attention-capability-bar`, NOT GNN regression.** Establish
by fetch what exists, and record absence as *not found*, never as *unoccupied*. If the
delta does not survive the fetch, that is the round's most valuable finding and it
lands early rather than late.

**HOUSE — IDLE.** Released only on a new missing-innovation death.

**NURSES** — inference and kernel engineers, whatever hack makes this module
comparable to vLLM while staying original work. They also own the standing failing
tests, which the fellows do not touch.

---

## 3. KILLS — immutable at first datum

**K-1.** The oracle admits any closed-form local fit above the decoder bar ⇒ **the
task is struck.** This is the E4 lesson: a static local-degree decoder read the label
at criticality at `0.4710` and an independent probe at `0.3360`, both inside
`PASS_BAR = 0.5`, and the substrate died. **The gate is not optional and its PASS half
carries its own non-degeneracy check.**

**K-2.** The curve is non-monotone in `t*` with disjoint CIs ⇒ **the `λ₂` theory is
wrong about this arm.** Report as a finding. Do not rescue it.

**K-3.** The curve is flat at every rung including `t* = 32` ⇒ **settling retires**
and the twin ships as pivot-routed mixture attention: `+0.111396` over softmax, CI
`[+0.100873, +0.121920]`, sd `0.016547`, four times tighter than settled. **A smaller
claim, fully earned, and still a product.**

---

## 4. RULES

**RULE 1 — one action per iteration**, then update `STATE.md`, `DONE.md` and the
`CHECKLIST.md` status column, and stop.

**RULE 2 — THE READING COMPLETES BY ITERATION 8.** Not the corpus, not the process:
the reading. A schedule can slip; the deciding measurement cannot be deferred again.

**RULE 3 — every iteration report ends with the scoreboard line.**

**RULE 4 — CAVEMAN binds every agent every iteration.** Short words, verdict first,
no preamble. **Numbers, identifiers, commands, math and quoted text pass BYTE-EXACT.**
Uncertainty stays visible. **Artifacts are exempt** — `DONE.md`, `D1.md`,
`CHECKLIST.md`, `METHODS.md`, commit messages, Lean files and docstrings stay in
normal precise English.

**RULE 4b — any widening of scope carries same-message critical-path arithmetic.**
State what it costs and what it displaces, in the message that proposes it.

**RULE 5 — EVERY KILL SHIPS A REPLACEMENT ROUTE.** Three shapes: **REROUTE** (the goal
lives, the method died, name the sharper method and why it is sharper), **REPRICE**
(the goal lives at another cost, arithmetic shown), **RETIRE** (the goal is
unreachable, measured reason, plus a replacement goal). Fellows are doctors. Inability
to find a route is an unfinished report, not a retirement, and a speculative
replacement is worse than an honest retirement.

**RULE 6 — never block on a measurement.** Something is always built alongside.

**RULE 8 — NAME THE UNSOLVABLE PART AS A FUNCTION AND SOLVE AROUND IT.** When a
quantity cannot be computed directly, write it as `f(...)`, solve everything adjacent
to it **exactly**, substitute into the expected result, and let the surroundings
determine what `f` must be. Then read its **behaviour** — constant, growing,
threshold, limit — and attack it with the derived constraint rather than a guess.
"Unmeasurable, moving on" is not an answer; a function with a measured shape is. Most
of the time the surrounding quantities are already in hand and `f` costs arithmetic
rather than compute.

**RULE 9 — LIGHTWEIGHT MATHEMATICS FROM WHEREVER IT LIVES, AND NOT FLAT BY DEFAULT.**
Reach for the structure that answers the question **whole**, including from obscure
corners, and prefer the cheap version of it. **Do not settle for flat Euclidean
geometry when the object is not flat.**

This is not decoration, and there is a live instance of it. **The readout of record is
NRMSE — a Euclidean distance — while the objects being predicted are fixed points of
row-stochastic maps.** Those live on a simplex, whose natural geometries are
Fisher–Rao, Hilbert projective, or total variation, none of them flat. A Euclidean
error on a simplex-valued target weights a move near the boundary the same as a move
near the centre, when the two are not comparable. **Whether that costs anything here
is measurable rather than arguable, and nobody has measured it.**

The campaign's record supports the rule: the results that survived came from Hilbert
projective geometry, Birkhoff contraction, Cheeger conductance, absorbing-chain
fundamental matrices and exact rectangle covers — not from flat approximations of any
of them. The rule's limit is **lightweight**: a structure that answers the question at
the cost of one solve is in scope; one that needs its own research programme is not,
and saying so is a REPRICE under RULE 5.

**RULE 7 — G1 fetches come BEFORE any build**, and absence is recorded as *not found*,
never as *unoccupied*.

---

## 5. STANDING PROHIBITIONS

* **No fourth contraction certificate.** Three died; proposing another is a stop.
* **No adjacent-measurement instruments.** The disease this project diagnosed is
  building an instrument that measures something next to the thing that matters,
  having it work perfectly, and finding it measured the wrong object.
* **Every must-fire's PASS half carries its own non-degeneracy check.** Fourteen
  vacuous controls have been struck across five authors; the fourteenth was a PASS
  case whose label was constant.

---

## 6. THE ESCALATION CHAIN

Nurses → fellow → Wilson → **Health Inspector** when the problem is NOT a leap,
because the Inspector is the better engineer → **Dr House on `fable`** when it IS a
leap, because House is a scientist.

---

## 7. COMPLETION

Line 1 of `DONE.md` carries the promise. The promise is **`HILBERT`** and it is output
only when it is completely and unequivocally true.

---

## 8. HOUSE ADDENDUM (round 8): why the module loses to softmax, and the two falsifiers that must run before any further ladder money

Appended by Dr House after the missing-innovation dispatch. Nothing above this
section is re-aimed; the pre-registered E-ladder reading in flight completes
exactly as registered. This section binds what may be *concluded* from it and
what runs next.

### 8.1 The value-path exclusion (measured, not argued)

`batched_pivots` excludes `(0, s-1)` and every attention row in this harness is
`tril(-1)`-strict, so no pivot row can place `x[s-2]` on the value path: the
maximum pivot index is `s-2`, and row `s-2` reads `j <= s-3`. Measured
(`e3_t1`, seed 0, perturbing token `s-2` by `+100`): the softmax cell's row-`s-1`
reading moves by `101.6983` (direct value pass-through); the pivot family's `av`
moves by `3.263746` and its log-gate by `14.19184` — weight channels only.
The `e3` label at `t* = 1` is `a[s-1] * b[s-2]`, concentrated on exactly the
demoted token, and every label term at every `t*` carries the factor `a[s-2]`.
**Therefore the `e3_t1`/`e3_t2` softmax-vs-pivot deficits are confounded by
construction and are not evidence about settling.** No headline sentence of the
form "softmax beats the equilibrium arms" may ship without the 8.3 falsifier
having run.

### 8.2 The fixed point iterates the wrong unknown

The settled cell iterates `alpha` inside an 8-simplex — a reparameterisation
within the twin's own output family `{alpha @ av}`. The solve can reallocate,
never add; measured `settled − twin = +0.002190` at `t* = 2`, sd ratio `3.874x`,
mean shift none. Literature agrees from three directions: softmax attention is
already the one-step-converged update of a fixed-point iteration (Ramsauer et
al., arXiv:2008.02217); iterating the *normalisation* to its fixed point buys
small gains (Sinkformer, arXiv:2110.11773); representation-space iteration —
depth — is where iteration pays (looped transformers, arXiv:2311.12424), and
`hop_k` composition needs depth `~ floor(log2 k) + 2` (arXiv:2402.09268). The
hop wall is the same fact seen from below: at `t* = 8` the best arm trains to
`0.860972` — on the 2-hop ceiling `0.866025` — and evals at `1.112208`, because
the 2-hop term is a 3-way product no depth-1 linear-mixing readout represents,
so training memorises noise instead.

### 8.3 The two falsifiers, in order, both cheap

**MOVE 1 — lift the exclusion.** New cells (`twin_plus`, `settled_plus`), by the
`etask_k5e.py` class-swap pattern: `exclude=(0,)` so `p = s-1` is a legal pivot
— then `av[s-1]` *is* the softmax cell's own row and softmax becomes an interior
point of the twin's function class — and/or inclusive-diagonal pivot rows
(`j <= p`). Falsifier: `e3_t1`, `n_train=2048`, seed 0 (~25 min box). If
`twin_plus` does not close at least half the `0.103` gap to softmax, the
exclusion is not the binding term and the deficit is optimisation; if it closes
and `settled_plus` still equals `twin_plus`, settling retires on clean ground
and the `+6` fallback branch fires with the confound removed.

**MOVE 2 — iterate the representation, not `alpha`.** A weight-tied looped cell,
input-injected: `x_{t+1} = x + A(x_t) @ x_t`, 3 loops, parameter count unchanged
at 4769. On `e3` this is credit-clean under §1.7d: softmax-looped vs softmax
involves no signed resolvent. Falsifier: `e3_t2` + `e3_t8`, `n_train=2048`,
seed 0, steps 150 and 600 (~40 min box; the 600 controls the undertraining
confound `E2_STEPS` already measured). If looped-3 neither beats single softmax
at `t* = 2` nor goes below `1.0` at `t* = 8`, depth-via-loop dies on this corpus,
and the honest headline is: one-step softmax wins on equilibrium tasks; the twin
ships as pivot-routed mixture attention on its earned `+0.111396`.

The signed arm's unrigged depth contest — signed-looped vs softmax-looped —
belongs on the X17 absorbing-chain corpus only, and only if Move 2 survives.
