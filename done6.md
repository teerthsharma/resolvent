# done6 — Round 6, CEQ v8.2, the Hilbert round

**Status at iteration 22 of 30. `HILBERT` is NOT emitted — neither `KEPT` nor
`BROKEN` is yet true.** The deciding measurement is UNRUN, and the round is still
running. This is a work-done, not a verdict.

Rounds 1–5 in `DONE_ARCHIVE_ROUND1.md`, `workdone2.md`, `done3.md`, `done4.md`,
`done5.md`. Iteration log in `DONE.md`; negative result in `D1.md`.

**SCOREBOARD: 4 of a ~45 ceiling.** Star delta +2, `κ<1` measured +2 (qualified).

---

## 0. Live right now, and the leap that just landed

### 0.1 Dr House was released at iteration 22, and returned a LEAP

**Trigger check, all three required and all three held.** A theory was dying — the
Birkhoff certificate does not apply to this map. The cause of death was **missing
innovation**, not missing evidence: more draws will not make `T` linear and more
seeds will not make `β` a measurement. And the thing it was going to deliver is
still wanted — it is D2's headline claim.

**He was NOT released on T1**, whose premise was measured false, nor on K-A, where
positivity did not fail. Released against a clean measurement is the one thing the
precedent forbids.

**Five minutes. No nurses. His answer, and it is one sentence:**

> **You certified the wrong map. Stop certifying `T` on the state cone. Fold one
> full iteration into pivot-weight space and certify the LINEAR piece that lives
> there.**

Two steps of `T` compose, in `w`-space (`k`-dimensional, `k ≈ 8`), as

```
w'_q  ∝  gate_q * ( Σ_p G_qp w_p ) ** β        G_qp = <a_q, a_p>
```

which is `P_β ∘ diag(gate) ∘ G`. The diagonal is a **projective isometry and drops
out**. `P_β` is **the dial — keep it**. And **`G` is a `k×k` entrywise-positive
LINEAR map**, so Lemmens–Nussbaum Thm 2.9's hypothesis is satisfied **verbatim,
with no nonlinear excuse**:

```
κ(T)  ≤  β · tanh( Δ(G) / 4 )
```

**The point is which cone.** The measured `Δ̂ = 101.3671 … 311.6091` nats lives on
the `d`-dimensional **state** cone at logit scale `|w| = 1.171e+01`, where softmax
makes the oscillation explode. **`Δ(G)` lives on the `k`-simplex of ~7.9 pivots.
Small cone, small diameter.**

### 0.2 Half of it was verifiable immediately, and it checks out

He noticed the number was **already in the corpse**. The attained-to-dial ratio
across the published β sweep [RUN, from numbers already in the record]:

```
    beta   attained        ratio
    0.25   0.240448   0.96179200
     0.5   0.480897   0.96179400
     0.9   0.865614   0.96179333

  spread = 2.000e-06   mean = 0.96179311
```

**Constant to `2e-06` across a 3.6× range in β.** A β-independent residual factor
is exactly what a linear factor multiplying the dial looks like — which is the
fingerprint he predicted before anyone measured a Gram.

His hypothesis, inverted: `tanh(Δ(G)/4) = 0.961793` ⟹ **`Δ(G) = 7.8772` nats**. He
predicted "≈ 7.88" without computing it.

**Beside the diameters actually measured:**

| cone | `Δ` | `tanh(Δ/4)` | `1−κ` |
|---|---|---|---|
| randn probe, **state** cone | `101.3671` | `1.000000` | `1.947350e-22` |
| trained, **state** cone | `83.6069` | `1.000000` | `1.399659e-18` |
| K-A float threshold | `76.246190` | `1.000000` | `5.551115e-17` |
| **House's `Δ(G)`, pivot cone** | **`7.8772`** | **`0.961793`** | **`3.820689e-02`** |

**The state-cone diameter saturates `tanh`. The pivot-cone diameter is nowhere
near it.** That is the entire leap.

### 0.3 The part with teeth, and it is what the round was actually missing

The certificate is finite **iff `G` is entrywise positive — iff the pivots overlap
in support.** Orthogonal pivots put zeros in the Gram, `Δ(G) = ∞`, and no
contraction credit beyond `β` survives.

Trained effective support is **`3.60068` of `~7.9` pivots**, so **trained attention
EARNS its contraction through pivot overlap.** That is a property of the attention
that **moves when the weights move**, and training can be watched tightening it —
which is precisely what `κ = β` was not.

### 0.4 Its status, stated plainly

**HYPOTHESIS. Not a finding, and it may not enter any verdict yet.** Dr House is
exempt from RED-first because five minutes does not fit a test, **and that
exemption is exactly why his output binds before it counts.**

- **`[RUN]`-verified:** the ratio constancy, from published numbers.
- **NOT measured:** `Δ(G)` itself. **Nobody has computed the trained pivot Gram.**

**He named his own binding test**, which is the right instinct: compute `Δ(G)` from
the trained Gram, predict `β · tanh(Δ(G)/4)`, and compare against `0.480897` at
`β = 0.5` and against the seed/scale-invariance panel. **If the ratio does not
track `tanh(Δ(G)/4)` per seed, the factorisation is wrong and this dies too.**

Until a fellow runs that, it stays in **Open**.

### 0.5 Still executing

**Chase's M3 quintuple** — `scale/m3_quintuple.py --n-train 8192`, PID 23104, **19
units journalled** (12 at iteration 15), lock released between buckets, still
advancing. **It remains UNRUN at the gate**: the RULE 2 deadline passed at
iteration 14 and any completion is recorded as a **post-deadline result, marked as
such**, which does not move round 6's scoreboard.

Phase D closed; the audit closed; Wilson closed. **The four claims that would move
the scoreboard are all now blocked on measurements nobody is running: `Δ(G)`, the
M3 headline cell, the trained-`Δ` sweep in `s`, and the aggregator retention
re-run.**

---

## 1. The one fact that reframes five rounds

**Every probe number this project has ever taken sits at a geometry the model does
not occupy.** Measured on one axis at last:

| geometry | mean causal `\|w\|` | `alpha` | max `Δ` |
|---|---|---|---|
| `torch.randn` probe (F-lam) | `1.171e+01` | one-hot, `log α min −182.7498` | `101.3671 … 311.6091` |
| m3 harness init | `0.00266492` | near-uniform, `−2.33938` | `6.3317` |
| **trained** | `0.454379` | lopsided, `−13.243`, eff support `3.60068` | **`83.6069`** |

Trained sits **170.5× above** the harness init and **~26× below** the randn probe,
both CI-disjoint.

**And the randn probe is not uniformly wrong — it is right about the diameter and
wrong about the weights.** On `Δ` trained is much nearer the randn end; on `alpha`
much nearer the harness end. A first reading that it overstated everything was
itself corrected.

---

## 2. What was established

**T1's premise is FALSE at trained projections.** `alpha` effective support runs
`7.49383 → 3.60068` of `7.91667` pivots, CI-disjoint; `alpha max` `0.19708 →
0.637833`. **An argmax lookup is 1.0.** Training concentrates the settled reading
but stops well short of degenerate. **The "expensive argmax" worry is refuted where
it matters, so T1 does not fire and Dr House was not released on it.**

**The free bind holds.** A model trained from scratch in the Phase D probe lands on
the published F-green cell **exactly**: `0.747528`, `CI [0.696849, 0.797716]`,
against `1.000335` at zero steps.

**`κ(T) = β`, exactly and attained.** The map factors through four cited theorems;
measured `0.500000000` — nine decimals — across **72 cells**, invariant in `s`,
`d`, `k`, logit scale and seed. β sweep tracks: `0.0→0.000000`, `0.25→0.240448`,
`0.5→0.480897`, `0.9→0.865614`.

**ARM S is born.** G3 bitwise identity 24/24, birth gate 1 (settled ≠ **one step**,
`0.986111`, CI `[0.925029, 0.999648]`, 92× over the 1% bar), birth gate 2
(gradcheck true at β = 0.25/0.5/0.9, must-fire fails at N=3).

**Degree-2 claims survive.** `E|L₃|/E|I|` = `0.049156` random-init, `0.044307`
trained — both far under the 0.5 bar. Exact-zero rate `1/24` trained against
`279/400` at the randn probe: **the 79% figure was a probe artifact. ARM P has
signal where the model lives.**

**The Star-Transformer delta, from the paper rather than its abstract.** Relay
pools **all** satellites (Eq 7), topology fixed by **position** (Eq 4),
`causal`/`autoregressive`/`top-k`/`interaction` all read **0**. The `mask` count of
15 is entirely the synthetic task *"Masked Summation"* — **a keyword sweep would
have concluded the opposite of the truth.**

---

## 3. What died, and it is the round's headline

**The Birkhoff certificate does not apply to this map.** Lemmens–Nussbaum Thm 2.9
requires a positive **LINEAR** map; six independent sources all carry "linear" in
the hypothesis; nonlinear order-preserving homogeneous maps get **nonexpansive
only**. `T`'s weights depend on `m`. A nonlinear map with a strict `tanh(Δ/4)`:
**NOT FOUND**.

**And at trained projections the implicit-gradient route is not affordable.**

| side | median `Δ` | max `Δ` | `κ(max Δ)` | `1−κ` | Neumann `N` |
|---|---|---|---|---|---|
| random-init | `3.7815` | `6.3317` | `0.919056` | `8.094389e-02` | **194** |
| **trained** | `17.6298` | **`83.6069`** | `1.000000` | `1.399670e-18` | **INFEASIBLE** |

`N = 3.924227e+19`. **Trained `Δ` sits ABOVE the K-A float threshold of
`76.246190`.**

**What survives is `κ = β`, and β is CHOSEN.** It sets `t*` and `N` directly. In
the words of the fellow who derived it and refused to dress it up: *"The
certificate is a dial with a known transfer function, not a measured property of
attention."*

**That is the leap gap Dr House is currently running on** — released at iteration
22, five minutes, hard stop, on the question *"where does a contraction constant
that is a measured property of attention come from, or is 'certified' the wrong
ambition?"* His output is a **hypothesis** and binds RED-first before any verdict
cites it.

---

## 4. Repairs that earned their keep

**The K-A float repair is now load-bearing on real data.** Made before any datum
landed: `tanh(Δ/4)` saturates to exactly `1.0` for `Δ ≥ 76.246190`, so a kill
written on `κ_cert ≥ 1` **cannot distinguish "positivity failed" from "diameter
merely large"**. At trained projections `κ_cert` reads exactly `1.0` while the
closed form `1−κ = 1.399670e-18` is nonzero. **K-A correctly does not fire** — `Δ`
finite, `0/24` cells infinite. Written the original way, **ARM S dies 30/30 at
iteration 0 on arithmetic alone.**

**The log-domain metric removed a blocker at negative cost.** `d_H(softmax u,
softmax v) = osc(u−v)` exactly, so the metric needs no `exp()`. float32 softmax
reads `d_H = +inf` at ARM A logit scales (146 / 1904 / 2028 underflowed entries);
log-domain reads finite. **It then also fixed ARM S's vertex collapse**, where the
true fixed point has coordinates `~e^-1000` and float64 rounded it to a corner. One
root cause — logit scale — four symptoms.

**The G2 inverted.** A published interval stopped reproducing. The cause was a
`torch.Generator` passed **by reference** to two bootstraps; the first consumed
`4,800,000` draws. Repairing the producer **restored both published intervals
exactly** — it was never a number moving, it was a producer acquiring a defect
after publication. **Zero of ten sites needed correcting.** The defect also
silently decoupled K1's *"on the SAME draws"* clause, unnoticed for a round.

---

## 5. Defects of ours, and who caught them

Nine controls were found unable to fire this round, across **four different
authors**. The pattern is now specific: **a hand-built minimal example is where a
control goes vacuous, because the smallest case is usually where the right and
wrong answers coincide.** Drawn instances with a count are the repair.

- **A 2×2 greedy control** could not fire — on two points the monotone assignment
  is forced. Replaced by drawn instances: greedy strictly worse in **111/400**.
- **A square matcher calibration** could not test its solver — equal pools make
  Cohen's *d* permutation-invariant. Replaced by a rectangular direction.
- **A naive paired bootstrap** returned a CI **excluding zero on null data**,
  understating by **2.1×**. Matching is part of the estimator; every replicate must
  re-match.
- **A `tril(-1)` row summing to exactly 0.0** made a bitwise control fire on a
  divide-by-zero rather than the float claim it advertised.
- **`pooled < tail`** is an identity, 400/400, cannot fail.
- **A closed form published under the word "Expanded"** was a truncated series —
  rel err `1.007e-03` → `5.328e-01`. Load-bearing consequences survived.
- **A κ derived from `mean(Δ)`** where Birkhoff's hypothesis is a **supremum**.
  Caught by its author, and **the fix flipped the sign of the answer.**
- **A "regime" error rather than a wrong number:** a leakage rule derived at masses
  `0.02–0.2` against a measured median of `1.2454e-20`. Correct mathematics,
  vacuous in practice.
- **A provenance claim of ours** — *"RED since 09:40 across five runs"* — was **7,
  not 10**, in **3** sessions, not five.

---

## 6. The audit

**All ten standing REDs are BY DESIGN. Zero rot.** Proven from the board rather
than the code: each reads RED on **every** recorded run, 6–13 runs apiece, **never
GREEN once**. Rot shows GREEN-then-RED.

**Three published claims struck:** our provenance sentence; `CHECKLIST.md:50`'s
reach half (**the kill was pre-registered in round 1 and its refutation recorded
twice — the sentence was simply never amended**); and `THEORY.md`'s contraction
guarantee, which holds for the **zero-action operator only** — intervention-
conditioned `ρ` reaches **1.4172**.

**Two scan gaps.** `THEORY.md` was outside `LEAD_DOCS` (closed). And the scan hunts
**numeric constants**, so a prose claim carrying no number is **structurally
invisible to it** — recorded, not repaired.

**Coverage is honest in mechanism, stale in value.** `collected == executed ==
passed`, zero skips; `.` and `tests` agree. But live reads twenty minutes apart
gave `200/1435` then `209/1446`. **Any quoted pair is a timestamp, not a fact.**

**Git cannot date any RED transition — because there was none.** Four files have a
single commit whose board entries **predate it by 8h33m**. Born RED.

---

## 7. Open, in order of what it would overturn

1. **The certificate is a dial.** Dr House is on it; a fellow must bind whatever he
   returns.
2. **M3 is UNRUN.** Declared so at the deadline; the breach's dominant cause is
   ours — the contract's triple was widened to a quintuple (**15 → 25 units, +67%**)
   inside a two-iteration window without extending it. **The design change was
   right; the schedule arithmetic was not done.** Any completion is post-deadline
   and marked.
3. **No within-sequence norm-matched control exists.** `select_pivots` takes top-k
   by norm, so every other within-sequence set is strictly lower; the matcher
   closed **zero** of the gap in **512/512** draws. **Not a bad control choice — no
   good one exists.** No mechanism claim about *which* tokens may be made.
4. **Trained `Δ` is one seed, one arm, s=64**, and `Δ` is the statistic most likely
   to move with `s` — and the one carrying the negative verdict.
5. **K-F is bracketed, not passed.** FLOP `1.010420`, dispatch **`23.00×`** (exactly
   `3.0` per settling step), clock `1.6546` lying between them. **The fix is named:
   a fused settling step removes `O(t*)` dispatches at zero FLOP cost.**
   `collect_callgrind` is **unavailable on this platform**, not merely undone.
6. **Aggregator retention `2.9% / 13.3% / 58.9%` not re-run** — still random-init,
   still rank-matched, still upper bounds.
7. **Four RED findings recorded in no document**, and the row-stochastic tradeoff
   they encode: unconstrained buys expressiveness and loses the contraction
   guarantee (`ρ > 1` under sustained action); row-stochastic buys the guarantee and
   **structurally cannot represent a decrementing loop**, since `sigmoid × softmax`
   is non-negative for any parameters.
