# R10 — iteration 17: a causal corpus with an acausal node, and controls that miss the front door

**Script task.** *"C-D = ∃ arm with NRMSE < 1.0 before any failure is scored; C-E
oracle executable, zero answer keys in files; C-F do()-bit flip moves labels with
effect ≥ pre-set δ, control inputs drawn from the PRODUCTION batch path
(L-SCOPE)."*

Run by MARS rather than SATURN, deliberately: C-F is the scope clause, and MARS is
the seat whose presumption P1 was withdrawn this round for keying on **who built
the input** instead of **where it enters**.

| clause | verdict | the number |
|---|---|---|
| **C-D** an arm reads < 1.0 | **HOLDS 12/12** | `khop_128` at **0.005092** vs a mean-predictor control at exactly 1.000000 |
| **C-E(i)** oracle executable | **HOLDS 12/12** | sensitivity to `g` 2.29e-01…6.93e-01, determinism 0.0, second route 1.554e-15 |
| **C-E(ii)** zero answer keys | **HOLDS**, with a denominator | **0** stored label vectors in **5,461 files / 142,271,756 bytes examined**; 56 two-coordinate disclosures in one file, reported |
| **C-F(a)** do()-bit moves by δ | **FAILS 1/12** after the δ strike | δ=0.5 imported out of its units — unreachable for `\|B\| > 7.5`; **what survives is one instance at exactly 0.0** |
| **C-F(b)** controls enter at the front door | **instrument works; the round's path does not** | classifier agrees with `p1prime` on **191/191**; the round's 12 controls enter **below** `make_rung`, 5 stages unexercised |

## A causal corpus that emits an acausal node

`reproduce-n1024-d4-t0.95`, seed `0x3a14000a`, ships a boundary node whose do()-bit
moves the label by **exactly 0.0 in every norm**.

**It is not a plant.** It is a boundary node all of whose neighbours are also
boundary nodes, and **nothing in `spec.stratify` forbids one**. Intervening on it
cannot propagate, because there is no interior vertex adjacent to it to propagate
into.

A corpus built to measure causal consequence emitted an instance where a causal
intervention has no consequence. It was found by a clause that files its threshold
**before** it looks — δ=0.5 was in the module header before the first measurement
— which is the only reason a reading of 0.000000 registered as a failure rather
than as an unremarkable small number.

## STRUCK: the 12/12 failure is a threshold imported out of its units

C-F's `delta = 0.5` carries its own provenance in MARS's rows —
`delta_source: scale/negation_scope.py:1587 (bar_verdict)`, the **chain task's**
`flipper_dependence > 0.5` clause. Measured from his shipped numbers:

| \|B\| | 4 | 7 | 11 | 19 | 41 | 80 |
|---|---|---|---|---|---|---|
| `fd_max` | 0.585 | 0.410 | 0.302 | 0.194 | 0.098 | 0.065 |

**`fd_max × |B|` is near-constant at 3.741** (range 2.341–5.220 over twelve rungs)
— harmonic measure, where one boundary node's influence is `~1/|B|`. So `fd ≥ 0.5`
requires **`|B| ≤ 7.5`**, and `stratify` targets λ₂ ∈ [0.90, 0.95], which forces
large boundaries: the corpus runs `|B| = 4…80`. **Ten of twelve rungs cannot reach
the threshold by construction.**

The source threshold is worse than merely inapplicable. The chain's own closed form
is `2/sqrt(t*)`, which drops below 0.5 at `t* > 16` — so that clause would fail on
**the chain itself** at `t*=32`, where it reads 0.3536.

MARS did the right thing — import a threshold rather than pick one that flatters
the result — and got a wrong number, because that rule has an unstated precondition
the round has now written down as **V-17**: a threshold travels only if its quantity
is *anchored*. NRMSE's 1.0 is predict-the-mean by construction and travels;
`flipper_dependence` is set by the generator's parameters and does not.

**WHAT SURVIVES THE STRIKE:** the zero-bit, below. It is independent of any
threshold.

## The round's own controls miss the front door

The classifier agrees with `scale/p1prime.py` on **191/191** files, so the
instrument is bound to the repo's existing measurement of the same property rather
than being MARS's opinion of it.

What it then measures: **the 12 controls this round has been using enter below
`make_rung`, leaving 5 stages unexercised.**

And the demonstration that makes L-SCOPE concrete instead of doctrinal:

> A hand-built star graph reads `fd = 2.000000` where every front-door instance
> reads under 0.36. **No production builder can emit that star.**

So a hand-built control **turns C-F from failed into ok**. The clause passes on an
input the production path cannot produce. That is precisely the P1 → P1′
withdrawal this campaign already made — a control keyed on the wrong thing passing
for the wrong reason — demonstrated in the corpus's own units.

## C-D holds, and its witness is in tension with C-B

`khop_128` — 128 rounds of the message passing the architecture is *supposed* to
do, **untrained** — reads 0.005092 against a mean-predictor control at exactly
1.000000. C-D is established.

**But the it.16 linear probe also reads under 1.0 on 10 of 12 rungs, and that is
the same reading that fails C-B.** One measurement satisfies the clause requiring
an arm to succeed and fails the clause requiring a probe to fail. The battery's two
clauses pull against each other on one number, which is worth stating: C-B and C-D
cannot both be satisfied by a corpus where a simple reader does well.

**Second result, on the k-hop ladder:** the arm does not cross 1.0 until somewhere
between `k = 32` and `k = 64`. At `k = 32` it reads **1.4782 — worse than
predicting the mean.** The label has a large mean and a small spread, so a
truncated arm pays for the mean before it earns any of the variance.

## Phase 1b's registration verdict

it.19 requires C-A…C-F all GREEN. Four fail:

| clause | why |
|---|---|
| C-A | split undeclared; 12 shared graphs and 43.66% leaked on the by-file reading |
| C-B | linear probe beats the bar on `gbar` features, median 0.7639 vs 0.9 |
| C-C | green, but prices the wrong tensor by 64× against a resource that never binds |
| C-F | **one** instance has a do()-bit moving the label by exactly 0.0 — a boundary node with no interior neighbour. Threshold-independent, so it survives the δ strike |

**NOT REGISTERED**, and the four failures need three different kinds of repair:

- **C-A, C-B — ill-posedness.** The corpus can declare its way out with no instance
  changed: a split (by arm gives 0 shared graphs) and a tensorisation.
- **C-C — a wrong formula.** `4·n·s·d·heads` should price the `[n,s,s]` operator,
  and the clause should say VRAM is not the binding resource for a CPU-only harness.
- **C-F — a generation gap.** `stratify` can emit a boundary node with no interior
  neighbour, so a `do()` has nowhere to propagate. **One instance, threshold-free,
  and the only part of C-F that survives its δ being struck.**

## What this iteration costs the round

The C-F(b) finding is the expensive one, because it is about method rather than
this corpus: **the 12 controls C-F uses enter below `make_rung`, leaving 5 stages
unexercised**, and the star graph shows that difference flips a clause's verdict.

**Scoped, because the careless reading is wider than the measurement.** It would be
easy to conclude that every planted control this round is invalidated. Checked, and
it is not. `r10_dual_oracle.py:232` takes `adj, boundary, g` straight from the
instance — **front door, from the corpus generator** — and applies the plant at
`:240` as `defect=plant is not None` **inside `absorbing_extension`**, corrupting
the *oracle under test* rather than the input. That is the correct construction for
testing an oracle. So it.15's rejection rates stand: **29/29** absorbing-side,
**17/17** Kirchhoff-side, and the **0/17** shared-plant blind spot, which matters
most and is built the same way.

**A finding about where control INPUTS enter is not a finding about where PLANTS
are injected.** The two were worth separating before the wider reading propagated
into this record — which it had, in this section's first draft.
