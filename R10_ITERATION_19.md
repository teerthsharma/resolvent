# R10 — iteration 19: NOT REGISTERED, with a route for every failure

**Script task.** *"ALL: registration verdict = C-A…C-F all GREEN; scoreboard +3;
MERCURY reprices 1c in the same message (RULE 4)."*

## The verdict

**NOT REGISTERED.** Two of six clauses hold; four fail.

| clause | verdict | the number |
|---|---|---|
| **C-A** split integrity | **FAILS** | 12 shared graphs, bit-identical labels, **43.66%** of eval rows — on the by-file reading |
| **C-B** within-split probe | **FAILS** | median NRMSE **0.7639** against bar 0.9, 25 of 28 rungs |
| **C-C** activation bytes | **GREEN, unearned** | prices the wrong tensor by **s/d = 64**; VRAM never binds |
| **C-D** an arm reads < 1.0 | **HOLDS 12/12** | `khop_128` at **0.005092** vs a mean-predictor control at exactly 1.000000 |
| **C-E** oracle executable, no answer keys | **HOLDS** | 0 structural hits in **5,461 files / 142,271,756 bytes examined** |
| **C-F** do()-bit moves by δ | **FAILS 1/12** | one boundary node with no interior neighbour: fd **exactly 0.0** |

**Scoreboard: no +3.** The gate is a conjunction and it is not satisfied.

## Three kinds of failure, three kinds of repair

Collapsing these into "four clauses failed" would hide that they need entirely
different work.

**C-A and C-B — ill-posedness.** The corpus declares neither a train/eval split
nor a tensorisation, so neither clause can be *evaluated*, let alone passed. Both
branches were measured rather than one being quietly chosen:

- **Split:** by file → 12 shared graphs, 43.66% leaked. By arm → **0 shared**,
  0 of 5,693. Declaring by arm turns C-A green **with no instance changed**.
- **Tensorisation:** geometric features → admissible **28/28**. Adding `gbar` →
  inadmissible **25/28**. Not a free choice: `gbar` is the first Neumann term of
  `u = P_IB g + P_II u`, so an arm that sees boundary values reads the label's own
  recursion.

**C-C — a wrong formula.** `4·n·s·d·heads` prices the `[n,s,d_model]` projection;
the peak is the `[n,s,s]` operator. At batch 8192, s=1024 that is 512 MiB claimed
against 32,768 MiB real — four times the whole card. And VRAM never binds anyway:
the host refuses at batch 8,832, VRAM not until 101,837. Bound executably at
`tests/loop/test_the_activation_clause_prices_the_peak_tensor.py`.

**C-F — a generation gap.** `stratify` can emit a boundary node whose neighbours
are all boundary nodes, so a `do()` has nowhere to propagate. One instance of
twelve, at the corpus's largest boundary (|B|=80). Threshold-free, so it survives
the strike against C-F's δ. Bound RED at
`tests/loop/test_every_boundary_node_can_propagate.py`. The route is one clause in
`admissible`, the mirror of the one already there.

## What was struck on the way to this verdict

**C-F's 12/12 failure was spurious.** δ=0.5 was imported from the chain task's
`flipper_dependence > 0.5` clause. Measured from the shipped rows,
`fd_max × |B| ≈ 3.741` — harmonic measure — so `fd ≥ 0.5` needs `|B| ≤ 7.5` while
the corpus runs `|B| = 4…80`. **Ten of twelve rungs cannot reach it by
construction.** The source threshold is worse than inapplicable: the chain's own
`2/sqrt(t*)` drops below 0.5 at `t* > 16`. Recorded as **MISTAKES.md V-17** — a
threshold travels only if its quantity is anchored.

**The scope clause over-fires.** it.18's census flags 19 of 31 controls and 17 are
not defects, because a refusal guard is *supposed* to be exercised on inputs
production cannot emit. `R ∩ D = ∅` is necessary, not sufficient. Recorded as
**V-14a**, which also resolves a contradiction this round created between "bind a
must-fire to a constructed input" and the scope rule.

## The state Phase 1c inherits

Registration is refused, so Phase 1c is formally blocked. The round proceeds under
two **declared** assumptions — split by arm, geometric features — with the cost of
being wrong stated in `R10_ITERATION_16.md`: if the arm is meant to see boundary
values, the corpus is **rebuilt, not re-declared**, and every S1/S2 result computed
under the assumption is void.

MERCURY's repricing of Phase 1c is running, carrying four measurements this round
produced: host RSS binds not VRAM; the linear RSS fit is a **floor** above n=49152
(8.2% low, slope rising 0.1102 → 0.1409); wall-clock is 502–532 s per cell at
n=49152 and ~345 s at n=32768; and N=8 is not optional, because two of eight seeds
fell the opposite side of the bar at n=32768.
