# CORRECTIONS 2 — the author's clarification of the deliverable, and two facts from the first design pass

## 0. THE DELIVERABLE IS THE PLAN (author, 2026-09-03, verbatim)

> wait the main goal of this research paper was to make the whole future plan so i can
> slowly develop it this is a current evidence report tf

Binding consequence for every planet: **the paper's body is the future programme** —
a roadmap the author can develop *slowly*, one small step at a time, each step
independently valuable, each with its own pass/kill number and price. The record
(what was proved, measured, refuted) is the *justification* for the plan and is
compressed to what the plan needs; it is not the paper's centre. Concretely:

- Every plan item is written for a developer working alone on the certified RTX 4060
  in evenings: **what to build (a specification, never code), what to prove, what to
  measure, the number that passes it, the number that kills it, the price in
  GPU-minutes, the MISTAKES.md mechanism it is designed against, and the file it
  produces.** Items are ordered cheapest-decisive-first (the record's own rule) and
  grouped into phases with explicit gates; each phase ends with a decision the
  author can take without an agent.
- Milestones are small. "Build BED-S" is not a milestone; "write the BED-S generator
  spec with the goal set and the BOS convention, run its constant-label census on 512
  draws, print label sd / class balance / discard count, PASS if sd > 0.05 and every
  class non-empty" is.
- The shape definition and its propositions stay (short, exact); the beds / binds /
  controls / metrics are written as the plan's reference apparatus; prior art is a
  table; the negatives are a section the plan's rules cite; Limits are one paragraph.
- The plan must be developable in *any order* consistent with the DAG — mark which
  nodes are independent — because the author will pick them up at his own pace.
- A "first evening" list: the five cheapest items that each settle something (0 GPU-s
  Lean targets and manifests; the BED-S census; the ~6 GPU-s capped run the record
  priced three times and never took; the ChaCAL control at gamma = 0.9 on BED-M as a
  smoke test of the solve path; the gamma-pinning LR-test instrument spec).

## 1. Two facts from the instrument design (RUN, one draw, s = 32, float64) that every later planet must respect

**F1 (BOS is absorbing by construction).** On any causal softmax `P`, row 0 is `e_0`
(`P_00 = 1`), so position 0 is an absorbing state whether or not it is declared. With
constraint and goal sets declared but BOS undeclared, `rho(Q) = 1.0` and `I - Q` is
singular; with BOS declared inside the goal set, `rho(Q) = 0.687` on the same draw.
**BED-S must declare BOS inside a boundary set** (the record's value-zero BOS sink,
`V15Fork.Asink`, is the same object seen from the value side), and the paper must say
which set. Designed against V-25 (a hypothesis no draw satisfies) and D-3.

**F2 (constraints must precede the query).** On a causal `P` walks move to `j <= i`,
so a constraint position placed after the query is unreachable: `q = 0.0` exactly
for it. BED-S places every constraint and the goal before the query position and
prints reachability at construction (V-8).

Also confirmed RUN in that pass: the C6 displacement identity
`dz = (I - gP')^{-1}(dV + g dP z)` to `1.2e-15`; Sherman–Morrison closed form vs
re-solve `1.2e-15`; `V = const` gives `dz = 0` bitwise (planted identity) and
`V ~ N(0,1)` gives `max|dz| = 0.1096` (rejection region); displacement is zero before
the intervened row (`4.4e-16`, forward-only consequences); reach-avoid sum-to-one on
`T` within `7e-16` with a goal set and `max_k q^(k) >= 1/K` (min `0.605` at `K = 3`)
without one; discounted reads at `g = 0.6` sum to `[0.136, 0.340]` (the delay share must
be printed beside every safest-move reading); the mixing matrix with absorbing rows
has min entry `0.0` and row sums `1/(1-g) = 2.5` (the read must carry the `(1-g)`
factor or declare its row sums).

## 2. Files from the first pass that exist and are inputs

`$SCRATCH/design/design_instrument.md` (525 lines, apparatus for the eleven NOT-FOUND
components) and `$SCRATCH/design/design_falsify.md` (fourteen bets with counters and
killers). `design_theory.md` was never written (session cap) and is produced fresh.
