# C9, corrected — the pinned bed for LAW L-NEVER

C9 is void. This document does not reinstate it; it replaces the
under-specified bed with a fully-specified one and restates the two
must-fires in the terms float64 can actually meet. Ruling on whether the
corrected certificate counts as C9 passing is left to a separate reviewer.

**Replacement route.** The object under repair is the arithmetic that
`hitting_time_read`/`committor` must get right for any causal-mask
consequence read in `ceqjepa/operator.py` — the piece of the CEQ apparatus
that turns a transition structure into an absorption verdict. Pinning the
bed so its reference numbers follow from a stated construction, rather than
from six free entries chosen to match a target, is what lets that arithmetic
serve north-star condition (1), understands causality: a causal read is only
evidence of understanding when the number it produces is reproducible from
the specification, not fitted to it.

## Why the old certificate was void

The specification fixed the labels — closed class `F = {0, 1, 2}`, absorbing
outcomes `WIN = 6` and `DRAW = 7`, transient states `{3, 4, 5}` — and never
fixed the transition weights. The transient block carries six free entries
(each of states 3, 4 and 5 distributes its row over up to three
destinations) against three equations (one target `r_gamma` per state), so
the system is underdetermined by three degrees of freedom. A value that is
not determined by the specification can be hit by any of infinitely many
weight choices that satisfy every structural claim identically; matching one
particular decimal triple this way is fitting the weights to the answer, not
reproducing the answer from the bed. The attending exhibited two such beds —
one reading `1.0526e-04 / 1.5512e-04 / 1.9564e-04` and another, using a
self-loop construction, reading `3.0600e-04 / 2.2900e-04 / 1.6900e-04` to
every printed digit of the paper's own numbers — which is the demonstration
that the triple does not pin the bed.

## The repair: pin the bed

The repair adopts the self-loop construction as part of the certificate,
which removes the six-against-three gap by construction rather than by
tuning: each transient state is given a **single** free parameter (its own
self-loop probability) and no transient-to-transient coupling, so the six
entries collapse to three unknowns before any target is applied, and the
three `r_gamma` targets then pin those three unknowns exactly. No parameter
is chosen by search; each is solved in closed form from the target it is
assigned to.

### The bed, completely

Eight states, index = label, `dtype = torch.float64`.

- **Closed class** `F = {0, 1, 2}`: each is pure self-identity, `W[f, f] = 1`,
  all other entries in that row `0`. Once entered, mass never leaves.
- **Absorbing outcomes**: `WIN = 6` (`W[6,6] = 1`) and `DRAW = 7`
  (`W[7,7] = 1`), two distinct sinks, neither reachable from the other.
- **Transient states** `T = {3, 4, 5}`: state `i` places mass only on itself
  and on `WIN` — `W[i, i] = p_i`, `W[i, WIN] = 1 - p_i`, and `W[i, j] = 0` for
  every other `j` (no state-to-state coupling among 3, 4, 5, and no leak to
  `DRAW` or to `F`). This keeps the transient-transient block diagonal, hence
  causal (lower-triangular) trivially.

That is all eight rows; nothing above is left free.

### Pinning the three self-loop probabilities

`state_solve` computes `z = (I - g Q)^{-1} V` with `V = 1`, and
`r_gamma = (1-g) z`. For a diagonal transient block with self-loop `p_i` and
no coupling, this closes in one line: `r_gamma_i = (1-g) / (1 - g p_i)`.
Solving for `p_i` at the target `r_i`:

```
p_i = (1 - (1-g) / r_i) / g
```

At `gamma = 0.9999`, `(1-g) = 1e-4`, target triple `r_3 = 3.06e-4`,
`r_4 = 2.29e-4`, `r_5 = 1.69e-4` (float64):

| state | `p_i` (float64) | `1 - p_i` (float64) |
|---|---|---|
| 3 | `0.6732699413732584` | `0.32673005862674165` |
| 4 | `0.5633751148041048` | `0.43662488519589515` |
| 5 | `0.4083248561543196` | `0.5916751438456804` |

### Reference values, as they follow from the pinned bed

Run through the shipped `state_solve`/`committor` on this bed, torch
`2.14.0+cpu`, float64:

`r_gamma` at `gamma = 0.9999`, transient states, via
`(1-g) * state_solve(Q, ones, g)`:

| state | `r_gamma` (float64) |
|---|---|
| 3 | `0.000306` |
| 4 | `0.00022899999999999996` |
| 5 | `0.000169` |

These follow from the construction and the closed-form `p_i` above — they
are not independently chosen. The triple in the task's "reference instance"
is recovered because the bed was pinned to produce it, and that pinning is
now part of the certificate rather than hidden behind an unstated choice of
weights.

**The triple is definitional, not predicted.** The transient block is
diagonal by construction — no coupling among states 3, 4 and 5 — so `Q` is
diagonal and the general `r_gamma = (1-g) (I - g Q)^{-1} V` collapses
state-by-state to the one-line identity `r_gamma_i = (1-g) / (1 - g p_i)`
used above to solve for `p_i`. Running `state_solve` on that same diagonal
`Q` and getting the target triple back checks that `state_solve` correctly
implements the identity it was handed; it is not evidence about causal
structure between transient states, because this bed has none to detect —
3, 4 and 5 cannot influence one another by construction.

## MUST-FIRE 1: `r_gamma = 1` on `F`

`state_solve` on the transient block `{0,...,5,7}` (`WIN` absorbing, `DRAW`
carried as a self-looping row since it is not declared absorbing here) gives,
on `F = {0, 1, 2}`:

| `gamma` | `r_gamma` on F (float64, all three states identical) | max `|.-1|` |
|---|---|---|
| 0.9 | `1.0` | `0.0` |
| 0.99 | `1.0` | `0.0` |
| 0.999 | `1.0` | `0.0` |
| 0.9999 | `0.9999999999999999` | `1.1102230246251565e-16` (one ulp) |

Nine of twelve guarded entries (three states times the first three gammas)
are bitwise `1.0`. The remaining three, at `gamma = 0.9999`, are one ulp
below `1.0` in float64 — not bitwise equal to it. "Bitwise `1.0` at every
gamma" is a claim float64 cannot deliver at the fourth gamma; the true
statement is bitwise at the first three and within one ulp at the fourth,
with the ulp count given above rather than a `< 1e-12` tolerance standing in
for the word "bitwise."

## MUST-FIRE 2: committor to `WIN`, off `F`

`committor(W, [F0, F1, F2, WIN, DRAW])`, column `WIN`, read at row = state
label (the function embeds absorbing rows back to `eye(k)` at their own
index, not at their position in the argument list):

| state | committor to WIN (float64) |
|---|---|
| 0 (F) | `0.0` |
| 1 (F) | `0.0` |
| 2 (F) | `0.0` |
| 3 (T) | `1.0` |
| 4 (T) | `1.0` |
| 5 (T) | `1.0` |
| 6 (WIN) | `1.0` |
| 7 (DRAW) | `0.0` |

Eight guarded entries evaluated: three bitwise-zero on `F`, three
bitwise-one on the transient states, one bitwise-one at `WIN` itself, one
bitwise-zero at `DRAW`. "Committor `= 1.0` off `F`" is false as a universal
claim — `DRAW` is off `F` and reads bitwise `0.0`, as a distinct absorbing
outcome must. That `0.0` is not evidence about this bed's dynamics: `committor`
never reads an absorbing state's own row of `P` at all — it hard-codes every
declared absorbing index to the one-hot row of `eye(k)` at that index,
regardless of what the matrix says there. (Checked directly: setting
`W[DRAW, WIN] = 0.5` in this bed still returns `0.0` for DRAW's committor to
WIN.) So `DRAW` reads `0.0` at `WIN` by embedding convention, not because
"no mass in this bed ever reaches WIN from DRAW." The claim only holds read
as "off `F`, among `{3, 4, 5, WIN}`"; restated that way, all four entries are
bitwise `1.0`.

## The arithmetic wall

`committor(W, [WIN, DRAW])` — `F` folded into the transient block instead of
declared absorbing — raises `SingularTransientBlockError`, confirmed by
running it: the transient-block diagonal at each of the three `F` positions
is exactly `1.0` (closed, causally self-only), so `I - Q` there is exactly
singular. This is the guard firing as a refusal rather than returning a
number, on the pinned bed exactly as on the original.

## Limits

Wall (a), `r_gamma = 1` on `F`, is a one-ulp statement at `gamma = 0.9999`,
not an exact one: float64 through the shipped triangular path returns
`0.9999999999999999` there, `1.110e-16` below `1.0`, not bitwise equal to
it. Separately, must-fire 2's companion inequality — `r_gamma < 1e-3` on
every transient state, checked on the repo-canonical twin and on a
strict one-factor twin whose sole change is a finite logit floor `-Delta` in
place of structural `-inf` — holds unconditionally as a statement about wall
(b) (`W_10 > 0` for every finite `Delta`, an analytic fact about softmax on
finite input, not a measured one). The `r_gamma < 1e-3` half is a statement
about logit scale, not about softmax: whether it holds depends on where the
finite floor `-Delta` sits relative to the bed's other logits, and that is a
property of a specific twin construction, not of the softmax map. This
certificate does not pin that twin — it names no `n`, no absorbing set, no
base logits, and no command that reproduces one, and no test guards it — so
no crossover value is reported here. A crossover is unmeasured in this
certificate.
