# The pivot-exclusion confound, and the falsifier that decides it

Written before `twin_plus` or `settled_plus` produced a single number. Owner:
Chase. Instrument: `scale/etask_k5e.py` (class swap, no bucket, no lock, so it
does not disturb the in-flight pre-registered ladder).

## 1. The confound, verified in the code and then measured

`ceq/bench.py:154` builds the causal mask as `torch.ones(s, s).tril(-1)` —
**strictly** causal, so row `i` reads only `j <= i-1`. `scale/m3_quintuple.py:129`
selects pivots with `exclude=(0, s-1)`, so the largest legal pivot is `s-2`, and
row `s-2` reads only `j <= s-3`.

**Therefore `v[s-2]` is invisible to every pivot row.** Softmax's own row is
`s-1`, which reads `j <= s-2`, so it sees that token directly.

`make_equilibrium_batch` at `t* = 1` puts `head = s-2` and zeroes `b[s-1]`, so
the label is concentrated on exactly the token the pivot rows cannot see.

Measured here, `e3_t1`, seed 0, `s=64`, perturbing `x[:, s-2, :]` by `+100`:

```
softmax row s-1 (a@x) moves   100.216400     <- DIRECT VALUE
pivot av moves                  2.060300     <- weight channel only
pivot log_gate moves           11.198860     <- weight channel only
```

So `0.819665 / 0.923118 / 0.978314` at `t* = 1` is **not a clean measurement of
settling**. It is partly a measurement of which cell can see the answer token.

This also explains `f`: `ceiling(1, 2 hops) = 0.000000`, measured, so the hop
budget left the shortfall entirely unexplained. This is what was binding.

**Root cause is a rationale, not a bug.** `scale/arm_s.py:104`: *"The query row
itself is excluded because a pivot reading of the row being settled is not an
independent reading of it."* Defensible; it became a capability ceiling.

## 2. THE PRESCRIBED LIFT DOES NOT WORK, AND THIS IS WHY IT IS NOT USED

The move as relayed was `exclude=(0,)` so that `p = s-1` becomes legal.
**Legal is not selected.** `pivot_probe.select_pivots` is a plain top-k over the
key-norm; `exclude` only sets entries to `-inf`. Measured over 32 drawn examples
at `e3_t1`, seed 0, untrained keys:

```
rank of s-1 by key-norm (0 = best):  min 10   median 19   max 27
in the top-8 for 0 of 32 examples
```

`exclude=(0,)` therefore changes **nothing** in `0 of 32` examples. Shipping it
as the fix would be a change that moves no number, reported as a fix — the
fifteenth vacuous control.

**THE LIFT USED INSTEAD RESERVES THE SLOT.** `k-1` pivots are content-selected
from `exclude=(0, s-1)` exactly as now, and `s-1` is **appended unconditionally**.
`k` is unchanged, the other `k-1` pivots are still content-selected, and `av`
now contains the softmax cell's own row by construction rather than by luck.

## 3. The mechanism bind — must fire BEFORE any training number is read

With the slot reserved, the `s-1` row of `av` **is** the softmax operator's row
`s-1` applied to `v`. So

    av[:, slot_of(s-1), :]  ==  (softmax_operator(q, k) @ x)[:, s-1, :]

to float tolerance. If that does not hold, the lift did not do what it claims
and no trained number from it may be read. `softmax` is then an **interior
point** of `twin_plus`'s function class: an `alpha` concentrating on that slot
reproduces the softmax reading exactly. That is the minimum condition for the
comparison to mean anything.

## 4. THE FALSIFIER, fixed before the run

`e3_t1`, `n_train = 2048`, `n_eval = 2048`, seed 0, `k = 8`, all other
parameters as the ladder.

Reference values already on disk (`results/etask_k5e_t1.txt`, same geometry,
same seed): `softmax 0.819665`, `twin 0.923118`. The gap is

    0.923118 - 0.819665 = 0.103453        half of it = 0.051727
    the threshold twin_plus must reach  = 0.923118 - 0.051727 = 0.871391

| outcome | what it licenses |
|---|---|
| `twin_plus > 0.871391` — closes **less than half** the gap | **THE MOVE DIES.** The exclusion was not what was binding; the deficit is optimisation or capacity. No re-reading of the ladder is licensed and the e3 numbers stand as they are. |
| `twin_plus <= 0.871391` **and** `settled_plus` within the ladder's own resolution of `twin_plus` | **The exclusion was binding, and settling still buys nothing once it is lifted.** Settling retires on clean ground and the `+6` branch fires **unconfounded**. A real result. |
| `twin_plus <= 0.871391` **and** `settled_plus` beats `twin_plus` beyond that resolution | The confound was masking a real settling effect. Escalate to 5 seeds on `t1`/`t2` before any claim — one seed decides nothing here. |
| the §3 mechanism bind fails | **No trained number is read at all.** The lift is broken, not the theory. |

**One seed. No interval.** This is a screen, not a reading: it decides whether
the 5-seed confirmation is worth `~2.5 h`, nothing more. No verdict about
settling ships from this file.

## 5. Cost, and what it does not touch

Measured at this geometry (`results/etask_k5e_t1.txt`): `twin 298.8 s`,
`settled 402.6 s`. The falsifier is `twin_plus + settled_plus ~ 700 s ~ 12 min`;
`softmax` and `twin` are already on disk at this seed and are **not** re-run.
The 5-seed `t1`/`t2` confirmation is `~2.5 h` and is spent **only if the move
survives**.

**The in-flight ladder is not touched.** This runs through `scale/etask_k5e.py`,
which is deliberately not bucketed and takes no lock, so the pre-registered
ladder completes exactly as registered.

## 6. What may NOT be concluded until this runs

`LOOP_PROMPT.md` §8: **no "softmax beats the equilibrium arms" headline ships
before this falsifier runs.** The e3 ladder keeps running, keeps printing
`SHAPE NOT READABLE` below three rungs, and keeps the `print_scope` caveat.

Two readings from the literature that bear on it, recorded so the result is not
over-read:

* **Softmax is Bayes-optimal on exactly the `t* = 1` task shape** — single-location
  regression, one softmax layer provably solves it, linear attention provably
  falls short (`arXiv:2410.01537`, ICLR 2025). `e3_t1` attacks softmax at its
  proven optimum, so a loss there is much weaker evidence than it looks.
* **The hop wall is depth and it is quantified** — `arXiv:2402.09268` Thm 4.2:
  `hop_k` needs `L = floor(log2 k) + 2`. These arms are depth 1; `t* = 8` needs
  about 5. An arm that trains onto its 2-hop ceiling and evaluates far above it
  is memorising noise, not failing to settle.

---

# RESULT — the move dies

`e3_t1`, `n_train = 2048`, `n_eval = 2048`, seed 0, run `2026-08-26 15:19:05`,
`results/etask_k5e_plus.txt`:

```
       e3_t1    1 twin_plus   4769  1.001431  0.938728    BEATS BAR   371.9
```

| quantity | value |
|---|---|
| `softmax` (reference, on disk) | `0.819665` |
| `twin` (reference, on disk) | `0.923118` |
| **`twin_plus`** | **`0.938728`** |
| gap `twin - softmax` | `0.103453` |
| pre-registered threshold | `<= 0.871391` |
| **fraction of the gap closed** | **`-0.1509`** |

`0.938728 > 0.871391`. **Outcome row 1 of §4 fires: THE MOVE DIES.** The lift did
not close half the gap; it did not close any of it. `twin_plus` is `0.015610`
*worse* than `twin` — it moved 15 % further from softmax.

**The confound is real and it is not the binding constraint.** Both statements
are now measured. The exclusion demonstrably hides the answer token
(`100.216400` against `2.060300`), and removing it demonstrably does not help.
Whatever costs the pivot cells `0.103453` at `t* = 1` is not pivot access.

Per §6, no re-reading of the e3 ladder is licensed by this file, and the e3
numbers stand as they were taken.

## `settled_plus` — finished, and it changes nothing

The sentence that stood here said `settled_plus` was still running. It has since
finished, in the same run as `twin_plus` (`results/etask_k5e_plus.txt`,
`2026-08-26 15:19:05`, same geometry, same seed):

```
       e3_t1    1 settled_plus   4769  1.001455  0.956787    BEATS BAR   318.6
```

| quantity | value |
|---|---|
| `softmax` (reference, on disk) | `0.819665` |
| `twin` (reference, on disk) | `0.923118` |
| `settled` (reference, on disk) | `0.978314` |
| **`settled_plus`** | **`0.956787`** |
| `settled_plus` − `twin` | **`+0.033669`** (worse) |
| `settled_plus` − `settled` | `−0.021527` (better) |
| gap to `softmax` | `0.137122`, against `twin`'s `0.103453` |

**The verdict is unchanged, and it was always going to be.** Outcome row 1 of §4
fires on `twin_plus` alone; rows 2 and 3 are the only rows `settled_plus`
appears in and **both are conditioned on `twin_plus <= 0.871391`**, which is
false. So this number was unreadable under the pre-registration before it was
taken, and it is unreadable now. It is recorded because the document promised
it, not because it licenses anything.

**The one thing that must not be read out of it.** `settled_plus` is `0.021527`
better than plain `settled`, and that is the only direction in this table where
lifting the exclusion helped anything. It is not evidence. It is one seed with
no interval on the difference, against a measured seed spread of `sd 0.064106`
for `settled` — three times the effect. Nothing here separates it from noise,
and no re-reading of the e3 ladder is licensed by this file (§6). Lifting the
exclusion still leaves `settled_plus` `0.033669` worse than plain `twin`, so the
conclusion stands in the form §4 fixed it: **whatever costs the pivot cells
`0.103453` at `t* = 1` is not pivot access.**
