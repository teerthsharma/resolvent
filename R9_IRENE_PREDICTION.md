# R9 — the competing prediction, filed before any R9 number existed

**Written 2026-08-30, R9 iteration 1, by the IRENE seat (the alternative).**
**No R9 cell has run. No vector-valued label exists in this repository at the
time of writing** — `scale/negation_scope.py:274 equilibrium_oracle` returns
shape `[n]`, and every arm's `forward` ends `[:, s - 1]`. Nothing below is
computed from an R9 measurement, because there is none. Every number is either
read out of the **completed scalar ladder** (`results/e_ladder_reading.txt`) or
derived in closed form from the batch builder, and both provenances are marked.

This file exists so that after the R9 numbers land, the table can print which
prediction the measurement matched — the contract's or this one.

---

## 1. The contract's prediction, stated as the contract states it

From `FINDINGS.md` C1–C5: every label in this repo is a scalar point prediction
at position `s-1` (`scale/m3_quintuple.py:311`), which is exactly the shape
where one softmax layer is provably Bayes-optimal (`arXiv:2410.01537`). The move
is a **vector-valued readout** — drop the `[:, s-1]` index for an `[n, s]`
label at zero new parameters — on the grounds that this leaves the regime where
softmax is proven optimal.

> **The contract predicts the pivot-routed arm's margin over softmax grows once
> the label is vector-valued.**

## 2. The competing hypothesis — one, committed to

> **DILUTION, AND WORSE THAN DILUTION. The vector-valued readout does not
> multiply the contrast by `s`; it divides it. At every position but one the
> pivot arms and softmax compute the same tensor, and at `s − t*` of the `s`
> positions the label is a zero-hop copy of an input channel that any arm reads
> for free. The pooled margin therefore shrinks toward zero at every rung, and I
> predict it lands below the pre-registered 13-seed resolution `0.027260` at all
> four — which is the floor the E ladder already fixed as "too small to matter
> at this scale".**

This is the third candidate in the dispatch, sharpened. The dispatch put it as
"the early positions carry almost no information". That is the weaker version
and it is not what the builder does. The builder does something more damaging:
the early positions carry **complete** information, legible at zero hops, from
the input the arm is already holding.

### 2a. Why the arms are identical at 63 of 64 positions — `READ`

`scale/m3_quintuple.py:302-306`, the whole forward:

    a = bench._softmax_operator(q, k)
    z = x + a @ x
    if self.base_cell not in ("softmax", "glance"):
        z = z.clone()
        z[:, s - 1] = x[:, s - 1] + self._alpha(q, k, x).to(z.dtype)
    h = self.mlp(z)
    return self.readout(h).squeeze(-1)[:, s - 1]

**The pivot term is written into row `s-1` and no other row.** The class
docstring says so itself at `:257-261`: *"Only row s-1 can reach the output
[...] so only that row is replaced. The other rows are left exactly as the
shipped forward produced them."* That was a correctness argument while the
readout was scalar. Under a vector readout it becomes the refutation: at every
position `p ≠ s-1`, `settled`, `twin`, `argmax` and `softmax` evaluate the
**same expression** `x + a@x`. The architectural difference between the arms
occupies exactly `1` of `s = 64` positions. Any difference the vector lane finds
at the other 63 is a training-trajectory difference, not a capability
difference, and the paired bootstrap was never designed to read one.

### 2b. Why `s − t*` of the positions are a free copy — `RUN`, steps shown

`scale/negation_scope.py:388-395`, `make_equilibrium_batch`:

    head = s - 1 - t
    a[:, :head + 1] = 0.0
    b[:, s - 1]     = 0.0

and the oracle is the forward scan at `:290-291`:

    for i in range(x.shape[1]):
        z = a[:, i] * z + b[:, i]

The natural `[n, s]` label — the one you get by not throwing away the loop's
intermediate state — is `z_p` at every `p`. Then:

* For `p ≤ head`: `a_p = 0`, so `z_p = b_p` **exactly**. And `b` is written into
  the input at `:393` as `x[:, :, CH_FLIP]`. **The label at those positions is a
  channel of the input at those positions**, reachable through the residual
  `z = x + a@x` with zero hops, by every arm, identically.
* For `p = head + m`, `1 ≤ m ≤ t* - 1`: `Var(z_p) = m + 1`.
* For `p = s - 1`: `b_{s-1} = 0`, so `Var(z_{s-1}) = t*` — the existing scalar
  label, and the only position that needs the full hop chain.

At `t* = 1`, `head = 62`: **63 of 64 positions are the free copy.** At
`t* = 32`, `head = 31`: 32 of 64.

This is the exact defect that `b[:, s-1] = 0.0` was introduced to remove.
`scale/negation_scope.py:363-376` records why, in the author's own words: with a
driver at the read position, *"a `1 / (t*+1)` share of the label's variance was
legible before a single training step and the untrained arm read BELOW the
bar"*, and `m3_capability.py` *"aborted with INSTRUMENT BROKEN on three of the
five rungs"*. The fix zeroed `b` at **one** position. A vector readout re-opens
the same hole at `s − t* − 1` other positions, none of which was ever zeroed.

**Checked against the shipped builder, not assumed.** Drawing
`make_equilibrium_batch(512, 64, 8, d_model=24, seed=0, t_star=t)` and running
the scan of `:290-291` over its own `x`:

    t* = 1   head = 62   z_p == b_p for p <= head: True (torch.equal, exact)
                         copy positions 63 of 64
                         scan's last element == the shipped scalar label: True
    t* = 8   head = 55   z_p == b_p for p <= head: True (torch.equal, exact)
                         copy positions 56 of 64
                         scan's last element == the shipped scalar label: True

The equality is bitwise, not approximate. The second line of each block also
settles a question about the label choice: **the prefix scan's last element is
the shipped scalar label**, so the `[n, s]` label in §6 is a strict superset of
the one the repo already trains against, and the vector lane cannot be accused
of changing the target at `s-1`.

### 2c. The dilution factor, in closed form — `DERIVED`, checked two ways

Total label variance, summing §2b:

    SumVar(t*) = (s - t*)·1  +  [t*(t*+1)/2 - 1]  +  t*  =  s - 1 + t*(t*+1)/2

At `s = 64`, and `w := Var(z_{s-1}) / SumVar` — the share of the pooled label
variance sitting at the **only** position where the arms differ:

| rung | `t*` | free-copy positions | `SumVar` | `w` | `sqrt(w)` |
|---|---|---|---|---|---|
| `e3_t1`  | 1  | 63 | 64  | 0.015625 | 0.125000 |
| `e3_t2`  | 2  | 62 | 66  | 0.030303 | 0.174078 |
| `e3_t8`  | 8  | 56 | 99  | 0.080808 | 0.284268 |
| `e3_t32` | 32 | 32 | 591 | 0.054146 | 0.232692 |

**Between 1.6 % and 8.1 % of the vector label's variance lives at the one
position where a pivot arm is architecturally distinguishable from softmax.**

Two paths that fail differently. The closed form above is a recursion on
`Var(z_p)`. The second path draws `4 x 10^5` instances of the builder's own
construction and measures the per-position variance directly — no formula
enters it. They agree:

| `t*` | `SumVar` closed | `SumVar` drawn | `sqrt(w)` closed | `sqrt(w)` drawn |
|---|---|---|---|---|
| 1  | 64  | 63.9771  | 0.125000 | 0.124786 |
| 2  | 66  | 65.9948  | 0.174078 | 0.174060 |
| 8  | 99  | 99.1311  | 0.284268 | 0.284668 |
| 32 | 591 | 589.6929 | 0.232692 | 0.232463 |

Largest disagreement in `sqrt(w)` is `0.00040`, at `t* = 8`. The predicted
margins in §3 move by less than `1.6 x 10^-5` if the drawn column is used
instead, which is four orders below the `0.027260` floor they are compared
against, so the choice of column cannot decide the prediction.

---

## 3. The instrument, and the predicted numbers on it

**Instrument.** The `[n, s]` lane of `scale/m3_quintuple.py` with the
`[:, s-1]` index dropped from `readout`, against the prefix-scan label `z_p`,
at the E ladder's own geometry — `s=64 d=24 steps=150 n_train=2048 n_eval=2048
k_piv=8 beta=0.5 t_max=21 n_neumann=21`, seeds `0 1 2 3 4`, own journal and own
weights directory per `FINDINGS.md` §C. Reported as pooled NRMSE over all
`n × s` entries, and as `delta = NRMSE_softmax − NRMSE_arm` through
`scale/m3_synthetic_settled.py::contrast`, `n_boot=10000`, `seed=0`, strict at
zero. **Positive delta means the pivot arm wins**, matching `contrast()`.

**Baseline — `READ`, `results/e_ladder_reading.txt`.** The scalar margins the
completed ladder printed, in the same sign convention (`g` and `g_twin` are
`softmax − settled` and `softmax − twin`):

| rung | `settled` scalar margin | `twin` scalar margin |
|---|---|---|
| `e3_t1`  | −0.197059 | −0.161034 |
| `e3_t2`  | −0.060746 | −0.043531 |
| `e3_t8`  | +0.035984 | +0.040268 |
| `e3_t32` | +0.059654 | +0.043620 |

### PREDICTION 1 — the margins, with numbers

If the free-copy positions are solved to a small residual, the arm difference is
confined to position `s-1` and the pooled margin is the scalar margin scaled by
`sqrt(w)`. That is the **best case for the contract**: any residual left on the
copy positions is common to both arms and enters the pooled NRMSE additively
under the square root, shrinking the margin further. So these are **upper
bounds in absolute value**, not point estimates.

| rung | predicted `settled − softmax` | predicted `twin − softmax` |
|---|---|---|
| `e3_t1`  | **−0.024632** | **−0.020129** |
| `e3_t2`  | **−0.010574** | **−0.007578** |
| `e3_t8`  | **+0.010229** | **+0.011447** |
| `e3_t32` | **+0.013881** | **+0.010150** |

**Headline: all eight are below `0.027260` in absolute value.** The largest is
`0.024632`. The contract says the margin grows; I say every one of the eight
lands inside the E ladder's own pre-registered 13-seed resolution floor, where
`LOOP_PROMPT.md` §1.8's *"too small to matter at this scale and is not chased"*
applies. The vector label does not sharpen the contrast. It dissolves it.

### PREDICTION 2 — the pooled NRMSE collapses at the shallow rungs, and it is the copy

Bracketing the free-copy positions at per-position NRMSE `≤ 0.30` and every
other position in `[0.90, 1.20]` — the range the scalar ladder already reads at
`s-1`:

| rung | predicted pooled NRMSE, every arm | current scalar NRMSE (`settled` / `softmax`) |
|---|---|---|
| `e3_t1`  | **[0.113, 0.333]** | 0.994399 / 0.797339 |
| `e3_t2`  | **[0.222, 0.415]** | 1.013958 / 0.953212 |
| `e3_t8`  | **[0.593, 0.822]** | 1.096009 / 1.131993 |
| `e3_t32` | **[0.875, 1.169]** | 1.103711 / 1.163365 |

**The trap this predicts, named now so it cannot be reported as a discovery.**
At `t* = 1` and `t* = 2` the vector lane will print pooled NRMSE numbers two to
seven times better than anything the scalar ladder ever produced, on rungs where
the scalar ladder printed row **G** — *credited nothing, at or above
predict-the-mean*. That improvement is **not** capability. It is 62–63 of 64
positions being an identity copy of `x[:, :, CH_FLIP]`. Three of the four rungs
also cross from above `1.0` to below it, which would read as row G clearing.
**Row G does not clear.** The bar moved because the label changed, not because
an arm learned anything, and any R9 report that quotes a sub-1.0 vector NRMSE
without the free-copy count beside it is overclaiming.

### PREDICTION 3 — the RED gate trips

`scale/m3_capability.py:319` requires the **untrained** arm to sit at or above
NRMSE 1.0 (`red_ok = ... r0t >= 1.0 and r0e >= 1.0`) and prints
`INSTRUMENT BROKEN` at `:322-324` otherwise. The mechanism that tripped it
before — a driver-free read position whose label is legible through the residual
at zero hops — is present at `s − t* − 1` positions under the vector label
instead of at one. I predict:

> **The 0-step pooled NRMSE reads below 1.0 at `t* = 1`, and the vector lane
> aborts `INSTRUMENT BROKEN`, unless the builder is changed to zero `b_p` at
> every `p` where `a_p = 0` — that is, `b[:, :head + 2] = 0.0` rather than
> `b[:, s - 1] = 0.0` alone.**

The magnitude is not predicted: the recorded precedent is `0.993760` train /
`0.993600` eval at a `1/(t*+1) = 0.5` legible share
(`scale/negation_scope.py:369-371`), and the vector label at `t*=1` puts the
legible share at `63/64`. Larger share, same mechanism, unknown constant. **The
direction is the prediction; the size is not.**

If the builder *is* changed to zero the drivers' partners, the free-copy
positions become identically zero, `y_p` has `sd = 0`, and
`scale/negation_scope.py:675-676` returns `nan` for every one of them. That
outcome is covered by the fall-through row in §5.

---

## 4. What would falsify me — each one binary, on the same instrument

1. **Any** `settled − softmax` or `twin − softmax` vector margin whose 95 %
   paired percentile CI excludes zero **and** whose `|delta| ≥ 0.027260`, at any
   rung. One such cell and PREDICTION 1 is dead.
2. The margin at `e3_t32` exceeding its scalar value `+0.059654`. That is the
   contract's claim in its cleanest form — margin grows — and it kills me
   outright.
3. Pooled vector NRMSE at `e3_t1` above `0.50`. My dilution arithmetic assumes
   the copy positions get solved; if they do not, §2c's variance shares are the
   wrong weights and PREDICTION 1's scaling does not follow.
4. The 0-step pooled NRMSE reading at or above `1.0` at `t* = 1` with the
   builder unchanged. PREDICTION 3 dies; 1 and 2 survive it.
5. A demonstration that the pivot term reaches a position other than `s-1` in
   the R9 arm. §2a is the load-bearing read; if R9's forward writes `_alpha`
   into every row, the "63 of 64 identical" argument does not apply and my
   `sqrt(w)` scaling is wrong in the contract's favour.

## 5. The fall-through row for R9 — written before R9's table exists

Whatever outcome table R9 ends up with, this is the row that catches "none of
the above". It is written now for the reason row **H** was written at 13:35:
a table whose rows all condition on the arms being comparable has no row for the
case where they are not.

> **Ω** | The two lanes do not read the same task. Any of: (a) the pooled vector
> NRMSE of **every** arm moves by more than `0.10` from its scalar value at any
> rung while every pairwise margin stays inside `±0.027260`; or (b) the free-copy
> positions `p ≤ head = s-1-t*` are not reported with their count and their own
> per-position NRMSE beside the pooled figure; or (c) any position's label has
> `sd = 0`, so `scale/negation_scope.py:672-677` returns `nan` and the pooled
> figure is an average over non-numbers | **NOT A CONTRAST, AND NOT READ AS
> ONE.** The vector lane is a **new task**, not a re-reading of `e3`. No margin,
> no NRMSE and no row from the scalar ladder may be carried into it, in either
> direction, and no arm may be credited or retired on it. Under (c) the reader
> prints `NOT READABLE` and no value, because `nan` propagating through a mean is
> the same failure as reporting a missing rung as a null. The obligation this row
> creates is a **matched-position control**: the same two arms compared at
> position `s-1` **only**, inside the vector lane, which must reproduce the
> scalar ladder's margins. If it does not, the vector lane's instrument is broken
> and nothing it printed is read.

Row Ω fires **before** any row of R9's table is evaluated, the way `M2_TRAINED`'s
patched row **0** does in `PREREGISTRATION_HOLE_AUDIT.md` §H-5.

---

## 6. Limits, collected once

The `[n, s]` label does not exist yet, so §2b's prefix scan `z_p` is *my* choice
of what "drop the `[:, s-1]` index" means — the loop's own intermediate state at
`scale/negation_scope.py:290-291`, which is the only reading that costs zero new
code. If R9 defines the vector label some other way — re-running the oracle per
position with `head` recomputed, or a windowed label — then §2c's variance shares
are wrong and **PREDICTION 1's numbers must be recomputed from the same formula
against the label actually built, before this file is scored.** The formula, not
the eight numbers, is what I am committing to; the eight numbers are the formula
evaluated at the label I expect. §2a is independent of that choice and survives
any of them.

The scalar margins in §3 are seed-means over 5 seeds at `n_train = 2048` and
three of the four rungs were credited **nothing** by row G — `settled` and `twin`
both sit above predict-the-mean at `t* = 2, 8, 32`. Scaling an uncreditable
margin by `sqrt(w)` produces an uncreditable predicted margin. PREDICTION 1 is
therefore a prediction about **what the instrument will print**, not about a
capability, at every rung except `e3_t1`. I predict the printed values and claim
nothing about what they would license.

The per-position NRMSE brackets in PREDICTION 2 (`≤ 0.30` on copies, `[0.90,
1.20]` elsewhere) are **not** derived. They are ranges chosen to bracket what the
scalar ladder reads at `s-1`, and the copy bound is a judgment that a trained
linear readout over `d = 24` channels solves an identity copy well. If the copy
positions read `0.60` instead of `0.30`, the `t*=1` band moves to roughly
`[0.11, 0.62]` and falsifier 3 stops being a clean test. That is the softest
number in this file and it is flagged rather than buried.

**No cell was trained to produce any figure above.** This seat holds neither
execute nor adjudicate. Two things were executed, both read-only draws against
the shipped builder and neither of them a training step: the `torch.equal` check
in §2b and the `4 x 10^5`-instance variance draw in §2c. Everything else is
`READ` against a cited line or `DERIVED` from one by arithmetic shown in full.

The adversarial pass on my own strongest claim, §2a: would it survive the logic
being deleted? If `scale/m3_quintuple.py:304` were changed to write `_alpha`
into every row rather than row `s-1`, §2a is false and the `sqrt(w)` scaling
goes with it. That is falsifier 5, and it is a **one-line change to R9's own
arm** — the cheapest way for the contract to beat this prediction is to make it
inapplicable rather than to out-measure it. If R9 takes that route, this file
scores as neither right nor wrong and says so; it is not a prediction about an
arm nobody has written.
