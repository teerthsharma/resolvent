# V12 PRICING — the five gauge-positional arms, priced before the round runs

**Producer.** `scale/vgpe_flops.py`. Run it with `python scale/vgpe_flops.py`;
it prints every table below and ends in an `assert`-based self-check that passes
(`ALL CHECKS PASSED`). Every number in this document was read out of that run.
Nothing here was typed from a formula that was not executed.

**Independence.** This model does not import `scale/vgpe.py` and was written
without reading it. It is an analytic derivation from the block algebra upward,
so it functions as a cross-check on that implementation rather than a
restatement of it. `F_base` and the `twin` add-on are reused from
`scale/m3_flops.py:cell_terms`, which is the house's existing analytic-FLOP
accounting for exactly those terms.

**No clock.** Per the house convention (`scale/m3_flops.py:3-6`) wall-clock on
this box is PROVISIONAL / CONTENDED and is not a measurement. Nothing in this
document is timed.

**Forward only.** `scale/m3_flops.py`'s `BASE_TERMS` counts a forward pass; the
transport backward (roughly 2× its forward) is not counted here either. The
omission is identical across all five arms, so it cannot move an arm ordering.
It is collected with the other omissions in **Limits**.

**M-8 compliance** (`MISTAKES.md:495`). Every arm below is priced at its own
primitive rate. No cross-arm ratio is carried. The shuffled-gauge plant is not
priced as "same as VGPE" by assertion — it is derived independently and then
*shown* equal, in `scale/vgpe_flops.py` self-check F, at every `n` tested.

---

## Conventions, both printed, because the spec's numbers are not all in one

| tag | rule | source |
|---|---|---|
| **HOUSE** | one multiply-add = 2 FLOPs; a matmul `[m,p]@[p,q]` costs `2*m*p*q` | `scale/m3_flops.py:41`, shipped counter `scale/arm_s.py:457-460` |
| **OPS** | each multiply and each add = 1 FLOP, from the algebra's closed form | the convention the round spec's `768` is in |

Mixing them is how a 2.3× error hides. The spec's headline comparison mixes
them: `6,144` is OPS and `524,288` is HOUSE.

## Shape

`d = 256`, `s = 1024`, `depth = 8`, `|A| = 16`, `hidden = 4d = 1024`,
`k_piv = 16`, `n_neumann = 21`, `n = 1` (per example; all transport terms are
linear in `n`, the Cayley build is not). `|A|` and `hidden` are **not fixed by
the spec** — both are swept in the producer and the sweeps are below.

---

## 1. THE FIVE-ARM TABLE

Per training step, forward, HOUSE convention, blockwise Cayley,
`n=1 s=1024 d=256 depth=8 |A|=16 hidden=1024 k=16`.

| # | arm | `F_base` | twin add-on | transport | Cayley build | **TOTAL** | pos % |
|---|---|---|---|---|---|---|---|
| 1 | `twin + RoPE` (control) | 2,416,443,392 | 18,358,272 | 2,097,152 | 0 | **2,436,898,816** | 0.0861% |
| 2 | `twin + VGPE` | 2,416,443,392 | 18,358,272 | 6,291,456 | 262,144 | **2,441,355,264** | 0.2684% |
| 3 | `twin + VGPE-abelian` | 2,416,443,392 | 18,358,272 | 3,145,728 | 65,536 | **2,438,012,928** | 0.1317% |
| 4 | `twin + shuffled-gauge` | 2,416,443,392 | 18,358,272 | 6,291,456 | 262,144 | **2,441,355,264** | 0.2684% |
| 5 | `softmax + RoPE` (baseline) | 2,416,443,392 | 0 | 2,097,152 | 0 | **2,418,540,544** | 0.0867% |

Formulas, each evaluated at the shape above:

* `F_base = 4*n*s*d^2 + 4*n*s^2*d + 4*n*s*d*hidden + 2*n*s*d`
  = `4·1·1024·65536 + 4·1·1048576·256 + 4·1·1024·256·1024 + 2·1·1024·256`
  (`scale/m3_flops.py:43-63`, reused not re-derived).
* twin add-on `= n*[2*s*d + 2*(k+1)*s*d + 2*k^2*s + 2*k*s*d + 2*k*d]`
  = 18,358,272 at `k=16`; `= 0.7597%` of `F_base`
  (`scale/m3_flops.py:65-77,129-132,214`).
* transport, arms 1 and 5 (RoPE) `= n*s*2*APPLY(b=2) = n*s*4*d` = `1024·2048`.
* transport, arms 2 and 4 `= n*s*3*BLOCKMULT(b=4) = n*s*24*d` = `1024·6144`.
* transport, arm 3 `= n*s*3*BLOCKMULT(b=2) = n*s*12*d` = `1024·3072`.
* Cayley build `= |A| * CAYLEY_BLOCK(b)`, paid once per optimizer step because
  `Ω_a` are parameters; `= 16·16384` for `b=4`, `16·4096` for `b=2`.

**Per-node rates** (HOUSE / OPS), the quantity every arm total is built from:

| arm | prefix per node | naive per node | prefix saving |
|---|---|---|---|
| `twin + RoPE` | 2,048 / 1,536 | 2,048 / 1,536 | 1.000× (RoPE tabulates, never composes) |
| `twin + VGPE` | 6,144 / 5,376 | 32,768 / 28,672 | 5.333× |
| `twin + VGPE-abelian` | 3,072 / 2,304 | 16,384 / 12,288 | 5.333× |
| `twin + shuffled-gauge` | 6,144 / 5,376 | 32,768 / 28,672 | 5.333× |
| `softmax + RoPE` | 2,048 / 1,536 | 2,048 / 1,536 | 1.000× |

---

## 2. "768 FLOPs/edge at d=256" — REPRODUCES, BUT NOT FOR THIS ROUND'S ARM

The block primitives at `d=256`. Applying `U` to a vector and composing `U` with
`V` are the *same* primitive in the block algebra — a quaternion times a
4-vector and a quaternion times a quaternion are the same product — which is why
one row covers both.

| primitive | formula | HOUSE | OPS |
|---|---|---|---|
| apply/compose, `b=2` unit complex (ABELIAN) | `2*b*d` / `(2b-1)*d` | 1,024 | **768** |
| apply/compose, `b=4` unit quaternion (NON-ABELIAN) | `2*b*d` / `(2b-1)*d` | 2,048 | **1,792** |
| MATREP compose, `b=2` | `2*b^2*d` | 2,048 | 1,536 |
| MATREP compose, `b=4` | `2*b^2*d` | 8,192 | 7,168 |
| apply, dense `d×d` | `2*d^2` | 131,072 | 130,816 |
| compose, dense `d×d` | `2*d^3` | 33,554,432 | 33,488,896 |
| Cayley build, dense | `4*d^3` / `2*d^3` | 67,108,864 | 33,554,432 |
| Cayley build, `b=2` blockwise (upper bound) | `(d/2)*4*b^3` | 4,096 | 2,048 |
| Cayley build, `b=4` blockwise (upper bound) | `(d/4)*4*b^3` | 16,384 | 8,192 |

**`768` reproduces exactly — as the `b=2` ABELIAN rate under OPS.**

| parametrization | value | vs the claim |
|---|---|---|
| `b=2` abelian, OPS | 768 | **reproduces exactly** |
| `b=2` abelian, HOUSE | 1,024 | 1.33× |
| `b=4` quaternion, OPS | 1,792 | **2.333×** |
| `b=4` quaternion, HOUSE | 2,048 | **2.667×** |

**The discrepancy, plainly.** The spec's own words are *"blockwise unit
quaternions at `d/4` cost"*. Quaternions are `b=4`. `b=4` at `d=256` is `1,792`
FLOPs/edge under the same convention that makes `768` correct for `b=2` — not
`768`. The headline figure prices the **non-commuting** arm at the **commuting**
arm's rate, and 768 is arm 3's number, not arm 2's. That is an M-8 carry inside
the round spec itself.

Both numbers stand: `768` is right for arm 3 and `1,792` is right for arms 2
and 4. Neither is right for the other.

### The dense Cayley, and whether it is per-step or per-edge

The spec's "~16.7M FLOPs per action type" is `16,777,216 = 256^3 = d^3` exactly
— the textbook `O(d^3)` with the constant taken as 1. The actual transform is an
inverse **and** a matmul, `U_a = (I−Ω_a)^{-1}(I+Ω_a)`:

* OPS, LU inverse `+` matmul = `2*d^3` = **33,554,432** (2.0× the claim)
* HOUSE = `4*d^3` = **67,108,864** (4.0× the claim)

**Per-step, not per-edge.** `Ω_a` are parameters. They move once per optimizer
step, so the build is paid `|A|` times per step and reused by every edge in the
batch. It is `n`-independent, and it amortizes.

**But the two spec claims are inconsistent, and this is the finding.** `768
FLOPs/edge` presumes the quaternion/blockwise path; `16.7M per action type`
presumes a dense `d×d` Cayley. They cannot both describe one implementation,
because a dense Cayley produces a dense `d×d` `U`, and a dense `U` must then be
**composed and applied densely** — 131,072 FLOPs/edge to apply and 33,554,432 to
compose, not 768. Pricing a dense build against a blockwise transport is what
makes the dense path look survivable. Priced consistently:

| path | transport (n=1) | build (n=1) | arm total | pos % |
|---|---|---|---|---|
| blockwise `b=4` | 6,291,456 | 262,144 | 2,441,355,264 | **0.2684%** |
| dense `d×d` | 34,628,173,824 | 1,073,741,824 | 38,136,717,312 | **93.6156%** |

The dense transport alone is **14.3× `F_base`**. The build amortizes with `n`
(below 1% of the transport by `n = 4`); the transport does not, so the dense
path is never free at any `n`.

---

## 3. "depth-8 path = 6,144 vs ~524,288 for one s=1024 attention row" — DIRECTION HOLDS, FIGURES DO NOT

| path form | HOUSE | OPS |
|---|---|---|
| naive, ABELIAN `b=2`, ONE vector, depth 8 | 8,192 | **6,144** |
| naive, QUATERNION `b=4`, ONE vector, depth 8 | 16,384 | 14,336 |
| naive, QUATERNION `b=4`, q AND k, depth 8 | 32,768 | 28,672 |
| **PREFIX**, QUATERNION `b=4`, q AND k | 6,144 | 5,376 |

`6,144` reproduces as **abelian, one vector, naive** under OPS — three
substitutions away from the arm the round headlines: commuting instead of
non-commuting, one transported vector instead of `q` and `k`, and the naive walk
that the Linus gate exists to remove.

**Coincidence trap, stated so it is not mistaken for agreement.**
`depth · (b=2, OPS) = 8 · 768 = 6,144` and `3 · (b=4, HOUSE) = 3 · 2048 = 6,144`.
Both are `24·d` at `d=256`; they share no term and diverge at any other depth.
The OPS value of the prefix quaternion path, `5,376`, breaks the tie.

**The attention row is half a row.**

| convention | logits row `q@K^T` | logits + `a@V` |
|---|---|---|
| HOUSE | **524,288** = `2*s*d` | 1,048,576 = `4*s*d` |
| OPS | 523,264 = `2*s*d − s` | 1,046,528 |

`524,288` is the HOUSE `q@K^T` row alone. Mixing a vector into the row costs
another `2*s*d` and the spec omits it.

**Held to one convention, arm 2's own prefix per-node cost against a row:**

| convention | VGPE prefix/node | vs logits-only | vs full row |
|---|---|---|---|
| HOUSE | 6,144 | 85.3× cheaper | 170.7× cheaper |
| OPS | 5,376 | 97.3× cheaper | 194.7× cheaper |

**Verdict: the direction holds at every combination — the gauge is two to three
orders of magnitude below an attention row.** The specific `85×` is not
reproduced by the arm the round headlines; the honest range is 64× to 195×
depending on convention and on whether the row is half or whole.

---

## 4. LINUS GATE: naive vs prefix, as a number

Prefix: `R_i = R_parent ∘ U_{a_i}` — one composition per node, reused by every
descendant. Naive: walk the root path once per node per transported vector.

Per training step at `n=1, s=1024`, HOUSE:

| arm | naive | prefix | **SAVED** | factor |
|---|---|---|---|---|
| `twin + VGPE` | 33,554,432 | 6,291,456 | **27,262,976** | 5.333× |
| `twin + VGPE-abelian` | 16,777,216 | 3,145,728 | **13,631,488** | 5.333× |

The factor is `2*depth/3` and is depth-driven, so it is the same number for both
gauge arms while the absolute saving is not — which is the M-8 point in
miniature.

| depth | factor | VGPE saved/node | abelian saved/node |
|---|---|---|---|
| 2 | 1.333× | 2,048 | 1,024 |
| 4 | 2.667× | 10,240 | 5,120 |
| **8** | **5.333×** | **26,624** | **13,312** |
| 16 | 10.667× | 59,392 | 29,696 |
| 64 | 42.667× | 256,000 | 128,000 |
| 1024 | 682.667× | 4,188,160 | 2,094,080 |

`depth = 1024` is the degenerate path-graph case: naive is `O(s^2)` transports,
prefix is `O(s)`.

### The Linus gate INVERTS on the dense path

The gate counts *compositions*: `O(n)` instead of `O(n·depth)`. That is a saving
only if one composition costs about what one application costs. In the block
algebra it does. Densely it does not — compose is `2*d^3` and apply is `2*d^2`,
so one composition costs `d` applications.

| path | naive/node | prefix/node | prefix is |
|---|---|---|---|
| blockwise `b=4` | 32,768 | 6,144 | **5.333× CHEAPER** |
| dense `d×d` | 2,097,152 | 33,816,576 | **16.125× DEARER** |

Break-even: `2*d^3 + 4*d^2 < 4*depth*d^2`, i.e. `depth > d/2 + 1`. At `d=256`
the dense prefix **ties at depth 129 and wins only from depth 130**. At the
spec's `depth = 8` it loses by 16.1×.

**The prefix saving is a property of the blockwise representation, not of the
tree.** A design that claims both the dense Cayley and the prefix saving is
internally inconsistent.

---

## 5. WHAT THE COST TABLE SAYS ABOUT THE CONTROLS

**Arm 4 (`shuffled-gauge`) is EXACTLY cost-matched to arm 2.** Difference: **0**,
in every term, at `n ∈ {1, 8, 64, 512}` — derived independently, then checked
(`scale/vgpe_flops.py`, self-check F). Shuffling the label → `U_a` map permutes
a lookup table: it changes *which* `U` is fetched, never *how many* are applied.
Any wall-clock gap between arms 2 and 4 is dispatch or contention, not work.
This is the strongest property the M5-class plant could have had.

**Arm 3 (`VGPE-abelian`) is NOT cost-matched, and this is a real confound.**

| quantity | abelian | VGPE | ratio |
|---|---|---|---|
| transport (n=1) | 3,145,728 | 6,291,456 | exactly 2 |
| Cayley build | 65,536 | 262,144 | exactly 4 |
| positional total | 3,211,264 | 6,553,600 | 2.040816 |
| arm total | 2,438,012,928 | 2,441,355,264 | 1.001371 |

No single constant relates the arms — which is precisely why each was derived
from its own primitives rather than scaled off the other. The non-commutativity
ablation is cheaper by construction, because commuting blocks are `2×2` and
non-commuting ones are `4×4`. **A wall-clock comparison of arms 2 and 3
confounds non-commutativity with block size.** The FLOP difference is 0.1369% of
the arm total, so it cannot explain a capability gap — but it can explain a
timing gap, and a timing gap must not be read as evidence about commutativity.

**Arms 1 and 5** differ by the twin add-on alone: 18,358,272 FLOPs = 0.7597% of
`F_base` at `k=16`.

### `k` sweep — no arm ordering moves

| `k` | `twin+RoPE` | `twin+VGPE` | `twin+VGPE-abelian` |
|---|---|---|---|
| 8 | 2,428,112,896 | 2,432,569,344 | 2,429,227,008 |
| 16 | 2,436,898,816 | 2,441,355,264 | 2,438,012,928 |
| 32 | 2,455,257,088 | 2,459,713,536 | 2,456,371,200 |

### `n` sweep — "FREE" holds on the blockwise path only

Positional cost as a share of **that arm's own** total:

| arm | n=1 | n=8 | n=64 | n=512 |
|---|---|---|---|---|
| `twin+RoPE` | 0.0861% | 0.0861% | 0.0861% | 0.0861% |
| `twin+VGPE` | 0.2684% | 0.2591% | 0.2579% | 0.2578% |
| `twin+VGPE-abelian` | 0.1317% | 0.1294% | 0.1291% | 0.1290% |
| `twin+shuffled-gauge` | 0.2684% | 0.2591% | 0.2579% | 0.2578% |
| `softmax+RoPE` | 0.0867% | 0.0867% | 0.0867% | 0.0867% |

Absolute delta of each arm against `twin+RoPE`, HOUSE, blockwise:

| arm | n=1 | n=8 | n=64 | n=512 |
|---|---|---|---|---|
| `twin+VGPE` | +4,456,448 | +33,816,576 | +268,697,600 | +2,147,745,792 |
| `twin+VGPE-abelian` | +1,114,112 | +8,454,144 | +67,174,400 | +536,936,448 |
| `twin+shuffled-gauge` | +4,456,448 | +33,816,576 | +268,697,600 | +2,147,745,792 |
| `softmax+RoPE` | −18,358,272 | −146,866,176 | −1,174,929,408 | −9,399,435,264 |

On the dense path the same column reads 93.6156% (n=1) to 93.4310% (n=512): the
build amortizes, the dense transport does not.

### `|A|` sweep — dense build against dense transport

| `|A|` | n=1 | n=8 | n=64 | n=512 |
|---|---|---|---|---|
| 4 | 0.7191% | 0.0905% | 0.0113% | 0.0014% |
| 8 | 1.4279% | 0.1807% | 0.0226% | 0.0028% |
| 16 | 2.8155% | 0.3608% | 0.0452% | 0.0057% |
| 64 | 10.3849% | 1.4279% | 0.1807% | 0.0226% |

---

## 6. "THE GAUGE IS FREE AT THE SHAPES THIS PROJECT SHIPS" — CONDITIONALLY TRUE

**True on the blockwise path.** Every gauge arm's positional cost is under
**0.27%** of its own forward at every `n` from 1 to 512. The most expensive arm
adds **4,456,448 FLOPs** to a **2,436,898,816-FLOP** control at `n=1`. That is
free by any reading.

**False on the dense-Cayley path**, at every `n`: **93.6%** of the arm total at
`n=1`, **93.4%** at `n=512`. The build amortizes; the dense transport it forces
does not.

**So the claim is true only under the parametrization that makes `768/edge`
wrong for this round's arm, and the parametrization that makes `16.7M/action
type` relevant makes the claim false.** The two spec cost claims select
different implementations, and "free" survives only on the blockwise one — where
the correct per-edge figure is 1,792 (OPS) or 2,048 (HOUSE), not 768.

---

## 7. NOT COUNTED — element counts, per-element FLOP NOT FOUND

Same treatment `scale/m3_flops.py:370-395` gives transcendentals: element counts
are exact, per-element FLOP cost is NOT FOUND in this repository's source.

| term | formula | elements at the shape |
|---|---|---|
| RoPE cos/sin table | `s*d/2`, built once per step, cacheable | 131,072 |
| Cayley divide, blockwise `b=2` | `|A|*d/2` | 2,048 |
| Cayley divide, blockwise `b=4` | `|A|*d/4` | 1,024 |
| quaternion renormalise | `|A|*d/4` (rsqrt, if enforced per step) | 1,024 |
| transport backward | ~2× the forward transport, all arms alike | 12,582,912 |
| gate normalise (twin) | `n*k` | 16 |

---

## Limits

`|A| = 16` and `hidden = 4d` are **not fixed by the round spec**; both are swept
above and any other value can be read off the formulas. The tree is assumed to
have `s = 1024` nodes and `s − 1` edges, with each node consuming exactly one
edge under the prefix scheme — so "per edge" and "per node compose" are the same
quantity here, and the two per-node applies (`q`, `k`) are per node and not per
edge. Backward is not counted for any arm; the omission is uniform, so it cannot
move an ordering, but it means every total is a forward-pass total and not a
step total. Elementwise and transcendental work is excluded from all totals and
listed separately, per house convention. `cayley_block` is an **upper bound** —
both `b=2` and `b=4` have closed forms cheaper than the generic inverse it
prices, and the bound was used so that no closed form had to be assumed about
code not yet in the tree. The `b=4` blocks are priced as unit quaternions
(left-isoclinic `SO(4)`); a general `SO(4)` block needs two quaternions and
doubles that term. This model has not been reconciled against `scale/vgpe.py`,
which did not exist in this worktree when the model was written — reconciling
the two is the cross-check this file was built to make possible, and it has not
been performed. Every number here is analytic; none is measured.
