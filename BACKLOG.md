# Backlog — the parity campaign

W10 measured the gap: pure signed operator **2.2965** val against softmax **1.8838**, at
identical parameters (3,319,296) and **54% more wall-clock**. Ratio **1.2191** against a
1.05 bar.

Attention did not arrive in one paper. Closing a 22% gap is a campaign across iterations,
not a single fix. What follows is ranked by expected value per GPU-minute, each with the
falsifier that kills it.

**Every hypothesis is measured the same way, or it is not measured:** `ceq/lm.py`,
byte-level TinyStories, identical parameter count, identical initialization seed, identical
data, identical steps, identical optimizer, against the softmax arm. The only free variable
is the operator. Report val loss ratio. Anything above 1.05 has not reached parity.

---

## The diagnosis the campaign is built on

Softmax concentrates: `exp` is a sharpening nonlinearity, so a few keys carry most of the
mass. The current operator normalizes raw logits by their row L1 norm, `A = rho * w / sum|w|`,
which is **linear in the logits and therefore flat**. It cannot focus.

Then `hops=3` compounds that flatness: a flat operator applied three times is flatter still.
The multi-hop structure, which is the point of the design, is currently amplifying the
operator's worst property.

**So the campaign's central hypothesis is that signedness and selectivity were conflated.**
Dropping the softmax bought the sign and paid with the focus. They are separable.

---

## H9 — caustic Theorem 2 applied to attention rows  · NEW, and the strongest untried lead

The author's own theorems, pointed at the operator instead of at a language model.

**T1 (separation).** For an injective ground relation on `n` entities producing `m` distinct
values, `err >= n - m`, computable with NO ground truth.

**T2 (game theory).** If a map sends `k` entities to one value, that is a pooling
equilibrium and bounds ANY downstream function at `1/k`. Not a large model above it, not a
longer chain, not a better decoder.

**The application nobody has made.** An attention operator IS such a map: query `i` produces
row `A[i,:]`. Two queries with the SAME row are pooled, and by T2 no later layer can
separate them. T1 then counts the damage with no labels: `err >= n - m` where `m` is the
number of distinct rows.

**Why signedness should help here, structurally rather than by hope.** A non-negative row is
a point on the simplex. A signed row with bounded L1 lives in the strictly larger cross-
polytope. Same L1 budget, more distinguishable rows available, so the orbit partition of
queries should be strictly finer. That is a claim about the SET the operator can reach, not
about a training run, so it should not evaporate at scale the way a val-loss delta might.

**RED test:** `test_signed_rows_pool_fewer_queries_than_softmax_rows` — cluster attention
rows at a fixed tolerance, count orbits `m`, compare `n - m` between arms at matched `n`,
matched parameters, >= 3 seeds. The signed arm must pool strictly less.

**Kill:** if the orbit counts match, signedness buys no separation and T2 says the two
operators are equally bounded downstream -- which would mean the property measured by
`sign_flip_rate` has no consequence anybody can use.

**Why this is worth more than another val-loss decimal.** It is label-free, it is a statement
about reachable sets rather than about optimization, and it is measured on the author's own
published theorem with a proof already in `lean/CEQ/OrbitBound.lean`
(`orbit_error_bound`, machine-checked).

## H1 — signed gate, softmax magnitude  · cheapest, highest expected value

    A_ij = sigma_j * P_ij        P = causal softmax (selectivity)
                                 sigma_j in [-rho, rho] (sign)

Keeps `exp` sharpening exactly as softmax has it, and moves the sign to a per-source gate.
Tier 3 is preserved: `sigma_j < 0` gives a negative influence Jacobian entry, and the row L1
stays bounded by `rho` so `CEQ.Nilpotent.pow_card_eq_zero` still applies unchanged.

A non-negative version of exactly this shape was already measured (`perron_operator`,
`A_ij = s_j P_ij` with `s_j` in `[0, rho]`). Making the gate signed is a one-line change to a
construction that already exists in this repo.

**RED test:** `test_signed_gate_softmax_magnitude_reaches_parity`.
**Kill:** ratio stays above 1.05 with the gate free to go negative. Then selectivity was not
the missing ingredient and the diagnosis above is wrong.

## H2 — difference of two softmaxes

    A = softmax(scores_a) - lambda * softmax(scores_b)

Signed by subtraction rather than by gating; both halves keep `exp` sharpening.

**Check Foreman's novelty verdict before spending on this.** He is testing whether this is
Differential Transformer (arXiv:2410.05258). If it is, H2 is not novel on its own — but the
combination with strict causality, nilpotency, and the exact multi-hop resolvent is a
different object, and the card must say precisely which part is which.

**RED test:** `test_difference_of_softmaxes_reaches_parity`.

## H3 — sharpen the L1 path instead of replacing it

    w' = sign(w) * |w| ** gamma      then L1-normalize

Recovers concentration without `exp`, with `gamma` learned per head. Tests the diagnosis
directly: if sharpening alone closes most of the gap, flatness was the cause.

**RED test:** `test_sharpened_l1_reaches_parity`, plus a `gamma` sweep so the shape of the
curve is reported and not just its best point.

## H4 — per-head mixing, not per-model

Let each head choose: some heads softmax, some signed. The cheapest honest win available,
because it cannot be worse than all-softmax if the mixing weight can reach zero.

**RED test:** `test_mixed_heads_beat_all_softmax`.
**Note:** this is a weaker claim than parity of the pure operator, and the card must not
report it as parity. It is "the signed head earns its slot", which is a real result and a
different one.

## H5 — hops is currently a liability, so tune it against the operator

`hops=3` was chosen before any of this was measured. With a flat operator more hops is worse.
With a sharpened one it may be better. Sweep `hops` in `{1, 2, 3, 4, 6}` for whichever
operator wins H1-H3 and report the curve.

**RED test:** `test_best_hops_is_not_one`. If `hops=1` wins, the multi-hop path sum is dead
weight and the module is a single-hop signed attention — which is a much smaller claim and
must be restated as one.

## H6 — the normalizer's non-smoothness at w = 0

`rho * w / sum|w|` is not differentiable where the row is zero, and its Jacobian carries a
rank-one correction. Foreman is measuring the gradient-norm distribution against the softmax
arm. If training is ill-conditioned rather than the operator being under-expressive, H1-H3
are treating a symptom.

**RED test:** whatever Foreman binds. Do not duplicate his work; read his verdict first.

---

## Rules that do not change

- **Deletion beats defence.** A hypothesis whose falsifier fails is deleted and recorded,
  not tuned until it passes.
- **Single-seed results are not results.** Every parity claim needs >= 3 seeds with the
  spread reported. A tau sweep on seed 0 already produced a false optimum once (nash at
  tau=4.0 looked like a win at 2.3444 and evaporated to 1/5 seeds).
- **Iso-parameter, and report wall-clock.** The current operator loses while spending 54%
  more compute. A hypothesis that reaches parity by spending 3x is not parity.
- **The instrument gets calibrated before it is trusted.** Four checkers in this repo have
  been internally consistent and externally wrong.

---

## Opened by round 4 (Cameron) — the capability result

**C1 — the 3× budget run on COGS.** `sgate` is 0.7734 in-distribution against softmax's
0.9258 at 3000 steps, one seed. That is the number that separates "slower to fit" from
"cannot fit", and it is measured before generalization is tested. Falsifier: at 9000 steps,
if `sgate`'s in-distribution gate does not close on 0.9258, the deficit is capacity, not
optimization, and C4 deletes the operator. Size: ~3 GPU-hours on the 4060 uncontended,
`python -m ceq.capability --split cogs --steps 9000 --seeds 0,1,2`.

**C2 — `signmag` has never been trained.** It reads the highest sign-flip rate on the
shipped diagnostic (0.1641 / 0.2969 at depth 1/2, against `sgate` 0.1484 / 0.0938) and has a
per-pair sign that no third token dilutes — the one construction whose decay argument does
not obviously apply. It has never seen COGS, SCAN, or a val-loss run. Falsifier: same table,
same budget; if it also reads 0.0000 on COGS-gen the family is done. Size: ~1 GPU-hour.

**C3 — SCAN addprim_jump could not carry the comparison and needs a replacement or a
budget.** Both arms 0.0000 at 4000 steps against a published 0.034. Either the budget is
short of the published recipe (encoder-decoder, tuned) or the split needs the `+T5`-style
0.430 ceiling row to be reachable before a 0.000-vs-0.000 result means anything. Decide
which before spending on it again. Size: half a GPU-hour to check whether more steps move
softmax off zero at all.
