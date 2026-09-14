# Component ledger: every piece to its own bar before anything trains jointly

Opened at commit `b4c6620`. The method is component-wise validation first and
joint training last. Nothing trains jointly until every row below reads PASS or
carries an explicit, measured waiver saying why the joint run proceeds without
it.

The reason this ledger exists rather than a plan: this project trained a full
model four times before anyone measured whether its encoder aggregated context.
It does not. Every result produced before `b4c6620` used an encoder that is
`x_t -> s_t`, whose docstring claimed `x_{<=t} -> s_t`, and whose scaling
exponent in the token count is therefore zero by construction. That single
unmeasured component is what retired the per-coordinate corner rule
(`docs/CORNER_RULE_RETIREMENT.md`). A joint run cannot tell you which of its
parts is the broken one; only a bar per part can.

## Rules for a row to read PASS

1. **Its own bar, stated before the run.** The bar names a target the component
   is supposed to reach and a baseline it must beat.
2. **A planted negative that fails the bar.** If the shipped component, or a
   trivial one, also clears the bar, the bar measures nothing and the row stays
   OPEN.
3. **Matched parameter counts**, printed beside every arm. An unmatched win is a
   tie.
4. **RED first.** The test fails before the component exists, verbatim failure
   recorded.
5. **A producer for every number**, per `L-PROSE` in `MISTAKES.md`, with the
   report's bound-over-reported ratio on its first line.

## The pieces

| id | component | bar it must clear | status |
|---|---|---|---|
| **P1** | Encoder | aggregate context causally; clear a target a position-wise map provably cannot express, with the shipped `Encoder` shown failing it and a permutation control that moves the new one and leaves the shipped one bitwise identical | **PASS** |
| **P2** | Read / operator | matched head-to-head against a plain softmax attention head at equal parameter count | **ANSWERED — costs 1.17–2.84x, buys 0.055 nats and zero decisions; the gate is dead at the shipped init** |
| **P3** | Head / probe | find the binding ceiling among probe rank, read width and encoder | **ANSWERED — the encoder binds** |
| **P4** | Causal machinery | one synthetic-bed causal claim, measured on real human positions | **FAIL — the claim was an identity; the learned read is at 0.9997** |

## The joint gate

Joint training starts when P1 through P4 read PASS, or when a row carries a
waiver stating the measured reason it is being carried forward broken and what
that costs the joint result. A waiver is a number and a sentence, not a
judgement call.

Two things the joint run must carry regardless, both learned the hard way:

- **A frozen-random arm**, identical to the trained arm except that its encoder
  receives no gradient. Without it a joint result cannot separate scale from
  luck. On the synthetic bed the frozen arm *beat* the trained one; on 115,628
  real games the trained arm won, 0.5998 nats against 0.2348 with a uniform
  baseline of `2*ln(64) = 8.3178`. Only the control made that difference
  readable.
- **A trivial baseline on the real target.** Every arm of the T4 run lost to
  "guess the most common square" at every context length — 0.0823, 0.0256,
  0.0123 for the trained arm at L = 16, 64, 128 against 0.0942, 0.0398, 0.0247.
  A joint run without that column beside it can report a win that is not one.

## Limits

Statuses are updated only from a report that carries its producer ratio. A row
moved to PASS on prose is a row that is still OPEN.

## P1 — PASS, 2026-09-14

`ceqjepa/causal_encoder.py`, bound by `tests/curvature/test_causal_encoder.py`
(RED first: `ModuleNotFoundError: No module named 'ceqjepa.causal_encoder'`;
now 12 passed in 227.37 s). Reported with a producer ratio of 24/24.

The bar is a nine-channel strict-prefix EMA bank over an i.i.d. Gaussian bed.
A position-wise map cannot clear it, and not as an optimisation accident: the
sum runs over `j < t` strictly and the bed is i.i.d., so `y_t` is independent of
`x_t`, `E[y_t | x_t] = 0`, and the MMSE of *any* measurable function of `x_t`
alone equals `Var(y_t)`. The ceiling is `R² <= 0` by information, not by
training. A plain prefix mean was rejected as the bar because it is
permutation-invariant, so an encoder clearing it perfectly *should* be
order-blind — the bar and the permutation control would have disagreed.

| arm | params | R² eval | R² train | MSE | MSE const |
|---|---:|---:|---:|---:|---:|
| shipped `Encoder` (planted negative) | 1769 | **−0.1936** | 0.1543 | 1.1147 | 0.9339 |
| shipped `Encoder`, 3.2× wider | 5577 | **−0.6086** | 0.4027 | 1.5023 | 0.9339 |
| `CausalEncoder`, span = self (null) | 1657 | −0.0544 | 0.0446 | 0.9847 | 0.9339 |
| `CausalEncoder` | 1657 | **+0.9863** | 0.9872 | 0.0128 | 0.9339 |

The shipped encoder scores *below the constant predictor*, and widening it 3.2×
makes it strictly worse, which is what the argument predicts. The winner carries
112 fewer parameters than the arm it beats. Three seeds, non-overlapping:
shipped −0.1936 / −0.1940 / −0.2040 against causal +0.9863 / +0.9890 / +0.9896.

Permutation control — varies the order of context positions 0..14, pins the
multiset, `x_15`, weights, seed, batch and dtype: shipped `bitwise_identical =
True`, `max_abs_change = 0.000000e+00`; `CausalEncoder` `False`,
`4.647257e+00`. Causality is asserted rather than assumed: perturbing `x_k`
moves every position `t < k` by exactly `0.000000e+00` in all four arms.

The self-span ablation is what isolates aggregation from everything else —
identical 1657 parameters, initialisation asserted parameter-by-parameter with
`torch.equal`, identical bed, optimiser, lr, steps and loss, varying only which
positions a query may see. The `1.041` R² gap is attributable to aggregation
alone, not to depth and not to positional encoding.

**Adoption is deliberately not done yet.** Swapping the class touches
`pi_jepa.py:844-845`, and `:787-788` is a hard break: the collapse check reaches
*through* the encoder and iterates `.net`, so `CausalEncoder` exposes `.net` as
an `nn.ModuleList` for exactly that reason. `:1307-1311` states "Encoder is a
position-wise map", which becomes false and must be rewritten rather than
deleted — it is the premise the corner-rule retirement rests on.

Limits: `S = 16` only, so this says nothing about length generalisation; the bed
is synthetic i.i.d. by choice, because i.i.d. is what makes the position-wise
ceiling a theorem rather than a measurement; and the encoder's own
representational exponent was not measured, which is a different claim from
tracking nine timescales.

## P3 — ANSWERED, 2026-09-14: the encoder binds, and the probe is exonerated

Producer ratio 1.00 over 114 claims. Target is the exact committor from
`chess_steps.oracle()["q"]` (368,452 positions, residual 9.645e-13), not move
identity. Split by ply-15 **position** index, not by walk, with the train/test
position intersection asserted 0 — a by-walk split would score a lookup.

**The rank worry was wrong and is retired.** `logit(q)` as a single feature
reaches mse 2.253433e-15, skill **1.000000**. The committor is a scalar function
of state, so d = 1 suffices; 7 columns are seven times more than needed and no
rank bottleneck of the 64-class kind exists on this label.

Skill = 1 − mse/mse(marginal), cluster bootstrap over 4,006 distinct test
positions, 4,000 resamples.

| features | d | linear | kNN | GBT |
|---|---:|---:|---:|---:|
| six ordinal board coords (**the bar**) | 6 | −0.0002 | +0.6818 | **+0.8851** |
| raw 769 one-hot board | 769 | +0.1875 | | +0.4982 |
| encoder hidden, frozen_random | 1024 | +0.2560 | | +0.3702 |
| read7, frozen_random | 7 | +0.0126 | −0.0153 | −0.0061 |
| read7, trained | 7 | +0.0095 | −0.0388 | −0.0397 |

Three candidates, one binds. **Not the probe:** on the 7 columns every nonlinear
reader scores *below* linear at every arm, while the same GBT reaches +0.8851
where structure exists — the reader works, the structure is absent. **Not the
read width:** opening the two coordinates `pi_assign` refuses moves skill by at
most +0.0085 against a gap of 0.8851. **The encoder:** +0.3702 at the 1024-wide
layer falls to −0.0080 at the 9-dim output of the very next layer, one
`nn.Linear(1024, 9)`, and it is already 0.5154 behind the bar before that matrix.

**Training never helps.** Over a ladder of 0…256 fit steps the maximum skill is
+0.0163, reached after the representation has collapsed, while effective rank
falls monotonically 7.0038 → 1.0425. With the detector on, the trained arm
refuses at step 20: erank 1.4825 against floor 1.5 while `std_min` reads
5.5404e-03, **5.5x above its own floor** — the third independent sighting of the
variance leg being blind to dimensional collapse.

**Retired with a measured reason:** the D_LATENT sweep. No width crosses,
because a sweep over D cannot exceed what the layer feeding it carries, and that
layer carries +0.3697 against a bar of +0.8851.

**Reroute, one constant:** 0.3869 of the reducible variance dies in the
12-plane × 64-square one-hot before any weight (raw769 +0.4982 against embed6
+0.8851, identical reader and rows). `ceqjepa/chess_steps.py:549 embed` already
ships the ordinal (file, rank) encoding that recovers it. Change `X_DIM` and the
feature builder, re-run the bed unchanged; the bar is pre-registered at +0.8851
and RED #7 in `tests/curvature/test_committor_read_ceiling.py` is the gate.

**Reprice, before any further fit here:** `NU = 100.0` drives that collapse in 64
steps on a three-piece bed. Until `pi_jepa.covariance_sweep()` is run on this
bed, every trained number on it is a number about the degeneracy and
`frozen_random` is the honest arm.

Incidental, real, unfixed: the D_LATENT sweep crashes at D = 5 —
`pi_jepa.py:878 AssertionError: the read carries a nonzero imaginary part`.

## P2 — ANSWERED, 2026-09-14: optimisation parity, and the gate is dead on arrival

Producer ratio 61/61. Scope stated so it cannot be misread: `CHARTER.md:53-56`
already closes *expressivity* parity. This measures **optimisation** parity —
does gradient descent on the family reach where gradient descent on the fixed
corner reaches. Nobody had measured it. Metric is next-move prediction, which the
north star rules out as a target; it is admissible only because both arms are
bottlenecked identically and every comparison is arm-against-arm, never against
a baseline.

Bed: 6,844 real games at ≥33 plies, split by game with `check_game_split`,
6,148 train / 696 holdout, 22,272 held-out decisions. Tokens are **moves only**,
no board, so position is recoverable only by mixing over the prefix — which is
what makes the operator, not an encoder, the thing being scored. CIs are a paired
game-level bootstrap over the 696 games, 10,000 resamples.

**Containment survives training bitwise.** `arm-beta1` against softmax:
Δexact `+0.00000`, zero-width CI, at all three seeds. They are the same trained
model.

**The beta axis is a per-row scalar gain and nothing more.** `num_ij` carries no
beta, so `W(beta) = softmax · Z^(1−beta)` exactly — worst
`|softmax_row − linear_row| = 1.110223e-16` over a `[3, 12, 12]` operator once
rows are normalised. A free beta lowers held-out log-loss by **0.046–0.063 nats**
(mean 0.055, p < 1e-4 at every seed) and moves exact accuracy **not at all**,
because `dO_i/dbeta = −log(Z_i)·O_i` is a per-row scale carrying no component
that re-ranks which `j` a row attends to. Beta lands at an interior optimum,
0.667 from above and 0.587 from below — not the softmax corner. **A downstream
RMS norm absorbs the whole axis to 1e-12**, which is a structural tension worth
naming: the committor bed wants a normaliser and the beta axis cannot survive
one.

**The gate cannot train from the shipped init.** See `MISTAKES.md` V-29: two
independent mechanisms, each exactly zero, and `identity_heads` sits on both. An
arm trained 2,000 Adam steps from it ends bitwise identical to softmax with
`m_head_weight_absmax = 0.0`, at **1.910x** the wall clock. Woken with a 1e-3
offset on each bias, both gradients live (`2.412696e+00`, `1.005618e-02`) and the
operator stays within 5e-2 of the corner — and it then **ties** softmax across
three seeds (−0.00310 / −0.00054 / +0.00153) at 1.910–2.839x the cost.

Round-robin timing, interleaved so background load falls on every arm equally:
softmax 23.845 ms/step; `arm-beta-free` 1.174x; `arm-beta1` 1.318x; `arm-gate`
1.910x; `arm-gate-live` 2.839x.

**Retired with a measured reason:** the committor bed at this capacity. Built
exactly (`q_residual = 9.645e-13`), 58,940 held-out scored positions split by
hash of the state index. Only one configuration has the mixing operator beating
its own deletion, at R² `+0.06317`, and its curve still oscillates
(−0.1267, −0.1020, −0.1767, +0.0601 at steps 500/1000/1500/2000). A 0.003-scale
arm difference cannot be read off an unconverged endpoint carrying 6% of the
variance. The fix is optimisation budget, not architecture — and every
architectural stabiliser is a normaliser, which deletes the beta axis.

**Decided default: fix `identity_heads` before any further arm is trained.**
Every gate result in this repo produced from it is a measurement of softmax
wearing 130 buffers.

Limits: three seeds, one bed, one layer, one head, `d_model` 64, `S` 32, 2,000
steps — this prices the family at small scale and says nothing at transformer
scale. p-values are uncorrected for 9 comparisons; Bonferroni at α = 0.0056
leaves `no-mix` and the three ΔCE results standing, which is the same
conclusion. Wall clock is CPU-only with softmax held in float64 too, so the
ratios price the operator and not the dtype.

## P3 reroute — 2026-09-14: the encoding was the loss, and the bar has an axis

The one-constant reroute ran with the bar pre-registered and the assertions
locked. Producer ratio 1.00; every one-hot figure was re-derived bit-identically
before anything changed.

| untrained / hid / gbt | skill | gap to bar |
|---|---:|---:|
| 769-dim one-hot (before) | +0.3696799025442123 | 0.5154409791866523 |
| 7-dim ordinal (after) | **+0.8424222612206883** | **0.0426986205101764** |
| the bar, `embed6` + gbt | +0.8851208817308647 | — |

**91.7161% of the gap closed.** RED #7 did not flip and the bar was not moved.

**The residual is the bar's coordinate system, not lost information.** An exactly
orthogonal change of basis — `||Q'Q - I||inf` at 6.661e-16, destroying nothing by
construction — costs the same reader far more than the whole remaining shortfall:
`embed6` falls 0.1413 and `raw7` falls 0.2500. File and rank are literally the
tree's split axes, so RED #7 as written is partly an axis-alignment test. The
bar stays where it was pre-registered; the rotation control is recorded beside
it. Moving a bar after seeing the result is the defect this project hunts.

**The loss relocated downstream.** Untrained arm, gbt, under the new encoding:
`raw7` +0.8901 at the encoder's own input, `hid` +0.8424 (−0.0477), `enc9`
+0.4583 (**−0.3842**), `read7` +0.2353 (**−0.2230**). The MLP story now lives at
the 9-dim projection and the read, an order of magnitude above the residual at
the widest layer.

**Sixteen plies of context contribute nothing at the widest tap.** `raw7` pushed
through a *random* `Linear(7,1024)/GELU/Linear(1024,1024)/GELU` — the hid tap's
exact shape, one observation, no context — scores +0.8424222625154502 against
the bed's +0.8424222612206883, matching to nine significant figures.

**The collapse is priced, and raising `nu` does not prevent it.** First run of
`covariance_sweep` on the chess walk bed: at nu = 1, 25 and 100 the effective
rank falls from 7.6544 and every arm would refuse **at step 20**; nu governs only
the partial recovery afterwards (erank_last 1.0042 / 1.8198 / 2.5374). The
ordinal encoding makes it fire *earlier*, at step 13, with the smallest
per-coordinate sd at 1.8131e-02 — **eighteen times above** its 1e-3 floor.
Purely dimensional. `frozen_random` remains the honest arm. One contrast worth
chasing: on the PGN corpus at hidden 256 the trained arm ran all 1,500 steps
without firing (erank_min 1.6108, erank_last 8.2523) — two things differ, so it
is a lead, not a conclusion.

**The width-sweep crash is diagnosed and is not what it said** — see `P-14` in
MISTAKES.md. A float32 overflow reported as a moved gate; the largest genuine
imaginary part is exactly 0.0 at every width tested, and the shipped mask never
fires.

Next: attack the 9-dim projection with the encoding pinned to ordinal.

## P3 CORRECTION and the projection's acquittal, 2026-09-14 (producer ratio 313/313)

**Two claims recorded above are wrong and are corrected here.**

**"Training never helps" was measured at a collapsed width.** The collapse is a
HIDDEN-WIDTH threshold, not a property of the corpus or the objective:

| hidden | outcome | erank_min |
|---|---|---|
| 128 | survives 1,500 steps | 1.5369977235209193 |
| 256 | survives 1,500 steps | 1.9238083034849818 |
| 512 | refuses at step 45 | sd 5.7238e-02 |
| 1024 | refuses at step 13 | sd 1.8131e-02, erank 1.4530 of 9 |

Only `hidden` varies across that sweep, so the corpus is exonerated. The
non-firing contrast recorded earlier differs from the bed in **three** places,
not two: corpus, hidden width *and* observation encoding. At hidden 256, where
the fit is a real 1,500 steps rather than thirteen and a traceback, **training
helps**: `enc9` +0.6279 against its untrained +0.5663, `read7` +0.3422 against
+0.1919, encoder max|dw| 0.5756, no refusal. Every future trained number on this
bed is taken at hidden 256.

**The 1024 to 9 projection is an innocent bystander.** All four candidates are
killed, each by its own measurement. *Width:* `raw7` zero-padded to nine columns
scores +0.8901038547930719, bit-identical to `raw7` and above the bar, so nine
coordinates carry the whole label. *Information:* `enc9` reconstructs the six
ordinal board coordinates at R2 0.9752 to 0.9916 with zero float32 collisions,
and those reconstructions read +0.7105 against +0.4583 taken off `enc9`
directly. *Training:* above. *Context:* the encoder never had any, since the
1024-wide tap is **bit-identical under fifteen different context plies**, max
abs difference 0.0.

What is left is the **basis**. An orthogonal rotation costs `embed6` -0.1413 and
`raw7` -0.2500 but `enc9` only -0.0044, because `enc9` is already generic and
has nothing left to rotate away. Sixteen random 1024 to 9 draws of the same
`hid` average +0.5439 and +0.5740 by family; the shipped draw at +0.4583 sits
below all sixteen. The width sweep climbs monotonically (+0.4583 at 9, +0.7221
at 32, +0.8276 at 256) and never reaches the bar. That is not a capacity curve,
since capacity saturates at 7, but the count of axis-aligned directions a tree
can use.

**Where the loss is irreversible: the read.** It reconstructs the six board
coordinates at R2 0.9182, 0.9105, **0.1487, 0.1329**, 0.7213, 0.7414, and
coordinates 2 and 3 are the white queen's file and rank, the piece the committor
turns on. Re-reading from those reconstructions recovers nothing (+0.2596)
against +0.7105 by the same route from `enc9`. Mechanism: the read is a 16-ply
**mean**, with five of its seven columns explained by the plain unweighted mean
at R2 0.9962 to 0.9980 and adding ply 15 changing nothing, and a mean blurs the
queen because she moves 0.6923 squares per ply against the white king's 0.0810.

**This bed cannot reward an aggregating encoder, and that is measured.** `q` at
ply 15 is a function of the ply-15 *position*, so the walk is Markov and the
prefix is conditionally independent of the label. `CausalEncoder` at matched
parameter count, varying only which positions a query may see: aggregation
**costs** 0.0810 at `enc9`, prefix +0.2692 against self-span +0.3503. The 16-ply
mean alone scores +0.2978 against ply 15's +0.8851, and concatenating them costs
0.0377. Moving the label to ply 19, the horizon the family actually predicts,
does not rescue it. P1's PASS on the i.i.d. EMA bed is not in question; its sign
flips on this target, which is exactly the limit its own entry names.

**Next round does not widen the latent and does not give the encoder context**,
both measured dead ends here. It attacks the read's blur, on a target that is not
Markov in the ply-15 position.

Limits: one bed, one endgame, one reader family, skill stated for the
gradient-boosted reader only. The draw ensemble is eight seeds per family, so
"below all sixteen" is an order statistic, not a p-value. The recovery route uses
label-free board coordinates the encoder cannot access, so it is a ceiling on
what the representation carries rather than a shippable probe. The two RED tests
binding this round live in the session scratchpad and are **not** in the tree.
The pre-registered bar was not edited and remains RED.
