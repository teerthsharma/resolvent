# V15 SATURN-2 — leak ruling on M-18's squared-feature gate repair

**Node:** SATURN-2, CEQ v15 composition round. MARS-register question, SATURN
seat. Scope: adjudicate whether the squared feature that recovers `R^2 =
1.000000` on `log|a|`/band-mask (`MISTAKES.md` M-18) is a legitimate
architectural fix or an oracle leak, and whether it changes the standing
nine-cell non-crossing census.

**Inputs read:** `MISTAKES.md` M-18 (:1483-1533); `V15_VENUS_PREDICTIONS.md`
(:0, :306-367); `scale/negation_scope.py` (channel constants :69-71, base
`make_batch` :84-106, chain-family header :209-283, `equilibrium_oracle`
:286-304, `equilibrium_hop_reading` :307-330, `make_equilibrium_batch`
:354-433, the C1/propagate "NO NEW CHANNEL" documentation :517-522);
`scale/m3_capability.py` (`Arm.forward` :141-167); `CEQ_V15_CONTRACT.md` PART
IV (:205, :227-230); `tests/mars_v15/test_skyline_gate_containment.py` (read
and independently re-run, not edited).

A parallel MARS node (`V15_MARS_ATTACKS.md`, `tests/mars_v15/`) filed a
finding mid-session that the raw per-step gate's presence in `x` is
by-design and not itself a leak, with its own attack narrowed to *composed*
(order ≥ 1) quantities. That claim is verified independently below (source
read, and its test suite re-run on this machine) rather than taken on
report — it changes which question this ruling has to answer, not the
evidence used to answer it.

---

## (a) Channel-by-channel read of `x`, chain family (BED-M, `t* ∈ {2, 8}`)

| channel | constant | file:line | content on the chain/equilibrium corpus |
|---|---|---|---|
| 0 | `CH_FLIP` | `scale/negation_scope.py:69,431` | `b`, the Gaussian per-step driver, **at every position** (`x[:, :, CH_FLIP] = b`); `b[:, s-1] = 0` (query carries no driver, `:430`) |
| 1 | `CH_PAYLOAD` | `scale/negation_scope.py:70,101-105` | zero everywhere except position `p = s-2`, where it holds one `N(0,1)` draw — **left over from the base `make_batch` call and never overwritten by the chain family.** Unit variance, vs. `0.01` variance ambient noise elsewhere — this asymmetry matters below. |
| 2 | `CH_NOISE` → `CH_DRIVE = CH_NOISE` | `scale/negation_scope.py:71,245,432` | **`a`, the Rademacher gate coefficient itself, at every position** (`x[:, :, CH_DRIVE] = a`): `a_i ∈ {-1, +1}` for `i > head`, `a_i = 0` for `i ≤ head` (`:426-429`, `head = s - 1 - t*`) |
| 3–15 | — | `scale/negation_scope.py:96` | ambient `randn(n, s, d_model) * 0.1`, untouched by the chain family |

**Documented, not incidental.** `scale/negation_scope.py:520-522`, the C1/propagate
family's own comment block, states this in so many words: *"NO NEW CHANNEL. The
encoding is the chain family's own: `CH_DRIVE` carries the Rademacher
coefficients `a`, `CH_FLIP` the Gaussian drivers `b`."* The contract's own R1
kill-condition text (`CEQ_V15_CONTRACT.md:209`) presupposes the same thing —
*"diagnose by linear probe on `log a` (should be near-exact)"* only makes sense
as a diagnostic if `a` is meant to be legible from `x` in the first place.

**`scale/m3_capability.py:141-145`** — every arm (`softmax`, `pivot_signed`,
`pivot_unsigned`, `windowed_signed`) is handed the **raw** `x` tensor for its
`q`, `k` projections and its residual (`z = x + a_op @ x`); no arm receives a
filtered or channel-masked view. Whatever is in `x[:, :, CH_DRIVE]` is
therefore visible to the arm and (per the contract's PART V) to the skyline as
well.

## (b) Is `a` recoverable from `x` by a fixed function? Yes — and so is
`sign(a)`, exactly; `|a|`/`log|a|` needs one nonlinearity `a` alone can't supply

`x[:, :, CH_DRIVE]` **is** `a` (line 432 is a direct tensor assignment, not a
derived encoding). So:

- `x → sign(a)`: plain linear probe on raw `x`, `R^2 = 1.000000` (odd probe on
  an odd target — no nonlinearity needed).
- `x → 1{|a| > 0}` (the band mask, `≡ log|a|`'s exponential): plain linear probe
  on raw `x`, `R^2 = 0.000325` (an EVEN target has zero linear correlation with
  an odd channel, by Rademacher symmetry — this is M-18's finding, reproduced
  below).
- `[x, x^2] → 1{|a| > 0}`: `R^2 = 1.000000` exactly. Squaring `CH_DRIVE` turns
  `a ∈ {-1, 0, +1}` into `a^2 ∈ {0, 1}`, which **is** the band mask, bit for
  bit. This is the repair M-18 reports.

So the target the arm is asked to produce is a **deterministic, order-0
(single-position, no cross-position combination) function of a channel that is
already in its input by design.** No latent is being discovered; a fixed
nonlinearity (square, or `|·|`) is being supplied to route around a
parametrization (`-softplus(Wx)`, monotone) that cannot express an even
function of an odd argument. That is the precise sense in which M-18's own
diagnosis — "the obstruction is evenness, not information" — is correct, and
it is a *representational*, not an *information-availability*, finding.

## (c) The discriminating experiment — real numbers, `scripts/v15_saturn2_leak_probe.py`

Four experiments, all closed-form least-squares (`numpy.linalg.lstsq`),
float64, no training, run against the real corpus generator only
(`make_equilibrium_batch`, imported unmodified). `n = 2048, s = 64, seed = 0`
unless noted.

### A — reproduce VENUS's row (`CH_DRIVE` disclosed, as shipped)

```
head = 61
sst_band = 3968.000000
r2_band_x_only = 0.000325
r2_band_x_and_xsq = 1.000000
sst_sign = 4095.178711
r2_sign_x_only = 1.000000
```

Matches `V15_VENUS_PREDICTIONS.md:322-339` exactly (`0.0003`-scale null on `x`
alone, `1.000000` with the square added, `1.000000` on `sign(a)` from `x`
alone).

### B — the instructed test: shield `CH_DRIVE` with independent noise, same `y`

`x[:, :, CH_DRIVE]` is overwritten **after** `y` is already computed from the
true `a`, with `randn * 0.1` from an independent generator seed — severing any
fixed-function path from `x` to `a`.

```
r2_band_x_only = 0.000135
r2_band_x_and_xsq = 0.166160
per_channel_r2 = c0:0.0037, c1:0.1623, c2:0.0000, c3..c15: 0.0000
CH_DRIVE=2 shielded-channel R^2 alone = 0.000003   (vs CH_PAYLOAD=1 R^2 = 0.162287)
```

The raw collapse (`1.000000 → 0.166160`) is real but not the whole story: the
per-channel breakdown shows the residual `0.166` lives almost entirely in
channel 1, `CH_PAYLOAD` — **a second, unrelated channel**, not a resurgence of
the shielded drive signal (`CH_DRIVE`'s own contribution is `0.000003`, i.e.
gone). The reason is a fixture coincidence, not a leak mechanism: `CH_PAYLOAD`
sits at a fixed position `p = s - 2` (`:105`) with unit variance, against
`0.01`-variance ambient noise everywhere else, and `p = s - 2` falls inside the
live band for *every* `t* ≥ 2` this contract runs (live band starts at
`s - t*`). `x^2` of `CH_PAYLOAD` is therefore a near-perfect *position* tell,
independent of `a` entirely.

Controlling for it (`CH_PAYLOAD` also shielded, isolating `CH_DRIVE`'s own
contribution):

```
-- B2: same shield, PLUS CH_PAYLOAD masked too (isolates CH_DRIVE) --
r2_band_x_only = 0.000112
r2_band_x_and_xsq = 0.003961
```

**`R^2` collapses from `1.000000` to `0.003961`** once `CH_DRIVE` is the only
thing withheld and the unrelated payload-position confound is also controlled
— a ~250× drop, indistinguishable from sampling noise on a 33-parameter fit
over 131,072 pooled rows. The perfect fit in (A) depends entirely on
`CH_DRIVE` carrying `a`; nothing else in `x` substitutes for it.

**These are the two numbers requested:** `R^2 = 1.000000` (disclosed) vs.
`R^2 = 0.003961` (shielded, confound-controlled).

### C — the order boundary: does `[x, x^2]` reach anything composed (order ≥ 1)?

Same pooled `[x, x^2]` feature set, now regressed against (i) the running
multi-step resolvent (`equilibrium_hop_reading` re-run at every prefix length
— the same construction `tests/mars_v15/test_skyline_gate_containment.py`
uses to define "composed"), live band only, and (ii) the actual R1/R2 scalar
label `y`, from the readout position's own `[x, x^2]` alone:

```
-- t*=2 (R1's cell) --
live_n = 2
r2_composed_x_only        = 0.248855
r2_composed_x_and_xsq     = 0.251389      (x^2 adds +0.0025)
sst_label = 4134.041248
r2_label_x_and_xsq_at_readout = 0.021978

-- t*=8 (R2's cell) --
live_n = 8
r2_composed_x_only        = 0.162630
r2_composed_x_and_xsq     = 0.163691      (x^2 adds +0.0011)
sst_label = 16178.056856
r2_label_x_and_xsq_at_readout = 0.011448
```

`x^2` buys essentially nothing against either composed target (`+0.001` to
`+0.003` `R^2`, against `+0.9997` on the order-0 band mask), and the pointwise
probe never gets close to the actual label (`R^2 ≈ 0.01–0.02` at the readout
position, using that position's own channels only). The nonzero baseline
(`~0.25` at `t*=2`, falling to `~0.16` at `t*=8`) is explained by
`z_{head+1} = a_{head+1} b_{head} + b_{head+1}` — the first live position's own
`CH_FLIP` term contributes additively and is directly visible; the
cross-position product `a_{head+1} b_{head}` is not, and no pointwise feature
of one position can supply it. The share explained **shrinks** as `t*` grows,
which is the signature of a probe that is not composing hops, not one that is.

### D — independent re-run of MARS's shipped containment tests

```
tests/mars_v15/test_skyline_gate_containment.py::test_raw_per_step_gate_is_present_by_documented_design_not_a_leak PASSED
tests/mars_v15/test_skyline_gate_containment.py::test_composed_order1_signal_is_absent_from_x_on_the_live_bedm_corpus PASSED
tests/mars_v15/test_skyline_gate_containment.py::test_containment_census_MUST_FIRE_on_a_planted_composed_leak PASSED
tests/mars_v15/test_skyline_gate_containment.py::test_containment_census_MUST_FIRE_on_a_rescaled_planted_leak PASSED
tests/mars_v15/test_skyline_gate_containment.py::test_composed_signal_absent_from_x_fed_to_the_v15_skyline SKIPPED
4 passed, 1 skipped in 1.60s
```

Confirms, on this machine, independently of the parallel node's report: the
order-0 gate's presence in `x` moves the oracle's output under perturbation
(the "is it really the gate" control), and no channel of `x` contains the
order-≥1 running resolvent — with the census shown, on the same corpus, to
actually catch a planted composed-quantity leak (both an exact copy and a
rescaled one) when one is deliberately introduced. The skyline-specific test is
a real SKIP (no `ceq.beds.bed_m`/`scale.arm_pl` skyline module exists yet on
disk), not a pass being claimed for unbuilt code.

## (d) Ruling

**LEGITIMATE**, precisely scoped.

The squared feature is a representational fix to the **order-0 gate
sub-problem only** — recovering `log|a_i|`/band-membership from the channel
that the corpus deliberately, documentedly discloses (`CH_DRIVE = a`,
`scale/negation_scope.py:245,432,520-522`), which the contract's own R1
kill-condition text already presupposes is legible. `x^2` supplies the one
missing ingredient — an *even* nonlinearity — that `-softplus(Wx)`, a monotone
function of a linear form, structurally cannot: `R^2` moves from `0.000325`
(no square, real signal, degenerate parametrization) to `1.000000` (square
added), and the discriminating experiment shows that number depends entirely
on the disclosed channel: shielding `CH_DRIVE` alone (holding everything else,
including the confound, fixed) collapses `R^2` to `0.003961` — not
"diminished," gone.

**What makes it legitimate rather than a leak, in MARS's own terms
(`tests/mars_v15/test_skyline_gate_containment.py:7-28`): it never leaves
order 0.** The forbidden thing is a *composed* (order ≥ 1, multi-position)
quantity sitting in `x` — a partial resolvent, a cumulative log-gate sum,
anything only the oracle should be able to produce. `x_i^2` is a pointwise
function of one channel at one position; it cannot and (per experiment C) does
not reach the composed quantities R1 actually scores. Adding it moves `R^2`
against the order-0 band mask from `0.0003` to `1.0`, and against the order-≥1
running resolvent from `0.249` to `0.251` (`t*=2`) or `0.163` to `0.164`
(`t*=8`) — noise-level. Against the actual R1/R2 label, from the readout
position's own features, it reaches `R^2 ≈ 0.01–0.02`. **No feature the arm is
given by this repair reaches order ≥ 2 in the sense that matters here — none
of it is a product across two or more positions**, which is the one thing a
Rademacher chain's resolvent actually needs to be computed. The skyline reads
the same disclosed order-0 signal the arm does (both consume raw `x`,
`scale/m3_capability.py:141-145`) and, per the independently re-run census,
neither has order-≥1 content available to copy either.

**What R1 still has to do, and what the nine-cell census still means.** Fixing
the gate's representability does not fix R1. `g(x_i)` (however parametrized)
supplies, at best, a per-position sign/magnitude read; ARM PL's actual burden
is combining `t*` of those reads multiplicatively into the resolvent's last
coordinate — exactly the capability `scale/m3_capability.py`'s own docstring
names as the reason this item exists at all ("the model cannot solve it by
retrieving one negative value — it must combine two positions
multiplicatively"). Experiment C is the direct measurement that this
combination is not obtained for free: even with `[x, x^2]` in hand, a
pointwise probe at the readout position explains under 2% of the label's
variance. So the correct restatement of the nine-cell census, per the parallel
node's sharpened question: those arms were never "failing to discover a hidden
gate" — the gate's raw ingredient was visible to every one of them by
construction, the whole campaign — but that was never sufficient to cross a
floor whose measured quantity is the *composed* path product. Whatever the
non-crossing failures were (parametrization unable to express the per-step
gate, as M-18 argues for the literal contract spec; or a composition mechanism
that never got built or trained enough to combine `t*` gated reads correctly),
they were not failures to access information already sitting, disclosed, in
`x`. The squared-feature repair closes the first possible failure mode
(gate representability) without touching, and without being able to touch,
the second (composition) — so it neither manufactures nor retires any of the
nine crossings on its own, and R1's actual floor test is unaffected by whether
this repair ships: it still has to be earned by the operator that combines the
gated reads across hops, exactly as before.

**One separate, already-registered finding this ruling does not relitigate:**
M-18's original complaint that the *literal* kill-condition probe (`log|a|` on
the live band, `SST = 0`) is degenerate regardless of arm behavior is correct
and orthogonal to the leak question — a probe can be non-discriminating
(M-18's finding) on a quantity that is nonetheless legitimately, by design,
available to the arm (this ruling's finding). VENUS's registered amendment
(probe `sign(a_i)` / gate accuracy `p` instead of `log a`) is the fix for that
separate problem and is unaffected by this ruling either way.
