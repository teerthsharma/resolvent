# E4' spread: the sampling distribution of the number the strike is conditioned on

SATURN, v-main.3M script iteration 12.

## What was measured

`tests/cameron/test_e4_rips_gate.py::test_the_degree_decoder_passes_at_criticality_so_e4_is_struck`
conditions the whole E4 strike on one inequality:

```python
score = fit_eval(local_features(pairs, [adjacency(1024, case.edges)], 0), y)
assert score < PASS_BAR            # PASS_BAR = 0.5, scale/rips_gate.py:60
```

The shipped reading is `0.4710453160486452`, a margin of `0.028955` under the bar.
`draw_balanced_marginal` (`scale/rips_gate.py:198`) seeds its pair sampler
`random.Random(0x33960000 ^ case.seed)` and its train/test split
`np.random.RandomState(0)`; neither is a parameter, so the sampling spread of the
deciding number was never measured when the strike was made.

Reproduce every row below with:

```
python -m scale.e4prime_spread          # writes results/r10_it12_e4prime_spread.jsonl
```

Run twice, the `.jsonl` is byte-identical: nothing here depends on wall clock, process
hash seed or thread count. Python 3.11.9, numpy 1.26.4, Windows 11 x86-64, branch
`feat/r9-causal-consequence` at `aa82df7`.

## The seed is reachable without editing tracked source

`draw_balanced_marginal` reads only `case.node_count`, `case.partition` and
`case.seed`. `GraphCase` is a frozen dataclass whose `edges` and `partition` are plain
fields, so `dataclasses.replace(case, seed=s)` re-seeds the sampler and carries the
same list objects through. `scale/e4prime_spread.py::_assert_graph_untouched` checks
`edges` and `partition` by identity (`is`) on every substituted case and raises if
either moved, so "only the draw changed" is proved per row rather than asserted.

**No tracked source was edited.** `scale/e4prime_spread.py` is a new file that calls
the shipped functions; `scale/rips_gate.py`, `ceq/rips.py` and
`tests/cameron/test_e4_rips_gate.py` are untouched.

Held fixed across every row: the graph, the partition, `n_each = 1024`, the split
permutation `np.random.RandomState(0)`, `radius = 0`, and `PASS_BAR = 0.5`.

## The 20 draw seeds (block A) -- the set the it.12 verdict is read on

Declared by rule before any score was read: the 20 consecutive `case.seed` values
after the shipped `0x33960006`. Re-run any single row by hand with
`dataclasses.replace(case, seed=<case_seed>)` then the two lines quoted above.

| # | `case.seed` | `random.Random` seed | score | `score < 0.5` |
|---|---|---|---|---|
| -- | `0x33960006` | 6 | **0.471045** | yes (**shipped**) |
| 1 | `0x33960007` | 7 | **0.506250** | **no -- reversal** |
| 2 | `0x33960008` | 8 | 0.470402 | yes |
| 3 | `0x33960009` | 9 | 0.471809 | yes |
| 4 | `0x3396000a` | 10 | 0.482391 | yes |
| 5 | `0x3396000b` | 11 | 0.468960 | yes |
| 6 | `0x3396000c` | 12 | 0.482072 | yes |
| 7 | `0x3396000d` | 13 | 0.468763 | yes |
| 8 | `0x3396000e` | 14 | 0.474285 | yes |
| 9 | `0x3396000f` | 15 | 0.466404 | yes |
| 10 | `0x33960010` | 16 | 0.459390 | yes |
| 11 | `0x33960011` | 17 | 0.473392 | yes |
| 12 | `0x33960012` | 18 | 0.490249 | yes |
| 13 | `0x33960013` | 19 | 0.486529 | yes |
| 14 | `0x33960014` | 20 | 0.472217 | yes |
| 15 | `0x33960015` | 21 | 0.480245 | yes |
| 16 | `0x33960016` | 22 | 0.479473 | yes |
| 17 | `0x33960017` | 23 | 0.487707 | yes |
| 18 | `0x33960018` | 24 | 0.490227 | yes |
| 19 | `0x33960019` | 25 | 0.489364 | yes |
| 20 | `0x3396001a` | 26 | 0.489310 | yes |

Errors: **0 of 20**. Every row returned a drawable balanced set and a finite NRMSE;
no absence is being read as a pass.

## mu-hat, sigma-hat, z

| quantity | value |
|---|---|
| shipped | 0.471045 |
| mu-hat | **0.479472** |
| sigma-hat (sample, ddof = 1) | **0.011083** |
| sigma-hat (population, ddof = 0) | 0.010802 |
| min / max | 0.459390 / 0.506250 |
| z = (shipped - mu-hat) / sigma-hat | **-0.7603** (ddof = 1) / -0.7801 (ddof = 0) |
| rank of shipped among the 21 draws | 6th smallest (29th percentile) |

The dispatch's "currently -0.78" is the ddof = 0 form. The ddof = 1 form, -0.7603, is
the one reported here as the headline: the 20 sweep draws are a *sample* from the draw
distribution and the shipped draw is a 21st independent draw from it, so the unbiased
variance estimator is the correct denominator. The distinction moves z by 0.020 and
moves nothing else.

## k, and the verdict under it.12's own rule

**k = 19 of 20.** One seed, `0x33960007` (`random.Random` seed 7), reads 0.506250 and
reverses `score < 0.5`.

it.12's rule: *strike stands iff sign(effect) holds in >= 18/20.*

**19 >= 18. The strike STANDS.**

Applied mechanically. No judgement was substituted for the threshold.

## Disagreement with MARS_REPORT_IT4, and what it is

`attic/tests/mars/MARS_REPORT_IT4.md` (Attack 2 table) claims for 20 draw seeds:
min 0.459390, max 0.506250, mean 0.479472, sd 0.010802, 1 of 20 reverses, shipped
0.008427 below the mean = 0.78 sd.

Four of those six reproduce **exactly**: min, max, mean and the reversal count. The
shipped offset reproduces exactly (-0.008427).

The sd does **not** agree: MARS 0.010802, measured here 0.011083. The gap is entirely
the estimator, not the data -- `0.011083 * sqrt(19/20) = 0.010802` to six decimals.
MARS reported the population sd of the twenty; the sample sd is 2.6% larger. His 0.78
sd follows from the same choice. This is a labelling defect in his row, not a
measurement error, and it makes his spread look 2.6% tighter and the shipped draw look
2.6% more extreme than they are.

**The seed sets coincided, and that is the more important caveat.** Four statistics
agreeing to six decimals is not two independent samples agreeing; it means the rule
declared here ("the 20 consecutive case seeds after the shipped one") reselected
MARS's seeds. Block A is therefore a *reproduction* of his row, and the it.12 verdict
is read on the same 20 draws his was.

## Block B: a disjoint 20, which is the genuinely new evidence

`case.seed` `0x33960100`..`0x33960113`, `random.Random` seeds 256..275. Full rows in
the `.jsonl` under `"role": "sweep_b"`.

| block | n | mu-hat | sigma-hat (ddof 1) | min | max | k | reversals | errors |
|---|---|---|---|---|---|---|---|---|
| A (= MARS's seeds) | 20 | 0.479472 | 0.011083 | 0.459390 | 0.506250 | 19 | 1 | 0 |
| B (disjoint) | 20 | 0.475633 | 0.010301 | 0.457715 | 0.495780 | **20** | 0 | 0 |
| A + B | 40 | 0.477553 | 0.010738 | 0.457715 | 0.506250 | 39 | 1 | 0 |

Block B does not enter the it.12 verdict, which is defined on 20 seeds. It says the
draw distribution is stable across a distant part of the seed space (mu-hat moves
0.0038, well inside one sigma-hat) and that the single reversal in block A is not a
property of that block: over 40 draws the bar sits 2.09 sample sd above the mean and
one draw in 40 crosses it.

## Falsifiability of the instrument

Two independent demonstrations that this harness can return a score at or above the
bar, so a sweep that failed to cross would be readable as the world rather than the
rig:

1. **Inside the sweep itself.** `0x33960007` returns 0.506250 >= 0.5. The bar is not
   unreachable for this case.
2. **On a different case, same decoder, same code path.** `draw_balanced_marginal` +
   `local_features(..., 0)` + `fit_eval` on `StableSparse_S2Rips_64` returns
   **0.892650**, matching the value asserted at
   `tests/cameron/test_e4_rips_gate.py:147`. Emitted as `"role": "instrument_check"`.

A third check on the same axis: the shipped configuration reproduces
`0.4710453160486452` bit-for-bit through this harness, so the rig computes the same
function the strike test computes.

## The binomial p the rule cites, and whether it is the right null

The rule states `C(20,<=2) * 0.5^20 ~ 2.0e-4`. The arithmetic is right:
`C(20,0)+C(20,1)+C(20,2) = 211`, and `211 / 2^20 = 2.012e-4`. That is the one-sided
probability of at most 2 failures in 20 fair coin flips.

**It is not the right null for what was measured, and it should not be cited as the
p-value of this result.** Three reasons, in order of severity:

1. **Nothing makes 0.5 the null crossing rate.** A fair-coin null is the correct null
   for a *sign test*, where the null hypothesis "no effect" genuinely implies each
   observation is equally likely to fall either side of zero. Here the statistic is a
   continuous NRMSE compared against a pre-registered constant `PASS_BAR = 0.5`, and
   the quantity in question is `P(score >= 0.5)` over draws -- an unknown rate `p` with
   no reason at all to be 0.5 under any hypothesis anyone holds. The measured
   distribution is a tight cluster at mu-hat 0.4776 with sigma-hat 0.0107 (40 draws);
   the bar sits 2.09 sd above it. Rejecting `p = 0.5` at 2.0e-4 rejects a hypothesis
   no one proposed and that the data refute trivially.

2. **The 20 draws are not 20 independent replications of the effect.** They share one
   graph, one partition, one split permutation, one decoder and one bar. The only
   thing resampled is which 2048 node pairs are drawn. So the sweep measures the
   sampler's contribution to the deciding number and nothing else -- not the graph
   draw, not the split, not the corpus. Whatever `p` this estimates, it is conditional
   on `sample_sphere(1024, 0x33960006)`, and a binomial null over "seeds" silently
   promotes it to a statement about the world.

3. **20 draws cannot bound the rate the strike actually needs.** The decision-relevant
   quantity is `P(score >= PASS_BAR)`. Point estimate 1/20 = 0.05; exact
   (Clopper-Pearson) 95% interval **[0.0013, 0.2487]**. On the 40 draws here,
   1/40 = 0.025 with 95% interval **[0.0006, 0.1316]**. A normal approximation on the
   40 draws gives 0.018. The honest statement is that the reversal rate is somewhere
   under about 13%, and 40 draws do not narrow it further. The rule's 2.0e-4 is three
   orders of magnitude tighter than anything this design can support, because it is
   the p-value of a different question.

**What the rule should have asked for**, had it been written after the data rather
than before: a bound on `P(score >= PASS_BAR)`, or equivalently the standardised
distance from mu-hat to the bar, `(0.5 - 0.477553) / 0.010738 = 2.09 sd` on 40 draws.
That number carries the finding: the strike is not a coin flip, but it is a two-sigma
call resting on a bar pre-registered at a round number, and one draw in 40 does cross
it.

## Limits

The sweep varies the draw seed only; the split seed (`np.random.RandomState(0)`) and
the graph seed (`sample_sphere(1024, 0x33960006)`) are held at their shipped values,
so no statement here bounds their contribution -- MARS's split-seed row is not
re-measured and is not relied on. Block A reselected MARS's seed set, so its agreement
with him is a reproduction rather than a confirmation; only block B is independent.
`PASS_BAR = 0.5` is taken as given and not audited. The verdict is read on block A
alone because it.12 specifies 20 seeds; on the pooled 40 the same rule gives k = 39/40
and the same verdict. No model was trained and no step of this measurement exceeded
about 60 MiB of host RAM, so `scale/vram_gate.py` was not consulted.
