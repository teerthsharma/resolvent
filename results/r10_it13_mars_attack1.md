# R10 it.13 — MARS attack #1, filed before measurement

**Filed by MARS. Bound by SATURN. The predictor and the measurer are different
agents, and every number in section 2 is committed before any measurement that
could settle it.** Nothing in this file was produced by running the binding.

---

## 1. THE CONFOUND C

**C = the label's variance law.** The it.8/9 record scores the whole grid against
reference lines — the 1-hop ceiling `sqrt((t*-1)/t*)` and the calibration band
`2/sqrt(t*)` — that are derivable only from `Var(y) = t*`, while the same record,
the producer that generated the grid, the pricing tool, the README and the
docstring of the test that guards the band all state the law as `N(0, t*+1)`; the
two cannot both hold, and which one holds decides whether the grid's reference
lines or the grid's stated warrant is the thing that is wrong.

**Why it is a confound and not a typo.** `Var(y)` is not part of the capability
the reading claims to measure — the reading is `NRMSE < 1.0`, normalised by the
labels' own empirical spread, so it is invariant to the law. But `Var(y)` sets
every line the reading is *scored against*: the ceiling that F-2 and F-3 are
built on, the flipper band that `bar_verdict` gates on, and the "different label
law per block" clause that is the record's stated reason the `t*` axis costs 3x
instead of being sliced at eval. A variable that moves the reference lines
without moving the reading is exactly a confound on the verdicts read off them.

**The mechanism, named for the taxonomy.** A claim that was true of the pre-fix
builder, restated in the present tense after the fix that falsified it, and
copied into six downstream sites — including the docstring of the test that
guards it. The arithmetic: the resolvent is `z* = sum_{m=0..t*} A^m b`, which has
`t*+1` terms because `A` is nilpotent of index `t*+1`, and that index is stated
three lines above the variance claim in the same docstring
(`scale/negation_scope.py:361` and `:367`). Then `b[:, s-1] = 0` was added to pass
the 0-step RED gate (`scale/negation_scope.py:375-385`), which kills the `m=0`
term. The surviving terms are the `t*` drivers at positions `head .. s-2`, each
unit-variance with unit-modulus path weight, so `Var(y) = t*`. The variance claim
was read off the nilpotency index and never decremented when the term it counted
was removed. `chain_flipper_dependence` (`scale/negation_scope.py:419`) states the
corrected count — *"a sum of the `t*` drivers at positions `head .. s-2` ... so it
is `N(0, t*)` exactly"* — in the same file, 52 lines below the claim it contradicts.

**The seven live sites, split.**

| states `Var(y) = t*+1` | states `Var(y) = t*` |
|---|---|
| `scale/negation_scope.py:367` (the source docstring) | `scale/negation_scope.py:419-421` (`chain_flipper_dependence`) |
| `scale/r10_capacity_sweep.py:16` (the producer's justification) | `scale/negation_scope.py:528`, `:649` |
| `scale/r10_it8_pricing.py:166` (printed into `results/r10_it8_priced_dag.txt`) | `MATHEMATICS.md:64` |
| `R10_ITERATION_08_09.md:125` (the record) | `DONE.md:476`, `:539`, `:581` |
| `README.md:189` | `scale/foreman_lambda2.py:550` |
| `tests/cameron/test_m3_etasks.py:38`, `:276` | — |

Every **executable** line takes `t*`: `chain_flipper_dependence` returns
`2.0/math.sqrt(t_star)`, `r10_capacity_sweep.py:146` computes the ceiling as
`sqrt((t*-1)/t*)`, and `test_the_chain_flipper_dependence_is_its_closed_form`
asserts `abs(want - 2.0/math.sqrt(t_star)) < 1e-12` — under a docstring that
derives `2/sqrt(t*+1)` from `N(0, t*+1)`. That test's stated reason and asserted
value disagree by a factor `sqrt((t*+1)/t*)`.

---

## 2. delta_C — PREDICTED EFFECT, FILED BEFORE MEASUREMENT

**Primary quantity.** `delta_1(t*) = mean_over_draws[ Var(y) ] - (t* + 1)`,
population variance (`unbiased=False`, matching `nrmse`'s normaliser), units:
label variance.

| block | filed `delta_1` | sign | filed 95% CI on the mean | filed `mean Var(y)` |
|---|---|---|---|---|
| `t*=2`  | **-1.000** | NEGATIVE | `[-1.006, -0.994]` | 2.000 |
| `t*=8`  | **-1.000** | NEGATIVE | `[-1.025, -0.975]` | 8.000 |
| `t*=32` | **-1.000** | NEGATIVE | `[-1.098, -0.902]` | 32.000 |

CI half-widths are `1.96 * t* * sqrt(2/4096) / sqrt(200)` = `t* * 0.0030625`,
filed from the chi-square spread of a sample variance at `n=4096`, `K=200` draws.

**"The confound's direction", concretely, so no favourable reading can be picked
later.** The attack fires if and only if **all three** hold **in all three
blocks**:

1. `delta_1 < 0` — the measured variance is BELOW the stated `t*+1`, not above;
2. the 95% CI on `delta_1` excludes 0;
3. `|delta_1 + 1.000| < 0.05` — the shortfall is ONE unit of variance, the single
   `m=0` term that `b[s-1] = 0` removed.

A negative `delta_1` of any other magnitude does **not** fire. Clause 3 is the
point of the attack: `-1` is the mechanism's signature, and a shortfall of `-0.4`
or `-2.3` would mean the mechanism is something MARS did not name, which is a
miss and is scored as one.

**Control A, filed to NOT fire.**
`delta_2(t*) = mean_over_draws[ nrmse(equilibrium_hop_reading(x, 1), y) ] - sqrt((t*-1)/t*)`.
Filed: **0.000**, with `|delta_2| < 0.002` in every block. This is the ceiling the
grid is read against. If `delta_2`'s CI excludes 0 by more than 0.002 anywhere,
MARS's control failed and the casualty is F-2/F-3 rather than the record's stated
law. MARS does **not** get to claim that outcome as this attack; it is filed here
as a miss in advance so the reinterpretation is closed off.

**Control B — an axis MARS concedes in advance, so nobody spends an iteration on
it.** `delta_3` = across-independent-eval-draw sd of the reading, for a
training-free predictor pinned at the crossing: `pred = c * equilibrium_hop_reading(x, 1)`
with `c = 0.248958`, the value that puts NRMSE at exactly 0.9723723934535010 on
the `t*=8` corpus in population (mirror root `c = 1.751042`). Filed: **sd = 0.00129**,
95% draw-interval half-width **0.0025**, against a crossing margin of 0.0276 —
11x smaller. **The `t*=8` crossing is not an eval-draw artifact and MARS will not
attack it there.** If `delta_3` comes back above 0.014, the concession is
withdrawn and the eval-draw axis is live again for a later iteration.

---

## 3. THE BINDING MEASUREMENT

**Population the CI is over:** independent draws from the shipped corpora, at the
shipped eval shape — `n=4096` (`N_EVAL`), `s=64`, `d=24`, `d_model=16` — with seeds
disjoint from every seed the grid used (grid: 0-7 train, 12345 eval).
**Sample size:** `K = 200` draws per block, 600 draws total. **No training.**

Save as `scale/it13_mars_binding.py` and run `python scale/it13_mars_binding.py`:

```python
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.negation_scope import M3_TASKS, nrmse, equilibrium_hop_reading

K, C8 = 200, 0.248958


def stat(a):
    m = sum(a) / len(a)
    sd = (sum((v - m) ** 2 for v in a) / (len(a) - 1)) ** 0.5
    return m, sd, 1.96 * sd / len(a) ** 0.5


for t in (2, 8, 32):
    bf = M3_TASKS["e3_t%d" % t][0]
    V, H, Sh = [], [], []
    for sd in range(20000, 20000 + K):
        x, y, _, _ = bf(4096, 64, 24, d_model=16, seed=sd)
        V.append(float(y.var(unbiased=False)))
        h = equilibrium_hop_reading(x, 1)
        H.append(nrmse(h, y))
        if t == 8:
            Sh.append(nrmse(C8 * h, y))
    mV, _, hV = stat(V)
    mH, _, hH = stat(H)
    cf = math.sqrt((t - 1) / t)
    print("t*=%2d  mean Var(y)=%.6f  delta_1=%+.6f  CI [%+.6f, %+.6f]"
          % (t, mV, mV - (t + 1), mV - (t + 1) - hV, mV - (t + 1) + hV))
    print("       1-hop NRMSE=%.6f  closed form=%.6f  delta_2=%+.6f  CI [%+.6f, %+.6f]"
          % (mH, cf, mH - cf, mH - cf - hH, mH - cf + hH))
    if t == 8:
        mS, sS, _ = stat(Sh)
        print("       CONTROL B: pinned NRMSE=%.6f  across-draw sd=%.6f  "
              "95%% draw half-width=%.6f  (crossing margin 0.027628)"
              % (mS, sS, 1.96 * sS))
```

**Affordability.** Peak resident set is one batch of `[4096, 64, 16]` float32 =
16.8 MiB plus torch's own footprint; estimate ~350 MiB, one process, no gradients,
no optimiser, no model. That is below the ~1 GiB threshold the round's resource
rule sets, so `scale/vram_gate.py` `preflight` is not required by that rule; if
SATURN wants a logged pass it is
`preflight(0, name="it13_mars_binding", host_mib=350)`. Runtime: 600 batch draws
plus a 64-iteration scan each, order 1-3 minutes single-process.

**What the cheap binding gives up.** It cannot touch the trained model: no
checkpoint was retained by `r10_capacity_sweep.py`, and the round forbids
training. So it measures the CORPUS and the CLOSED FORMS — the reference lines —
and not the arm. It therefore cannot decide *why* the arm reads 0.9724; it decides
only what 0.9724 is being scored against. That is deliberate: the reference lines
are where the two incompatible laws live, and they are free to measure.

---

## 4. MARS'S OWN FALSIFIER

**MARS withdraws attack #1, in full and without reinterpretation, if the measured
95% CI on `delta_1` covers `0` in any one of the three blocks** — that is, if the
measured mean `Var(y)` is statistically indistinguishable from `t*+1` at `t*=2`,
`t*=8` or `t*=32`. One block suffices; there is no partial credit and no
"fires at `t*=8` only".

**MARS also withdraws if `|delta_1 + 1.000| >= 0.05` in any block.** A shortfall
that is not exactly one unit of variance falsifies the stated mechanism (the lost
`m=0` term), and the attack was filed on the mechanism, not on the sign.

**MARS does not get to convert a failed control into a hit.** If `delta_2` fires
instead — the measured 1-hop reading departing from `sqrt((t*-1)/t*)` by more than
0.002 with a CI excluding 0 — that is a larger finding than this attack and it is
handed to the record as SATURN's, not claimed as MARS's. Attack #1 is scored a
miss in that branch.

**Two defences are closed in advance.** (i) *"The docstring is describing the
pre-fix builder."* `R10_ITERATION_08_09.md:125` is in the present tense about the
shipped registry — *"`M3_TASKS` registers ... and gives a different label law"* —
and `scale/r10_it8_pricing.py:166` prints it as the live justification for the
grid that ran. (ii) *"`t*+1` versus `t*` is a rounding detail."* It is a factor
`sqrt((t*+1)/t*)` on the band, which is 22.5% at `t*=2`; section 5 prices it.

---

## 5. WHAT THE ATTACK COSTS IF IT FIRES

**Claim 1, quoted verbatim from `R10_ITERATION_08_09.md`:**

> "`t*` is a **corpus**, not an eval slice: `M3_TASKS` registers `e3_t{2,8,32}` as
> `partial(make_equilibrium_batch, t_star=t)`, which zeroes the sub-diagonal at
> `head = s-1-t*` and gives a different label law `N(0, t*+1)` and a different bar
> `2/sqrt(t*)` per block."

The clause "gives a different label law `N(0, t*+1)`" becomes unsupported, and the
sentence is shown to be self-contradicting: `2/sqrt(t*)` — the bar the same
sentence states, and the one `chain_flipper_dependence` actually returns — is
derivable only from `N(0, t*)`. This sentence is the record's whole warrant for
pricing the `t*` axis at 3 trainings rather than 1 slice; the pricing conclusion
survives (the corpora genuinely differ), the stated reason does not.

**Claim 2, quoted verbatim from `R10_ITERATION_08_09.md`:**

> "Bar: NRMSE < 1.0 (predict-the-mean). Calibration bar CALIBRATED in all three
> blocks; nothing below is credited from an uncalibrated block."

Under the law the same record states, the `t*=2` block does not calibrate. The
shipped journal reads `"flipper_dependence": 1.4012436552018552`
(`results/r10_it8_capacity_softmax_t2.jsonl`). Against `2/sqrt(t*)` = 1.4142135624
the deviation is 0.0129699, inside `bar_verdict`'s `flipper_tol = 0.05`. Against
`2/sqrt(t*+1)` = 1.1547005384 the deviation is **0.2465431**, 4.93x the tolerance —
`bar_verdict` returns `False`, and `r10_capacity_sweep.main` hits
`ABORT: calibration bar failed; crediting nothing.` (`scale/r10_capacity_sweep.py:157`)
before a single cell of that block runs. The record reports both the law and the
CALIBRATED status. They cannot both hold, and the journal says which one does.

The same arithmetic in the other two blocks, from the shipped journals:

| `t*` | measured | `2/sqrt(t*)` | dev | `2/sqrt(t*+1)` | dev | verdict under `t*+1` |
|---|---|---|---|---|---|---|
| 2 | 1.4012436552 | 1.4142135624 | 0.012970 | 1.1547005384 | **0.246543** | **ABORT** |
| 8 | 0.7130855231 | 0.7071067812 | 0.005979 | 0.6666666667 | 0.046419 | passes, at 93% of tol |
| 32 | 0.3687045648 | 0.3535533906 | 0.015151 | 0.3481553119 | 0.020549 | passes |

**Claim 3, the live trap, which is the finding's real cost.**
`tests/cameron/test_m3_etasks.py::test_the_chain_flipper_dependence_is_its_closed_form`
asserts `abs(want - 2.0/math.sqrt(t_star)) < 1e-12` under a docstring deriving
`2/sqrt(t*+1)`, and the module header at `:38` states the shipped closed form as
`2 / sqrt(t* + 1)`. The guard's stated reason and its asserted value disagree.
A consistency pass that made the code match the docstring — the exact kind of pass
this repo runs — would change `chain_flipper_dependence` to `2/sqrt(t*+1)` and
abort the `t*=2` block, destroying the one block whose N=8 seed CI at (150, 2048)
is the only it.11-compliant reading in the grid (`[0.9462, 0.9591]`, LEARNABLE).
The test would still pass, because the assertion would have been "fixed" too.

**What the attack does NOT cost, stated so it is not oversold.** The crossing
`0.9724`, the ceiling `0.9354`, F-1, F-2, F-3, the LEARNS/NO READING verdicts and
the priced DAG's 9-trainings conclusion all survive: every executable line in the
path takes `Var(y) = t*`, so no number in the grid moves. What falls is the
record's stated warrant, the producer's stated justification
(`scale/r10_capacity_sweep.py:16`), the pricing tool's printed output
(`results/r10_it8_priced_dag.txt` via `scale/r10_it8_pricing.py:166`),
`README.md:189`, and the docstring of the test that guards the band.
