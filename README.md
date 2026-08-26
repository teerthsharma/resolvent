# ceq — pivot-routed attention with an optional fixed-point settle

**A query row's reading is replaced by a mixture over content-selected pivot rows, and
those mixture weights can be driven to a fixed point; at matched parameters the mixture
beats causal softmax on every seed, and the fixed point on top of it buys nothing
measurable.**

`https://github.com/teerthsharma/resolvent`

---

## Abstract

`ceq` is an attention arm that replaces query row `i`'s single softmax reading with a
weighted mixture over `k` content-selected pivot rows, and optionally drives the mixture
weights to the fixed point of a settling map before reading out. The insight that makes
the fixed point affordable is that every iterate of the settling map is a positive
combination of the same `k` pivot rows, so the whole orbit is determined by its weight
vector on the `k`-simplex: the iteration closes in `k` dimensions, never touches an
`s`-sized object after setup, and adds `O(t* k²)` to an `O(s² d)` glance. In the Hilbert
projective metric the map contracts with modulus exactly `β`, independent of sequence
length, logit scale and pivot content. Measured on `negation_scope` at a
parameter-matched `n_params = 4769` across all five arms and five seeds, the settled arm
and its unsettled twin each beat causal softmax on every seed — `+0.108437` and
`+0.111396` NRMSE, exact 95 % intervals excluding zero. The settle itself is worth
nothing: `settled − twin = −0.002959` with an interval covering zero, while a one-hot
control is *worse* than plain softmax. The gain is the mixture, not the equilibrium. A
Lean 4 core of 39 theorems across six modules proves that the equilibrium oracle is not
the arm's own resolvent and that no truncation of it is ever exact.

## Background

### Why route through pivots rather than deepen the stack?

**A single attention row is one reading of the context, and it is normalised over the
whole context.** Causal softmax gives row `i` a distribution over `j < i` and the output
`A[i] V`. Every token's contribution is divided by a denominator that sums over `s`, so
the influence of any fixed token decays as the context grows. Routing the reading through
a small set of pivot rows fixes the number of paths that reach the readout, which is a
different quantity from the denominator and moves independently of it.

**The orbit lives in `k` dimensions, not in `s`.** The settling map is defined over the
token support, but every iterate is a positive combination of the `k` pivot rows, so
substituting the mixture back into the map closes it on the `k`-simplex with no reference
to `s` at all. The Gram matrix `G = A_P A_Pᵀ` and the product `A_P V` are each formed
once, at the same order as the glance; the loop after that is one `k × k` matrix-vector
product per step. The statistics therefore live on an object whose size does not grow
with sequence length.

**The certificate is a contraction modulus, not a spectral gap.** `α ↦ G α` is linear and
positive, hence non-expansive in the Hilbert projective metric; `x ↦ x^β` is
order-preserving and homogeneous of degree `β`; multiplication by the fixed positive gate
is a positive diagonal scaling and cancels in the coordinate ratios. Composing gives
`κ(T) ≤ β` exactly (Lemmens–Nussbaum, `arXiv:1304.7921`, Thm 2.9 and Prop 2.6). Birkhoff's
`tanh(Δ/4)` bound is not used and would not help: at this geometry it reads exactly `1.0`
in 30 of 30 measured cells (`scale/foreman_hilbert.py`).

**A fixed point is only testable against a label that has one.** Both oracles in the
original corpus are closed-form functions of the input —
`x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` and `x[:, :, CH_FLIP].sum(dim=1) ** 2` — with no
equilibrium anywhere in them. An arm that runs a Neumann series to convergence and then
predicts a product of two numbers ties an arm that does one normalisation and predicts
the same product, and the tie carries no information about settling. The equilibrium
corpus below exists to remove that confound, and it changes what the near-zero contrast
means without changing its value.

### Prior Art

Every row is a system or a published result, with the reference that carries the number.
The last column is the one where the alternatives are stronger.

| System | Reference | What gets iterated | Does iterating help there? |
|---|---|---|---|
| Softmax attention, one step | Ramsauer et al., ICLR 2021, `arXiv:2008.02217` | nothing — the update *"converges with one update"* | n/a; and softmax is Bayes-optimal on the single-location shape this corpus uses (`arXiv:2410.01537`) |
| Sinkformer | Sander, Ablin, Blondel, Peyré, AISTATS 2022, `arXiv:2110.11773` | the normalisation, to a doubly-stochastic fixed point in 3–5 steps | yes, for small accuracy gains — the nearest published relative of the settled cell |
| Deep equilibrium models | Bai, Kolter, Koltun, NeurIPS 2019, `arXiv:1909.01377` | the whole representation, to a root-found fixed point | *"equal or superior perplexity"* — parity, at real solver cost |
| Looped transformers | Yang, Lee, Nowak, Papailiopoulos, ICLR 2024, `arXiv:2311.12424` | the representation, weight-tied with input injection | **yes — matches standard transformers at under 10 % of the parameters** |
| DeltaNet WY | ships in sglang and `flash-linear-attention` | nothing; the signed multi-hop path sum is closed-form | n/a — **and it reads at or above this module on this module's own probe at every context length from 16 up** |
| Deep Equilibrium Algorithmic Reasoner | Georgiev, Wilson, Buffelli, Liò, NeurIPS 2024, `arXiv:2410.15059` | the reasoner state, on ~10 CLRS-30 tasks, with no ground-truth step count | yes, and it is the closest occupant of the equilibrium-label framing |
| `ceq` settled arm | this repository | the pivot weight vector `α` on the `k`-simplex | **no — `settled − twin = −0.002959`, 95 % CI `[−0.042903, +0.031557]`, covering zero** |

Two further occupancies are recorded rather than argued around. Equilibrium-shaped labels
as a *capability bar* are occupied by CLRS (Veličković et al., ICML 2022,
`arXiv:2205.15659`) and, for attention specifically, by Sanford, Hsu and Telgarsky
(`arXiv:2402.09268`), whose Theorem 4.2 gives `L = ⌊log₂ k⌋ + 2` as the depth needed for
`hop_k`. The full fetch, including what was searched for and **not found**, is in
[`PRIOR_ART.md`](PRIOR_ART.md) §4.

## Theoretical Foundation

### 1) The glance

Causal softmax attention gives query row `i` a distribution over the tokens before it and
reads out their values:

$$
A_{ij} \;=\; \frac{\exp(q_i \cdot k_j)}{\sum_{j' < i} \exp(q_i \cdot k_{j'})}, \qquad
\mathrm{out}_i \;=\; \sum_{j<i} A_{ij}\, v_j
$$

One pass, no fixed point. This is the `glance` arm, and it is bound bitwise to the
`softmax` baseline at `t_max = 0`.

### 2) The settling map, and why it closes in `k` dimensions

Let `P` be a set of `k` content-selected pivot rows, `a_p` the attention row of pivot `p`,
and `gate_p` row `i`'s own softmax weight on `p`. The settling map on the simplex over the
token support is

$$
w_p(m) \;=\; \mathrm{gate}_p \,\langle a_p, m\rangle^{\beta}, \qquad
T(m) \;=\; \frac{\sum_p w_p(m)\, a_p}{\sum_p w_p(m)}
$$

Every iterate of `T` is a positive combination of the `k` fixed vectors `a_p`, so writing
`m = Σ_q α_q a_q` gives `⟨a_p, m⟩ = (G α)_p` with `G = A_P A_Pᵀ`, and the iteration closes
in `k` dimensions:

$$
\alpha \;\leftarrow\; \mathrm{normalise}\!\left(\mathrm{gate} \odot (G\alpha)^{\beta}\right)
$$

This is the same map, not an approximation of it: the two orbits agree exactly once the
first step is taken, and `scale/arm_s.py::g3_bind` binds that identity by value rather
than asserting it. The shipped forward runs in the log domain; the probability-domain form
was checked against it and agrees to `2.998307e-44` / `1.314968e-17` / `7.758876e-13` at
seeds 0/1/2.

Positivity is structural rather than sampled: `G_pq = ⟨a_p, a_q⟩ ≥ a_p[0] a_q[0] > 0`
because coordinate 0 lies on the causal support of every non-empty row, and `gate_p > 0`
because `p < i` puts `p` on row `i`'s support. Float underflow is the one gap that
argument does not close — `a_p[0]` can reach exactly `0` in float32 — and it is stated as
open rather than closed.

### 3) The contraction modulus

In the Hilbert projective metric on the `k`-simplex,

$$
\kappa(T) \;\le\; \beta
$$

exactly, independent of `s`, of the logit scale and of the pivot readings. The shipped
setting is `β = 0.5`. The implicit-gradient backward uses `N = 21` Neumann terms, taken
from `arm_s.neumann_for(0.5)` — routed through the published `κ = tanh(Δ/4)` inverse,
`Δ = 2 log((1+β)/(1−β))` — so `N` is fixed before the run rather than tuned after it. The
backward never forms the Jacobian; each term is one vector-Jacobian product through a
single re-evaluation of the step map at the fixed point.

### 4) The equilibrium corpus, and its truncation law

The equilibrium tasks take their labels from an absorbing-chain solve — `N = (I − Q)^{-1}`,
`B = N R`, the fixed point of `z ← Q z + R`. That is not a closed-form function of any
bounded neighbourhood, so an arm that iterates has something to compute.

The label provably requires iteration, with a closed form rather than a demonstration. An
`h`-hop truncated reading — `h` is the hop budget here, not the pivot count `k` of the
sections above — incurs

$$
\mathrm{NRMSE}(h, t^\*) \;=\; \sqrt{\frac{t^\* - h}{t^\*}}
$$

Measured, against a predict-the-mean bar of `1.0`:

| `t*` | `h=0` | `h=1` | `h=2` | `h=4` | `h=8` | `h=16` | `h=32` |
|---|---|---|---|---|---|---|---|
| 1 | 1.000248 | 0.000000 | | | | | |
| 2 | 1.000023 | 0.701860 | 0.000000 | | | | |
| 8 | 1.000007 | 0.932740 | 0.863514 | 0.714329 | 0.000000 | | |
| 32 | 1.000004 | 0.983744 | 0.969120 | 0.934971 | 0.867263 | 0.704263 | 0.000000 |
| 63 | 1.000030 | 0.990328 | 0.986177 | 0.972203 | 0.938035 | 0.866643 | 0.705080 |

`h = 0` is **exactly** the bar — the closed form is exactly `1.0` there, and the
drawn-data test asserts `>= 1.0` with the closed-form match held under `0.03`. That
property had to be earned:
the first encoding gave the query token a driver, so `1/(t*+1)` of the label was legible at
zero hops and a 0-step RED gate aborted three of five rungs. Setting `b[s-1] = 0` in
`make_equilibrium_batch` makes zero hops exactly predict-the-mean.

The construction is deliberately nilpotent rather than contractive: the sub-diagonal is
zeroed at and before `head = s − 1 − t*`, so `A` is nilpotent of index `t* + 1` on the read
coordinate and the resolvent *terminates* instead of converging. A contraction would make
`t*` a tolerance, and it has to be a hop count. Coefficients are Rademacher and drivers
Gaussian, so the label is exactly `N(0, t*+1)` — which is what puts the truncation error in
closed form with no constant fitted. Generator `make_equilibrium_batch` and oracle
`equilibrium_oracle` in `scale/negation_scope.py`; the closed form is executable as
`ceiling(t_star, hops)` in `scale/e_ladder.py`; the check that the whole family exists for
is `tests/cameron/test_m3_etasks.py::test_a_fixed_k_hop_truncation_cannot_get_the_chain_label`,
parametrised over `t* ∈ {2, 8, 32}`, which asserts the closed-form match, the bar at
`h = 0`, monotone decrease in `h`, and NRMSE exactly `0.0` at `h = t*`.

### 5) The oracle is not the arm's own resolvent

An arm whose label is the object its own forward computes measures nothing. Two theorems
in `lean/CEQ/OracleSeparation.lean` close that, and they turn on nilpotency rather than on
label shape. The arm's operator is strictly lower triangular, so `Aⁿ = 0`. The oracle's `Q`
is non-negative with symmetric support and at least one positive entry, so it doubles on
the diagonal and is never nilpotent, since `n < 2ⁿ`. Hence

$$
\texttt{oracle\_ne\_resolvent} : Q \neq A
$$

and the theorem the truncation ladder actually needs,

$$
\texttt{truncation\_never\_exact} : \textstyle\sum_{k<N} Q^k \;\neq\; (1-Q)^{-1} \quad \text{for every } N
$$

For the arm's own operator the same sum *does* become an equality at `N = n`. So every rung
of the ladder leaves a real residual, and that is proved rather than assumed. `SymmSupport`
is deliberately weaker than symmetry, because `D⁻¹W` is not symmetric at unequal degrees,
and `zero_not_a_counterexample` records that the edge hypothesis is load-bearing.

## Implementation

### Five arms, one geometry, one parameter count

Arms 3, 4 and 5 share every parameter tensor and every line of the forward except how the
pivot weight vector `α` is obtained, so a contrast between them is a contrast between
settling rules and not between architectures.

| arm | `α` | hops | role |
|---|---|---|---|
| `softmax` | — | 1 | the baseline, reproduced against its published reading before anything else runs |
| `glance` | — | 2 | ARM S at `t_max = 0`; bound **bitwise** to `softmax`, so its row is a structural zero |
| `settled` | log-domain fixed point, `β = 0.5`, implicit gradient `N = 21` | 2 | the arm under test |
| `twin` | the normalised gate — setup paid, loop not run | 2 | isolates the settle from the routing |
| `argmax` | one-hot at `argmax(gate)` | 2 | the attribution control: does the mixture reduce to a lookup? |

The three settling cells are genuinely distinct objects at this geometry rather than
numerically identical ones, which is the precondition the project had previously failed
five times. Measured at random init, mean over seeds 0–4:

| `k` | `‖settled−twin‖/‖twin‖` | `‖settled−argmax‖/‖argmax‖` | gate max | uniform value |
|---|---|---|---|---|
| 8 | 1.107578e-01 | 5.874960e-01 | 0.129379 | 0.125 |
| 16 | 1.164525e-01 | 6.086490e-01 | 0.063773 | 0.0625 |
| 32 | 1.277286e-01 | 6.331066e-01 | 0.032274 | 0.03125 |

`scale/arm_s.py` is written per example; the deciding run trains on `[n, s, d]` with
`n = 8192`, so the map is batched in `scale/m3_quintuple.py` and bound **bitwise** against
the per-example original by `_bind_batched_against_arm_s`, which runs before any cell and
aborts the run if it fails.

### The Lean core

Lean 4.7.0 with mathlib. `lake build CEQ` exits 0, there is **zero `sorry`**, and
`#print axioms` lists only `[propext, Quot.sound, Classical.choice]` — no `sorryAx`.
**39 theorems across six modules**: `Contraction` 5, `Nilpotent` 4, `Occupancy` 3,
`OracleSeparation` 12, `OrbitBound` 5, `Refcount` 10. Every declaration is a `theorem`;
there are no `lemma`s, so the count is not theorems-plus-lemmas rounded up. A naive
grep for an indented `theorem` reads 40, because `Refcount.lean:39` is the word inside a
doc comment.

| theorem | statement |
|---|---|
| `CEQ.OracleSeparation.oracle_ne_resolvent` | the absorbing-chain oracle `Q` is not the arm's operator `A`, via nilpotency |
| `CEQ.OracleSeparation.truncation_never_exact` | `∑_{k<N} Qᵏ` is the inverse of `(1−Q)` at no `N` |
| `CEQ.OracleSeparation.zero_not_a_counterexample` | negative control: the edge hypothesis is load-bearing |
| `CEQ.Nilpotent.pow_card_eq_zero` | strictly lower-triangular ⇒ `Aⁿ = 0`, over any `CommRing`, with **no sign hypothesis** |
| `CEQ.Nilpotent.one_not_nilpotent` | negative control: `.tril(0)` is nilpotent at no power, so `.tril(-1)` cannot be weakened |
| `CEQ.Occupancy.occupancy_telescope` | `(1−A)·∑_{k<N}Aᵏ = 1−Aᴺ`, over any ring |
| `CEQ.Contraction.weighted_contraction` | Perron certificate ⇒ contraction in the weighted sup norm |
| `CEQ.Contraction.expander_expands_l2` | negative control: `!![1,0;1,0]` is row-stochastic with `σ_max = √2`, refuting the `σ_max` form of the contraction claim |

The proofs are bound to the shipped tensors rather than to a paper statement.
`tests/foreman/test_oracle_separation_binding.py` passes 6/6 in 103.46 s: reachability from
the root module, `#print axioms` on five names **with a planted `sorryAx` seen to fire**,
the three Lean hypotheses checked entrywise on the actual `Q`, and a must-fire in which the
arm's own `.tril(-1)` operator is correctly *rejected* by the support check.

## Results

### The deciding measurement

Task `negation_scope`, geometry `s64_d24_st150_ntr8192_nev512_b21`: sequence length `s = 64`,
**flipper distance** `d = 24` (*not* `d_model`, which is 16), 150 steps,
`n_train = 8192`, `n_eval = 512`, settle cap `t_max = 21`, `k = 8` pivots, five seeds,
and `n_params = 4769` on **every** arm — 256 + 256 for `wq`/`wk`, 2048 + 128 + 2048 + 16
for the MLP at `hidden = 128`, 16 + 1 for the readout. Asserted against the module rather
than against a file by
`tests/chase/test_m3_capability_harness.py::test_param_counts_are_equal_across_every_arm`. Metric is eval NRMSE, so lower is better and `1.0` is
predict-the-mean. Journal `results/m3_quintuple_v2.jsonl` at commit `9629616`;
pre-registered outcome table in `M3_QUINTUPLE_PREREGISTERED_READING.md`, written before the
file produced a number.

| arm | seed 0 | 1 | 2 | 3 | 4 | mean | sd | beats predict-the-mean |
|---|---|---|---|---|---|---|---|---|
| `argmax` | 1.022910 | 0.994867 | 1.011705 | 1.010804 | 1.013609 | **1.010779** | 0.010115 | **no — credited with nothing** |
| `softmax` | 0.877168 | 0.889523 | 0.919148 | 0.890175 | 0.885603 | **0.892323** | 0.015866 | yes |
| `glance` | 0.877168 | 0.889523 | 0.919148 | 0.890175 | 0.885603 | **0.892323** | 0.015866 | yes |
| `settled` | 0.753581 | 0.768802 | 0.874658 | 0.816071 | 0.706317 | **0.783886** | 0.064106 | yes |
| `twin` | 0.767403 | 0.784397 | 0.794505 | 0.798001 | 0.760328 | **0.780927** | 0.016547 | yes |

Contrasts are `δ = NRMSE(reference) − NRMSE(arm)`, so positive means the arm has the lower
error. At five seeds the paired resample space is finite — `5**5 = 3125` tuples with 126
distinct values — so the intervals below are the **exact percentile over all 3125 paired
resamples**, with zero Monte-Carlo error, rather than a `B = 10000` draw. Strict at zero:
an interval touching zero does not exclude it.

| arm | reference | `δ` | exact 95 % CI | seeds favouring arm | verdict |
|---|---|---|---|---|---|
| `settled` | `twin` | **−0.002959** | `[−0.042903, +0.031557]` | 3/5 | **no difference** |
| `settled` | `softmax` | **+0.108437** | `[+0.068181, +0.147110]` | 5/5 | settled wins |
| `twin` | `softmax` | **+0.111396** | `[+0.100873, +0.121920]` | 5/5 | twin wins |
| `argmax` | `softmax` | **−0.118456** | `[−0.134115, −0.102786]` | 0/5 | softmax wins |
| `settled` | `argmax` | +0.226893 | `[+0.175040, +0.276921]` | 5/5 | *a win over a failure — see below* |
| `glance` | `softmax` | +0.000000 | `[+0.000000, +0.000000]` | 0/5 | structural zero, not a measured tie |

Substrate: Windows 11, Python 3.11.9, torch 2.5.1, CPU-only with threads pinned to 2 inside
`scale/m3_quintuple.py`. Wall-clock is not reported: the box is contended, `meta.seconds`
in the journal is labelled PROVISIONAL by its own producer, and no seconds figure enters a
verdict sentence. Cost, where it is reported at all, is reported as analytic FLOPs
(`scale/m3_flops.py`).

### Reproduction

```bash
# the deciding measurement itself (bucketed; one unit is one (cell, k, seed))
python scale/m3_quintuple.py --cells softmax glance settled twin argmax \
                             --seeds 0 1 2 3 4 --ks 8

# the table, assembled from the journal, with the negatives included
python -m scale.capability_table          # writes results/capability_table_v0.{json,md}

# the exact-enumeration intervals, direct from the journalled per-seed values
python -c "
import json, itertools, statistics
d = json.load(open('results/capability_table_v0.json'))
p = {a['arm']: a['per_seed'] for a in d['arms']}
idx = list(itertools.product(range(5), repeat=5))
def exact(arm, ref):
    dd = [p[ref][i] - p[arm][i] for i in range(5)]
    v = sorted(sum(dd[i] for i in t) / 5 for t in idx)
    q = lambda f: (lambda pos, lo: v[lo] * (1 - (pos - lo)) + v[min(lo + 1, len(v) - 1)] * (pos - lo))(f * (len(v) - 1), int(f * (len(v) - 1)))
    return statistics.fmean(dd), q(0.025), q(0.975)
for a, r in [('settled','twin'), ('settled','softmax'), ('twin','softmax'), ('argmax','softmax')]:
    print('%-8s - %-8s %+.6f  [%+.6f, %+.6f]' % ((a, r) + exact(a, r)))
"

# the proofs
cd lean && lake build CEQ
```

The falsifier suite collects **1,737 tests in 80.23 s** at HEAD on the machine described
above (`python -m pytest --collect-only -q tests/`). Running it in full takes over three
hours and has never completed in one pass; the gates that bind a claim in this document
are named beside that claim.

### What the numbers say

**Both arms beat softmax on every seed, by about the same margin.** `+0.108437` and
`+0.111396`, intervals excluding zero, 5/5 seeds each. The twin's interval is the tighter
of the two.

**The fixed point is worth nothing on top of the routing.** The point estimate for
`settled − twin` is *negative* — the unsettled twin is very slightly ahead — and the
interval covers zero comfortably in both directions. Three of five seeds favour settled,
which is what a coin does. What settling does add is variance: `sd 0.064106` against the
twin's `0.016547`, a factor of `3.874×` at the same batches, the same initialisation and
the same parameter count. There is a structural account of that, and it is not flattering:
the settled arm iterates `α` inside an 8-simplex over `{α @ av}`, which is a
reparameterisation **inside the twin's own function class**. It can reallocate; it cannot
add.

**The gain is the mixture, and a lookup is worse than nothing.** Collapsing the pivot
weights to a single one-hot reading does not merely lose the gain, it lands `−0.118456`
*below plain softmax*, 0/5 seeds, interval well clear of zero. `argmax` also fails its own
bar at mean `1.010779` — above `1.0`, so it does not beat predict-the-mean and is credited
with nothing. That makes `settled − argmax = +0.226893` a win over a failure, which
licenses nothing; it is recorded here because the largest positive number in the table is
the least meaningful one.

**The headline cell is undecided by arithmetic fixed before the run, not by the data.** The
per-draw Ville e-process reads `E_t = 0.9978` in the settled direction and `1.0019` in the
twin direction at `t = 5`, against `THRESHOLD = 40.0`. Its ceiling `max_attainable(5) =
3.80169140625` is below that threshold and `MIN_T_MIXTURE = 13`, so five seeds cannot cross
in either direction whatever the data say.

## Quick Start

```bash
git clone https://github.com/teerthsharma/resolvent
cd resolvent
pip install -r requirements.txt

python run_calib.py --self-test        # 4/4 bit-identical calibration, exit 0
python -m scale.capability_table       # the table above, rebuilt from the journal
python -m ceq.hf.smoke                 # the shippable package, CPU only
python -m ceq.diagnose                 # the sign-flip diagnostic, for another operator
```

`python -m scale.capability_table --artifact ceq/hf_artifact` assembles the Hub package —
the model card and the modelling code. **It ships no weights**, deliberately: a
random-init `model.safetensors` measures 1,901,686,656 bytes and publishing it would be
publishing noise. The only trained checkpoints on disk are `pivot_unsigned`, a different
arm family.

Benchmark corpora are not redistributed. COGS (Kim & Linzen 2020), SCAN (Lake & Baroni
2018) and TinyStories (Eldan & Li 2023) carry their own licences; `data/README.md` has the
provenance and the exact fetch commands.

## Requirements

Developed and measured on Python 3.11.9. No `python_requires`, `setup.py` or
`pyproject.toml` is declared, so no supported range is claimed or tested. Pins are the
versions actually installed and exercised, read off the
installed packages rather than taken from PyPI's latest — see `requirements.txt` for the
reasoning behind each.

| package | pin | note |
|---|---|---|
| `torch` | `2.5.1` | the measurement loop is CPU-only; the local build is `2.5.1+cu121` but the `+cu121` local segment resolves only from PyTorch's own index |
| `transformers` | `==5.3.0` | pinned tight: `ceq/hf/modeling_ceq.py` sets six private `PreTrainedModel` attributes that are undocumented HF internals and can be renamed in a patch release |
| `numpy` / `scipy` | `1.26.4` / `1.17.1` | |
| `ripser` / `persim` | `0.6.14` / `0.3.8` | reference oracle for the persistent-homology gate |
| `datasets` / `huggingface_hub` | `4.8.4` / `1.7.1` | Hub packaging only |
| `pytest` | `9.0.3` | |
| `triton-windows` | `3.7.1.post27`, `sys_platform == "win32"` | imported at module scope by `ceq/mz_kernel.py`, exercised only behind CUDA compute capability ≥ 8.0. On Linux/macOS the package is `triton`; no version is pinned there because none could be verified on this machine |

Lean 4.7.0, pinned by `lean/lean-toolchain`; mathlib4 at git tag `v4.7.0`, pinned by
`lean/lake-manifest.json` to rev `a45ae63747140c1b2cbad9d46f518015c047047a`.
CUDA is optional throughout: every test parametrises over `cpu` and `cuda` and skips the
`cuda` leg when `torch.cuda.is_available()` is false. GPU figures elsewhere in the
repository were taken on an RTX 4060 Laptop (`sm_89`, 8.0 GiB, 24 SMs) and a utilization
number from a laptop part does not transfer to an A100.

## Limitations

Collected here rather than scattered through the sections above.

**Every task in the original corpus is static.** Both registered oracles are closed-form
functions of the input with no fixed point anywhere in them:
`negation_scope.oracle` is `x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]`
(`scale/negation_scope.py:69`) and `counter_squared_oracle` is
`x[:, :, CH_FLIP].sum(dim=1) ** 2` (`scale/negation_scope.py:128`). A settling arm has
nothing to settle toward on either, so the headline `NO DIFFERENCE` is evidence about
these tasks and not about settling in general — the equilibrium corpus exists precisely
because that contrast could not carry the claim.

**The deciding ladder is pre-asymptotic at every rung.** The rungs `t* ∈ {1, 2, 8, 32}` sit
against mode times `20.5615` and `169.3116`, so `λ₂^{t*}` is the wrong predictor *at the
rungs* even though `λ₂` is the right asymptotic rate. Reading the curve with `λ₂^{t*}`
makes it look wrong for arithmetic reasons rather than architectural ones. The closed form
`q_t − q = (u ⊙ Qᵗ s − s ⊙ Qᵗ u)/(s ⊙ s_t)` is the right predictor and costs one solve.

**The ladder is one rung of four.** `results/m3_quintuple_v2.jsonl` holds 35 rows: the 25
`negation_scope` units reported above, plus 10 `e3_t1` units (settled ×5, twin ×5) taken at
a *different* geometry, `n_train = 2048, n_eval = 2048`. `t* ∈ {2, 8, 32}` have zero rows.
`scale/e_ladder.py` reads the same journal and refuses a verdict on a partial ladder, so no
row of the pre-registered outcome table has fired.

**The built table on disk is behind this document.** `results/capability_table_v0.{md,json}`
records `journal_commit` and `head_commit` both at `9629616` and prints the Monte-Carlo
interval family; regenerate it with `python -m scale.capability_table` before citing it as
current. Three clauses of its own `Limits` string have also gone stale — it says
`scale/m3_quintuple.py` has no `--task` flag (it has one, at line 542), it cites a
`CHECKLIST.md` line for the defective `+0.146551` endpoint that now reads `+0.147110`, and
it gives "the quintuple journals metrics only" as the reason the consequence-fidelity
column is empty, though per-cell weights now exist under `results/m3_quintuple_v2_weights/`.
The column is still empty; the reason given for it is not the current one.

**Five seeds cannot decide anything anytime-validly.** `eprocess.MIN_T_MIXTURE = 13` and
`max_attainable(5) = 3.80169140625` against `THRESHOLD = 40.0`. Every interval in this
document is fixed-sample, read once, and carries no anytime validity. A real gap below
roughly `0.05` NRMSE reads `NO DIFFERENCE` at five seeds whether or not it is real
(`M3_QUINTUPLE_PREREGISTERED_READING.md:87`).

**No weights are published.** The Hub package is the card and the modelling code. The
consequence-fidelity column of the capability table is consequently an empty one — it needs
trained per-arm weights to intervene on, and `scale/m3_quintuple.py` journals metrics
rather than tensors.

**The pivot family cannot see the token the equilibrium label depends on.** `pivots_of`
excludes the query row and drops pivot 0, so the largest legal pivot is `s − 2`, and a
strictly causal `tril(-1)` makes that row read only `j ≤ s − 3`. Softmax's own row is
`s − 1` and reads `j ≤ s − 2`. Measured on `e3_t1`, seed 0, perturbing token `s−2` by
`+100`: the softmax cell moves `101.6983` while the pivot `av` moves `3.263746`. Every
reading of `settled` against `twin` against `softmax` **on the equilibrium tasks** is
confounded by that exclusion and is not reported here as evidence. The root cause is a
defensible independence argument in a docstring — *a pivot reading of the row being
settled is not an independent reading of it* — that became a capability ceiling.

**The arms do not reach their own hop ceiling.** At `t* = 8` the best arm evaluates
`1.112208` against a measured 2-hop ceiling of `0.866025`, short by `+0.246`, while
training to `0.860972` — sitting on the ceiling and memorising noise. Depth is the likely
binding constraint: `hop_k` needs `L = ⌊log₂ k⌋ + 2` (`arXiv:2402.09268`, Thm 4.2) and
these arms are depth 1, where `t* = 8` wants about 5.

**Two interval families exist in this repository under one label.** The shipped
`results/capability_table_v0.md` prints the Monte-Carlo `B = 10000, seed = 0` percentile
bootstrap; this document prints the exact enumeration, and they differ at two endpoints
(`settled − twin` lower: exact `−0.042903` against Monte-Carlo `−0.048587`;
`argmax − softmax` upper: exact `−0.102786` against `−0.102204`). One further endpoint,
`+0.146551` for `settled − softmax`, is in neither family and is carried as an open defect
rather than adopted.

**The earlier `sgate`/`tgate` operator line is a negative result and stays one.** The
context-stability numbers this project published for several rounds were measured on
`_causal_tgate_operator`, which appears nowhere in the shipped path; the shipped `sgate`
carries a denominator that sums over context and its routed sign-flip rate reads
`0.16511 / 0.02732 / 0.00000 / 0.00000` at `s = 8/32/128/512`, slope `−1.298` against a
pre-registered bar of `−0.3`. On capability it lost outright: COGS generalization exact
match `0.0000` (0/512) against a softmax control's `0.0293` (15/512) at 3,652,096 matched
parameters, one-sided Fisher exact `p = 2.7502788939e-05`, against a published
from-scratch encoder-decoder at `0.35 ± 0.06` (`arXiv:2010.05465`). The full autopsy is in
[`PROGNOSIS.md`](PROGNOSIS.md) and [`D1.md`](D1.md); it is kept because a falsification
harness that caught seventeen of its own broken instruments — including the one that
invalidated its own headline — is the part of this repository most worth reading.

## License

MIT.

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app)*
