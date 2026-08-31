# V15 — audit of the contract's own arithmetic, before anything is built on it

Coordinator node, it.2. Written because the previous round's `c3cce2c` did the
same thing to `v-main.8d/8e` and found errors there: a contract's numbers are
load-bearing from the moment an arm is measured against them, and every number
below is one an iteration will later cite as a threshold.

Method: each claim re-derived numerically in float64 and run, not checked by
inspection. Probe scripts are inline in this document so the reading is
reproducible. `scipy 1.x`, `numpy 1.26.4`.

This audit checks the contract **against itself and against arithmetic**. It does
not check it against prior art — that is `n1`'s node under L-EQ.

---

## SUMMARY

| # | claim | verdict |
|---|---|---|
| A-0a | `floor_1 = sqrt((t*-1)/t*)` = 0.7071 at `t*=2` | **CONFIRMED** |
| A-0b | `floor_1` = 0.9354 at `t*=8` | **CONFIRMED** |
| A-0c | C-PAR CI first fits at `N=23` | **CONFIRMED** |
| A-0d | C-PAR power 0.80 first at `N=70` | **CONFIRMED** (see the correction note) |
| A-0e | `T* = DeltaDeltaE / ln m` is the parallel-saddle crossover | **CONFIRMED** |
| A-0f | scoreboard ceiling 38 | **CONFIRMED, and it is exact, not approximate** |
| A-0g | GL binds: `alpha -> 0` identity, `alpha -> 1` cumulative sum | **CONFIRMED** (only via the recurrence — see A-3) |
| **A-1** | "TOST retires to `N >= 23` runs" | **FINDING — licenses a measurement with 6.7% power** |
| **A-2** | `w_k = (-1)^k C(-alpha, k)` as written | **FINDING — the direct transcription returns NaN at the contract's own bind** |
| **A-3** | `H-hat = alpha-hat + 1/2` | **FINDING — shipped without its stationarity hypothesis** |
| **A-4** | "Ceiling approximately 38 from the current 22%" | **FINDING — `22%` has no live producer in the repository** |

Four findings. Three are in clauses the contract itself introduces; A-2 and A-3
are both instances of the exact defect class **L-EQ** was written to catch, and
they are in the contract that introduces L-EQ. Filed accordingly.

---

## CONFIRMED, with the numbers

### A-0a / A-0b — the floors

```
t*=2 : floor_1 = sqrt(1/2) = 0.707107
t*=8 : floor_1 = sqrt(7/8) = 0.935414
t*=32: floor_1 = sqrt(31/32) = 0.984251
```

Contract's `0.7071` and `0.9354` are correct to the digits quoted. R1's and R2's
thresholds stand.

### A-0c — the C-PAR confidence interval first fits at N=23

Half-width of the 90% CI on a two-sample difference, in units of sigma, against
the `Delta_eq = 0.5 sigma` margin:

```
half-width(N) = t(.95, 2N-2) * sqrt(2/N)

N =  8 : 0.8807  fits = False      <- the registered seed count
N = 22 : 0.5071  fits = False
N = 23 : 0.4955  fits = True       <- first N
N = 70 : 0.2799  fits = True
```

`N=23` is exact: `N=22` misses by `0.0071 sigma`. The pre-v13 census figure is
correct.

### A-0d — power 0.80 first at N=70

**A correction is recorded here against this audit's own first pass.** A 60,000-
draw Monte Carlo read `N=69: 0.802` and this document's draft called the
contract off by one. At 4,000,000 draws (`se(power) ~ 0.0002`):

```
N = 23 : power = 0.0669
N = 67 : power = 0.7830
N = 68 : power = 0.7910
N = 69 : power = 0.7985      <- below 0.80
N = 70 : power = 0.8059      <- first N at or above 0.80
N = 71 : power = 0.8132
```

`N=69` sits `0.0015` below the bar — inside the noise of a 60k run and outside
the noise of a 4M run. **The contract's `N=70` is correct.** The 60k reading was
the audit's error, not the contract's, and it is left in this document rather
than deleted because it is a live instance of the repo's `M-4` mechanism (a
single-sample interval read as if it settled a boundary).

### A-0e — the parallel-saddle crossover

Two channels: one saddle at barrier `E`, and `m` parallel saddles at `E + ddE`.
Rates `exp(-E/T)` and `m * exp(-(E+ddE)/T)`. At `ddE = 1.0`, `m = 5`,
`T* = 1/ln 5 = 0.6213`:

```
T = 0.5592 (0.9 T*) : rate_low = 1.6725e-01  rate_high_m = 1.3986e-01  higher wins = False
T = 0.6213 (   T* ) : rate_low = 2.0000e-01  rate_high_m = 2.0000e-01  equal
T = 0.6835 (1.1 T*) : rate_low = 2.3151e-01  rate_high_m = 2.6799e-01  higher wins = True
```

The higher barrier becomes the faster channel above `T*` and not below it.
BED-1's rule — label by committor, never by barrier height — is arithmetically
justified, not just stylistically preferred.

### A-0f — the scoreboard ceiling

```
[2, 12, 4, 2, 8, 4, 2, 1, 3] -> sum = 38
```

Exactly 38. The contract writes "approximately 38"; the hedge is unearned and
should be dropped, since a reader cannot tell whether "approximately" conceals a
line item that was not written down.

---

## FINDINGS

### A-1. "TOST retires to `N >= 23` runs" licenses a measurement with 6.7% power

**Mechanism.** The contract permits a TOST verdict from `N = 23` upward, on the
grounds that the CI first fits there. But fitting the CI and being able to
*declare* equivalence are different events. At `N = 23`, with `Delta_eq = 0.5
sigma` and two bit-identical arms — the most favourable case that can exist —
achieved power is **0.0669**. A TOST run at `N = 23` returns NO VERDICT about
93.3% of the time when the arms are literally the same operator.

This is the repository's own **M-5: a process that cannot cross its own
threshold**, and separately **M-9: a verdict whose finest achievable p cannot
reach the alpha it quotes**. Both are already in `MISTAKES.md`. The contract
reintroduces the mechanism in the clause that retires the previous instance
of it.

**What is actually true.** The two numbers in the contract are both right and
they answer different questions:

| N | what it is the first N for |
|---|---|
| 23 | the CI half-width fits inside the margin **for some draw** |
| 70 | the CI half-width fits inside the margin **with probability 0.80** |

**Proposed amendment** (not applied — the contract is the author's):
> TOST retires to `N >= 70` for a VERDICT. `N in [23, 70)` licenses a CI
> statement only, and any such row must carry the achieved-power column so a
> reader cannot mistake a 7%-power NO VERDICT for evidence of difference.

Note this amendment is already half-present: Mars's standing attack #2 is
"parity by underpowering" and requires exactly that achieved-power column. The
attack and the clause it attacks are both in the same contract.

### A-2. `w_k = (-1)^k C(-alpha, k)` cannot evaluate its own bind

**Mechanism.** The contract fixes the fractional head by the Grunwald-Letnikov
weights `w_k = (-1)^k C(-alpha, k)` and binds them at two points: `alpha -> 0`
gives identity, `alpha -> 1` gives cumulative sum. Transcribed directly to
`scipy.special.binom(-alpha, k)`, the `alpha = 1` bind returns **NaN**:

```
alpha=0.0 : scipy.binom = [ 1. -0. -0. -0. -0. -0.]
alpha=0.3 : scipy.binom = [ 1.  0.3  0.195  0.1495  0.123337  0.10607]
alpha=1.0 : scipy.binom = [nan nan nan nan nan nan]      <- the contract's own bind
```

`binom(-1, k)` hits scipy's negative-integer branch. The mathematics is fine —
`(1-z)^-1 = sum_k z^k`, so `w_k = 1` for every `k` — but the formula as written
in the contract, handed to the obvious library call, produces NaN at exactly the
point the contract uses to certify the construction.

**Why this matters beyond a library quirk.** A bind test written against the
scipy form and asserting "no exception raised", or using `np.allclose(...,
equal_nan=True)`, would pass on NaN. That is a **V-10** (a gate whose threshold
is satisfied by construction) waiting to be written, and the S-K head is
scheduled for it.12-14.

**Route, verified.** Use the ratio recurrence:

```
w_0 = 1 ,  w_k = w_{k-1} * (alpha + k - 1) / k
```

```
alpha=0.0 : [1. 0. 0. 0. 0. 0.]            BIND identity  : True
alpha=0.3 : [1. 0.3 0.195 0.1495 0.123337 0.10607]   matches scipy exactly
alpha=1.0 : [1. 1. 1. 1. 1. 1.]            BIND cumsum    : True
```

The recurrence agrees with scipy wherever scipy is defined and is finite where
scipy is not. **Any implementation of the S-K head must use the recurrence, and
its bind test must assert on real numbers with `equal_nan=False`.**

### A-3. `H = alpha + 1/2` ships without its stationarity hypothesis

**Mechanism.** The contract specifies BED-K's power-law bed as "power-law `K`
(fBm-type, `H > 0.5`)" and its recovery target as `H-hat = alpha-hat + 1/2`. The
relation is correct for fractionally integrated noise. Its hypothesis is not
stated: ARFIMA(0, d, 0) is stationary and invertible **iff `|d| < 0.5`**, so the
relation's domain is `alpha in (0, 0.5)` and therefore `H in (0.5, 1.0)`.

The contract gives a lower bound on `H` and no upper bound. At `alpha >= 0.5`
the kernel weights do not decay fast enough for finite variance:

```
alpha=0.25 : w_199 = 5.203e-03   sum_w =  4.147   finite variance = True
alpha=0.45 : w_199 = 2.763e-02   sum_w = 12.244   finite variance = True
alpha=0.50 : w_199 = 3.997e-02   sum_w = 15.948   finite variance = False
alpha=0.60 : w_199 = 8.077e-02   sum_w = 26.869   finite variance = False
```

Generating BED-K at `alpha >= 0.5` produces a non-stationary series whose
"Hurst exponent" is not a Hurst exponent of a stationary process, and every
estimator R3 scores against it would be reporting a number with no population
value behind it.

**This is precisely the defect L-EQ names**: the theorem's statement was carried
into the contract without its hypotheses. It is in the same document that
introduces L-EQ, which is the strongest available evidence that L-EQ is a real
and recurring class rather than a one-off strike. Filed under the author's name,
as L-EQ itself is.

**Route.** BED-K registers the box `alpha in (0, 0.5)`, `H in (0.5, 1.0)`,
before any cell runs, and the generator refuses `alpha >= 0.5` rather than
silently producing it.

### A-4. "the current 22%" has no live producer

**Mechanism.** The SCOREBOARD section reads "Ceiling approximately 38 from the
current 22%". A repo-wide search for that baseline:

| hit | file:line | what it actually is |
|---|---|---|
| `22%` | `BACKLOG.md:7` | prose — "Closing a 22% gap is a campaign across iterations" |
| `22%` | `DONE_ARCHIVE_ROUND1.md:523` | a different quantity — an arm "spent 54% more wall-clock to lose by 22%" |

The two live scoreboards both read something else:

| file:line | value |
|---|---|
| `README.md:55` | **Scoreboard: 0 of 39.** |
| `workdonenew.md:44` | **Scoreboard: 0 of 39.** Itemised in section 8 |

`0 of 39` is `0%`, not `22%`. No file in the tree produces `22%` as a scoreboard
position. This is **P-1 (a number with no live producer)** and possibly also
**P-3 (a stale claim never retracted)** if `22%` was a scoreboard figure from an
earlier round that the two current documents have since superseded.

**Consequence if left.** The scoreboard "moves only at it.10, 17, 21, 23, 30".
Every one of those moves is computed as a delta from a baseline. A baseline with
no producer makes all five deltas uncheckable, which is the whole point of
having a scoreboard.

**Route.** Either the baseline is `0 of 39` — in which case the ceiling sentence
should read "38 from the current 0 of 39" — or `22%` measures something the
repository does not currently compute, and the thing it measures needs naming
and a producer before it.10.

---

## WHAT THIS AUDIT DID NOT CHECK

Stated so a later reader cannot mistake silence for a pass.

- **`rho_P` gate at 2** ("antisymmetric `J` gives 2 EXACTLY; `sqrt(2)` was the
  random-matrix value"). Not checked — `rho_P` is not defined in the contract
  text and the definition lives in X32, which this node did not open. This is a
  live `[V]`-grade statement under L-EQ until someone runs an instance.
- **Pesin deficit `0.0003` / `0.156`.** Not reproduced. Pesin's identity relates
  metric entropy to positive Lyapunov exponents under an SRB / absolutely
  continuous invariant measure; the contract states neither hypothesis. Same
  class as A-3, unverified rather than refuted.
- **`C_OPERATOR = 3.9 x 3.4 B/elt` and the 6.63x factor.** Covered by `n2b`
  (`V15_N2B_SIZING_CITATIONS.md`), not re-derived here.
- **Every `[RUN: ...]` figure in PART I** (`4.0e-15`, `3.1e-15`, `1.1e-16`,
  `1e-9`, `0.990`, `0.0375`, `0.519`, `6.4e-16`). These are inherited from the
  pre-v13 census and are not re-run by this node. Under L-TIME they are
  `INHERITED` and whichever iteration cites one must re-measure it.
