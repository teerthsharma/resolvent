# REMOVAL-ECHO v1 — pre-registration

Fixed before any data is downloaded. The predictions and counters below are the
author's, stated in the protocol that commissioned this work; they are recorded
verbatim and are not edited after data exists. A later change to any number in
this file is a change to the record and needs a row in `docs/canon/CORRECTIONS.md`.

## The question

Does a system's reaction to a dated removal break additivity — "one number per
thing" — and can that breaking be measured rather than inferred?

## The statistic

For units A, B with outcome f over subsets:

    Shapley  phi_A = sum_S [|S|!(n-|S|-1)!/n!] (f(S u A) - f(S))
    Mobius   I(A,B) = f(AB) - f(A) - f(B) + f(empty)

ADDITIVE iff I = 0 for every pair. "Breaks the rule" iff some I != 0 with a
confidence interval that excludes zero.

## Why this replaces the curl programme

The curl programme failed on five domains in a row, each for one of two reasons:
the system could not break the rule, or the removal hid the reaction. The first
reason has a name. A SUPERPOSITION system cannot break additivity — power grids
obey Kirchhoff's voltage law, so the curl of any flow response is zero by
theorem. An EQUILIBRIUM system can: removing an edge from a Braess network lowers
equilibrium travel time. Admission test for any domain: after the removal, does
the rest RE-EQUILIBRATE? If not, the domain is refused, not tested.

## Pre-registered predictions and counters

Counters are point estimates, stated as the outcome the author would bet on if
forced to bet against the prediction.

| domain | measurement | prediction | counter |
|---|---|---|---|
| D1 Perturb-seq double knockouts (Norman et al. 2019) | fraction of gene pairs with I(A,B) != 0 at BH q = 0.05 | **> 20%** of tested pairs | **< 5%** survive BH — most synergy is noise at this depth |
| D2 NBA injuries vs RAPM | RAPM residual variance on injury weeks vs placebo weeks | **x1.5** with CI | **x1.0** — RAPM already absorbs rerouting |
| ECHO, per domain | lag of peak impulse response | D1 at one measurement step; D2 at 1-3 games | — |

D3-D5 are sized after D1/D2 and are not run this week unless D1 fails admission.

## Kills, fixed now

- A domain without re-equilibration is **refused**, not tested.
- An I != 0 outside the Benjamini-Hochberg set is **not a finding**.
- An effect without its placebo rank is **not a number**.
- A lag peak inside the null band is **no echo**.
- Any counterfactual not from synthetic control or a planted double is **struck**.

## Must-fires, required before any domain number is reported

- A planted ADDITIVE game reads I = 0 to 1e-12.
- A planted SYNERGY reads its planted value.
- Synthetic-control placebo rank on a donor with no removal is approximately uniform.
- The planted-additive must-fire is re-run at the SAME noise level as the real data,
  because additivity broken by measurement noise is the most likely false positive.

## Statistical design, fixed now

The unanimity rule this project used before (a bar of +0.020 across 5 of 5 seeds)
has a power ceiling of 0.5^5 = 3.1% at any noise level when the true effect sits at
the bar, and gets weaker with every seed added. It is not used here. Every result
is a point estimate with a confidence interval; every family of tests is corrected
by Benjamini-Hochberg at q = 0.05; every effect is ranked among its placebos.
