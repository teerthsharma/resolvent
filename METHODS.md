# METHODS — the anytime-valid decision statistic

Round 7, Foreman, iteration 0. This file derives the e-process that §1.4 of the
contract puts in place of a deadline, and states the conditions under which the
derivation holds. It is normal English by RULE 4's artifact exemption. Every
constant here is fixed from the moment the first seed lands.

Companion file: `scale/sprt.py`, which carries the Bernoulli sibling of this
test and its own pre-registration. The e-process below is the continuous-outcome
instrument; the SPRT is retained for kills whose outcome is a coin.

---

## 1. What is being bought

An **e-process** for a null `H0` is a nonnegative supermartingale `E_t` adapted
to the data filtration `F_0 ⊆ F_1 ⊆ ...` with `E[E_0] ≤ 1`. **Ville's
inequality** states that under `H0`,

    P( sup_t E_t ≥ 1/alpha ) ≤ alpha

for every `alpha` in `(0,1)`, where the supremum is over **all** `t`, not over a
sample size fixed in advance.

**An attribution precision, because the usual citation is loose.** Ville's own
Théorème 1 (1939 thesis, printed p. 84) is stated for a **martingale** — his
condition (14) is an *equality* in conditional expectation together with
`s_0 = 1` — and reads `Pr{ B.sup_(n) s_n ≥ λ } ≤ 1/λ` for `λ > 1`. The
**supermartingale** form with `E[M_0]` on the right, which is the form this file
uses, is the modern statement: Wang & Ramdas, arXiv:2304.01163v3 §2.1 eq. (3),
`P[ sup_{0≤n<∞} M_n ≥ C ] ≤ C^{-1} E[M_0]` for `M` a nonnegative supermartingale;
equivalently Durrett, *Probability: Theory and Examples*, Version 5, Exercise
4.8.2, `P(sup X_n > λ) ≤ EX_0/λ`. The two differ in exactly the way that matters
here — supermartingale rather than martingale, and an expectation rather than a
fixed initial value — so the round cites the modern form, and the proof below is
self-contained in any case.

The proof is one line and worth having in front of the reader, because
everything this file is for lives in it. Let `a = 1/alpha` and let
`tau = inf{ t : E_t ≥ a }`. For every fixed `n`, `tau ∧ n` is a bounded stopping
time, so optional stopping for supermartingales gives `E[E_{tau∧n}] ≤ E[E_0] ≤ 1`.
On the event `{tau ≤ n}` we have `E_{tau∧n} = E_tau ≥ a`, and `E_{tau∧n} ≥ 0`
always, so `a · P(tau ≤ n) ≤ E[E_{tau∧n}] ≤ 1`, i.e. `P(tau ≤ n) ≤ 1/a = alpha`.
Letting `n -> ∞` and using monotone convergence on the increasing events
`{tau ≤ n}` gives `P(tau < ∞) ≤ alpha`, which is the claim.

**The consequence the contract wants.** Nothing in that argument mentions a
sample size, a deadline, or a stopping rule chosen by the experimenter. The
bound is over the running maximum, so an analyst who looks at `E_t` after every
seed, stops when it pleases them, resumes after an interruption, and reads the
number at an audit has still not spent more than `alpha` of type-I error. **A
slipped schedule therefore stops invalidating a verdict.** It changes what the
number *is* — a run stopped early has less evidence — but not whether the number
may be believed. Three rounds of this project were breached by deadlines. This
is the mathematics that makes the fourth one a smaller event: the audit reads
the current `E_t`, and if it has not crossed, the round prints
`undecided at evidence E_t = [value]` and says so, rather than slipping.

At `alpha = 0.05` the threshold is `1/alpha = 20` exactly. That is the
`E_t ≥ 20` in the contract's decision rule, and it is where the number 20 comes
from.

---

## 2. The construction, and the martingale property proven

**Setup.** `d_i` is the paired per-seed difference already shipped and tested in
`scale/m3_synthetic_settled.py::contrast`, whose docstring fixes the sign:

    delta_s = NRMSE_twin(s) - NRMSE_settled(s)

so a **positive** `d_i` means the settled arm has the **lower** error, i.e. it
wins. `d_i` is `F_i`-measurable. `B > 0` is a bound on `|d_i|` fixed **before
the first seed** (§4.1 is about how, and it is the part that can go wrong). The
null for the settled direction is

    H0 :  E[ d_i | F_{i-1} ] ≤ 0   for every i

— "settled does not beat twin". Note this is a condition on **conditional**
means only. It does not require the `d_i` to be independent, identically
distributed, Gaussian, or symmetric, and it is *composite*: it is a whole family
of distributions, and the guarantee below holds uniformly over all of them. The
twin direction is the mirror process built on `-d_i`, run in parallel.

**Single-lambda process.** For `lam` a constant in `[0, 1/2]`,

    E_t^lam  =  Prod_{i<=t} ( 1 + lam * d_i / B ),    E_0^lam = 1  (empty product)

*Nonnegativity.* `|d_i| ≤ B` gives `d_i/B` in `[-1,1]`, and `lam ≤ 1/2` gives
`lam*d_i/B` in `[-1/2, 1/2]`, so every factor lies in `[1/2, 3/2]`. The factors
are not merely nonnegative, they are bounded **away from zero**, which is why the
cap is `1/2` and not the `lam < 1` that bare nonnegativity would allow: a process
that can hit exactly zero is dead for the rest of the run and can never recover,
and a factor near zero makes the log-scale arithmetic that the implementation
will use lose precision at exactly the seed that matters most.

*Supermartingale.* `E_{t-1}^lam` is `F_{t-1}`-measurable and nonnegative, and
`lam` and `B` are constants, so

    E[ E_t^lam | F_{t-1} ]  =  E_{t-1}^lam * ( 1 + lam * E[ d_t | F_{t-1} ] / B )
                            <=  E_{t-1}^lam

because `lam >= 0`, `B > 0`, and `E[d_t | F_{t-1}] ≤ 0` under `H0`. With
`E[E_0^lam] = 1` this is an e-process. **The direction of the inequality is
carried entirely by `lam >= 0`**, which is why the grid is one-sided: a negative
`lam` would test the other tail and reverses the sign, and that is what the
mirror process is for, not a wider grid.

*Predictable lambda.* If `lam_i` is `F_{i-1}`-measurable rather than constant —
chosen from the data seen so far — the identical computation goes through,
because `lam_t` factors out of the conditional expectation exactly as a constant
does: `E[1 + lam_t*d_t/B | F_{t-1}] = 1 + lam_t*E[d_t|F_{t-1}]/B ≤ 1`.
Predictability is the whole requirement. A `lam_t` that peeks at `d_t` is not
predictable and the step fails.

**The grid mixture — the step that can be wrong.** Fix a grid
`lam^(1), ..., lam^(K)` in `[0, 1/2]` and weights `w_1, ..., w_K` with
`w_k >= 0`, `Sum_k w_k = 1`, **all chosen before the first seed**. Define

    Ebar_t  =  Sum_k w_k * E_t^{lam^(k)}

Nonnegativity is immediate: a nonnegative combination of nonnegative terms.
`E[Ebar_0] = Sum_k w_k * 1 = 1`. For the supermartingale property, conditional
expectation is linear and the sum is finite, so no interchange theorem is
needed at all:

    E[ Ebar_t | F_{t-1} ]  =  Sum_k w_k * E[ E_t^{lam^(k)} | F_{t-1} ]
                          <=  Sum_k w_k * E_{t-1}^{lam^(k)}
                           =  Ebar_{t-1}

using the single-lambda result term by term, and `w_k >= 0` to preserve each
inequality under the sum. **The property holds.** For a continuous mixture over
a prior on `[0,1/2]` the same argument runs with Tonelli in place of the finite
sum, which applies because the integrand is nonnegative; the grid is used here
because it is what gets implemented and it needs no measurability side
conditions.

**Why the mixture is not a separate object.** The two facts above look like two
theorems and are one. Since every factor is at least `1/2`, `Ebar_{t-1} > 0`
always, so the posterior weights
`w_k^{(t-1)} = w_k * E_{t-1}^{lam^(k)} / Ebar_{t-1}` are well defined,
`F_{t-1}`-measurable, nonnegative, and sum to 1. Then

    Ebar_t / Ebar_{t-1}  =  Sum_k w_k^{(t-1)} * ( 1 + lam^(k) * d_t / B )
                         =  1 + lambar_{t-1} * d_t / B

with `lambar_{t-1} = Sum_k w_k^{(t-1)} * lam^(k)`, which is a convex combination
of grid points, hence lies in `[0,1/2]`, and is `F_{t-1}`-measurable, hence
**predictable**. Telescoping,

    Ebar_t  =  Prod_{i<=t} ( 1 + lambar_{i-1} * d_i / B )

**The grid mixture is exactly a single predictable-lambda process whose `lam` is
the posterior mean of `lam` under the prior `w`.** It is a special case of the
predictable-lambda result, not an extension of it. This is worth stating for two
reasons beyond economy. First, it says what the mixture is *for*: it is an
automatic, data-driven choice of `lam` that needs no tuning and no held-out
split, and it converges on whichever grid point the data favour. Second, it says
precisely what would break it, in §4.3: the derivation used `w_k` **constant** in
the step `Ebar_t = Sum_k w_k * E_{t-1}^{lam^(k)} * (1 + lam^(k)*d_t/B)`.
Re-weighting the outer mixture by hand at each step produces a `lambar` that is
not the posterior mean, the telescoping identity fails, and there is no theorem
left.

**Backstop.** A nonnegative supermartingale converges almost surely to a finite
limit. Durrett, Version 5, **Theorem 4.2.12**, verbatim: *"If X_n ≥ 0 is a
supermartingale then as n → ∞, X_n → X a.s. and EX ≤ EX_0."* The
Robbins–Siegmund almost-supermartingale theorem is the general form — in the
Neri & Powell restatement (arXiv:2410.15986v2, Theorem 1.1), the hypothesis is
`E[X_{n+1} | F_n] ≤ (1 + A_n) X_n − B_n + C_n` with `Sum A_i < ∞` and
`Sum C_i < ∞` a.s., and our case is `A_n = B_n = C_n = 0`, where it reduces to
Durrett 4.2.12. Under `H0`, `Ebar_t` therefore does not wander: it settles. A
reading of `undecided at E_t = [value]` is consequently a statement about a
converging quantity rather than a snapshot of something diverging, which is what
makes the printed value interpretable at an audit rather than merely honest.

---

## 3. The decision rule

Immutable from the first seed.

| condition | verdict |
|---|---|
| `Ebar_t >= 20` on the settled process | settled beats twin at `alpha = 0.05`, anytime-valid, at that `t` |
| `Ebar_t >= 20` on the mirror process | twin beats settled at `alpha = 0.05`, anytime-valid — **K-1 fires** |
| neither by phase end | `undecided at evidence E_t = [value]`, printed, both processes |

Running both directions and rejecting when **either** crosses `20` spends up to
`alpha = 0.05` on each, so the two-sided family-wise rate is bounded by `0.10`.
If the round wants `0.05` two-sided it must use `1/0.025 = 40` on each process.
**This is a choice, and it must be made before the first seed, not after.** It is
pinned by value in the must-fire test.

---

## 4. The four conditions, and what each one breaks

The derivation in §2 is short and each line of it is a hostage. These are the
places where a correct implementation and an incorrect one differ by nothing
visible in the output.

### 4.1 `B` must bound `|d_i|` a priori — and NRMSE does not supply one

This is the live problem, and it is not a formality.

`d_i = NRMSE_twin(s) - NRMSE_settled(s)`. NRMSE is `RMSE / std(y)`, which is
**not bounded above**: a diverging arm has arbitrarily large NRMSE. The repo's
own record shows the range is wide and routinely exceeds 1: `2.1166` at
`n_train=128` and `1.3165` at `512` (`CHECKLIST.md:333`), a seed-mean of
`1.007076` at `2048` (`CHECKLIST.md:868`), and `5.8198` for attention on an OOD
cell (`ceq/nash.py:4`). **There is no a-priori `B` available from the NRMSE
scale itself.**

Taking `B = 1.0` from the absolute bar is **wrong**. The bar at `NRMSE = 1.0` is
a *credit* rule — the outcome table's last row refuses credit to any cell whose
seed-mean is at or above `1.0` — and it is applied to the **seed-mean at the
end**. It does not bound any individual arm's NRMSE, so it does not bound
`|d_i|`.

Two consequences if `B` is set too small and some `|d_i| > B` occurs:
`lam_i*d_i/B` can fall below `-1`, a factor goes **negative**, and the process is
no longer nonnegative. Ville's inequality then does not apply, and neither does
the convergence backstop. Every anytime-valid claim in the round is void, and
nothing in the printed output looks unusual — the number keeps updating.

**Clipping `d_i` after the fact does not repair this.** Clipping `d` to
`[-B, B]` raises the value of every `d` below `-B`, which raises the conditional
mean, and `H0` is exactly a statement that the conditional mean is not positive.
A repair that can push the mean up is a repair that inflates the type-I error it
was supposed to protect.

**The repair that works: clip the OUTCOME, not the difference, and do it in the
definition.** Pre-register a ceiling `C` and define the outcome of each arm as
`min(NRMSE, C)`, then difference:

    d_i  =  min( NRMSE_twin(s), C )  -  min( NRMSE_settled(s), C )

so `|d_i| ≤ C` by construction and `B = C` is a-priori by fiat rather than by
hope. This is legitimate where post-hoc clipping is not, because it defines a
**new bounded outcome variable** before any data exist, and `H0` is then stated
about that variable: *"settled does not beat twin on the `C`-clipped error
scale"*. `min(·, C)` is monotone nondecreasing, so the clip never flips the sign
of a per-seed comparison; it only shrinks magnitudes. And because the round
already refuses credit to any cell at or above `1.0`, a ceiling of `C = 2.0`
leaves the clipped scale and the claimable scale in agreement everywhere the
claim is permitted to live: the clip discards only distinctions between degrees
of failure that the outcome table discards anyway.

**`C = 2.0` is the recommended value.** It must be fixed in code, asserted, and
pinned by value in the must-fire test before the first fresh seed. The assertion
`|d_i| ≤ B` must be a hard failure that **voids the process**, not a clamp — a
clamp is the post-hoc clipping this section just rejected, wearing a different
name.

A bounded-by-construction alternative exists and is cheaper: take the outcome to
be `sign(d_i)` in `{-1, 0, +1}` with `B = 1`. It needs no ceiling and no
pre-registration argument. It also discards effect magnitude, which is most of
the information at five seeds, and it is the Bernoulli case that `scale/sprt.py`
already covers. It is named here as the fallback if `C` cannot be agreed, not as
the default.

### 4.2 `lam_i` must be predictable

`lam_i` may depend on `d_1, ..., d_{i-1}` and must not depend on `d_i`. Under the
grid mixture this is automatic — §2 shows the effective `lambar_{i-1}` is a
function of the posterior weights at time `i-1` — which is a reason to prefer the
mixture over a hand-tuned schedule. A `lam` fitted once at the end of the run to
the whole sequence is not predictable and produces a number with no guarantee
attached.

### 4.3 The mixture weights `w_k` must be constant

Fixed before the first seed, never re-normalised toward whichever `lam` is
currently winning. §2 shows why: the derivation requires `w_k` to pass through
the conditional expectation as a constant, and the posterior-mean identity that
makes the mixture legitimate is what hand re-weighting destroys. The
**posterior** weights `w_k^{(t)}` move, and they are supposed to — that is the
mixture adapting. The **prior** weights `w_k` do not. Reporting the posterior
weights per step is a good diagnostic; feeding them back in as new priors is the
bug.

### 4.4 No outcome-dependent seed exclusion

Every seed that runs must contribute a factor. Dropping a seed **because of how
it came out** — a failed gate, an outlier, a crashed run whose crash correlates
with the arm — makes the inclusion indicator `F_i`-measurable but not
`F_{i-1}`-measurable, and then `E[ I_i*d_i | F_{i-1} ] ≤ 0` does not follow from
`E[ d_i | F_{i-1} ] ≤ 0`. The supermartingale property is lost and the printed
`E_t` is not an e-value.

The contract's existing gate does **not** trip this, and it is worth recording
why rather than trusting it: the `NRMSE ≥ 1.0` rule is applied to the
**seed-mean of a whole cell at the end**, as a rule about what the cell is
credited with, so it filters *cells*, not the `d_i` within a cell. Applying the
same threshold **per seed**, to decide which `d_i` enter the product, would break
the process. The two readings of one number differ by everything.

If a seed genuinely cannot produce a value — a crash unrelated to the arm — the
honest handling is to record it, contribute `d_i = 0` (a factor of exactly 1,
which is no evidence in either direction), and report the count of such seeds
beside the final `E_t`. That is conservative and it is auditable.

---

## 5. What this does not say

Collected here and not repeated above.

1. **An e-value is not a p-value.** `E_t = 20` does not mean `p = 0.05`. It
   means the type-I error of the rule "reject when `E_t` first reaches 20" is at
   most `0.05` at any stopping time. The two coincide in no useful sense and the
   round must not print one as the other.
2. **Ville bounds type-I error only.** Nothing here gives power. A true effect
   smaller than the harness's own resolution floor — recorded at `~0.05 NRMSE at
   5 seeds`, with paired sd `0.056889` against unpaired `0.057089`
   (`CHECKLIST.md:820`) — will leave `E_t` near 1 indefinitely. `undecided`
   means the evidence did not accumulate; it is not evidence of no effect, and
   the e-value's distance from 1 is the only quantitative statement available.
3. **`H0` is about conditional means of the clipped outcome**, not about
   distributions, medians, or the unclipped NRMSE. A verdict transfers to the
   unclipped scale only by the sign-preservation argument in §4.1, which is
   per-seed and does not by itself transfer statements about means.
4. **The guarantee is per-process.** Two directions run simultaneously spend
   `alpha` each; see §3. Adding a third comparison later without re-stating the
   budget spends error that no line of this file accounts for.
5. **Nothing here is about the training run.** `d_i` arrives from
   `scale/paired_arm.py::train_and_predict` unchanged. If that path moves, the
   `d_i` are from a different experiment and the accumulated `E_t` is not
   continuous across the change — a moved training path starts a **new**
   process, it does not resume the old one.

---

## 6. Sources

Citations carry `[U]` for a fetched source with a locator and `[V]` for a fact
verified inside this repo. Anything not yet at `[U]` is marked as such and is
Foreman's to close before the reading is published.

| object | status |
|---|---|
| Ville's inequality, **martingale** form | `[U]` Ville, *Étude critique de la notion de collectif*, Gauthier-Villars 1939, **Théorème 1, printed p. 84**, full text at numdam.org. Hypothesis as printed: condition (14) with `s_0 = 1` and conditional-expectation **equality**; `λ > 1` |
| Ville's inequality, **supermartingale** form (the one used here) | `[U]` Wang & Ramdas, arXiv:2304.01163v3, §2.1 p. 4 eq. (3); and Durrett Version 5, Exercise 4.8.2 |
| Doob supermartingale convergence | `[U]` Durrett, *Probability: Theory and Examples*, **Version 5 (11 Jan 2019), Theorem 4.2.12** — *"If X_n ≥ 0 is a supermartingale then as n → ∞, X_n → X a.s. and EX ≤ EX_0."* Specializes Theorem 4.2.11. **Durrett 4th ed. Theorem 5.2.9 was NOT verified** — no 4th-edition text retrieved |
| Robbins–Siegmund almost-supermartingale theorem | `[U]` **bibliographic only — ORIGINAL NOT REACHED.** Robbins & Siegmund, "A convergence theorem for non negative almost supermartingales and some applications", in *Optimizing Methods in Statistics* (ed. J. S. Rustagi), Academic Press, New York, 1971, **pp. 233–257**; ISBN 012604550X; DOI `10.1016/b978-0-12-604550-5.50015-8`; reprinted in *Herbert Robbins Selected Papers*, Springer 1985, pp. 111–135. ScienceDirect 403, Semantic Scholar `openAccessPdf: CLOSED`, Internet Archive scan access-restricted. **Statement used is a restatement**, Neri & Powell arXiv:2410.15986v2 Theorem 1.1, labelled as such |
| Safe-anytime-valid framing | `[U]` Ramdas, Grünwald, Vovk & **Shafer**, "Game-Theoretic Statistics and Safe Anytime-Valid Inference", *Statistical Science* 38(4), 2023, DOI `10.1214/23-sts894`; arXiv:2210.01948v2 §2.2, §2.5. **Author is Shafer, not Wang** — an earlier draft of this file had that wrong |
| grid-mixture supermartingale property | **proven above, §2** — self-contained, no citation required |
| sign convention `delta_s = NRMSE_twin - NRMSE_settled` | `[V]` `scale/m3_synthetic_settled.py::contrast` docstring |
| NRMSE range exceeding 1.0 | `[V]` `CHECKLIST.md:333`, `CHECKLIST.md:868`, `ceq/nash.py:4` |
| resolution floor `~0.05 NRMSE`, paired sd `0.056889` | `[V]` `CHECKLIST.md:820` |
| absolute bar `NRMSE = 1.0` as a credit rule on the seed-mean | `[V]` `M3_QUINTUPLE_PREREGISTERED_READING.md` §5, last row |

The proof in §2 is the load-bearing item and it depends on no citation at all.
The fetches supply the standard names and the backstop, not the argument — which
is the reason the one source that could not be reached (Robbins & Siegmund 1971,
paywalled and access-restricted in every located copy) costs this file nothing.
Its role is played, for our special case, by Durrett Theorem 4.2.12, which was
reached.

Further prior-art corrections affecting the Hankel program rather than the
statistics live in `PRIOR_ART.md`.
