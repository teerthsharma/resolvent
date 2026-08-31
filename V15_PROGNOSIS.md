# R11 PROGNOSIS — CEQ v15 / v15.1 / v15.2

Written at it.27 with five nodes still running, so that a report exists whatever
they return. Amendable; every line below is what the tables permit **as of this
commit**, and anything a late node changes is marked when it is amended.

---

## 1. THE ONE CLAIM SENTENCE THE TABLES PERMIT

The contract asks for exactly one. This is it, and it is a **composition** claim,
not a capability claim:

> **A single causal softmax head, carrying a prefix-scan in its key logit and a
> value-zero sink, is bitwise standard attention at its identity setting and
> reproduces an exact path-product label on the same head — to `6.66e-16`, with
> four planted mutilations of the construction failing at `O(1)`.**

What that sentence does **not** say, and must not be extended to say:

- It does not claim a capability advantage over softmax. **None has been
  measured.** `C-CAP` stands at 0 crossings in 9 cells, `ĥ ≤ 0.389`.
- It does not claim parity at the gate's own identity point. The bind is at
  *both auxiliary heads zero*, whose content is: **the modification enters only
  through the key logit, additively, and vanishes at zero.**
- It does not claim novelty of the components. SSD/Mamba-2, GLA, RetNet,
  negative-eigenvalue SSMs, MWU/Hedge, Mori–Zwanzig-in-ML, ARFIMA/fractional
  nets, generating-partition estimators and Hamiltonian nets all reached
  `[V-eq]`, and VORT (arXiv:2605.08966, verified) occupies the fractional head
  outright.
- It does not survive if `ceq/lm.py`'s `Attention("softmax_x")` is not standard
  causal attention. That is the reference the bitwise claim rests on, and its own
  header asserts the equivalence at the shipped constants.

---

## 2. SCOREBOARD

Against the contract's 38-point ceiling, from the RUL-7 baseline of `0 of 39`.

| event | pts | status |
|---|---|---|
| Lean train-gate green | **+2** | **EARNED** |
| BED-K registered | **+2** | **EARNED** |
| R1 floor crossing | +12 | running at it.25 |
| R2 `ĥ > 1` | +4 | **the same event as R1's crossing** — the scoreboard double-pays |
| R4 two-sided parity | +8 | **BLOCKED** — R3 unscoreable (M-20) |
| R5 CK with committor | +4 | BED-1 running |
| R6 intervention halved | +2 | **unscoreable as written** (Venus, pre-data) |
| conservation + Lyapunov columns | +1 | no tables cut |
| package | +3 | not attempted |

**4 of 38 confirmed.** `+14` of the ceiling was established unreachable **before
any cell ran**, by reading the contract against arithmetic and against the code.

### The claim ladder, which is the scoreboard that matters

| bar | at it.0 | at it.27 |
|---|---|---|
| **C-PAR** | TOST unreachable at N=8; identity bind assumed | **instrument repaired** — bitwise, on an amended operator, with the label bind on the same head |
| **C-CAP** | 0 of 9 cells, `ĥ ≤ 0.389` | unchanged; R1 running |
| **C-TS** | NOT BUILT | BED-1 running |

---

## 3. WHAT THE ROUND ESTABLISHED THAT NO CELL COULD HAVE

**Nine contract statements checked, six wrong, all six over-crediting the
project.** Errors distributed by chance do not share a sign. The direction is the
finding, and the design rule is in the R11 round entry.

**The parity clause was refuted and then repaired.** Refuted three independent
ways — Lean row sums (`i+1 ≠ 1`), prior art (`g ≡ 0` lands on *linear* attention,
Dao & Gu §2.4 has no softmax), and plain torch outside the Lean kernel. Repaired
under L-AMEND by a telescope: `(1−a_j)e^{−C_j} = e^{−C_j} − e^{−C_{j−1}}`, whose
boundary term **is** the BOS sink.

**Two hoped-for headlines refuted, one of them the coordinator's own.**
"Softmax's normalizer is the obstruction to path products" is **false** — it
holds only with the drives as values; a position-local rescale dissolves it
uniquely at `γ_j = 1/(1−a_j)`. **The normalizer is a change of units, not an
obstruction.**

**Eight failure mechanisms filed**, each with an instance, a number, and a check:
`M-17`, `M-18` (carrying a correction against its own author), `M-19`, `M-20`,
`P-10`, `P-11`, `V-23`, `V-24`.

**The campaign's central negative is re-diagnosed.** The nine non-crossings were
never a gate-learning failure — `CH_DRIVE` **is** `a`, disclosed by design. The
binding constraint is multiplicative composition along paths, bounded by
measurement: order-0 features reach `R² = 0.01–0.02` against the scored label.

---

## 4. NEXT-X LIST

**X₃₆ — the approximation bound.** Lean #12 is an exact-identity result. The
power-law bed is not scan-blind (`R² = 0.604` at `H = 0.75`) and that does not
contradict it. **Every "X cannot represent Y" claim in this campaign needs an
approximation bound before it can be read as "X cannot fit Y."** This is the
largest unclosed gap in the round's own logic.

**X₃₇ — reconcile R3's two registrations.** PART III and PART IV predict inverse
outcomes for the same bed (M-20). The author picks one; the cell cannot run
before that, because whichever number arrives confirms whichever half is quoted.

**X₃₈ — the device predecessor.** `--device cuda` aborts by design because
`calibrate_bar` is structurally CPU-only, and a threshold carried across a device
boundary is V-22. **637 CPU-hours against 47 GPU-hours turns on this one
re-certification.**

**X₃₉ — the signed variant.** Handed back open by the fork node: it needs a
target-side `χ(P_i)` multiplier, which fails V-24 unless applied by machinery
already inside the transformer class.

**X₄₀ — the scoreboard's double-count and its missing baseline.** `NRMSE < floor₁`
and `ĥ > 1` are one event; `+12` and `+4` pay twice. And "the current 22%" has no
producer anywhere in the tree.

**X₄₁ — re-measure X₃₅'s flatness against a trained arm.** The instrument is
certified at the *oracle's* scope. A trained arm carries approximation error the
oracle does not, and *the detector cannot outrun the model it subtracts.*

---

## 4b. LATE RESULTS — landed after §1–§4 were written

**X₃₅ is occupied, in closed form, since 1993.** The composition is Basseville &
Nikiforov §7.2.4 (orig. Willsky & Jones 1976), equation for equation: `(7.2.110)`
is X₃₅b's re-propagation, `(7.2.123)` the onset, `(7.2.126)` the magnitude,
`§7.2.5` the localization, `§7.2.6 Def 7.2.1` the detectability condition. Those
1993 equations **discharge both X₃₅a must-fires on the first attempt**. The
learned-model form is arXiv:2604.25655 (28 Apr 2026), whose **Theorem 3.1 is
X₃₅a's must-fire as a theorem**, with the delta's KILL appearing as its
local-identifiability eigenvalue. What survives is a **transfer gap**: the
residual belongs to a learned model whose `Γ` is not enumerable.

**LATENT LOCALIZATION ACCURACY has a ceiling set by theory.** A 1-latent and a
3-latent generator give observed covariances agreeing to `5.6e-17` — identical CI
oracles, so FCI/RFCI/FCI+/LPCMCI must return the same PAG, and completeness
guarantees it never resolves at any sample size. The column is admissible only as
*recovery of a planted node under a known generator*.

**The memoryless comparator is vindicated with a better argument than the delta
had.** Shewhart ARL₀ measured `80.11 ± 0.40` against a predicted `80.52`, and
under AR(1) it rises to `1.71×` — **conservative**. CUSUM moves the opposite way,
to `0.46×`. The two charts sit a factor `3.6` apart on the same data.

**The struck KK probe is rebuilt and fires**: `0.9899` against a null of
`1.37e-15`. Its diagnosis eliminated one of four candidate causes by measurement
(a `1/√M` normalizer cannot print an exact zero) and **refused to guess between
the two surviving families**, specifying the one-run experiment that separates
them. The CRB does not exist for a discrete onset; a **Ziv–Zakai** bound ships
instead, agreeing with direct evaluation to `0.00e+00`.

**BED-1 is built — C-TS has a bed.** 28 tests pass. The strike reproduces to
every digit and **the labels disagree exactly where theory says they must**: at
`1.1 T*` the barrier label reads `lo` while the committor reads `hi`. A **third,
independent route** confirms it from the dynamics — simulated walkers cross the
five-fold *higher* barrier more often, empirical `lo:hi = 0.86679 < 1`, agreeing
with the committor ratio to 0.34%.

**And BED-1 reports a negative it did not have to volunteer.** The `q = 1/2`
guard partition's **own** Pesin deficit is `0.009654`, **11.9% of `h`** — not near
zero, so **the contract's guards are not a generating partition.** That is R5's
distribution-predict verdict, arrived at against the contract's S-G design, and
printed rather than buried. It also *observed* Bollt's non-monotonicity on its own
bed rather than citing it: two **equally** misplaced cuts differ from each other
by `7.0e-4`, more than one of them differs from the correct cut (`2.6e-5`).

Its methodological line is the best in the round: **"13/13 green on the first
implementation run isn't evidence"** — so it ran five mutants and killed all five.

---

## 5. THE HONEST SUMMARY

R11 spent twenty-seven iterations making the next round's measurements mean
something and took one. It repaired the contract's central clause, refuted two of
its own hoped-for headlines, filed eight mechanisms, and established that two of
the contract's scheduled cells cannot be scored as written — none of which is on
the scoreboard.

**Whether that was worth it turns on R1**, which is running as this is written.
If it crosses, the campaign has its first capability point on an arm that is
provably equal to softmax. If it does not, the round meets the contract's stated
Floor with a sharper diagnosis than the campaign has ever had: not *"unlearnable
at budget"* but a measured order boundary showing exactly what the arm's features
reach and what they do not.
