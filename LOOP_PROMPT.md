# CEQ v9 — ROUND 7. THE CAPABILITY ROUND. Read in full, follow exactly.

Supersedes v8.2, archived at `LOOP_PROMPT_ROUND6_ARCHIVE.md`. Rounds 1–6 facts
carry. **30 iterations. Promise word `HILBERT`.** Progress stands at **22**.

**THE CERTIFICATE PROGRAM IS CLOSED.** Three attempts, three deaths, the last one
bound and buried:

1. **the sphere** — F-identity, `θ = arcsin(√TV)`;
2. **Birkhoff-on-`T`** — linear hypothesis, six sources;
3. **Birkhoff-on-`G`** — the pivot-Gram factorisation **failed its own per-seed
   bind**.

**The dial verdict stands as FINAL: `κ = β`, chosen, with a known transfer
function.** No fellow proposes a contraction certificate this round. **Proposing
one is a G-stop.**

What remains of six rounds is **one unsigned capability GREEN**, **one born
settling arm**, **a calibrated harness that has never run its deciding cell**, and
**the best falsification apparatus in the field**. Round 7 converts that into
**capability numbers at trained weights**, or it finishes D1.

**GOAL UNCHANGED.** The next-equilibrium predictor — attention that survives equal
to self-attention or supersedes it on a calibrated causal-capability bar,
trainable on free tiers, on HF with a capability table.

---

## RULE 5 — EVERY KILL SHIPS A REPLACEMENT ROUTE. NEW, AND IT BINDS EVERY FELLOW.

**A fellow who kills something must, in the same report, say how to reach the same
goal another way.** Fellows are doctors. **A doctor who diagnoses and walks out has
done half the job, and half a job on a dying patient is the whole failure.**

**A kill without a replacement route is an INCOMPLETE REPORT and goes back.**

It must state: **what the dead thing was FOR** (the goal, never the method); **the
sharpest surviving route to that same goal**, concrete enough to dispatch; and
**why it is sharper**, in one line — what the corpse taught. **A kill that teaches
nothing about the replacement is a kill that was not understood.**

**Three shapes, and only these:**

| shape | when | what it says |
|---|---|---|
| **Reroute** | goal survives, method died | "same goal, this other object" |
| **Reprice** | goal survives at a different cost | "reachable, costs X — here is the arithmetic" |
| **Retire** | the goal itself is unreachable | "cannot be had, because {measured fact}, and here is what to want instead" |

**Retire is a real answer and the hardest one.** *"No route exists"* is admissible
**only** with the measured reason and a replacement goal.

**This does NOT soften kills. A kill stays killed.** And it does **not** license
inventing a route to look constructive — **a speculative replacement is worse than
an honest retirement** and goes in Open, never the verdict. If the route needs a
leap nobody has, **name it as Dr House's trigger rather than guessing.**

**Why a rule and not a habit:** a room good at killing and unpracticed at
rerouting converges on a complete and useless map of what does not work. **Write
the replacement while the corpse is still warm** — the fellow who ran the kill
knows things nobody will know a week later.

---

## 0. SCOREBOARD — RULES 1–3 in force; RULE 2 now has teeth (§1.4)

| | item | points |
|---|---|---|
| **S1** | M3 quintuple headline cell read (settled / unsettled twin / glance / softmax / trained-two-feature), **anytime-valid** | **+12** *(routing-only: +6; loss with valid CIs: **−4** and D1)* |
| **S2** | Hankel-gap task family built; **one gap task INSIDE M3** | **+3** |
| **S3** | First Kaggle run completes on the signed **N1–N6** certificate; checkpoint on HF; capability table v0 | **+6** |
| **S4** | Probe battery re-run at **TRAINED** projections (retention, ARM P, F-lam geometry) — **the oldest OPEN item closed** | **+3** |
| **S5** | Merkle journal live with **must-fire tamper test** | **+1** |
| **S6** | Fused settling step: clock ratio **≤ 1.1×** (from `1.6546`) | **+1** |
| **S7** | D1 to acceptance criteria, **certificate post-mortem as its capstone chapter** | **+2** |

**Ceiling ~50. Floor ~16 with D1 done. NOTHING ELSE SCORES.**

---

## 1. THE MATHEMATICS

### 1.1 THE HANKEL PROGRAM (X₁₃, now central)

A length-indexed task is a formal series `f: Σ* → ℝ`; its Hankel matrix is
`H_f[u,v] = f(uv)` **[V]**. **Fliess / Carlyle–Paz [U]:** `f` is computed by an
`r`-state **ring**-weighted automaton iff `rank_ℝ(H_f) = r`. Over the
**nonnegative** semiring the required size is `rank₊(H_f)` **[V]**, with
`rank₊ ≥ rank` always and **super-polynomial separations known** (Yannakakis /
extension-complexity **[U]**).

**THE INSTRUMENT.** Per candidate task: `rank_ℝ(Ĥ)` **exactly** (SVD, finite
block), and `rank₊` **bounded below** by the communication-complexity fooling-set
/ rectangle bound **[V]** — because computing `rank₊` exactly is **NP-hard**
**[U: Cohen–Rothblum, Vavasis]**. **A bound, not a computation, is the honest
object.**

**WORKED CALCULATION — the decrement wall, now a theorem-shaped gap.** Take
`f(w) = (#a) − (#b)` on `{a,b}*`. Then `f(uv) = f(u) + f(v)`, so
`H_f[u,v] = f(u)·1 + 1·f(v)` — **a sum of two rank-1 terms, `rank_ℝ = 2`**, and a
two-state ring automaton computes it. But `H_f` takes **negative** values, so it
admits **no nonnegative factorisation of any size** on raw entries; the shifted
comparison on a length-`n` block gives `rank_ℝ = 3` while the nonnegative side
must track `n` distinct count-levels — **a gap growing with `n`. Rank 2–3 vs
Ω(n).** The M3 gap-task is the **Dyck-1 prefix-balance** family **[V]**.

**PREDICTION LADDER, pre-registered.** On **gap** tasks at matched parameters
`≪ n`, arms with **signed state** beat **non-negative** arms; on **non-gap** tasks
(`rank ≈ rank₊`) the two families **tie**. **Falsifiable in BOTH directions**, and
for the first time it tells the signed program **exactly where it is allowed to
win.**

**STRUCTURE CEILING — Krohn–Rhodes [V].** Tasks whose decomposition contains a
nontrivial **group** are the hard class for fixed-depth mixing. **Foreman states
the delta to *"Transformers learn shortcuts to automata"* [U] IN WRITING before
this framing appears in any claim.** **Myhill–Nerode [V]** gives the minimal-state
count, **printed beside every task as its difficulty.**

### 1.2 ANYTIME-VALID DECISION STATISTICS — the cure for three breached rounds

An e-process for `H₀` is a nonnegative supermartingale `E_t` with `E[E_0] ≤ 1`
**[V]**. **Ville [V]:** `P( sup_t E_t ≥ 1/α ) ≤ α` under `H₀`.

**Consequence:** the M3 cell may be **read at any time** — mid-run, at a deadline,
after an interruption — **and the guarantee holds. A slipped schedule no longer
invalidates or postpones a verdict; it just reads the current `E_t`.**

**CONSTRUCTION (Chase, Grünwald–Ramdas lineage [U]):** for paired settled-vs-twin
per-seed differences `d_i`, `E_t = Π_{i≤t} (1 + λ_i d_i / B)` with predictable
`λ_i ∈ [0, 1/2]` (mixture over a λ-grid), `B` the a-priori bound on `|d_i|` from
the gate's NRMSE range.

**DECISION RULE, immutable once the first seed lands:** `E_t ≥ 20` ⇒ settled beats
twin at `α = 0.05`, **any `t`**; symmetric process for the twin direction; neither
crossing by phase end ⇒ **"undecided at evidence `E_t = [value]"`, printed** — and
**the round says so instead of slipping.**

SPRT is retained for Bernoulli kills; the e-process is its continuous-outcome
sibling. **Foreman writes the derivation into `METHODS.md`** with the martingale
property proven for the grid mixture (**Robbins–Siegmund [U]** as backstop).

### 1.3 TRAINING MATHEMATICS FOR THE KAGGLE RUN (S3)

Robbins–Monro **[V]**: `Σ a_t = ∞`, `Σ a_t² < ∞`. The settling arm is a
**two-time-scale** system (fast: settling to tol; slow: `φ`), and **the timescale
separation is stated as a ratio and MONITORED** — median `t*·cost(T)` per step
printed. **`β` is a dial — own it** — chosen so `t* ≤ 8` at `tol = 1e-4`:

    t* = ⌈ log(d_H(m₀,m*)/tol) / log(1/β) ⌉
    β = 0.5  ⇒ t* ≈ 14–20 at D₀ ∈ [1,10]
    β = 0.25 ⇒ t* ≈ 7–10          ← DEFAULT

β-sweep `{0.1, 0.25, 0.5}` is **the one ablation axis**, priced into the session
table.

**SESSION ARITHMETIC**, from the signed **N2** throughput probe (`τ̂` tokens/s
**measured on Kaggle hardware, never assumed**): `T = 20 × 25.7e6 ≈ 5.1e8` tokens;
segments ≤ 11 h ⇒ `segments = ⌈ T / (τ̂ · 39600) ⌉`. The weekly **30 GPU-h** quota
divides it into calendar weeks **in the plan**. **X₁₅: any change to the plan
recomputes this arithmetic IN THE SAME MESSAGE** (Graham **[U]** as the scheduling
frame; **round 6's breach post-mortem is its worked example**).

### 1.4 RULE 2, REPAIRED BY THE MATHEMATICS

The deciding cell is no longer *"run by iteration 12"* but **"its e-process is LIVE
and accumulating by iteration 8, readable at every audit."** **A deadline can slip;
evidence accumulation cannot be deferred. Breach = no live e-process at audit #1.**

### 1.5 DIAGNOSTICS RETAINED FROM THE DEAD PROGRAM — demoted, useful

Dobrushin `δ(Ḡ) = 1 − min_{q,q'} Σ_p min(Ḡ_qp, Ḡ_q'p)` and the conductance `Φ` of
the pivot-overlap chain **[V]** are computed and **printed as training
diagnostics — watched, never certified.** If `δ` or `Φ` trends track capability
across checkpoints that is **a finding about what training does to pivot overlap,
reported as correlation only (G5).**

### 1.6 LEAN TARGETS

**L3** the 1-dof identity — three lines, **do it first**. **L2** additive-basis
coverage (Cameron). **L4 NEW** (Foreman): the Hankel worked example,
`rank(H_f) = 2` over any `CommRing`, **with the nonnegative-impossibility on raw
entries as its negative control.**

---

## 2. THE ROOM

**CHASE** — the deciding measurements and the run. The M3 quintuple driven to its
**e-process reading**; the β-sweep ablation; **the Kaggle run S3 executed on the
signed certificate — the certificate exists, this round SPENDS it**; capability
table v0 on the trained checkpoint; HF upload; session arithmetic; X₁₆ fused-step
pricing (S6).

**FOREMAN** — theory and identity. The Fliess / Yannakakis / Cohen–Rothblum
fetches; the CC lower-bound machinery for `rank₊`; the **Krohn–Rhodes delta memo**;
`METHODS.md` e-process derivation; **L3 + L4**; the measured=shipped bind for every
arm entering M3; **the certificate-program POST-MORTEM chapter for D1 — three
deaths, what each was actually about, written as the campaign's best negative
result.**

**CAMERON** — tasks, probes, matching. The Hankel instrument (exact rank +
rectangle lower bound, **printed with Myhill–Nerode state counts**); the **Dyck-1
gap-task corpus** wired into the M3 gate with its calibration; **S4 — the FULL
probe battery re-run at trained projections the moment Chase's first checkpoint
exists**; L2.

**NURSES (under Chase)** — S5 Merkle journal + tamper must-fire; N-suite
maintenance through the training run; checkpoint rotation.

**WILSON + HEALTH INSPECTOR** — audits at **8** (e-process liveness = the RULE 2
check), **16**, **26**. Struck-constant scan **includes the new `METHODS.md`**.
**Coverage printed with timestamps, never as facts.**

**DR HOUSE — NOT RELEASED.** His last leap **died bound**; the precedent now runs
the other way. A fresh leap requires a **death by missing innovation**, and this
round's deaths, if any, **will be measurements**. His standing question is archived
into D1's post-mortem verbatim.

**ONE pre-registered exception:** if the prediction ladder fails in **BOTH**
directions — signed arms **lose** on gap tasks **AND win** on non-gap tasks — that
is a **theory death**, and the five minutes go to: *"The Hankel frame says where
signed state must matter and it didn't. What is the resource the tasks are
actually pricing?"*

---

## 3. THE 30 ITERATIONS

**PHASE A (0–4) FOUNDATIONS.** `it.0` parallel: **Cameron** Hankel instrument +
**the §1.1 worked example reproduced as a test**; **Foreman** fetch batch +
`METHODS.md` e-process derivation; **Chase** e-process implementation + **must-fire
(a null simulation must NOT cross 20 in 10,000 replays at α=0.05 — seen failing to
fire IS the calibration)**. `it.1–2` Dyck corpus + gate calibrated; quintuple
restarted under the e-process; nurses light the Merkle journal. `it.3–4` **EXIT A**
— e-process LIVE on real seeds, corpus calibrated, L3 done.

**PHASE B (5–9) THE DECIDING CELL.** Quintuple accumulates; **audit #1 at it.8
reads `E_t` on the record**. Cameron finishes the rank/rank₊ table for every M3
task. Foreman: L4. **EXIT B** — either `E` crossed (S1 scored, **both directions
honest**) or **"undecided at `E_t = [x]`" printed and the run CONTINUES INTO PHASE
C rather than blocking it.**

**PHASE C (10–17) THE TRAINING RUN.** S3 on the certificate: segments per §1.3,
**resume verified bitwise at the first boundary (N1's bind on live fire)**,
β-sweep, Dobrushin/Φ per checkpoint. Cameron launches S4 on checkpoint 1. **EXIT
C** — a trained checkpoint on HF, **or the failing segment named with its recovery
path exercised.**

**PHASE D (18–24) CAPABILITY AT TRAINED WEIGHTS.** Capability table v0 — the M3
task, the gap task, the matched-parameter comparison, published refs beside,
**e-process / SPRT statistics only**. Cameron: S4 complete, **the trained twin for
every historical probe number, deltas printed**. Foreman: **prediction-ladder
verdict with its pre-registered kill**. Audit #3 at it.26.

**PHASE E (25–30) CLOSURE.** D1 final: certificate post-mortem, identity chapter
with L3, the five-round instrument ledger, **the Hankel frame as the methods
capstone**. `it.29` prognosis with every scoreboard item's final state and round
8's X-candidates **declared** (X-next: **does settling depth `t*` correlate with
gap-task capability** — the dose-response question the dial makes askable).
`it.30` **STOP on the prognosis.**

---

## 4. KILLS — immutable at first datum

**K-1** e-process crosses for the **TWIN** ⇒ settling loses at `α=0.05`
anytime-valid; **equilibrium clause CUT project-wide**; claim routing-only forever.
**K-2** prediction ladder fails **both** directions ⇒ Hankel frame dead, **House
exception fires**, D1. **K-3** gap task solved **equally** by the non-negative arm
⇒ the signed program is **capability-irrelevant on its own best terrain**; retired
in writing. **K-4** training run cannot cross a session boundary **bitwise** ⇒ S3
halts, **the N1 certificate is REVOKED, not patched mid-run**; round 8 re-certifies.
**K-5** rank/rank₊ machinery disagrees with the worked example ⇒ **the instrument
is broken and nothing downstream of it is read.**

**G-NEW: no contraction-certificate proposal this round.** Three deaths; **the
post-mortem is the deliverable.** All prior **G1–G8**, **X₁₅** same-message
schedule arithmetic, **G5** scope law, the **value-comparison law** and the
**must-fire law** are in force every iteration.

---

## 5. THE ESCALATION CHAIN — binding

    nurses  ->  fellow  ->  Wilson  ->  { Health Inspector | Dr House }

**Nurses** are inference and kernel engineers; mandate is whatever hack makes this
module comparable to vLLM while staying original work. Raw evidence only, never
conclusions, never a stance. **Fellows** dispatch in ONE message, same question,
different stances, **test-bound: a finding needs a RED test shown, or it goes to
Open** — and **RULE 5: a kill needs a replacement route, or the report is
incomplete.** Each fellow uses superpowers (`test-driven-development`,
`systematic-debugging`, `dispatching-parallel-agents`). **Wilson** reports verified
facts only and routes the fork: **not a leap ⇒ Health Inspector**, who is the
better engineer; **a leap ⇒ Dr House on `fable`**, five minutes, hard stop, no
nurses, one leap or "no leap", **a HYPOTHESIS that a fellow binds RED-first or it
stays in Open forever.**

## 6. RULE 4 — THE REGISTER

**Every agent reports in caveman.** Short words, verdict first, no preamble.
**Numbers, identifiers, commands, math and quoted text pass BYTE-EXACT.**
Uncertainty stays visible. **Artifacts are exempt** — `DONE.md`, `D1.md`,
`CHECKLIST.md`, `METHODS.md`, commit messages, Lean files and docstrings stay in
normal precise English.

## 7. STANDING POLICY

**Never block on a measurement.** Every optimisation ships a **bitwise** bind; a
faster path that moves a number is a **new arm**. Long runs bucket. Threads pinned
**in the probe file**. Every number carries its command, seed, thread count and
control. `inspector.py` is tri-state. **Absence is NOT FOUND, never "unoccupied".**

## 8. COMPLETION — line 1 of `DONE.md`

`HILBERT: KEPT` requires **D2**: *"a settling attention module, settling depth set
by an owned dial with a known transfer function, beating softmax and its own
unsettled twin under anytime-valid statistics at the budget where softmax provably
passes, trained free on a certified resume path, on HF with a capability table
whose tasks carry their Hankel rank and nonnegative-rank bounds"* — **every clause
naming the kill it survived.** Requires **S1's +12 branch AND S3**.

`HILBERT: BROKEN — <item> <which kill fired>` otherwise. **An honest BROKEN
outranks an unfinished KEPT, and D1 ships either way.**

Emit `<promise>HILBERT</promise>` only when one is completely and unequivocally
true.
