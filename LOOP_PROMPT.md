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
| **S1** | The deciding cell reads settled vs twin **on an E-task (§1.7)** at matched parameters, statistics per §1.8 | **+12** *(routing-only: +6; loss with valid CIs: **−4** and D1)* |
| **S2** | **The E-task family (§1.7) built, calibrated, and registered in `M3_TASKS`**; difficulty dial `t*` printed per task; rank/rank₊/Myhill–Nerode printed beside each task **as metadata only** | **+3** |
| **S3a** | **Capability table v0 published on HF from EXISTING artifacts** — phaseD weights, `results/m3_capability.txt` numbers, consequence fidelity (§1.7c) per arm — every verdict stated even where it reads NO DIFFERENCE | **+3** |
| **S3b** | First Kaggle run completes on the signed **N1–N6** certificate; trained checkpoint on HF; capability table v1 at those weights | **+3** |
| **S4** | Probe battery re-run at **TRAINED** projections (retention, ARM P, F-lam geometry) — **the oldest OPEN item closed** | **+3** |
| **S5** | Merkle journal live with **must-fire tamper test** | **+1** |
| **S6** | Fused settling step: clock ratio **≤ 1.1×** (from `1.6546`) | **+1** |
| **S7** | D1 to acceptance criteria, **certificate post-mortem as its capstone chapter** | **+2** |

**Ceiling ~50. Floor ~16 with D1 done. NOTHING ELSE SCORES.**

---

## 1. THE MATHEMATICS

### 1.1 THE HANKEL PROGRAM — RETIRED AS A CAPABILITY FRAME (House amendment, it.7)

**The frame died at trained weights and stays dead.** The trained NON-NEGATIVE
arm puts a negative sign on a token's influence in `0.492188` of drawn
interventions; swapping `GELU` for `nn.Identity()` at the same trained weights
takes that to `0.000000` (Foreman F-J3). The readout `Linear → GELU → Linear`
supplies the sign the operator withholds, so no arm in this project is a
nonnegative weighted automaton and `rank₊` bounds nothing about any of them.
**No operator-algebra property (rank, rank₊, cone, contraction constant) may be
claimed to predict a capability of a trained arm.** The instrument survives:
`ceq/hankel.py`'s exact rank, the rectangle bound, and Myhill–Nerode counts are
printed beside every task as **difficulty metadata**, never as a prediction.
The §1.1 prediction ladder is struck; §1.7's ladder replaces it. The text below
is retained as the record of what the instrument computes.

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

**PREDICTION LADDER — STRUCK by the House amendment.** Its joint 3 is refuted
at trained weights (F-J3, above). Ladder E in §1.7 is the replacement.

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

**DECISION RULE, immutable once the first seed lands:** `E_t ≥ 40` ⇒ settled beats
twin, **any `t`**; symmetric process for the twin direction; neither crossing by
phase end ⇒ **"undecided at evidence `E_t = [value]"`, printed** — and **the round
says so instead of slipping.**

**THE THRESHOLD IS 40, NOT 20, AND THE ARITHMETIC IS WHY (Chase, it.7).** Ville
bounds *one* process: `P(sup_t E_t ≥ τ) ≤ 1/τ`. Two directions are live — nobody
knows in advance which arm wins — so the event actually reported is their **union**
and the bound is `N_DIRECTIONS / τ`. Then `2/τ ≤ 0.05` forces **`τ ≥ 40`**, giving
`α = 0.025` per direction. A threshold of 20 buys `0.10`, not `0.05`. Fixing this in
prose ("say *settled crossed*, never *some direction crossed*") is valid arithmetic
enforced by a habit rather than by a constant; the constant is enforced by
`scale/eprocess.py:213-216`.

**WHAT THE THRESHOLD COSTS, stated so no one re-litigates it:** `MIN_T_MIXTURE`
rises `11 → 13` and `MIN_T_SINGLE_ARM` `8 → 10`. **This strengthens the iteration-3
kill of the five-seed cell rather than weakening it** — `max_attainable(5) =
3.80169140625` against `40` — and **the ceiling does not depend on `B`**, because the
largest legal increment is `d = B` and the factor is `1 + λ` whatever `B` is. Pinned
by `test_the_ceiling_does_not_depend_on_the_bound`.

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

### 1.7 THE E-TASK FAMILY — EQUILIBRIUM BECOMES THE LABEL (House amendment, it.7)

**The finding that forces this section:** in seven rounds the word "equilibrium"
appears in this project only about the MODEL — the α fixed point, the Karcher
mean, the QRE stance — and **never once about the label.** A grep of `DONE.md`,
`DONE_ARCHIVE_ROUND1.md`, `BACKLOG.md`, `D1.md`, `THEORY.md`, `THEORY2.md`,
`PRIOR_ART.md` and every task module for an equilibrium-labeled task returns
zero hits. The goal sentence — *"the next-equilibrium predictor"* — names the
prediction TARGET. Every task in `M3_TASKS` is a static function of the input
with no equilibrium anywhere in its oracle, so the settling arm has been asked
to show a settling advantage on tasks where settling has nothing to compute.
The deciding contrast was near-zero by construction (gate near uniform at M3
geometry, `‖settled−twin‖/‖twin‖ ≈ 0.11`), which is why it needed 13+ seeds.

**The family, all in the M3 tensor format (`make_batch` geometry, executable
oracle, no stored answer key):**

* **E1 — the anchor.** Tokens carry the rows of a strictly lower-triangular
  signed `A` and a vector `b`; the label is one coordinate of the settled state
  `z* = (I − A)⁻¹ b` (finite by nilpotency). The ceq path-sum computes exactly
  this object, so E1 is a **calibration anchor and a must-fire, never a
  headline** — a win here is a rigged demo and is credited nothing, but a
  settled arm that cannot beat NRMSE 1.0 on E1 means the harness is broken and
  nothing downstream of it is read (K-5E).
* **E2 — consequence.** The label is a coordinate of the NEW fixed point of a
  small contraction system after a one-token `do()`-shock: a damped
  best-response game at `τ > τ*` (`ceq/nash.py::safe_tau` already computes the
  well-posedness threshold), or a Markov row edit with the stationary
  distribution as label. One token's intervention decides the downstream
  equilibrium — the goal's own sentence, as an oracle.
* **E3 — dose-response.** An iterate-to-stable update rule with known
  stabilization time `t*`; `t*` is the printed difficulty dial. This makes the
  already-declared X-next question (*does settling depth correlate with
  capability*, Phase E) measurable in this round instead of the next one.

**LADDER E, pre-registered, falsifiable in BOTH directions.** At matched
parameters, the settled arm beats the twin and softmax on tasks with
`t*` **above** the model's hop budget, and **ties** on `t* ≤ 1`
(retrieval-regime) tasks. Settled loses on deep-`t*` tasks ⇒ settling buys no
equilibrium capability (**K-2E**). Settled wins on `t* ≤ 1` tasks ⇒ the win is
capacity leakage, not settling, and the claim is routing-only (G4 shape).

### 1.7c CONSEQUENCE FIDELITY — the capability frame that replaces Hankel

Over drawn single-token `do()`-interventions whose oracle effect `Δy` is in
closed form (every E-task and every `M3_TASKS` oracle provides it), report per
arm at trained weights: **(a)** the fraction of interventions where
`sign(model Δŷ) = sign(oracle Δy)`, with its exact interval; **(b)** the
regression slope of `Δŷ` on `Δy` with CI. This measures the trained network's
behavior — the only measurement plane that survived F-J3 — and it is computable
for softmax, so it is a comparison, not a self-portrait. It is a **capability
table column**, and the honest end is pre-registered: if softmax at matched
parameters matches the signed and settled arms on consequence fidelity at every
distance `d ≥ 256`, the operator contributes nothing behavioral and the signed
program retires on its own terms (RULE 5: Retire).

### 1.7d CREDIT ROUTING ON THE E-LADDER, AND THE E4 CANDIDATE (House amendment, it.10)

**E3 shares E1's oracle.** `M3_TASKS` registers every `e3_t{t}` with
`equilibrium_oracle` — the same signed path sum for which E1 is declared a
rigged demo (`scale/negation_scope.py:635` and `:640`). The settled arm's own
resolvent computes that object, so **`settled − softmax` on any e3 task is
credited nothing**: it inherits E1's rig at every chain length. Credit on the
E-ladder flows through exactly one contrast — **`settled − twin` at matched
parameters, read across the `t*` ladder** — because the twin carries the
identical mixture and differs only in the iteration. The believable signature
is dose-response: the contrast compatible with zero at `t* ≤ 1` and growing
with `t*`. Flat in `t*` ⇒ routing-only (K-2E). Nonzero at `t* ≤ 1` ⇒ capacity
leakage (G4 shape). `e2_consequence` does not share the oracle and keeps both
contrasts.

**E4 candidate — the Rips corpus (`ceq/rips.py`), admitted CONDITIONALLY.**
The label must be the post-bridge global connectivity fact — same-component
for a marked node pair, tokens carrying incidence rows — never the component
count, which local density statistics predict away from criticality. Near the
transition one bridge edge flips the label and the iteration depth required is
the graph diameter, which diverges at the critical point: a real `t*`, not a
knob. **Admission gates, RED-first, both must print before any E4 number is
read:** (a) a `k`-hop label-propagation truncation is bounded away from the
label and tightens with `k` on the critical cases (§1.7 property 2, verbatim);
(b) a static decoder on local degree statistics FAILS on the critical cases —
it is expected to pass on `StableSparse_S2Rips_64` and
`SupercriticalDense_S2Rips_256`, and if it also passes at criticality, E4 is
another adjacency and is struck without appeal.

### 1.8 STATISTICS RULE — effect-first (House amendment, it.7)

`scale/eprocess.py` proves `MIN_T_MIXTURE = 13`: no contrast enters the
e-process unless its plan runs `≥ 13` seeds. Any other contrast is read ONCE as
a pre-registered fixed-sample paired bootstrap at exactly `N` seeds, `N` fixed
by power analysis on the pilot SD before the data land. **An effect that needs
more than 13 seeds at the deciding cell's geometry is reported as "too small to
matter at this scale" and is not chased — the 436-unit path is closed.** The
anytime-valid machinery is kept for what it is calibrated for, not worn as a
ceremony by cells that cannot cross.

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

**DR HOUSE — RELEASED at it.7, after the 4.9 frame death.** His output is the
§1.7/§1.7c/§1.8 amendment and four ranked hypotheses in `house-events.jsonl`.
Per the standing rule they are **hypotheses, not findings**: each re-enters the
differential and is bound RED-first by a fellow — Cameron owns the E-task
builds, Chase the deciding cell and the S3a table, Foreman consequence fidelity
and the Ladder E pre-registration — or it stays in Open forever.

**ONE pre-registered exception:** if Ladder E fails in **BOTH** directions —
settled arms **lose** on deep-`t*` tasks **AND win** on `t* ≤ 1` tasks — that is
a **theory death**, and the five minutes go to: *"The E-frame says where
settling must matter and it didn't. What is the resource the tasks are actually
pricing?"*

---

## 3. THE 30 ITERATIONS

**House redirect at it.7, binding on the remaining phases:** where a phase below
says "Dyck corpus", "gap task" or "rank/rank₊ table", read "E-task family
(§1.7)" for the capability-bearing work and keep rank/rank₊ as printed metadata.
The running twin arm completes and is read under §1.8 as a fixed-sample paired
comparison — work already paid for is read, not discarded. S3a (the table from
existing artifacts) is Phase-C work that does NOT wait for the Kaggle run.

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

**K-1** e-process (or the §1.8 fixed-sample read) decides for the **TWIN** ⇒
settling loses; **equilibrium clause CUT project-wide**; claim routing-only
forever. **K-2E** Ladder E fails in **both** directions — settled loses on
deep-`t*` E-tasks AND wins on `t* ≤ 1` tasks ⇒ the settling program has no
equilibrium capability; D1, and RULE 5's Retire route is the deliverable.
**K-3E** E2/E3 solved **equally** by the twin at matched parameters ⇒ settling
is **capability-irrelevant on its own home terrain** — the terrain it was built
for, not an adjacent one; retired in writing. **K-4** training run cannot cross
a session boundary **bitwise** ⇒ S3b halts, **the N1 certificate is REVOKED,
not patched mid-run**; round 8 re-certifies. **K-5** rank/rank₊ machinery
disagrees with the worked example ⇒ **the metadata columns are not printed**
(the instrument is broken; capability reads are unaffected, per §1.1's
retirement). **K-5E** the settled arm cannot beat NRMSE 1.0 on E1, the task its
own forward computes ⇒ **the harness is broken and nothing downstream is read.**

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
whose tasks carry their Hankel rank and nonnegative-rank bounds as metadata, at
least one task equilibrium-labeled per §1.7"* — **every clause naming the kill
it survived.** Requires **S1's +12 branch AND S3a AND S3b**.

`HILBERT: BROKEN — <item> <which kill fired>` otherwise. **An honest BROKEN
outranks an unfinished KEPT, and D1 ships either way.**

Emit `<promise>HILBERT</promise>` only when one is completely and unequivocally
true.
