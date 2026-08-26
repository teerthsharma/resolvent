# done7.md — CEQ v9, round 7, handoff at iteration 11

Written for whoever picks this up next, including a later version of the same
session. It states what exists, what it cost, what is still missing, and how far the
whole product is from done — with the derivation of that number rather than an
assertion of it.

---

## 0. THE ANSWER, FIRST: **≈ 35 % of the whole product**

That number is not a feeling. It is the weighted count in §1, and it splits very
unevenly, which is the part that matters more than the number:

| half | state | why |
|---|---|---|
| **The engineering** — a module that trains, beats its baseline, and ships with an honest table | **≈ 80 %** | measured, journalled, reproducible, sealed |
| **The scientific claim** — that the module understands causality and consequence | **≈ 5 %** | **the deciding measurement has been taken zero times** |

**The product is gated on the second half, not the average of the two.** A module
that ships beautifully and does not do the thing is not the product that was asked
for.

**The single sharpest sentence in this file:** in seven rounds, the module has never
once been asked to predict an equilibrium. That is not a metaphor — it is a property
of the corpus, established at iteration 7 and confirmed on every registered task.

---

## 1. THE DERIVATION OF 35 %

Each component is scored against evidence in this repository, not against intent.

| # | component of the product as asked | % | evidence |
|---|---|---|---|
| 1 | An attention module that trains at the smallest scale | **100** | `n_params = 4769`, 25 journalled units, `results/m3_quintuple_v2.jsonl`, bit-identical on replay |
| 2 | Survives equal to, or supersedes, self-attention | **100**, narrow | `settled − softmax = +0.108437`, CI `[+0.068181, +0.147110]`, **5/5 seeds**; `twin − softmax = +0.111396`, CI `[+0.100873, +0.121920]`, **5/5**. One task family, one geometry. |
| 3 | A capability table, honest about negatives | **85** | `results/capability_table_v0.md`, 6 587 B; prints `NO DIFFERENCE` where true; tamper must-fire **seen to fire both directions** (`+0.497041` / `−0.502959` when planted). Not uploaded. |
| 4 | On HuggingFace | **40** | `ceq/hf_artifact/` built and locally verified; **not uploaded** (needs the author's say-so) and **ships no weights** — a random-init `model.safetensors` measures 1 901 686 656 B and was correctly refused. 3 trained checkpoints exist on disk (2 108 341 B each) but are `pivot_unsigned`, not settled/twin. |
| 5 | Trainable on free tiers | **20** | trains on 2 CPU threads at ≈ 300–800 s per unit, so plausible; **the Kaggle run has not been started.** Plausible ≠ demonstrated. |
| 6 | **Understands causality and consequence** | **5** | **This is the product. See §2.** |

Weighting components 1–5 at one share each and component 6 at five shares — because
it is the thing that was asked for and the rest is scaffolding for it — gives
`(100 + 100 + 85 + 40 + 20 + 5×5) / (5 + 5×20) ≈ **35 %**`.

**Cross-check, independent of that arithmetic.** Dr House, given the whole repository
and asked the same question, answered in iterations rather than percent: **two
fellow-iterations to the first honest reading, four to five to a shipped product, if
the reading goes our way.** Those are consistent: the scaffolding is nearly done and
the reading is untaken.

---

## 2. WHY COMPONENT 6 IS AT 5 %, AND NOT AT 0 OR AT 50

**It is not 0.** The instruments to take the reading now exist and several are good:

* **Consequence fidelity** — the one instrument in this project that measured the
  actual trained object and returned a decisive number: the trained *non-negative*
  arm puts a negative sign on `0.492188` of drawn interventions, and swapping `GELU`
  for `nn.Identity()` on identical weights takes that to `0.000000`. It is now §1.7c
  of the contract, the capability metric of record.
* **Eight registered tasks**, four of them equilibrium-labelled (`e1_anchor`,
  `e2_consequence`, `e3_t{1,2,8,32}`).
* **An anytime-valid evidence process**, calibrated in both directions, with its
  control seen to fire.

**It is not 50.** Every reading that would support the claim is missing:

* **The consequence-fidelity column is `NOT MEASURED` for every arm**, because
  `scale/m3_quintuple.py` journals metrics and saves no per-cell weights. Chase is
  closing that now.
* **The `t*` dose-response curve — the one thing — has never been run.**
* **`e3_t*` inherits a rigged oracle.** All four rungs bind `equilibrium_oracle`, the
  same oracle as `e1_anchor`, whose label is the signed path sum that the ceq
  resolvent already computes. On the ladder, only `settled − twin` is creditable.
* **`e2_consequence` is registered and unmeasured.**
* **E4, the one non-synthetic substrate, was struck by its own gate** at iteration 11.

---

## 3. WHAT ROUND 7 ESTABLISHED, IN ORDER OF IMPORTANCE

**1. The corpus was the binding defect, and it went unnoticed for seven rounds.**
Every registered oracle was a closed-form function of its input —
`x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` and `x[:, :, CH_FLIP].sum(dim=1) ** 2`.
Neither has a fixed point. **The settling arm ran a Neumann series to convergence and
then predicted a product of two numbers.**

**2. The deciding contrast is therefore correct and uninformative.**
`settled − twin = −0.002959`, exact 95 % CI `[−0.042903, +0.031557]`, covering zero,
3/5 seeds favouring settled. **Not evidence that settling fails — evidence that
settling has never been tested.**

**3. The gain is the mixture, not the fixed point, and not a lookup.** The one-hot
control is **worse than plain softmax**: `argmax − softmax = −0.118456`, CI
`[−0.134115, −0.102786]`, 0/5 seeds. Reading one pivot is worse than reading none.
And `argmax` fails its own bar at `1.010779`, so the largest positive number in the
table — `settled − argmax = +0.226893` — is a win over a failure and licenses nothing.

**4. The five-seed cell could never have decided anything, and this was provable
before the first seed.** Bounded increments and a betting fraction capped at ½ force
every factor `≤ 1.5`, so `MIN_T_MIXTURE = 13` and the maximum attainable evidence at
five seeds is `3.80169140625` against a threshold of `40`. **`undecided` here is
structural, not a data outcome.**

**5. The capability frame died at trained weights.** `rank₊ > rank` bounds the state
count of an automaton with a *linear* value path; the arm's readout is
`Linear → GELU → Linear`, and the nonlinearity supplies the sign the operator
withholds. Retired to task metadata.

**6. Settling adds variance and no mean.** `sd 0.064106` against the twin's
`0.016547` — **3.874×** — on the same batches, the same initialisation and the same
parameter count, with per-seed deltas alternating in sign.

---

## 4. THE ONE THING, AND WHAT BLOCKS IT

**The `settled − twin` dose-response across the `t*` ladder, at trained weights. One
curve.** Predicted flat at `t* ≤ 1` and growing with `t*`. That curve is
consequence-awareness; nothing else currently on the table is.

**What blocks it is small and named:**

1. `scale/m3_quintuple.py` had no `--task` flag — `scale/m3_capability.py:246` has
   one. Chase is porting it.
2. It saves no per-cell weights, which is exactly why the fidelity column is empty.
3. Then: `settled` and `twin` across `e3_t1 / e3_t2 / e3_t8 / e3_t32`, five seeds,
   matched parameters, fixed-sample.

**The kill and the fallback are both pre-registered.** If the interval covers zero at
every rung including `t* = 32`, settling retires and **the twin ships** as
pivot-routed mixture attention: `+0.111396` over softmax, CI
`[+0.100873, +0.121920]`, sd `0.016547`, four times tighter than settled. **A smaller
claim, fully earned, and still a product.**

---

## 5. THE GRAPH SUBSTRATE: RECOVERED, PORTED, STRUCK, REROUTED

`ceq/rips.py` ports six geodesic Vietoris–Rips graphs on `S²` from the author's own
merged `google-deepmind/mujoco#3396`, recovered at commit `5d91d878` where the file
was added and removed inside the same pull request — cut upstream for scope, not for
any defect. The port matches an independent derivation **6/6 on edges and 6/6 on
components**, and both bridge cases satisfy `pre_bridge == components + 1`.

**Gate (a), truncation: PASSED.** A 1-hop reading is worse than the mean; accuracy is
monotone in `k`; exact only at the diameter, which reaches **62**.

**Gate (b), the decoder must-fire: STRUCK IT.** A static local-degree decoder reads
the label at criticality at **`0.4710`**, and an independently written probe reads
**`0.3360`**, both inside `PASS_BAR = 0.5`.

**Root cause, one line:** the bridge joins the two *nearest* components, which on `S²`
is always a speck against the giant — measured `3 × 222` and `1016 × 4`. A speck
saturates at radius 2–3, so "am I in the merged component" degenerates into "did my
own ball grow".

**The reroute is measured:** join the two *largest* components instead. The local leak
closes from `0.1565` to `0.9951` at radius 3, and the control fires — a planted
degree-sum label reads `0.0000`, a planted balanced one `0.5530`. **E4′ passes both
gates at `n = 1024`, is not yet registered in `M3_TASKS`, and is not admissible at
`n = 64`, which still leaks at `0.0055`.**

**And gate (a) was measuring the wrong quantity, which generalises:** reachability
between two named nodes needs the diameter, but *deciding* the label never requires
reaching the second node. **A diverging diameter does not imply a task is not locally
decidable.** Any future substrate needs the decoder gate, not just the truncation
gate.

---

## 6. THE ERROR RECORD — fourteen vacuous controls, four of them from here

A control that cannot fail is worse than no control, because it reads as evidence.
Fourteen have been struck in this campaign across five authors. The four authored or
relayed from the main session:

1. A genesis assertion that sliced a digest to zero characters and asserted a label
   was present in the result — `X in "" + X`, true for every input. **Caught before
   shipping.**
2. A 2×2 greedy must-fire that could not fire, because on two points the monotone
   assignment is forced. **Replaced with drawn instances.**
3. **A journal scan that read only top-level keys of records that nest their payload
   under `value`.** Reported "zero readings above 1.0"; the truth is 22 of 68. **It
   shipped, stood two iterations, and was used to strike a colleague's evidence.**
4. **A gate relayed as "that contrast IS the control" whose prescribed pass-case has
   every node in one component**, so its label is constant and it could never have
   controlled anything. **Caught by the fellow it was handed to.**

`scale/journal_scan.py` now makes class 3 a mechanism rather than a discipline: the
caller supplies a witness that must be recovered, and a scan whose witness is missing
**raises** instead of returning an empty list. **Its own tests found two bugs in it
before it shipped**, the worse being that the witness was checked against paths
*walked* rather than paths the *selector keeps* — so a transposed selector passed. A
witness that cannot fail on a wrong selector is precisely the disease it was written
to cure.

**The standing rule this produces:** every must-fire's PASS half now needs its own
non-degeneracy check before dispatch, and every reported absence needs a planted case
that changes it.

---

## 7. STATE OF THE BOARD

| item | points | state |
|---|---|---|
| S1 — headline cell, anytime-valid | +12 / +6 / −4 | **read, routing-only branch.** Not anytime-valid and provably cannot be at 5 seeds. |
| S2 — Hankel-gap family, one gap task in M3 | +3 | **corpus half closed** (`counter_squared` runs, gap `rank 3` vs `rank₊ ≥ 6` at `s = 64`); **frame retired the same day**, so the gap is metadata. |
| S3a — table + HF from existing artifacts | +3 | **shipped, not uploaded** |
| S3b — Kaggle run | +3 | not started |
| S4 — probe battery at trained projections | +3 | partial |
| S5 — Merkle journal + tamper must-fire | +1 | **earned** |
| S6 — fused settling step ≤ 1.1× | +1 | target quantified; gate arithmetically unreachable at `7/6 = 1.1667` |
| S7 — D1 to acceptance | +2 | open |

**Scoreboard: 23.**

---

## 8. WHAT THE NEXT SESSION SHOULD DO, IN ORDER

1. **Take the reading.** Chase's `--task` port plus weight saving, then `settled` and
   `twin` across the four `e3` rungs at five seeds. Print the `e3` rig caveat beside
   every row. **This is the only thing that moves component 6.**
2. **Fill the consequence-fidelity column** from the saved weights, for every arm
   including softmax, with Clopper–Pearson exact intervals.
3. **Register E4′** in `M3_TASKS` — it needs the M3 tensor batch format, which is a
   build rather than a measurement, and it is the only non-synthetic substrate that
   has passed both gates.
4. **Then, and only then, decide whether to upload.** The package is built and the
   upload is a one-command action that needs the author's say-so; it should carry
   real weights, not random-init ones.

**Do not** propose another contraction certificate — three died and the contract
makes a fourth a stop. **Do not** build another instrument that measures something
adjacent; that is the disease this round diagnosed.
