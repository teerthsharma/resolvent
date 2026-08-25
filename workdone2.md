# Work done — round 2, iterations 34–46

Round 1 is archived in `DONE_ARCHIVE_ROUND1.md` (5,922 lines). This covers the
second round, which began after M2 went RED on the shipped operator.

**Round 2 produced no new capability. It produced a repository that can be
believed.** Every headline number in this project before iteration 34 was either
measured on an operator that ships nowhere, fitted to a withdrawn exponent, or
asserted without a test that could contradict it. That is now fixed, and the
fixing is the result.

---

## 1. What was found, in order

| # | iteration | finding |
|---|---|---|
| 1 | 34 | README and MODEL_CARD described a working operator; neither stated the verdict |
| 2 | 35 | **`‖A^hops‖ = 1.471448` was fabricated** — not reproducible at any of 1,800 settings |
| 3 | 36 | **`LOOP_PROMPT.md` carried killed numbers as law**, under "must NEVER be re-derived" |
| 4 | 37 | Three existing binds were all *presence* checks; nothing asserted *absence* |
| 5 | 38 | **`COSTS` shipped `slope: -1.389`** to HuggingFace while the README withdrew it |
| 6 | 39 | Every M2/M5 number is measured at **random initialization** |
| 7 | 40 | `lake build \| tail; echo $?` reports **tail's** exit status, not the build's |
| 8 | 41 | M2's tail zeros rest on 213 and **60** draws (CP upper `0.05963` at s=512) |
| 9 | 42 | **The kill survives `floor = 0`** — the instrument is not convicted |
| 10 | 43 | My own iteration-39 novelty claim was **false**; PROGNOSIS already recorded it |
| 11 | 45 | The Inspector reported attribution HITS with **zero matching lines** |
| 12 | 46 | **The journal is only bitwise-reproducible at a fixed thread count** |

---

## 2. The verdict on the operator, unchanged and now permanent

Shipped operator `sgate`, `PROTOCOL: SCALING`, `c` drawn from the pivot set:

| s | 8 | 32 | 128 | 512 |
|---|---|---|---|---|
| pivot-routed | 0.16511 | 0.02732 | 0.00000 | 0.00000 |
| dense control | 0.16511 | 0.03655 | 0.00000 | 0.00000 |

Slope **−1.298** against a pre-registered bar of **−0.3**. Routing is **worse**
than dense (−1.088), which is the opposite of the thesis.

**Three attempts to overturn it, all failed:**

1. **Weak tail?** The two well-measured points alone give
   `log10(0.02732/0.16511)/log10(4)` = **−1.2977** against the published
   **−1.2980**. The zeros contribute **0.0003**. A tail re-measurement cannot
   rescue it.
2. **Floor artifact?** The statistic is `lo*hi < 0 AND min(|lo|,|hi|) > floor` —
   the second condition is a magnitude gate at `1e-6`, the same floor that
   invalidated `−1.389`. Paired on identical draws: **`floor = 0` gives −1.1150**
   (R² 0.9350). The kill fires with the gate removed, and removing it makes the
   slope **steeper**.
3. **Untrained tensor?** The instrument was tested instead of the operator, and
   survived: the same probe reads **−0.034 for `tgate`** against **−1.298 for
   `sgate`**, so it is not returning a constant. Per
   `M2_TRAINED_PREREGISTERED_READING.md` — written **before** that test — no
   trained number can overturn this.

The floor does discard a growing share of genuine sign changes — **3/44 at s=8,
10/19 at s=32, 2/2 at s=128** — so the published `0.00000` reads stronger than
the data supports. The verdict is unaffected.

---

## 3. The instrument taxonomy — the transferable finding

Sorted by **what they compared**:

| | instruments | false readings |
|---|---|---|
| compared **structure** (regex, slices, substrings, exit codes) | **9** | **9** |
| compared **values** (numbers against numbers) | **3** | **0** |

The nine: the LOCK slice boundary; the LOCK-line scraper matching prose; the
provenance audit missing a line break; the ARMS-DISTINCT dispatch slice broken by
a refactor; a line-scoped strike-marker check whose markers sat one line away; a
substring search for `809` that matched `0.038097`; `lake build | tail; echo $?`;
`grep | head -3 &&` reporting hits on empty input; and a "published number" check
that printed its target instead of comparing it.

The three that have never failed: the calibration gate, the journal's bitwise
replay, and layer 1 of the struck-constant bind.

**And the harder lesson: naming a class does not prevent it.** Iteration 40 wrote
*"the defect is ANY pipeline, because the shell reports only the last stage"* —
and iteration 45 wrote `| head -3 &&` anyway. Iterations 37, 38 and 45 each
reproduced a class recorded one to five iterations earlier. **Every defence that
held is a test. None of the rules held.**

That is why `inspector.py` exists: eight checks, no decision after a pipeline,
each comparing a value, **each with a must-fire control that must be seen to
fire**. An all-green run whose controls stayed silent exits nonzero, because a
blind check is worse than a failing one.

---

## 4. Why fourteen passing tests missed a fabricated number

`‖A^hops‖ = 1.471448` appeared in four files. The adjacent test asserted
`got > 1e-3` — **an inequality**. Both the true value (`0.880500`) and the false
one satisfy it. It also sat beside a real reading (`(64,2) = 1.476635`), so it
never looked wrong.

**A test that pins an inequality cannot protect an exact number quoted from it.**
All eight measured `(s, hops)` values are now asserted at `abs=5e-7`.

---

## 5. What is now bound

| bind | asserts |
|---|---|
| `test_measured_operator_is_shipped` | arm → operator, resolved from the dispatch |
| `test_arms_distinct` | 28 arm pairs differ bitwise at seed 0 |
| `test_theorem_hypotheses_hold_at_shipped_settings` | 8 tail norms pinned at `abs=5e-7` |
| `test_no_struck_constant_ships` | walks `COSTS` for withdrawn values; 9 documents |
| `test_resume_checkpoint` | resumed weights **bitwise** vs uninterrupted run |

`pytest tests/loop tests/w11 tests/chase/test_resume_checkpoint.py` → **100
passed**. `lake build CEQ` → exit 0, **27 theorems, zero `sorry`**.

---

## 6. State of the deliverable

- **Resume works and is bitwise-verified.** `LOOP_PROMPT.md` claimed "~15 lines of
  missing checkpoint/resume" — the code was already there. Now proven at the
  bitwise end, with a must-fire control that no-ops `AdamW.load_state_dict`.
- **Nothing has trained above 3.65M** against a 300M bar. The free-T4 shape is
  25.7M — **8.6% of the gate**. A reading there could not settle the question.
- **No trained weights are published.**

---

## 7. What we do not have — stated as an equation

Everything above is bookkeeping. This is the actual open problem.

Let `t_c` be the contribution of a perturbed token `c` to the hop-2 sum, and
`B_k(s)` the background it must overcome:

```
flip(s)  =  P[ sign(t_c)  flips  sign( Σ_{p ∈ P \ {c}} t_p ) ]
         ≈  P[ |t_c|  >  |B_k(s)| ]
```

Pivot routing was built on the assumption that fixing `|P| = k` fixes the
background at `k` terms instead of `s`. **It does — and the rate decayed anyway,
at −1.298.** So the term *count* was never the mechanism.

The measured decay says:

```
|B_k(s)|  grows with s   even though   k  is constant
```

**Why**: `select_pivots` takes the top-`k` of `s` candidates by salience. Those
survivors are **extreme order statistics**, whose magnitude grows like
`√(2 log s)`. A perturbed token `c` is re-drawn to a **typical** value. So
selection shrinks the number of competitors while inflating each one — and the
perturbed token is swamped by exactly the tokens selection promoted.

```
        t_c  ~  typical            (O(1),  no s dependence)
   B_k(s)    ~  k · max-of-s       (O(√log s),  grows without bound)
   ⇒  flip(s) → 0                  for ANY fixed k
```

**Therefore:**

```
X  =  an aggregation over hop ≥ 2 in which the surviving background
      does NOT inherit the scale of the tokens that selection promoted
```

Not "fewer terms" — that was tried and is dead. **Scale-free comparison.**

Ruled out already, by measurement or by prior art:

| candidate | status |
|---|---|
| fewer background terms (pivot routing) | **dead** — this document, −1.298 |
| non-Archimedean / max-plus | **dead** — G1 fires, Tropical Attention arXiv:2505.17190 |
| hierarchical criticality (Dyson tree) | **dead** — no zero crossing of slope(θ) |
| sign-determinacy (SNS pivot block) | **dead** — sign-blind by the twin test |
| certified selection (group testing) | **dead** — sign-blind by the twin test |

**Every route in the contract broke the same hypothesis — the term count — and
none of them touched the scale.** That is the gap, and it is where the next
architecture has to start.
