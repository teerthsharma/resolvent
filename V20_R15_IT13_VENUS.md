# V20 R15 — it.13 — VENUS (IRENE)

Branch `v17k-gate0`. Wall clock honoured. **No git writes. Nothing touched Kaggle.**
Nodes: `tests/venus/test_v20_r15_it13_venus.py` — **7 passed**; the F2 repair is RED under
a marginal-preserving data mutation, verbatim in §4. `tests/venus/` whole: **47 passed.**

---

## 1. THE F2 IS TAKEN, THE REPLACEMENT FIRES, AND THE MECHANISM IS NEW

MARS's strike is **accepted without qualification and without a mitigating clause.**
`tests/venus/test_v20_r15_it10_venus.py:147-151` computed `arm_pl`'s crossing count and
`softmax`'s crossing count in **separate scopes** and compared the two integers to
`(8, 0)`. C14 — the round's flagship *repair*, the row that turned the it.8 unpaired
headline into a paired one — is carried by the word **`paired`**, and that node never
looked at a seed set.

**The one-line replacement is shipped, in place, inside the existing loop**
(`tests/venus/test_v20_r15_it10_venus.py:150`, located by node, not by memory —
`test_the_repair_is_present_in_the_it10_node_and_not_only_in_prose` prints the line
number it finds and fails if there is not exactly one):

```python
assert {s for (k, s) in tab if k == "arm_pl"} == {s for (k, s) in tab if k == "softmax"}, \
    "unpaired: the arms did not run on the same seeds"
```

### 1.1 It fires. The proof is a mutation that preserves both counts.

`and False` proves nothing here, and neither does mutating the code: **the old code is
correct for what it says.** The defect is only visible under a mutation of the *data* that
breaks the joint while leaving **every per-arm statistic bitwise unchanged.**
`VENUS_MUT=disjoint_softmax` relabels every `softmax` record onto `{900, 908..915}`,
disjoint from `arm_pl`'s `{0, 8..15}`; no `eval_nrmse` is touched, so both counts are still
`8` and `0`. `[RUN]`, one command, two nodes, verbatim in §4: **the count node PASSES, the
repaired predicate RAISES.** Its passing *is* the defect and it is now labelled as such in
the file.

### 1.2 The mechanism — `V-26`, and it is not `D4`

**A marginal assertion standing in for a joint claim.**

The prose claims a **relation between two collections** — *these are the same seeds*. The
assertion was over a **function of each collection separately** — the count of `A` below
`floor₁` and the count of `B` below `floor₁`. Two marginals cannot express a joint. No
value of the constants `(8, 0)` says anything whatever about `A = B`; the predicate's
*vocabulary* has no term for it.

**This is why it is a different shape from the Inspector's `D4`.** `D4` is a
**quantitative** loosening — the prose says `= 0.7176`, the node says `> 0.5`; tightening
the tolerance repairs it, and a code mutation exposes it. `V-26` is a **categorical**
loosening: **no tolerance ever repairs it, and no code mutation ever exposes it**, because
the code computes exactly what it claims to compute. It is invisible to the whole mutation
apparatus this round has been using.

**The detection rule, general and cheap:** *for any claim about a relation between two
collections, the falsifying mutation is the one that preserves every per-collection
statistic.* Marginal-preserving data mutation. If no such mutation turns the suite red, the
suite is not asserting the relation. That rule is what produced §4's RED in one attempt.

**Why this shape is worth its own row rather than a note under `D4`:** the D4 audit found
15 sites by comparing assertion tolerance against prose precision. That search is
**structurally incapable of finding V-26** — there is no tolerance to compare. The 15 D4
hits were found and this one was not, for four iterations, on the round's most-cited row.

Appended to `MISTAKES.md` as **`V-26`**, this office's own, read back at §4.

### 1.3 The pairing of the two failures is the real finding

`[CITED] V20_R15_IT12_MARS.md`, Task A-1, verified by `[RUN]` against the unmutated tree.

| | property checked | file read | state |
|---|---|---|---|
| **VENUS `it.10:147`** | crossing **counts** — not pairing | **right** file (`t="rescore"`) | **GREEN**, and green for a reason unrelated to the claim |
| **MARS `it.9 agg_vs_cells.py:67`** | **pairing** — the right property | **wrong** key (`t="cell"`, banked before MERCURY's repair) | **RED**, and red for a reason unrelated to the claim |

**One claim, two nodes, and the union of their two errors is exactly one working node.**
Neither office was watching C14. The round's instrumentation *looked* redundant on its
flagship row and was in fact absent. **The repair for MARS's half is his one-line
re-point at `results/v20_r15_it10_mercury_rescore.jsonl`; VENUS's half is shipped here.**

`[RUN]` The pairing is **true** in the data — `arm_pl [0, 8..15]` vs `softmax [0, 8..15]`,
identical — asserted at `test_the_shipped_repair_fires_on_exactly_that_mutation`. **F2 was
the right grade: unbound, not refuted. C14's number stands; nothing was watching it.**

---

## 2. TASK A — THE RE-FORECAST. THE UPDATE'S SIGN IS **DOWN** ON W3 FOR THE FIRST TIME.

> **1. W3 `arm_pl` — 2. W1 `arm_smprime`**

| ordering at it.29 | it.5 | it.7 | it.10 | **it.13** | Δ |
|---|---|---|---|---|---|
| **W3 ≻ W1** | 0.72 | 0.78 | 0.84 | **0.81** | **−0.03** |
| **W1 ≻ W3** | 0.20 | 0.14 | 0.09 | **0.12** | **+0.03** |
| no ordering — both eliminated on clause (1) | 0.08 | 0.08 | 0.07 | **0.07** | **0.00** |

**The ordering does not change. The sign of the update does**, for the first time in four
filings. Priced item by item, including the four items that moved it by **zero**.

### 2.1 `arm_smprime` seed 2, draw-checked — the only item that moved the number `[RUN]`

Bound at `test_smprime_seed_2_crosses_on_all_three_draws_and_is_the_only_cell_that_does`
and `test_the_smprime_rescore_reproduces_the_banked_cells_bitwise_16_of_16`:

- `0.203920 / 0.208055 / 0.216517`; **worst reading `0.490590` below `floor₁`**;
- **the only cell of sixteen that crosses on any draw** — nearest miss is seed 8, `0.144954`
  *above* the floor, so there is no second candidate;
- **16 of 16 bitwise control** at draw `12345` against the frozen journals;
- on **this office's own** fragility measure, `0.012597 / 0.490590 = 0.0257`, against
  `arm_pl` seed 15's `0.999` — **38.9×**, asserted in-node at `38.0 < r < 39.5`.

**Why this is a real update and not a courtesy.** At it.10 §2.6 this office wrote that W1's
mass *"never rested on rate… it rests on clause (2), BED-K(a), still unrun"* and that
**nothing that iteration touched it.** That was correct then and it is false now. Seed 2 is
**the tournament's best cell**, W1 holds it, and it has now survived the exact instrument
objection (`n_eff = 1`) that was used to discount W1's evidence while W3's was being
credited. **The asymmetry this office complained of at it.7 §3.4, which it.10 recorded as
*inverted*, is closed.** Both arms are now draw-checked; the comparison is symmetric for
the first time in the round.

### 2.2 Why the move is `0.03` and not more — the conditional the payout runs through

Seed 2's draw-check is evidence about **depth**, not **rate**. W1 is `1 of 16`; W3 is
`8 of 9`. Depth pays out **only under readings of clause (1) that are not a pure rate bar**,
and **⟨CLAUSE_1_TAIL⟩ is unruled.** So the unconditional shift is the depth-evidence shift
times the mass on non-rate readings, and that second factor is exactly the quantity the
author has not yet set. **Under a pure rate bar this iteration changed nothing for W1.**
Had the ruling been in and non-rate, the move would have been several times larger; this
office is not entitled to price the ruling and does not.

**And a second reason to hold back, which the round has not connected.** `[CITED]
V20_R15_IT12_MARS.md` B-2, `[RUN]`-verified here by re-running
`tests/mars_v20/test_it12_the_four_constants.py` — **5 passed** — the cell whose deletion
flips `ρ(beta)` from `−0.032353` to `+0.089286` is **seed 2**. **The same single cell
carries W1's entire remaining probability mass and the sign of the correlation that struck
the M1 transfer.** Those two conclusions looked independent and share one point. W1's
evidence is draw-robust (`0.0257`) *and* has `n_cells = 1`; those are different properties
and only the first improved. **A high-leverage point being also your only witness is a
reason to move less, not more, on the news that the witness is steady.**

### 2.3 The four items that moved the forecast by exactly zero, each with its reason

| item | Δ | why |
|---|---|---|
| **`ρ(beta) = −0.0324`'s sign is one cell** | **0.00** | It is not in a clause of this ranking. It carries C11 and the M1-transfer strike, neither of which orders W1 against W3. **The ordinal evidence MARS names as untouched — best cell `beta +1.344`, worst `+1.990` — is the part this office ever read.** Its *leverage structure* moved §2.2; its *sign* moves nothing here. |
| **Q2/W1 re-graded `F1 + const` → `F4`** | **0.00** | A **grade**, and this ranking is scored at it.29 **against clauses**. The *measurement* under it — Q2's `ceq` surface is `ceq.hankel` only, a BED-K object cited against BED-M wings — is evidence about a supporting argument's admissibility, not about clause (1) or (2). See §3. |
| **`1.084523` withdrawn, true value `1.0845223424`** | **0.00** | A hand-subtraction of two six-decimal displays, on W3's side, correctly withdrawn. It never entered a clause. **The withdrawal is against W3 and this office's number did not move toward W1 on it either** — a constant nobody's forecast read is a constant whose repair nobody's forecast owes anything to. |
| **`L-GRADE` does not exist** | **0.00** | §3. |

### 2.4 One prior pre-registration is unscored, and this office is not claiming it

it.10 §2.2 pre-registered *seed 15 flips first, then 12*, against a fourth eval draw.
MERCURY's Experiment B is the only fourth draw that exists and **failed its own control on
9 of 9 `arm_pl` cells** (`instrument_hash 945d850d…` against the round's `5d41a63d…`), and
he declined to score this office on a broken run. **That pre-registration therefore stands
UNTESTED — neither held nor failed — and it is not counted in §5.** A forecaster whose
killer never got a draw has no result to bank.

### 2.5 The falsifier, complement form over the full outcome space, dated 2026-09-02

**W3 ≻ W1 at `0.81` is wrong if any of —**

<!-- CLAUSES-BEGIN -->
1. eight fresh `arm_pl` seeds `16–23` land the pooled count **below `18/24`** below `floor₁`; **or**
2. **any** eval draw returns an `arm_pl` crossing count other than `8/9` on the rescored cells; **or**
3. `arm_smprime` returns **≥ 3** crossings on any eight fresh seeds; **or**
4. BED-K(a) runs and W1's seed-2 statistic separates the arms in W1's favour; **or**
5. seed 2 is re-scored on a fourth draw with a passing bitwise control and **stops crossing**,
   which removes W1's only witness and sends this `−0.03` back and further.
<!-- CLAUSES-END -->

Clause 5 is new and is the complement of §2.1: the same measurement that moved the number
down is registered as the thing that can move it back up. Each clause is a set complement
over an observable that exists on every cell — the `M-7` repair, held at it.8 and it.10.

**Stated once, not re-litigated:** `⟨AUTHOR_COUNTER_RANKING⟩` is **NOT MEASURED**, so under
**D-CALIB-2** this ranking is formally unreadable and it.29 cannot open a VENUS row.
**D-CALIB-3 authorises sign, never size** — the `−0.03` is a **sign statement with a
magnitude attached for auditability**, not a calibrated quantity.

---

## 3. TASK B — THE FORECAST DOES NOT READ A GRADE, AND THAT IS NOW TEST-BOUND

**Plainly: no. This ranking has no dependence on the `F0–F4` scale, in any clause, at any
filing, from it.5 forward.**

The five clauses in §2.5 range over exactly two kinds of object: **`eval_nrmse` against
`floor₁` on a named cell**, and **a bed that has or has not been run**. Both are readable
off `results/*.jsonl` and `scripts/`. **Not one clause names an `F`-token, a
LEAPABLE/TERMINAL verdict, a ledger row, or a contract law.**

That is asserted rather than claimed: `test_the_forecast_never_reads_an_F_grade` extracts
the delimited block above out of **this file** and fails on any `\bF[0-4]\b` inside it. If
a future filing smuggles a grade into a clause, the node goes red on the report itself.

**What this buys, stated at the size it is worth.** The it.35 leap gate sorts F1/F2/F3;
five `F4` rows already carry LEAPABLE/TERMINAL verdicts; `M14` is `F3` in the contract and
`F4` in the ledger; and the `+6` and the annex's `+4` both require counts over a scale with
no text. **All of that is blocked. None of it is upstream of this ranking.** So the ranking
is scoreable at it.29 on a repository whose grading law is missing — which appears to make
it the one artifact in the round that survives the absence.

**Two qualifications, because "survives the missing law" is a strong claim.**

1. **Survival is not evidence of quality.** The ranking is grade-independent because it was
   written against measurements, and that was a construction choice made at it.5 for a
   different reason. It is not foresight about `L-GRADE` and is not claimed as such.
2. **The dependence that *does* exist is on a different missing law.** The ranking is
   blocked on **⟨CLAUSE_1_TAIL⟩** — an unruled convention, not a missing rubric — and on
   **⟨AUTHOR_COUNTER_RANKING⟩** being NOT MEASURED. **Two of three items on this office's
   critical path are author rulings; the third is a bed run.** Immunity to the missing
   grading scale does not make the forecast unblocked. It makes it blocked on things that
   are cheaper.

---

## 4. THE NODES — RED FIRST, VERBATIM, AGAINST UNMUTATED CODE

`[RUN] VENUS_MUT=disjoint_softmax python -m pytest tests/venus/test_v20_r15_it13_venus.py -q -k "count_pair or shipped_repair"`

```
.F                                                                       [100%]
    def test_the_shipped_repair_fires_on_exactly_that_mutation():
        tab = _rescore()
        if MUT == "disjoint_softmax":
            assert not _paired(tab)          # the repair would raise -> RED
>           raise AssertionError(
E           AssertionError: unpaired: the arms did not run on the same seeds - arm_pl [0, 8, 9, 10, 11, 12, 13, 14, 15] vs softmax [900, 908, 909, 910, 911, 912, 913, 914, 915]
1 failed, 1 passed, 5 deselected in 2.07s
```

**Read the `1 passed`.** That is `test_the_count_pair_is_blind_to_a_disjoint_seed_block` —
the shipped it.10 predicate, on the same mutated data, still returning `(8, 0)` on all
three draws. **The two lines together are the strike and the repair in one run.**

Unmutated:

```
python -m pytest tests/venus/test_v20_r15_it13_venus.py -q      ->  7 passed in 0.94s
python -m pytest tests/venus/ -q                                -> 47 passed in 4.21s
python -m pytest tests/mars_v20/test_it12_the_four_constants.py
                tests/jupiter/test_v20_r15_it12_constants.py -q -> 11 passed in 14.28s
```

The last line is this office **re-running the adversary's and the sibling's constant nodes
rather than citing their reports** — MARS's `ρ(beta)` sign-flip node and JUPITER's
`1.084523` withdrawal are green on this tree, at this hour.

Node printouts:

```
pairing assertion lives at test_v20_r15_it10_venus.py:150
smprime seed2 draws=[0.20392, 0.208055, 0.216517] worst margin below floor=0.490590
fragility: smprime seed2 0.0257 vs arm_pl seed15 0.999 = 38.9x
bitwise control at draw 12345: 16 of 16
clause block, zero F-grade tokens
```

**Evidence classes.** `[READ]` `tests/venus/test_v20_r15_it10_venus.py:147-155`;
`V20_R15_IT12_MARS.md` Task A-1, B-2; `V20_R15_JOURNAL.md:2008-2125, 2149-2339`;
`results/v20_r15_it11_mercury_smprime.jsonl`. `[RUN]` the seven nodes above, the
mutation, and the two sibling suites. `[DERIVED]` the fragility ratio, the `38.9×`, the
`−0.03`. **No `GUESS` is asserted anywhere in this report.**

---

## 5. TASK C — THE FORECASTER LEDGER, SIGNED, WITH THIS OFFICE'S OWN `F2` ON IT

Three rows are scored here as required, then the round's new instrumentation row.
**Signed: VENUS (IRENE), 2026-09-02.**

| # | office | object | filed | outcome | **could the killer fire?** | grade |
|---|---|---|---|---|---|---|
| **4** | **VENUS** | `5 of 8` crossings, PI `2–7` | it.7 | **7 of 8**, top edge, **inside** | **YES** — `0/8` and `8/8` were both reachable and both outside | **HELD** |
| **5** | **VENUS** | `frac_gate_annihilated == 0.0` on `8 of 8` | it.7 | `9 of 9`, then **54/54 readings × 3 draws** | **YES, and it did not** | **HELD — strongest form** |
| **2** | **VENUS** | mechanism: `frac ∈ {0.4967, 0.5033}` | it.5 | **FALSIFIED** at seeds 11, 13 | **NO — blind** (`M-7` HOLE: the falsifier did not range over the outcome space) | **FAILED, and the falsifier failed worse than the prediction** |
| **N** | **VENUS** | **`tests/venus/…it10_venus.py:147-151` certifying C14** | it.10 | **F2 — unbound, not refuted** | **NO — structurally.** A marginal predicate cannot fire on a joint claim under *any* data | **DEFECTIVE (`V-26`), repaired it.13, repair demonstrated RED** |
| **N+1** | **MARS** | his own it.9 pairing node, `agg_vs_cells.py:67` | it.9 | **RED now**, against `t="cell"` where MERCURY banked `t="rescore"` | checks the **right** property against the **wrong** file | **CORRECT PROPERTY, VOID READING** |

**The two-office row, which is the one worth keeping.** Rows N and N+1 are **the same
claim**. VENUS shipped a node that is green for a reason unrelated to C14; MARS shipped a
node that is red for a reason unrelated to C14. **A green node without the check and a red
node against the wrong file, on the round's flagship correction, for four iterations.**
Neither state is *wrong-looking* from the outside — a passing suite and a known-red dispute
both read as instrumented. **The failure mode is that redundancy was assumed from the
existence of two nodes rather than measured from what either asserts.**

**The self-score across rows 2, 4, 5 and N.** Rows 4 and 5 held with killers that could
fire. Row 2 failed with a killer that could not. **Row N is a third category and it is the
worst of the three**: not a prediction that failed, and not a killer blind to *some*
outcomes — **a killer blind to every outcome.** Rows 2 and N share one ancestor: in both,
the object asserted was **narrower than the object claimed**, and in both the gap was
invisible because the assertion was true. `M-7` was that gap in a *prediction*; `V-26` is
that gap in a *certification*. **This office has now made the same class of error on both
sides of the pre-registration discipline it enforces.**

**Unscored and named:** it.10 §2.2's fragility pre-registration — no valid fourth draw
exists (§2.4). **`⟨AUTHOR_COUNTER_RANKING⟩` remains NOT MEASURED.** Stated once.

---

## 6. WHAT it.14 OWES, PRICED

1. **⟨CLAUSE_1_TAIL⟩** — still the largest item, and now the *only* thing between this
   ranking and a scored clause (1). **0 GPU-s.**
2. **⟨L_GRADE_RUBRIC⟩** — blocks the `+6`, the annex `+4`, and the it.35 gate. **Does not
   block this ranking** (§3). **0 GPU-s.**
3. **MARS's it.9 node re-pointed** at `results/v20_r15_it10_mercury_rescore.jsonl` — the
   other half of the C14 repair, and it goes green on a true fact. **0 GPU-s.**
4. **A marginal-preserving mutation sweep** over every node that certifies a *relation*
   between two collections. `V-26` was found by an adversary on one row; the D4 audit
   could not have found it, and nothing has looked for its siblings. **0 GPU-s.**
5. **A fourth eval draw with a passing bitwise control**, which scores it.10 §2.2 and
   clause 5 of §2.5 at once.
