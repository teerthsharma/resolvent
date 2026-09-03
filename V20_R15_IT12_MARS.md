# V20 R15 - it.12 - MARS (MORIARTY, standing adversary)

**Standing:** three of four it.9 strikes upheld, one repaired by measurement. This
iteration attacks the round's **test discipline** (A), the **four unasserted constants**
(B), and the **corrections index** (C).

**Evidence classes:** `RUN` / `READ path:line` / `CITED` / `DERIVED`. No `GUESS` asserted.
**No `and False` red anywhere in this filing.** Every red below is a **data mutation**
applied to a copy of journalled data, against unmutated code.

**Search proof, same style as the strikes:** `grep -rn "1\.084523" --include=*.md .`
returns `V20_R15_IT6_JUPITER.md:426,427,494` - the three prose hits the Inspector names
independently at `V20_R15_IT11_INSPECTOR.md:59`. The instrument finds what is there.

---

## TASK B - THE ADJUDICATION HOLDS, ON MARS'S OWN ARITHMETIC

The brief's condition: *"If they hold, say so and the adjudication stands on your
arithmetic rather than on an unbound shell command."*

**They hold. To every digit published.** `[RUN]`

```
n = 16   seeds [0..15]
beta         rho=-0.032353  p=0.905320
qk           rho=+0.717647  p=0.001748
abs(beta-0)  rho=-0.032353  p=0.905320
BEST  seed 2 nrmse 0.203920 beta +1.343933
WORST seed 13 nrmse 1.203324 beta +1.989505
```

Against `V20_R15_JOURNAL.md:1408-1412`, character for character. **The it.8 adjudication
of JUPITER over VENUS - corrections-index row C11 - is not decided by a number nobody can
reproduce.** MARS reproduced it from a **wider source set** than the original
(`results/v17k_r4_floor.jsonl` + `retake` + `v20_r15_it6_seeds8_15.jsonl`, three journals
where the it.7 loader reads two) and with **scipy**, where JUPITER's concurrent it.12 node
(`tests/jupiter/test_v20_r15_it12_constants.py:82-97`) hand-rolls midrank Pearson on
numpy. **Two implementations, two source sets, agreement at six decimals.**

Node: `tests/mars_v20/test_it12_the_four_constants.py::test_it8_adjudication_spearman_recomputed_exactly`

**RED FIRST, by data mutation, verbatim** (`MARS_MUT=swap_qk` exchanges two cells' `qk`
values; the assertion is untouched):

```
        assert n == 16
>       assert round(qk.correlation, 6) == 0.717647
E       assert 0.1 == 0.717647
E        +  where 0.1 = round(0.09999999999999999, 6)
E        +    where 0.09999999999999999 = SignificanceResult(statistic=0.09999999999999999, pvalue=0.7125163701282782).correlation
```

`+0.7176 -> +0.1000`. The node bites.

### B-2. THE STRIKE THAT IS LEFT: the SIGN of `-0.0324` is one cell

`-0.0324` is load-bearing. It carries C11 (`V20_R15_JOURNAL.md:47`) and section 2.2's
strike of the M1 transfer (`V20_R15_IT7_JUPITER.md:113`). The Inspector priced it as *"a
failure to reach significance"*, Fisher-z 95% CI `[-0.520, +0.471]`.

**It is weaker than that.** `[RUN]` Drop **one** cell - seed 2, the single crossing cell
the round treats as exceptional in every other section:

```
EX-SEED2 beta rho=+0.089286 p=0.751673   (n=15)   <-- SIGN FLIPS
EX-SEED2 qk   rho=+0.657143 p=0.007770            <-- survives, still p<0.01
```

**The sign of the number that struck the M1 transfer is decided by a single point.** `qk`
is robust to the same deletion; `beta` is not. This does **not** refute C11 - C11's
ordinal evidence (best cell `beta +1.344`, worst `+1.990`, corner cell `-0.060`
second-worst) is a fact about ranks and is unaffected. **What is void is the use of
`rho = -0.032` as if it were a measurement.**

Node: `::test_the_beta_rho_sign_is_decided_by_a_single_cell`

### B-3. THE DEDUP IS WORTH 0.179069 OF THE HEADLINE

`[RUN]` The banked record carries **20** `smp_values` rows, not 16. On the raw 20:
`rho(qk) = +0.538578`, `rho(beta) = -0.154312`. The headline moves **0.179069**.

The dedup is **correct** and MARS certifies it: the seed-0 and seed-1 duplicates are
**bitwise identical across all three journals**, now asserted inside `_cells()` rather
than assumed - so the retake-wins precedence in the it.7 loader is immaterial, a fact
JUPITER's node does not establish and this one does. It was disclosed in prose
(`V20_R15_JOURNAL.md:1406`). Node:
`::test_the_dedup_is_an_unbound_preprocessing_choice_worth_0_18_of_rho`

### B-4. THE ATTACK THAT DID NOT FIRE - AND WHO CLOSED IT

**The four-unasserted-constants strike is dead, killed inside its own iteration.**
`tests/jupiter/test_v20_r15_it12_constants.py` (258 lines, landed it.12) binds all three.
It also **withdraws `1.084523`**: measured `1.0845223424`, so the prose's last digit is
wrong (`:138-181`). MARS's node is kept, reframed as a standing guard that goes RED if
that file is deleted. **Credit to JUPITER; the debt was real and it is paid.**

---

## TASK A - THE SLACK AUDIT

**One target fired of five.** The four that did not are reported, including clause (1)
CP - which the brief named and which this office audited and **cleared** rather than
leaving flagged.

### A-1. STRIKE - C14's repair says **paired**; the node certifying it never checks pairing

C14 (`V20_R15_JOURNAL.md:51`) is the round's model correction: the it.8 headline was
**unpaired**, `p = 6.730e-04` against an honest `p = 0.012821`, *"a factor of 19"*. The row
closes: *"Now repaired by measurement - **8 of 9 vs 0 of 9, paired**, three draws."*
MERCURY states it as *"same seeds, same draws, same process"*
(`V20_R15_IT10_MERCURY.md:202-203`).

**The word carrying the entire correction is `paired`. The binding node does not assert it.**

`[READ] tests/venus/test_v20_r15_it10_venus.py:147-151`:

```python
for e in (12345, 12346, 20260902):
    pl = sum(1 for (k, _), v in tab.items() if k == "arm_pl"  and v[e] < FLOOR1)
    sm = sum(1 for (k, _), v in tab.items() if k == "softmax" and v[e] < FLOOR1)
    assert (pl, sm) == (8, 0), (e, pl, sm)
```

Two **independent** counts over whatever seeds each kind happens to carry. **Nothing
compares the `arm_pl` seed set to the `softmax` seed set.** `[RUN]` Rebuild the shipped
`_rescore()` and counting logic against a mutated copy of the journal in which every
`softmax` record's seed is moved to a disjoint block (`900,908..915` against `arm_pl`'s
`{0,8..15}`) - pairing is now **false by construction** - and:

```
python -m pytest tests/mars_v20/test_it12_slack_mutations.py -q
..                                                                       [100%]
2 passed in 0.22s
```

**The shipped assertion still reads `(8, 0)`.** Its greenness *is* the strike. This is
**exactly the D4 mechanism** the Inspector found in 15 places - an assertion looser than
the prose it certifies - landing on **the round's flagship correction**.

**And this office does not overstate it. `[RUN]` The pairing is TRUE in the data:**

```
arm_pl  [0, 8, 9, 10, 11, 12, 13, 14, 15]
softmax [0, 8, 9, 10, 11, 12, 13, 14, 15]
IDENTICAL SEED SETS: True
```

**Grade F2 - unbound, not refuted.** C14's number is right. Nothing in the suite would
notice if it stopped being right.

**REPLACEMENT ROUTE, one line, inside the existing loop:**

```python
assert {s for (k, s) in tab if k == "arm_pl"} == {s for (k, s) in tab if k == "softmax"}, \
    "unpaired: the arms did not run on the same seeds"
```

**The corroborating fact the round has not connected.** MARS's own it.9 node
`tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py:67` **does** check pairing, and it is
**RED right now on the unmutated tree**:

```
E  AssertionError: softmax cells exist only for seeds [0, 1, 2, 3, 4, 5, 6, 7]; the fresh
E  seeds [8, 9, 10, 11, 12, 13, 14, 15] have no softmax control anywhere in results/
```

It is red because it reads `t="cell"` records while MERCURY's repair was banked as
`t="rescore"` in a new file. **The round holds a green node that certifies the headline
without pairing, and a red node that checks pairing against the wrong journal. Neither is
watching the claim.** Re-point the it.9 node at
`results/v20_r15_it10_mercury_rescore.jsonl` and it goes green on the true fact.

### A-2. ATTACKS THAT DID NOT FIRE

| target | binding node | why it did not fire |
|---|---|---|
| **seed 2's crossing** `0.203920` | `tests/venus/test_v20_r15_it5_ranking_record.py:97` | **Faithful.** `pytest.approx(0.203920, abs=5e-7)` - the value, not a boolean, at a tolerance narrower than the printed digits. |
| **`FREEZE-SHA256`** | `tests/saturn/test_v20_r15_freeze_manifest.py` | **Faithful.** `assert m.group(1) == digest(rows())` - real `hashlib.sha256` recomputed from the manifest text, not a stored string compared to itself. Its sibling `test_moving_one_citation_by_one_line_fires_both_binds` proves the digest moves on a one-line tamper. |
| **`12 of 16`** (C13) | `tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py` | **Not eligible.** The node is **already RED** on the unmutated tree - `assert (len(crossed) > len(cells)/2) == bool(agg["crosses"])` is catching the n=9 / n=8-ex-seed-9 flip. A red node cannot be attacked by "green survives mutation". The dispute is live and correctly instrumented. |
| **clause (1) CP figures** | `tests/venus/test_v20_r15_it5_ranking_record.py:42-47,75-89` | **Faithful - audited in the last minutes and cleared.** `cp_lower` is a real `scipy.stats.beta.ppf` recompute, and critically `:76` derives the count from the data (`x = sum(1 for c in cells[kind].values() if c["eval_nrmse"] < FLOOR_1)`) rather than hardcoding it - the D4 shape MARS was hunting. The literal `cp_lower(7, 8)` / `cp_lower(8, 8)` at `:87-89` is a statement about the *estimator* (only a clean sweep clears `0.5`), not about the data, and is correctly scoped as such. **One residual, named not struck:** `n = 8` is a literal at every call site while the round now scores **9** cells; if the cell count moves, the node computes the wrong CP silently. Replacement: `cp_lower(x, len(cells[kind]), ...)`. |

---

## TASK C - THE CORRECTIONS INDEX

### C-1. THE BRIEF'S TWO NAMED GAPS ARE CLOSED. THE PREMISE IS STALE.

`[READ] V20_R15_JOURNAL.md:52-53`. The index runs **C1..C17**, not C1..C15, and the two
rows the brief says are missing **both exist**:

- **C16** covers the it.9 restatement, and covers it harder than the complaint asked -
  *"MARS filed the report breaking it at 03:21:02; this record's mtime is 03:21:06. **Four
  seconds.**"* It also names its own failure: *"C13 and C14 point at it.8 and missed this
  restatement entirely"*, corrected **it.11 - INSPECTOR, on the index itself**.
- **C17** covers the GPU timing - no `torch.cuda.synchronize()`, run-order `rho = +0.7029,
  p = 0.0024` against the gate's `+0.5197`, and *"this office kept quoting the ratios
  anyway"*.

**Two corrections against the brief this office was handed.** The brief also cites the
it.9 restatement at `V20_R15_JOURNAL.md:1736`; that line is *"The coordinator's call on
the three columns"*. The restatement is at **`:1745-1747`**.

### C-2. C9 READS STRONGER THAN ITS TARGET - CONFIRMED

The Inspector's flag holds. C9's closing clause - *"so something drives it past the
ceiling and what that is remains unknown"* - is said by **neither** target. it.8's
CORRECTION 8 (`V20_R15_JOURNAL.md:1432-1459`) says only *"Not determined... No journal
stores `u`"*. it.9 MARS (`V20_R15_IT9_MARS.md:409-421`) then **measures it and closes it**:
*"`a_hat_max == 1.0` is saturation, not coincidence, and it is now measured rather than
inferred"*, with a closed-form mechanism (pre-clamp `u` at step 0 is deterministic in
seed + pinned eval batch through the `g==1.0` lerp identity). **The index preserves a
mystery its own target dissolved.** Per the index's own governing rule - *"the entries
below are the record; this table is only an index into it"* - the clause must go.

### C-3. TWO ROWS ARE GENUINELY MISSING, AND ONE IS THE MOST CONSEQUENTIAL SENTENCE OF it.2

Verified by `[READ]`, and `grep` over the whole table `:34-54` returns **zero** mentions of
either:

- **`N = 1 primitive`.** Filed `V20_R15_JOURNAL.md:339-355` (it.2, JUPITER's merge
  verdict); struck at `:499-501`: *"**`N = 1 primitive` is STRUCK as UNBOUND.** The merge
  verdict - the single most consequential sentence of it.2, and the one this office
  propagated into the it.2 record - has no node asserting it."* **A verdict this office
  propagated, struck as unbound, with no index row.**
- **`Every pair separates by >= 0.30`.** Filed `:357-376` (it.2, SATURN's distinctness
  measurement); withdrawn at `:533`: *"**'Every pair separates by `>= 0.30`' is false and
  is withdrawn.**"* **An explicit withdrawal of a published constant, with no index row.**

Both are structurally identical in kind to C1..C17 - an early published verdict later
struck or withdrawn by the room. **The index closed the debt for it.3 onward and left
it.2's two headline claims on the page.** A reader who opens it.2 and stops there reads
both.

### C-4. A CITATION IN AN APPEND-ONLY FILE HAS DRIFTED

C16's own embedded quote (`V20_R15_JOURNAL.md:2047-2048`) cites the restatement as living
at `V20_R15_JOURNAL.md:1694`. Line 1694 today is *"`dist_to_skyline` is `None` on 34/34"*
(CORRECTION 11). **In a file whose first invariant is append-only, an internal line
citation should not move.** Either the citation was wrong when written, or the invariant
is not being held. This office does not have the evidence to say which and does not
assert one. **It is the right object for it.13, and the fix is cheap: the index sits above
the append marker in an untracked file with no hash - give it one.**

### C-5. NOT AUDITED

C1, C2, C3, C11, C12 were skim-checked only and are **flagged unverified, not passed**.
C16 was not checked word-for-word against its own target. C4-C8, C10, C13-C15, C17 were
verified against target text and read **faithful**.


### C-6. INCIDENTAL - THE EVENT LOG ITSELF

`[RUN]` Found while validating this filing's own logging. `house-events.jsonl` holds
**12791** non-empty lines. **4 are unparseable** by `json.loads` (invalid backslash
escape) at lines **1899, 2937, 2938, 5871**, and **116 more carry no `t` field** (round-4
`chase` events use `kind` instead). Any audit that iterates this file with `json.loads`
either crashes on line 1899 or silently drops those records. **The round cites this file
as evidence and 120 of its records are not readable as records.** MARS's own 13 it.12
events parse and every one carries `t`.

---

## STANDING

**Struck this iteration:** two - C14's `paired` (F2, unbound not refuted, one-line
replacement shipped); C9's closing clause (stronger than its target, must be deleted).
**Missing rows filed:** two, both from it.2 - `N = 1 primitive` and `>= 0.30`.
**Upheld against MARS:** the it.8 Spearman adjudication (C11), reproduced exactly on
independent arithmetic and an independent source set. **It stands.**
**Closed by a sibling mid-iteration:** the four-constants absence (JUPITER).
**Left standing, named:** C1-C3/C11/C12/C16 unverified; the
`:1694` citation drift; 120 unreadable `house-events.jsonl` records.

**Nodes shipped**
- `tests/mars_v20/test_it12_the_four_constants.py` - 5 nodes, `5 passed in 1.57s`, RED
  under `MARS_MUT=swap_qk` (data mutation, verbatim above)
- `tests/mars_v20/test_it12_slack_mutations.py` - 2 mutation demonstrations,
  `2 passed in 0.22s`, green **as the strike**
