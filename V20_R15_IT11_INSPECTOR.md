# V20 R15 — it.11 — HEALTH INSPECTOR

**Office:** the LOG. Authority is over whether a claim is **BOUND**, never whether it is
**correct**. A struck cell is not a refuted result; it is a result the record does not
yet carry.

**Task set by this office's own it.8–9 ruling:** *"Four of twelve cells were not reached
at all"* … *"audit Q2–Q5. Then the `+6` is earned on the merits it already has."*
JUPITER took the repair half. This is the other half.

---

## 1. THE FOUR CELLS NOT REACHED — Q2 THROUGH Q5

Each cell was put to four questions: **RED first?** · **does the HOW-BAD gap cite
something that exists?** · **does the cited test assert what the prose claims** (the D4
question) · and, where the claim is a constant, **is the constant algebraically forced?**

### Q2 / W1 — `R2_1·d = 1.000000000000000`, spread `0.000e+00` — **STRUCK, VACUOUS**

- **RED is manufactured.** `V20_R15_IT6_JUPITER.md:431-439` labels its own red *"the
  annex-comparison assertion deliberately conjoined with `False`"*; the traceback reads
  `E assert (0.0 < 1e-06 and False)`. The real predicate was **already true, diff exactly
  `0.0`** — only the appended `and False` failed. Mirrored at `house-events.jsonl:12119`.
  The run line is `1 failed, 3 passed`, and the node carrying the `R2_1·d` claim —
  `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:108` — is **among the three that
  passed**. It was never red.
- **D4 — the tolerance is nine orders wider than the printed number.**
  `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:125`
  `assert spread < 1e-6, "R2_1 * d products not constant: %r" % products`
  The prose prints `1.000000000000000` and `0.000e+00`; the assertion licenses anything
  under `1e-6`. Worse, the constant the grade is **named for** — `err_1 = 0.9746794345` —
  has no numeric assertion at all. `:105` is `assert np.isfinite(err_1)`.
- **The finding is arithmetic on an identity matrix.** `_delay_hankel` (`:63-72`) builds
  `H = I_d` with a zero row on top, so `HᵀH = I_d` exactly, every singular value is
  `1.0`, and `R2_k = k/d` **exactly, for every `d`, forever**. `err_1 = sqrt(1-1/20) =
  sqrt(0.95) = 0.9746794345` is a closed form. Zero spread across `{5,10,20}` is not
  evidence for the construction; it is a consequence of it, and would hold at any `d`
  never run. **`CEQ_V20_R15_CONTRACT.md:249` already says so: `F1 by construction.`**
- The HOW-BAD gap is the bare string **`§8`** (`V20_R15_IT6_JUPITER.md:310`, copied into
  `V20_R15_JOURNAL.md:1165`) — a section reference standing in for a gap, which this
  office already struck at `V20_R15_IT89_INSPECTOR.md:620-630`.

### Q2 / W3 — the nonexistence — **STRUCK; the theorem survives, the cell does not**

- **The nonexistence itself is sound, and it is not the test's doing.** `ceq/hankel.py:294-299`
  returns `NEG_ENTRY` on `H.min() < 0`. "Any size" ranges over the **inner dimension of the
  nonnegative factorisation**, not over block size, and `nonneg × nonneg ⇒ entrywise
  nonneg` is a one-line theorem that lifts a finite negative entry to the infinite Hankel.
  **This office records the mathematics as correct.** What it will not record is the test
  as evidence for it: the W3 block is a **single hardcoded `n = 4`** matrix
  (`tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:139`), no size loop anywhere, and
  the target `_sign_alt` (`:129-134`) has codomain `{±1}` **by construction**, so
  `assert H.min() < 0` cannot fail. Both asserts are identities.
- **`H[0,2] = −1.0` is never asserted** — `:152` *prints* it; the only assertion (`:153`)
  is a sentinel identity `bound.bound is NEG_ENTRY`. And the value is the return of a
  hardcoded literal `val *= 1.0 if c == "+" else -1.0` (`:133`). `rank_R = 1` is likewise
  forced: `H[u,v] = f(u)f(v)` is an outer product for any multiplicative series.
- **`1.084523` does not exist in any Python file.** Repo-wide, three hits, **all prose**
  (`V20_R15_IT6_JUPITER.md:426,427,494`). It is a hand-subtraction of two printed numbers.
  The cited test's terminal assertion is `assert len(viol) >= 1`
  (`tests/jupiter/test_v20_r15_it6_q1_exact_class.py:173`).
- **D4 again, verbatim.** `:208-209` prints `0.5303282490` to ten digits and asserts
  `residual > 0.1`. A deliberately wrong PAVA (unweighted pooling) yields
  **`0.7070481503` — off by 33% — and passes.**
- **The planted control is an identity.** `:216-223` sets `t = x**2` and `z = (x**2)[perm]`,
  so `z` and `t_perm` are **bitwise the same array**; `_isotonic_fit` sorts by the feature,
  the PAVA pooling loop never executes, and the residual is zero by construction **for any
  implementation whose pooling logic is broken**. The wrong PAVA passes *both* asserts.
  The seed-0 permutation is decorative — `argsort` at `:194` undoes it exactly, so the
  report's justification at `:214-215` is false.

### Q3 / W3 — `lambda_hat < 0 ⟺ nrmse < 0.7`, margin `0.451211` — **STRUCK on provenance and on D4; the separation is real**

- **The threshold `0.7` is not preregistered.** It appears in this role in exactly two
  places, both authored at it.7, both after the data: `V20_R15_IT7_JUPITER.md:171` and
  `tests/jupiter/test_v20_r15_it7_q3.py:110`. **The repo's preregistered bar is
  `floor₁ = 0.7071067811865476`** (`CEQ_V20_R15_CONTRACT.md:116`) and it is used **twelve
  lines away in the same file**, at `:78`. The author had the registered constant in hand
  and wrote a different number for this one claim.
- **This office declines to call the claim rescued, and also declines to call it dead.**
  The admissible-threshold window is `(0.6621282051474511, 1.1133392329955414]`, width
  `0.4512110278480903` — the preregistered `0.7071…` sits inside it, and substituting it
  leaves the claim **8/8 and 16/16**. The defect is provenance. But it cuts the other way
  too: because a continuum 0.45 wide works, **"8/8 at 0.7" carries no information about
  `0.7`**, and `0.451211` is a restatement of the bimodality, not a fitted separator.
- **D4.** `:110` genuinely asserts the biconditional, both directions. But `:111-112`
  certify only `> 1.11 − 0.67 = 0.44`, while the prose (`V20_R15_IT7_JUPITER.md:200`)
  calls `0.451211` *"the constant"*. **Slack `0.011211`** — the same defect, in the same
  file, nine lines below the line the round already struck as D4
  (`V20_R15_IT89_INSPECTOR.md:669-677`, STRIKE I-14). The n=16 restatement
  (`tests/jupiter/test_v20_r15_it9_q6.py:193-210`) **asserts no margin at all.**
- **The `2^-8` reading is wrong.** Class balance is **5 neg / 3 pos** over 8, so the
  permutation null is `1/C(8,3) = 0.017857` — **4.6× weaker** than the `0.0039` that a
  balanced reading implies. Over 16 it is 12/4, `1/C(16,4) = 0.00054945`.
- **The margin shrank `0.451211 → 0.432761`** and both figures are recomputed exact from
  the JSONL. The shrinkage is caused by a **new negative raising the low ceiling** — not
  by a point landing in the gap — so it is *not* the fitted-threshold signature. It is
  disclosed (`V20_R15_JOURNAL.md:1480`) and nowhere interpreted, and the stale n=8 figure
  still stands at `V20_R15_LEAP_LEDGER.md:27`.
- **The honest grade:** the it.7 filing is **F2 + no constant**. The F1 belongs to the
  it.9 n=16 node, which is stated at the **preregistered** floor and confirmed on 8
  unselected cells — and which still asserts no margin.

### Q3 / W1 — `qk` partitions 16 cells, Spearman `+0.7176` vs `beta` `−0.0324` — **STRUCK; and this office is struck with it**

- **RED is genuine here, and this office says so.** `V20_R15_IT7_JUPITER.md:246-252` carries
  a real verbatim failure — `E assert (12 == 13)` at `tests/jupiter/test_v20_r15_it7_q3.py:83`
  — on a bucket-membership count, with `house-events.jsonl:12146` logging `"red_first": true`,
  and the `qk_band` mutation reaches the ordering assertion. **This is the only true
  red-then-green in the four cells.** But it is a red on bucket counts, never on a
  correlation.
- **D4, one rung worse than D4.** D4 was a test *wider* than the prose. Here **there is no
  test at all.** `grep -rn spearman` over `tests/`, `scripts/`, `ceq/` returns nothing for
  these cells; the word `beta` appears **zero times** in the whole test file. The cited
  `[RUN]` node (`:75-85`) is two inequalities on block extrema. The report tags the numbers
  `[DERIVED]` at `V20_R15_IT7_JUPITER.md:94` and then attaches `[RUN]` on the *next line*,
  so a reader takes the run as certifying the table at `:91-92`. It certifies neither cell.
- **`−0.0324` is load-bearing and measures nothing.** It carries the §2.2 strike of the M1
  transfer (`:113`, *"the number that kills it is ρ = −0.032"*) and §2.3's "MISNAMED"
  ruling (`:135`). Recomputed: `ρ = −0.032353, p = 0.905320, n = 16`; two-sided 5% critical
  `|ρ|` at n=16 is `0.4973`; **Fisher-z 95% CI is `[−0.520, +0.471]`**. A true `beta`
  ordering of `ρ = +0.45` is entirely consistent with this measurement. It is a failure to
  reach significance at n=16, **not a measurement of absence** — and the report carries
  **no p-value, no CI, no power statement, no n-caveat** anywhere in the file. The office
  knows how to state the caveat; it states it one section later for the other wing (`:180`).
- **"Zero overlap, sixteen of sixteen" is FALSE.** The test asserts two inequalities
  spanning `1 + 12 + 2 = 15` cells. **Seed 3, `qk = 1.853493`, is in no block and in no
  assertion.** The report contradicts itself in six lines: `:103` lists exactly the two
  inequalities (seed 3 absent), `:104` declares sixteen of sixteen, `:100` draws seed 3 as
  its own group (four regimes) while `:91` says three.
- **THIS OFFICE PROPAGATED IT.** `V20_R15_IT89_INSPECTOR.md:660` repeats *"qk partition
  with zero overlap 16/16"* verbatim, without opening the test. **That is the second time
  this round this office has repeated a claim without opening the file it rests on** — the
  first is C12 in the corrections index. **Self-strike, and it is the same mechanism.**
- What *is* real: all 16 qk values are distinct, but the blocks are defined by
  `frac_gate_annihilated`, not by qk, so "zero overlap" is a genuine and
  mutation-sensitive claim that qk-rank respects an outcome-side blocking — and it holds
  on the 15 cells asserted. `+0.7176` is real (`p = 0.001748`) and also asserted by nothing.

### Q4 / W1 and Q4 / W3 — `s = 64` on 40 of 40, `S` a module constant — **THE GRADE HOLDS. This office confirms F3 by its own hand.**

This was the one cell the brief singled out, on the ground that if true *"the grade is not
a judgement — it is a fact about the instrument."* **It is true.**

`scripts/v15_r1.py:137`, read directly:

```python
S, D = 64, 24                  # the shape every e3 row in results/ uses
```

A bare module-level tuple assignment. **`S` is not a flag.** The full `add_argument` block
of the same file is `:547-558` and contains **nine** flags — `--seeds`, `--n-train`,
`--n-eval`, `--steps`, `--threads`, `--arms`, `--trace-every`, `--device`, `--tag` — and
**no `--s` and no `--S`**. The brief's "six siblings" understates it: **nine** siblings are
flags while the one variable Q4 is *about* is frozen in the source.

Independently swept every `results/**/*.jsonl`. This office's raw sweep found 84 records
carrying an `s` field, 78 at `64` and 6 at `8`; a stricter cell filter resolves the six —
they are **bind-check rows, not cells** (`PUBLISHED_BIND` is annotated
`# V15_ARM_PL.md section 2, at s = 8`, `scripts/v15_r1.py:141`). On cells:

```
Q4's own scope (test_v20_r15_it8_q4_q5.py:15-16) : 34 cells, distinct s = {64}
every results/ journal in the repository          : 65 cells, distinct s = {64}
```

**65 of 65. The distinct-value set is `{64}` at every scope anyone has quoted.**
Also invariant on all 34: `t_star={2}`, `steps={150}`, `d={24}`, `n_train={2048}`,
`n_eval={4096}`.

**TWO CORRECTIONS AGAINST THIS OFFICE'S OWN BRIEF.** The brief said *"40 of 40"* and
*"six siblings"*. Q4's scope is **34**, not 40 — the 40 is the *Q6* dedup census at
`V20_R15_IT89_INSPECTOR.md:869`, and the same file quotes a **third** figure, `55/55`, at
`:860`. **Three numbers circulate for one census across two of this office's own
documents.** And the siblings number **nine**, not six — JUPITER's report says seven and
cites `:547-557`, silently dropping `--tag` at `:558`. **Both errors run in the report's
disfavour: it undercounts its own case.** The claim is robust to which count you believe.

**The consequence is structural and this office states it plainly.** Any Q4 claim of the
form *"X varies with `s`"* or *"X is invariant in `s`"* is **not underpowered — it is
untestable**, because the independent variable has `n = 1` and no code path exists by
which it could have had more. That is not a grade this office can soften on appeal, and it
is not a defect a rerun fixes: it needs a flag first. **F3 STANDS, and it is the most
solid finding in the theory table** — the only cell whose grade rests on a fact rather
than on an inference.

**And the provenance discipline is intact, which this office records in JUPITER's
favour.** Both wings do state an S-scaling law — `Θ(n·S²·d_model)` at
`V20_R15_IT8_JUPITER.md:107` and `Θ(n·S²·d)` at `:227` — and with `n = 1` in `S` these are
observationally identical to `Θ(n·S)` or `Θ(n·S³)`. **The report knows and says so.** The
exponents are tagged `[DERIVED]`/`[READ]` from operator source, **never `[RUN]`**; `[RUN]`
attaches only to the constant, the spread and the census; and `:247` states it outright —
*"there is no law in `S`, because `S` never varied."* This is not a claim overreaching its
evidence. It is a claim correctly fenced, and it is the best-disciplined cell in the table.

**The load-bearing assert is clean, and D4 does NOT recur on it.**
`tests/jupiter/test_v20_r15_it8_q4_q5.py:60-67` asserts `{r["s"] for r in cells} == {64}`
— **set equality**, no tolerance, no bound. The green certifies exactly the prose. The red
is genuine too: `V20_R15_IT8_JUPITER.md:415-425` records `assert 34 == 26`, the draft
counting 26 and correcting **upward** against itself, and a seeded planted negative
(`out[0]["s"] = 128`, `:32-33`) takes the node RED, proving it load-bearing where a natural
red is physically unavailable. **This is the pattern the other cells lack.**

**One defect in the instrument, and it is real.**
`tests/jupiter/test_v20_r15_it8_q4_q5.py:236` asserts `"--s " not in runner` — with a
**trailing space**. The canonical form is `ap.add_argument("--s", …)`, quote after the
name, never a space. Verified empirically: append a working `--s` flag to the source and
**the assertion still evaluates `True`**. The negative check cannot fail on the string it
exists to forbid. The claim it guards is true — this office established the absence by
grep of the parsed source, not by substring — **but the guard is not**. Repair:
`'add_argument("--s"' not in runner`.

---

## 2. JUPITER'S REPAIR — DID IT LAND, AND DOES IT EARN THE POINT

### 2a. §3.4 re-pointed at `equilibrium_oracle` — **LANDED, and it is good work**

`scale/negation_scope.py` is newly modified this iteration (clean at session start), and
`tests/jupiter/test_v20_r15_it11_q6_oracle.py` is new. Run by this office:

```
tests/jupiter/test_v20_r15_it11_q6_oracle.py  ....  5 passed in 2.78s
```

The new node is written the way this office has spent six reports asking for. It asserts
**exactly**, not approximately, and it says so in its own comments:

```python
assert _w1(pred, z) == 0.0                      # exact, not approx
assert _nrmse(pred, z) == NRMSE_PERMUTED        # bitwise, not `approx sqrt 2`
assert _nrmse(pred, z) > 1.0                    # worse than predict-the-mean
```

It also carries a **non-degeneracy control** (`:108-114`) — `_w1(2.0*z, z) > 0.0` and
`_w1(z+1.0, z) == pytest.approx(1.0, rel=1e-4)` — which is the thing missing from every
planted control audited above. **This office records it as the correct pattern.**

### 2b. The `≈` at `V20_R15_LEAP_LEDGER.md:131` — **LANDED, late in the iteration. This office was wrong at first read and says so.**

**Read at 09:23 into the iteration it had not landed; read at 12:40 it had.** The line now
carries `NRMSE ≈ √2` — **`≈`** — and goes further than ordered, adding the measured
`1.421901` against the comparison value in the same row. The finding below is the state at
first read, preserved because the reasoning is what made the repair checkable; **the repair
is complete and this office withdraws the NOT-LANDED verdict.**

`V20_R15_LEAP_LEDGER.md:131` still reads, verbatim:

> `W1 = 0.0` exactly at `NRMSE = √2`

A bare `=`. Its own source report already reads `≈` — `V20_R15_IT9_JUPITER.md:301`:
*"permuted oracle scores `W1 == 0.0` exactly at `NRMSE ≈ √2 > 1`."* And the equality is
**false**, by this office's arithmetic:

```
sqrt(2)  = 1.4142135623730951
measured = 1.421901019003236
diff     = 0.00768745663014081
```

**JUPITER's own new test asserts the refutation.** `tests/jupiter/test_v20_r15_it11_q6_oracle.py:39`
sets `NRMSE_PERMUTED = 1.421901019003236`, and `:77-78` assert:

```python
assert NRMSE_PERMUTED != math.sqrt(2.0)
assert abs(NRMSE_PERMUTED - math.sqrt(2.0)) > 1e-3
```

So the tree now contains **a green test whose explicit purpose is to deny what the ledger
line states**. This is a one-character repair that was specifically ordered, was
understood well enough to be *tested*, and was not made. It is small — and it is exactly
the class of defect (a bare `=` on an approximate quantity) that this round has struck
four times.

### 2c. The census journalled — **LANDED, and in the strongest available form**

Same correction: absent at first read (journal ended at 1894 with `WHAT it.11 OWES`),
present at second (1998 lines). `V20_R15_JOURNAL.md:1897-1898` journals it **with a
machine-readable copy** — `results/v20_r15_it11_jupiter_census.jsonl`, three `t="census"`
records, verified present and parseable by this office:

```
{"t":"census","it":11,"office":"JUPITER","question":"Q6 -- does any banked cell carry a
 distributional object?","n_cells":40,"by_kind":{"arm_pl":16,"arm_smprime":16,"softmax":8},
 "n_union_keys":60,"n_vector_valued_fields":0,"n_histogram_fields":0,
 "n_quantile_fields":0,"n_density_fields":0,"n_prediction_sample_fields":0,...}
{"t":"census","it":11,...,"question":"Q1/W1 -- is pathProd_eq_Wp's hypothesis satisfied on
 any registered bed?","declaration":"pathProd_eq_Wp","file":"lean/CEQ/V16Domain.lean",
 "line":176,"hypothesis":"forall k, 0 < m k","n_registered_beds":3,
 "beds":["BED-M","BED-K","BED-1"],"beds_satisfying_hypothesis":0,...}
```

**A census emitted as data rather than as prose is the correct answer to STRIKE I-3**, and
it reaches Q1's Lean half — the cell this office named as unreached — with a declaration,
a file, a line, and a bed count of zero. `n_cells: 40` also settles the census denominator
this office had circulating at three different values.

---

## 3. THE DEBT THIS OFFICE CALLED LARGER THAN THE `+6`

### The corrections index LANDED, and it is a real instrument

`V20_R15_JOURNAL.md:24` — `## CORRECTIONS INDEX — READ BEFORE ANY ENTRY BELOW`. Fifteen
rows `C1..C15`, columns `# | the claim, as filed | where filed | what is true | corrected at`.
**Every row carries the substance, not a pointer** — C13 states the pooled sd `0.2125` vs
eight-cell `0.0203` and the n=9/n=8 flip; C14 states `p = 0.012821` against `6.730e-04`
and names the factor of 19. A row reading "see it.10" would have hidden what changed;
none of them does. It names the agent who forced each correction, most of them against
its own author. **This office asked for a forward-pointing table and got one.**

### But it fails on its own terms, in four ways

**U1 — it misses the exact sentence this office complained about.** The finding named the
`12 of 16` / `7 of 8 vs 0 of 8` pair *"still standing in the journal, unretracted."* The
site is the it.**9** DISTANCE line, `V20_R15_JOURNAL.md:1736`, which this office quoted at
`V20_R15_IT89_INSPECTOR.md:410` and `:523`. C13's `where filed` reads "it.8 DISTANCE";
C14's reads "it.8". **Neither names it.9.** A linear reader who reaches `:1736` is carried
nowhere. The index corrects the first statement of the claim and leaves the restatement
standing — which is the failure mode it was built to prevent.

**U3 — no row at all for the finding that voids every price in the round.**
`V20_R15_JOURNAL.md:1638-1653`: no `torch.cuda.synchronize()`, strongest correlate of
`secs` is run order at `ρ = +0.7029`, and the journal's own words, *"IT INVALIDATES EVERY
GPU TIMING THIS ROUND HAS QUOTED."* That retroactively supersedes `0.3455 GPU-h` — which
is the *"what is true"* column of **C3 itself** — plus `10.67x`/`1.078x` (`:419-420`),
`15.907 s` vs `14.852 s` (`:1658`), and STRIKE 6's `25.0x` (`:573`). **The largest
unpointered class in the file, and it undercuts a row the index already contains.**

**C9 — a pointer that reads STRONGER than its target.** C9 asserts *"MARS later showed
pre-clamp `u` strictly exceeds 1.0 on 6 of 6 zero-step cells."* The journal says the
opposite at `:1444`: *"Not determined: whether pre-clamp `u` exceeds `1.0` or lands on it
exactly"*, still open at `:1541` and `:1707`. The result exists only in
`V20_R15_IT9_MARS.md:383-422`, outside the journal. **A reader following C9 into it.9
finds the question open.** The index's guard clause (`:58-60`) forbids a row reading *more
gently* than its entry; it has no clause for a row reading more strongly, and that is the
direction it actually failed in. Also: C10's `where filed` says *"repeated it.9"* when all
three occurrences are it.7, and C2 credits the Inspector for a correction the coordinator
self-attributed.

**No pointer in any of the 15 rows carries a line number.** Every one is
iteration-granular, pointing at a ~200-line region — and in three cases at a file that is
not the journal.

### CAN A CORRECTION BE SOFTENED IN THE INDEX? **YES.**

The index sits at `:24-64`, **above** the file's only append-discipline marker
(`<!-- APPEND THE NEXT ITERATION BELOW THIS LINE -->`, `:1894`). It is therefore the one
**mutable region of an append-only file**. There is no hash, no `DO NOT EDIT` marker, and
no test: the string `CORRECTIONS INDEX` occurs **once in the entire repo** and is
referenced by nothing. The journal is **untracked by git**, so an edit leaves no trace
anywhere. The sole guard is the prose at `:58-60`, a norm with zero enforcement — **and it
is already violated, by C9, in the direction it does not cover.** The failure mode this
office predicted has recurred one level up, exactly as predicted, and it is live now.

### IS THE DEBT DISCHARGED? **NO — reduced. Of MARS's four it.9 strikes, ONE is discharged and THREE stand.**

The it.10 numbers **verify**: `8 of 9 arm_pl` vs `0 of 9 softmax` paired over three draws
`12345/12346/20260902`, gap `0.2360` with `floor₁` inside, **54 scorings, zero flips**
(`V20_R15_IT10_MERCURY.md:38,110,129,145-153,186-207`). This office confirms them.

- **DISCHARGED — S-c**, the unpaired `softmax` control. Repaired by measurement; VENUS's
  `test_softmax_has_never_been_run_past_seed_7` went RED while her report was open. Clean.
- **STANDS — S-a**, the seed-9 hinge. it.10 reproduced both computations bitwise and
  **took no ruling**. The journal concedes it at `:1883-1884`.
- **STANDS — S-b**, `sign_acc_ci` hi `1.0179795219893695` and `gate_r2_ci` hi
  `1.1121099749560528`, both outside the estimand's range. **Not repaired, not journalled,
  not indexed — entirely off the record.**
- **STANDS — S-d**, the seed-9 magnitude envelope. The sign separator survived; the
  magnitude envelope was never re-taken.

**The brief's "two of four" is one too many.** `n_eff = 1` is §5.2, the surviving half of
STRIKE 10 — **not one of MARS's four** (`V20_R15_IT9_MARS.md:542` names the four RED nodes
and excludes the eval-pin RED). Counting it in while S-d drops out silently swaps the
denominator. And MERCURY declined to call even that lifted: 54 scorings buy `n_eff = 3`,
not unbounded, and `ci_hi` moves `0.100` across draws with **no published interval
containing that term** (`:1784-1788`). **Narrowed, not lifted.**

**Provenance gap:** `grep "V20_R15_IT10" V20_R15_JOURNAL.md` returns **0 hits**. The
1894-line journal cites nurse reports six times in total. it.10's headline numbers stand
with no file:line to the reports that produced them.

### Q5 / W1 and Q5 / W3 — **BOTH STRUCK, but not for the reason the brief gave**

**This office corrects its own framing first.** A cell is not unbound merely because its
headline quantity is violated 18% of the time — **a threshold is not a bound**, and the
it.8 report is the document that says so out loud (`V20_R15_IT8_JUPITER.md:317-318`:
*"calling it a floor is the thing that has to stop"*). Worse for the brief's reading: the
6 "violations" and the round's 6 "crossings" are **the same six cells** —
`scripts/v15_r1.py:17-19` states in source that *"`NRMSE < floor_1` AND `h_hat > 1` ARE ONE
EVENT"*. So "violated by 6 of 34" is the round's headline achievement re-narrated as a
scandal. **The measurement is clean; the defect is in the noun.**

The cells are unbound for two sharper reasons, both D4:

- **Q5/W1 — the graded constant has ZERO test coverage.** The grade at `:355-356` rests on
  *"predicted band `+0.20 … +0.27`, observed `+0.145 … +0.220`."* Grep of the whole test
  file for `0.145`, `0.220`, `0.2317`, `0.2663`, `0.20`, `0.27` → **zero hits.** The
  `[RUN]` markers at `:309-313` bind the `floor_1` identity and the `h_hat` identity,
  which nobody contested, and **nothing the grade rests on**. The band re-derives correctly
  (`+0.144954 … +0.219610` over 12; softmax `+0.231689 … +0.266293` over 8) — **correct and
  untested.** It is also a **postdiction stated in prediction voice**: the "prediction" is
  `[DERIVED]` at `:348-351` from softmax cells sitting in the same banked journal.
- **Q5/W3 — the test is wider than the prose.** Prose (`:394-395`) grades on *"`0.0488` on
  `ci_hi` at `n = 5`"*. The cited node,
  `tests/jupiter/test_v20_r15_it8_q4_q5.py:163-164`, asserts `hi < 0.7071067811865476`.
  **That asserts margin > 0. A margin of `1e-9` passes.** It does not assert `0.0488`, does
  not assert `ci_hi = 0.6582633033`. **The D4 mechanism this office corrected at it.9 was
  already live in the it.8 report it was correcting from.**

**"6 of 34" is SELF-REFUTED at the larger denominator.** On the record this office itself
banked — adding `results/v20_r15_it8_armpl_b.jsonl`, declared landed at
`V20_R15_LEAP_LEDGER.md:139-141` — the count is **13 of 40 (32.5%)**, not 6 of 34. Already
struck at `house-events.jsonl:12518` (*"it.9 entry says 34 of 34 and 6 of 34 while its own
report establishes 40"*) and **still standing unamended at `V20_R15_LEAP_LEDGER.md:70`.**
Also: 34 contains 2 duplicate records, so on distinct cells the it.8 rate is 6/32.

Unretracted "floor"-as-bound uses with no violation attached:
`V20_R15_LEAP_LEDGER.md:144-145`, `:153-155`, `:160`; `V20_R15_JOURNAL.md:261, 432, 621,
778, 1029, 1088, 1472, 1545` (eight DISTANCE lines) and `:1838`.

**M11/M12 — confirmed, and vacuous.** Zero producing `.py` is true. But **no M11/M12
number appears in the ledger or journal at all** — every figure lives only at
`CEQ_V20_R15_CONTRACT.md:227,231`, and all four mentions in the two target files already
carry an explicit non-reproducibility disclaimer. **There is no unbound number to report.**

**M16 — UNBOUND for its stated purpose, and worse than stated.** `V20_R15_WING_MANIFEST.md:22-23`
puts W1 = `arm_smprime` and W3 = `arm_pl`, both BED-M; BED-K(a) is unrun
(`V20_R15_JOURNAL.md:1549`). The `1/d` anchor is computed **only inside a test**, on a
synthetic `_delay_hankel(20)` block. `V20_R15_IT8_JUPITER.md:473` says it itself: *"do not
cite it on W1/W3 at all."* Correct self-indictment, not a hidden defect.

**RED — self-attested, not logged.** `V20_R15_IT8_JUPITER.md:429-437` carries a real
Q5-specific red with a verbatim traceback (`0.6454312` → `0.6454112028697345`). But
`house-events.jsonl:12318` logs only `"reds_recorded": 2` — no node id, no traceback, and
**no `red_first` field**, where it.9's events at `:12388-12389` do carry one.

---

## 4. THE PLANTED NEGATIVE — VERIFIED BY BREAKING IT

This office ordered the new planted negative verified **by making it fail**. Six mutations
were applied to the guarded implementation (`scale/negation_scope.py`), never to the test's
own assertion. **The file was restored and the digest matches:**
`555613bd6704b6674b5214ec5f0087929f8ad10d3a93a6b54494fc52aee47e0d` before and after.

**CAN IT FAIL? YES — 5 of 6 mutations went RED.** That alone clears the bar the struck
predecessor failed. But the *shape* of the discrimination is not what the file claims.

| mutation | the bug it imitates | result |
|---|---|---|
| **A** — `+b[:,i]` → `−b[:,i]` | **sign error in the recursion** | **GREEN — NOT CAUGHT** |
| B — delete `b[:, s-1] = 0.0` | dropped normalization | RED (pin only) |
| C — `a *= 1.1` | 10% coefficient perturbation | RED (pin only), margin `2.05e-4` |
| D — delete `a[:, :head+1] = 0.0` | **the planted t\*-band removed** | RED (pin only) |
| E — drop `+ b[:,i]` entirely | label collapses to 0 | RED (crash + 3 of 4) |
| F — `range(n)` → `range(n-1)` | off-by-one | RED (pin only) |

**Two findings.**

**(1) A genuine sign bug is invisible.** Mutation A returns *exactly* `−z`
(`torch.equal(z_mut, -z) → True`). W1 of a permutation is 0 either way and NRMSE is
scale-and-sign invariant, so the pin reads the identical float. **The negative cannot see a
sign error in the very oracle it guards.**

**(2) The test named "THE NEGATIVE THAT CAN FAIL" is the one that cannot.**
`test_q6_marginal_W1_inverts_the_ranking_against_nrmse` stayed **GREEN under B, C, D and
F** — every non-degenerate wrong implementation. It fires only under total degeneracy (E).
The reason is structural: for any non-degenerate iid marginal,
`E[(z_perm − z)²] = 2·Var(z)`, so `nr_a ≈ √2` regardless of implementation; `w1_a == 0` is
a permutation identity; and the opposed-orderings assert follows from those two. **The
rank-inversion claim is the old identity wearing a second coat**, and its docstring's claim
that it *"depends on the oracle's actual values, not on the formula"* is false for every
mutation except a constant label.

**Seed sweep, unmutated:** `w1_a` is **exactly `0.0` at all six seeds** and `w1(z+1, z)` is
**exactly `1.0` at all six** — identities, not measurements. Only `nr_a` varies
(`1.3973 … 1.4291`, spread `0.032`), and **only the seed-4096 pin exploits it.**

So all real discriminating power lives in **one line** —
`assert _nrmse(pred, z) == NRMSE_PERMUTED`, a bitwise-pinned regression constant that
caught 4 of 6. Worth keeping; it is a fingerprint of the draw, not the argument the file
makes. Note `pytest.approx` at any sane tolerance would **not** have caught C (margin
`2.05e-4`) — the bitwise `==` is doing the work, which vindicates the exactness discipline
even as it exposes the reasoning.

**Repair, and it is free:** `equilibrium_hop_reading(x, k)` is already in the module with
closed-form NRMSE `sqrt((t*−k)/t*)` — exactly `1.0` at `k=0`, exactly `0` at `k=t*`.
Asserting the k-ladder against that closed form kills A, B, C, D and F on a **derived**
quantity instead of a pinned float, and it would catch the sign error.

---

## 5. THE LEDGER

**13 audited, 9 struck.**

| # | check | verdict |
|---|---|---|
| 1 | Q2/W1 `R2_1·d = 1.0`, spread `0.000e+00` | **STRUCK — VACUOUS.** `HᵀH = I_d`; fake red; `err_1` asserted only `isfinite` |
| 2 | Q2/W3 nonexistence `rank_+` | **THEOREM SOUND, CELL STRUCK.** `n=4` fixed, no size loop; asserts are identities |
| 3 | Q2/W3 constants `1.084523`, `0.5303282490` | **STRUCK.** First exists only in prose; second asserted `> 0.1` at 10 printed digits |
| 4 | Q2/W3 planted `squared_feature` control | **STRUCK — IDENTITY.** `z` bitwise equals `t_perm`; a wrong PAVA passes both asserts |
| 5 | Q3/W1 `qk` partition "16 of 16" | **STRUCK.** 15 asserted; seed 3 in no block, contradicted six lines apart |
| 6 | Q3/W1 Spearman `+0.7176` / `−0.0324` | **STRUCK.** No Spearman anywhere in the tree; `−0.0324` CI `[−0.520, +0.471]` |
| 7 | Q3/W3 biconditional + margin | **PARTIAL.** Biconditional asserted; margin `0.451211` certified at `0.44`; `0.7` unregistered |
| 8 | **Q4/W1 + Q4/W3 — `S` is a module constant, `s` has one value** | **UPHELD. F3 STANDS.** `s = {64}`, 34/34 and 65/65; nine sibling flags; set-equality assert |
| 9 | Q4 `"--s "` guard | **STRUCK.** Trailing space; cannot fail on the flag it forbids |
| 10 | Q5/W1 constant `+0.145 … +0.220` | **STRUCK.** Zero test nodes; postdiction in prediction voice |
| 11 | Q5/W3 constant `0.0488` | **STRUCK.** Cited assert requires only margin `> 0` |
| 12 | Q5 `6 of 34` | **SELF-REFUTED.** 13 of 40 on the record this office banked; unamended at ledger `:70` |
| 13 | JUPITER's four repairs | **ALL FOUR LANDED.** §3.4, the `≈`, the census, and a negative that fails 5 of 6 |

**Self-strikes, this office's own:** propagating "16/16" at `V20_R15_IT89_INSPECTOR.md:660`
without opening the test — **the same mechanism as C12**, twice in one round; circulating
**three denominators** (34 / 40 / 55) for one census across `:860` and `:869`; and framing
Q5 around the floor violations when the violations *are* the crossings.

**Tree statement.** `git status --porcelain` tracked-modified set is
`house-events.jsonl`, `pytest.ini`, `scale/ledger.py` — **identical to session start**.
One mutation was made and reverted: `scale/negation_scope.py`, digest
`555613bd6704b6674b5214ec5f0087929f8ad10d3a93a6b54494fc52aee47e0d` **before and after**,
matching HEAD. It was dirty for roughly 76 seconds mid-audit and is clean now.
**Concurrent it.11 paths, not this office's:** `V20_R15_IT11_MERCURY.md`,
`results/v20_r15_it11_{jupiter_census,mercury_draw4,mercury_smprime}.jsonl`,
`scripts/v20_r15_it11_mercury_{draw4,report,smprime}.py`,
`tests/mercury/test_v20_r15_it11_smprime_draws.py`,
`tests/jupiter/test_v20_r15_it11_q6_oracle.py`. **No git writes. Nothing touched Kaggle.**

---

## 6. THE RULING

### IS THE THEORY TABLE BOUND, SO THAT THE `+6` IS EARNED?

# NO. The `+6` is REFUSED a second time.

**And this office states plainly that the refusal is narrower than it was, and that it is
not a refusal of the work.**

**What JUPITER was ordered to do, he did — all four.** §3.4 is re-pointed at
`equilibrium_oracle` and its new node asserts `== 0.0` exactly with a non-degeneracy
control, which is the pattern six reports have been asking for. The `≈` landed. The census
landed **as data**, three `t="census"` records, reaching Q1's Lean half. And the planted
negative was rebuilt into something that **dies under 5 of 6 mutations**, where its
predecessor died under none. **The repair is not the reason for the refusal.**

**The reason is that four of the eight cells this office had never reached turn out to be
worse than ungraded — they are ungrounded.** Not wrong: **unasserted.**

- **`1.084523` does not exist in any Python file.** Three occurrences, all prose.
- **`+0.7176` and `−0.0324` are computed nowhere in the tree.** `grep -rn spearman` over
  `tests/`, `scripts/`, `ceq/` returns nothing for these cells.
- **Q5/W1's graded band has zero test nodes.**
- **Q2/W1's headline constant is asserted `isfinite`.**

**D4 is not a defect. It is the table's dominant mode.** This office found it in **seven**
of thirteen checks: a tolerance nine orders wider than the printed number (Q2/W1), a
one-sided floor at `0.1` under ten printed digits (Q2/W3), a margin of `0.44` under a
prose constant of `0.451211` (Q3/W3), an assert requiring only `> 0` under `0.0488`
(Q5/W3), and — three times — **no assertion at all**. A table where the modal relationship
between prose and test is "the test is wider, or absent" is not bound, and no count of
green runs changes that.

**The one cell that is bound is bound because it is a fact.** Q4's F3 rests on
`S, D = 64, 24` at `scripts/v15_r1.py:137`, nine sibling flags, `s = {64}` on 65 of 65
records, and a **set-equality** assertion with a real red behind it. **It is the only cell
in the table whose grade this office cannot argue with**, and it is the template for what
the other eleven need.

**The debt is REDUCED, not DISCHARGED.** Of MARS's four it.9 strikes, **one** is
discharged, **three** stand — S-a the seed-9 hinge, S-b the out-of-range CIs (off the
record entirely), S-d the magnitude envelope. The brief's "two of four" counts `n_eff = 1`,
which is §5.2 and **not one of the four**. The corrections index is a real instrument whose
rows carry substance, and it fails on its own terms: it **misses the it.9 restatement that
was the entire complaint** (`V20_R15_JOURNAL.md:1736`), has **no row** for the finding that
voids every GPU timing in the round, and **C9 reads stronger than its target**, asserting
settled what the journal calls undetermined at three sites.

**And the recursion this office predicted is live.** The index sits at `:24-64`, **above**
the file's only append marker at `:1894` — the one mutable region of an append-only,
**untracked** file, with no hash, no marker, and no test. Its guard clause forbids a row
reading *more gently* than its entry. **It has no clause for the direction it actually
failed in**, and C9 already failed in it. **Yes, a correction can be softened in the index.**

### WHAT WOULD EARN THE `+6`

Not more runs. **Assertions.** Four numbers need a node that can go red — `1.084523`,
`+0.7176`, `−0.0324`, and Q5/W1's band — and four existing nodes need their tolerance
narrowed to the precision their prose prints. Q3/W3 needs the **preregistered**
`0.7071067811865476`, which sits twelve lines from the `0.7` that was used. The planted
negative needs the `equilibrium_hop_reading` k-ladder so a **sign error** stops being
invisible. And the index needs line-number pointers, a row for U3, and a hash.

**That is a bounded list, and every item on it is small.** The table is one iteration of
assertion-writing away from a ruling this office would grant. It is not there yet.

**SCOREBOARD. `+6` REFUSED — the table is not bound, and the reason is that four of its
constants are asserted by nothing.** `+2` stands from it.4. Repair: **all four items
LANDED**, and this office records that it was wrong twice about the state of the tree and
corrected itself both times against the record.

---

## 7. WHAT THIS OFFICE DID NOT REACH

Named, because the round's rule is that an unreached cell is not a passed cell.

1. **The grade rubric itself.** F1 / F2 / F3 / F4 / "+ const" are applied throughout the
   table and this office **never located a written definition**. A scale with no rubric
   cannot be audited for misgrading — only for evidence, which is what was done. **If no
   rubric exists, that is a larger finding than any single cell** and it belongs at it.12.
2. **A systematic sweep for the D4 family across the whole suite.** Seven instances were
   found inside the eight cells in scope. The suite has not been swept for the pattern
   outside them, and on a base rate of 7/13 the expected yield elsewhere is not small.
3. **A tautology and skip census** — tests with zero assertions, `xfail`/`skipif` nodes,
   and what `pytest.ini` (currently modified) excludes via `addopts`. Not opened.
4. **Q1 and Q6** were out of scope this iteration and are not ruled on here.
5. **The three unreached numbers in the false-precision class** —
   `0.432760655774893`, `1.421901019003236`, `0.5303282490` are each printed to 15-16
   significant figures on evidence of 8 to 40 samples. The first two are now genuinely
   pinned (a bitwise regression constant is legitimately 16 digits); **the general question
   of whether printed precision matches evidential precision across the table was not
   answered.**

---

## 8. LATE FINDING — THE `[RUN]` MARKER ITSELF (arrived after the ruling; it strengthens it)

A sweep of the round's evidence tags landed after §6 was written. **It does not change the
ruling. It removes the last doubt about it.**

```
[RUN] markers, it.6-it.10 + journal   153
  name a test node or pytest cmd       31
  name a shell command                 30
  NAME NOTHING RUNNABLE                92   <-- 60%
Nodes, it.6-it.10                     128
  RED-THEN-GREEN (strict)              26
  GREEN-ONLY     (strict)             102   <-- 80%
  GREEN-ONLY     (generous)            90   <-- 70%
Tests hidden by norecursedirs         216   (8 attic + 208 kaggle)
```

**Sixty percent of the round's `[RUN]` tags cite no artifact.** A tag you cannot re-run is
a tag you cannot ask "was this ever red?" — which is this office's whole standard.

**Three of it.6's five REDs are theatre**, and this office had already caught one of them:
they were produced by mutating the **test**, not the data — `assert diff <= 0.0`, an
assertion conjoined with `and False`, and an edited expected line number. **An assertion
conjoined with `False` fails whatever the data says.**

**The largest unbound block in the round is 21 nodes with zero REDs claiming RED-first.**
`V20_R15_IT9_SATURN.md:12` reports *"21 passed in 0.46s"*; the *"RED first"* claim at
`:251` rests on **four `house-events.jsonl` entries appended before the run** — a
declaration of intent, not a failing test.

**D4-1 is still in the tree and the prose it certifies is still false.**
`tests/jupiter/test_v20_r15_it7_q3.py:96` admits `abs(live) < 0.05`; the journal says
*"under 3%"* at `:1235`, `:1336` and `:1387`; **the true maximum is 4.76%, and the
falsifier lives inside the assertion's slack.** One acknowledgement at `:46`, three
uncorrected publication sites. **Struck as unrepaired at it.9; still unrepaired.**

**Seven more D4 instances** outside the eight cells, the sharpest being a published cost
constant `10.171×` certified by `assert 9.0 < ratio < 11.0` (~2000× looser), a five-digit
`1.4859×` certified by a bare `> 1.48`, and `crude/exact = 1.0` certified by
`crude >= exact - TOL` — **which the journal itself diagnoses at `:245` and which is still
shipped.**

**And one tautology is load-bearing.** `tests/mars_v20/test_v20_r15_it9_eval_pin.py:161`
carries the common-mode claim — that a scale shift *"cannot manufacture a DIFFERENTIAL
between `arm_pl`'s 12/16 and `softmax`'s 0/8"* — on the single assertion
**`assert d["std_y"] > 0`**, the standard deviation of a 4096-element random tensor. **It
cannot fail.** `V20_R15_IT9_MARS.md:349` treats the conclusion as established.

**Revised ledger: 16 audited, 12 struck.** The `+6` refusal stands and is now
over-determined: **80% of the round's nodes were born green, and 60% of its evidence tags
point at nothing.** This office's §7 item 2 — that the D4 base rate outside the eight cells
would not be small — is answered: it is eight more instances, and the original is live.

**One item for it.12 above all others.** `pytest.ini`'s repair moved root collection from
`201 errors, 0 tests` to `2895 tests`. `V20_R15_JOURNAL.md:238` asserts that *"every agent
result this round predates the fix and was produced by an explicitly-named path — no claim
rested on a root run."* **That sentence is a claim about every it.1–it.10 invocation in the
round, and it is bound by nothing.** It is the largest unverified claim the round makes
about its own evidence, and it should be the first thing it.12 tests.

---

## 9. TWO LATE FINDINGS THAT INDICT THIS OFFICE, NOT JUPITER

**(a) THERE IS NO RUBRIC. F0–F4 are never defined anywhere.** Searched the contract, the
ledger, the journal and every V20_R15 report. The only gloss in existence is one
self-issued clause mid-report — `V20_R15_IT6_JUPITER.md:496`, *"F1 means 'with the
constant'"*. **F2 and F3 are never characterised at all**, and F4 is used for three
incompatible things: unattempted (`V20_R15_LEAP_LEDGER.md:28`), struck (`:24-25`), and
domain-empty (`:130-131`). **"Misgraded" is therefore unfalsifiable by construction** —
twelve grades on a scale nobody wrote down. **This office has been ruling BOUND / NOT
BOUND against grades it never required a definition for.** That is a larger finding than
any cell in this report, and it belongs at the head of it.12.

**(b) THIS OFFICE'S OWN "FOUR NOT REACHED" WAS INCOHERENT WHEN WRITTEN.**
`V20_R15_IT89_INSPECTOR.md` states three incompatible counts:
`:454-478` names **ten** unreached (*"eight of the twelve cells"* plus Q1 both wings);
`:550` says **four**, *"Q1's Lean half among them"*; `:836` names the four as **Q4/W1,
Q4/W3, Q5/W1, Q5/W3** — **Q1 not among them, contradicting `:550` 286 lines earlier** —
and then `:860-863`, **24 lines below**, adjudicates all four **BOUND**, closing at `:945`
with *"Twelve of twelve cells adjudicated."* **Ten, four, and zero, in one document.** And
Q1's Lean half was closed CLEAN at `:569-591` **in the same file that says it was not
reached.**

**The brief this office wrote for itself this iteration was built on that sentence.** It
was wrong in its count, wrong in its membership, and wrong in the "40 of 40" and "six
siblings" it carried into Q4. **Every one of those errors was found by the room, against
this office, in the same iteration.** The corrections index has a name for this pattern —
C12 — and this is its third instance.

**Q1's Lean half, ruled properly at last:** zero `sorry`, zero `admit`, zero `axiom` in
project Lean; the six `sorry` greps are doc-comments asserting their own absence; the four
cited declarations sit exactly at `lean/CEQ/V16Domain.lean:105, 121, 129, 176` with
`#print axioms` covering 18 declarations at `:529-546`. **The Lean is real and it is the
strongest thing in the table.** Two caveats: **W3 has no Lean at all** (`Lean #21 [S]`
withdrawn at it.2, never replaced — which is why W3 is F1 and W1 is F0), and the theorem
nearest the arm, `pathProd_eq_Wp`, carries `∀k, 0 < m k`, **satisfied by 0 of 3 registered
beds and 0 of 2048 BED-M rows.**

**Constants, counted properly: 7 reproducible, 17 bare — 71% bare.** Thirteen are printed
past their evidence, the worst being `0.6582633033288883` — **16 significant figures from
n = 5**, a t-CI endpoint whose own sd carries ~35% relative error.

**And the §3.4 numbers do not port.** `NRMSE_B` published as `0.098296619951725` in three
artifacts measures `0.098296619747453` on a second machine (rel. `2e-9`, thread-count
dependent float reduction order). Harmless numerically; **three artifacts now carry
15-figure constants that are not machine-portable.**

**Final count: 12 of 12 reached, 6 bound, 4 struck, 2 repaired-but-unaudited. The it.11
audit LOWERED the bound count from 8 to 6.** That is the correct direction for an
instrument that is working.
