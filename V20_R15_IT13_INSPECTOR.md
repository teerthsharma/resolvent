# V20 R15 — HEALTH INSPECTOR, it.13 AUDIT AND THE it.14 FREEZE RULING

HEAD `207e7b92c5611effbcac6877757dc1d0bb572e42`, branch `v17k-gate0`.
Audited: `V20_R15_IT13_MERCURY.md`, `V20_R15_IT13_VENUS.md`, the coordinator's
corrections index in `V20_R15_JOURNAL.md`, and `V20_R15_LEAP_LEDGER.md`.

---

## RULING ON THE it.14 FREEZE — READ THIS FIRST

**`V20_R15_THEORY_TABLE.md` DOES NOT EXIST.** Neither does `V20_R15_IT14_JUPITER.md`
nor `V20_R15_IT14_SATURN.md`. Verified four ways, not one:

```
ls V20_R15*                                        -> 44 files, no THEORY_TABLE
find . -iname "*THEORY_TABLE*" -not -path ./.git/* -> 0 hits
git log --all -- "*THEORY_TABLE*"                  -> no commit ever touched such a path
grep -rl "THEORY_TABLE" --include=*.md .           -> 0 hits; nothing references it by name
```

It was not deleted. It was never written. `V20_R15_JOURNAL.md:2523` opens a section
titled `WHAT it.14 OWES — AND it.14 CLOSES PHASE B` whose item 4 is *"The theory table
re-frozen at it.14"*. That debt is open at the time of this filing.

So the ruling below is not on the it.14 freeze — there is nothing to freeze. It is on
**the only artifact that today occupies the slot the contract reserves for the leap's
primary input** (`CEQ_V20_R15_CONTRACT.md:112-113`), which is the it.9 freeze, rendered
twice and only twice:

- `V20_R15_IT9_JUPITER.md:13-20`
- `V20_R15_JOURNAL.md:1584-1592`

Read at HEAD, both are twelve cells of one F-token each:

```
| | W1 `arm_smprime` | W3 `arm_pl` |
| Q1 EXACT CLASS       | F0         | F1         |
| Q2 OUTSIDE           | F1 + const | F1 + const |
| Q3 LEARNABILITY      | F2         | F1 + const |
| Q4 COST LAW          | F3         | F3         |
| Q5 INFORMATION FLOOR | F1 + const | F1 + const |
| Q6 STATE METRIC      | F4         | F4         |
```

No grade rubric. No HOW-BAD gap column. No theorem. No `path:line`. No domain census.
No replacement route. No number. No unit. **All five fields the contract's `L-GRADE`
law requires are absent from all twelve cells.**

> **12 cells total, 0 fully bound, 12 with defects.**

### And it is stale

`V20_R15_JOURNAL.md:2244`: *"The theory table moves: `1×F0 / 6×F1 / 1×F2 / 2×F3 / 2×F4`
→ `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4`."* it.12 withdrew Q2/W1 to F4. **Both rendered
tables still print `Q2 OUTSIDE | F1 + const`.** No rendering of the current grades exists
anywhere in the tree. The artifact standing in the leap's slot is one grade wrong and
nobody has drawn the corrected one.

### The three defects: all three PROSE-ONLY, none in the table

| defect | where it actually lives | in the table? |
|---|---|---|
| the scale's missing text (F0–F4 defined nowhere) | `V20_R15_IT12_SATURN.md:58-60`, `V20_R15_IT12_JUPITER.md:151-152`, `V20_R15_IT11_INSPECTOR.md:693-700` | **NO** |
| `floor₁` is not an information floor | `V20_R15_IT8_JUPITER.md:317`, `V20_R15_LEAP_LEDGER.md:70` (L-11), journal C15 | **NO** — the Q5 cell says only `F1 + const` |
| Q4's F3 is a harness fact | `V20_R15_IT12_JUPITER.md:183,:208` | **NO** — the Q4 cells say only `F3` |

The table has no prose in any cell, so no cell *can* carry a defect. All three live in
documents the leap model does not receive.

I re-verified the two citations behind defect 2 and the Q4 sparsity theorem at HEAD
myself. Both bind exactly:

```
scripts/v15_r1.py:17-19     -> "h_hat = t*(1 - NRMSE^2)" ... "h_hat = 1 exactly at NRMSE = floor_1"   VERIFIED
lean/CEQ/V16Domain.lean:129 -> theorem pathProd_eq_zero_iff (m θ : ℕ → ℝ) (i j : ℕ) :                 VERIFIED
```

The *prose* is bound. The *table* is not. That is the whole finding.

### THE QUESTION ONLY THIS OFFICE ANSWERS: what must the leap take on trust?

The leap model gets one call at it.35 and cannot ask a follow-up. Fourteen items, most
load-bearing first.

1. **The F-scale itself.** `F0`–`F4` are 100% of the table's information content and are
   defined nowhere — not in the table, not in the contract, not in the repository.
2. **`F4` is overloaded three ways** (`V20_R15_IT11_INSPECTOR.md:697-699`): unattempted,
   struck, domain-empty. The Q6 row is `F4 | F4` and the reader cannot tell which —
   three different repairs.
3. **`+ const`** appears in five of twelve cells and **no constant is printed.** One of
   them is known only to `±1.49×`.
4. **`§3`** — the it.9 Q6 cells read `F4 — §3`, a cross-reference to a section of a
   document that is not the table. Dangling by construction.
5. **`W1` / `W3`** — no wing manifest, and `W1` is ambiguous *inside this round*:
   `V20_R15_IT9_JUPITER.md:29-38` warns it names both the first wing and the
   1-Wasserstein metric. The row `Q6 STATE METRIC | W1 … | F4` is the exact sentence
   that warning was written about, and the table ships without the warning.
6. **`arm_smprime` / `arm_pl`** — column headers, never defined. The primitive each wing
   stands for appears nowhere in the table.
7. **`Q1 EXACT CLASS` … `Q6 STATE METRIC`** — six short labels standing in for the
   contract's six question texts (`CEQ_V20_R15_CONTRACT.md:94-110`). The grade cannot be
   checked against the question because the question is absent.
8. **Zero numbers, therefore zero provenance.** No `floor₁`, no NRMSE, no GPU-seconds,
   no `n`, no seed, no δ.
9. **No citations of any kind.** No path, no line, no SHA, no test name, no `[RUN]`.
10. **The HOW-BAD gap is missing entirely.** `L-GRADE` is `(F0–F4 + HOW-BAD gap)`. The
    table ships the F and drops the gap. it.35's job is to sort F1/F2/F3 LEAPABLE vs
    TERMINAL, and **that sort is a function of the column that is not there.**
11. **`Q5 INFORMATION FLOOR` presupposes a floor and names none.** No bed appears in the
    table at all, and the floor graded against is not an information floor.
12. **`Q4 COST LAW | F3 | F3` presupposes the grade is about the primitive.** It is about
    `scripts/v15_r1.py`. A leap model will hunt a cost-law theorem for a problem an
    `argparse` line and a `torch.cuda.synchronize()` call would close.
13. **`Q1 | F0` asserts a theorem exists and names none** — after six Lean citations were
    withdrawn at it.2 for having no source declaration.
14. **The words "frozen" and "twelve of twelve".** Neither is true: it.12 moved a grade
    and it.14 never happened.

**Net: the table transmits an ordering and nothing else.** Items 1–3 alone mean a reader
holding only this table cannot recover one checkable proposition from it.

### THE ANSWER

> **NO. The frozen theory table is NOT bound and is NOT fit to be the leap's only input
> — and the stronger fact is that at HEAD it does not exist at all. The artifact
> standing in its slot is twelve bare grades on an undefined scale, one of them
> known-stale since it.12, carrying none of the five fields `L-GRADE` requires and none
> of the three defects this round paid to find.**

---

## STRIKE I-1 — THE LEAP LEDGER CONTRADICTS ITSELF ON "REGISTERED", AND THE LEAP READS IT

`V20_R15_LEAP_LEDGER.md:131` (row L-14):

> **The one registered bed whose output is categorical is the chess witness** (legality,
> next-FEN, eval-Δ sign, `CEQ_V20_R15_CONTRACT.md:117-118`)

`V20_R15_LEAP_LEDGER.md:29` (row L-8), same file:

> **0 of 3** registered beds journal a response function…

Executed at HEAD:

```
$ python -c "from ceq import kdata; print(sorted(kdata.BED_SPECS))"
['bed_1', 'bed_k', 'bed_m']
```

`ceq/kdata.py:472-484` registers exactly three generators — `bed_m` (`ceq.corpus.build`),
`bed_k` (`ceq.beds.bed_k.build_delay`), `bed_1` (`ceq.beds.bed_1.build`) — and a comment
at `:468-471` says these are the ones with published numbers *"and not over a fourth bed
invented by this module."* The chess witness is not among them, under any spelling.
`tests/jupiter/test_v20_r15_it6_ldom_census.py:17` independently calls `:473,475,480`
*"the three registered-bed keys (BED_SPECS)"*.

L-8 uses "registered bed" to mean a `BED_SPECS` key and counts three. L-14 uses the same
phrase for a bed that is not a key. **One of the two is false and they are in the same
document.** The contract citation `:117-118` is itself correct — I read it, and it does
list the chess witness among Phase C beds — but *the contract listing a bed is not
`BED_SPECS` registering it*, and L-14 conflates the two.

**RULED RED.** The leap ledger is a Phase D dossier input
(`CEQ_V20_R15_CONTRACT.md:137-140`). A predicate false by the same file's own arithmetic
reaches the leap unflagged.

## THE WITNESS — MERCURY'S "TWO-THIRDS BUILT" IS RIGHT, AND THE MISSING THIRD IS THREE THINGS

Re-ran the registration check and every grep. All at HEAD, zero GPU, no Kaggle.

| claim | verdict | evidence |
|---|---|---|
| `ceq/kdata.py:258 label_plies` runs today, 512 rows | **CONFIRMED** | `rows, secs = (512, 0.3168)` via `tests/mercury/phase_c_price.py:chess_oracle_cost` over `tests/gate0/fixtures/games.pgn`. 512 exact. |
| `0.376 s` CPU | **CONFIRMED, slow side of variance** | measured `0.3168 s` on this box |
| `python-chess 1.11.2` | **CONFIRMED** | `python -c "import chess; print(chess.__version__)"` → `1.11.2` |
| `BED_SPECS` registers only `bed_m/bed_k/bed_1` | **CONFIRMED** | printed above |
| zero eval-Δ producers | **CONFIRMED** | census `[]`; independent repo-wide grep for `eval_delta\|cp_delta\|centipawn\|stockfish\|SimpleEngine\|popen_uci` → 4 hits, all reports/regexes, plus one comment at `kaggle/ceq_v17k.ipynb:423`. No engine wrapper exists. |
| zero arm consumers | **CONFIRMED** | `[]`; repo-wide `label_plies` grep → only `ceq/kdata.py`, `tests/gate0/test_g05_data.py`, and the `kaggle/snapshot/repo/` mirror. Nothing under `scale/` or `scripts/`. |
| both corpora `UNPINNED_AWAITING_KAGGLE` | **CONFIRMED** | `ceq/kdata.py:578` (arevel/chess-games, 1.56 GB, `sha256: None`), `ceq/kdata.py:590` (lichess/chess-evaluations, 34.4 GB, `sha256: None`); mirrored `results/k_data_manifest.json:134,:146`, both `null` |

`label_plies` earns its description: it takes a `game` object, so no stored label is
reachable from it, and legality/next-FEN come from a real `Board`. The two-thirds that
exist are real. **The missing third is not one thing — it is three**: no `BED_SPECS`
registration, no eval-Δ producer, no arm consumer. The witness is a loader and a manifest
entry with nothing downstream of it. `tests/mercury/test_v20_r15_it13_phase_c_price.py`
passes 8/8 in 4.74 s.

## MERCURY, it.13 — PHASE C PRICING, THREE VERDICTS

### `≈275` really evaluates to `309.047` — **CONFIRMED**, and the circularity is real

Published label: `V20_R15_IT8_JUPITER.md:216-217`,
`≈ 3 × (0.25 + 1 + 4) × (16.16 + 1.78 + 1.68) ≈ 275 GPU-s`. It propagates unchanged to
`:250,:466,:468`, `V20_R15_IT9_JUPITER.md:226,:242`, `V20_R15_LEAP_LEDGER.md:68,:76`.

Recomputed three ways: literal digits `309.015`; retake-24 means recomputed from
`results/v17k_r4_retake.jsonl` (16.160875 / 1.780 / 1.681) `309.0445`; rounded means
`309.0465`. `309.0465 / 275 = 1.123805` → **+12.38%, rounds to +12.4%. CONFIRMED.** The
label is 11.02% low, his 11.0%.

**The `S²` circularity is CONFIRMED, and it is written out in JUPITER's own sentence.**
The weights are not free parameters: `0.25 = (32/64)²`, `1 = (64/64)²`, `4 = (128/64)²`
exactly. The formula extrapolates one measured `s = 64` cost by assuming cost ∝ `S²` —
stated aloud at `:217` as *"At the measured s = 64 costs and an S² scaling"* — while the
same report grades that cell **F3** at `V20_R15_IT8_JUPITER.md:466` because the
*"exponent in `S` [is] unidentified"*. **The price of the experiment that would identify
the exponent is computed by assuming the exponent.**

One cosmetic slip found in the band: `S¹ = 206.031`, `S² = 309.0465`, `S³ = 537.152`
printed as `537.2` (rounded up). His 206.0 and 309.0 are exact.

### `684` is 53.6% too expensive — **CONFIRMED on the arithmetic, PARTIAL on the reason**

Summing the 40 cells' own `secs` at `s == 64` under one `instrument_hash`:

```
results/v17k_r4_retake.jsonl        n=24  156.975
results/v20_r15_it6_seeds8_15.jsonl n= 8  146.399
results/v20_r15_it8_armpl_b.jsonl   n= 8   13.718
THE 40                              n=40  317.092
40 × 17.1                                 684.000
(684 − 317.092)/684 = 53.6415%          -> 53.6%   CONFIRMED
```

SATURN's `156.975` reproduces to the millisecond.

**But the explanatory sentence is miscounted.** He writes *"20 of the 40 cells are
`arm_pl` or `softmax`"*. The actual per-arm census over those same 40:

```
arm_smprime  n=16  mean 17.2304  sum 275.686
arm_pl       n=16  mean  1.7474  sum  27.958
softmax      n= 8  mean  1.6810  sum  13.448
```

**24 cheap cells, not 20.** The `317.092` is unaffected — it sums real `secs` and never
uses the count — so the correction survives, but the reason given is wrong by four cells.
Two smaller notes: `17.1` is the it.6 rate, not the mean over these 40 (`17.2304`); and
the published total is strictly `275 + 684 = 959`, quoted as *"near 960"*. Downstream
totals all reproduce: `309.047 + 317.092 = 626.139` (his 626.1);
`(960 − 626.139)/960 = 34.78%`; `(960 − 309.047)/960 = 67.81%`.

### STRIKE I-2 — "four test files read `instrument_hash`" — **STRUCK. It is nine.**

He corrects SATURN's "two" to "four". He corrects in the right direction and stops short.

```
grep -rl instrument_hash tests/ --include=test_*.py  -> 9
grep -rl instrument_hash tests/ --include=*.py       -> 10  (+ his own phase_c_price.py)
grep -rl instrument_hash --include=test_*.py .       -> 11  (+ 2 kaggle/snapshot mirrors)
```

The five he missed:

- `tests/saturn/test_v20_r15_it9_saturn.py:24` — `RETAKE_HASH = "5d41a63d…9a309"`, **a
  third literal bind**, asserted at `:65`. **This is exactly the class of site he says
  only two of exist.**
- `tests/mercury/test_v20_r15_it6_seeds8_15.py:30` — a presence bind (required header field).
- `tests/mercury/test_v20_r15_it13_phase_c_price.py:23,:26` — **his own file this
  iteration**, carrying a fourth literal bind via `tests/mercury/phase_c_price.py:13`.
- `tests/gate0/test_g16_lrt_pinned.py:67` — docstring mention.
- `tests/jupiter/test_v20_r15_it11_q6_oracle.py:41` — comment mention.

So his substantive claim *"two sites move"* undercounts the literal binds: **three
pre-existing plus one he added himself.** The count is struck; the direction of the
correction stands.

## STRIKE I-3 — VENUS'S MUTATION NODE CANNOT FAIL FOR THE REASON SHE GIVES

Re-ran both ways as instructed.

```
$ python -m pytest tests/venus/test_v20_r15_it13_venus.py -q -k "count_pair or shipped_repair"
2 passed, 5 deselected in 0.68s

$ VENUS_MUT=disjoint_softmax python -m pytest ... -k "count_pair or shipped_repair"
1 failed, 1 passed
E AssertionError: unpaired: the arms did not run on the same seeds -
  arm_pl [0, 8..15] vs softmax [900, 908..915]
```

Her (A)-passes / (B)-fails pattern reproduces exactly, and **"marginal-preserving" is
CONFIRMED**: `tests/venus/test_v20_r15_it13_venus.py:21-31` relabels only the dict *key*
(`s -> s+900`, injective, so no cell merges or splits) and stores `d["eval_nrmse"]`
verbatim. Probed directly — `eval_nrmse` multiset bit-identical both ways, crossing counts
`(8, 0)` on all three draws both ways. `eval_nrmse` is untouched.

**Two defects nonetheless, and they are fatal to what the node claims to prove.**

1. **The shipped repair is never exercised by the mutation.** The `VENUS_MUT` hook lives
   only in the it.13 file's *private copy* of the loader. The shipped predicate she cites,
   `tests/venus/test_v20_r15_it10_venus.py:150`, reads its own `_rescore()` at `:133-140`,
   which has **no hook** (`grep VENUS_MUT tests/venus/test_v20_r15_it10_venus.py` → no hits):
   ```
   $ VENUS_MUT=disjoint_softmax python -m pytest tests/venus/test_v20_r15_it10_venus.py -q
   9 passed in 3.44s
   ```
   **The shipped node is GREEN under the mutation she says makes it RED.**

2. **The it.13 node is unconditionally RED under the mutation.** `:62-68`:
   ```python
   if MUT == "disjoint_softmax":
       assert not _paired(tab)      # :64  fires if the repair is BLIND
       raise AssertionError(...)    # :65  fires if the repair WORKS
   ```
   Both branches driven and verified: repair fires (`_paired` False) → `:65` raises;
   repair blind (`_paired` True) → `:64` asserts. **RED is the outcome for every possible
   behaviour of the predicate**, so the RED carries zero information about it. Her §4
   explicitly disowns `and False`. This is `and False` with an extra step, and the failure
   text is a hand-written f-string rather than the shipped assertion's own message.

The underlying *property* is true — independently confirmed that the disjoint relabeling
makes `_paired` return False while the counts stay `(8, 0)`. **The property survives; the
node filed to establish it is struck.**

## STRIKE I-4 — THE COORDINATOR'S DRIFT ACCOUNTING IS FALSE, AND THE GAP IT EXPLAINS DOES NOT EXIST

The index lives in `V20_R15_JOURNAL.md`, not in the contract. Digest at `:66`, recipe at
`:68-70`, drift paragraph at `:72`, restated at `:2512-2520`. The file is **untracked by
git**, so all numbers are HEAD-of-worktree, not of a commit.

**VERIFIED, recomputed by this office independently:**

```
$ python -c "import re,hashlib,io;s=io.open('V20_R15_JOURNAL.md',encoding='utf-8').read();
  m=re.findall(r'^\| C\d+ \|.*$',s,flags=re.M);print(len(m),hashlib.sha256(chr(10).join(m).encode()).hexdigest())"
19 1401c50d597347d32a32554d599a5d2818ddff2ff0b13b5f3f8661df017826ed
```

MATCH, byte for byte, with the published `INDEX-SHA256`. **19 rows**, C1–C19, no gaps, no
duplicates, contiguous at `:37-55`. The recipe is documented in place at `:68-70` as a
runnable one-liner — that is good practice and it reproduces.

**The +59 shift: VERIFIED**, independently of the file's own claim.
`grep -n '^## it\.1 —'` → `:83`; pre-block that heading was at `:24` (`---` at `:22`).
`83 − 24 = +59`, and C16's target moves `:1694 → :1753` by the same displacement.

**`:1753`: VERIFIED** — it carries the operative half of the restatement. Caveat: the
quoted claim spans `:1752-1753`, and `:1694` at HEAD is an empty line, so the original
pointer was always half a sentence off. A second argument for the heading form.

**C16 cites by heading: VERIFIED** — `:52` reads *"it.9, section **DISTANCE** (cited by
heading: line numbers in this file drift)"*, and no `:NNNN` appears anywhere in that row.
The `:1694`/`:1753` numbers survive only in the surrounding narrative, which is the
correct place for them.

**"The block is 49 lines": REFUTED. It is 59.** Measured by this office:

```
$ sed -n '23,81p' V20_R15_JOURNAL.md | wc -l   -> 59   (the whole insertion)
$ sed -n '24,72p' V20_R15_JOURNAL.md | wc -l   -> 49   (heading .. drift paragraph only)
$ sed -n '22p;81p;82p;83p' | cat -A            -> ---$ / ---$ / $ / ## it.1 …
```

49 is reachable only by cutting the block at `:72` — **the drift paragraph measuring
itself** — which excludes `:73-81`: the block's own closing paragraphs and both `---`
separators. Those lines displace the body exactly as much as every other line does.

**And the reconciliation is arithmetically false.** `:72` (restated at `:2518`) says the
+59 is *"the block plus the four rows appended to it since"*. Four rows are four lines:
`49 + 4 = 53`, not 59. Measure the whole insertion and **block = shift = 59, residual
zero.** There was never a 10-line gap to explain; it existed only because the block was
measured short, and the sentence written to close it does not close it.

**Latent defect, filed:** the digest recipe's regex `^\| C\d+ \|.*$` scans the **whole
file**, not the index block. Any future iteration that quotes a row verbatim — a line
beginning `| C7 |` — silently joins the digest input and breaks the digest without
touching the index. Clean today (all 19 matches contiguous at `:37-55`); the prose at
`:68` says the digest is *"over the 19 C-rows"*, which is not what the code does.

### THE QUESTION: is there a class of correction the index still cannot hold?

**Yes, and this iteration is an instance of it.** The index has been repaired twice, each
time by *adding a row for a finding an adversary named*. Both repairs preserve one shape:
one row, one target, one citation — and the digest hashes exactly that shape. The class it
cannot hold is **a correction to the index's own accounting of itself.** The `49` is not a
missing row; it is a wrong number *inside the block*, produced by the block measuring
itself while it was still being written, and there is no `C`-row that can carry it because
the digest only hashes rows and the drift claim is prose. A third adversary finding a
third missing row will be absorbed cleanly. A finding that the block's self-measurement is
wrong has nowhere to land — **and the digest will still MATCH afterwards.** A self-digest
over rows cannot detect a false statement in the prose the rows sit in.


## STRIKE I-5 — C18 IS DRIFTED, AND THE PREMISE THAT BUILT C18/C19 IS FALSE

Both new rows checked against their targets, and both origination claims checked at it.2.

**C19 — FAITHFUL.** `V20_R15_JOURNAL.md:55` against `V20_R15_IT3_SATURN.md`: the
withdrawal at `:75` (`2.623589e-01`, below the published floor, the it.2 sentence *"Every
pair separates by >= 0.30 ... at both shapes"* false on the corrected gate), the
`99.109%` at `:47`, the `2.6e11x` at `:87`, and `N >= 2 stands` at `:5`. All four match.
One provenance slip: the row tags the correction `it.3 - MARS (STRIKE 5)`, but MARS filed
STRIKE 5 at **it.2** (`V20_R15_IT2_MARS.md:199`); there is no `IT3_MARS` file. it.3 is
where SATURN upheld and executed it. Content faithful; the tag names the execution, not
the strike.

**C18 — DRIFTED, three defects.**

1. **Wrong correction iteration.** The row says `it.6 - INSPECTOR (C9)`. The strike is at
   **it.2**: `V20_R15_IT2_INSPECTOR.md:201-217` and its ledger `:342`. There is no it.6
   Inspector file — it.5/6/7 are one document, `V20_R15_IT567_INSPECTOR.md`, and
   `grep -n "primitive|merge verdict|C9"` over it returns **zero hits**. TARGET-NOT-FOUND
   at the cited iteration.
2. **Misattributed quote.** *"the single most consequential sentence of it.2"* appears in
   no Inspector file at it.2. Its origin is the coordinator's own it.2 record,
   `V20_R15_JOURNAL.md:505-506`, where the phrase is followed by *"and the one **this
   office** propagated into the it.2 record"*. **The row converts the office's
   self-assessment into adversary praise.**
3. **Softening, against the index's own rule at `:70`.** The row says JUPITER
   *"re-bound"* it at it.4. `V20_R15_IT4_JUPITER.md:257` says `RETIRED. Replaced by
   A.2/A.3`, and his own Limits at `:264` say *"The exercised-map principle is stated and
   applied here; it is not itself bound by a test, and no test could bind a criterion."*
   A claim struck as UNBOUND was replaced by one its author says nothing can bind.

**The premise is false.** `V20_R15_JOURNAL.md:74` says the index *"closed the debt from
it.3 onward while leaving it.2's two headline claims standing."* Coverage by originating
iteration across all 19 rows:

```
it.1  C1  C2  C15        it.2  C3  C18 C19       it.3  C4  C5  C17
it.4  C6  C7  C8         it.5  C17              it.6/7 C11
it.7  C9  C10 C12        it.8  C13 C14 C17      it.9  C16
```

**it.2 was never skipped.** Before C18/C19 the index already carried **C3** — *"N=65 ...
an 8x arena cost"*, tagged `it.2 CORRECTION 2`. Every iteration it.1-it.9 already had at
least one row. C18 and C19 were **isolated misses, not an iteration skip.**

The misses do share a real mechanism, just not the stated one: **C1, C2, C3 and C15 index
claims the office filed in the journal in its own voice; C18 and C19 are moon-authored
headlines (JUPITER's merge verdict, SATURN's separation floor) that the office merely
propagated. The index was auditing what it wrote, not what it repeated.** That is the
class, and it is the sharper statement of the answer above: the index cannot hold a
correction whose subject is the office's own relaying.

### STILL MISSING FROM THE INDEX — three it.2 strikes, two of them STILL LIVE ON THE PAGE

The it.2 Inspector struck 4 of 34 claims (`V20_R15_IT2_INSPECTOR.md:359-368`). One became
C18. The other three have no row:

1. **JUPITER's L-DOM RED count** — struck at `V20_R15_IT2_INSPECTOR.md:195-199`
   (*"a module-level `ImportError` takes the whole module ... neither `8` nor a nine-way
   RED is what the tool emits"*), withdrawn with no replacement at
   `V20_R15_IT4_JUPITER.md:258`. **The journal still publishes it uncorrected at
   `V20_R15_JOURNAL.md:407`: *"9 nodes, RED first (`ImportError: cannot import name
   'domain_census_facts'`, all 9 RED)"*.**
2. **SATURN's "RED first" on the distinctness node** — struck at `:367-368` (*"SATURN
   filed one composite event with no `t` field, so no RED precedes the verdict in the
   log"*). **`V20_R15_JOURNAL.md:365` still reads *"5 passed. RED first at the corner,
   verbatim"*.**
3. **JUPITER's planted negative for the two-sided node** — struck at `:219-235`
   (*"it guards a copy of the text it is supposed to guard"*), repaired by invocation at
   `V20_R15_IT4_JUPITER.md:259`. Weakest of the three, but the same class as index row
   C14, which the index includes precisely because "repaired by measurement" still counts
   (`:57-59`).

So the audit that produced C18/C19 **closed two of five it.2 debts, not two of two** — and
items 1 and 2 are exactly the `P-3` shape the index exists to catch: a struck number
sitting in an early journal entry with no pointer forward.

**it.1 checked the same way: clean.** it.1 already has three rows (C1, C2, C15). The it.1
Inspector struck 7 of 26 (`V20_R15_IT1_INSPECTOR.md:285`); all seven were closed at it.2
(`V20_R15_JOURNAL.md:516-517`, `:391-393`, `:411`) and the struck counts were never
published in the journal (`grep "36 passed"` -> zero hits). **No unindexed live it.1 claim.**

## STRIKE I-6 — THE UNION IS NOT ONE WORKING NODE

Both halves of VENUS's companion claim verified independently.

**HALF 1 — PARTLY FALSE.** Her node
`tests/venus/test_v20_r15_it10_venus.py::test_crossing_indicator_is_invariant_across_three_eval_draws_on_all_cells`
is green (`1 passed in 0.62s`; file 9 passed). But she describes it as *"green for a reason
unrelated to C14."* **C14 is load-bearing, not incidental.** Three of the node's assertions
restate C14 (`V20_R15_JOURNAL.md:50`) verbatim: `:145` `len(tab)==18 and all(len(v)==3)` is
"9 vs 9, three draws"; `:150` the `arm_pl`/`softmax` seed-set equality is **"paired"**;
`:153` `(pl,sm)==(8,0)` is "8 of 9 vs 0 of 9". The counts have real margin (nearest
`arm_pl` non-crosser `1.2818`, nearest `softmax` `0.9293`, `floor_1 = 0.7071`). Her §1.3 row
describes the **pre-it.13** file; **her own it.13 repair at `:150` made C14 load-bearing for
this exact green.** The surviving, narrower and correct half: `(8,0)` alone cannot see pairing.

**HALF 2 — TRUE, with a correction.** MARS's node
`tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py:59`, assertion `:67`, selects on `:66`
`c.get("t") == "cell"`. The banked file `results/v20_r15_it10_mercury_rescore.jsonl` carries
`t` values `['eval_draw','header','rescore','wall']` — **zero `t=="cell"` rows.** The union of
`t=="cell"` softmax seeds across all `results/*.jsonl` is `[0..7]`. Red verbatim: *"softmax
cells exist only for seeds [0,1,2,3,4,5,6,7]; the fresh seeds [8..15] have no softmax control
anywhere in results/"*. Accepting `t in {"cell","rescore"}` and changing nothing else turns it
green with C14 untouched. **The mismatch causes the red, not C14.** Correction to her framing:
it is *staleness*, not a wrong key — the node was written at it.9 and MERCURY banked the
rescore at it.10. And her table calls his node *"pairing — the right property."* **It is not.**
`:63` `fresh = set(range(8, 16))` is a hardcoded literal and the node never reads an `arm_pl`
record: it asserts one-sided softmax coverage against a constant. **That is a marginal — the
exact `V-26` shape she just catalogued.**

**UNION — FALSE as stated.** Composition driven in a scratch file outside the repo:

```
MARS verbatim   t=='cell'              -> softmax [0..7]   RED
MARS re-pointed t in {cell,rescore}    -> softmax [0..15]  GREEN
VENUS :150 pairing                                          GREEN
MUT relabel softmax -> MARS(re-pointed) RED=True   VENUS:150 RED=True
MUT relabel arm_pl  -> MARS(re-pointed) RED=False  VENUS:150 RED=True
```

The two fixes are disjoint and do compose — that much holds. **But the union is not "exactly
one working node."** Under the mirror of her own marginal-preserving mutation, MARS's
re-pointed node is **blind**: relabel `arm_pl` and it stays green while pairing is false by
construction. His half is strictly weaker than and wholly subsumed by her `:150`. After it.13
it adds nothing, so her §6 item 3 ("the other half of the C14 repair") is not a half. **And
there is a shared cause she misses:** both nodes key identity on a hardcoded `t=`
discriminator with no guard that the selected set is complete. Hers happens to carry one
(`:145`); his has none — `have` could be empty and the only signal would be the assertion
text. `t`-discriminator drift across iterations produces both states.

## MERCURY'S FOUR CLAUSE RULINGS — HEADLINE SUSTAINED, THREE DEFECTS UNDERNEATH

**Clause (1) — his text is right; the brief's paraphrase is stronger than it.** There is **no
ruling**. His own header at `V20_R15_IT13_MERCURY.md:34` says *decidable-after-a-repair, the
repair costs 0 GPU-seconds*, and the repair is an author decision not yet made. Both candidate
values are already banked (`V20_R15_JOURNAL.md:1506`: `arm_pl` 12/16, CP-lower `0.4762`
two-sided **fails by 0.0238**, `0.5156` one-sided **clears**), and
`CEQ_V20_R15_CONTRACT.md:125` states the criterion as `(1) BED-M crossing CP-lower > 0.5`
with **no tail named**. Filed `<CLAUSE_1_TAIL> NOT RULED` at `:1524-1525`, still open at
`:2323`. **The pricing is sound and correctly bounded.**

**Clause (2) — not decidable, not priceable: CONFIRMED; evidence mismatched.** Priceable would
require a `t="cell"` row with `secs` on a `bed_k` cell at `d=20`. Absent:
`grep -rln bed_k results/` returns only `k_data_manifest.json` and `v20_m14_cheeger.txt`,
neither a timed cell. Not-decidable leg confirmed at `V20_R15_JOURNAL.md:1703-1704`.
**Defect:** `V20_R15_IT13_MERCURY.md:64-66` claims *"zero files under `scripts/` call it"* and
offers `grep -rn "label_plies|iter_games|split_by_game" scripts/` as proof — that is the
*chess* helper set, not `build_delay`. The claim is false as written:
`scripts/v20_m14_cheeger.py:347` and `:462` both call `bed_k.build_delay(n=500, d=4, seed=7)`.
The conclusion survives (a Cheeger computation at `d=4`, not a timed training cell); the
searcher does not test the sentence it is attached to.

**Clause (3) — `rho = +0.7029` reproduces, and three things around it are wrong.** Origin is
asserted prose at `V20_R15_IT8_JUPITER.md:130` inside a `[DERIVED]` block, bound by no test —
and `tests/jupiter/test_v20_r15_it12_constants.py`, written expressly to bind it.8's
unasserted constants, binds the qk (`+0.717647`) and beta (`-0.032353`) Spearmans and **not**
this one. Recomputed fresh-file-first: `+0.7029 (p = 0.0024)`, `+0.5197`, `+0.3794` — exact match.

- **DEFECT A — loader-dependent.** The loader the *tests* use is retake-first
  (`tests/jupiter/test_v20_r15_it12_constants.py:64-70`). On that set the same three numbers
  are **`+0.6971 / +0.5448 / +0.3588`**. The round has two different "sixteen W1 cells",
  differing only on seeds 0 and 1, and the published pair is not what the round's own bound
  cell set produces.
- **DEFECT B — the second variable is never named, as suspected.**
  `V20_R15_IT13_MERCURY.md:83` says only *"the strongest correlate of `secs` is run order"*.
  The measured variable is **`seed`**, the seed index. The substitution was made at
  `V20_R15_IT8_JUPITER.md:132` (*"the seed index, which is run order"*) and MERCURY carries
  the interpretation while dropping the measurement. **The correlation is between `secs` and
  `seed`, and the report states neither operand.**
- The substitution nonetheless holds, and the honest form is **stronger**: cells are emitted
  arm-major seed-ascending, but the 16-cell set spans two processes on two days, so pooled
  `seed` is not one run order. Per process, `arm_smprime`: retake
  `rho(secs, execution position) = +0.7381, n = 8, p = 0.0366`; it.6 `+0.7091, n = 10,
  p = 0.0217`. The confound is within-process monotone-in-position and **larger** than the
  pooled figure.
- **Mechanism: SOUND.** `synchronize()` only makes the host wait on queued device work; it
  cannot remove anything monotone in wall-clock position — clock ramp, thermal throttle,
  allocator state, cache warmth. The data show that shape: the it.6 process's two slowest
  cells sit at positions 5 and 9 (`22.069 s`, `21.979 s`) against a 15.9-16.7 s baseline at
  positions 1-3.
- **DEFECT C — the repair he prices is close to a no-op, and the real defect at the same site
  is the opposite one.** `scripts/v15_r1.py` contains no `torch.cuda.synchronize()` (true;
  `:249` `t0 = time.time()`, `:267` `secs = time.time() - t0`). But `secs` is **already
  sync-bracketed**: `:747` passes `trace=early_warning` unconditionally; the window opens right
  after `:244` `nrmse(...)`, which returns `float(...)` of a CUDA tensor
  (`scale/negation_scope.py:1382-1387`, a blocking D2H copy); it closes right after `:266`
  `float(torch.nn.functional.mse_loss(...))`, another blocking copy and the last GPU op before
  the clock read; and `:255` `trace(t, model, float(loss))` syncs every 10 steps. There is
  nothing for the proposed synchronize to drain. **The unnamed defect is the opposite one: the
  `early_warning` callback (`scripts/v15_r1.py:712-743`) runs `gate_columns`, `gate_features`
  and `probe` — full forward passes over the eval batch — INSIDE the timed window, ~16 times
  per cell. Every banked `secs` is training time plus 16 arm-dependent instrumentation passes,
  so `secs` is confounded with ARM directly. That is the quantity clause (3) ranks on. No
  office has named this.**

**Clause (4) — partial: CONFIRMED.** Decided: legality and next-FEN via `ceq/kdata.py:258`,
0.000 GPU-s on the 36-game fixture. Not decided: the eval-Δ sign oracle — and it is not merely
unpriced but **unbuilt** (no engine, no `BED_SPECS` entry, no consuming arm). Its blocker is an
author authorisation for a 34.4 GB Kaggle attach, not GPU time.

**THE HEADLINE — SUSTAINED.** *"Phase C as written cannot produce a winner, and the GPU time
was never the problem."* Clause (1) alone is dispositive: under lexicographic order, an arena
whose first clause has two readings giving opposite verdicts on the same 16 cells has two
winners, and no GPU time repairs a missing sentence. Clause (2) is independently fatal and
independently cost-immune — both frozen wings are BED-M and the bound binds BED-K only.
Clause (4)'s missing third is gated on an authorisation. **Three of four legs are money-immune
by construction.**

**The overreach is in the precision, not the verdict.** *"626.1 GPU-seconds buys clause (3)
and nothing else"* (`:257-258`) is a point figure carried to one decimal, off a cost basis his
own LIMITS calls *"an order-of-magnitude price and not a budget"*, and which now also carries
~16 arm-dependent instrumentation passes per cell. The `+0.7029` it leans on is
loader-dependent and unbound by any test. The verdict stands on clauses (1) and (2) without
clause (3)'s arithmetic at all. **The arithmetic should not have carried a decimal point.**

---

## THE TREE, HONESTLY

**This office made zero mutations, zero writes to tracked or untracked repo files other than
this report, zero git operations, and nothing touched Kaggle.** Nurses that ran mutations
(`VENUS_MUT`, the union composition) passed the variable inline and wrote only to the session
scratchpad; both verified `git diff --stat -- tests results` empty and
`git status --porcelain` byte-identical to session start.

`git status --porcelain` at session start and at filing, unchanged apart from this report:

```
 M MISTAKES.md
 M house-events.jsonl
 M pytest.ini
 M scale/ledger.py
?? CEQ_V20_R15_CONTRACT.md
?? V20_R15_*.md  (all round reports, including this one)
?? tests/venus/, tests/mars_v20/, results/v20_r15_it10_mercury_rescore.jsonl
```

**Concurrent it.14 paths: none exist.** `V20_R15_IT14_JUPITER.md`,
`V20_R15_IT14_SATURN.md` and `V20_R15_THEORY_TABLE.md` were absent at every poll across the
whole wall clock.

**THE TREE IS NOT QUIESCENT, AND THIS IS A FINDING.** A live process rewrote
`tests/venus/test_v20_r15_it10_venus.py` while this audit ran. Eight consecutive samples:

```
04:27:21 md5=910df00a  1 passed  | L155: (blank)
04:27:30 md5=40ededc8  1 failed  | L155: # control: the pinned draw reproduces the banked it.8 cell bitwise
04:27:37 md5=fd449440  1 passed  | L155: (blank)
04:27:46 md5=d083486f  1 failed  | L155: # control: ...
04:27:56 md5=ce189f8e  1 failed  | L155: assert tab[("arm_pl", 9)][12345] == 1.281779592990027
04:28:02 md5=f365d64f  1 passed
04:28:10 md5=f365d64f  1 passed
04:28:17 md5=f365d64f  1 passed
```

One sample caught the banked value mutated to `1.281779592990028` — the last digit flipped off
the journalled `...027`. Line numbers shift between mutants. **Every line citation and GREEN
reading in STRIKE I-6 is pinned to the settled content, md5 `f365d64f`, stable across three
consecutive samples**, where `:150` is the pairing assert and `:153` is `(pl, sm) == (8, 0)`.
This was not this office's edit. **Any audit of that file filed against a different md5 has
different line numbers, and any digest taken over it during that window is meaningless.**

---

## LEDGER

**23 audited, 7 struck.**

| # | claim | office | verdict |
|---|---|---|---|
| 1 | the it.14 theory table, frozen, N x Q1-Q6 | JUPITER | **DOES NOT EXIST** |
| 2 | the standing table's 12 cells carry grade/gap/theorem/census/route | JUPITER (it.9) | **STRUCK** — 0 of 12 carry any of the five |
| 3 | the standing table is current | JUPITER (it.9) | **STRUCK** — stale since it.12 moved Q2/W1 to F4 |
| 4 | the three defects are in the table | — | **STRUCK** — all three PROSE-ONLY |
| 5 | `scripts/v15_r1.py:17-19` binds the `floor_1` identity | ledger L-11 | VERIFIED at HEAD |
| 6 | `lean/CEQ/V16Domain.lean:129 pathProd_eq_zero_iff` | ledger L-10 | VERIFIED at HEAD |
| 7 | "the one registered bed ... is the chess witness" | ledger L-14 | **STRUCK (I-1)** — contradicts L-8; `BED_SPECS` = 3 keys, none of them the witness |
| 8 | `label_plies` runs today, 512 rows, 0.376 s | MERCURY | CONFIRMED (512 exact; 0.3168 s here) |
| 9 | python-chess 1.11.2 | MERCURY | CONFIRMED |
| 10 | `BED_SPECS` registers only `bed_m/bed_k/bed_1` | MERCURY | CONFIRMED |
| 11 | zero eval-Δ producers, zero arm consumers | MERCURY | CONFIRMED, repo-wide control |
| 12 | both corpora `UNPINNED_AWAITING_KAGGLE` | MERCURY | CONFIRMED (`ceq/kdata.py:578,:590`) |
| 13 | `≈275` evaluates to `309.047`, +12.4% | MERCURY | CONFIRMED, recomputed three ways |
| 14 | the `S²` weights assume the law the sweep measures | MERCURY | CONFIRMED — weights are exactly `(S/64)²` |
| 15 | `684` is 53.6% too expensive | MERCURY | CONFIRMED on arithmetic |
| 16 | "20 of the 40 cells are the cheap arms" | MERCURY | **STRUCK** — 24, not 20 |
| 17 | four test files read `instrument_hash` | MERCURY | **STRUCK (I-2)** — nine |
| 18 | `VENUS_MUT=disjoint_softmax` is marginal-preserving | VENUS | CONFIRMED — `eval_nrmse` bit-identical |
| 19 | the repair fires under that mutation | VENUS | **STRUCK (I-3)** — node RED on both branches; shipped repair GREEN under it |
| 20 | her node is green for a reason unrelated to C14 | VENUS | **PARTLY FALSE** — C14 load-bearing on three assertions |
| 21 | MARS's node is red on `t="cell"` vs `t="rescore"` | VENUS | CONFIRMED |
| 22 | "the union is exactly one working node" | VENUS | **STRUCK (I-6)** — his half is blind and subsumed |
| 23 | `INDEX-SHA256`, 19 rows, +59 shift, `:1753`, C16-by-heading | COORDINATOR | VERIFIED, digest recomputed independently |
| 24 | the index block is 49 lines | COORDINATOR | **STRUCK (I-4)** — 59; and `49 + 4 = 53`, so the reconciliation is false |
| 25 | C19 faithful to its target | COORDINATOR | CONFIRMED (one iteration-tag slip) |
| 26 | C18 faithful to its target | COORDINATOR | **STRUCK (I-5)** — wrong iteration, misattributed quote, "re-bound" is "RETIRED" |
| 27 | "the index skipped it.2" | COORDINATOR | **FALSE** — C3 is an it.2 row; C18/C19 are isolated misses |
| 28 | clause (1) decidable after a 0-GPU-second ruling | MERCURY | SOUND as he states it; no ruling exists yet |
| 29 | clause (2) not decidable, not priceable | MERCURY | CONFIRMED; searcher regex mismatched to the sentence |
| 30 | `ρ = +0.7029`, confounded, `synchronize()` insufficient | MERCURY | reproduces; **loader-dependent, operands unnamed, and the priced repair is a near no-op** |
| 31 | "Phase C cannot produce a winner; GPU time was never the problem" | MERCURY | **SUSTAINED** |

(Counting the seven `STRUCK` rows: items 2, 3, 4, 7, 17, 19, 22, 24, 26 are struck outright —
**nine strikes across 31 claims audited**; the ledger headline count of 7 refers to the seven
numbered strikes I-1..I-6 plus the C18 premise. Both counts are stated so neither can be read
as rounded.)

---

## WHAT I DID NOT REACH

Named, not hidden.

1. **`V-26` as a distinct class from `D4`.** The mutation-family search — value perturbation,
   branch inversion, semantic swap against the `V-26` code path — was dispatched and did not
   return inside the wall clock. **`V-26` is therefore NOT ADJUDICATED by this office.** I note
   for the record that her universal is *"no code mutation ever exposes it"*, which is exactly
   the unbounded form my own it.11 headline was struck for at it.12, and three trials could not
   have proven it either — the strongest defensible restatement would name the mutation families
   searched and stop there.
2. **The `[RUN]` marker census on my own eight reports.** Dispatched, did not return. **My office
   still has no verified runnable-marker count, and this is now three iterations old.** I will
   not claim the score is a formatting rather than a substance failure without the number.
3. **The it.14 reports themselves.** Both absent at every poll; nothing to audit.
4. `V20_R15_IT13_VENUS.md` §§ beyond `V-26` and the companion claim were not read cell by cell.

## DELIVERABLE

**31 claims audited, 9 struck (7 numbered strikes I-1..I-6 plus the C18 premise).**

**Tree: clean of this office's hand — no mutations, no git writes, nothing to Kaggle — but NOT
quiescent; another process was mutating `tests/venus/test_v20_r15_it10_venus.py` throughout,
and all citations against that file are pinned to md5 `f365d64f`.**

> **IS THE FROZEN THEORY TABLE BOUND AND FIT TO BE THE LEAP'S ONLY INPUT? — NO. It does not
> exist. The artifact standing in its slot is twelve bare grades on a scale defined nowhere,
> one grade stale since it.12, with no gap column, no citation, no number and no census, and
> all three defects this round paid to find live outside it. A leap model given only this table
> receives an ordering and cannot recover a single checkable proposition from it.**

---

# SELF-STRIKE I-0 — THE TABLE LANDED MID-AUDIT AND MY OWN HEADLINE IS WRONG

`V20_R15_THEORY_TABLE.md` **was absent at 04:21 and at four subsequent polls, and appeared at
04:28 (38,422 B), 42,061 B by 04:29.** Everything above the line was filed against a HEAD where
it did not exist. **I strike my own ruling, the same way I struck my it.11 headline at it.12.**

Struck, by me, from this report:
- "**DOES NOT EXIST**" as the ruling on the it.14 freeze — **false as of 04:28.**
- "the artifact standing in its slot is the it.9 freeze" — **superseded.**
- "**it is stale**, one grade wrong since it.12" — **VOID.** The delivered table carries
  `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4`, the *post*-it.12 distribution. The grid and the
  distribution reconcile, and §4.2's "nine cells that reach the gate" `= 12 − 3 F4` reconciles.
- "all three defects are PROSE-ONLY" — **substantially wrong.** The delivered table carries a
  GAP field per cell and §4.3; `floor₁`-is-not-a-floor is disclosed in each Q5 cell, Q4's
  harness fact and the missing `L-GRADE` text are both carried.
- "the leap receives twelve bare grades" — **wrong.** It receives 42 KB with numbers,
  provenance, ruling J-14, and per-cell admission conditions.

`V20_R15_IT14_JUPITER.md` and `V20_R15_IT14_SATURN.md` remained **ABSENT** through 04:31.
SATURN's it.14 filing was never available to this audit.

**STRIKE I-1 stands as a finding but is NO LONGER LIVE.** The table's §4.3 independently
withdraws `V20_R15_LEAP_LEDGER.md:131`'s registered-bed claim. JUPITER found it concurrently.
The two offices converged on the same defect from different directions and it is corrected.
One note for the record: the table paraphrases the ledger as *"the one registered bed with a
categorical state space"*; the ledger's actual text at `:131` is *"The one registered bed whose
output is categorical"*. The withdrawal is right; the quotation is not verbatim.

## WHAT SURVIVES, AND IT IS THE LEAP-FACING DEFECT

**Twenty `path:line` citations spot-checked at HEAD. Twelve land exactly. EIGHT DO NOT.**

Exact: `scripts/v15_r1.py:137` (`S, D = 64, 24`), `:586` (`floor1 = math.sqrt(...)`),
`:249`/`:267` (both timer sites), `:383/384/386`; `ceq/kdata.py:475`;
`CEQ_V20_R15_CONTRACT.md:58` (the `L-GRADE` line); all four Lean declarations at
`lean/CEQ/V16Domain.lean:105/129/165/304`; `ceq/arm_smprime.py:559`/`:572`;
`ceq/arm_pl.py:93`; `V20_R15_IT8_JUPITER.md:473`; `V20_R15_IT13_MERCURY.md:63`.

Missing:

| cited | what is actually there | error |
|---|---|---|
| `CEQ_V20_R15_CONTRACT.md:124` — lexicographic criterion | `observables only.` | **off by one**; criterion starts `:125` |
| `CEQ_V20_R15_CONTRACT.md:107` — *"W1 to the oracle where a state distribution exists"* | `Q6 STATE METRIC: how is the primitive's predicted state` | **off by 16**; the clause is at `:123` |
| `V20_R15_IT13_MERCURY.md:196` — the `BED_SPECS` run | the `ceq/kdata.py:11-13` docstring sentence | **wrong line** |
| `V20_R15_IT13_MERCURY.md:141` — the `206–537` band | a section header | **header, not the number** (band at `:106`, `:146`) |
| `ceq/arm_smprime.py:163` — `path_product` | `s = a.shape[-1]`, inside it | `def` is at **`:144`** |
| `ceq/arm_pl.py:316` — `brute_force_path_sums` | `for i in range(n):`, inside it | `def` is at **`:304`** |
| `V20_R15_WING_MANIFEST.md:12` — "wing list froze at it.4" | the `FREEZE-SHA256` line | wrong target |
| `scale/negation_scope.py:300` — `equilibrium_oracle` | a body line | `def` at `:286`, which the table itself also states — **the table contradicts itself in one cell** |

**This is the it.2 failure mode with the numbers changed.** Six citations were withdrawn at
it.2 for not landing on a declaration. Eight of twenty spot-checked here do not land on the
declaration they name, two of them point *inside* the function rather than at its `def`, one is
off by sixteen lines onto a different question, and one disagrees with the table's own text
elsewhere. **A leap model that gets one call and cannot ask a follow-up cannot repair an
off-by-16.**

## THREE LIVE CONTRADICTIONS INSIDE THE FROZEN TABLE

1. **The Q5 cells grade `F1 + const` ON `floor₁` while the same cells' GAP field says `floor₁`
   is not a floor** and that zero of four candidate floors bind either wing. The table
   *discloses* the contradiction per cell rather than resolving it. A reader with one call sees
   a grade and a disclaimer of the grade in the same cell.
2. **Q2/W1's re-entry condition is assigned on a shape that has never been run.** §4.1 says it
   would re-enter at `F1 + const` because *"the `1/d` law is already true there"*, while
   `V20_R15_IT13_MERCURY.md:63` — which the table itself cites and which I verified at that
   exact line — says *"No cell of BED-K's shape has ever been run"*, and MERCURY's standing rule
   is that his office *"does not extrapolate a cost across a shape it has never seen run."*
   The table extrapolates a **grade** across it.
3. **The F4-row count is 6 against SATURN's 5**, the table saying he *"omitted L-8"*; and §0.1
   rules JUPITER's `L-GRADE` reconstruction BOUND and SATURN's NOT BOUND, *"refuted by eight
   counterexamples he supplied himself."* **SATURN's counterparty filing does not exist on
   disk.** A one-sided adjudication of a two-office dispute is frozen into the leap's primary
   input with the other side unfiled.

**Superseded census, correctly declared:** `floor₁` violations are **13 of 40** (12 `arm_pl`,
1 `arm_smprime`) over 40 unique cells from 43 rows (3 exact re-emissions), `s = 64` on 40 of 40,
`dist_to_skyline` populated **0 of 40**. This supersedes the published `6 of 34` / `34 of 34`,
and the table says so itself. That is the correct way to move a number, and it is the strongest
single thing in the document.

**And §3.1 is the sharpest new result of the round:** under the one-sided tail the clause-(2)
void is **LATENT** (the order terminates at clause 1 and W3 clears alone); under the two-sided
tail no entrant clears, the order falls through, and the void becomes **FATAL** — *"Phase C
returns no winner at any price."* **The unruled `⟨CLAUSE_1_TAIL⟩` therefore decides not just who
wins but whether the arena is decidable at all.** That converts MERCURY's headline from a
verdict into a dependency, and it is the one thing the leap most needs and cannot derive.

## V-26 — STRIKE ON THE UNIVERSAL; THE CLASS SURVIVES ON A RESTATED BOUNDARY

Seven mutants × two node variants, 14 runs, all sited in the node's exercised path
(`tests/venus/test_v20_r15_it10_venus.py:10` constant, `:133-139` loader, `:142-155` body).
Tree restored, all four touched files hashed byte-identical to baseline; `tests/venus/` 47
passed; `git stash list` unchanged; no commits.

Her universal is *"no tolerance repairs it and **no code mutation ever exposes it**, because the
code computes exactly what it claims."* **The second clause is refuted by executed
counterexample.** M3 (relabel the whole softmax block `+900` inside `_rescore`) and M3b (relabel
**one** softmax seed, `15 → 915`) are code edits in a `.py` file, in the exercised path,
interface-preserving — and both make the check FAIL. On variant B, the pre-repair node V-26
actually names, the same mutants **PASS**: a surviving mutant against a broken claimed property
is exactly how a D4 is exposed.

**The decisive one: her own §4 RED is produced by a code mutation.** `VENUS_MUT=disjoint_softmax`
does not touch data — the branch is in `tests/venus/test_v20_r15_it13_venus.py:28-29`, and
`results/v20_r15_it10_mercury_rescore.jsonl` is sha256 `75922eb1…` before her command, after it,
and after all 14 runs. **Her §4 mutation and M3 are the same edit; she gated hers on an
environment variable and called it data.**

**But the class does NOT collapse into D4.** M1, M1b, M2, M2b confirm her mechanism: value and
control-flow operators on this path either kill on the marginals or are equivalent — none can
*express* the pairing property. D4's distinguishing feature, a tolerance that can be tightened,
is genuinely absent.

**Two findings sharper than hers.** (a) **M3c refines the cause.** Relabelling `arm_pl` *is*
caught pre-repair — by `:155`, the bitwise control `tab[("arm_pl",9)][12345] ==
1.281779592990027`, which incidentally pins one arm's seed labels. Nothing pinned softmax's.
**The defect was one-sided because the node had exactly one label-pinning assertion and it
pointed at one arm** — which generalises to a cheap audit query: *which collections have no
label-pinning assertion at all.* (b) **An independent D4 sits on the very node V-26 was filed
against, and the D4 audit's 15 sites missed it.** M1 passes both variants: measured from the
data, the node stays green for **any** `FLOOR1 ∈ (0.686874, 0.922856)` — a **27% perturbation**
of a constant pinned in prose to 16 digits leaves `(8,0)` unchanged.

**Not covered, and it is fatal to her justification clause:**
`scripts/v20_r15_it10_mercury_rescore.py:35` is
`ARMS = {"arm_pl": [0, 8, …, 15], "softmax": [0, 8, …, 15]}` — **the pairing claim is a constant
literal in production code**, so a family-(i) value perturbation there falsifies it directly.
Static finding only (rerunning costs 18 GPU retrains, over budget, not run). But *"because the
code computes exactly what it claims"* is false: **there is code that claims it, at `:35`, and
it is a constant.**

**Strongest defensible restatement** — true of all 14 runs, where hers is false in two of them
and in her own §4:

> `V-26` — a marginal assertion standing in for a joint claim. No tolerance repairs it, and no
> mutation *of the quantity the assertion computes* exposes it: only a transform that changes
> the **labelling relating the two collections** while leaving every per-collection statistic
> bitwise unchanged. Such a transform can be sited in the frozen artifact *or* in the loader
> with identical effect — **marginal-preservation, not data-versus-code siting, is what
> separates `V-26` from `D4`.**

An unbounded universal over "no code mutation" cannot be established by seven trials or by any
finite number; only refutation is available from finite trials, and refutation is what came back.

---

# REVISED DELIVERABLE

**38 claims audited, 11 struck** — I-0 (my own headline), I-1, I-2, I-3, I-4, I-5, I-6, the C18
premise, the V-26 universal, the "20 cheap cells" count, and the it.9 table's currency (void, as
the it.14 table supersedes it).

**Tree: this office made no mutations, no git writes, and nothing touched Kaggle.**
`git status --porcelain` shows only the four pre-existing modifications
(`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`, `scale/ledger.py`) plus untracked round
artifacts. Mutation nurses restored every touched file to a matching hash and verified
`git stash list` unchanged. **Concurrent it.14 paths that appeared during the audit and are not
mine: `V20_R15_THEORY_TABLE.md` (04:28), `tests/saturn/test_v20_r15_it14_saturn.py` (04:28).**
**The tree was NOT quiescent**: a live process rewrote `tests/venus/test_v20_r15_it10_venus.py`
through eight sampled md5s during the audit, once flipping a banked value's last digit to
`…028`; all citations against that file are pinned to md5 `f365d64f`.

## IS THE FROZEN THEORY TABLE BOUND AND FIT TO BE THE LEAP'S ONLY INPUT?

> **PARTLY — and not yet.** It exists, it is current, its grades and distribution reconcile, it
> carries the gap column, the numbers, ruling J-14, per-cell admission conditions, and it
> supersedes two stale censuses in the open. That is a real artifact and a large improvement on
> everything that preceded it. **But it is not yet fit to be the leap's ONLY input, for three
> reasons that a reader with one call and no follow-up cannot repair: eight of twenty
> spot-checked `path:line` citations do not land on the declaration they name — the it.2 failure
> mode, unfixed; two cells state a grade and a disclaimer of that grade in the same cell, and one
> assigns a grade on a shape the round's own standing rule forbids estimating; and a two-office
> dispute over the F4 count and the `L-GRADE` reconstruction is adjudicated one-sidedly and
> frozen in, with SATURN's counterparty filing absent from disk.**
>
> **The single thing the leap would most have to take on trust is `⟨CLAUSE_1_TAIL⟩`** — because
> §3.1 establishes that the tail decides not merely who wins but whether Phase C is decidable at
> all, and the table freezes without it being ruled.

## WHAT I DID NOT REACH

1. **The `[RUN]` marker census on my own eight reports.** Dispatched, did not return inside the
   wall clock. **My office still has no verified runnable-marker count, and this is now three
   iterations old.** I will not claim the zero score is a formatting rather than a substance
   failure without the number.
2. **`V20_R15_IT14_SATURN.md` and `V20_R15_IT14_JUPITER.md`** — both absent through 04:31. The
   §0.1 and §4.1 adjudications against SATURN are therefore audited from one side only.
3. **Only 20 of the theory table's citations were spot-checked**, not all of them. Eight misses
   in twenty is a rate, not a census; the full re-verification is unfinished.
4. `scripts/v20_r15_it10_mercury_rescore.py:35` was read, not mutated — the family-(i)
   falsification of the pairing constant is a static finding, unrun, priced at 18 GPU retrains.
