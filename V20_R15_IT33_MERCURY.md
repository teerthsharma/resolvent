# V20 R15 — IT.33 — MERCURY (arena and measurement)

**`438` retired for a digest this office re-read. `309.047` withdrawn from all three
mercury addresses, the list grep-produced and asserted empty. Landings `53 → 60 of 129`,
CLAUSE D. One new RED: the line cited for *band only, NO POINT* is the last line quoting
the point.**

---

## 1. REPAIR 1 — THE `438` PIN IS RETIRED FOR A DIGEST, RE-READ NOT CITED

`tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:142` asserted
`len(_lines("V20_R15_LEAP_LEDGER.md")) == 438` under the failure message *"the ledger
started growing again"* — a node stating the defect it dies of. A cardinality cannot see a
rewrite.

**The digest was re-read by this office, not taken from SATURN's spec.**

    [RUN] 2026-09-02 19:36:19 IST — clock read by
          python -c "datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Kolkata'))"
    [RUN] python -c "import hashlib,pathlib;
          b=pathlib.Path('V20_R15_LEAP_LEDGER.md').read_bytes();
          print(hashlib.sha256(b).hexdigest(), len(b),
                len(b.decode('utf-8').splitlines()))"
    -> 6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485  45670  438

**The independent read agrees with SATURN's spec to the byte.** That agreement is the
content of the re-read: the figure is now this office's, not a `[CITED: SATURN]`.

The node now asserts the pair `(45670, sha256)`, and carries the repair's own negative
**entirely in memory** — `(("x" + chr(10)) * 438)` is 438 lines, passes the retired
predicate, and fails the digest. The retired predicate is therefore demonstrated
insufficient at the address that used it, not merely asserted to be.

    [RUN] python -m pytest ...it24_seal_and_manifest_gap.py::test_m24b_... -q
    -> 1 passed in 0.38s

**NOT REACHED:** the second pin, `tests/jupiter/test_v20_r15_it23_fence_and_argument.py:194`.
It is JUPITER's office and it is unstamped. Same digest applies.

---

## 2. REPAIR 2 — `309.047` WITHDRAWN, AND THE GREP FOUND AN ADDRESS THE STRIKE DID NOT

### 2.1 RED first, verbatim, against unmutated code

    [RUN] python -m pytest tests/mars_v20/test_v20_r15_it31_the_repairs_of_it29_it30.py\
          ::test_mars31c_the_m30a_withdrawal_has_the_same_reach_defect_as_62 -q

    E  AssertionError: M-30a names one address; 2 others still assert it:
       ['tests/mercury/test_v20_r15_it13_phase_c_price.py:48',
        'tests/mercury/test_v20_r15_it13_phase_c_price.py:55']
    1 failed in 0.89s

### 2.2 The address list, grep-produced — **MARS-31-C named 2; the scan finds 5, of which 3 are mine**

MARS-31-C scanned **one file**. A scan of all of `tests/` for lines whose first token is
`assert` and which carry the literal:

| address | kind | disposition |
|---|---|---|
| `tests/mercury/test_v20_r15_it13_phase_c_price.py:48` | live point | **withdrawn it.33** |
| `tests/mercury/test_v20_r15_it13_phase_c_price.py:55` | live point | **withdrawn it.33** |
| `tests/mercury/test_v20_r15_it30_admission.py:107` | live point | **withdrawn it.33** — *a file-scoped scan cannot see it* |
| `tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py:195` | `not in` — **enforces** the withdrawal | kept, it is not a quote |
| `tests/mars_v20/test_v20_r15_it31_the_repairs_of_it29_it30.py:179` | MARS's unmutated-instrument control | **out of office**, named not repaired |

**This is the `62` defect caught one iteration later on the withdrawal that named it.**
MARS's strike found two addresses because it looked in the file the strike was about; the
third lives in the same office, in the node that admitted the three numbers.

### 2.3 What the assertion is now

`tests/mercury/test_v20_r15_it13_phase_c_price.py`,
`test_the_point_on_the_band_is_withdrawn_from_every_assertion_in_this_office`:

* the literal is built as `"309." + "047"` so the node is **not its own hit**;
* **liveness**: `assert hits` — the scanner must return non-zero somewhere or it is broken;
* **empty**: the mercury-scoped list is `[]`;
* **exact partition, not a bound**: the surviving two are compared as a **sorted tuple of
  `(file, enforces)`**, so a backfilled deletion in either office turns this node RED
  rather than passing silently. `MONOTONE` is retired; this node does not use `>= N`.

### 2.4 The ruling applied is MARS's, and it costs this office a finding

Band and point are **one monotone family in `e ∈ {1,2,3}`**; containment is a consequence of
the ordering, never a measurement. **`206–537 GPU-s`, BAND ONLY, NO POINT.**

The node now asserts the ordering `lo < mid < hi` (the fact that makes containment free) and
the **endpoints** `(206.031, 537.152)`. It no longer asserts a midpoint.

**The price, stated rather than absorbed: it.13's `+12.4% against its own label` does not
survive the withdrawal.** `275` lies *inside* `206–537`, so the band cannot price the label
at all. What survives is that `275` is not an endpoint and that the exponent is unmeasured.

In `test_v20_r15_it30_admission.py` the three-number admission keeps `309.015` — the formula
**as written**, which is not on the band — and replaces the midpoint assertion with JUPITER's
closing detail, computed rather than quoted:

    assert round(literal, 3) != round(jupiter_it8_formula(), 3)   # two objects at 3 dp
    assert round(literal, 1) == round(jupiter_it8_formula(), 1)   # one number at 1 dp

That pair is the reason a rounded point cannot be reinstated: **at one decimal the point does
not carry the distinction it was made to draw.**

### 2.5 A node that was RED because the repair landed

`test_table_still_carries_the_stale_275_at_354` recorded `~275 GPU-s` at
`V20_R15_THEORY_TABLE.md:354` as a live defect. **JUPITER's it.32 edit landed it.** The node
is flipped to assert the landing, re-read at that address rather than cited:

    :354  ... LEAPABLE by 206–537 GPU-s, band only, no point (`V20_R15_IT13_MERCURY.md:146`) ...

and it now also asserts **no `309` came back at the address the band replaced**.

---

## 3. M-33a — NEW, **RED** — THE CITATION FOR *NO POINT* RESOLVES TO THE POINT

`V20_R15_THEORY_TABLE.md:184` **and** `:354` both carry the it.32 ruling and both cite
**`V20_R15_IT13_MERCURY.md:146`** as its authority. That line is this office's own it.13
RETIRE row:

    | **RETIRE the re-take (SATURN's split)** | **206 – 537** (point 309.0) | 4 — ...

**The pointer lands and the target asserts what the citing line withdraws** — at one decimal,
which by §2.4 is the one resolution at which the point is meaningless. The withdrawal was
applied in the citing office and never at the address the citation resolves to. This is the
same reach defect as `62` and as MARS-31-C, one hop further out: **not a second live
address in a test, but the cited target of the withdrawal itself.**

`tests/mercury/test_v20_r15_it33_band_only_and_clause_d.py::test_m33a_...` — **RED**,
verbatim above in the run log. Its control is GREEN and discriminates *the target is clean*
from *the reader is blind*: the reader is shown to see `(point 626.1)` at `:148`.

**Whose to repair:** the target is a mercury file, the two citing lines are JUPITER's it.32
edits. This office files it RED rather than editing an it.13 filing under a 20-minute clock;
the edit is one parenthesis at `V20_R15_IT13_MERCURY.md:146`.

---

## 4. LANDINGS CENSUS — **`53 → 60 of 129`. CLAUSE D. `7 of 7` LAND.**

**Clause D — the leap-row block, table lines `335–380`.** Disjoint from clause C
(`179–192`); the overlap is asserted empty in the node, so the 14 are not re-counted.

| table line | citation | what is there | verdict |
|---|---|---|---|
| `:335` | `V20_R15_IT567_INSPECTOR.md:482` | *"as the head noun, which is what the clause requires"* | **LANDS** |
| `:339` | `tests/saturn/test_v20_r15_it12_saturn.py:319` | `def inadmissible_leapable_rows() -> dict[str, str]:` | **LANDS** |
| `:346` | `tests/saturn/test_v20_r15_it12_saturn.py:329` | `if head.startswith("none") or not head:` | **LANDS** |
| `:347` | `V20_R15_LEAP_LEDGER.md:99` | `TERMINAL rows name no field, which is the rule.` | **LANDS** — verbatim |
| `:354` | `V20_R15_IT13_MERCURY.md:146` | the RETIRE row, `206 – 537` **(point 309.0)** | **LANDS — and contradicts, M-33a** |
| `:368` | `V20_R15_LEAP_LEDGER.md:130` | the `L-13` row, `F4 — the metric has no object on this bed` | **LANDS** |
| `:380` | `tests/saturn/test_v20_r15_it12_saturn.py:343` | `KNOWN_INADMISSIBLE = {"V-it7","L-2","L-9","L-13","L-14"}` | **LANDS** |

**`0` FAIL. `1` contradicts its citing line, named.** `7` is added; **`53 → 60`. No rate,
no extrapolation to `129`.** `:354` is scored a LANDING because **this census scores
addresses**: the pointer resolves. The defect is in the target's text and is filed
separately as M-33a, so that one finding is not counted twice in two currencies.

**The denominator has not moved in this census.** JUPITER's it.32 edits took the citation
population `131 → 133`; `129` is the census denominator this office has been counting
against since it.26 and is left untouched, because changing a denominator mid-count is the
`28 vs 25` defect. **A note for the next office, measured not asserted:** this office's own
extractor over `V20_R15_THEORY_TABLE.md` (443 lines) returns **122 pointer occurrences over
71 lines**, against the round's `129`/`133`. **Three populations, three rules, one name.**
That gap is unclosed and is not claimed as a finding here.

---

## 5. RADIUS

**Declared radius:** `tests/mercury/`, `tests/mars_v20/`, `tests/jupiter/`.

| command | before | after |
|---|---|---|
| `pytest tests/mars_v20/...::test_mars31c_...` | **1 failed** (verbatim, §2.1) | **passed** |
| `pytest tests/mercury/...it24...::test_m24b_...` | passed on a count | **passed on a digest** |
| `pytest tests/mercury/ -q` | *not taken before the edit* | `20 failed, 158 passed, 2 xfailed` |

`test_mars31c2_the_275_withdrawal_left_its_only_load_bearing_address_live` also passes —
JUPITER's it.32 table edit, not this office's work.

**LIMITS.**

* **The full-suite retake returned NO COUNTS, and the fault is in this office's own
  command.** `[RUN] python -m pytest tests/ -q -p no:cacheprovider --timeout=300 2>&1 | tail -5`
  was started at the top of the iteration and exited having printed only a `pytest-timeout`
  dump inside a `torch.autograd` backward pass — **no `N failed / M passed` line at all**.
  Two defects, both this office's: the run was piped through `tail -5`, which discards the
  summary whenever the run ends in a timeout traceback rather than a summary; and it was
  left running **across the edits it was meant to baseline**, so even a summary would have
  measured a moving tree. **The `93 → 91 failed` comparison is therefore NOT reproduced
  here, and no claim of `none broken` outside the declared radius is made.** The `20 failed`
  in `tests/mercury/` are in `it28_named_artifacts` and `it29_withdrawals`, neither touched
  this iteration, and they have **no before-count**. A clean retake is the first item for
  it.34: take it **before** the edit, keep the summary line, and pin the seed of the torch
  node that timed out.
* `M-33a` is filed RED and adds one failure by construction.
* `tests/jupiter/...it23...:194` (the second `438` pin) and
  `tests/mars_v20/...it31...:179` (the surviving point) are **named, not repaired** — other
  offices.
* No git writes. Nothing touched Kaggle.
