# V20 R15 · it.29 · SATURN (WATSON — instruments)

**Two repairs, both to this office's own prose. The instruments were sound; the paragraphs
beside them were not.** All readings `2026-09-02`, container clock, `13:03Z–13:10Z`.
Working tree at `207e7b9` **plus untracked files** — see §3, where that turns out to be the
whole story about tiers.

Instrument: `tests/saturn/test_v20_r15_it29_planted_negative_rederived.py`
(md5 `5524431bf49c701231a9baa07e833daf`, 7 nodes, **2 RED / 5 passed**, `[RUN] 13:09:44Z`).
Subjects: `tests/saturn/test_v20_r15_it27_wing_arm_citation.py` md5
`d4286376667b9e78eccd214a8e29c08b`, `V20_R15_LEAP_LEDGER.md` md5
`4ca866a027a1bc69433d0cd764078094`, `ceq/arm_pl.py` md5 `3770f8842c00af6ea95cf6ea4849415a`,
`ceq/arm_smprime.py` md5 `690b61470d2bbef37c536ba647b8755f`.

---

## 1. REPAIR 1 — what actually fires in the it.27 planted negative

The INSPECTOR's strike is upheld in both halves, by measurement against unmutated code.

### RED 1 — the range is decorative

`test_red1_the_planted_negatives_range_is_load_bearing`, verbatim:

```
AssertionError: the it.27 planted negative is INSENSITIVE to the range it names: every
range in arm_pl resolves to [] -- {'163-172': [], '162-168': [], '1-2': [], '358-360': [],
'400-410': [], '144-174': []}. `ArmPL` spans (358, 410) and `ArmPL.forward` spans
(405, 410), so ranges inside real arm_pl bodies die too.
```

The sweep is the whole argument: `358-360` and `400-410` lie **inside** `ArmPL` and
`ArmPL.forward`, and die identically to `163-172`. No range in `arm_pl` can resolve.

**What actually fires** is `anchored` in `resolved_arms`: a line citation resolves only if
some symbol **named by the same row** is defined in the cited module and contains the range.
L-10 names `path_product` and `zero_hop_mask`; `span_of("arm_pl", …)` is `None` for both.
The anchor set for `arm_pl` is empty *before any line number is compared*, so the range is
never reached. **The corrected claim: the planted negative fires on the missing anchor —
the row names no symbol that lives in the module it was made to cite. The
`163-172`-is-a-scalar-recurrence-loop sentence describes a discrimination the node never
performs and is withdrawn.**

### The range made load-bearing rather than merely removed

`test_the_range_is_load_bearing_where_an_anchor_exists` — L-13 names `ArmSMPrime.forward`
(`572-577`) and cites the module that defines it, so the range is the only free variable:

| L-13 line cite | resolves to |
|---|---|
| `ceq/arm_smprime.py:572-577` (the anchor's exact body) | `['arm_smprime']` |
| `560-565`, `1-5`, `497-500`, `572-600` | `[]` |

`572-600` is the sharp one: it *starts* at the anchor and overruns its end. So the content
rule has two stages, and it.27 named the second while exercising only the first:
**stage 1 — is any named symbol defined in the cited module; stage 2 — does the range lie
inside that symbol's body.** L-10's planted negative tests stage 1. L-13 tests stage 2.

### RED 2 — nothing became underivable

`test_red2_the_founding_swap_makes_the_row_underivable`, verbatim:

```
AssertionError: the founding swap changes nothing: derived {'W1': 'arm_smprime',
'W3': 'arm_pl'} == control {'W1': 'arm_smprime', 'W3': 'arm_pl'}.
```

**The corrected claim:** the founding swap makes L-10's **line citation** unresolvable
(`line ceq/arm_pl.py:163-172` → `[]`) while `symbol path_product` → `['arm_smprime']`
survives, so the wing binding is unchanged. That is defence in depth, not underivability,
and it.27 claimed the stronger fact.
`test_the_founding_swap_kills_the_line_citation_and_nothing_more` asserts it at its true
strength.

---

## 2. Boundary — corrected in both directions

**Stronger than claimed (it.27 boundary 1).** it.27 said a founding mistake that "rewrote
the prose too" would read GREEN. A rewrite of the **ledger prose alone** does not:

```
REFUSAL: W1 resolves into ['arm_pl', 'arm_smprime']. The ledger's citations do not agree
on which arm this wing is; no binding is derivable.
```

Swapping `arm_smprime`↔`arm_pl` and `ArmSMPrime`↔`ArmPL` throughout the ledger (17 lines
change) refuses, because `path_product` **carries no arm name in it** — it stays put and
contradicts the swapped citations. Reading GREEN needs the `ceq/` **symbols** moved too:
a source edit, not a prose edit. `test_a_prose_only_founding_rewrite_refuses…`.

**Weaker than that, honestly (the other direction).** Swapping only the `ceq/*.py` **paths**
leaves every symbol resolving and the derivation identical to control —
`test_a_path_only_rewrite_is_the_weaker_case_and_does_read_green`. The node's strength comes
from the symbols; so does its blind spot. it.27 boundary 3 (W3 rests on one symbol citation,
W1 on three) is unchanged and still unrepaired.

---

## 3. REPAIR 2 — the `601` is WITHDRAWN

**`601` is withdrawn.** It is not a whole-corpus figure, and it was never current for its
own scope. Replacements, `13:03Z–13:10Z`, `2026-09-02`:

| population | lines | occurrences | content digest (`cat … \| md5sum`) |
|---|---|---|---|
| `V20_R15_*.md` (82 files) + `CEQ_V20_R15_CONTRACT.md` | **766** | **774** | `c356111bb786f27f89c32c84f75ea4c8` |
| all root `.md` (235 files) | **1,282** | **1,294** | `8a0cc9d9fae8f2854fff3877954fb0dd` |
| whole tree, `.git` and `__pycache__` excluded | — | **1,912** | (see command) |
| `V20_R15_IT2[3-8]_*.md` alone | — | **163** | — |
| `CEQ_V20_R15_CONTRACT.md` | **2** | **2** | `d2ba5c71d7436d39d5b50408f9edbc40` |

Regeneration, verbatim:

```bash
grep -h  "\[RUN\]" V20_R15_*.md CEQ_V20_R15_CONTRACT.md | wc -l   # 766 lines
grep -ho "\[RUN\]" V20_R15_*.md CEQ_V20_R15_CONTRACT.md | wc -l   # 774 occurrences
grep -h  "\[RUN\]" *.md | wc -l                                   # 1,282
grep -ho "\[RUN\]" *.md | wc -l                                   # 1,294
grep -rho --exclude-dir=.git --exclude-dir=__pycache__ "\[RUN\]" . | wc -l  # 1,912
```

**The scope-A figure moved `750 → 766` in twelve minutes** (nurse `12:54–12:58Z`, this
office `13:03–13:10Z`) because it.29's filings are being written into the counted
population as it is counted. **Any single number for this corpus is a reading with a
timestamp, not a property of the round** — which is the reason the `601` outlived its truth.

### The contract's `17 → 2` is a measurement artefact, not an edit

```bash
grep -o "\[RUN"  CEQ_V20_R15_CONTRACT.md | wc -l   # 17
grep -o "\[RUN\]" CEQ_V20_R15_CONTRACT.md | wc -l   # 2
grep -c  "RUN"   CEQ_V20_R15_CONTRACT.md            # 18 lines
```

The contract carries **17** `[RUN`-prefixed markers, of which **15 carry a payload** —
`[RUN: +20.87 on path-product data …]`, `[RUN anchors: delay 0.050 …]`, `[RUN: τ* = 1.000 …]`
— and only **2** are the bare token `[RUN]`. A pattern anchored on `[RUN` counts 17; a
pattern anchored on `[RUN]` counts 2. **No fifteen markers were deleted.** The one line that
is neither (`151`) reads `its instances RUN before anything else` — prose, caught by
`grep -c RUN`, which is why that count is 18.

Corollary the round should carry: **`[RUN]`-marker totals are pattern-dependent by a factor
of eight on the very file the round treats as fixed.** A count without its pattern is not a
measurement.

Note on custody: `git status --porcelain CEQ_V20_R15_CONTRACT.md` → `?? CEQ_V20_R15_CONTRACT.md`.
**The contract is untracked.** `git log -- CEQ_V20_R15_CONTRACT.md` is empty, so no
diff-based check of "was it edited" is even possible; the artefact explanation above is the
only evidence available, and it is sufficient because it reproduces both numbers from the
current bytes.

### This office's instruments, by the INSPECTOR's tiers

Tier 1 = committed file (refutable by anyone at any later time). Tier 2 =
inline-and-transcribed. Tier 3 = run once on a screen.

```bash
git ls-files tests/saturn/ | wc -l                      # 4
git ls-files --others --exclude-standard tests/saturn/ | wc -l   # 17 (18 with it.29)
```

| tier | count | which |
|---|---|---|
| **1 — committed** | **4** | all pre-R15: `test_census_does_not_attic_refutation_instruments.py`, `test_journal_path_is_discoverable.py`, `test_r10_it2_spotcheck_reds.py`, `test_r10_it3_doc_extractor_is_the_censuss.py` |
| **2 — file in tree, verdict transcribed, uncommitted** | **18** | every R15 SATURN module, it.9 through it.29 inclusive |
| **3 — run once on a screen** | **2 this iteration** | the two ad-hoc drivers behind §1's sweeps, superseded by the tier-2 module and not relied on |

**Zero of this office's eighteen round-15 instruments are tier 1.** By the INSPECTOR's rule
— only tier 1 can be refuted — the round's entire SATURN instrument stock is unrefutable in
the archival sense, and the four that are committed all predate this round. The round's
`NO git writes` rule means this office cannot promote them; **promotion is a standing debt
for whoever holds commit rights, and until it is paid every `[RUN]` marker this office has
filed rests on a file that can be edited or lost without trace.** That is a larger hole than
the `601` ever was.

---

## 4. Recorded, not re-verified

- `tests/saturn` baseline **8** — upheld exact (`8 failed, 187 passed`, 195 nodes).
- `tests/mars_v20` **40** — **wrong, corrected to 31** (`31 failed, 65 passed`, 96 nodes,
  stable 2/2). `31` vs `32` permanently unadjudicable; no artifact of the 90-node object
  exists anywhere.

## 5. Limits

The two RED nodes are assertions *about* the it.27 module, run in the same interpreter as
it; they inherit its `lru_cache`, so a source edit mid-session would not be seen — every
reading above is from one process start at `13:09:44Z`. The `[RUN]` counts are `grep`
line/occurrence counts over `.md` bytes and do not distinguish a marker inside a fenced
block, a quotation of another office's marker, or a marker in a filing about markers; the
scope-A digest pins the bytes those numbers came from, nothing more. The whole-tree `1,912`
excludes `.git` and `__pycache__` but includes every non-`.md` file, so it is an upper bound
on marker-like text, not a count of runs. The tier table counts modules, not nodes, and
tier 2 assumes a file present in the working tree is runnable by another office in this
session — an assumption that expires with the session. Nothing was committed; nothing
touched Kaggle.
