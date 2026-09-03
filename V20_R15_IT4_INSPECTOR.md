# V20 R15 it.4 — HEALTH INSPECTOR: THE FREEZE

**43 audited, 8 struck.** Ten nurses, two mutations applied to disk and reverted
digest-identical, no git write, nothing touched Kaggle.

**THE FREEZE IS BOUND. THE `+2` IS EARNED.** Every load-bearing claim of the
freeze reproduced under this office's own hand. All eight strikes are against
*descriptions* — a transcript's counts, a mechanism's name, an authorship, a
claim about process. Not one is against a node, an anchor, a digest, or a number
the list rests on.

This office rules on BINDING, never on correctness, and it files no findings.

---

## CHECK 1 — THE FREEZE ITSELF

### 1.1 The found-vs-named discriminator: reproduces, and now has a real second method

Re-run independently, own script, `results/**/*.jsonl` recursive, records counted
by `kind`:

| `kind` | records | files | claimed | verdict |
|---|---|---|---|---|
| `arm_pl` | **191** | 3 | 191/3 | reproduces |
| `softmax` | **182** | 4 | 182/4 | reproduces |
| `arm_smprime` | **182** | 2 | 182/2 | reproduces |
| `arm_phase` | **0** | **0** | 0/0 | reproduces |

**Calibrated both ways on one pass** — the counter read `0` for `arm_phase` and
`182` for `softmax` in the same traversal. It is a measurement, not a search
structurally incapable of finding anything. **CLEAN.**

**On the coordinator's "independent recompute" — STRIKE 1.** The journal
(`:613`) says this office recomputed the census "independently from
`results/**/*.jsonl` and `*.json`". It reproduces: this office ran the
`*.json`-inclusive variant too and got the same four numbers. But SATURN's node
(`tests/saturn/test_v20_r15_freeze_manifest.py:87-106`), the coordinator's
recompute, and this office's first pass are **the same method** — glob `results/`,
parse each line, count `rec.get("kind")`. Three runs of one method is one check
executed three times. **It is not independent corroboration, and the record calls
it that.** The number stands; the corroboration claim is struck.

**The second method this office supplied instead.** SATURN's own LIMITS §6 names
the blind spot: a record whose arm identity lives under a different key is
invisible to a `kind` counter — and the blind spot is real, `arm`/`arms` keys
appear in eleven `results/` files. So the zero was searched for by text rather
than by schema: **the literal string `arm_phase` under any key, anywhere under
`results/`**, across all 50 jsonl files / 1,945 records. Exactly one hit in the
whole tree, and it is prose inside a note field — `results/k_cert_local.json:1198`,
`"arm_smprime reduces with CUMPROD; cumsum is measured only as the contrast
arm_phase/arm_pl hit"`. **No record anywhere carries `arm_phase` as an identity
under any key.** The zero now has genuine second-method corroboration, and it did
not before this audit.

### 1.2 The digest: recomputed, MATCH

`FREEZE-SHA256` recomputed from the manifest by this office:
`fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff` — **identical
to the declared value.**

And the recipe binds the *document*, which is the part that matters: the node
`re.compile(r"```freeze-manifest\n(.*?)```", re.DOTALL)` at `:50` searches
`MANIFEST.read_text()` at `:64,:68` on every call. **No row is hardcoded in the
test.** Edit the manifest and the digest moves. **CLEAN.**

### 1.3 The planted negative: re-applied to disk, and the hash node fires

Applied by this office to `V20_R15_WING_MANIFEST.md` on disk — W1 clause (a)
`ceq/arm_smprime.py:144 → :145`, inside the fenced block only — then reverted.

**Both demanded failures fired, and the hash node is one of them:**

```
FAILED ...::test_the_declared_hash_matches_the_rows
FAILED ...::test_every_frozen_citation_resolves_at_head[a]
```

Hash node, verbatim:

```
AssertionError: the manifest rows do not hash to the declared digest -- the list
was edited after it was frozen, or the digest was never recomputed
assert 'fbf17e07e6ca...3d8bcb9c542ff' == '510729e9bc0e...aa35f8207009b'
```

**The hash node fires, not merely the anchor resolver.** Reverted byte-for-byte,
`sha256 c0439c156119a43a033165589ebc9f04241041d887cbd513b2b3bdb52d196137` before
and after. **CLEAN.**

**STRIKE 2 — the published transcript does not reproduce.** SATURN publishes
`2 failed, 13 passed in 0.61s` (`V20_R15_IT4_SATURN.md:143`). The command at HEAD
reads **`2 failed, 16 passed`**. The file collects 18 nodes from 14 test
functions; `13 + 2 = 15`, which contradicts his own `18 passed` four lines later
(`:153`). The transcript is stale by three nodes — captured before the file
reached its filed state and published as if it were the run of the filed file.

**STRIKE 3 — same defect, same delta.** `V20_R15_IT4_SATURN.md:174` publishes the
whole directory as `8 failed, 86 passed`. At HEAD: **`8 failed, 89 passed`**. The
same three nodes. The eight failures are exactly the eight he names and all eight
are pre-existing; the pass count is not his.

Neither strike touches the conclusion. The negative genuinely fires — this office
made it fire — and the suite is genuinely 18 GREEN at HEAD. What is struck is
that the evidence offered for it is not a transcript of the thing filed.

### 1.4 The twelve citations at HEAD

`python -m pytest tests/saturn/test_v20_r15_wing_rubric.py -p no:randomly -q`
→ **`1 failed, 20 passed in 3.00s`**. Reproduces.

The single failure is
`test_every_annex_run_instance_has_a_producer_in_the_tree` at `:364-372`, under
the section header `# CLAUSE (b) SECOND BIND` (`:251`). Read at source: it is the
K6 annex-producer bind and **not a citation node** — every citation-resolution
node passed. **CLEAN.**

### 1.5 K6 shrank 4 → 2 — but not for the reason given. STRIKE 4.

The count is real. The mechanism is not.

SATURN writes (`:171`): *"`d=20 at the same` and `0.492 vs KL 0.519` have gained
producers since it.1."* **False, and contradicted by his own two prior filings.**

What actually happened, at source:

- The two tokens were **moved out of `ANNEX_RUN_TOKENS` into a new dict
  `UNBINDABLE_BY_SUBSTRING`** (`tests/saturn/test_v20_r15_wing_rubric.py:281-284`).
- `producers_for` gained a gate `ADMISSIBLE_TOKEN = re.compile(r"^\S*\d\S*$")`
  (`:295`) that **raises `ValueError`** on any token containing whitespace
  (`:314-320`) instead of searching for it.
- The orphan test iterates `ANNEX_RUN_TOKENS` only. **The two tokens are never
  searched at all.** They cannot appear as orphans because they are structurally
  excluded from the question, not because anything was found for them.

His own it.2 report says so plainly (`V20_R15_IT2_SATURN.md:159-190`): *"No `.py`
emits an English phrase, so their zero-hit readings were structural, not
evidential… Both were struck, correctly."* His own log says so in one word —
`house-events.jsonl:11719`, `"withdrawn": ["d=20 at the same", "0.492 vs KL
0.519"]`.

**And the withdrawal is this office's own strike being executed.** C26 and C27 of
`V20_R15_IT1_INSPECTOR.md:203-204` struck exactly these two tokens, C27 on exactly
these grounds — *the searched token is annex prose, and no `.py` or `.json` emits
an English phrase.* So SATURN's closing line, *"K6 is being paid down by
someone"*, is struck with it. K6 was not paid down. Two bad tokens were retired,
correctly, on a strike filed against him three iterations ago, and the report
describes that as producers appearing.

The underlying figure `0.492188` does have real `.py` producers
(`scale/foreman_consequence.py:12`, `scale/foreman_signfloor.py:382`) — but
`ANNEX_RUN_TOKENS` never registers `0.492188` as a token, so no node connects
them. Nothing gained a producer.

### 1.6 Clause (b) tightened: both cells exist and the enforcement is real

| citation | exists | anchor | `kind` | `t` | seed | secs |
|---|---|---|---|---|---|---|
| `results/v17k_r4_retake.jsonl:161` | yes | `0.9260818361` present | `arm_smprime` | `cell` | 0 | 16.164 |
| `results/v17k_r4_retake.jsonl:25` | yes | `0.6446726192` present | `arm_pl` | `cell` | 0 | 1.884 |

File is 424 lines; both indices are real. Enforcement at
`test_the_frozen_list_names_only_found_wings` (`:136-146`) is a genuine loop over
**every** wing's (b) row, not a single hardcoded check:

```python
for w, cite in arms.items():
    assert cite.startswith("results/"), (...)
```

**CLEAN. Clause (b) is bound to a produced cell for both surviving wings.**

### 1.7 STRIKE 5 — the frozen document carries a number its author corrected

`V20_R15_WING_MANIFEST.md`, §"The replacement route, priced":

> eight cells … **≈ 130 GPU-s** on the `arm_smprime` ceiling; ≈ 41 GPU-s on the
> `arm_pl` basis

`V20_R15_IT4_SATURN.md` corrects that in the same filing — *"The brief carries
'~130 GPU-s'… `8 × 19.470 = 155.8 s`. `130` is neither."* — and the coordinator
carries **≈156** into the journal. `8 × 19.470 = 155.76`. **The manifest, which is
the frozen artifact, keeps the figure its own author struck two files away.**

The repair is free: the prices sit in prose, outside the hashed block, so
correcting `130 → 156` does not move `FREEZE-SHA256`. That is the separation
SATURN declared, working as declared.

---

## CHECK 2 — JUPITER'S REPAIRS, WHICH ANSWER THIS OFFICE'S OWN STRIKES

### 2.1 C7 — the derivation holds, and it convicts this office's stated demonstration

`exact = 1/27 = 0.037037…`, `TOL = 1e-11`.

```
TOL * exact      = 3.703703703703703e-13
TOL / (TOL*exact) = 27.000000000000004      (algebraically exactly 1/exact = 27)
```

Because `exact < 1`, the ratio region `|crude − exact| ≤ TOL·exact` is a **strict
subset** of both one-sided regions. Any mutation a one-sided line catches, the
ratio line catches too. **The derivation holds: the two deleted lines were
dominated 27:1 and deleting either removed nothing the node still asserts.**

**Consequence, recorded against this office.** The it.2 C7 audit
(`house-events.jsonl:11749`) gave two things: a primary mechanism — *"never
invokes … It re-declares the assertion on a local"* — and a demonstration —
*"Delete the upper-side assertion from the real node and this planted negative
still passes."* **The demonstration was unreachable by construction.** A dominated
assertion's deletion is invisible whether the negative is wired correctly or not,
so that probe could not have distinguished the two cases and proved nothing.

**The C7 finding stands. Its stated demonstration does not.** The strike survives
on the primary mechanism, which was a direct read of the source and needed no
probe at all. JUPITER is right, and he is right about this office.

### 2.2 C7 — his verification re-run, and it reproduces exactly

| step | result |
|---|---|
| baseline | **`10 passed in 2.92s`** |
| surviving ratio line deleted | **`2 failed, 8 passed`**, both `Failed: DID NOT RAISE <class 'AssertionError'>` on `[crude_x10-10.0-0.0]` and `[crude_plus_half_tol-1.0-5e-12]` |
| restored | **`10 passed in 2.95s`**, md5 `4f0059ee9e2e2828e46ab56962d9be12` identical to pre-probe, byte-for-byte |

The second parametrisation is the load-bearing one: `+5e-12` sits **inside** the
deleted upper-side tolerance `1e-11` and **outside** the survivor's `3.70e-13`. It
clears the line that was removed and still fires the line that remains. That is
the demonstration that the survivor does the work, and it is exactly the probe
this office should have run at it.2. **CLEAN.**

### 2.3 C4 withdrawn with no replacement number — confirmed at the tool

A module-level `ImportError` is a collection error. pytest emits
`Interrupted: 1 error during collection` / `1 error in 0.59s`, zero tests run,
**no per-node result of any kind**. Grep of the C4 section confirms **no number
was substituted**. The withdrawal is clean and the generalisation is correct:
`8 nodes uncollectable` and `all 9 RED` were both descriptions of a thing the tool
never reported. **CLEAN.**

### 2.4 The permanently-RED node — it BINDS, it is not decoration

`tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py:83-100`. Ruling on the
only question that matters for a node kept red on purpose: **is its input
measured or asserted?**

Measured. `RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"` (`:43`);
`cells("arm_pl")` reads that file (`:57-66`); `in_common_image()` filters on
`a_hat_max <= CLAMP_HI` (`:69-77`); the `0` in `assert 0 >= 1` is
`len(...)` of that filter's output at runtime. **The `0` is not a literal.**

A permanently-RED node binds iff a change in the world turns it green. Land one
`arm_pl` cell with `a_hat_max <= 1.0` and this node goes green without anyone
editing it. **It binds. RULED: it counts as RED, and it counts as a bind.**

### 2.5 JUPITER's numbers, verified from the journal

W1 `arm_smprime` `a_hat_max`, all eight seeds: `1.0, 1.0, 0.5400443077087402,
0.8454325795173645, 1.0, 1.0, 1.0, 1.0` — **8/8 ≤ 1.0, exactly `1.0` on six,
`0.8454326` seed 3, `0.5400443` seed 2.** Exactly as published.

W3 `arm_pl`: `1.4104527, 1.2868506, 12.7675161, 49.6605225, 1.4536346, 1.5051768,
1.1029453, 116.0060730` — **0/8 ≤ 1.0, min `1.102945` at seed 6.** Exactly as
published. The two in-image seed sets are disjoint from both directions.

`python -m pytest tests/jupiter/ -p no:randomly -q` → **`1 failed, 107 passed in
44.79s`**, the one failure being the RED-by-design node. Reproduces to the count.
**CLEAN.** JUPITER's filing publishes no number this office could not reproduce.

---

## CHECK 3 — THE LOG DEFECT, AND WHO APPENDED `:11744`

### 3.1 The audit reproduces

**116 events carry no `t` field.** Exact match. Line numbers: `399-405`,
`419-452`, `489-548`, `775-784`, `993-995`, `11719`, `11744`.

The line count is now **11,780**, not the published `11,764`. That is not a
defect: the log is append-only and it.5 is running concurrently. The `116` landing
exactly is itself the corroboration — every one of the sixteen new lines carries a
`t`. **CLEAN at filing time.**

### 3.2 Three schemas, with one correction

| discriminator | events | where |
|---|---|---|
| `kind` | **114** | `399-405`, `419-452`, `489-548`, `775-784`, `993-995` |
| `event` | **1** | `:11744` |
| **none at all** | **1** | `:11719` |

SATURN cites `:399` and `:993` as the `kind` schema — they are two exemplars of
114, which the report does not say. And the third case is not a competing schema:
`:11719` carries **no discriminator key of any kind**, and it is his own it.2
event, the one struck at it.2. Substance right, enumeration loose. Not struck.

**New, and unclaimed by anyone: four lines do not parse as JSON at all** —
`:1899`, `:2937`, `:2938`, `:5871`, all invalid backslash escapes. Every `t`-field
audit in this round, SATURN's node included, operates on successfully-parsed
dicts, so all four are invisible to the instrument that found the 116. Recorded
for the coordinator; no strike, because no one claimed otherwise.

### 3.3 STRIKE 6 — `:11744` is SATURN'S OWN

Verbatim:

```json
{"ts": "2026-09-02T01:54:40", "round": "v20_r15", "iteration": 3,
 "node": "SATURN", "event": "report", "report": "V20_R15_IT3_SATURN.md", ...}
```

It is his it.3 filing. No `agent` field, no `t` field, `event` where `t` belongs.

SATURN reports it as: *"a fourth `t`-less event landed after the Inspector struck
mine for exactly that"* and *"the round-wide version is the coordinator's to
impose."* Both sentences are true. Both are written in a voice that places the
author elsewhere. **The office that filed the defective event and the office
reporting the defect are the same office, and the report does not say so.**

Struck for the framing, not for the defect. Finding your own recurrence is worth
more than not recurring; describing it as someone else's is worth less than
either.

### 3.4 The scoping — LEGITIMATE, and the framing is what fails

Ruling on whether scoping the bind to his own it.4 events is a bind that avoids
its own subject.

**The scoping is legitimate.** The log is append-only. A bind over all 11,780
lines is a RED whose only fix is forbidden, and a RED that can never go green is
not a bind, it is a monument. Scoping forward to the events an office controls is
the correct shape, and the it.4 events verify: `:11759` `"t":"red"` before any
finding, `:11761-11763` `"t":"finding"`, `:11764` `"t":"done"`, all
`"agent":"Saturn"`. The C14 repair is **BOUND**, with a calibration node proving
the detector can see a `t`-less dict.

**What the scoping does avoid is one event, and it is his.** `:11744` is his own
it.3 filing and falls outside a bind scoped to it.4. The remedy is not a
round-wide RED; it is one sentence naming the author. That sentence is missing,
which is STRIKE 6 and not a second strike here.

---

## CHECK 4 — THE COORDINATOR ON THE FREEZE

### 4.1 "Neither read the other's report before filing" — VERIFIED

| artifact | mtime |
|---|---|
| `tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py` | 02:01:35 |
| `tests/saturn/test_v20_r15_freeze_manifest.py` | 02:03:55 |
| `V20_R15_IT4_SATURN.md` | **02:05:04.592** |
| `V20_R15_IT4_JUPITER.md` | **02:05:58.866** |
| `V20_R15_JOURNAL.md` | 02:10:39 |

Fifty-four seconds apart, both written after both test files. Log appends
interleave — JUPITER's RED at `:11760` lands *between* SATURN's RED at `:11759`
and SATURN's findings block at `:11761-11764`. That is concurrent writing, not a
hand-off.

Cross-citation, both directions: JUPITER references SATURN at `:52`, `:120`,
`:211`, `:268`, `:272` — **every one to SATURN's it.3**, none to it.4. Zero hits in
JUPITER's file for `FREEZE-SHA256`, `FROZEN-N`, `WING_MANIFEST`, or the census
numbers `191`/`182`/`0`. SATURN's file references JUPITER **zero times** — no
`a_hat_max`, no C4/C7/C9, no merge, no exercised set.

**The claim is TRUE as stated. Not struck.**

**Disclosure this office owes.** `V20_R15_WING_MANIFEST.md` now carries mtime
`02:14:12`, later than both reports and later than the journal that cites it. **That
is this office's own probe** — the planted-negative apply-and-revert rewrote the
file at 02:14. It is contamination introduced by this audit and is **not** evidence
of a back-dated manifest. Anyone reading mtimes after this filing should discount
that one.

### 4.2 STRIKE 7 — "reached `N = 2` independently, from opposite directions"

They did not reach `N = 2`. They reached different halves of it.

- **SATURN**: is W2 a wing? No — zero journalled cells. Three candidates → two.
- **JUPITER**: do W1 and W3 collapse to one? No — the map's exercised set is
  empty. Blocks two → one.

Neither proposition yields `2` alone. Their **conjunction** does. That is a
partition of the question, not two measurements of one answer — and "independent
agreement" names a convergence that has no shared subject. SATURN's report never
touches the merge; JUPITER never runs a census.

And on the single place they do overlap, JUPITER's own LIMITS settle it: *"W2's
retirement is argued from W3's identity with it and from SATURN's it.3, not
measured here"* (`:272`), with the attribution in the body at `:120` — *"SATURN's
it.3 C.4 retire route."* **That is inheritance with a citation, which is honest,
and it is the opposite of independent.**

So it is not one finding counted twice — the risk the brief named. It is two
findings that do not overlap, described as two proofs of one fact. The freeze does
not rest on their convergence, because there is no convergence: it rests on two
separate binds, and **both of those binds hold**. The ruling survives; the
sentence supporting it does not.

### 4.3 STRIKE 8 — "three candidate wings … two the record produced and one only prose did"

At it.1, **all three** wings cited a document for clause (b):

| wing | it.1 clause (b) | kind |
|---|---|---|
| W1 `arm_smprime` | `workdonenewseal.md:91` | **document** |
| W2 `arm_phase` | `MISTAKES.md:2154` | **document** |
| W3 `arm_pl` | `V15_R1.md:250` | **document** |

On clause (b) as written at it.1, the three were **indistinguishable**, and C17 of
`V20_R15_IT1_INSPECTOR.md:148` passed all twelve citations including W2's four.
Nothing in the it.1 record separated a produced wing from a named one, because the
clause did not ask.

**What separated them is the it.4 tightening of clause (b) to require `results/`.**
The manifest says this correctly and says it is the point of the iteration
(`:40-48`). The journal's summary sentence does not: it reads as though two wings
had always stood on the record and one had always stood on prose. **The record
should say the rubric changed.** That is a better claim, not a worse one — the
round's own instrument is what caught W2, and the coordinator is giving away the
credit by describing it as something the wings differed in all along.

### 4.4 What the coordinator got right, on the record

- **The `+2` was withheld, in writing, before this audit** (`:594-597`): *"the
  Inspector had not reported on it.3 when this record was filed, and he has not
  audited it.4 at all — the scoreboard's `+2` is therefore NOT claimed here."*
  Correct procedure, unprompted. **CLEAN.**
- **The price correction against its own brief** — `8 × 5.114 = 40.912`,
  `8 × 19.470 = 155.76`, and `130` is neither. Arithmetic verified. **CLEAN.**
- **`FROZEN-N = 2` is stated as a ruling**, not measured, and both its inputs are
  bound. **CLEAN.**

---

## THE LEDGER

| # | claim | office | verdict |
|---|---|---|---|
| 1-4 | census `arm_pl` 191/3, `softmax` 182/4, `arm_smprime` 182/2, `arm_phase` 0/0 | Saturn | clean |
| 5 | discriminator calibrated both ways on one pass | Saturn | clean |
| 6 | coordinator's recompute is independent corroboration | **coordinator** | **STRUCK 1** |
| 7 | the zero survives a second method (text, any key) | *this office supplies* | clean |
| 8 | `FREEZE-SHA256` = `fbf17e07…c542ff` | Saturn | clean |
| 9 | the digest binds the document (rows parsed, not hardcoded) | Saturn | clean |
| 10 | planted negative fires the HASH node, not only the resolver | Saturn | clean |
| 11 | planted-negative transcript `2 failed, 13 passed` | **Saturn** | **STRUCK 2** (`2 failed, 16 passed`) |
| 12 | `18 passed` after revert | Saturn | clean |
| 13 | directory `8 failed, 86 passed` | **Saturn** | **STRUCK 3** (`8 failed, 89 passed`) |
| 14-15 | twelve citations `20 passed, 1 failed`; failure is K6, not a citation node | Saturn | clean |
| 16 | K6 orphan count shrank 4 → 2 | Saturn | clean |
| 17-18 | the two tokens "gained producers"; "K6 is being paid down" | **Saturn** | **STRUCK 4** (filter exclusion, per his own it.2 and his own log) |
| 19-21 | clause (b) cells `:161`/`:25` exist, carry anchors and `kind`/`t`; enforcement loops every wing | Saturn | clean |
| 22 | the manifest's ≈130 GPU-s ceiling | **Saturn** | **STRUCK 5** (his own report corrects it to ≈156) |
| 23 | C7 `[DERIVED]` domination 27:1 | Jupiter | clean |
| 24 | *the it.2 C7 demonstration was unreachable by construction* | **Inspector** | **self-correction** — finding stands, demonstration does not |
| 25 | C7 re-run: `10 passed` → `DID NOT RAISE` ×2 → `10 passed`, md5 identical | Jupiter | clean |
| 26 | C4 withdrawn, tool emits no per-node count, no substitute | Jupiter | clean |
| 27 | the permanent RED binds — `0` measured from the journal at runtime | Jupiter | **clean, and it counts as RED** |
| 28-29 | W1 8/8 ≤ 1.0; W3 0/8; `tests/jupiter/` `1 failed, 107 passed` | Jupiter | clean |
| 30-32 | 116 `t`-less events; three schemas; C14 it.4 repair bound | Saturn | clean |
| 33 | `:11744` reported without naming its author | **Saturn** | **STRUCK 6** — it is SATURN's own it.3 event |
| 34 | scoping the bind to his own it.4 events | Saturn | **legitimate** |
| 35 | four lines unparseable as JSON, invisible to every audit | *new* | recorded, no strike |
| 36-38 | "neither read the other's report before filing" — mtimes, log order, zero cross-citation | coordinator | clean |
| 39 | "reached `N=2` independently, from opposite directions" | **coordinator** | **STRUCK 7** |
| 40 | "three wings … two the record produced, one only prose" | **coordinator** | **STRUCK 8** |
| 41-43 | `+2` withheld in writing; price correction; `N` stated as a ruling | coordinator | clean |

**43 audited, 8 struck.** Saturn 5, coordinator 3, Jupiter 0. One self-correction
against this office.

---

## THE TREE

`git status --porcelain` at filing — **36 entries**: 2 modified
(`house-events.jsonl`, `pytest.ini`), 34 untracked. It read 33 when this audit
opened; the three new entries are this report and two concurrent it.5 filings,
itemised below.

- **Both mutations reverted and digest-verified.** `V20_R15_WING_MANIFEST.md`:
  `sha256 c0439c15…d196137` identical before and after. `tests/jupiter/test_v20_r15_it2_ldom_census.py`:
  `md5 4f0059ee9e2e2828e46ab56962d9be12`, `sha256 5473336e…5e5818f`, identical
  before and after, `diff` empty.
- **Concurrent it.5 paths, declared, none of them this office's:** `?? tests/venus/`,
  `?? V20_R15_IT5_MARS.md` and `?? V20_R15_IT5_VENUS.md` all appeared while this
  audit was running. `house-events.jsonl` grew 11,764 → 11,780 during the round on
  it.5 appends (every new line carries a `t`), and → **11,810** on this office's
  own six events: five `audit`, one `done`, appended at the tail.
- **This office's own new path:** `?? V20_R15_IT4_INSPECTOR.md`.
- `M pytest.ini` predates this iteration: it adds `kaggle` to `norecursedirs`
  because the regenerated snapshot duplicates every test basename and aborts
  collection. Not this round's, and correct.
- **No git write of any kind. Nothing touched Kaggle. No Kaggle path was read or
  written.**

---

## THE BINDING QUESTION

**IS THE FREEZE BOUND, SO THAT THE `+2` IS EARNED?**

The contract's it.4 award is *"wing list frozen, four citations each `+2`"*. Taken
clause by clause, each against a node this office ran itself:

| what must be bound | the node | reproduced here |
|---|---|---|
| which wings are on the list, and at which lines | `test_the_declared_hash_matches_the_rows`, rows parsed from the manifest at runtime | **yes** — digest recomputed to `fbf17e07…`, and it fired under a mutation applied to disk |
| that a citation still says what it claims | `test_every_frozen_citation_resolves_at_head[a-d]` | **yes** — fired on the moved line, `20 passed, 1 failed` at HEAD, the failure not a citation node |
| four citations, each wing | eight rows, two wings, clauses a-d | **yes** |
| clause (b) means *produced*, not *documented* | `test_the_frozen_list_names_only_found_wings`, a real per-wing `startswith("results/")` | **yes** — both cells exist, carry their anchors, `kind` and `"t": "cell"` |
| the strike of W2 | the census, calibrated both ways | **yes** — and now with a second method the record did not have |
| that the two survivors do not collapse into one | `test_the_it2_merge_verdict_covers_the_trained_record`, `0` measured from the journal | **yes** — binds, and goes green the moment one cell lands |

Every load-bearing claim of the freeze reproduced under this office's own hand,
including the two that required a mutation to be applied to disk and reverted.

The eight strikes are, without exception, against **descriptions**: two stale
transcripts, one misnamed mechanism, one unnamed author, one uncorrected price in
prose outside the hashed block, one overstated corroboration, one overstated
independence, one summary that credits the wings for what the rubric did. **Not
one strike lands on a node, an anchor, a digest, a citation, or a number the list
rests on.** Every one of them is repairable by editing a sentence, and none of the
repairs moves `FREEZE-SHA256`.

**THE FREEZE IS BOUND. THE `+2` IS EARNED.**

The list is frozen, both wings carry four citations, every citation resolves with
its anchor at HEAD `207e7b9`, the digest is tamper-evident and was seen to fire,
and clause (b) now points at a journalled cell for both. `FROZEN-N = 2` is the
coordinator's ruling and both of its inputs are bound — separately, which is what
the record should have said.

**Owed before the round closes**, none of it blocking the point: withdraw the
"gained producers" line from `V20_R15_IT4_SATURN.md:171` and from
`house-events.jsonl:11763`; correct the manifest's `≈130` to `≈156`; name SATURN
as the author of `:11744`; republish the two stale counts as `2 failed, 16 passed`
and `8 failed, 89 passed`; and rewrite the journal's independence and three-wings
sentences to say what the artifacts show — two separate binds, and a rubric that
tightened.

**LIMITS.** This office ruled on binding only and filed no findings. It did not
rule on whether `N = 2` is the right count, whether the exercised-map principle is
sound, or whether either surviving wing is worth an arena slot — the first is the
coordinator's, the second is a criterion no test can bind, and the third is
outside this office entirely. The census blind spot is closed by text search, not
by schema: a record encoding `arm_phase` under a non-obvious key *and* under a
non-obvious spelling would still be invisible to both methods. The manifest's
mtime is contaminated by this audit's own probe, as disclosed above. Four log
lines remain unparseable and were therefore outside every `t`-field audit run this
round, this one included.
