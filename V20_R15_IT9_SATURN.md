# V20 R15 it.9 — SATURN (WATSON). THE HASH QUESTION IS ANSWERED, AND THE ANSWER IS NOT THE ONE THE ROUND FEARED.

**Adding a journalled column to `scripts/v15_r1.py` DOES move
`instrument_hash 5d41a63d…9a309`. It does NOT cost a re-take of any banked cell.
Those are two different facts and the round has been treating them as one.**

The freeze holds at HEAD with zero drift. The ledger's schema hole is repaired
in both directions — folded backward for the reader, gated forward for the
writer — and a second, harder class of unreadable record was found while doing
it. **No GPU-seconds were spent. No git write. Nothing touched Kaggle.**

Node: `tests/saturn/test_v20_r15_it9_saturn.py`, **21 passed in 0.46s**
`[RUN] python -m pytest tests/saturn/test_v20_r15_it9_saturn.py -q`.
Four REDs were posted to `house-events.jsonl` before the findings, each with a
`t` field.

---

## A. THE GOVERNING QUESTION, ANSWERED FIRST BECAUSE IT PRICES THE OTHER TWO

### A.1 Yes, the hash moves — and it is arithmetic, not judgement

`[READ] scale/identity_manifest.py:210-215`:

```python
data = pathlib.Path(path).read_bytes()
file_hash = _sha(data)
reach_hash = _sha(*(_code_fingerprint(f).encode() for f in reaches))
return {..., "hash": _sha(file_hash.encode(), reach_hash.encode())}
```

and the function's own docstring at `:184` states the rule without hedging:

> `file` — sha256 of `path`'s own bytes, read off disk. **Moves on ANY edit to
> the named file, prose included** — deliberately NOT run through
> `_code_fingerprint`'s docstring-dropping rule.

`scripts/v15_r1.py:174` calls exactly that:
`INSTRUMENT_MANIFEST = identity_manifest.instrument_manifest(__file__, reaches=INSTRUMENT_REACHES)`.

`[RUN]` `test_a_one_byte_edit_to_the_instrument_moves_the_instrument_hash`
copies the shipped instrument, appends **one comment line** — smaller than any
of the three asked-for columns — recomputes with `reaches=()` so only the `file`
component can move, and asserts three things: `reach` identical, `file`
different, composite `hash` different. **GREEN.** All three column additions
land in that one file (`:801-809`, the identity-manifest `base.update` at
`:834-`, and `gate_columns` at `:343`), so all three move it.

### A.2 But nothing in production reads the field

`[RUN]` `test_no_production_path_refuses_on_an_instrument_hash_mismatch` walks
every `.py` under `ceq/` and `scale/` for the literal `instrument_hash` and
**finds zero**. `[READ]` the field is *written* at `scripts/v15_r1.py:614`
(header) and `:830` (cell) and read by:

| reader | what it does on a mismatch |
|---|---|
| `ceq/**`, `scale/**` | **nothing — the string does not occur** |
| `tests/mercury/test_v20_r15_it6_seeds8_15.py:30` | lists it among the invariant header fields of a **regime diff between two banked files** |
| `tests/mercury/test_v20_r15_it8_armpl_and_clamp.py:36` | `assert new["instrument_hash"] == ret["instrument_hash"]`, again between two banked files |

**There is no refusal. `refuse_if_changed` (`scale/identity_manifest.py:218`)
compares `config/code/shapes/rng` on a CELL manifest and never sees this field
at all.** So the edit does not invalidate one banked number. What it does is
**fork the pool for a reader that binds on the hash** — every cell produced
after the edit reads a different instrument than the 24 retake cells and the
19 it.6/it.8 cells.

### A.3 The fork, priced

Re-taking the deciding cells so a hash-binding reader sees the new columns on
them, `[DERIVED]` from the `secs` column of `results/v17k_r4_retake.jsonl`:

| arm | cells | mean | total |
|---|---|---|---|
| `arm_pl` | 8 | 1.780 s | 14.240 s |
| `arm_smprime` | 8 | 16.161 s | 129.287 s |
| `softmax` | 8 | 1.681 s | 13.448 s |
| **all 24** | | | **156.975 s** of cell time |

**Under three minutes of cell time on this box.** **NOT RUN this iteration** —
the brief forbids starting it here, and it is priced rather than begun.

**The cheaper route, and it does not need the re-take at all.** MERCURY's §1.4
established that the invocation reproduces **bitwise** on seed 0 across a
five-day gap. So the pool can be re-established by a **value** bind — bitwise
equality of the columns both sides share — instead of a **hash** bind, and the
two `tests/mercury` assertions above are the only things that would have to
change. That is a two-line edit to two test files against a 157-GPU-second
re-take, and it is the recommended route.

### A.4 What was NOT implemented, and exactly why

**None of the three columns was written into `scripts/v15_r1.py` this
iteration**, and the reason is a scope call, not the hash:

1. **The hash answer removes the *cost* objection but not the *timing* one.**
   MERCURY records at it.8 `[CITED] V20_R15_IT8_MERCURY.md:380` that he ran with
   the instrument *"unmodified by this node"*, and the round has three more
   planets who may run before it re-freezes. Editing the shared instrument
   mid-iteration, with no run inside this iteration to validate the emit,
   silently re-prices every batch anyone takes afterwards. **The coordinator
   rules on that, with A.1–A.3 in hand; this office does not rule on it
   unilaterally at minute 15 of a 20-minute wall.**
2. **Item 1 has a route that costs neither GPU-seconds nor the hash.** The
   pre-clamp question — does `u` exceed `1.0` or land on it — is answerable at
   the **zero-step twin without touching the instrument at all**, because the
   0-step model is rebuilt under the cell's own seed
   (`scripts/v15_r1.py:801`, `torch.manual_seed(seed); m0 = make_arm(kind, S)`)
   and `ceq/arm_smprime.py:109-113` `magnitude(u)` is a pure function of the
   head output. A **test** under `tests/saturn/` can rebuild `m0` on CPU, call
   `model.heads(x)` and read `max_j u_j` **pre**-`clamp` directly. A test is not
   the instrument: it moves no hash and journals no column. **Priced: one test
   node, 0 GPU-s, ~15 minutes.** It answers the 0-step twin only; the *trained*
   pre-clamp `u` still needs the emit and the fork.
3. **Items 2 and 3 are the same edit and should ship together.** `smp_values` on
   the `_0step` record and `beta/qk/route/g` as cell fields both come from the
   arm's config; splitting them into two hash moves buys two forks for one
   question. `[READ] scripts/v15_r1.py:834-840` `base.update(...)` is where the
   cell's identity manifest is assembled and where the four fields belong — the
   same `base` that JUPITER's Q1/W1 **F0, HOW-BAD `0 of 24`** is about.

---

## B. THE FREEZE, RE-RUN AT HEAD — ZERO DRIFT, FIVE ITERATIONS ON

`[RUN] python -m pytest tests/saturn/test_v20_r15_freeze_manifest.py
tests/saturn/test_v20_r15_wing_rubric.py -q` → **38 passed, 1 failed**. The one
failure is `test_every_annex_run_instance_has_a_producer_in_the_tree` — the it.1
**K6** annex-producer bind (`M9-F1 d=65`, `M2 theta 1.3e-3`), which is **not a
citation node** and was already failing at the freeze. Every citation node is
green.

Re-run independently in this iteration's own file rather than trusted from it.4:

- `test_the_frozen_digest_still_matches_the_list_at_head` — 8 rows,
  `sha256` over the sorted, newline-joined rows =
  **`fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff`**. Match.
- `test_every_frozen_citation_resolves_with_its_anchor_at_head` — parametrised
  over all **8** rows, each asserting the anchor occurs on that exact line.
  All green.

**`[READ] git rev-parse --short HEAD` → `207e7b9`, unchanged since the freeze.**
The tree moved, HEAD did not: the two new result journals
(`results/v20_r15_it8_armpl_b.jsonl`, `results/v20_r15_it8_armpl_seeds8_15.jsonl`),
the ten new test files and the modified `pytest.ini` are **uncommitted**. So the
it.4 manifest's claim to bind repo HEAD `207e7b9` is still literally true, and
**the P-6 risk this office named at it.4 has not fired**: no cited line has
moved and no anchor has softened across five iterations.

**The reservation stands unchanged.** The digest still covers the *list* and not
the cited files' *content*. `V16_ARM_SMPRIME.md:529` could be rewritten around
its anchor and the digest would not notice. That is why the anchor check is a
separate node re-run at every HEAD, and why it was re-run here rather than
inherited.

---

## C. THE LOG SCHEMA — REPAIRED FORWARD, FOLDED BACKWARD, AND A HARDER CLASS FOUND

### C.1 The blast radius, re-measured

`[RUN]` over `house-events.jsonl` at HEAD:

| | count |
|---|---|
| lines | 12,367 (pre-append) |
| records with a `t` field | 12,247 |
| **records with NO `t` field** | **116** |
| **lines no JSON parser can read** | **4** |

The 116 spell the type as `kind` (`:399`, `:993`) or `event` (`:11744`), and
`scale/ledger.py`'s three type accessors all filtered the literal:
`tests()` `if e.get("t") == "test"`, `findings()`, `audits()`. **Type is the
FIRST predicate in all three**, so those records matched no filter at any
status. Unbindable by construction, exactly as MARS measured.

### C.2 The repair reuses this module's own established pattern

`scale/ledger.py` already normalises three fields the file spells more than one
way — `_status` (`status`/`state`), `_agent` (case), `_iteration` (int/string).
**The type field is the fourth instance of the same defect, in the field every
accessor filters on first, which makes it the widest.** So the fix is the
pattern already there, one field over — not a new module:

- **`scale/ledger._t(event)`** — reads `t`, then `kind`, then `event`,
  case-folded. `t` wins when present, so a record carrying both keeps its
  author's meaning. `tests()`, `findings()` and `audits()` now route through it.
- **`scale/ledger.append(event, path=None)`** — **the forward-only schema gate.**
  Refuses a non-dict, refuses an event whose `t` is missing / empty / not a
  string, and refuses anything that does not survive a `json.dumps` →
  `json.loads` round trip. Returns the line it wrote. **`kind`/`event` are READ
  for history and REFUSED on new appends** — the history stays a monument, the
  future is gated.

**Measured recovery `[RUN]`:**

| | literal `t` filter | after the fold | recovered |
|---|---|---|---|
| type `test` events | 11,307 | 11,401 | **+94** |
| **`test` events at status RED** | **1,321** | **1,349** | **+28** |
| typeless records now folding | — | — | **115 of 116** |
| still typeless after the fold | — | — | **1** |

**Twenty-eight REDs were invisible to the reader the Inspector uses to decide
what is bound.** One record remains typeless and carries none of the three
spellings; it is left visible as a residual rather than folded by guesswork.

### C.3 The harder class, found while measuring the first

**`[RUN]` four lines — `1899`, `2937`, `2938`, `5871` — carry INVALID JSON
escapes** (`\e` and friends, from hand-written Lean and LaTeX pasted into a
`text` field). They *look* well-formed and three of them even carry a correct
`t` field, but **no parser can read them**: `scale/ledger.read()` drops them
silently and `unparseable()` is the only thing that counts them.

This class is strictly worse than the 116 — those parse and fail to bind; these
never arrive. **The append gate closes it by construction**: every record now
goes out through `json.dumps`, and the encoder cannot emit an escape the decoder
rejects. The round-trip assertion in `append()` proves it per call rather than
trusting the encoder.

`test_the_unparseable_lines_are_counted_not_silently_dropped` pins the census at
those four exact line numbers, so a fifth is a failure and not a shrug.

### C.4 The gate is dogfooded

This iteration's seven GREEN events were written **through `ledger.append`**,
not through a raw `open(...,"a")`. `[RUN] python scale/ledger.py` →
`demo OK: 12,370 events, 1349 red, 20 from HOUSE, status keys in use
['state','status'], 4 unparseable lines skipped and counted`.

---

## D. THE NODES

`tests/saturn/test_v20_r15_it9_saturn.py`, **21 passed in 0.46s**:

| node | binds |
|---|---|
| `test_a_one_byte_edit_to_the_instrument_moves_the_instrument_hash` | A.1 — the governing answer |
| `test_the_instrument_at_head_is_the_one_the_banked_cells_name` | the control: the fork does not predate this iteration |
| `test_no_production_path_refuses_on_an_instrument_hash_mismatch` | A.2 — the price ceiling |
| `test_the_frozen_digest_still_matches_the_list_at_head` | B — `fbf17e07…c542ff` |
| `test_every_frozen_citation_resolves_with_its_anchor_at_head[8]` | B — eight anchors at HEAD |
| `test_the_reader_folds_every_event_type_spelling` | C.2 — ≥28 recovered REDs, and `t` beats `kind` |
| `test_the_append_gate_refuses_a_malformed_event[6]` | C.2 — six refusal classes, each asserting **nothing was written** |
| `test_the_append_gate_writes_a_record_the_reader_can_bind` | C.2 — the round trip, on a string carrying a backslash |
| `test_the_unparseable_lines_are_counted_not_silently_dropped` | C.3 — the four-line census |

**RED first.** Four RED events were appended to `house-events.jsonl` at
`2026-09-02T03:20:00`, **each with a `t` field**, naming the four nodes and
their claims, before any of the findings above. Seven GREEN events followed at
`03:31:00` through the new gate.

---

## E. LIMITS

The three asked-for columns are **not** in `scripts/v15_r1.py`; §A.4 states the
reason and prices each route. The pre-clamp answer for the *trained* cells still
requires the emit and therefore the fork; only the 0-step twin has a
hash-free route. The freeze digest still does not cover the cited files' content,
so the anchor nodes must be re-run at every HEAD and never inherited — including
from this document. One ledger record remains typeless and four remain
unparseable; the repair is forward-only by design and does not rewrite them.
The 28 recovered REDs are recovered **for a reader that calls `scale.ledger`** —
any reader still grepping the file literally is unrepaired, and this office has
not audited who they are. No GPU-seconds, no git write, nothing to Kaggle.

| | |
|---|---|
| GPU time | **0 s** |
| files changed | `scale/ledger.py` (+`_t`, +`append`, 3 filters rerouted), `house-events.jsonl` (11 appends), `tests/saturn/test_v20_r15_it9_saturn.py` (new), `V20_R15_IT9_SATURN.md` (new) |
| files NOT changed | `scripts/v15_r1.py`, `ceq/**`, `V20_R15_WING_MANIFEST.md` |
