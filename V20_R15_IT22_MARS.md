# V20 R15 it.22 — MARS (adversary)

Branch `v17k-gate0`, repo HEAD `207e7b9`, this box (Windows 11, Git for Windows
bash). **3 repairs attacked, 3 struck.** Every RED below runs against the shipped
tree with nothing mutated — no `git stash`, no edit to any file the round owns,
no edit to the code under test.

Instrument: **`tests/mars_v20/test_it22_the_repairs_of_it21.py`** (4 nodes, `4 failed`).

| # | it.21 repair | verdict | RED node |
|---|---|---|---|
| STRIKE 1 | SATURN's re-imported `WING_ARM` pin | **struck — a fifth clause, not a second witness** | `test_the_wing_pin_is_a_fifth_clause_not_a_second_witness` |
| STRIKE 2 | JUPITER's `J-21a` heading instrument | **struck — the resolver cannot see a fenced code block; fails OPEN and CLOSED** | `test_a_shell_comment_in_a_fenced_block_launders_an_ambiguous_anchor`, `test_a_heading_quoted_inside_a_fence_kills_three_live_anchors` |
| STRIKE 3 | The coordinator's `INDEX-SHA256` re-declaration | **struck — the digest is right, the sentence publishing it is stale by one row** | `test_the_index_recipe_states_a_row_count_it_no_longer_reads` |
| upheld | `C32`'s `38,271` byte claim | **verified, not struck** | `[RUN] wc -c V20_R15_IT18_INSPECTOR.md` → `38271` |
| not reached | SATURN's `_owns_watchdog` body | — | see §5 |

**`[RUN] python -m pytest tests/mars_v20/test_it22_the_repairs_of_it21.py -q` →
`4 failed in 0.54s`.**

---

## 1. STRIKE 1 — the pin raises the price of the move from four edits to five

SATURN re-imported `WING_ARM = {"W1": "arm_smprime", "W3": "arm_pl"}` on the
ruling that *"the direction it is read in"* is what licenses it: struck at it.14
as a **source**, restored as an **assertion target**. His own limit named the
cost — *"wing identity now rests on a two-entry map that nothing corroborates."*

It is worse than uncorroborated, and the reason is his own threat model. `V-26`
is an editor who moves **every clause of the manifest together**. The pin lives
in the same tree, was typed by the same office, and is checked only against
`_arm_of`, which derives the arm name **from the manifest's own clause-(a)
citation** — the very clause the move edits. So the editor now needs five edits
instead of four. That is a **price**, not a **witness**. A witness is a record
the editor does not control.

**The witness was already open on the desk.** `tests/saturn/test_v20_r15_it14_saturn.py`
reads `V20_R15_LEAP_LEDGER.md` at `:31`, `:367` and `:435` — for the `L-16`
recount and for `## RULING J-17d`. That same file binds both wings to both arms
in prose written by the ledger office, at seven separate rows
(`:22` `Q1 / W3 `arm_pl``, `:23` `Q1 / W1 `arm_smprime``, `:120`, `:121`,
`:124`, `:126`, …). The module opens the corroborator twice for other facts and
does not open it for this one.

### RED, verbatim, unmutated

```
E  AssertionError: WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:39
E  ({'W1': 'arm_smprime', 'W3': 'arm_pl'}). The only node that checks it --
E  test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set -- compares it
E  to `_arm_of`, which derives from the manifest's own clause-(a) citations, i.e. the
E  very clauses the V-26 move edits. So the pin is not a second witness; it is a fifth
E  clause. An editor who moves the four clauses and this one line leaves both guardian
E  nodes green. Meanwhile this same module already opens V20_R15_LEAP_LEDGER.md for two
E  other facts, and that file binds W1 -> arm_smprime, W3 -> arm_pl in prose written by
E  a different office, which no node reads. The corroborator was on the desk and was
E  not opened.
E  assert not True
```

The node also **establishes the route is available before it strikes**: its first
assertion is that `_ledger_binding()` — a regex `(W\d+)\s+`(arm_[a-z_]+)`` over
the ledger — returns exactly `{"W1": ["arm_smprime"], "W3": ["arm_pl"]}`. That
assertion **passes**. The corroborator agrees with the pin today; nothing checks
that it still will tomorrow.

### Route — **reroute**: derive the pin, do not type it

Replace line 39 with the ledger read. The map becomes a **derivation from a file
in a different office's custody**, and the V-26 move stops being an edit count
and becomes a contradiction:

```python
WING_ARM = _ledger_binding()   # V20_R15_LEAP_LEDGER.md, seven corroborating rows
```

**Measured reason.** Cost is one function this file already has the imports for
(`LEDGER` is defined at `:31`). The gain is the only thing the four-clause move
cannot satisfy: an editor who moves the manifest must now also edit
`V20_R15_LEAP_LEDGER.md`, a file the freeze manifest does not own and whose rows
are cited by `L-1`, `L-2`, `L-9`, `L-10`, `L-11`, `L-12` and `L-14`. If the
derivation ever returns 0 or 2 arms for a wing, that is a REFUSAL and the pin
says so — the same shape `J-21a` uses one strike below.

**Retire nothing.** The pin's *assertion* is correct and should stay; only its
*provenance* is wrong.

---

## 2. STRIKE 2 — `J-21a`'s resolver has no idea what a fenced code block is

The heading instrument is 32 lines and one of them decides everything:

```python
def _level(line: str) -> int:
    s = line.lstrip()
    return len(s) - len(s.lstrip("#")) if s.startswith("#") else 0
```

`_level` counts leading `#` on **any** line. Markdown does not: a `#` inside a
` ``` ` fence is a shell comment, not a heading. `resolve()` uses `_level` twice —
once to count candidate headings (RESOLVE) and once to find the section boundary
(SCOPE) — so **every fenced `#` line in a LIVE file is a level-1 heading to this
instrument**, and level 1 is same-or-shallower than every section key in the
census.

`J-21a` states uniqueness at both steps *"is the entire mechanism."* Both steps
break, in opposite directions.

### RED A — FAIL-OPEN, silent. A shell comment launders an ambiguous anchor

```
E  AssertionError: FAIL-OPEN. On the real file the resolver REFUSES:
E  REFUSED: 'F4' appears on 4 lines in section '## it.7' of V20_R15_LEAP_LEDGER.md, not 1
E     After a 4-line ```bash fence carrying one `# comment` is inserted inside the
E     section, it returns line 24 with full confidence.
E     The `# comment` is read as a level-1 heading, the section ends there, and 3 of
E     the 4 occurrences are scoped out of existence. Uniqueness was not established;
E     it was manufactured by markdown the instrument cannot parse.
E  assert (False)
E   +  where False = isinstance(24, str)
```

This is **JUPITER's own third refusal negative** — `REFUSED: 'F4' appears on 4
lines in section '## it.7'`, quoted verbatim in his §3 — turned into a confident
`:24`. His ruling says the resolver exists so that it will not *"report a line
number with full confidence"* on a section it has mis-scoped. It does exactly
that, and nothing in the census can tell, because a laundered anchor **passes**.

### RED B — FAIL-CLOSED, loud. Quoting a heading kills three live anchors

```
E  AssertionError: FAIL-CLOSED. Appending five lines that quote `## it.7` inside a
E  ``` fence -- the house style of every report this round has filed -- kills all three
E  LIVE anchors at once:
E     H10: REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1
E     H11: REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1
E     H9:  REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1
E     J-20b retired digit anchors because they expire on the next append to a LIVE file.
E     J-21a expires on a different append to the same LIVE file.
```

**The premise of `J-20b` is that a LIVE file gets appended to and a digit anchor
expires.** The replacement expires on the same event, one append later. `H9`,
`H10` and `H11` are three of the thirteen occurrences that make the number `129
of 129`, and five lines of the round's own house style — a quoted heading in a
fence — take all three at once.

Both nodes prove their **premise on the real file first** (`F4` is genuinely
ambiguous today; `H9`/`H10`/`H11` genuinely resolve today) and grow their input
**in memory only** — the same methodology as JUPITER's
`test_the_heading_anchor_SURVIVES_AN_APPEND_THAT_BREAKS_THE_DIGIT`. **No file the
round owns was written.**

**Latency, stated honestly.** A sweep of both anchored files today finds **0
ghost headings inside fences** in `V20_R15_JOURNAL.md` (4,499 lines) and **0** in
`V20_R15_LEAP_LEDGER.md` (438 lines). The defect is **latent, not live** — and
`J-20b` was written on the premise that these two files grow every iteration, so
a latency measured at one iteration is not a defence. `V20_R15_JOURNAL.md:88`
already carries a ` ```bash ` fence; it is above the anchored sections, not
inside one, and nothing prevents the next append from putting one inside.

### Route — **reprice**: a five-line fence flag, at the top of `resolve`

```python
def _levels(src):
    """_level per line, with fenced blocks forced to 0."""
    out, fence = [], False
    for l in src:
        if l.lstrip().startswith("```"):
            fence = not fence
            out.append(0)
        else:
            out.append(0 if fence else _level(l))
    return out
```

`resolve` then indexes `_levels(src)` in both places it currently calls
`_level(...)`. **Measured reason.** Five lines, one pass, no new dependency; it
closes RESOLVE and SCOPE together because both read the same list. Against the
`J-20b` alternative — refuse every anchor into a LIVE file — this keeps the 13
scored and keeps `129 of 129` honest instead of dropping it to `116 of 129`.

**Second half of the route — retire the substring key.** `heads` matches on
`key in l`, so `## it.7` also matches `## it.7 bis`. Anchor on the heading line's
**normalised full text** (`l.strip() == key`) or keep the substring and accept
that any future heading extending an existing one is a silent double-match.
JUPITER already paid for this once: *"three of the ten re-anchors (`H10`, `H18`,
`H117`) needed a longer `want` or a longer key precisely because the short form
hit this."* Three of ten is not a corner case; it is the common path.

### The flag the INSPECTOR raised and did not pursue

He is right and it costs one line. `tests/jupiter/test_v20_r15_it21_heading_anchor.py`
**reimplements** the it.20 landing check inside `not_landing_by_heading` as
`want not in at`, rather than importing the it.20 predicate. The two agree today.
The it.21 file already imports `CENSUS`, `WITHDRAWN`, `WAN_SEAL` and `line_at`
from the it.20 module; **import the predicate too** and the reimplementation
stops being a second place for the landing rule to be wrong — which is precisely
the argument it.14 used to delete a hardcoded map, and precisely the argument
STRIKE 1 above re-applies to SATURN.

---

## 3. STRIKE 3 — the index digest is correct and its own count is stale by one

**Recomputed, as instructed.** Running the file's **own published recipe** on the
unmodified `V20_R15_JOURNAL.md`:

```
[RUN] python -c "import re,hashlib,io;s=io.open('V20_R15_JOURNAL.md',encoding='utf-8').read();
                 print(hashlib.sha256(chr(10).join(re.findall(r'^\| C\d+ \|.*$',s,flags=re.M)).encode()).hexdigest())"

rows   32   (C1 .. C32, no gaps)
digest 5f07ca0e4385711c590ef151044aa4cacc44bbbf34154c331dc8d6c843749ffa
```

**The digest is CORRECT.** It reproduces the published value exactly, over 32
rows, with `C32` present. SATURN's node `6 passed` stands on the arithmetic.

**The sentence that publishes it does not.** `V20_R15_JOURNAL.md:86` reads:

> **INDEX-SHA256 = `5f07ca0e…`** over the **31** `C`-rows, sorted as written.

The recipe reads **32**. `C32` was appended and the count in the publication
sentence was not moved with it.

### RED, verbatim, unmutated

```
E  AssertionError: the index publishes `INDEX-SHA256 = 5f07ca0e...` over "the **31**
E  `C`-rows". The published recipe, run on the unmodified file, reads **32** rows
E  (C1..C32) and reproduces that digest exactly. The digest is right and its own count
E  is stale by one: `C32` was appended without moving the number. A reader who counts
E  31 and stops has been told the wrong population by the sentence whose only job is
E  to state it.
E  assert 31 == 32
```

### The ruling asked for: does removing the count weaken the recipe?

**The question is moot in this tree and the answer is the opposite of the
premise.** The prose was **not** changed to *"over the `C`-rows"* — it still
reads *"over the **31** `C`-rows"*. So this iteration is not a test of a
count-free recipe; it is a demonstration of what a **stale** count does, which is
worse than either alternative.

**On the merits, stated anyway.** A count is the **only** part of this recipe
that can detect a **deletion**. The digest cannot: delete `C7` and the digest
changes, but a reader has nothing to compare the new digest against except the
sentence that was rewritten in the same edit. The count is the independent
handle — count the rows yourself, compare to the published number, *then* trust
the digest. That is the same argument MARS made in the pid channel at it.20:
`$PIDFILE` failed because a file naming a number is not evidence for the number
unless something outside the file states what the number should be.

**SATURN's objection to a literal count is real but mis-aimed.** A literal count
does rot on every append — this tree proves it, at one iteration's distance. The
fix is not to delete the count; it is to make the count **derived and asserted**,
which is the same route STRIKE 1 gives the wing pin:

### Route — **reprice, do not retire**: assert the count, do not narrate it

1. Correct `V20_R15_JOURNAL.md:86` to **32**. One character.
2. Extend the published recipe by one line so it prints the count with the digest
   and states the expected range, making the sentence self-checking:
   ```
   rows 32 (C1..C32, contiguous)   digest 5f07ca0e…
   ```
3. Keep `test_the_index_recipe_states_a_row_count_it_no_longer_reads` (or
   SATURN's equivalent) in the tree, so the next append that moves the digest and
   not the sentence is RED at the next `pytest` rather than at the next audit.

**Measured reason.** Cost is one edit plus one already-written node. The gain is
the deletion channel: with 3, removing `C7` fails the contiguity assertion even
if the digest and the sentence are rewritten together, because `C1..C31` with a
`C32` present is not contiguous. Neither the digest alone nor a count-free recipe
catches that.

### `C32`'s row — checked, and it is honest

`C32` asserts `wc -c` reads `38,271`. **`[RUN] wc -c V20_R15_IT18_INSPECTOR.md`
→ `38271`.** Re-run on this box, this minute, by this office. The number is
correct. Whether the coordinator ran it or restated the INSPECTOR's cannot be
read out of the tree — but the it.21 rule asks for a number that survives a
`[RUN]`, and this one does. **No strike.** The two timings `C32` challenges
(`14m45s`, `15m10s`) remain unsourced, which is what the row says.

---

## 4. Route summary

| strike | action | cost | what it buys |
|---|---|---|---|
| 1 | **reroute** — `WING_ARM = _ledger_binding()` | 1 line, imports already present | the V-26 move must contradict a file in another office's custody, not just cost a fifth edit |
| 2a | **reprice** — `_levels()` fence flag in `resolve` | 5 lines, one pass | closes RESOLVE and SCOPE together; keeps `129 of 129` instead of dropping to `116 of 129` |
| 2b | **retire** — substring key match | `l.strip() == key` | ends the double-match that already forced 3 of 10 re-anchors to lengthen |
| 2c | **reroute** — import the it.20 landing predicate | 1 import | one place for the landing rule, per it.14's own argument |
| 3 | **reprice** — assert the count, print it with the digest | 1 char + 1 node | restores the deletion channel a bare digest cannot cover |

---

## 5. Limits

`_owns_watchdog`'s body was **not opened** this iteration and remains where the
it.20 audit left it — unreached. SATURN's two stated limits are therefore
untested by this office: the `trap` is still proven only on the shipped body text
at a one-second deadline rather than through a real `start 1`, and a `SIGKILL`ed
watchdog still runs no trap. The clock was spent on the three named repairs and
the watchdog was the fourth. **Not reached, not cleared.** STRIKE 2's fence
defect is **latent**: 0 ghost headings exist in either anchored file today, so
the RED is a projection of the next append rather than a report of a current
miss — the same evidentiary standing as JUPITER's own append negative, and it
should be read with the same weight, no more. STRIKE 3's ruling on removing the
count is delivered against a prose change that does not exist in this tree.
`_ledger_binding()`'s regex is a MARS instrument, not a round-blessed one; it
returns exactly one arm per wing today and is not proof that it always will.

## 6. Tree statement

**No mutation was made to any file the round owns, so nothing needed reverting.**
Both heading strikes grow their input in memory; the pin strike reads source with
`ast` and writes nothing; the index strike is a read and a hash. `git status
--porcelain` at filing, honestly and in full:

```
 M MISTAKES.md
 M house-events.jsonl        <- appended by scripts/iteration_timer.sh check, this office's own runs
 M pytest.ini
 M scale/ledger.py
?? tests/mars_v20/test_it22_the_repairs_of_it21.py   <- this filing's instrument
?? V20_R15_IT22_MARS.md                              <- this document
?? V20_R15_JOURNAL.md, V20_R15_LEAP_LEDGER.md, V20_R15_THEORY_TABLE.md,
?? V20_R15_WING_MANIFEST.md, CEQ_V20_R15_CONTRACT.md, scripts/, results/, V20_R15_IT*_*.md
```

The four ` M` entries were already modified at this iteration's start and are not
this office's. `house-events.jsonl` is appended by `iteration_timer.sh check`,
which this office ran three times. Everything else untracked is the round's
standing set plus concurrent it.22 work — **SATURN and MERCURY are both live in
this tree**, and no claim here rests on a file either of them owns being frozen.

**No git writes were made. Nothing touched Kaggle.**
