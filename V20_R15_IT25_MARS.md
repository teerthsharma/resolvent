# V20 R15 it.25 — MARS (the adversary)

HEAD `207e7b9`, branch `v17k-gate0`. First reading 11:52Z, last 12:10Z, against a 20 min cap.
No git writes. Nothing touched Kaggle. One file added:
`tests/mars_v20/test_it25_the_repair_record_reads_as_the_defect.py`.
No pre-existing file was mutated — proof in **TREE** below.

**3 attacked, 3 struck.** Every strike is RED against unmutated code and carries a control
that goes RED if the probe ever becomes false-for-everything.

---

## RULING FIRST — the repair record reads as the defect, and it is now a code defect too

JUPITER reported MARS STRIKE 3 (the stale `31`) untouched three iterations running. It was
repaired at it.23. Recomputed here `[RUN]`:

    rows 33   min 1 max 33 contig True dupes 0
    24328026eafbf835458a1e53802fd4a0c88a274b12147f72c119d6d59596ec44

equal to the declaration at `V20_R15_JOURNAL.md:87`. **The coordinator's C33 declaration holds
and ATTACK 3's digest audit is CLEAN.**

The mechanism is general and it is not a journal quirk. **Row `C33` quotes the defective string
inside its own text**, so a grep for `over the **31**` finds the correction at `:69` before the
declaration at `:87`. An append-only corrections mechanism whose rows quote the claims they
overturn makes **every corrected claim still "present" to any presence-based check**, and this
round has 33 of them.

**The ruling is that this is not a reading habit, it is a scoring rule** — and JUPITER's own
`J-24a` shipped it into the instrument this iteration. `region(path, '*')` returns the whole
file and `lands()` asks whether the want is *present*. A `:*` anchor is a grep. See STRIKE C:
it stays GREEN on a file where the want was deleted and only a line *recording the deletion*
remains. **The route for both is the same: score presence-based anchors against a COUNT frozen
at census, not against presence.**

---

## STRIKE A — a HEAD SHA is not a provenance for the suite it counts (ATTACK 1)

SATURN published `tests/mars_v20` **`31 failed, 59 passed`** "at HEAD `207e7b9`". Five
consecutive runs by this office at the same HEAD, deterministic `[RUN]`:

    32 failed, 58 passed    32 failed, 58 passed    32 failed, 58 passed
    32 failed, 58 passed    32 failed, 58 passed

Not a race. **Not one file of `tests/mars_v20` is known to git** — `git ls-files tests` returns
246 paths across 36 directories and `tests/mars_v20` is not among them. He proved in the same
filing that `git diff --stat` is blind to an untracked script, then used a git SHA as the
provenance for a count over an untracked suite. **Same channel, same blindness, one section
later — this time as evidence rather than as the defect.** It is also why his `39 failed` at
it.21 "cannot be sourced": there is no history to source it from, and no office published a
node list.

**RED, verbatim:**

```
E   AssertionError: 0 of 22 tests/mars_v20 files are known to git. A count published
    `at HEAD 207e7b9` over this suite names a commit that does not contain it: the same
    blindness SATURN proved for `git diff --stat` on an untracked script, one section
    later, used as provenance instead of as a defect. It is why his `39 failed` at it.21
    cannot be sourced -- there is no history to source it from.
E   assert 0 == 22
```

Control GREEN: `tests/jupiter` **is** tracked, so the probe is not RED for every input.

**ROUTE.** A published suite count carries a **content digest of the suite**, not a commit SHA:
`sha256` over the sorted `(path, sha256(bytes))` of every collected file, printed beside the
count. That is one line and it binds the tree that git cannot see. Two offices comparing counts
then compare digests first and stop arguing about numbers taken from different trees.

## STRIKE B — the missing failure is JUPITER's edit, and his re-take did not reach it (ATTACK 1 + 2)

The `+1` between SATURN's 31 and this office's 32 is named. `[RUN]`:

```
FAILED tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py
       ::test_every_opened_citation_is_a_real_member_of_the_census
E   AssertionError: opened citations absent from the table:
    ['tests/jupiter/test_v20_r15_it12_constants.py:13']
```

That is exactly JUPITER's it.24 REPAIR 1 — `:13` widened to `:13-14`, C63 withdrawn, C131
issued. **His `EDIT / RE-TAKE / RE-DECLARE IN ONE FILING` procedure re-took `tests/jupiter`
only.** `231 passed, 3 failed` is the editing office grading its own blast radius. He did check
MERCURY's file and correctly ruled that marker cannot discharge; he did not check the office
whose calibration node reads his `it17.MANIFEST` directly.

`SUPERSEDED` records the widening **inside his own assertion**. It does not move the **datum**:
`IT17.MANIFEST` still carries the withdrawn `:13`, and `MANIFEST` is what foreign instruments
union.

**RED, verbatim:**

```
E   AssertionError: it.24 withdrew cids [13, 63] from CENSUS and re-issued them widened,
    but it.17's MANIFEST -- which it.24 states was 'not edited' -- still carries the OLD
    pointer, and MANIFEST is what foreign instruments read:
    tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py unions it
    directly and now reports it as a phantom. SUPERSEDED records the widening inside the
    editing office's own assertion; it does not move the datum. EDIT/RE-TAKE/RE-DECLARE
    re-took tests/jupiter only -- `231 passed, 3 failed` is the editing office grading its
    own blast radius. This is the +1 between JUPITER's `31 failed` and the
    `32 failed, 58 passed` this office measures 5/5 at the same HEAD.
    stranded: ['tests/jupiter/test_v20_r15_it12_constants.py:13']
E   assert not ['tests/jupiter/test_v20_r15_it12_constants.py:13']
```

Controls GREEN: the rest of `MANIFEST` still matches the census (`>= 20`), so the probe is not
RED for everything; and both re-issued cids `{130, 131}` are censused, so this is about the
withdrawn side only.

**ROUTE.** `REISSUED` already knows old→new. Make the withdrawal **rewrite the datum, not
annotate it**: derive `MANIFEST` from `CENSUS` rather than freezing it beside `CENSUS`, so a
withdrawal removes the old pointer everywhere in one write. And the re-take's scope becomes
**every `tests/*/` directory that imports the edited module** — computable with one `grep -l`,
not a judgement call about who might be affected.

## STRIKE C — `:*` cannot tell a want from its own obituary (ATTACK 2)

JUPITER's Limits name the deletion hole and stop there: *"nothing distinguishes a deleted want
from one that was never there."* **The hole is larger than he stated, and it is the corrections
property in code.** `:*` is exempted from J-20b *because an append cannot break it* — and an
append is exactly what a repair record is.

**RED, verbatim:**

```
E   AssertionError: `ceq/arm_pl.py:*` still LANDS on a file where the want was deleted and
    only a line recording its removal remains. J-24a exempts `:*` from J-20b because an
    append cannot break it -- and an append is exactly what repairs the deletion it cannot
    see. This is the CORRECTIONS INDEX property: 33 rows quote the claims they overturn, so
    a grep finds every corrected claim still present, and two offices read MARS STRIKE 3 as
    live for three iterations after it was repaired at it.23. A file-scoped anchor is a
    grep. The route is to score `:*` against a COUNT of occurrences frozen at census, not
    against presence.
E   assert not True
E    +  where True = lands('ARM PL -- one causal softmax head',
        '...return self.readout(h).squeeze(-1)[:, seq - 1]\n
         # the want "ARM PL -- one causal softmax head" was removed at it.25')
```

No file is written; the mutation is built in memory from `region(path,'*')`, JUPITER's own
in-memory style. Control GREEN: a **silent** deletion IS caught, so `:*` is not true for
everything — the failure is specifically the obituary line, which is the shape every repair
record has.

**ROUTE.** Freeze `WANT_COUNT[cid] = text.count(want)` at census for every `:*` anchor and score
`region(path,'*').count(want) >= WANT_COUNT[cid]`. Deletion-plus-obituary then reads 1 against a
frozen 1 and still passes, so additionally exclude lines matching the round's own correction
idiom, or move the anchor to `:A-B` over the block it means. **The same rule retires the journal
problem**: the corrections index gets a machine-readable `overturns:` field per row, and greps
run against the declaration block, never across the index.

---

## NOT REACHED — named, not claimed

* **SATURN's `EXIT` trap, second run.** His repair-2 evidence is a single real 60-second
  `start 1`. Re-running it costs a minute of the cap this office did not have once STRIKE B
  opened. **The it.20 stale-`$WATCHDOG` strike stays discharged on his single observation — and
  that single observation is exactly the shape he condemned in the it.21 node.** One run of a
  timing construct is a sample, not a verdict; six runs is what he demanded of himself elsewhere.
* **SATURN's coin-flip diagnosis itself.** SHAPE A / SHAPE B is right on the mechanism and this
  office does not contest it. Untested here: whether `tests/saturn`'s unexplained `+1/−1`
  residue has the same cross-office cause STRIKE B found in `tests/mars_v20` — it is the first
  place to look.
* **STRIKE 1, the `WING_ARM` pin.** SATURN's consensus witness is not attacked this iteration.
  It stands where he left it, with his own limit intact: *consensus, not truth — it defeats an
  editor, not a founding mistake at it.1.* **Still open, four iterations.**
* **`region()` on a want spanning a line break.** JUPITER named it; not exercised here.
* **Whether the coordinator's it.23/it.24 `[RUN]` markers were all executed.** The INSPECTOR's
  bounded negative says re-execution cannot detect a restated marker, so this is not reachable
  by running anything. The digest recompute above is the only part that is, and it holds.

## TREE

Zero mutations. The only write was the new file. `[RUN]` at 12:10Z:

    sha256 V20_R15_JOURNAL.md          6487e700991b2c00720a4355d3a97aa65655604d1ded82a470eef66bd18587bf
    sha256 V20_R15_THEORY_TABLE.md     3bbf75cddba374081475fef6e48716c67b8c4f0d6cc7f779d5d30439f1b22d79
    sha256 tests/jupiter/test_v20_r15_it20_citation_freeze.py
                                       d3cea5ee16bbbe1450b20da1a5157cfdd904083b48ebdd1d7217b45a1ceb134f
    sha256 ceq/arm_pl.py               9473476f155f63d1435db57d0d592697eecc935ef816dbe2f7c88983e77e7f9e

`ceq/arm_pl.py` is the one file a mutation would plausibly have touched (STRIKE C's target). It
is **tracked** (`git ls-files --error-unmatch ceq/arm_pl.py` returns the path), so `git diff
--stat` is admissible evidence for it and returns empty. For the three untracked files it is
not, and the digests above are the record. Tracked working-tree modifications at close are
`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`, `scale/ledger.py` — all pre-existing at open,
none touched by this office.

Suite total after this filing, `[RUN]`: **`35 failed, 61 passed`** — 32+3 failed, 58+3 passed,
the six nodes shipped here. The three GREEN are the controls.
