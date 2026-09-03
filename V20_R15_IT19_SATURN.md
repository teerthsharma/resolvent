# V20 R15 — it.19 — SATURN (WATSON)

Three instrument repairs. Each opens on what the instrument currently does, carries
its RED as its own executable event before the finding, and ships a planted negative
that was applied and observed to fire.

Machine: the author's Windows 11 box, `bash` via Git for Windows, `python -m pytest`.
Tree: branch `v17k-gate0`, working tree at the it.19 dispatch.

---

## REPAIR 1 — THE PID CHANNEL LAUNDERED "UNKNOWN" AS "CONFIRMED"

### What the instrument did

`scripts/iteration_timer.sh check` reported a liveness verdict. The it.18 repair had
promoted the heartbeat from *check-call spacing* to *process liveness* by requiring
corroboration from `.claude/iteration.pids` — and the missing-file arm set
`ANY_DEAD=1`, the **permissive** value:

```sh
    if [[ -f "$PIDFILE" ]]; then
      while read -r p; do
        [[ -n "$p" ]] && ! kill -0 "$p" 2>/dev/null && ANY_DEAD=1
      done < "$PIDFILE"
    else
      ANY_DEAD=1   # nothing registered: cannot corroborate either way, say so below
    fi
```

`.claude/iteration.pids` does not exist and nothing in the repo writes it. So the
only state this repo is ever in is the one that prints, flatly:

> `NN m NN s of it had NO live process.`

about processes it never inspected. **"No evidence" and "evidence of death" were the
same sentence.** Three further defects sit in the same eleven lines:

| case | pidfile | old verdict |
|---|---|---|
| absent | — | interruption |
| empty | 0 bytes | ordinary elapsed |
| one live pid | `1234\n` | ordinary elapsed |
| one dead pid | `999999\n` | interruption |
| one live **and** one dead | both | **interruption** |
| `printf '1234'`, no newline | — | registers nothing |

`ANY_DEAD` is `∃ dead` and is never cleared, so **one finished nurse of four
licensed the interruption verdict while three were still running.** And `while read
-r p` drops a final line written without a trailing newline, so a registry written
by `printf` is silently short by one — a defect that survives into the watchdog,
which is the pidfile's only remaining reader and the thing that issues `kill`.

### RED — `t`, before the finding

```
$ python -m pytest tests/saturn/test_v20_r15_it19_pid_channel.py -q
FAILED ::test_absent_and_empty_pidfile_give_the_same_verdict
FAILED ::test_check_asserts_no_process_fact_it_cannot_source
FAILED ::test_one_live_pid_does_not_read_as_death
FAILED ::test_last_pid_without_a_trailing_newline_is_registered
4 failed in 1.61s
```

### The repair — retire the channel, rename the output

A check nobody feeds launders unknown as confirmed, so the channel is **retired**
rather than corrected, and the output now names the quantity it actually holds:

```sh
    echo "iteration $IT: ...m...s elapsed, ...m...s left of ...m"
    if [[ $GAP -gt 300 ]]; then
      echo "   ...m...s since the last check call. That is CHECK-CALL SPACING,"
      echo "   not liveness: nothing registers pids, so this instrument cannot tell a"
      echo "   suspended session from a busy one. Not evidence, and not a licence."
    fi
```

The OVERDUE branch offered `Corroborate with a dead pid`, which no writer can
supply; it now says the instrument cannot corroborate and that OVERDUE means stop.
The watchdog's loop became `while read -r p || [ -n "$p" ]`, so a pid written
without a trailing newline is registered by the one reader that kills on it.

If liveness is ever genuinely wanted the route is in the source as a `ponytail:`
note: give `$PIDFILE` a writer **first**, then predicate on `all dead AND ≥1
registered` — never `∃ dead`, and never with a missing file as the permissive
default.

### GREEN, and the planted negative applied

```
$ python -m pytest tests/saturn/test_v20_r15_it19_pid_channel.py -q
....                                                                     [100%]
4 passed in 1.43s

$ python -m pytest tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py -q
...                                                                      [100%]
3 passed in 0.89s
```

**Planted negative (applied to `scripts/iteration_timer.sh` on disk, then reverted):**
the `ANY_DEAD` block and the newline-dropping loop were put back verbatim.

```
MUTATION APPLIED (repair reverted)
FAILED ::test_absent_and_empty_pidfile_give_the_same_verdict
FAILED ::test_check_asserts_no_process_fact_it_cannot_source
FAILED ::test_one_live_pid_does_not_read_as_death
FAILED ::test_last_pid_without_a_trailing_newline_is_registered
4 failed in 2.21s

=== REVERTED, re-verify GREEN ===
....                                                                     [100%]
4 passed in 1.58s
```

The `--force` half was closed by the coordinator this iteration; the script and the
prompt now agree.

---

## REPAIR 2 — THE CLAUSE-(b) REPAIR CERTIFIED THE MANIFEST AGAINST ITSELF

### What the instrument did

The it.14 repair replaced a hardcoded wing→arm map with

```python
def _arm_of(wing: str) -> str:
    for w, clause, cite, _anchor in rows():
        if w == wing and clause == "a":
            return pathlib.Path(cite.rpartition(":")[0]).stem
```

*"because a hardcoded map would be a second place for the manifest to be wrong"*.
It was also **the only place the manifest could be caught being wrong.** After the
repair, clause (b) is checked against clause (a), and clause (a) against nothing —
a closed loop inside one file, which cannot contradict itself about a wing's
identity and can only be uniformly wrong about it.

MARS's mutation, reused rather than reinvented: exchange W1's and W3's clause-(a)
**and** clause-(b) citations together. The `V-26` preconditions are asserted and
pass — same row count, same `(wing, clause)` pairs, same multiset of citations, only
their wing changed. W1 — whose clauses (c) and (d) still cite `V16_ARM_SMPRIME.md`
and the 15.970 smprime price — comes out certified `arm_pl`.

A second defect in the same file:
`test_every_frozen_wing_is_found_and_not_merely_named` was
`@pytest.mark.parametrize("wing", ["arm_smprime", "arm_pl"])` — **arm names**, with
no wing argument at all. It asked *does this arm exist?* and never *is this arm this
wing's?*, so it could not witness a mismatch in either direction.

### RED — `t`, before the finding

```
$ python -m pytest tests/saturn/test_v20_r15_it19_wing_identity.py -q
FAILED ::test_the_identity_swap_is_caught_by_a_semantic_node
FAILED ::test_the_found_wing_node_takes_a_wing_and_not_an_arm
2 failed, 2 passed in 0.71s
```

The two that pass are the calibration (`the_true_rows_pass_every_semantic_node`) and
the `V-26` precondition — both asserted **before** the falsification is claimed.

One correction against this office's own first draft: the harness initially invoked
the found-wing node with wing ids while it was still parametrized on arm names,
which made it fire for the wrong reason and handed the strike a false GREEN. The
harness now reads the node's **own** `parametrize` list, so it never invents the
node's arguments. Zero of seven shipped nodes fire under the swap.

### The repair — a second witness, outside clause (a)

Clause (c) is the wing's **accepted kill**, cited independently of the module path
clause (a) names. One assert, satisfied by the true rows:

```python
def test_every_wings_arm_is_corroborated_outside_clause_a():
    for wing in sorted({r[0] for r in rows()}):
        arm = _arm_of(wing)
        cite, anchor = [(c, a) for w, cl, c, a in rows() if w == wing and cl == "c"][0]
        assert _norm(arm) in _norm(f"{cite} {anchor}")
```

`_norm` folds runs of non-alphanumerics to `_`, so `V16_ARM_SMPRIME.md:529` and
`ARM PL with the parity claim RETIRED` both spell their arm the way clause (a)'s
module stem does. The found-wing node is now parametrized on **wing ids read from
the manifest**, deriving the arm from the wing under test.

### GREEN, and the planted negative applied

```
$ python -m pytest tests/saturn/test_v20_r15_it19_wing_identity.py \
    tests/saturn/test_v20_r15_freeze_manifest.py \
    tests/mars_v20/test_it18_wing_identity_is_self_certified.py -q
..........................                                               [100%]
26 passed in 0.73s
```

**Planted negative (MARS's swap, applied to the real rows, nodes called unchanged):**

```
every_wings_arm_is_corroborated_outside_clause_a(): AssertionError: W1 derives arm
'arm_pl' from its clause-(a) module path, but its clause (c) -- the kill it
accepted -- names neither: V16_ARM_SMPRIME.md:529 / 'exp_scan'. Clause (a) and
clause (c) disagree about which wing this is, and only one of them can be right.
```

**One of eight nodes fires, and it is the new one.** The found-wing node does not
fire under the swap and is not expected to — both arms remain journalled — but it
can now witness a wing whose clause (a) names an arm with zero journalled cells,
which it previously could not.

---

## REPAIR 3 — THE THEORY DIGEST WAS BLIND TO WHAT THE LEAP READS

### What the instrument did

`THEORY-SHA256` is a digest over the twelve §1 grade tokens. `theory_cells()`'s own
docstring closes *"Rows only"*. JUPITER made **29 citation repairs inside the twelve
`### CELL` bodies** and measured the digest **bit-identical**, `9989f0ef…c63057`,
twice. The it.35 leap and the it.15 arena read the cell bodies; the freeze covered
none of them.

The per-cell keys collide besides. `it12.row_digest` hashes the row **value** alone,
so twelve cells yield **seven distinct digests**. JUPITER named Q4/W1 = Q4/W3 and
Q6/W1 = Q6/W3 by hand; the census is wider — **eight of twelve cells share a digest
with another cell**, so two thirds of the table is unlocatable. Q4/W1 and Q4/W3 are
byte-identical in the grade matrix, so no digest over the rows can tell the two
wings apart: swap them and nothing moves.

### RED — `t`, before the finding

```
$ python -m pytest tests/saturn/test_v20_r15_it19_theory_digest.py -q
FAILED ::test_two_wings_cells_are_distinguishable_by_the_digest_of_record
FAILED ::test_a_citation_edit_inside_a_cell_body_moves_the_digest
FAILED ::test_the_twelve_per_cell_digests_are_twelve_distinct_values
FAILED ::test_the_declared_cells_digest_matches_the_table_at_head
4 failed, 3 passed in 0.60s
```

### The repair — adopt `THEORY-CELLS-SHA256`, keyed per cell

`cell_bodies()` reads the twelve `### CELL` sections. `cell_digest(key, body)` hashes
`f"{key}|{body}"`, so two cells with equal content **cannot** share a digest.
`cells_digest()` hashes the sorted keyed per-cell digests — joint over (cell, body).
`it12.row_digest` is left untouched, because the it.12 `LEDGER-ROWS BASELINE` is
declared against it.

**Recomputed, not copied.** JUPITER's `6e347350…82189` and this office's value
differ, because the recipe differs: his is over the raw bodies, this one is over the
**keyed** per-cell digests, which is the property that makes a Q4/W1 ↔ Q4/W3 swap
move it. His number is not wrong; it is a digest of a different thing.

The population moved as J-18a says it would — *a census count is a dated
measurement, not a table constant*. Measured at it.19, on the table at HEAD of the
working tree:

```
citations (path:line in backticks) = 129
```

129, not 123. It agrees with `tests/jupiter/test_v20_r15_it18_citation_landing.py`'s
`POPULATION_AT_IT18 = 129`.

```theory-cells-freeze
# The digest of record for V20_R15_THEORY_TABLE.md, over the twelve `### CELL`
# bodies -- what the it.35 leap and the it.15 arena actually read. Per-cell digests
# are KEYED (`sha256("Qn/Wm|body")[:16]`), so two cells with equal content cannot
# collide. Supersedes THEORY-SHA256, which covers the twelve grade tokens and is
# retained only as the thing being graded.
#
# RE-DECLARED at it.28, 2026-09-02T12:51:57Z, under the it.27 subject-provenance discipline
# (stat -> read -> stat). The it.19 freeze was taken before JUPITER applied
# MERCURY's three named replacements; five cell bodies moved (Q1/W1, Q2/W3,
# Q3/W1, Q6/W1, Q6/W3) and the digest read RED for two iterations. The SUBJECT
# STAMP below is part of the declaration: a freeze that does not name the bytes
# it froze cannot be told from a stale verdict.
SUBJECT = V20_R15_THEORY_TABLE.md sha256=942e4208893444cd mtime_ns=1788352690049018500 size=43461 torn=False
SUBJECT-MTIME-UTC = 2026-09-02T12:38:10.049019Z
THEORY-CELLS-SHA256 = bef437d4e5d2612d878891f28e7f973f8208684b55e376d9dd35c91565e47d40
Q1/W1 175765879224b808
Q1/W3 be071fc4772b9e22
Q2/W1 a2701819199bde6c
Q2/W3 84c403bb34523303
Q3/W1 bc5b4074f96286a0
Q3/W3 b419ed3666872953
Q4/W1 667cfdf878ed6205
Q4/W3 b486e1a00b63e9f7
Q5/W1 c243b350d11d6f47
Q5/W3 b06897865f5e4590
Q6/W1 fd11b3f5715a00ba
Q6/W3 3b01ae50adf21e26
```

Twelve distinct of twelve, against seven of twelve before.

### GREEN, and the planted negative applied

`::test_the_planted_negative_fires_against_the_declaration` copies the **real** table
to disk, appends `EDITED-CITATION` to a line inside the first `### CELL` body, and
requires the instrument to move the whole digest **and name the one cell**. Detection
is not the bar; location is. The superseded behaviour is pinned as a premise in
`::test_a_citation_edit_inside_a_cell_body_moves_the_digest`, so re-adopting the row
digest fires a node rather than passing quietly.

---

## SUITE ACCOUNTING, WITH THE CONTROL

Two files were edited this iteration: `scripts/iteration_timer.sh` and
`tests/saturn/test_v20_r15_freeze_manifest.py`. Three files were added, all under
`tests/saturn/`.

```
$ python -m pytest tests/saturn/ -q
9 failed, 155 passed in 12.67s

$ python -m pytest tests/saturn/ -q | grep ^FAILED | sed 's/::.*//' | sort | uniq -c
      3 tests/saturn/test_journal_path_is_discoverable.py
      3 tests/saturn/test_r10_it2_spotcheck_reds.py
      1 tests/saturn/test_v20_r15_it14_saturn.py
      1 tests/saturn/test_v20_r15_wing_rubric.py
      1 tests/saturn/test_v20_r15_wings_distinct_percell.py

$ python -m pytest tests/saturn/test_v20_r15_freeze_manifest.py -q
20 passed in 0.48s
```

**Zero of the nine standing REDs are in a file this iteration edited or added**, and
they are distributed over five files, none of them touched. The control: the one
file this iteration modified runs 20/20, and the three added files run 4/4, 4/4 and
7/7. `tests/mars_v20/` carries 28 standing failures on the same run; the two MARS
files bearing on these repairs —
`test_it18_timer_liveness_defaults_permissive.py` and
`test_it18_wing_identity_is_self_certified.py` — run **5 passed**, both having been
RED before the repairs.

No `and False`, no `or True`, no short-circuited assertion is shipped in the three
new files. The it.12 `or True` disjunct is not re-introduced.

---

## THE FACT THAT IS NOT AN INSTRUMENT PROBLEM

`git show HEAD:V20_R15_THEORY_TABLE.md` → **not in HEAD.** The leap's primary input
is untracked, along with the whole `V20_R15_*` set, this record included.

**Consequence, established at it.12 and still holding: every `git log -S` over this
round's prose is structurally incapable.** A search for when a claim entered the
round's record cannot return anything, because none of it has ever been committed —
and it returns *nothing found*, which reads identically to *the claim was never
made*. Any it.19+ argument resting on a `git log -S` over `V20_R15_*` is resting on
the same shape as REPAIR 1's pidfile: an absent channel answering in the permissive
value.

**Nothing was `git add`ed.** No dispatched agent performs a git write, and whether
this round's record is committed is the author's call.

---

## LIMITS

The clause-(c) corroboration in REPAIR 2 is a **substring** test over the citation
path and its anchor, normalised. It catches an exchange between two wings whose
kills name different arms, which is the mutation on the table; it would not catch a
manifest in which clause (a) and clause (c) were moved together. Two clauses is a
second witness, not an independent one — the third would be clause (d)'s price, and
it is not asserted here. REPAIR 3's `cells_digest` covers the twelve `### CELL`
bodies and **not** the prose between sections, the §1 matrix (still `THEORY-SHA256`),
or the content of the files the citations point at — that is the resolver's job and
must be re-run at every HEAD. The `CITATION-POPULATION = 129` in the freeze block is
a dated measurement of the working tree, not a constant of the table, and it will
move again the next time a citation is repaired by splitting. REPAIR 1 leaves the
timer with **no** liveness verdict at all: a genuinely suspended session is now
indistinguishable from a busy one by this instrument, which is the honest reading
until `$PIDFILE` has a writer. None of the three repairs was measured on a machine
other than the author's box, and none of the three tests requires a GPU.
