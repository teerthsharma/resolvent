# SATURN (WATSON) — iteration 3 report

Unit: four new `MISTAKES.md` entries from this round's cross-audit, plus the
pilot-transfer fold. Role unchanged.

**Status: DONE_WITH_CONCERNS.**

Worktree fast-forwarded `b43da46 → f4c9d03` before any edit.
`tests/cameron/test_harmonic_attribution.py` was not touched — confirmed by
`git diff --name-only HEAD | grep -c harmonic` → `0`.

---

## 1. What landed

`MISTAKES.md` goes from 31 entries to **35**. (My iteration-1 report said "22
entries" — that was a miscount of my own file; the correct figure was 31, and it
is now mechanically counted by the test described in §4.)

| new | class | one line |
|---|---|---|
| **P-8** | provenance | A headline that states an upper bound as a price |
| **M-8** | measurement | Pricing every arm at one arm's rate |
| **M-9** | measurement | A verdict whose finest achievable p cannot reach the α it quotes |
| **V-13** | vacuous | A search whose walk includes nested checkouts |

**M-3** (pilot spread taken as the realised spread) absorbed the cost and power
faces of the same failure, as instructed. Closing check 7 was extended so the
achievable-p test sits beside the ceiling-arithmetic rule it generalises.

Placement note: V-13 is appended at the end of class V rather than inserted
beside V-7, so nothing renumbers. Five files and two reports already cite
`MISTAKES.md V-1 / V-8 / D-1 / D-4 / M-2`; all still resolve.

---

## 2. Claim ledger

Every claim below was re-verified at source. I did not take the dispatch's
numbers on trust, and four of them needed correcting.

| # | Claim | Class | check: |
|---|---|---|---|
| 1 | The `29.6 h` headline and its `5.9 h` measurement | READ | `results/r9_systems_gate.md:196` and the SUPERSEDED block at `:200-222` |
| 2 | The caveat that predicted the miss was already in the gate | READ | `results/r9_systems_gate.md:328` — *"an implementation that batches the query rows differently could beat it"* |
| 3 | …and so is the diagnosis | READ | `:332-334` — *"The caveat was correct, was written in the right place, and was still not enough"* |
| 4 | `STATE.md:21` over-prices the ladder `3.0×` | READ | `results/r9_pricing.md:175`; M22 at `mercury-report.md:49`, measured `6,047 s = 1.68 h` against `~5 h` |
| 5 | `twin` has no settle loop and skips the Gram | READ | `need_gram = self.base_cell == "settled"`, `scale/m3_quintuple.py:404` and `:444` |
| 6 | The twin/settled ratio inverts across tasks | READ | M42/M45: `1.61×` dearer at `negation_scope`/ntr8192 (medians `508.74` vs `315.65`), `2.1×` cheaper at `e3_t1`/ntr2048 |
| 7 | The `rglob` walk descended into sibling worktrees | READ | `tests/deimos/test_deimos_r9_iteration1.py:265-291`, and the repaired test asserts `hits == [ceq/nash.py]` |
| 8 | Nine worktrees, not ten | RUN | `git worktree list \| grep -c ".claude/worktrees"` → **9**; total entries **10**, the tenth being the primary checkout |
| 9 | The phantom count is not stable | RUN | `find .claude/worktrees -name nash.py \| wc -l` → **9** today against Deimos's **10** |
| 10 | N=5 is a sign test; finest two-sided p is `0.0625` | READ | `progress.md:1013-1021`: unanimity `385/385`, 4–1 `20–44 %`, 3–2 `0–3.7 %` |
| 11 | Two in-tree homes for the headline CI **disagree** | RUN | `ceq/hf_artifact/README.md:35` `[+0.066232, +0.147110]` vs `CHECKLIST.md:1168` `[+0.068181, +0.147110]` — see §3 |
| 12 | `n+` is already printed in two places | READ | both rows above carry `5/5` |
| 13 | The pilot-transfer warning is in the code, not a report | READ | `scale/m3_flops.py:115-116` — *"A pilot cost measured at small `s` must not be scaled to full geometry by this term"* |
| 14 | Every `MISTAKES.md` citation resolves and is tracked | RUN | 35 entries, **72 numbered citations**, 19 bare references, all clean but two allowlisted |
| 15 | The citation checker can fail | RUN | RED on the pre-fix text: returns `['mercury-report.md:49']`; three planted breaks all caught |

### Four corrections I had to make to the dispatch's own numbers

Stated plainly because each would have shipped into the file that documents
line-reference drift.

1. **`m3_quintuple.py:293-295`** was cited for `need_gram=False`. At `f4c9d03`
   that range is `batched_log_alpha_step`. The real sites are `:404` and `:444`
   (docstring at `:215`). This is P-6 in the citation for M-8, so the entry
   cites the **symbol** and the current lines.
2. **"nine agent worktrees"** — correct, but `git worktree list` returns **ten**
   entries because the primary checkout is one of them. The entry now states
   both numbers and which is which, and adds the measurement that the phantom
   count has already moved from 10 to 9.
3. **"The original ladder estimate, same shape"** as a third instance of M-8 —
   I could not source a third document distinct from the two. Mercury *counts*
   three (`mercury-report-it2.md:216`), the third being the subject of his own
   §4.4. The entry therefore evidences **two** and attributes the count of three
   to him, rather than inventing a bullet.
4. **Neptune's pilot quote** as given in the dispatch is a paraphrase. The entry
   uses his words: *"It nearly caught me: the `s=64` projection was drafted from
   the pilot ratio before being measured."*

One more, mine: I first wrote that the over-finding search claimed the symbol
was "read in eleven places". That was arithmetic I invented. The entry now
states only what was measured — twelve files against a tracked truth of one,
with eleven of the twelve being copies of the file being asked about.

---

## 3. A finding made while writing, recorded not fixed

**The repository's most-quoted result has two in-tree homes whose intervals
disagree.** Same point estimate `+0.108437`, same upper bound `+0.147110`, lower
bounds `0.001949` apart:

    ceq/hf_artifact/README.md:35   [+0.066232, +0.147110]
    CHECKLIST.md:1168              [+0.068181, +0.147110]

Neither row names the run that produced it, so there is no way from inside the
tree to tell which is the transcription error. This is P-1 attached to the
published headline, and the artifact README is the one that ships. It is
recorded in M-9 and **not** fixed: picking one without finding the producer
would make the disagreement invisible, which is the worse outcome. Whoever owns
`ceq/hf_artifact` should resolve it against the journal.

---

## 4. Scope-adjacent addition, flagged

`tests/cameron/test_mistakes_citations_resolve.py`, 7 tests. Not in my brief,
and here is the argument for it: `MISTAKES.md` is now 35 entries whose entire
value is the citation on each, and **both of the failure classes it documents
have already happened to it.** In iteration 1, four citations were wrong on
first write and two of those drifted because the same commit's edits moved the
lines being cited (P-6, committed by the entry that names P-6). This iteration,
`mercury-report.md:49` shipped without its directory — the checker caught it
before the commit, which is claim 15's RED.

P-5's own rule is *"assert cited symbols exist — one test that imports every
name a docstring names is cheaper than the audit that finds them missing."* This
is that test, for the document. It asserts every `file:line` exists, is
git-tracked (so a reader can actually open it), and is in range.

**Its planted-negative half is the load-bearing part.** A checker reporting
"every citation is fine" is an absence claim, and V-7 and V-13 are both about
absence claims from searches that could not have found anything — so three
deliberate breaks (missing file, line past EOF, line below 1) run through the
same extractor and the same predicate and must be caught. Without that, a regex
that silently matched nothing would print a clean bill of health forever. The
two legitimately non-resolving references are allowlisted with their reason and
**checked from both ends**: if one starts resolving, the exemption fails rather
than sitting there covering future real breaks.

---

## 5. Test summary

| file | result |
|---|---|
| `tests/cameron/test_mistakes_citations_resolve.py` | **7 passed** (RED demonstrated on the pre-fix text) |
| `tests/cameron/test_c1_propagate_registration.py` | 23 passed |
| `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` | 6 passed |

36 passed, 0 failed. The full suite was not run. No source file was modified
this iteration — the only code added is the test.

---

## 6. What I could not validate

Every entry rests on documents rather than on re-running the experiments they
record, and that is a real limit rather than a formality: I verified each
citation's location and quoted text by direct read, and I independently
re-measured only the two things that were cheap — the worktree count (9, against
the dispatch's phrasing) and the live phantom-copy count (9, against Deimos's
10). I did not re-time a single arm, so the `3.0×`, `5×`, `1.61×`, `2.1×`,
`15.5 h` and `5.9 h` figures are all `READ` from other planets' reports and the
in-tree results files, not `RUN`. If Mercury's or Neptune's timings are wrong,
M-8 and P-8 inherit the error wholesale, and neither entry would detect it.

The `385/385` sign-test measurement is Venus's and I did not reproduce it; the
`0.0625` follows from `2/2^5` by arithmetic I did check, but whether the shipped
`contrast()` actually behaves as a sign test at N=5 is her measurement alone.
M-9 claims this is general to every 5-seed reading in the repository — that is a
`DERIVED` generalisation from one instrument's behaviour, and I did not
enumerate the readings it would cover.

I could not source a third distinct instance for M-8 and said so in the entry
rather than padding it; if the coordinator has the third document, the entry
should be amended rather than left at two. Relatedly, I did not resolve the
`+0.108437` interval discrepancy in §3 — I found it, confirmed both lines, and
stopped, because resolving it needs the journal and the artifact's owner.

Finally, the new test binds citations to files and line ranges but **not to
content**: a cited line that still exists but now says something else passes.
That is precisely the P-6 half that bites hardest, and closing it would need
each entry to carry a quoted fragment the test could re-match. I did not build
that, and the file remains vulnerable to a citation whose target has been
rewritten under it.
