"""it.23 -- J-21a made fence-aware, and the argument layer ruled on.

MARS struck the it.21 heading resolver in BOTH directions (`V20_R15_IT22_MARS.md`,
STRIKE 2): `_level()` counts leading `#` on any line and knows nothing about
fenced blocks, so a `# comment` inside a fence is a level-1 heading (FAIL-OPEN: an
ambiguous anchor is laundered into a confident line number) and a heading quoted
inside a fence is a duplicate heading (FAIL-CLOSED: three LIVE anchors die).

  RULING J-23a -- A FENCE IS NOT A HEADING. ADOPTED.
  `_levels(src)` forces every line inside a fence to level 0 and is indexed at
  BOTH `_level` call sites in `resolve`. Five lines, one pass, no dependency.

  RULING J-23b -- THE EXACT-KEY HALF OF THE ROUTE IS REJECTED, WITH A NUMBER.
  MARS also asked for `l.strip() == key` in place of `key in l`. [RUN] That breaks
  ALL NINE keyed re-anchors -- every key in `HEADING_CENSUS` is a PREFIX of its
  heading line, e.g. `## it.7` against `## it.7 - opened, seven rows`. Under the
  exact form each resolves to ZERO headings and REFUSES, taking `129 of 129` to
  `116 of 129` in one edit. That is the outcome J-20b was avoiding. And the
  silent double-match he fears cannot happen: two matches is `len(heads) != 1`,
  which is a REFUSAL. The substring form fails CLOSED and LOUD already. Rejected.

  RULING J-23c -- ONE LANDING PREDICATE. ADOPTED (INSPECTOR it.20, MARS it.22).
  it.21 restated the it.20 landing rule as `want not in at`. it.20 now exports
  `lands(want, line)`; both modules call it. One place for the rule to be wrong.

  RULING J-23d -- `THREE_LINES_LOW` IS A REAL DEFECT AND ITS PROPOSED REPAIR IS
  ALSO WRONG. MERCURY is right that `CEQ_V20_R15_CONTRACT.md:239` carries `[V]`,
  not `F3`. He places `F3` at `:242`. [RUN] `:242` reads "an exact sweep-cut
  conductance; first task of the annex, and" and carries no `F3` either. `F3` is
  at `:241`, `:243` and `:244`. So the repair is NOT a pointer edit: no line in
  that item carries `F3` uniquely, and a pointer that lands uniquely needs a
  different `want` -- which under J-20a is a WITHDRAWAL plus a NEW CITATION, not
  a repair. `C13` is REFUSED-PENDING-REISSUE, not silently moved.

  RULING J-23e -- A COMPOUND CLAIM MAY NOT CITE ONE LINE. `COMPOUND_HALF` is a
  real defect and the rule it needs is general: a cell asserting N propositions
  must carry N pointers, or one range. `+0.717647` is at `:13`, `-0.032353` at
  `:14`; the cell says "both @ :13" and every location instrument scores it GREEN
  because the half it can see is true. Repairable as `:13-14`.

  RULING J-23f -- `LINE_ONE_IDIOM` IS A MISSING NOTATION, NOT A WRONG POINTER.
  [RUN] 8 occurrences across 6 files, 6.2% of the 129-occurrence census, spell
  "this file" as `path:1` -- indistinguishable from "line 1 of this file". The
  round is GIVEN THE NOTATION: `path:*` means THE FILE and is not line-landed;
  `path:1` continues to mean line 1 and IS line-landed. The idiom is retired at
  the next table edit; until then the 8 are counted here so the population cannot
  drift silently.

  RULING J-23g -- `LIVE` IS A HAND-MAINTAINED LIST AND THE INSTRUMENT NOW SAYS SO.
  Commit history cannot see it: `V20_R15_LEAP_LEDGER.md`, `V20_R15_JOURNAL.md`,
  `CEQ_V20_R15_CONTRACT.md` and `V20_R15_THEORY_TABLE.md` are ALL untracked, so
  all four have zero commits and the criterion does not separate the refused from
  the scored. The measurable join criterion that does not need git: A FILE IS LIVE
  IF ITS LINE COUNT MOVED BETWEEN TWO READS ONE ITERATION APART. Applied now, that
  criterion CONTRADICTS the hand-list: `V20_R15_JOURNAL.md` grew 3908 -> 4829
  since the it.20 census (LIVE, correctly), but `V20_R15_LEAP_LEDGER.md` is 438
  lines -- EXACTLY its it.20 census value, three iterations later. The ledger is
  on the list because an office typed it there. The list is NOT retired here (a
  static file may be appended tomorrow, and a criterion that flips a file's
  scoring status every iteration is worse than a judgement with an owner), but it
  is no longer asserted as measured. This test is what makes an unannounced edit
  to it RED.
"""

from __future__ import annotations

import re

from tests.jupiter.test_v20_r15_it20_citation_freeze import (
    LIVE_FILES,
    ROOT,
    lands,
)
from tests.jupiter.test_v20_r15_it21_heading_anchor import (
    HEADING_CENSUS,
    _level,
    _levels,
    resolve,
)

LEDGER = "V20_R15_LEAP_LEDGER.md"
CONTRACT = "CEQ_V20_R15_CONTRACT.md"
CONSTANTS = "tests/jupiter/test_v20_r15_it12_constants.py"
FENCE = "`" * 3


def _src(path: str) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()


# --------------------------------------------------------------- J-23a

def test_a_shell_comment_in_a_fence_no_longer_launders_an_ambiguous_anchor():
    """FAIL-OPEN, closed. The planted negative in this office's own words."""
    src = _src(LEDGER)
    control = resolve(LEDGER, "## it.7", "F4", src)
    assert isinstance(control, str) and control.startswith("REFUSED"), (
        f"premise gone: 'F4' is no longer ambiguous in that section ({control})")

    head = next(i for i, l in enumerate(src) if l.startswith("##") and "it.7" in l)
    first = next(i for i in range(head, len(src)) if "F4" in src[i])
    grown = (src[:first + 1]
             + [FENCE + "bash", "# a shell comment", "echo hi", FENCE]
             + src[first + 1:])

    after = resolve(LEDGER, "## it.7", "F4", grown)
    assert isinstance(after, str) and after.startswith("REFUSED"), (
        f"FAIL-OPEN. On the real file the resolver REFUSES: {control}\n"
        f"   With a fenced shell comment inserted it returns line {after} with "
        f"full confidence. The refusal J-21a calls load-bearing came back as a "
        f"number.")


def test_a_heading_quoted_inside_a_fence_no_longer_kills_the_three_live_anchors():
    """FAIL-CLOSED, closed. All three `## it.7` anchors survive the quote."""
    src = _src(LEDGER)
    live = {c: w for c, (p, k, w) in HEADING_CENSUS.items()
            if p == LEDGER and k == "## it.7"}
    assert len(live) == 3, f"premise gone: {len(live)} anchors on '## it.7', not 3"
    grown = src + ["", FENCE, "## it.7", FENCE, ""]
    dead = {c: resolve(LEDGER, "## it.7", w, grown) for c, w in live.items()}
    assert all(isinstance(v, int) for v in dead.values()), (
        "FAIL-CLOSED. Five appended lines quoting `## it.7` inside a fence "
        "retire three of the thirteen occurrences that make `129 of 129`:\n   "
        + "\n   ".join(f"{c}: {v}" for c, v in sorted(dead.items())))


def test_the_fence_flag_is_the_only_thing_that_changed():
    """`_levels` and `_level` agree line for line once fences are removed, so the
    repair narrows nothing it was not aimed at."""
    src = [l for l in _src(CONTRACT) if not l.lstrip().startswith(FENCE)]
    assert _levels(src) == [_level(l) for l in src]


# --------------------------------------------------------------- J-23b

def test_the_exact_key_form_would_break_every_keyed_reanchor():
    """[RUN] behind the rejection. Not an opinion: a count."""
    broken = []
    for cid, (path, key, _) in HEADING_CENSUS.items():
        if key is None:
            continue
        if not any(l.strip() == key for l in _src(path)):
            broken.append(cid)
    keyed = [c for c, (_, k, _) in HEADING_CENSUS.items() if k is not None]
    assert len(keyed) == 9 and sorted(broken) == sorted(keyed), (
        f"the exact-key half of MARS's route breaks {len(broken)} of {len(keyed)} "
        f"keyed anchors: {sorted(broken)}. Every key is a PREFIX of its heading "
        f"line, so each resolves to ZERO headings and REFUSES.")


def test_a_second_heading_extending_the_key_refuses_rather_than_double_matching():
    """The risk the exact-key form was meant to remove is already a refusal."""
    src = _src(LEDGER) + ["## it.7 bis"]
    got = resolve(LEDGER, "## it.7", "machine-true and domain-empty", src)
    assert isinstance(got, str) and "matches 2 headings" in got, got


# --------------------------------------------------------------- J-23d/e/f

def test_the_contract_line_the_table_cites_for_F3_carries_V_and_not_F3():
    """MERCURY's defect, and his replacement pointer, both measured."""
    src = _src(CONTRACT)
    assert "[V]" in src[238] and "F3" not in src[238], src[238]
    assert "F3" not in src[241], (
        f"MERCURY's proposed :242 does carry F3 after all: {src[241]!r}")
    where = [i + 1 for i, l in enumerate(src[235:250], start=235) if "F3" in l]
    assert where == [241, 243, 244], where


def test_the_compound_constant_claim_needs_two_lines_and_the_cell_cites_one():
    src = _src(CONSTANTS)
    assert lands("+0.717647", src[12]) and not lands("-0.032353", src[12])
    assert lands("-0.032353", src[13])


def test_the_file_pointer_idiom_is_RETIRED_at_it24_and_the_count_carried_over():
    """it.23 counted the population so it could not drift; it.24 APPLIED the
    notation. The same 8 occurrences across the same 6 files are now `:*`, and
    the digit form of the idiom is gone. This is what makes a relapse RED."""
    txt = (ROOT / "V20_R15_THEORY_TABLE.md").read_text(encoding="utf-8")
    assert re.findall(r"[A-Za-z0-9_./-]+\.(?:py|lean|md|jsonl):1\b", txt) == []
    hits = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|lean|md|jsonl):\*", txt)
    assert len(hits) == 8 and len(set(hits)) == 6, (len(hits), sorted(set(hits)))


# --------------------------------------------------------------- J-23g

def test_LIVE_is_a_hand_list_and_one_of_its_three_members_has_not_moved():
    """The ruling, made RED-able. If the ledger starts growing again, or the list
    changes, this says so instead of the list quietly being right."""
    assert LIVE_FILES == frozenset(
        ["V20_R15_JOURNAL.md", "house-events.jsonl", "V20_R15_LEAP_LEDGER.md"])
    assert len(_src(LEDGER)) == 438, (
        "the ledger moved off its it.20 census length; the hand-list and the "
        "growth criterion now agree and J-23g's example must be re-taken")
    assert len(_src("V20_R15_JOURNAL.md")) > 3908
    for f in LIVE_FILES:
        assert (ROOT / f).is_file()
