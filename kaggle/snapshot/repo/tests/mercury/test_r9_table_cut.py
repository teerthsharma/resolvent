"""Every count the capability card states must be counted, not stored.

Neptune's repair replaced the card's false `--task` denial with a true sentence
that carries three numbers: *"60 of the 100 rows ... (15 each at `e3_t1`,
`e3_t2`, `e3_t8`, `e3_t32`, against 25 bare `negation_scope` keys)"*. Two of
those were true of an 85-row journal and went stale when one more unit landed;
`100` was never right for this journal at all. A count asserted about a file
that grows is a stale claim with a delay fuse — the same defect class the clause
was rewritten to repair.

These tests ground on the journal rather than on prose: they recount it and
require the rendered card to agree, so they go red the next time the journal
moves and the generator does not.

Run this file alone:

    python -m pytest tests/mercury/test_r9_table_cut.py -q
"""
from __future__ import annotations

import collections
import json
import re

import pytest

from scale import capability_table as CT
from scale.m3_quintuple import task_of


@pytest.fixture(scope="module")
def census() -> dict:
    """Recount the journal here, independently of the generator."""
    counts: collections.Counter = collections.Counter()
    n_rows = 0
    with open(CT.JOURNAL, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n_rows += 1
                counts[task_of(json.loads(line)["key"])] += 1
    e3 = {k: v for k, v in counts.items() if k.startswith("e3_")}
    return dict(n_rows=n_rows, n_e3=sum(e3.values()), e3=e3,
                n_bare=counts[CT.TASK])


@pytest.fixture(scope="module")
def card() -> str:
    return CT.render(CT.build(str(CT.JOURNAL)))


def test_the_journal_is_not_the_size_the_stored_string_claimed(census):
    """Anti-vacuity: if the journal ever really holds 100 rows with 60 e3 and
    15 apiece, the tests below would pass against the stale literal too."""
    stale = census["n_rows"] == 100 and census["n_e3"] == 60 and \
        set(census["e3"].values()) == {15}
    assert not stale, (
        "the journal now matches the old stored literal exactly; these tests "
        "can no longer distinguish a counted number from a stored one")


def test_the_card_states_the_real_row_total(card, census):
    assert f"of the {census['n_rows']} rows" in card


def test_the_card_states_the_real_e3_total(card, census):
    assert re.search(rf"\b{census['n_e3']} of the {census['n_rows']} rows", card)


def test_the_card_states_the_real_per_task_breakdown(card, census):
    for task, n in sorted(census["e3"].items()):
        assert f"{task} {n}" in card, f"card does not carry {task} {n}"


def test_the_card_states_the_real_bare_key_count(card, census):
    assert f"{census['n_bare']} bare `negation_scope` keys" in card


def test_no_count_survives_in_the_stored_template():
    """The template must carry placeholders, not numbers."""
    assert "60 of the 100 rows" not in CT.LIMITS
    assert "15 each at" not in CT.LIMITS
    assert "against 25 bare" not in CT.LIMITS


def test_the_census_reads_the_journal_it_is_handed(tmp_path):
    """Adversarial pass: the counter must move when the file moves.

    A census that returns the same numbers for a different journal is a stored
    string wearing a function's clothes.
    """
    j = tmp_path / "tiny.jsonl"
    rows = [
        {"key": "softmax_k0_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1"},
        {"key": "twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1"},
        {"key": "settled_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t8"},
        {"key": "softmax_k0_s64_d24_st150_ntr8192_nev512_b21_sd0"},
    ]
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n_e3, n_rows, breakdown, n_bare = CT._task_census(j)
    assert (n_e3, n_rows, n_bare) == (3, 4, 1)
    assert breakdown == "e3_t1 2, e3_t8 1"


# --------------------------------------------------------------------------
# the clobber guard, re-verified against the live folder before the cut
# --------------------------------------------------------------------------

def test_regenerating_over_the_live_artifact_keeps_the_weight_facts(tmp_path):
    """Neptune's guard, exercised on the REAL artifact directory.

    `tests/neptune/test_capability_table_truth.py` builds its fixture folders
    from scratch. This one copies the shipped `ceq/hf_artifact` and regenerates
    in place, which is the operation a table cut actually performs.
    """
    import shutil
    live = CT.ROOT / "ceq" / "hf_artifact"
    if not (live / "weights" / "MANIFEST.json").exists():
        pytest.skip("no manifest on disk; the denial would be true")
    copy = tmp_path / "hf_artifact"
    shutil.copytree(live, copy)

    CT.write_artifact(CT.build(str(CT.JOURNAL)), copy)
    body = (copy / "README.md").read_text(encoding="utf-8")

    assert (copy / "weights" / "MANIFEST.json").exists(), "manifest destroyed"
    assert "carries NO trained weights" not in body
    assert "no `--task` flag" not in body
    for w in CT.shipped_weights(copy / "weights" / "MANIFEST.json"):
        assert w["file"].rsplit("/", 1)[-1] in body


# --------------------------------------------------------------------------
# the second clobber: the CLI's default output is the frozen v0 evidence
# --------------------------------------------------------------------------
#
# BOARD.md:182-191 records v0 as frozen evidence to be cited as historical, and
# v1 as the current build, "because OUT_MD/OUT_JSON hardcode the v0 paths and
# the plain command would clobber committed evidence". That left the rule in a
# document and the landmine in the tool: cutting the table the documented way
# meant hand-editing two module constants first.

def test_the_cli_targets_the_current_build_by_default():
    ns = CT._argparser().parse_args([])
    assert ns.version == "v1", (
        "the default cut must not overwrite the frozen v0 evidence")


def test_out_paths_name_the_version_they_are_asked_for():
    md, js = CT.out_paths("v1")
    assert md.name == "capability_table_v1.md"
    assert js.name == "capability_table_v1.json"
    md0, js0 = CT.out_paths("v0")
    assert md0.name == "capability_table_v0.md"
    assert js0.name == "capability_table_v0.json"


def test_writing_the_frozen_version_stays_reachable_but_explicit():
    """v0 must remain writable -- the guard is that it takes saying so."""
    ns = CT._argparser().parse_args(["--version", "v0"])
    assert ns.version == "v0"


# --------------------------------------------------------------------------
# provenance must name the repository, not the machine that cut the table
# --------------------------------------------------------------------------

def test_the_journal_provenance_is_repo_relative():
    """An absolute path is provenance nobody else can resolve.

    The field recorded `C:\\Users\\seal\\Desktop\\New folder (32)\\results\\...`,
    and cutting from a git worktree made it worse: it named a temporary
    directory that is deleted when the worktree is. `journal_commit` beside it
    already computes `journal.relative_to(ROOT)`; the path field now uses it.
    """
    t = CT.build(str(CT.JOURNAL))
    j = t["provenance"]["journal"]
    assert j == "results/m3_quintuple_v2.jsonl", j
    assert "\\" not in j and ":" not in j, "not a portable repo-relative path"


def test_an_out_of_tree_journal_still_records_its_real_path(tmp_path):
    """Only paths inside the repo are relativised; anything else is reported
    as given, because there is no repo-relative name for it."""
    j = tmp_path / "elsewhere.jsonl"
    j.write_text((CT.JOURNAL).read_text(encoding="utf-8"), encoding="utf-8")
    t = CT.build(str(j))
    assert t["provenance"]["journal"] == str(j)
