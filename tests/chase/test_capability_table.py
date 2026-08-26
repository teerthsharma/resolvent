"""CHASE round 7 it.9 - the capability table v0, and the rule that makes it honest.

Subject: `scale/capability_table.py` (does not exist when this file is first run
- that is the RED state, and it is deliberate).

WHY THIS EXISTS. S3a is worth +3 and it does not depend on winning. The table
prints NO DIFFERENCE wherever that is what is true, and the headline cell
(`settled` vs `twin`) is exactly such a cell. A table that shows only wins is
marketing; the tests below pin the negatives into it.

THREE THINGS THE TESTS ENFORCE, all of them failure modes this repo has
actually shipped:

1. THE NEGATIVE CELL IS NOT A CONSTANT. Twelve vacuous controls have been
   struck across five authors, and the twelfth was a scan that could not have
   returned a hit on any input. So `NO DIFFERENCE` is not trusted until the
   same builder, on a journal with a planted gap, is SEEN to print a win -- in
   both directions. That is the must-fire, and it is the reason the negative
   in the shipped table means anything.

2. THE INTERVAL IN THE TABLE IS THE INTERVAL THE JOURNAL RECORDS. The prose
   deliverables and the run log disagree today: `results/m3_quintuple.txt:51`
   records `[-0.048587, +0.031557]` from the `B=10000` bootstrap that ran,
   while `CHECKLIST.md:1167` reports `[-0.042903, +0.031557]`, which is the
   EXACT enumeration over all 5**5 = 3125 resamples. Both are defensible
   estimators; only one of them was labelled. The table may carry at most one
   instrument per number, and the test recomputes it rather than trusting a
   transcription.

3. `undecided` IS STRUCTURAL, NOT A DATA OUTCOME. `MIN_T_MIXTURE = 13` and
   `max_attainable(5) = 3.80169140625 < THRESHOLD = 40.0`, so five seeds cannot
   cross in either direction whatever the data say. A table that prints
   `undecided` without its ceiling reads as weak evidence when it is in fact a
   proved impossibility, so the ceiling travels with the word.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import capability_table as CT                           # noqa: E402
from scale import eprocess as EP                                   # noqa: E402
from scale.m3_synthetic_settled import contrast                    # noqa: E402

JOURNAL = ROOT / "results" / "m3_quintuple_v2.jsonl"
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------- helpers

def _rows(path=JOURNAL):
    out = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cell = r["key"].split("_")[0]
        seed = int(r["key"].rpartition("_sd")[2])
        out.setdefault(cell, {})[seed] = r["value"]
    return out


def _planted(tmp_path, name, *, winner):
    """A copy of the real journal with one arm made unambiguously better.

    The gap is 0.5 NRMSE per seed, which is ten times the pre-registered
    resolution floor of ~0.05, so a bootstrap that can resolve anything at all
    must resolve this. Nothing else about the journal changes.
    """
    src = _rows()
    loser = "twin" if winner == "settled" else "settled"
    out = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cell = r["key"].split("_")[0]
        if cell == winner:
            r["value"] = dict(r["value"],
                              eval_nrmse=r["value"]["eval_nrmse"] - 0.5)
        out.append(json.dumps(r))
    assert loser in src and winner in src
    p = tmp_path / name
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def _contrast_row(table, arm, ref):
    for row in table["contrasts"]:
        if row["arm"] == arm and row["ref"] == ref:
            return row
    raise AssertionError(
        "no contrast row {} vs {} in {}".format(
            arm, ref, [(r["arm"], r["ref"]) for r in table["contrasts"]]))


# ------------------------------------------------------- 1. the honest negative

def test_the_headline_cell_prints_no_difference_with_its_interval():
    """`settled - twin` covers zero, and the table says so in the verdict cell."""
    t = CT.build()
    row = _contrast_row(t, "settled", "twin")
    assert row["verdict"] == "NO DIFFERENCE", row
    assert row["ci_lo"] < 0.0 < row["ci_hi"], row
    assert row["delta"] == pytest.approx(-0.0029590637137927533, abs=0.0)
    assert row["n_seeds"] == 5
    assert row["seeds_favouring_arm"] == 3


def test_the_verdict_names_the_arms_in_the_row_and_not_two_fixed_names():
    """`m3_synthetic_settled.verdict_of` hardcodes SETTLED/TWIN, so the shipped
    run log prints `argmax vs softmax -> TWIN WINS` at
    `results/m3_quintuple.txt:48` -- a row whose verdict names an arm that is
    not in it. That is the G3 shape (an arm reporting another arm's name) and
    the table may not reproduce it."""
    t = CT.build()
    row = _contrast_row(t, "argmax", "softmax")
    assert row["verdict"] == "softmax WINS", row
    assert "TWIN" not in row["verdict"].upper(), row


def test_the_interval_is_the_journal_bootstrap_and_carries_its_estimator_name():
    """One instrument per number. Recomputed here, not transcribed."""
    r = _rows()
    want = contrast([r["twin"][s]["eval_nrmse"] for s in SEEDS],
                    [r["settled"][s]["eval_nrmse"] for s in SEEDS],
                    n_boot=10000, seed=0)
    row = _contrast_row(CT.build(), "settled", "twin")
    assert row["ci_lo"] == want["ci_lo"], (row["ci_lo"], want["ci_lo"])
    assert row["ci_hi"] == want["ci_hi"], (row["ci_hi"], want["ci_hi"])
    assert row["estimator"] == "paired percentile bootstrap B=10000 seed=0"


# ------------------------------------------------- 2. MUST-FIRE, both directions

@pytest.mark.parametrize("winner,expected", [("settled", "settled WINS"),
                                             ("twin", "twin WINS")])
def test_a_planted_gap_moves_the_no_difference_cell(tmp_path, winner, expected):
    """THE MUST-FIRE. The same builder, the same cell, a planted 0.5 gap.

    Without this, `NO DIFFERENCE` in the shipped table is indistinguishable
    from a verdict function that cannot return anything else.
    """
    real = _contrast_row(CT.build(), "settled", "twin")
    assert real["verdict"] == "NO DIFFERENCE"

    p = _planted(tmp_path, "planted_{}.jsonl".format(winner), winner=winner)
    row = _contrast_row(CT.build(journal=p), "settled", "twin")
    assert row["verdict"] == expected, row
    assert row["verdict"] != real["verdict"]


def test_the_bitwise_bound_pair_reads_an_exact_zero(tmp_path):
    """G3 binds `glance` bitwise to `softmax`, so their contrast is exactly
    0.000000 with a zero-width interval. The table must be able to print a row
    that is zero BY CONSTRUCTION and label it as such, rather than as a
    measured tie -- the failure this project logged as "zero BY CONSTRUCTION
    mapped to GREEN"."""
    row = _contrast_row(CT.build(), "glance", "softmax")
    assert row["delta"] == 0.0 and row["ci_lo"] == 0.0 and row["ci_hi"] == 0.0
    assert row["verdict"] == "NO DIFFERENCE"
    assert "bitwise" in row["note"].lower(), row["note"]


# --------------------------------------- 3. undecided is structural, not weak

def test_the_undecided_cell_carries_the_ceiling_that_makes_it_structural():
    t = CT.build()
    cell = t["eprocess"]
    assert cell["decision"] is None
    assert cell["t"] == 5
    assert cell["e_settled"] == 0.9978465225545524
    assert cell["e_twin"] == 1.0019156582048507
    assert cell["ceiling_here"] == EP.max_attainable(5) == 3.80169140625
    assert cell["threshold"] == EP.THRESHOLD == 40.0
    assert cell["min_t_mixture"] == EP.MIN_T_MIXTURE == 13
    assert cell["can_decide"] is False
    text = cell["text"]
    assert "undecided at evidence E_t = 0.9978465225545524" in text
    assert "structural" in text.lower()
    assert "3.80169140625" in text and "40.0" in text and "13" in text


def test_the_rendered_table_puts_the_ceiling_beside_the_word_undecided():
    md = CT.render(CT.build())
    assert "undecided at evidence E_t" in md
    assert "3.80169140625" in md
    assert "MIN_T_MIXTURE = 13" in md
    assert "NO DIFFERENCE" in md


# ------------------------------------------------------------- 4. provenance

def test_every_number_carries_its_provenance():
    t = CT.build()
    p = t["provenance"]
    for key in ("journal", "journal_commit", "head_commit", "n_seeds", "seeds",
                "n_boot", "boot_seed", "task", "geometry", "n_params",
                "built_by"):
        assert key in p and p[key] not in (None, "", []), (key, p.get(key))
    assert p["n_seeds"] == 5 and p["seeds"] == [0, 1, 2, 3, 4]
    assert p["journal"].endswith("m3_quintuple_v2.jsonl")
    for row in t["contrasts"]:
        assert row["n_boot"] == 10000 and row["n_seeds"] == 5
        assert row["estimator"]
    for row in t["arms"]:
        assert row["n_params"] == 4769
        assert row["task"] == "negation_scope"
        assert len(row["per_seed"]) == 5


def test_an_arm_at_or_above_the_predict_the_mean_bar_is_credited_with_nothing():
    """`argmax` reads seed-mean 1.010779, above the section-5 credit bar 1.0.
    Its contrast against softmax is real and negative; the table must still
    refuse to credit it."""
    row = [r for r in CT.build()["arms"] if r["arm"] == "argmax"][0]
    assert row["nrmse_mean"] > 1.0
    assert row["beats_predict_the_mean"] is False


# ------------------------------------------------------------- 5. what it omits

def test_the_limits_paragraph_is_collected_once_and_names_what_is_missing():
    t = CT.build()
    lim = t["limits"]
    assert isinstance(lim, str)
    low = lim.lower()
    for needle in ("static", "five seeds", "consequence fidelity",
                   "counter_squared", "trained"):
        assert needle in low, (needle, lim)
    md = CT.render(t)
    # collected ONCE, at the end -- not scattered through the evidence sections
    assert md.count("## Limits") == 1
    assert md.index("## Limits") > md.index("## Contrasts")


def test_the_consequence_fidelity_column_is_declared_not_measured_with_a_reason():
    """1.7c is a capability-table column, and it is not computable from these
    artifacts: `scale/m3_quintuple.py` journals metrics only and saves no
    per-cell weights, so there are no trained `settled`/`twin`/`argmax` weights
    to intervene on. An empty column with a reason is honest; a fabricated one
    is not. The instrument itself is Foreman's (1.7c)."""
    t = CT.build()
    col = t["consequence_fidelity"]
    assert col["measured"] is False
    assert "foreman" in col["owner"].lower()
    assert "weights" in col["reason"].lower()
    for row in t["arms"]:
        assert row["consequence_fidelity"] == "NOT MEASURED"


# -------------------------------------------------------------- 6. the artifact

def test_the_upload_path_is_built_and_reported_but_never_executed(tmp_path,
                                                                  monkeypatch):
    """LEAP 4 stops at the repo boundary. `write_artifact` produces the files
    and `upload_command` names the call; neither may touch the network."""
    def explode(*a, **k):  # noqa: ANN001
        raise AssertionError("the table must not reach huggingface_hub")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", explode)

    out = CT.write_artifact(CT.build(), tmp_path)
    have = {p.name for p in pathlib.Path(out).iterdir()}
    assert "README.md" in have and "capability_table_v0.json" in have
    body = (pathlib.Path(out) / "README.md").read_text(encoding="utf-8")
    assert "NO DIFFERENCE" in body and "undecided at evidence E_t" in body

    cmd = CT.upload_command(out, "OWNER/REPO")
    assert "OWNER/REPO" in cmd and "upload_folder" in cmd
    assert os.environ.get("HF_TOKEN") is None or "HF_TOKEN" not in cmd


def test_the_package_carries_the_files_push_refuses_to_upload_without(tmp_path):
    """`ceq/hf/train.py:297` gates `push()` on three names. A folder holding a
    card and nothing else is not a package that gate would accept, and finding
    that out at upload time is finding it out too late."""
    train = pytest.importorskip("ceq.hf.train")
    out = CT.write_artifact(CT.build(), tmp_path / "pkg", with_model=True)
    have = {p.name for p in pathlib.Path(out).iterdir()}
    need = {"config.json", "configuration_ceq.py", "modeling_ceq.py"}
    assert need <= have, sorted(have)
    src = pathlib.Path(train.__file__).read_text(encoding="utf-8")
    assert 'need = {"config.json", "configuration_ceq.py", "modeling_ceq.py"}' in src
    assert "README.md" in have


def test_the_package_ships_no_random_init_weights(tmp_path):
    """A card reporting trained NRMSE beside a random-init 1.9 GB
    `model.safetensors` is the claim shape this table exists to refuse."""
    out = pathlib.Path(CT.write_artifact(CT.build(), tmp_path / "pkg",
                                         with_model=True))
    assert not list(out.glob("*.safetensors")), sorted(p.name for p in out.iterdir())
    assert not list(out.glob("*.bin"))
    assert sum(p.stat().st_size for p in out.iterdir()) < 1_000_000
    body = (out / "README.md").read_text(encoding="utf-8")
    assert "NO trained weights" in body


def test_the_module_does_not_import_huggingface_hub_at_module_scope():
    src = (ROOT / "scale" / "capability_table.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith(" "):
            assert "huggingface_hub" not in stripped, line
