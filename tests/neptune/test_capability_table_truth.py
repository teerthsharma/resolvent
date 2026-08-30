"""The capability-table generator must not assert what its own artifact denies.

`scale/capability_table.py` writes `ceq/hf_artifact/README.md` through
`write_artifact`, which calls `render`. Two of `render`'s assertions were true
when v0 was cut and became false afterwards, and nothing failed when they did:

  1. "This package carries NO trained weights" -- commit `0162bdd` shipped five
     verified `twin` probe checkpoints plus `weights/MANIFEST.json` into that
     same directory. The README on disk was corrected by hand; the generator was
     not, so the next table cut silently reverts the correction.
  2. "`scale/m3_quintuple.py` ... has no `--task` flag" -- `--task` is registered
     on that file's `_argparser`.

Both tests ground on the artifact rather than on prose: they read the manifest
and they build the real parser, so they stay green only while the generator
agrees with what is on disk. `test_manifest_is_non_empty_and_verified` is the
anti-vacuity guard -- with the manifest gone or empty the denial would be TRUE,
and the other weight tests skip rather than pass for the wrong reason.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from scale import capability_table as CT
from scale import m3_quintuple as M3Q

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ceq" / "hf_artifact" / "weights" / "MANIFEST.json"


def _shipped() -> list[dict]:
    if not MANIFEST.exists():
        pytest.skip("no weight manifest on disk; the denial would be true")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["shipped"]


@pytest.fixture(scope="module")
def card() -> str:
    return CT.render(CT.build(str(CT.JOURNAL)))


def test_manifest_is_non_empty_and_verified():
    """The premise of every weight test below. Without it they are vacuous."""
    shipped = _shipped()
    assert shipped, "manifest present but ships nothing -- the denial is not false"
    assert all(w["verified"] for w in shipped)


def test_render_does_not_deny_weights_while_the_manifest_ships_them(card):
    assert "carries NO trained weights" not in card
    assert "does not exist yet" not in card


def test_render_names_every_shipped_checkpoint(card):
    """A weakened denial is not enough; the card must carry the positive fact."""
    for w in _shipped():
        name = pathlib.Path(w["file"]).name
        assert name in card, "shipped checkpoint absent from the card: " + name


def test_the_no_weights_branch_still_exists_and_still_fires(monkeypatch):
    """Both branches must execute, or the guarded one is untested by the rest.

    With no manifest the denial is TRUE and the card must make it, otherwise the
    fix has merely replaced one unconditional assertion with another.
    """
    t = CT.build(str(CT.JOURNAL))
    monkeypatch.setattr(CT, "shipped_weights", lambda *a, **kw: [])
    card = CT.render(t)
    assert "carries NO trained weights" in card
    assert "## Weights shipped" not in card


def test_shipped_weights_is_empty_when_the_manifest_is_absent_or_damaged(tmp_path):
    assert CT.shipped_weights(tmp_path / "absent.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert CT.shipped_weights(bad) == []
    unver = tmp_path / "unver.json"
    unver.write_text(json.dumps({"shipped": [{"file": "x", "verified": False}]}),
                     encoding="utf-8")
    assert CT.shipped_weights(unver) == []


def test_write_artifact_describes_the_folder_it_writes(tmp_path):
    """The landmine itself, end to end.

    `write_artifact` overwrites `README.md` in `out_dir`. Written into a folder
    that carries the manifest, the README must name the checkpoints; written
    into an empty folder it must not, because there are none there to name. The
    card speaks for the directory it sits in, not for this repository.
    """
    shipped = _shipped()
    t = CT.build(str(CT.JOURNAL))

    with_w = tmp_path / "with_weights"
    (with_w / "weights").mkdir(parents=True)
    (with_w / "weights" / "MANIFEST.json").write_text(
        MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    CT.write_artifact(t, with_w)
    body = (with_w / "README.md").read_text(encoding="utf-8")
    assert "carries NO trained weights" not in body
    for w in shipped:
        assert pathlib.Path(w["file"]).name in body

    without = tmp_path / "no_weights"
    CT.write_artifact(t, without)
    bare = (without / "README.md").read_text(encoding="utf-8")
    assert "carries NO trained weights" in bare
    assert pathlib.Path(shipped[0]["file"]).name not in bare


def test_task_flag_claims_match_the_real_parser():
    """`--task` is on the parser, so no generator string may say it is not."""
    opts = {s for act in M3Q._argparser()._actions for s in act.option_strings}
    assert "--task" in opts, "premise gone: --task no longer on the parser"
    for name, text in (("TASK_SOURCE", CT.TASK_SOURCE), ("LIMITS", CT.LIMITS)):
        assert "no `--task` flag" not in text, name + " denies a flag that exists"
        assert "has no --task" not in text, name + " denies a flag that exists"
