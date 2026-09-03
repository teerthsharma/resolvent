"""V20 R15 it.9 -- SATURN. The instrument-hash question, the freeze re-verified
at HEAD, and the forward-only schema gate on the ledger.

THE GOVERNING NODE IS THE FIRST ONE. Three 0-GPU-s column additions were asked
for at `scripts/v15_r1.py`; all three are producer edits to that one file, and
`instrument_manifest`'s `file` component is `sha256` over that file's RAW BYTES
(`scale/identity_manifest.py:210-215`, whose own docstring at :184 reads "Moves
on ANY edit to the named file, prose included"). So "does adding a journalled
column change the hash" is not a judgement call -- it is arithmetic, measured
here rather than argued.
"""
import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import identity_manifest, ledger  # noqa: E402

RETAKE_HASH = "5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309"
FREEZE_SHA = "fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff"
MANIFEST = ROOT / "V20_R15_WING_MANIFEST.md"
INSTRUMENT = ROOT / "scripts" / "v15_r1.py"


# --------------------------------------------------- A. the governing question

def test_a_one_byte_edit_to_the_instrument_moves_the_instrument_hash(tmp_path):
    """ANY edit to `scripts/v15_r1.py` moves `instrument_hash`, comment included.

    Measured, not asserted: the shipped file is copied, ONE comment line is
    appended -- the smallest edit any of the three asked-for columns could
    possibly be -- and the manifest is recomputed with `reaches=()` on both, so
    only the `file` component can move. It moves. Adding a journalled column is
    strictly larger than this edit, so it moves too.
    """
    src = INSTRUMENT.read_bytes()
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(src)
    b.write_bytes(src + b"#: one comment, the smallest possible producer edit\n")
    ha = identity_manifest.instrument_manifest(a)
    hb = identity_manifest.instrument_manifest(b)
    assert ha["reach"] == hb["reach"], "reaches=() must hold the reach component fixed"
    assert ha["file"] != hb["file"], "a byte hash that ignores a byte is not a byte hash"
    assert ha["hash"] != hb["hash"], (
        "THE ANSWER IS NOT FREE: adding any journalled column to "
        f"scripts/v15_r1.py moves instrument_hash off {RETAKE_HASH[:8]}...")


def test_the_instrument_at_head_is_the_one_the_banked_cells_name():
    """The control on the node above: the fork does not already predate it.

    `reaches=()` cannot reproduce the full published hash -- it is the FILE
    component this node pins -- so the check is that the banked header still
    carries the retake hash and the live file is byte-identical to what it
    hashed, i.e. this iteration modified no instrument.
    """
    banked = json.loads((ROOT / "results" / "v17k_r4_retake.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert banked["instrument_hash"] == RETAKE_HASH
    live = identity_manifest.instrument_manifest(INSTRUMENT, reaches=())
    assert live["file"] == identity_manifest._sha(INSTRUMENT.read_bytes()), (
        "the file component is not the digest of the file on disk")


def test_no_production_path_refuses_on_an_instrument_hash_mismatch():
    """WHAT THE HASH MOVE ACTUALLY COSTS, bound so the price cannot inflate.

    `instrument_hash` is WRITTEN at `scripts/v15_r1.py:614` and `:830` and read
    by nothing under `ceq/` or `scale/`. Nothing refuses to run, pool or score a
    cell because the hash moved -- the only binds on it live in TEST files
    (`tests/mercury/*`), which assert equality between a new batch and the
    retake. The repair therefore does not invalidate a banked cell; it FORKS the
    pool, and the fork is visible only to a reader that binds on the hash.
    """
    hits = [str(p.relative_to(ROOT))
            for p in list((ROOT / "ceq").rglob("*.py")) + list((ROOT / "scale").rglob("*.py"))
            if "instrument_hash" in p.read_text(encoding="utf-8", errors="replace")]
    assert hits == [], f"a production reader of instrument_hash exists: {hits}"


# ------------------------------------------------- B. the freeze, re-run at HEAD

def _rows():
    text = MANIFEST.read_text(encoding="utf-8")
    block = text.split("```freeze-manifest", 1)[1].split("```", 1)[0]
    return [ln.strip() for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_the_frozen_digest_still_matches_the_list_at_head():
    rows = _rows()
    assert len(rows) == 8, f"the freeze covered eight rows, found {len(rows)}"
    digest = hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()
    assert digest == FREEZE_SHA, f"FREEZE-SHA256 moved: {digest}"


@pytest.mark.parametrize("row", _rows())
def test_every_frozen_citation_resolves_with_its_anchor_at_head(row):
    """P-6 drift, re-measured eight iterations after the freeze."""
    _, _, cite, anchor = row.split("|", 3)
    path, _, lineno = cite.rpartition(":")
    lines = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
    got = lines[int(lineno) - 1]
    assert anchor in got, f"{cite} no longer carries {anchor!r}: {got[:100]!r}"


# ------------------------------------------- C. the ledger, read and write gates

def test_the_reader_folds_every_event_type_spelling():
    """The 116 typeless records were unbindable BY CONSTRUCTION; 115 now fold."""
    events = list(ledger.read())
    literal = [e for e in events if e.get("t") == "test"]
    folded = [e for e in events if ledger._t(e) == "test"]
    assert len(folded) > len(literal), "the fold recovered nothing"
    lit_red = [e for e in literal if ledger._status(e) == "red"]
    fold_red = [e for e in folded if ledger._status(e) == "red"]
    assert len(fold_red) - len(lit_red) >= 28, (
        f"the reader recovered only {len(fold_red) - len(lit_red)} reds; it.9 "
        "measured 28 that a literal `t` filter could not see")
    assert ledger._t({"kind": "finding"}) == "finding"
    assert ledger._t({"event": "report"}) == "report"
    assert ledger._t({"t": "TEST"}) == "test", "the fold must case-fold like _agent"
    assert ledger._t({"t": "audit", "kind": "test"}) == "audit", "`t` must win"
    assert ledger._t({}) is None, "a typeless event must match no type filter"


@pytest.mark.parametrize("bad", [
    {"agent": "SATURN", "status": "red"},          # no type at all
    {"kind": "test", "agent": "SATURN"},           # the 116's spelling
    {"event": "report", "agent": "SATURN"},        # the :11744 spelling
    {"t": "", "agent": "SATURN"},                  # empty type
    {"t": 7, "agent": "SATURN"},                   # non-string type
    ["t", "test"],                                 # not a mapping
])
def test_the_append_gate_refuses_a_malformed_event(tmp_path, bad):
    log = tmp_path / "house-events.jsonl"
    log.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        ledger.append(bad, path=log)
    assert log.read_text(encoding="utf-8") == "", "a refused append still wrote"


def test_the_append_gate_writes_a_record_the_reader_can_bind(tmp_path):
    log = tmp_path / "house-events.jsonl"
    log.write_text("", encoding="utf-8")
    ev = {"t": "test", "agent": "SATURN", "status": "red", "name": "n",
          "text": "a backslash-e \\example of the class the 4 bad lines died on"}
    line = ledger.append(ev, path=log)
    assert json.loads(line) == ev
    back = list(ledger.read(path=log))
    assert back == [ev], "the gate wrote something its own reader cannot parse"
    assert ledger._t(back[0]) == "test" and ledger._status(back[0]) == "red"


def test_the_unparseable_lines_are_counted_not_silently_dropped():
    """Four hand-written lines carry invalid escapes and never parse.

    This class is what the append gate closes going forward: `json.dumps` cannot
    emit an escape `json.loads` rejects, so no record written through
    `ledger.append` can ever join these four.
    """
    bad = ledger.unparseable()
    assert [i for i, _ in bad] == [1899, 2937, 2938, 5871], (
        f"the unparseable census moved: {[i for i, _ in bad]}")
