"""A declared identity field that no record carries is not part of identity.

THE DEFECT, in one line of `scale/identity_manifest.py`:

    cfg = {k: record[k] for k in CONFIG_FIELDS if k in record}

`CONFIG_FIELDS` is the module's declaration of what makes two measured objects
different. The comprehension silently narrows it to whatever the record happens to
carry. A field declared essential and then absent is treated as "not part of
identity" rather than "identity is incomplete", and nothing anywhere says so.

MEASURED, round 10 iteration 3. Of the 16 declared `CONFIG_FIELDS`, exactly one --
`device` -- is absent, and it is absent from **60 of 60** weight records inspected
out of 165 on disk. The other fifteen are present in every record. No producer has
ever written `device`. So no published cell's manifest has ever included the device
it was computed on, and this repository parametrises `[cpu]`/`[cuda]` throughout.

The guard itself is sound where it can see. Injecting `device` into a record does
change the hash and the refusal fires naming the field, exactly like `cell` and
`beta`. `kind` and `torch_version` -- the two fields round 10 iteration 2 found had
no must-fire case -- also fire correctly when mutated. That gap was coverage, not
breakage, and this file does not re-litigate it. The hole is different: the guard
cannot fire on a field that is never there to mutate.

WHY THIS IS THE SAME DEFECT THE REPOSITORY ALREADY NAMED. `scale/journal_scan.py`
exists because a scan once "reported an absence it has not earned" -- it could not
descend to where the values lived, returned no hits, and that false absence was used
to strike a colleague's evidence. MISTAKES.md carries it as V-7. `manifest()` makes
the same move one level up: it reports an identity it has not earned, over a config
it silently truncated. A search that cannot find a thing and a hash that cannot see a
field are the same error wearing different clothes.

WHY IT MATTERS HERE SPECIFICALLY. The module's own comment gives the rationale for
keeping `torch_version` in the list:

    "a number taken under a different torch IS a number taken under a different
     object"

A number taken on a different DEVICE is a number taken under a different object by
exactly that argument, and `device` is in `CONFIG_FIELDS` because someone agreed.
The round-10 contract pre-seeds this module as "proven this branch; carried, not
re-audited", on the strength of a published cell -- and that cell is one of the 60
with no `device` key.

THE ROUTE. Two shapes, both small, and the choice is the module owner's:
  (a) `manifest()` records which declared fields were absent, so the manifest states
      its own coverage and a later reader can see the identity was partial;
  (b) producers write `device`, and `manifest()` refuses a record missing any
      declared field -- the strict form, matching `journal_scan.py`'s witness rule.
(b) is the stronger guarantee and the larger change: it invalidates the 60 existing
records as inputs, which under L-G2 means re-reads, not edits. (a) is honest and
cheap and leaves the published numbers alone. This test goes green under either.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import identity_manifest as IM  # noqa: E402

#: The published cell the round-10 contract pre-seeds as this module's proof.
PUBLISHED_UNIT = ROOT / (
    "results/m3_quintuple_v2_weights/"
    "settled_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1.pt"
)

#: How many records to open. Loading all 165 is slow and 60 already measured a
#: unanimous result; a sample that finds the field absent everywhere it looks is
#: enough to bind the claim, and the claim is scoped to the sample.
SAMPLE = 60


def absent_fields(record) -> list[str]:
    """Declared CONFIG_FIELDS the record does not carry."""
    return [f for f in IM.CONFIG_FIELDS if f not in record]


def load(path: pathlib.Path):
    import torch
    return torch.load(path, weights_only=True)


def test_the_checker_finds_a_planted_absence():
    """Must-fire. A checker that reports no absence must be able to find one."""
    probe = {f: 0 for f in IM.CONFIG_FIELDS}
    assert absent_fields(probe) == [], "the checker reports absences in a complete record"
    del probe[IM.CONFIG_FIELDS[0]]
    assert absent_fields(probe) == [IM.CONFIG_FIELDS[0]], (
        "the checker cannot see a planted absence, so a clean report means nothing"
    )


def test_the_guard_still_fires_on_every_field_that_is_present():
    """Must-not-fire in the other direction. The hole is absence, not the mechanism.

    Round 10 iteration 2 flagged `kind` and `torch_version` as having no must-fire
    case. Both fire correctly when the field exists; that was coverage, not breakage,
    and this test records it so the two findings are not confused.
    """
    if not PUBLISHED_UNIT.exists():
        pytest.skip("published cell not fetched here")
    rec = load(PUBLISHED_UNIT)
    man = lambda r: IM.manifest(r, callables=[IM.manifest], params=r["state_dict"])
    stored = man(rec)
    for field, new in [("cell", "twin"), ("kind", "__mutated__"),
                       ("torch_version", "0.0.0")]:
        if field not in rec:
            continue
        moved = dict(rec)
        moved[field] = new
        with pytest.raises(IM.IdentityMismatch) as e:
            IM.refuse_if_changed(stored, man(moved))
        assert field in str(e.value), f"the refusal does not name {field}"


def test_injecting_the_absent_field_changes_the_hash():
    """Premise. If adding `device` did nothing, the field would be decorative and
    this finding would be about the declaration, not the record."""
    if not PUBLISHED_UNIT.exists():
        pytest.skip("published cell not fetched here")
    rec = load(PUBLISHED_UNIT)
    man = lambda r: IM.manifest(r, callables=[IM.manifest], params=r["state_dict"])
    base = man(rec)["hash"]
    assert man({**rec, "device": "cpu"})["hash"] != base, (
        "adding a declared field leaves the hash unchanged; the field is decorative"
    )


def test_the_published_cell_carries_every_declared_identity_field():
    """THE DEFECT, on the one cell the contract pre-seeds as this module's proof."""
    if not PUBLISHED_UNIT.exists():
        pytest.skip("published cell not fetched here")
    missing = absent_fields(load(PUBLISHED_UNIT))
    assert not missing, (
        f"the published cell omits declared CONFIG_FIELDS {missing}. "
        "`manifest()` drops them silently at "
        "`cfg = {k: record[k] for k in CONFIG_FIELDS if k in record}`, so the cell's "
        "identity was computed over "
        f"{len(IM.CONFIG_FIELDS) - len(missing)} of {len(IM.CONFIG_FIELDS)} fields "
        "and no reader is told. Either record the coverage in the manifest, or "
        "refuse a record missing a declared field."
    )


def test_no_weight_record_omits_a_declared_identity_field():
    """The blast radius, over a sample. Scoped to the sample by construction."""
    records = sorted((ROOT / "results").rglob("*.pt"))[:SAMPLE]
    if not records:
        pytest.skip("no weight records fetched here")
    offenders: dict[str, int] = {}
    seen = 0
    for p in records:
        try:
            rec = load(p)
        except Exception:
            continue
        if not isinstance(rec, dict) or "state_dict" not in rec:
            continue
        seen += 1
        for f in absent_fields(rec):
            offenders[f] = offenders.get(f, 0) + 1
    assert seen, "no loadable weight records in the sample; the check is vacuous"
    assert not offenders, (
        f"over {seen} records, declared CONFIG_FIELDS are absent: "
        + ", ".join(f"{f} in {n}/{seen}" for f, n in sorted(offenders.items()))
        + ". A field declared to make two objects different, that no producer writes, "
        "makes no two objects different."
    )
