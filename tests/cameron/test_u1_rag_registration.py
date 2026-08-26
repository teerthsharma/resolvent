"""U1 -- THE RAG-MULTIHOP TWIN REGISTRATION (contract v10.1, U-layer).

WHAT THIS FILE BINDS. The e3-harmonic task family registers TWICE in
`M3_TASKS`: once under its equilibrium-prediction name (`e3_t*`) and once as
`rag_multihop_t*` under document-graph naming -- positions are CHUNKS, the
chain head is the ANCHOR DOCUMENT, the band positions are RETRIEVED
DOCUMENTS, the last position is the QUERY CHUNK. The contract mandates the
two registrations carry IDENTICAL TENSORS: the same seed must produce
bitwise-equal (x, y, f, p) under both names, because the rag reading is a
NAMING of the same corpus object, not a second corpus. A twin whose tensors
drifted would let the rag column quietly become a different task while
sharing the e3 numbers' provenance.

WHY RED-FIRST. Every test here was watched failing against a registry with
no rag_multihop entries (KeyError), before any rag code existed.

THE METADATA CHANNEL. The M3 example format is [n, s, d_model] with every
channel owned by convention (`CH_FLIP`, `CH_PAYLOAD`, `CH_NOISE`, then the
game block, then the E4' block up to `CH_QB`). At d_model=16 exactly one
channel is unowned: `CH_DOC`. The rag registration exposes document-graph
metadata THERE, and only on request (`mark_documents=True`): the default
call must stay BITWISE the e3 call, so metadata marking cannot be part of
the default batch. The marked variant leaves channels 0..CH_DOC-1 untouched
and keeps the executable-oracle property -- `equilibrium_oracle` reads only
CH_FLIP and CH_DRIVE, so relabelling documents cannot move any label.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import negation_scope as NS                              # noqa: E402

S_DEFAULT, D_DEFAULT = 64, 24
RAG_NAMES = ("rag_multihop_t1", "rag_multihop_t2",
             "rag_multihop_t8", "rag_multihop_t32")
E3_NAMES = ("e3_t1", "e3_t2", "e3_t8", "e3_t32")


# ------------------------------------------------------- 0. the registration --
@pytest.mark.parametrize("rag_name,e3_name", list(zip(RAG_NAMES, E3_NAMES)))
def test_rag_multihop_is_registered_with_house_shape(rag_name, e3_name):
    """Each rag sibling sits in M3_TASKS with the shipped 4-tuple shape --
    `(batch_fn, oracle_fn, feature_fn, fd_fn)` -- because
    `m3_capability.py:248` unpacks exactly four names. The oracle and feature
    slots are the CHAIN family's own: identical tensors means identical
    oracle, not a parallel implementation that could drift."""
    assert rag_name in NS.M3_TASKS, rag_name
    entry = NS.M3_TASKS[rag_name]
    assert len(entry) == 4, (rag_name, len(entry))
    batch_fn, oracle_fn, feature_fn, fd_fn = entry
    assert callable(batch_fn) and callable(fd_fn)
    #: the twin shares the chain family's oracle/features BY CONTRACT
    assert oracle_fn is NS.equilibrium_oracle
    assert feature_fn is NS.equilibrium_features
    #: and the difficulty dial is the sibling rung's dial, exactly
    e3_entry = NS.M3_TASKS[e3_name]
    assert fd_fn(S_DEFAULT) == e3_entry[3](S_DEFAULT)
    t_star = int(rag_name.rsplit("_t", 1)[1])
    assert NS.E_T_STAR[rag_name](S_DEFAULT) == t_star


# ------------------------------------- 1. IDENTICAL TENSORS, by the contract --
@pytest.mark.parametrize("t_star", [1, 2, 8, 32])
@pytest.mark.parametrize("seed", [0, 1])
def test_rag_batches_are_bitwise_the_e3_batches(t_star, seed):
    """THE CONTRACT CLAUSE. Same arguments, same seed => bitwise-equal batch
    under both names: torch.equal on x and y, equality on the returned
    positions. Anything less lets the rag column become a second corpus."""
    xe, ye, fe, pe = NS.M3_TASKS[f"e3_t{t_star}"][0](
        128, S_DEFAULT, D_DEFAULT, d_model=16, seed=seed)
    xr, yr, fr, pr = NS.M3_TASKS[f"rag_multihop_t{t_star}"][0](
        128, S_DEFAULT, D_DEFAULT, d_model=16, seed=seed)
    assert torch.equal(xr, xe)
    assert torch.equal(yr, ye)
    assert fr == fe and pr == pe


def test_the_preexisting_registry_entries_are_untouched():
    """Adding twins must not move any shipped number: the original two tasks
    and one chain rung reproduce their oracles bitwise, and the shipped
    one-sided flipper clause slot stays None for `negation_scope`."""
    x, y, f, p = NS.M3_TASKS["negation_scope"][0](64, S_DEFAULT, D_DEFAULT,
                                                  d_model=16, seed=0)
    assert torch.equal(y, NS.oracle(x, f, p))
    xc, yc, fc, pc = NS.M3_TASKS["counter_squared"][0](64, S_DEFAULT,
                                                       D_DEFAULT,
                                                       d_model=16, seed=0)
    assert torch.equal(yc, NS.counter_squared_oracle(xc, fc, pc))
    assert NS.M3_TASKS["negation_scope"][3] is None
    xe, ye, fe, _pe = NS.M3_TASKS["e3_t8"][0](64, S_DEFAULT, D_DEFAULT,
                                              d_model=16, seed=3)
    assert torch.equal(ye, NS.equilibrium_oracle(xe, fe, _pe))


# ------------------------------------------ 2. the document-graph naming ------
def test_document_metadata_channel_exists_and_is_free_at_shipped_width():
    """CH_DOC is the one channel above the E4' block, and the shipped
    d_model=16 has room for it. If the layout ever changes this test names
    the collision instead of letting two families silently share a
    channel."""
    assert NS.CH_DOC == NS.CH_QB + 1
    assert NS.CH_DOC < 16


@pytest.mark.parametrize("t_star", [1, 8])
def test_metadata_is_present_when_asked_and_absent_by_default(t_star):
    """Default call: bitwise the e3 batch, so CH_DOC carries nothing. Asked
    call: CH_DOC carries document ids, channels below it are UNTOUCHED, and
    the executable oracle recomputed from the marked tensor still reproduces
    the label bitwise -- relabelling documents cannot move a label that reads
    only CH_FLIP and CH_DRIVE."""
    kw = dict(t_star=t_star, d_model=16, seed=0)
    base = NS.make_equilibrium_batch(64, S_DEFAULT, D_DEFAULT, **kw)
    xr, yr, fr, pr = NS.make_rag_multihop_batch(64, S_DEFAULT, D_DEFAULT,
                                                **kw)
    assert torch.equal(xr, base[0]) and torch.equal(yr, base[1])

    xm, ym, fm, pm = NS.make_rag_multihop_batch(64, S_DEFAULT, D_DEFAULT,
                                                **kw, mark_documents=True)
    assert fm == fr and pm == pr
    assert torch.equal(xm[:, :, :NS.CH_DOC], xr[:, :, :NS.CH_DOC])
    assert not torch.equal(xm[:, :, NS.CH_DOC],
                           torch.zeros_like(xm[:, :, NS.CH_DOC]))
    assert torch.equal(ym, NS.equilibrium_oracle(xm, fm, pm))

    ids = xm[0, :, NS.CH_DOC]
    #: the naming is present and structured: the query chunk, the anchor and
    #: the retrieved band carry DISTINCT ids, and the unretrieved pre-context
    #: carries the not-retrieved sentinel.
    head = S_DEFAULT - 1 - t_star
    assert float(ids[S_DEFAULT - 1]) != float(ids[head])
    assert float(ids[head]) != float(ids[head + 1])
    assert float(ids[0]) == -1.0


def test_marking_requires_room_and_refuses_to_overflow():
    """The builder refuses a d_model that cannot hold the metadata channel
    rather than writing past the tensor -- same refusal class as
    `make_e4prime_batch`'s width clause."""
    with pytest.raises(ValueError):
        NS.make_rag_multihop_batch(8, S_DEFAULT, D_DEFAULT, t_star=1,
                                   d_model=NS.CH_DOC, seed=0,
                                   mark_documents=True)


# --------------------------------------- 3. the calibration hooks still work --
def test_the_rag_twin_bar_calibrates_through_its_own_registration():
    """A registration that cannot pass `calibrate_bar` through its OWN entry
    is decoration. Rung t*=1 at its measured n-floor, through the rag
    entry's four slots, with the chain's closed-form flipper dependence."""
    batch_fn, oracle_fn, feature_fn, fd_fn = \
        NS.M3_TASKS["rag_multihop_t1"]
    cal = NS.calibrate_bar(n=256, s=S_DEFAULT, d=D_DEFAULT,
                           batch_fn=batch_fn, oracle_fn=oracle_fn,
                           feature_fn=feature_fn, steps=150, lr=0.02)
    ok, why = NS.bar_verdict(cal, flipper_dependence=fd_fn(S_DEFAULT))
    assert ok, (why, cal)
