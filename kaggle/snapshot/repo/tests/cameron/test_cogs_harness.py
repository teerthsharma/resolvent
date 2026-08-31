"""COGS, wired. The blocker in `harness.run("cogs")` was a fixed 256-way head.

Round 2 stopped here: `lm.TinyLM` was built byte-level, COGS needs 874 word
types and sequences past 500, and another agent was live in `lm.py`. That agent
is done, so these are the tests that unblock it.

WHY THIS BENCHMARK AND NOT ANOTHER VAL-LOSS DECIMAL. The module's one
distinguishing property is content-conditional sign -- whether a third token can
decide if j helps or hurts i. Next-byte prediction on TinyStories does not need
that property and cannot see it. COGS can: arXiv:2108.12284 Table 3 runs four
attention variants through one fixed recipe on this task and reads 0.80 / 0.81 /
0.78 / 0.77, so the benchmark moves about four points when the attention changes
and nothing else does. TinyStories val loss has no published precedent of
responding to an attention change at all.

That four points is smaller than what this project claimed for three rounds. The
"Edge Transformer 0.874 against a 0.784 control" figure is from the
GRAPH-PREDICTION version of COGS, not this one -- see
`test_the_cogs_reference_carries_the_altered_attention_precedent`.

Every test parametrizes over cpu and cuda per the standing rule.
"""
from __future__ import annotations

import pytest
import torch

from ceq import harness, lm

# ------------------------------------------------------------------ the bars
# Pre-registered here, in code, before any COGS number was measured. A bar
# written after the result is not a bar.
#
# RESOLUTION GATE. `tests/cameron/test_arc_reality.py` is this project's
# standing lesson: two arms near zero separate nothing and the run says
# nothing. COGS in-distribution accuracy is 0.96-0.99 for the published
# encoder-decoder, so if the softmax control cannot clear a generous floor on
# the in-distribution split at the shared budget, the generalization column is
# noise and must not be read.
RESOLUTION_FLOOR = 0.20
# WIN BAR. Absolute exact-match margin the signed arm must take on the
# generalization split, median over seeds.
WIN_MARGIN = 0.02


def test_tinylm_head_matches_a_requested_vocab(device):
    """`lm.TinyLM` must size its embedding and head to a requested vocab.

    The 256-way head is a byte-level default, not a design commitment. COGS has
    874 word types; truncating them to 256 would report a number that looks
    like a COGS score and is not one.
    """
    m = lm.TinyLM("softmax", d=32, n_layers=1, n_heads=2, seq=16, vocab=874,
                  seed=0).to(device)
    out = m(torch.zeros(2, 16, dtype=torch.long, device=device))
    assert out.shape == (2, 16, 874), out.shape
    assert m.tok.num_embeddings == 874, m.tok.num_embeddings


def test_a_bigger_vocab_does_not_disturb_the_byte_default(device):
    """The default must stay exactly what every published number here used."""
    m = lm.TinyLM("softmax", d=32, n_layers=1, n_heads=2, seq=16, seed=0).to(device)
    assert m.head.out_features == lm.VOCAB == 256, m.head.out_features


def test_cogs_vocab_covers_every_token_in_all_three_splits(device):
    """One vocabulary over train, test and gen, or the eval is scoring UNKs.

    COGS' generalization split introduces no new word types -- that is the
    point of the benchmark, the words are familiar and the STRUCTURES are new.
    A vocabulary built from train alone would still cover it, but building from
    all three makes that a checked fact rather than an assumption.
    """
    vocab = harness.build_vocab("cogs")
    for which in ("train", "test", "gen"):
        for src, tgt in harness.load_cogs(which):
            missing = [w for w in src + tgt if w not in vocab]
            assert not missing, (which, missing[:5])


def test_cogs_training_sequences_fit_the_context(device):
    """Nothing in the training split may be silently truncated.

    Truncation on the train side is not a fair-to-both-arms artifact: it
    removes supervision, and a benchmark scored against a published 0.35 must
    train on the same data that number was trained on.
    """
    longest = max(len(s) + len(t) + 2 for s, t in harness.load_cogs("train"))
    assert longest <= harness.COGS_SEQ, (longest, harness.COGS_SEQ)


def test_greedy_decode_batched_matches_one_at_a_time(device):
    """The batched decoder is a speedup, so it must be a parity oracle first.

    21,000 generalization items at one forward pass per output token is not a
    run anyone finishes. Batching is the fix; a batched decoder that quietly
    disagrees with the per-item one would make every score below unfalsifiable.
    """
    vocab = {"a": 0, "b": 1, "<pad>": 2, "<sep>": 3, "<eos>": 4}
    inv = {i: w for w, i in vocab.items()}
    m = lm.TinyLM("softmax", d=32, n_layers=1, n_heads=2, seq=16, vocab=5,
                  seed=0).to(device).eval()
    cmds = [["a"], ["a", "b"], ["b", "b", "a"]]
    one = [harness.greedy_actions(m, c, vocab, inv, device, max_new=6) for c in cmds]
    many = harness.greedy_batch(m, cmds, vocab, inv, device, max_new=6)
    assert many == one, (many, one)


@pytest.mark.slow
def test_run_cogs_emits_a_three_column_table(device):
    """`run("cogs")` must return both arms AND the published reference.

    The harness refuses a two-column result on purpose: a table without the
    published number invites the reader to supply a baseline from memory, and
    this project has already had to strike claims that turned out to be
    memories.
    """
    r = harness.run("cogs", steps=2, device=device, max_eval=4)
    assert set(r["arms"]) == {"softmax", "sgate"}, sorted(r["arms"])
    assert r["reference"]["source"] == "arXiv:2010.05465", r["reference"]
    assert r["arms"]["softmax"]["n_params"] == r["arms"]["sgate"]["n_params"], (
        "parameter counts must be identical, not merely similar: "
        f"{r['arms']['softmax']['n_params']} vs {r['arms']['sgate']['n_params']}")
    assert 0.0 <= r["arms"]["softmax"]["exact_match"] <= 1.0


def test_scoring_two_splits_reuses_one_trained_model(device):
    """COGS needs two numbers off one model, and training it twice is the bill.

    The in-distribution split is the resolution gate and the generalization
    split is the result, and they must come from the SAME model or the gate
    does not gate anything -- a control that cleared 0.20 on a differently
    seeded run says nothing about the run that produced the headline. Splitting
    `train_and_score` into `train_model` + `score` makes that structural, and
    halves a 46-minute bill on the way.
    """
    m = harness.train_model("softmax", "cogs", steps=2, device=device, seed=0,
                            bs=4, lr=1e-3)
    a = harness.score(m, "cogs", "test", device=device, max_eval=4, max_new=8)
    b = harness.score(m, "cogs", "gen", device=device, max_eval=4, max_new=8)
    assert a["eval_split"] == "test" and b["eval_split"] == "gen"
    assert a["n_eval"] == b["n_eval"] == 4
    assert m.vocab == harness.COGS_VOCAB, m.vocab
    # same model, so a rescore must be bit-identical, not merely close
    assert harness.score(m, "cogs", "gen", device=device, max_eval=4,
                         max_new=8)["n_solved"] == b["n_solved"]
