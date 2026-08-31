"""The eval harness: a number the day the model finishes training.

Requirements this file pins, from the round brief:

  * runs on CPU, needs no GPU
  * three columns, always: **the model**, a **matched-parameter softmax
    control**, and a **published reference point**
  * a self-contained scorer ships with it -- `openai/gsm8k` sees 1,076,823
    downloads/month against `tasksource/corr2cause` at 49, and the difference is
    whether you can score the thing without writing your own harness first

WHY SCAN AND NOT ARC-AGI. `test_arc_reality.py` measures ARC-AGI's resolving
power at 120 tasks: 5 solved tasks minimum before a gap is visible, a 95% upper
bound of 2.47% on an observed zero, and Fisher p = 1.0 between two arms that
both score zero. SCAN's length split has 3,920 test items and a published
from-scratch transformer baseline of 0.00 with a demonstrated ceiling of 1.00.
Same dollar of CPU, roughly thirty times the resolving power, and a reference
number that a parameter-matched model is actually comparable to.

THE COMPARISON IS PARAMETER-MATCHED, NOT FRONTIER. A 0.5B model against o3 or
Claude Opus is a category error and this harness does not offer that column. The
published reference is a from-scratch transformer of comparable size on the same
split, which is the only comparison that says anything about an operator.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench, harness

# Published split sizes, Lake & Baroni 2018 (arXiv:1711.00350), counted off the
# canonical files at github.com/brendenlake/SCAN. If a future refactor silently
# swaps in a resampled split, these numbers change and the published reference
# points stop being comparable.
PUBLISHED_SIZES = {
    ("addprim_jump", "train"): 14670, ("addprim_jump", "test"): 7706,
    ("length", "train"): 16990,       ("length", "test"): 3920,
    ("simple", "train"): 16728,       ("simple", "test"): 4182,
}


def test_the_scan_splits_load_at_their_published_sizes(device):
    """Comparability check, and the reason the data is vendored rather than
    regenerated from the grammar.

    A locally regenerated SCAN would be a different dataset wearing the same
    name, and every published baseline attached to it would become a guess.
    """
    for (split, which), n in PUBLISHED_SIZES.items():
        pairs = harness.load_scan(split, which)
        assert len(pairs) == n, (split, which, len(pairs), n)


def test_the_scorer_is_exact_match_and_one_wrong_token_scores_zero(device):
    """CALIBRATION of the scorer itself, before it is used on anything.

    SCAN is scored by exact sequence match. A scorer that gave partial credit
    would report a healthy-looking number for a model emitting the right actions
    in the wrong order, which is precisely the failure SCAN exists to detect.
    """
    gold = ["I_JUMP", "I_WALK", "I_WALK"]
    assert harness.exact_match(["I_JUMP", "I_WALK", "I_WALK"], gold)
    assert not harness.exact_match(["I_JUMP", "I_WALK"], gold)          # short
    assert not harness.exact_match(["I_WALK", "I_JUMP", "I_WALK"], gold)  # order
    assert not harness.exact_match(["I_JUMP", "I_WALK", "I_RUN"], gold)   # token
    assert not harness.exact_match([], gold)


def test_every_published_reference_carries_a_citation(device):
    """A reference number with no source is a memory, and memories are what
    this project already struck once.

    Each entry must name a paper and say what model produced it, so the
    comparison column can be audited without re-reading this file's git history.
    """
    assert harness.REFERENCES, "no published reference points registered"
    for key, ref in harness.REFERENCES.items():
        assert ref["model"], key
        if ref["score"] is None:
            # A NULL entry records a search that found nothing. It is the
            # opposite of an uncited number: it says out loud that no published
            # comparison exists, so a later reader cannot assume one was found
            # and omitted. It must say so in its note and must not fake a cite.
            assert ref["source"] == "none", (key, ref)
            assert "not found" in ref["note"], (key, ref)
            continue
        assert ref["source"].startswith(("arXiv:", "aclanthology")), (key, ref)
        assert 0.0 <= ref["score"] <= 1.0, (key, ref["score"])


def test_the_table_has_a_model_a_matched_control_and_a_published_reference(device):
    """The three columns, enforced structurally.

    A harness that reports only the model's own number invites the reader to
    supply their own baseline from memory. All three arrive together or the
    table is not built.
    """
    table = harness.run("length", steps=2, device=device, seed=0, max_eval=16)
    assert "sgate" in table["arms"], (
        "the default challenger arm is the campaign's shipped operator, "
        "A = rho*(softmax(w) - lam*softmax(-w))/(1+lam), not the L1 form it "
        "replaced")
    assert "softmax" in table["arms"]
    assert table["reference"]["source"], "no published reference in the table"
    for arm in table["arms"].values():
        assert 0.0 <= arm["exact_match"] <= 1.0
        assert arm["n_eval"] == 16


def test_both_arms_have_identical_parameter_counts(device):
    """Not "within 10%". IDENTICAL.

    The operator carries no parameters of its own, so any difference means the
    two models are not the same model and the comparison is about capacity.
    """
    table = harness.run("length", steps=2, device=device, seed=0, max_eval=8)
    counts = {k: v["n_params"] for k, v in table["arms"].items()}
    assert len(set(counts.values())) == 1, counts


def test_scan_length_resolves_a_gap_that_arc_agi_cannot(device):
    """The whole benchmark-choice argument, as one arithmetic comparison.

    Both benchmarks are exact-match. They differ only in how many items they
    score, and that difference is what decides whether a run can say anything.
    """
    n_scan = PUBLISHED_SIZES[("length", "test")]
    n_arc = 120

    k_scan = bench.min_successes_for_separation(n_scan, k_baseline=0, alpha=0.05)
    k_arc = bench.min_successes_for_separation(n_arc, k_baseline=0, alpha=0.05)

    rate_scan, rate_arc = k_scan / n_scan, k_arc / n_arc
    assert rate_scan < rate_arc / 20, (rate_scan, rate_arc)
    # and an observed zero is a far tighter statement on SCAN
    assert (bench.zero_success_upper_bound(n_scan)
            < bench.zero_success_upper_bound(n_arc) / 20)


@pytest.mark.slow
def test_the_harness_produces_a_number_on_cpu_with_no_gpu(device):
    """End to end, CPU only, no accelerator touched.

    This is the acceptance test for "a number the day the model finishes
    training": if this passes today at two steps, it passes at 20,000 steps
    with nothing changed but the argument.
    """
    table = harness.run("length", steps=20, device=torch.device("cpu"),
                        seed=0, max_eval=32)
    assert table["device"] == "cpu"
    for arm in table["arms"].values():
        assert isinstance(arm["exact_match"], float)


# --------------------------------------------------------------------- COGS

COGS_SIZES = {"train": 24155, "dev": 3000, "test": 3000, "gen": 21000}


def test_cogs_loads_at_its_published_sizes(device):
    """COGS is the benchmark this round actually recommends, ahead of SCAN.

    Kim & Linzen 2020 (arXiv:2010.05465). The generalization set is 21,000
    items and the published from-scratch transformer scores 0.35 +/- 0.06 --
    NONZERO and nowhere near either floor or ceiling, which is what SCAN's
    length split is not and what ARC-AGI is very much not.
    """
    for which, n in COGS_SIZES.items():
        assert len(harness.load_cogs(which)) == n, which


def test_the_cogs_reference_carries_the_altered_attention_precedent(device):
    """The reason COGS beats SCAN as the first choice -- restated, smaller.

    This project cited "Edge Transformer 0.874 against a Universal Transformer
    control at 0.784" as proof that COGS moves when the attention changes. Its
    own source says otherwise: arXiv:2112.00578 Table 4's top section is the
    GRAPH-PREDICTION version of COGS, and the 0.784 comparison row is quoted
    from another paper without a standard deviation. Neither number is the
    sequence-generation task scored by this harness.

    The precedent that survives is smaller and real. Csordas et al. 2021 Table
    3 runs four attention variants through one fixed training recipe on the
    sequence-generation task: 0.80 / 0.81 / 0.78 / 0.77. Attention moves this
    benchmark about four points, not fifty-two. That is still more than
    TinyStories val loss can show about a signed operator, which is why COGS is
    still the right benchmark -- but the claim is now the size of its evidence.
    """
    base = harness.REFERENCES[("cogs", "transformer_from_scratch")]
    alt = harness.REFERENCES[("cogs", "altered_attention")]
    assert base["score"] == pytest.approx(0.35, abs=0.01)
    assert alt["score"] == pytest.approx(0.81, abs=0.01)
    assert alt["source"] == "arXiv:2108.12284"
    struck = harness.REFERENCES[("cogs", "edge_transformer_graph_prediction")]
    assert "graph" in struck["note"].lower(), "the retraction must stay visible"
    assert harness.REFERENCES[("cogs", "no_decoder_only_reference")]["score"] is None


def test_cogs_no_longer_refuses_and_no_longer_truncates(device):
    """The blocker this test used to guard is gone; what it guarded is not.

    The refusal was correct while `lm.TinyLM` had a fixed 256-way head: a
    truncated COGS reports a number that looks like a COGS score and is not
    one. The head is sized to the task vocabulary now, so the refusal is
    replaced by the property it was standing in for -- every training sequence
    fits, and the vocabulary covers every token in all three splits.
    """
    vocab = harness.build_vocab("cogs")
    assert len(vocab) == harness.COGS_VOCAB == 874, len(vocab)
    longest = max(len(harness.encode(p, vocab))
                  for p in harness.load_cogs("train"))
    assert longest <= harness.COGS_SEQ, (longest, harness.COGS_SEQ)


def test_no_scan_split_is_silently_truncated_by_the_context_window(device):
    """The guard that makes the SCAN numbers trustworthy.

    The length split trains on short sequences and tests on long ones. If the
    context window clipped the test set, the benchmark would report a low score
    for a reason that has nothing to do with compositional generalization, and
    the arm comparison would be measuring the truncation.

    Measured: every SCAN sequence in every split fits, longest 59 against 64.
    """
    for split in ("simple", "addprim_jump", "length"):
        vocab = harness.build_vocab(split)
        for which in ("train", "test"):
            longest = max(len(harness.encode(p, vocab))
                          for p in harness.load_scan(split, which))
            assert longest <= harness.SEQ, (split, which, longest, harness.SEQ)
