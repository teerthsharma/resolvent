"""W11 -- every claim in the README must name a test that exists.

LOOP.md completion condition 3: "Every claim in the model card has a test name and
a reproduction command." A document full of test names that do not resolve is
worse than a document with none: it reads as evidence and is not.

This is the same failure mode that has already bitten this project three times,
each time from a checker that was internally consistent and externally wrong:

  * a parity test that compared the gated path against this repo's own
    `stock_attention`, both wrong in the same way -- perplexity 89400.180 against
    1.667 before real-model parity caught it
  * a `sorry` detector that split on "--" only and fired on CEQ.lean's own
    sentence "No `sorry` anywhere", failing exactly when the code was correct
  * an eviction test that asserted bitwise invisibility in a window where it is
    provably false

So this file checks the README against pytest's actual collected node ids, not
against a hand-maintained list, and it calibrates itself first.

NO DEVICE PARAMETRIZATION, DELIBERATELY. Test collection has no device axis, and
a fake `@parametrize("device", ...)` would run the identical subprocess twice and
report two passes for one fact. Every numerical test in this repo does run on cpu
and cuda; this one has nothing to vary.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

# tests/dir/file.py::test_name   (optionally ::test_name only, on a following line)
NODE = re.compile(r"tests/[\w/]+\.py::(\w+)")
BARE = re.compile(r"`::(\w+)`")


def _collected() -> set[str]:
    """Every test function pytest can actually collect, by bare name.

    Collected rather than grepped: a test that exists in a file pytest cannot
    import is not a test, and a grep would happily report it as one.
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        "--no-header", "-p", "no:cacheprovider", "tests"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    names = set()
    for line in r.stdout.splitlines():
        if "::" in line:
            leaf = line.split("::")[-1].strip()
            names.add(leaf.split("[")[0])          # drop the device parametrization
    return names


@pytest.fixture(scope="module")
def collected():
    got = _collected()
    if not got:
        pytest.skip("pytest collected nothing; cannot validate claims")
    return got


def test_the_collector_actually_finds_known_tests(collected):
    """Calibrate the instrument before trusting it. A checker that returns an
    empty set would pass every claim below vacuously."""
    assert len(collected) > 50, len(collected)
    assert "test_signed_operator_reaches_negative_influence" in collected
    assert "test_eviction_is_not_retroactive" in collected


def test_readme_exists_and_leads_with_limits(collected):
    """House style: the limits come first, before any result.

    Checked positionally rather than by vibe -- the Limits heading must appear
    before the section that states the distinguishing property.
    """
    txt = README.read_text(encoding="utf-8")
    lim = txt.find("## Limits, first")
    # Heading renamed in iteration 10. It read "The one measured distinguishing
    # property" until FOREMAN measured SimA (arXiv 2206.08898, 2022) at
    # -3.929583e-01 on the same instrument, so it is neither the one nor
    # distinguishing. The positional invariant this test exists for is unchanged.
    prop = txt.find("## The signed influence property, and who already had it")
    assert lim != -1, "README has no Limits section"
    assert prop != -1, "README has no property section"
    assert lim < prop, "the limits do not come first"


def test_every_test_name_in_the_readme_resolves(collected):
    """The claim-to-evidence link. Any name that does not resolve is a claim with
    no evidence behind it wearing the costume of one."""
    txt = README.read_text(encoding="utf-8")
    named = set(NODE.findall(txt)) | set(BARE.findall(txt))
    assert named, "README cites no tests at all"
    missing = sorted(n for n in named if n not in collected)
    assert not missing, f"README names tests that do not exist: {missing}"


def test_the_reproduce_commands_are_real(collected):
    """A reproduction command that does not run is not a reproduction."""
    txt = README.read_text(encoding="utf-8")
    assert "python -m pytest tests/ -q" in txt
    assert "lake build CEQ" in txt
    assert (ROOT / "lean" / "lakefile.lean").exists()
    assert (ROOT / "lean" / "CEQ.lean").exists()


def test_deleted_requirements_are_stated_as_deleted(collected):
    """C2 says a failed falsifier DELETES its requirement. The README must say so
    for each, or the document quietly keeps six requirements while the repo has
    two."""
    txt = README.read_text(encoding="utf-8")
    for r in ("R1", "R2", "R3", "R4", "R6"):
        assert re.search(rf"\|\s*{r}\s*\|.*\*\*DELETED", txt), f"{r} not marked deleted"
    assert re.search(r"\|\s*R5\s*\|.*\*\*ALIVE", txt), "R5 not marked alive"


def test_the_failed_intervention_result_is_in_the_limits(collected):
    """The single most important negative result must not be buried. All three
    arms were worse than predicting the mean, and the README has to say it where
    a reader sees it."""
    txt = README.read_text(encoding="utf-8")
    lim = txt.find("## Limits, first")
    prop = txt.find("## The signed influence property, and who already had it")
    limits = txt[lim:prop]
    assert "worse than a constant predictor" in limits
    assert "2.6151" in limits and "4.2107" in limits


# ------------------------------------------------------------ the model card

CARD = ROOT / "MODEL_CARD.md"


def test_model_card_exists_and_leads_with_limits(collected):
    """Completion condition 4. Checked positionally, not by vibe."""
    txt = CARD.read_text(encoding="utf-8")
    lim = txt.find("## Limits, first")
    add = txt.find("## What it adds")
    assert lim != -1, "model card has no Limits section"
    assert add != -1, "model card never says what it adds"
    assert lim < add, "the limits do not come first"


def test_every_test_name_in_the_model_card_resolves(collected):
    """Completion condition 3. A cited test that does not exist is a claim
    wearing the costume of evidence."""
    txt = CARD.read_text(encoding="utf-8")
    named = set(NODE.findall(txt)) | set(BARE.findall(txt))
    assert named, "model card cites no tests at all"
    missing = sorted(n for n in named if n not in collected)
    assert not missing, f"model card names tests that do not exist: {missing}"


def test_the_model_card_states_what_parity_actually_cost(collected):
    """The parity claim must arrive with its price attached.

    This test previously pinned "correction term and not a replacement". That
    claim was superseded when the difference-of-softmaxes operator reached
    median 1.0334 on 5/5 seeds. Re-pointing the test rather than deleting it:
    the successor claim is stronger and therefore needs MORE guarding, not less.

    Three things must appear in the Limits section, above any result:
      * the parity numbers themselves, so the claim is checkable
      * the tuning asymmetry -- four operator knobs against softmax's one
      * that it is 3.3M parameters and NOT a scaling claim, since the widening
        measured on the original operator has not been re-measured here
    """
    txt = CARD.read_text(encoding="utf-8")
    lim = txt.find("## Limits, first")
    add = txt.find("## What it adds")
    limits = txt[lim:add]
    assert "1.0334" in limits, "the parity ratio is not in the limits"
    assert "5/5 seeds" in limits, "the seed count is not stated"
    assert "four knobs" in limits, "the tuning asymmetry is not stated"
    assert "UNTESTED" in limits, "the longer-budget caveat is missing"
    assert "WIDENS with training" in limits, (
        "the original operator's widening gap was dropped; it is still true of "
        "that operator and must survive the successor claim")


def test_the_model_card_declares_no_weights(collected):
    """It describes a module, not a checkpoint. Saying so prevents the most
    obvious misreading of an HF card."""
    txt = CARD.read_text(encoding="utf-8")
    assert "not a trained checkpoint" in txt
    assert "There are no weights here" in txt


def test_the_single_sample_perplexity_is_flagged_as_such(collected):
    """The -1.49% at alpha = 0.05 is one 99-token sample. A card that prints it
    without that qualifier is reporting a result it does not have."""
    txt = CARD.read_text(encoding="utf-8")
    assert "one 99-token sample and is not a\nresult" in txt or \
           "is not a result" in txt, "the single-sample caveat is missing"
