"""Every interval the card renders must name the estimator that produced it.

The card prints two columns both labelled `95% CI`, and they are NOT the same
estimator:

  * the Arms table's `marginal 95% CI` is a percentile bootstrap over EVAL
    POINTS within one seed, at `negation_scope.bootstrap_ci`'s own default
    `n_boot`, journalled per unit as `marg_lo`/`marg_hi`;
  * the Contrasts table's `95% CI` is a PAIRED percentile bootstrap over the
    five SEEDS at `N_BOOT` with `BOOT_SEED`.

The JSON already ships `n_boot`, `boot_seed` and a per-row `estimator` string,
and the contrasts prose names the family -- but a reader of the rendered tables
sees `95% CI` twice and has no way to tell that two different procedures
produced them. These tests ground on the RENDERED text, because that is the
artefact the reader gets.

    python -m pytest tests/mercury/test_r9_estimator_named.py -q
"""
from __future__ import annotations

import inspect
import re

import pytest

from scale import capability_table as CT
from scale import negation_scope as NS


@pytest.fixture(scope="module")
def card() -> str:
    return CT.render(CT.build(str(CT.JOURNAL)))


def _headers(card: str) -> list:
    return [ln for ln in card.splitlines() if ln.startswith("| ") and "CI" in ln]


def test_the_card_renders_more_than_one_interval_column(card):
    """Premise. With only one interval column there would be nothing to confuse."""
    assert len(_headers(card)) >= 2


def test_no_rendered_interval_header_is_a_bare_95_percent_ci(card):
    for h in _headers(card):
        for col in h.split("|"):
            col = col.strip()
            if "CI" not in col:
                continue
            assert re.search(r"bootstrap|B=\d+", col), (
                "interval column names no estimator: " + col)


def test_the_arms_header_names_its_own_bootstrap_and_its_size(card):
    n = inspect.signature(NS.bootstrap_ci).parameters["n_boot"].default
    hdr = next(h for h in _headers(card) if "marginal" in h)
    assert "eval points" in hdr, "the arms CI is over eval points, not seeds"
    assert "B={}".format(n) in hdr, "the arms CI does not state its B"


def test_the_contrasts_header_names_the_paired_estimator(card):
    hdr = next(h for h in _headers(card) if "delta" in h)
    assert "paired" in hdr
    assert "B={}".format(CT.N_BOOT) in hdr
    assert "seed={}".format(CT.BOOT_SEED) in hdr


def test_the_two_intervals_are_genuinely_different_estimators():
    """Adversarial pass: if they were the same, naming them apart would mislead."""
    n_arm = inspect.signature(NS.bootstrap_ci).parameters["n_boot"].default
    assert n_arm != CT.N_BOOT, (
        "the two intervals now share a B; re-check whether they are still "
        "different procedures before keeping them labelled apart")
