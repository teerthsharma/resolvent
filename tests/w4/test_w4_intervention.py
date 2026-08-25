"""W4 -- intervention generalization. The requirement the module exists for.

THE KILL CONDITION. If the signed arm does not separate from APPNP under
intervention, tier 3 buys nothing measurable and the module is deleted.

WHY EXECUTED CODE AND NOT PHYSICS. Physics pretraining was measured
indistinguishable from a sham prior with its action column permuted: 0.3667 vs
0.3654, against 0.3654 for matched-lambda scratch. Executed code has a free
ground-truth oracle -- the interpreter -- and an interventional corpus built from
it recovers `dy/da = -1` where the confounded observational corpus gives +0.63, a
sign flip.

THE THREE ARMS, which are the three measured tiers:

  tier 1  softmax attention   pairwise similarity. A third token cannot move the
                              RATIO of two weights in a row.
  tier 2  APPNP               non-negative multi-hop. The ratio moves, but the
                              minimum influence-Jacobian entry is EXACTLY 0.
  tier 3  ceq signed path sum negative influence reachable, measured at -0.900.

The control is APPNP, NOT vanilla attention. A linear resolvent stage was
measured reproducing APPNP to 2.22e-16, so beating vanilla attention proves
nothing about this module.

WHAT THE CORPUS PUTS IN THE WAY. Every program contains a negation gate --
`if flag: acc = -acc` -- so one token flips the sign of the entire outcome. That
is the operation the Kleene star of a non-negative matrix provably cannot
express in any ordered semiring. If tier 3 is worth anything, this is where it
shows, and if it does not show here it will not show anywhere.

THE SPLIT. Every (op, flag) cell except ('-', 1) is trained; that cell alone is
tested. Literals 1..9 appear on both sides, so no arm is asked to read an
embedding row it never updated. The held-out cell requires composing a
subtraction with a negation -- two sign flips seen separately and never
together.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ceq import arms, corpus

N_TRAIN, N_TEST = 384, 128
STEPS = 400
SEED = 0


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def data():
    return corpus.build(n_train=N_TRAIN, n_test=N_TEST, seed=SEED)


# ------------------------------------------------------- the corpus is honest

def test_the_oracle_is_the_interpreter_not_a_formula(data):
    """Every outcome must be a real execution. A closed-form label would make
    this a curve-fitting benchmark with a causal story attached."""
    rows = data["train"][:32]
    for r in rows:
        assert r["y"] == corpus.execute(r["src"]), f"label disagrees with CPython: {r['src']}"


def test_exec_agrees_with_a_subprocess_interpreter(data):
    """`exec` is used for speed. Spot-check that it agrees with a genuinely
    isolated CPython subprocess, so the speed trade did not change the oracle."""
    for r in data["train"][:8]:
        assert corpus.execute_subprocess(r["src"]) == r["y"], r["src"]


def test_intervention_changes_the_outcome_and_the_tokens_together(data):
    """A do() here is an edit to the token sequence whose outcome is re-measured
    by the interpreter. If the pairs were observational the corpus would be
    confounded and the whole comparison void."""
    changed = sum(int(r["y"] != r["y_base"]) for r in data["train"])
    assert changed > 0.5 * len(data["train"]), changed


def test_the_negation_gate_actually_inverts_the_outcome(device, data):
    """The property tier 1 and tier 2 provably cannot express.

    Checked directly rather than by a linear regression on the flag. A linear
    main effect is the WRONG instrument here: the gate flips a sign, so its
    marginal coefficient averages to roughly zero over a corpus symmetric in
    `op`, and an earlier version of this test failed for exactly that reason
    while the corpus was behaving correctly.
    """
    for r in data["train"][:32]:
        flipped = corpus.source(r["n"], r["op"], r["c"], 1 - r["flag"])
        assert corpus.execute(flipped) == -r["y"], r["src"]


def test_the_held_out_cell_is_a_composition_of_seen_tokens(data):
    """Every token id in the test split must appear in training. Otherwise the
    comparison measures embedding coverage, which is what the first version of
    this corpus accidentally measured."""
    seen = {t for r in data["train"] for t in r["tokens"]}
    unseen = {t for r in data["test"] for t in r["tokens"]} - seen
    assert not unseen, f"test uses token ids never trained: {sorted(unseen)}"


def test_the_held_out_composition_never_appears_in_training(data):
    """OOD by composition. Train sees both sign flips separately, never
    together."""
    ho = data["held_out"]
    assert all((r["op"], r["flag"]) != ho for r in data["train"])
    assert all((r["op"], r["flag"]) == ho for r in data["test"])


# ------------------------------------------------------------ the arms are fair

def test_the_three_arms_have_matched_parameter_counts(device, data):
    """An arm that wins with more parameters has not won."""
    counts = {n: arms.build(n, data["vocab"], device=device).n_params()
              for n in ("attention", "appnp", "signed")}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi <= 1.10 * lo, counts


def test_appnp_control_is_actually_nonnegative(device, data):
    """The control has to really be tier 2. If its operator went signed, the
    comparison below measures nothing."""
    m = arms.build("appnp", data["vocab"], device=device)
    a = m.operator(torch.randint(0, data["vocab"], (1, corpus.SEQ_LEN), device=device))
    assert float(a.min()) >= 0.0, float(a.min())


def test_signed_arm_operator_is_actually_signed(device, data):
    m = arms.build("signed", data["vocab"], device=device)
    a = m.operator(torch.randint(0, data["vocab"], (1, corpus.SEQ_LEN), device=device))
    assert float(a.min()) < 0.0, float(a.min())


# --------------------------------------------------------- W4's own falsifier

def test_signed_arm_separates_from_appnp_under_intervention(device, data):
    """RED first. THE KILL CONDITION for the module.

    Report OOD NRMSE on the held-out composition against the APPNP control, on
    identical data, matched parameters, matched training budget. If the signed
    arm does not beat APPNP, tier 3 buys nothing measurable.

    PASSES. Median over 5 seeds: attention 5.8198, appnp 4.2107, signed 2.6151.
    Signed beats appnp in 4/5 seeds, median ratio 0.6770. The ordering is the
    predicted tier ladder and it reproduces.

    Read this together with `test_signed_arm_generalizes_rather_than_merely_
    degrading_less`, which is RED. Relative separation is real; generalization
    is not.
    """
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED)
    signed, appnp = got["signed"]["ood_nrmse"], got["appnp"]["ood_nrmse"]
    assert signed < 0.8 * appnp, (
        f"signed {signed:.4f} vs appnp {appnp:.4f} -- no separation under "
        f"intervention; tier 3 buys nothing and the module is deleted. {got}")


def test_every_arm_beats_predicting_the_mean(device, data):
    """Deadness guard. An arm at NRMSE 1.0 learned nothing, and a comparison
    between two arms that both learned nothing is not a result."""
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED)
    for name, g in got.items():
        assert g["train_nrmse"] < 0.9, (name, g)


def test_signed_arm_generalizes_rather_than_merely_degrading_less(device, data):
    """RED. The absolute form of R2, and the one the requirement actually asks
    for: "must generalize under intervention where a similarity baseline fails".

    NRMSE 1.0 is the predict-the-mean baseline. Measured OOD medians over 5
    seeds: attention 5.8198, appnp 4.2107, signed 2.6151. All three are ABOVE
    1.0, so every arm is worse than a constant predictor on the held-out
    composition. The signed arm fails 1.6x less badly than the control; it does
    not generalize.

    The relative test above was the weaker falsifier and it was mine to get
    wrong. This is the one the requirement specifies, it is RED, and the
    ordering result must not be reported without it.
    """
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED)
    assert got["signed"]["ood_nrmse"] < 1.0, (
        f"signed OOD NRMSE {got['signed']['ood_nrmse']:.4f} >= 1.0: worse than "
        f"predicting the mean. Tier 3 degrades more gracefully than tier 2 but "
        f"does not generalize under intervention. {got}")


def test_the_tier_ordering_is_the_predicted_one(device, data):
    """What DID survive: attention > appnp > signed on OOD error, the tier
    ladder in the predicted direction. Bound so the ordering claim is not
    carried by the deleted generalization claim."""
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED)
    assert got["attention"]["ood_nrmse"] > got["appnp"]["ood_nrmse"] > got["signed"]["ood_nrmse"], got
