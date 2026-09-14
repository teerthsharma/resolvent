"""RED-first: what bounds reading the COMMITTOR off the shipped 7-column read.

WHAT THIS FILE BINDS. `ceqjepa/pi_jepa.py:195` fixes `D_LATENT = 9` and
`pi_assign`'s frozen vector refuses two of those nine, so the read emits 7
columns (`ceqjepa/pi_jepa.py:329 read_per_coordinate`, `cols`). The question is
whether that 7 is what bounds the architecture, or whether the bound sits in the
encoder that feeds it or in the probe that reads it.

The target is the one the north star names -- `CEQ_V20_R15_CONTRACT.md:42-45`,
"predicting the NEXT STATE toward equilibrium, not the next token" -- taken here
as the transition-path committor of `ceqjepa/chess_steps.py:458 _committor`:
q_loss(s), the probability that a uniform-random-play walk from s reaches
checkmate before a draw. It is EXACT (residual 9.645e-13 over 1,085 iterations,
`python -m ceqjepa.chess_steps`), it is encoder-independent, and it is a SCALAR,
so none of the rank arithmetic that applies to a 64-way move label applies here.

WHY EVERY TEST BELOW WAS RED WHEN IT WAS WRITTEN. Each asserts the consequence of
a hypothesis about where the bound sits. The hypotheses are the ones a reader of
the T4 run would form; the point of the file is that the code refutes them, in
the order they are stated, and the verbatim failures are recorded in
`RED_VERBATIM` beside each test.

THE ARTIFACTS. These tests score a bed that is built, not shipped: 60,000
uniform-random-play walks of 20 plies in K+Q vs K, split by the PLY-15 POSITION
index so no position carries its label into both sides. The builder is
`scratchpad/ceiling/kqk_bed.py` and the reader `scratchpad/ceiling/kqk_ceiling.py`.
If the artifacts are absent the tests FAIL and say so -- L-SURFACE: a check that
cannot run is RED, never skipped.
"""
import json
import os
import pathlib

import numpy as np
import pytest

ART = pathlib.Path(os.environ.get(
    "CEQ_CEILING_DIR",
    pathlib.Path(os.environ.get("TEMP", "/tmp")) / "claude" /
    "C--Users-seal-Desktop-New-folder--32-" /
    "bb16374f-0874-425e-b13e-7a1d3ce67564" / "scratchpad" / "ceiling"))

TAG = "kqk"


def _need(p):
    if not p.exists():
        raise AssertionError(
            "the artifact %s is absent, so this check cannot run; under L-SURFACE "
            "a check that cannot run is RED. Rebuild it with\n"
            "  python scratchpad/ceiling/kqk_bed.py --walks 60000 --hidden 1024 "
            "--steps 1500 --batch 256 --arms untrained,frozen_random,trained\n"
            "  python scratchpad/ceiling/kqk_ceiling.py kqk" % p)
    return p


@pytest.fixture(scope="module")
def labels():
    return np.load(_need(ART / ("%s_labels.npz" % TAG)))


@pytest.fixture(scope="module")
def table():
    d = json.loads(_need(ART / ("%s_ceiling.json" % TAG)).read_text())
    return d


def _row(table, arm, features, reader):
    hits = [r for r in table["rows"]
            if r.get("arm") == arm and r.get("features") == features
            and r.get("reader") == reader]
    if not hits:
        raise AssertionError("no row for arm=%r features=%r reader=%r in the "
                             "ceiling table; the sweep did not produce it"
                             % (arm, features, reader))
    if "failed" in hits[0]:
        raise AssertionError("the row arm=%r features=%r reader=%r FAILED to fit: "
                             "%s" % (arm, features, reader, hits[0]["failed"]))
    return hits[0]


# ---------------------------------------------------------------------------
# 1. THE ARITHMETIC HYPOTHESIS, RE-AIMED AT A SCALAR TARGET
# ---------------------------------------------------------------------------

RED_VERBATIM = """All seven RED, verbatim, at commit b4c6620 on WIN-16QAL06O9GB,
python 3.11.9, torch 2.14.0+cpu, numpy 2.4.6, sklearn 1.9.0, against the bed
`kqk_bed.py --walks 60000 --hidden 1024 --steps 1500 --batch 256`:

  CEQ_CEILING_DIR=<scratchpad>/ceiling python -m pytest \\
      tests/curvature/test_committor_read_ceiling.py -q      # exit 1, 7 failed

E AssertionError: a one-column read reached mse 3.7e-05 against the marginal's
E   0.000147: one real coordinate carries the committor exactly, so 7 columns are
E   not arithmetically narrow and no negative result on this bed is a statement
E   about the width
E   assert 3.69879468107272e-05 > (0.5 * 0.0001469879077232036)

E AssertionError: opening the two refused coordinates moved committor skill from
E   -0.0397 to -0.0334: the refusals are not where the information went, so the
E   read's column count is not the bound
E   assert -0.033414955972912 > (-0.03970984069534089 + 0.01)

E AssertionError: the 7-column read reaches skill -0.0397 where the same reader on
E   the same forward pass reaches 0.2165 one layer upstream: the information is
E   present at 1024 and absent at 7, so the projection and not the reader is the
E   bound
E   assert -0.03970984069534089 >= (0.9 * 0.2165420794003985)

E AssertionError: the trained 7-column read scores mse 1.528248e-04 against the
E   marginal's, skill -0.0397, gain 95% CI [-9.595e-06, -2.157e-06]: the lower
E   bound does not clear zero, so on this bed the trained read does not beat a
E   predictor that learned nothing
E   assert -9.594570957943828e-06 > 0.0

E AssertionError: the trained encoder reaches committor skill -0.0397 and the
E   FROZEN RANDOM encoder reaches -0.0061 off the identical read: the JEPA fit is
E   not what puts the committor into the representation
E   assert -0.03970984069534089 > -0.006149817482146602

E AssertionError: the trained 7-column read reaches committor skill -0.0397 where
E   the SAME reader on six raw piece coordinates reaches 0.8851: the architecture
E   is behind a stock tree on the board itself by 0.9248 of the reducible variance
E   assert -0.03970984069534089 > 0.8851208817308647

E AssertionError: the encoder's widest layer reaches committor skill 0.3697
E   against six raw piece coordinates at 0.8851: even before the 9-dim projection,
E   and with the reader held identical, the architecture's representation is
E   behind the board
E   assert 0.36967990254421235 > 0.8851208817308647
"""


def test_one_column_is_too_narrow_to_carry_the_committor(labels):
    """The width hypothesis in its sharpest form, and it is FALSE.

    VARIES: the number of feature columns handed to the probe -- exactly one.
    PINS:   the probe family (the shipped `nn.Linear` head with its 64 outputs
            replaced by 1, soft binary cross-entropy, Adam, the shipped
            `probe_lr`), the rows, the label, the seed.

    The single column is the exact committor itself, which is a legal feature
    precisely because the question is arithmetic: CAN 1 real number carry q, not
    DOES the encoder produce it. If one can, seven can, and the width is not the
    bound.
    """
    import torch
    qtr = labels["q_train"].astype(np.float64)
    qte = labels["q_test"].astype(np.float64)
    x_tr, x_te = qtr[:, None], qte[:, None]
    mu, sd = x_tr.mean(0), np.maximum(x_tr.std(0), 1e-8)
    a = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    b = torch.tensor((x_te - mu) / sd, dtype=torch.float32)
    y = torch.tensor(qtr, dtype=torch.float32)
    torch.manual_seed(5501)
    head = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(4000):
        loss = lossfn(head(a).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(head(b).squeeze(-1)).numpy().astype(np.float64)
    mse = float(np.mean((p - qte) ** 2))
    marginal = float(np.mean((qtr.mean() - qte) ** 2))
    assert mse > 0.5 * marginal, (
        "a one-column read reached mse %.3g against the marginal's %.3g: one real "
        "coordinate carries the committor exactly, so 7 columns are not "
        "arithmetically narrow and no negative result on this bed is a statement "
        "about the width" % (mse, marginal))


# ---------------------------------------------------------------------------
# 2. THE READ-WIDTH HYPOTHESIS: 7 OF 9, AND THE TWO THE RULE REFUSED
# ---------------------------------------------------------------------------

def test_opening_the_two_refused_coordinates_buys_committor_skill(table):
    """VARIES: the read's column count, 7 -> 9, by setting beta = 1.0 at the two
    coordinates `pi_assign` refuses (`half_walk`, `osc_shape`).
    PINS:   the encoder weights, the predictor weights, the beta at the other
            seven coordinates, the reader, the rows, the label, the seed.
    """
    a = _row(table, "trained", "read7", "gbt")
    b = _row(table, "trained", "read9", "gbt")
    assert b["skill"] > a["skill"] + 0.01, (
        "opening the two refused coordinates moved committor skill from %.4f to "
        "%.4f: the refusals are not where the information went, so the read's "
        "column count is not the bound" % (a["skill"], b["skill"]))


# ---------------------------------------------------------------------------
# 3. THE ENCODER HYPOTHESIS: WHAT THE 9-DIM LATENT THROWS AWAY
# ---------------------------------------------------------------------------

def test_the_read_reaches_the_encoders_own_hidden_layer(table):
    """VARIES: where the reader taps the SAME forward pass -- the 7-column read,
            versus the encoder's 1024-wide penultimate activation one layer
            upstream of the 9-dim projection.
    PINS:   the encoder weights, the reader family, the rows, the label, the seed.
    """
    r = _row(table, "trained", "read7", "gbt")
    h = _row(table, "trained", "hid", "gbt")
    assert r["skill"] >= 0.9 * h["skill"], (
        "the 7-column read reaches skill %.4f where the same reader on the same "
        "forward pass reaches %.4f one layer upstream: the information is present "
        "at 1024 and absent at 7, so the projection and not the reader is the bound"
        % (r["skill"], h["skill"]))


# ---------------------------------------------------------------------------
# 4. THE ONE THE NORTH STAR ASKS: DOES TRAINING BEAT A PREDICTOR THAT LEARNED NOTHING
# ---------------------------------------------------------------------------

def test_the_trained_read_beats_the_marginal_committor(table):
    """VARIES: the encoder arm -- `trained` against the constant `marginal`.
    PINS:   the read width, the reader, the rows, the label, the seed, the split.

    The bar is the marginal committor rate on the training split: a predictor that
    learned nothing. The interval is the cluster bootstrap over distinct test
    POSITIONS reported beside the point estimate.
    """
    r = _row(table, "trained", "read7", "gbt")
    assert r["ci95"][0] > 0.0, (
        "the trained 7-column read scores mse %.6e against the marginal's, skill "
        "%.4f, gain 95%% CI [%+.3e, %+.3e]: the lower bound does not clear zero, "
        "so on this bed the trained read does not beat a predictor that learned "
        "nothing" % (r["mse_vs_q"], r["skill"], r["ci95"][0], r["ci95"][1]))


def test_training_beats_the_frozen_random_encoder_on_the_committor(table):
    """VARIES: `requires_grad` on the online encoder -- the one line that separates
            `trained` from `frozen_random` in `pi_jepa_chess.train_arm`.
    PINS:   initialisation seed, batch sequence, step count, optimiser settings,
            the mask, the read width, the reader, the rows, the label.
    """
    t = _row(table, "trained", "read7", "gbt")
    f = _row(table, "frozen_random", "read7", "gbt")
    assert t["skill"] > f["skill"], (
        "the trained encoder reaches committor skill %.4f and the FROZEN RANDOM "
        "encoder reaches %.4f off the identical read: the JEPA fit is not what "
        "puts the committor into the representation" % (t["skill"], f["skill"]))


# ---------------------------------------------------------------------------
# 6. THE BAR THE NORTH STAR ACTUALLY SETS: "capable on ground they cannot occupy"
# ---------------------------------------------------------------------------

def test_the_architecture_beats_six_raw_board_coordinates(table):
    """VARIES: the representation the committor is read off -- the architecture's
            own 7-column read against `ceqjepa/chess_steps.py:549 embed`, six raw
            board coordinates (file and rank of each of the three pieces, scaled
            to [0,1]) chosen in that module precisely so that nothing about the
            label leaks into them.
    PINS:   the reader (HistGradientBoostingRegressor, max_iter 400, seed 5501),
            the rows, the split, the label, the bootstrap.

    This is the comparison the north star's third clause asks for, at the
    smallest possible scale: a reader with no encoder at all, on coordinates a
    beginner would write down.
    """
    r = _row(table, "trained", "read7", "gbt")
    e = _row(table, "shared", "embed6", "gbt")
    assert r["skill"] > e["skill"], (
        "the trained 7-column read reaches committor skill %.4f where the SAME "
        "reader on six raw piece coordinates reaches %.4f: the architecture is "
        "behind a stock tree on the board itself by %.4f of the reducible variance"
        % (r["skill"], e["skill"], e["skill"] - r["skill"]))


def test_the_encoders_widest_layer_beats_six_raw_board_coordinates(table):
    """The same bar, taken at the architecture's most favourable tap.

    VARIES: the representation -- the encoder's 1024-wide penultimate activation
            against the same six raw coordinates.
    PINS:   the reader, the rows, the split, the label, the bootstrap.
    """
    h = _row(table, "untrained", "hid", "gbt")
    e = _row(table, "shared", "embed6", "gbt")
    assert h["skill"] > e["skill"], (
        "the encoder's widest layer reaches committor skill %.4f against six raw "
        "piece coordinates at %.4f: even before the 9-dim projection, and with "
        "the reader held identical, the architecture's representation is behind "
        "the board" % (h["skill"], e["skill"]))
