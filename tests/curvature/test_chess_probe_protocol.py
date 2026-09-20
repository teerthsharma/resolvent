"""RED-first: what the staged chess probe protocol measures that it does not name.

Target: the Kaggle entry script `kaggle_pkg/pi_jepa_chess.py`. Point
CEQ_KAGGLE_PKG at the directory holding it (default: <repo>/kaggle_pkg).

Each test names one way the run can emit a number that reads like a result:

  1. the evaluation label ply is a different ply at every evaluation length,
     so "length generalisation" and "game phase" move together
  2. the probe is fit on a ply-uniform window sampler and scored at a single
     fixed ply, so even the 1x number is off the probe's own fit distribution
  3. the summary's declared headline, acc_exact, is a JOINT metric and the
     evaluation dict carries no joint control to compare it against
  4. a different set of holdout games is scored at every length
  5. the measured train/holdout position overlap is reported as one aggregate
     and never used to split the 1x accuracy it contaminates
  6. a rank-7 linear head on a 7-column read emits a near-constant prediction

REPRICED BY AUDIT, 2026-09-14, at commit 391a2d0 on WIN-16QAL06O9GB. The eight
failures below are honest and reproduce; several numbers stated in the report
that shipped them were not produced by any test here and are struck. Struck:
every figure from a hidden=1024 configuration (this file runs the shipped
width), the "32x width buys 0.005 nats" delta, the populations 615 / 153 / 769,
the majority_exact 0.0260 comparison, and the repricing of the train/holdout
overlap from 42.7% to 2.86%. None of those has a producer in this file. What
survives, and is produced here: the probe cross-entropy against a uniform
8.3178, and the read's rescaling across the evaluation lengths.

The subject of this file, kaggle_pkg/pi_jepa_chess.py, filters no material and
loads no distance-to-mate label -- verified by grep at 391a2d0 -- so nothing
measured here is a statement about a K+Q-vs-K endgame target.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_PKG = str(REPO / "kaggle_pkg")
KPKG = pathlib.Path(os.environ.get("CEQ_KAGGLE_PKG", _DEFAULT_PKG))
PGN = pathlib.Path(os.environ.get("CEQ_PGN", str(KPKG / "_smoke_prefix.pgn")))

# L-SURFACE. A check that cannot run is a FAILED check, never a skipped one.
# A module-level raise here reads as a collection error and takes the whole
# session down with it; a module-level skip reported `1 skipped`, exit status
# 5, which reads as "nothing to run" rather than "eight REDs did not execute".
# So collection never fails on the kernel's absence -- only the fixture below
# does, the same pytest.fail idiom the `pipeline` fixture already uses -- and
# every one of the eight tests goes red on its own setup.
_KERNEL_PRESENT = (KPKG / "pi_jepa_chess.py").exists()
_KERNEL_MSG = (
    "the subject of this file is absent: no pi_jepa_chess.py under %s. "
    "Set CEQ_KAGGLE_PKG to the directory holding the staged kernel, or "
    "track the kernel. These eight checks are RED by intent; they must "
    "never report as skipped, because a skip is indistinguishable from a "
    "suite that has nothing to say." % KPKG)

if _KERNEL_PRESENT:
    sys.path.insert(0, str(KPKG))
    import pi_jepa_chess as K  # noqa: E402
else:
    K = None

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _require_kernel():
    """Autouse fixtures run first within their scope, ahead of `pipeline`, so
    this is what every test in the module hits when the kernel is absent."""
    if not _KERNEL_PRESENT:
        pytest.fail(_KERNEL_MSG)


S_LEN, HORIZON = 16, 4
LENGTHS = [16, 64, 128]


def _games(lens, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i, m in enumerate(lens):
        out.append(dict(
            key="g%d" % i,
            codes=rng.integers(0, 13, size=(m, 64)).astype(np.uint8),
            stm=(np.arange(m) % 2).astype(np.uint8),
            frm=rng.integers(0, 64, size=m).astype(np.int16),
            dst=rng.integers(0, 64, size=m).astype(np.int16)))
    return out


def _fake_store(n_games=400, plies=200):
    return K.Store(_games([plies] * n_games, 0), torch.device("cpu"), torch)


# ------------------------------------------------------------------ 1 and 4

def test_eval_label_ply_is_the_same_at_every_length():
    """The scored label must sit at one ply, or length is confounded with phase."""
    st = _fake_store()
    plies = {}
    for L in LENGTHS:
        starts, n = st.head_starts(L + HORIZON, None)
        off = (starts + L - 1) - st.base[: starts.numel()]
        plies[L] = sorted(set(off.tolist()))
    got = sorted({p for v in plies.values() for p in v})
    assert len(got) == 1, (
        "the evaluation scores a different ply at every length: %r -- a change "
        "in accuracy from 1x to 8x is a change in game phase as much as a "
        "change in context length" % (plies,))


def test_the_same_holdout_games_are_scored_at_every_length():
    """Length must vary alone; the scored population must not vary with it."""
    st = K.Store(_games([20, 70, 140, 300] * 60, 1), torch.device("cpu"), torch)
    scored = {}
    for L in LENGTHS:
        starts, n = st.head_starts(L + HORIZON, 4096)
        scored[L] = frozenset(starts.tolist())
    sizes = {L: len(s) for L, s in scored.items()}
    assert len(set(scored.values())) == 1, (
        "a different holdout population is scored at each length: %r -- the 8x "
        "number is measured on long games only, a selected subpopulation, not "
        "the same games read further" % (sizes,))


# ------------------------------------------------------------------ 2

def test_probe_training_plies_cover_the_evaluation_ply():
    """The probe is fit where it is scored, or the 1x number is already OOD."""
    st = _fake_store()
    gen = torch.Generator().manual_seed(5501 + 101)
    starts = st.sample_starts(20000, S_LEN, gen)
    lab = starts + S_LEN - 1
    gi = torch.bucketize(lab, st.base, right=True) - 1
    off = (lab - st.base[gi]).tolist()
    frac = {L: sum(1 for o in off if o == L - 1) / len(off) for L in LENGTHS}
    assert min(frac.values()) >= 0.5, (
        "the probe's fit distribution over label plies is uniform over the "
        "game; the evaluation scores a point mass at ply L-1. fraction of "
        "probe training labels landing on each evaluated ply: %r" % (frac,))


# ------------------------------------------------------------------ 3, 5, 6

@pytest.fixture(scope="module")
def pipeline():
    """The real pipeline at smoke size: corpus, split, model, probe, evaluate."""
    if not PGN.exists():
        pytest.fail("no PGN at %s -- these checks are RED by intent and a "
                    "missing corpus must report red, not skipped" % PGN)
    pj, pa, kdata = K.load_repo_modules(REPO)
    cfg = K.Config(repo_dir=str(REPO), pgn=str(PGN),
                   out_dir=str(KPKG / "_chase_out"))
    cfg.max_games, cfg.hidden, cfg.batch = 1500, 32, 8
    cfg.probe_steps, cfg.probe_batch, cfg.eval_batch = 1500, 256, 256
    cfg.print_every, cfg.min_eval_games, cfg.device = 1000, 1, "cpu"
    beta, _ = K.load_mask(pj, pa)
    games, _ = K.load_games(PGN, kdata, cfg.max_games, S_LEN + HORIZON)
    train_g, held_g, _ = K.split_games(games, kdata)
    dev = torch.device("cpu")
    tr, ho = K.Store(train_g, dev, torch), K.Store(held_g, dev, torch)
    model = K.build_model(pj, beta, cfg, dev)
    head, pinfo = K.train_probe(torch, model, tr, cfg, S_LEN, dev, "chase")
    maj = K.majority_at(tr, S_LEN - 1, torch)
    out = K.evaluate(torch, pj, model, head, ho, cfg, S_LEN, HORIZON, maj,
                     "chase", 1)
    return dict(pj=pj, model=model, head=head, tr=tr, ho=ho, cfg=cfg, out=out,
                pinfo=pinfo, train_g=train_g, held_g=held_g)


def test_the_headline_metric_ships_with_a_control(pipeline):
    """summarise() calls acc_exact the verdict; a joint metric needs a joint control."""
    out = pipeline["out"]
    assert "majority_exact" in out, (
        "acc_exact is printed as the verdict and the only control beside it is "
        "majority_from, a marginal. keys returned by evaluate(): %r"
        % (sorted(out.keys()),))


def test_the_1x_accuracy_is_split_by_train_overlap(pipeline):
    """0.42 of 1x positions are also in train; the accuracy must be split on that."""
    out = pipeline["out"]
    have = [k for k in out if "seen" in k or "overlap" in k or "dedup" in k]
    assert have, (
        "fen_overlap measures the contaminated fraction and evaluate() never "
        "sees it; the 1x accuracy is one pooled number over both strata. keys "
        "returned by evaluate(): %r" % (sorted(out.keys()),))


def test_the_probe_prediction_is_not_a_constant(pipeline):
    """A read of 7 columns through one linear layer must still vary per position."""
    p = pipeline
    st, cfg, model, head = p["ho"], p["cfg"], p["model"], p["head"]
    starts, n = st.head_starts(S_LEN + HORIZON, cfg.max_eval_games)
    idx = st.window(starts, S_LEN)
    with torch.no_grad():
        logits = head(K.read_at_last(model, st.feats(idx)))
    pf = logits[:, :64].argmax(-1)
    pt = logits[:, 64:].argmax(-1)
    nf, nt = len(set(pf.tolist())), len(set(pt.tolist()))
    top = max(pf.bincount(minlength=64).tolist()) / max(n, 1)
    assert nf >= 8 and nt >= 8, (
        "over %d holdout positions the head predicts %d distinct from-squares "
        "and %d distinct to-squares; the single most-predicted from-square "
        "covers %.4f of them. a metric this head cannot move is not measuring "
        "the encoder" % (n, nf, nt, top))


def test_the_read_magnitude_does_not_track_the_context_length(pipeline):
    """Same games, same scored ply, only L differs: the read must not rescale.

    beta is 1.0 on four read columns and 0.0 on three. Z_i sums over the whole
    context, so at 8x the beta=0 columns carry roughly 8x the magnitude they
    carried at 1x while the beta=1 columns carry roughly an eighth. A head fit
    at 1x and an NRMSE taken against an encoder target that does NOT rescale
    both move with that factor before any information changes hands.
    """
    p = pipeline
    st, model = p["ho"], p["model"]
    qual = (st.lens >= 132).nonzero(as_tuple=True)[0]
    if qual.numel() < 3:
        pytest.fail("corpus has %d holdout games of 132 plies, under the 3 this "
                    "check needs: it did not run, which is a failure and not a "
                    "skip" % qual.numel())
    base = st.base[qual]
    mag = {}
    for L in LENGTHS:
        idx = st.window(base + (127 - L + 1), L)
        with torch.no_grad():
            r = K.read_at_last(model, st.feats(idx))
        mag[L] = float(r.abs().mean())
    ratio = {L: mag[L] / mag[16] for L in LENGTHS}
    assert max(ratio.values()) / min(ratio.values()) < 1.5, (
        "the read rescales with the context length on identical rows: mean|read| "
        "%r, ratio to 1x %r. the 4x and 8x columns of the results table move by "
        "this factor before any generalisation is measured" % (mag, ratio))


def test_the_probe_carries_information_about_the_human_move(pipeline):
    """Two 64-way heads at chance cost 2*ln(64) nats. The fit must buy some back."""
    import math
    t = pipeline["pinfo"]["probe_loss_trace"]
    chance = 2.0 * math.log(64.0)
    tail = sum(t[-100:]) / len(t[-100:])
    assert chance - tail >= 0.50, (
        "after %d steps the probe cross-entropy is %.4f against a uniform-chance "
        "%.4f: the read buys %.4f nats out of the %.4f available across both "
        "squares. every accuracy printed off this head is a marginal, not a read"
        % (len(t), tail, chance, chance - tail, chance))
