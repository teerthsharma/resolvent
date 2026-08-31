"""Is there an interventional domain that is ALREADY linguistic?

THEORY.md section 2 concedes observational text cannot identify p(y|do(x)), picks
MuJoCo because interventions there are real, and then admits (risk 3) that
physics->language transfer is assumed.

These tests measure that assumption instead of assuming it, and measure whether
executed source code supplies a real do() operator with an oracle-measured
outcome over units that are already tokens.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import interventional as iv

TRUE_DA = -1.0  # d y / d a for template 0, by construction of the source text


def test_outcome_is_measured_by_the_interpreter_not_by_the_test():
    """The in-process runner must agree with a real CPython subprocess.

    If it does not, the 'oracle' is the test's own arithmetic and the whole
    interventional claim is circular.
    """
    for a, u, t in [(0, 0, 0), (7, 1, 0), (13, 1, 1), (4, 0, 2), (20, 1, 2)]:
        src = iv.render(a, u, t)
        out = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True, timeout=30
        )
        assert out.returncode == 0, out.stderr
        assert iv.run_program(src) == float(out.stdout.strip())


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            "observational",
            marks=pytest.mark.xfail(
                strict=True,
                reason="observational text cannot identify do(); measured sign flip",
            ),
        ),
        "interventional",
    ],
)
def test_do_effect_recovered_from_executed_code(mode):
    """Recover dy/da from a corpus of Python programs whose outputs were executed.

    Same estimator, same program family, same sample count. The only difference
    is whether `a` was set by an edit or observed alongside the hidden style `u`.
    """
    c = iv.code_corpus(600, mode=mode, seed=0)
    only_t0 = c["t_lit"] == 0
    slope = iv.ols_slope(c["a_lit"][only_t0], c["y"][only_t0])
    assert slope == pytest.approx(TRUE_DA, abs=0.25), f"{mode} slope={slope:.3f}"


def test_physics_pretraining_beats_a_sham_prior_of_equal_strength():
    """THEORY.md section 2's load-bearing assumption, given every advantage.

    Both domains share state/action/target shapes so an operator fitted on one
    can be applied to the other with no adapter at all. Transfer is the generous
    kind: ridge the code fit toward the physics-fitted W instead of toward zero.
    Held-out interventions use edit magnitudes never seen in training, which is
    the stated win condition (intervention generalization).

    Two controls, both at IDENTICAL total regularisation strength, because
    shrinking toward any prior is itself regularisation and a 64-sample fit is
    helped by regularisation whatever it shrinks toward:

      scratch : shrink toward zero at lam = lam0 + prior_w
      sham    : shrink toward an operator fitted on physics with the action
                column permuted -- same shape, same scale, dynamics destroyed

    Physics->language transfer is real only if the physics prior beats BOTH.
    Averaged over 8 independent draws, because a single draw is noise.
    """
    phys = iv.physics_corpus(4000, seed=1)
    W_phys = iv.fit_bilinear(phys["z"], phys["a"], phys["z_next"])
    W_sham = iv.fit_bilinear(phys["z"], iv.shuffle_actions(phys["a"], seed=7),
                             phys["z_next"])
    lam0, pw = 1e-3, 1.0
    scratch, sham, physics = [], [], []

    for s in range(8):
        few = iv.code_corpus(64, mode="interventional", seed=100 + s,
                             edit_range=(-3, 3))
        ood = iv.code_corpus(400, mode="interventional", seed=200 + s,
                             edit_range=(7, 9))

        def err(W):
            return iv.nrmse(iv.predict(W, ood["z"], ood["a"]), ood["z_next"])

        scratch.append(err(iv.fit_bilinear(few["z"], few["a"], few["z_next"],
                                           lam=lam0 + pw)))
        sham.append(err(iv.fit_bilinear(few["z"], few["a"], few["z_next"],
                                        lam=lam0, prior_W=W_sham, prior_w=pw)))
        physics.append(err(iv.fit_bilinear(few["z"], few["a"], few["z_next"],
                                           lam=lam0, prior_W=W_phys, prior_w=pw)))

    e_scratch, e_sham, e_phys = (float(np.mean(x)) for x in (scratch, sham, physics))
    assert e_phys < e_scratch and e_phys < e_sham, (
        f"physics prior NRMSE={e_phys:.4f} vs matched-lam scratch={e_scratch:.4f} "
        f"vs sham (action-shuffled) prior={e_sham:.4f} -- mean of 8 draws. "
        "Physics pretraining is indistinguishable from shrinking toward a prior "
        "with the dynamics destroyed; the apparent gain is regularisation strength."
    )


def test_intervention_conditioning_is_what_generalizes_on_code():
    """The win condition, inside a single already-linguistic domain.

    Train on edits in [-3,3], test on edits in [7,9] never seen in training.
    Three predictors, one training set, no transfer step anywhere:

      do-nothing    : the edit changes nothing
      action-blind  : linear map z -> z', ignores which edit was applied
      bilinear      : sigmoid's phi(z,a) = [z ; a (x) z ; a ; 1]

    Knowing the intervention must help at unseen magnitudes, or nothing
    interventional was learned.
    """
    train = iv.code_corpus(4000, mode="interventional", seed=4, edit_range=(-3, 3))
    ood = iv.code_corpus(400, mode="interventional", seed=5, edit_range=(7, 9))

    e_nothing = iv.nrmse(ood["z"], ood["z_next"])
    W_blind = iv.fit_bilinear(train["z"], np.zeros_like(train["a"]), train["z_next"])
    e_blind = iv.nrmse(
        iv.predict(W_blind, ood["z"], np.zeros_like(ood["a"])), ood["z_next"])
    W = iv.fit_bilinear(train["z"], train["a"], train["z_next"])
    e_bilinear = iv.nrmse(iv.predict(W, ood["z"], ood["a"]), ood["z_next"])

    assert e_bilinear < e_blind < e_nothing, (
        f"bilinear={e_bilinear:.4f} action-blind={e_blind:.4f} "
        f"do-nothing={e_nothing:.4f}"
    )


def test_program_text_is_the_carrier_of_the_intervention():
    """The intervention is an edit to a token sequence, not a force vector.

    Distinct interventions must produce distinct source text; if the corpus is
    not textually distinguishable there is nothing linguistic about it.
    """
    c = iv.code_corpus(200, mode="interventional", seed=6)
    srcs = {iv.render(int(a), int(u), int(t))
            for a, u, t in zip(c["a_lit"], c["u_lit"], c["t_lit"])}
    assert len(srcs) > 50
    assert all(s.lstrip().startswith("def run") for s in srcs)
