"""RULING 10' -- "PINNED" BY LIKELIHOOD RATIO. The criterion that replaces `5*delta_beta`.

WHY THE OLD CRITERION HAD TO GO, stated as the test that would have caught it.
RULING 2a defined PINNED as `|beta_i - 1| <= 5*delta_beta_i`, with `delta_beta`
read off an identical-seed training pair. The re-take measured that pair BITWISE
on the certified device -- `delta_nrmse = 0.0` -- so the tolerance collapsed to
exactly zero and every run read the "moved" branch for a reason about the pair's
determinism rather than about training. RULING 10' does not patch that floor: it
observes that Wilks' randomness is over the DATA and not over the optimizer, so a
bitwise training pair is IRRELEVANT to the test, and replaces the criterion with

    Lambda = 2 * [ LL_eval(beta_final) - LL_eval(beta == 1) ]

from two deterministic forward passes on the held-out eval split. `delta_beta`
survives as a diagnostic and `beta_summary` still computes it; it is no longer
the criterion, and `test_beta_summary_calls_the_delta_beta_branch_a_diagnostic`
is the receipt that the module says so.

THE THREE CONSTANTS ARE ALL OFF THE SHELF AND NONE IS INVENTED HERE.
`3.841` is chi-squared at 1 dof, 0.95. `ln n` is BIC's, `n` the eval count. The
`2` is Wilks'. There is no `k`, and there is no interpolation between `3.841`
and `ln n` -- the region between them has its own verdict string and
`test_the_interval_verdict_is_the_rulings_exact_words` pins it.

THE NON-DEGENERACY HALF OF EVERY PASS, because this repo has struck 15 vacuous
controls and the criterion being replaced broke precisely because nobody
measured its degenerate case before adopting it:

  * `Lambda == 0.0` at `beta == 1` is only a result if `Lambda != 0` when beta
    moves -- both halves are asserted in the same test.
  * the restore proof compares with `torch.equal` and its planted negative is a
    ONE-ULP change to the same tensor, so the comparator is shown to be capable
    of failing.
  * the power line is checked against the formula AND against a second,
    independent path (the Wald form), because two forms that agree by
    construction agree about nothing.
"""
import json
import math
import os

import pytest
import torch

from ceq.hf import modeling_ceq as M
from ceq.hf.configuration_ceq import CEQConfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPE = dict(vocab_size=32, hidden_size=32, num_hidden_layers=2,
             num_attention_heads=4, max_position_embeddings=16)

#: The ruling's own two `[RUN]` figures, quoted so the arithmetic is checkable
#: rather than inherited. `V17K_RULINGS.md`, RULING 10'.
AUTHOR_LN_N = 8.29
AUTHOR_N = 4000
AUTHOR_DEPARTURE = 0.02
AUTHOR_POWER = 0.32


def _determinism_state():
    """The flag AND its `warn_only` half. Reading only
    `are_deterministic_algorithms_enabled()` loses `warn_only`, and restoring
    from that lossy read turns an ambient `(True, warn_only=True)` into STRICT
    -- which raises on the next backward through `cumprod`. Found the hard way:
    `scripts/v15_r1.py:575` sets `(True, warn_only=True)` process-wide and does
    not restore it, `tests/gate0/test_g14_instrument_hash.py` calls into it, so
    by collection order this file inherits `warn_only=True` and a lossy restore
    here silently escalated it for every test that ran afterwards."""
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.fixture(autouse=True)
def _pinned_regime():
    """One thread and RULING 1's regime, both pinned and both restored.

    THREADS: a float32 reduction changes order with the thread count, and
    `tests/gate0/test_g03_persist.py` sets this at MODULE scope, so a file that
    does not pin it is order-dependent on the rest of the suite -- the defect
    `V17_R2_BETA.md` section 4.3(ii) records.

    DETERMINISM: this file is pinned to `warn_only=True`, RULING 1's regime for
    the round, so its readings do not depend on what an earlier module left in a
    process-wide flag. The one test that needs STRICT sets it for itself.
    """
    threads, det = torch.get_num_threads(), _determinism_state()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(det[0], warn_only=det[1])
    torch.set_num_threads(threads)


def _model(seed=0, **over):
    torch.manual_seed(seed)
    cfg = CEQConfig(operator="smprime", **{**SHAPE, **over})
    m = M.CEQForCausalLM(cfg)
    m.eval()
    return m


def _ids(batch=8, seq=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, SHAPE["vocab_size"], (batch, seq), generator=g)


def _eval_batch(**kw):
    ids = _ids(**kw)
    return dict(input_ids=ids, labels=ids)


def _set_beta(model, values):
    with torch.no_grad():
        for b, v in zip(model.model.layers, values):
            b.self_attn.beta.fill_(float(v))
    return model


def _trained(steps=30, seed=0, lr=0.02):
    """A model whose `beta` is somewhere the optimizer put it.

    NOT A TRAINING RESULT. 30 steps at a 27,914-parameter shape on one repeated
    batch exists so the observed information is positive and the Wald form has
    something to cross-check; no number out of it says anything about the arm.
    """
    m = _model(seed=seed)
    m.train()
    batch = _eval_batch()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        m(**batch).loss.backward()
        opt.step()
    m.eval()
    return m


def _state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _power_chi2_1(lam, crit):
    """EXACT for 1 dof, not an approximation: a noncentral chi-squared with one
    degree of freedom and noncentrality `lam` IS `(Z + sqrt(lam))**2`, so its
    survival function is two normal tails. stdlib `math.erf`, no scipy."""
    s, r = math.sqrt(crit), math.sqrt(lam)
    return _phi(r - s) + _phi(-s - r)


# --------------------------------------------------------------- the statistic

def test_lambda_is_exactly_zero_when_beta_is_already_one_and_not_when_it_moved():
    """THE DEGENERATE HALF AND ITS MUST-FIRE, in one test on purpose.

    `Lambda == 0.0` at `beta == 1` is not evidence of anything by itself: a
    function that returns `0.0` unconditionally passes it. The second half is
    the control -- the SAME model, the SAME batch, one `beta` moved -- and it
    must be non-zero.
    """
    batch = _eval_batch()
    at_one = M.beta_lrt(_set_beta(_model(), [1.0, 1.0]), **batch)
    assert at_one["lambda_joint"] == 0.0, at_one["lambda_joint"]
    assert all(r["lambda"] == 0.0 for r in at_one["per_parameter"]), at_one

    moved = M.beta_lrt(_set_beta(_model(), [1.4, 1.0]), **batch)
    assert moved["lambda_joint"] != 0.0
    assert moved["per_parameter"][0]["lambda"] != 0.0
    #: and the beta that did NOT move still reads exactly zero, so the
    #: per-parameter statistic is per-parameter and not a broadcast copy.
    assert moved["per_parameter"][1]["lambda"] == 0.0


def test_lambda_is_two_n_times_the_loss_difference_the_ruling_writes():
    """Second, independent path: `beta_substitution` already forwards the model
    at trained beta and at beta<-1 and returns both losses. `Lambda` must be
    `2*n*(loss_at_one - loss_trained)` off those, since the model's loss is a
    MEAN cross-entropy and `LL = -n * loss`."""
    batch = _eval_batch()
    m = _set_beta(_model(), [1.3, 0.8])
    sub = M.beta_substitution(m, **batch)
    got = M.beta_lrt(m, **batch)
    n = got["n_eval"]
    assert n == int((batch["labels"][:, 1:] != -100).sum())
    expect = 2.0 * n * (sub["loss_at_beta_one"] - sub["loss_trained"])
    assert got["lambda_joint"] == pytest.approx(expect, rel=1e-9, abs=1e-9)


def test_the_second_pass_sets_every_beta_to_exactly_one():
    """The ruling says the second pass sets every beta to EXACTLY 1 without
    otherwise touching the model. Asserted by recording what the forward
    actually saw, not by trusting the call site."""
    seen = []
    m = _set_beta(_model(), [1.3, 0.8])
    real = m.forward

    def spy(**kw):
        seen.append([float(b.self_attn.beta) for b in m.model.layers])
        return real(**kw)

    m.forward = spy
    M.beta_lrt(m, **_eval_batch())
    assert [1.3, 0.8] == pytest.approx(seen[0], abs=1e-6), seen
    assert [1.0, 1.0] in seen, seen
    #: exactly 1.0, bitwise, not "close to 1"
    assert any(v == [1.0, 1.0] for v in seen), seen


def test_the_model_is_bit_identical_after_the_statistic_and_the_check_can_fail():
    """RESTORE PROOF, `torch.equal`, never `allclose` -- plus its planted
    negative, because a comparator that cannot fail proves nothing. The plant is
    ONE float32 ulp on the same tensor the substitution writes."""
    m = _set_beta(_trained(), [1.25, 0.75])
    before = _state(m)
    M.beta_lrt(m, **_eval_batch())
    after = _state(m)
    assert set(before) == set(after)
    for k in before:
        assert torch.equal(before[k], after[k]), k

    #: PLANT: one ulp on beta itself. The same comparator must reject it.
    with torch.no_grad():
        b = m.model.layers[0].self_attn.beta
        b.copy_(torch.nextafter(b, b + 1))
    planted = _state(m)
    assert not all(torch.equal(before[k], planted[k]) for k in before)


def test_the_statistic_leaves_the_gradient_state_alone():
    """`.grad` is the census's input (`beta_census` reads it after
    `loss.backward()`), so a statistic that populates or clears it corrupts a
    quantity RULING 10' explicitly leaves standing."""
    m = _trained()
    batch = _eval_batch()
    m.zero_grad(set_to_none=True)
    m(**batch).loss.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in m.named_parameters()}
    M.beta_lrt(m, **batch)
    for n, p in m.named_parameters():
        if grads[n] is None:
            assert p.grad is None, n
        else:
            assert torch.equal(grads[n], p.grad), n


def test_the_statistic_restores_the_training_flag():
    m = _trained()
    m.train()
    M.beta_lrt(m, **_eval_batch())
    assert m.training is True
    m.eval()
    M.beta_lrt(m, **_eval_batch())
    assert m.training is False


def test_labels_are_required_because_n_is_read_off_them():
    """`n` is the eval count. It sets `ln n` AND the power line, so guessing it
    from a shape rather than reading it off the scored targets is how a
    tolerance gets invented. Refuse instead."""
    with pytest.raises(ValueError, match="labels"):
        M.beta_lrt(_model(), input_ids=_ids())


def test_an_operator_without_beta_yields_no_verdict_rather_than_a_pinned_one():
    """`sgate` has no `beta`. "No beta" is not "beta pinned"; the absence must
    read as absence."""
    m = M.CEQForCausalLM(CEQConfig(operator="sgate", **SHAPE))
    got = M.beta_lrt(m, **_eval_batch())
    assert got["n_beta"] == 0
    assert got["lambda_joint"] is None
    assert got["verdict_joint"] is None
    assert got["per_parameter"] == []


# ------------------------------------------------------------- the three-way verdict

def test_the_pinned_boundary_is_inclusive_at_3_841_and_the_constant_is_not_ours():
    ln_n = math.log(4000)
    assert M.CHI2_1_AT_95 == 3.841
    assert M._lrt_verdict(3.841, ln_n) == "PINNED"
    assert M._lrt_verdict(math.nextafter(3.841, 5.0), ln_n) != "PINNED"
    assert M._lrt_verdict(0.0, ln_n) == "PINNED"
    #: a NEGATIVE Lambda reads PINNED: beta_final fits the eval split WORSE than
    #: beta == 1, which is no evidence against beta == 1.
    assert M._lrt_verdict(-60.0, ln_n) == "PINNED"


def test_moved_needs_strictly_more_than_ln_n():
    ln_n = math.log(4000)
    assert M._lrt_verdict(math.nextafter(ln_n, 100.0), ln_n) == "MOVED"
    assert M._lrt_verdict(ln_n, ln_n) != "MOVED"
    assert M._lrt_verdict(1e6, ln_n) == "MOVED"


def test_the_interval_verdict_is_the_rulings_exact_words():
    ln_n = math.log(4000)
    mid = (3.841 + ln_n) / 2.0
    assert M._lrt_verdict(mid, ln_n) == "rejected at 0.95, below description-length"


def test_the_verdict_changes_only_where_a_constant_is_crossed():
    """"The verdict changes when it should and only then." A sweep across both
    constants must produce exactly the three verdicts, in order, with exactly
    two changes."""
    ln_n = math.log(4000)
    grid = [0.0, 1.0, 3.8, 3.841, 3.8411, 5.0, 8.2, 8.294, 8.2941, 20.0]
    seen = [M._lrt_verdict(x, ln_n) for x in grid]
    changes = [i for i in range(1, len(seen)) if seen[i] != seen[i - 1]]
    assert len(changes) == 2, list(zip(grid, seen))
    assert seen[0] == "PINNED" and seen[-1] == "MOVED"
    assert set(seen) == {"PINNED", "rejected at 0.95, below description-length",
                         "MOVED"}


def test_the_interval_is_empty_below_47_eval_items_and_that_is_not_an_interpolation():
    """`ln n < 3.841` for `n <= 46`, so the middle verdict is UNREACHABLE on a
    small eval split and every rejection is a MOVED. This is a property of the
    two shelf constants, not a rule anyone chose, and it is recorded rather than
    patched with an invented interpolation."""
    assert math.log(46) < 3.841 < math.log(47)
    small = math.log(46)
    assert M._lrt_verdict(4.0, small) == "MOVED"
    big = math.log(47)
    assert M._lrt_verdict(3.843, big) == "rejected at 0.95, below description-length"


def test_a_beta_the_data_prefers_crosses_and_a_beta_planted_at_one_does_not():
    """PLANT AND CONTROL. Same model, same eval batch, only beta differs.

    The plant that must cross is the beta the OPTIMIZER put there, not one this
    test moved by hand -- see the next test for why those are different things.
    """
    batch = _eval_batch()
    m = _trained()
    trained_beta = [float(b.self_attn.beta) for b in m.model.layers]
    moved = M.beta_lrt(m, **batch)
    assert moved["lambda_joint"] > M.CHI2_1_AT_95, moved["lambda_joint"]
    assert moved["verdict_joint"] == "MOVED"
    assert moved["per_parameter"][0]["verdict"] == "MOVED"

    at_one = M.beta_lrt(_set_beta(m, [1.0, 1.0]), **batch)
    assert at_one["lambda_joint"] == 0.0
    assert at_one["verdict_joint"] == "PINNED"
    assert all(r["verdict"] == "PINNED" for r in at_one["per_parameter"])
    _set_beta(m, trained_beta)


def test_a_beta_far_from_one_that_the_data_dislikes_reads_pinned_and_that_is_the_point():
    """THE DIFFERENCE BETWEEN RULING 10' AND THE CRITERION IT REPLACED, as a
    test rather than as prose.

    `5*delta_beta` was a DISTANCE criterion: far from 1 meant moved. `Lambda` is
    a LIKELIHOOD criterion. A `beta` planted far from 1 in a direction the data
    dislikes makes the model fit WORSE, so `Lambda` is large and NEGATIVE and
    the verdict is PINNED -- the data does not reject `beta == 1` in favour of
    that point, because that point is worse. That is the criterion working, not
    failing, and the two criteria are therefore not nested: the numbers below
    are `[MEASURED]` this box, this run, and the distance criterion would call
    the same model MOVED.
    """
    batch = _eval_batch()
    m = _set_beta(_trained(), [0.4, 0.4])
    got = M.beta_lrt(m, **batch)
    assert abs(0.4 - 1.0) > 0.0                       # far by any distance rule
    assert got["lambda_joint"] < 0.0, got["lambda_joint"]
    assert got["verdict_joint"] == "PINNED"
    #: and the distance diagnostic RULING 10' retired disagrees, loudly
    diag = M.beta_summary(
        [{"step": 0, "name": [r["name"] for r in got["per_parameter"]],
          "beta": [0.4, 0.4], "grad": [0.1, 0.1],
          "requires_grad": [True, True]}], delta_beta=[1e-3, 1e-3])
    assert diag["branch"] == "B" and diag["pinned"] == [False, False]


# --------------------------------------------------------- the Wald cross-check

def test_the_observed_information_is_a_second_derivative_and_not_a_squared_first():
    """The Hessian-diagonal read, checked against a central finite difference of
    the SAME loss in beta. Its planted negative is the squared gradient, which
    is what a careless Fisher implementation returns instead."""
    batch = _eval_batch()
    m = _set_beta(_trained(), [1.15, 0.9])
    got = M.beta_lrt(m, **batch)
    n = got["n_eval"]
    h = 1e-3
    for i, row in enumerate(got["per_parameter"]):
        b = m.model.layers[i].self_attn.beta
        was = float(b)
        vals = []
        for d in (-h, 0.0, h):
            with torch.no_grad():
                b.fill_(was + d)
            with torch.no_grad():
                vals.append(float(m(**batch).loss))
        with torch.no_grad():
            b.fill_(was)
        fd = (vals[0] - 2 * vals[1] + vals[2]) / (h * h)
        assert row["I_beta_per_obs"] == pytest.approx(fd, rel=2e-2), (i, row, fd)
        assert row["I_beta_total"] == pytest.approx(n * row["I_beta_per_obs"],
                                                    rel=1e-9)
    #: the plant: the squared first derivative is a different number entirely
    loss = m(**batch).loss
    betas = [b.self_attn.beta for b in m.model.layers]
    g1 = [float(x) for x in torch.autograd.grad(loss, betas)]
    for row, g in zip(got["per_parameter"], g1):
        assert row["I_beta_per_obs"] != pytest.approx(g * g, rel=1e-3)


def _beta_to_eval_mle(model, batch, iters=12):
    """Newton on `beta` ALONE against the eval batch, using the same exact
    double backward `beta_lrt` uses. This exists because the LRT/Wald
    equivalence is claimed AT THE MLE and nowhere else -- testing it at a beta
    fit on a different split would be testing a claim nobody made."""
    betas = [b.self_attn.beta for b in model.model.layers]
    for _ in range(iters):
        g1 = torch.autograd.grad(model(**batch).loss, betas, create_graph=True)
        h = [float(torch.autograd.grad(g1[i], b, retain_graph=True)[0])
             for i, b in enumerate(betas)]
        g = [float(x) for x in g1]
        if max(abs(x) for x in g) < 1e-8:
            break
        with torch.no_grad():
            for b, gi, hi in zip(betas, g, h):
                if hi > 0:
                    b.sub_(gi / hi)
    return model, max(abs(x) for x in g)


def test_the_wald_form_and_the_lrt_agree_at_the_mle_for_a_small_departure():
    """THE SECOND INDEPENDENT PATH. `(beta-1)^2 * I_beta` against the same
    `3.841`, with `I_beta` the observed information -- asymptotically the same
    statistic as `Lambda`, computed from a Hessian instead of from a second
    forward pass. They must track where the asymptotics claim they do: at the
    MLE, for a departure small against the curvature."""
    batch = _eval_batch()
    m, score = _beta_to_eval_mle(_trained(), batch)
    assert score < 1e-6, score
    rows = M.beta_lrt(m, **batch)["per_parameter"]
    small = [r for r in rows
             if 0 < abs(r["beta_final"] - 1.0) < r["beta_min_detectable"]]
    assert small, rows          # the shape must actually produce such a row
    for r in small:
        assert r["wald_rel_gap"] < 0.10, r
        assert r["wald_agrees"] is True


def test_the_two_forms_disagree_where_the_asymptotics_stop_and_that_is_reported():
    """NON-DEGENERACY OF THE AGREEMENT ABOVE. Two forms that agree everywhere
    agree about nothing, so the disagreements are pinned too -- and both are
    findings about the ASYMPTOTICS, not about the implementation:

      (i)  OFF THE MLE the expansion of `Lambda` carries a first-order score
           term that the Wald form drops entirely. `beta_final` from a real run
           is fit on the TRAINING split and is never the eval split's maximiser,
           so this is the normal case, not a corner.
      (ii) FAR from the null the quadratic approximation to `LL` fails on its
           own terms, MLE or not.

    CASE (i) DISAGREES ON THE VERDICT, NOT MERELY ON THE NUMBER, and that is the
    finding this test exists to pin: [MEASURED] this box, this run, at
    `beta = 1.02` on a 30-step model, `Lambda = 7.21` (MOVED) against
    `Wald = 1.15` (does not reject) -- OPPOSITE sides of the same `3.841`. The
    Wald form drops the score term, and the score is exactly what is non-zero
    off the MLE. So the Wald is NOT a free confirmation of the LRT; where the
    two split, `wald_agrees` reads False and the card must print both.

    CASE (ii) keeps the verdict and moves the number, since far past the
    resolution both statistics are large.
    """
    batch = _eval_batch()
    off_mle = M.beta_lrt(_set_beta(_trained(), [1.02, 1.0]),
                         **batch)["per_parameter"][0]
    assert off_mle["I_beta_total"] > 0
    assert off_mle["wald_rel_gap"] > 0.5, off_mle
    assert off_mle["lrt_rejects_at_95"] is True, off_mle
    assert off_mle["wald_rejects_at_95"] is False, off_mle
    assert off_mle["wald_agrees"] is False, off_mle
    #: and the report SAYS so rather than printing one of the two
    line = [l for l in M.lrt_report(M.beta_lrt(_set_beta(_trained(), [1.02, 1.0]),
                                               **batch)).splitlines()
            if off_mle["name"] in l][0]
    assert "DISAGREES" in line, line

    m, _ = _beta_to_eval_mle(_trained(), batch)
    far = [r for r in M.beta_lrt(m, **batch)["per_parameter"]
           if abs(r["beta_final"] - 1.0) > 2 * r["beta_min_detectable"]]
    assert far, "the shape must produce a departure well past its resolution"
    assert any(r["wald_rel_gap"] > 0.2 for r in far), far
    assert all(r["wald_agrees"] for r in far), far


def test_a_non_positive_observed_information_is_reported_and_never_clamped():
    """An UNTRAINED model at beta == 1 sits at a local MAXIMUM of the eval loss
    along beta, so the observed information is NEGATIVE and the Wald statistic
    does not exist. Observed information is only guaranteed non-negative at an
    MLE and beta_final is not one. The number must be reported as measured, with
    `wald` absent -- not clamped to zero, not made positive by an abs()."""
    got = M.beta_lrt(_set_beta(_model(), [1.0, 1.0]), **_eval_batch())
    rows = got["per_parameter"]
    assert all(r["I_beta_total"] < 0 for r in rows), rows
    assert all(r["wald"] is None for r in rows), rows
    assert all(r["beta_min_detectable"] is None for r in rows), rows
    assert all("information" in (r["note"] or "") for r in rows), rows
    #: and the LRT verdict still exists -- the Wald is a cross-check, not a gate
    assert all(r["verdict"] == "PINNED" for r in rows)


# ------------------------------------------------------------------ the power line

def test_the_power_line_is_the_formula_the_ruling_prints():
    batch = _eval_batch()
    got = M.beta_lrt(_set_beta(_trained(), [1.1, 1.1]), **batch)
    n = got["n_eval"]
    for row in got["per_parameter"]:
        assert row["I_beta_total"] > 0, row
        expect = math.sqrt(M.CHI2_1_AT_95 / (n * row["I_beta_per_obs"]))
        assert row["beta_min_detectable"] == pytest.approx(expect, rel=1e-9)
        #: the same number read the other way round, since I_total == n*I_per_obs
        assert row["beta_min_detectable"] == pytest.approx(
            math.sqrt(M.CHI2_1_AT_95 / row["I_beta_total"]), rel=1e-9)


def test_the_minimum_detectable_departure_is_a_fifty_percent_power_point():
    """What the printed resolution MEANS, checked rather than asserted in prose:
    a departure of exactly `|beta-1|_min` puts the noncentrality at `3.841`,
    which is 50 % power -- NOT a threshold below which nothing is ever detected
    and above which everything is."""
    assert _power_chi2_1(M.CHI2_1_AT_95, M.CHI2_1_AT_95) == pytest.approx(0.5, abs=1e-3)
    assert _power_chi2_1(0.5 * M.CHI2_1_AT_95, M.CHI2_1_AT_95) < 0.5
    assert _power_chi2_1(4.0 * M.CHI2_1_AT_95, M.CHI2_1_AT_95) > 0.5


def test_the_authors_ln_n_figure_reproduces():
    """`[RUN] ln n = 8.29 at n = 4000`. Arithmetic, and it reproduces."""
    assert round(math.log(AUTHOR_N), 2) == AUTHOR_LN_N


def test_the_authors_32_percent_figure_pins_an_unstated_fisher_information():
    """`[RUN] a real 2 % departure is detected only 32 % of the time at n = 4000`.

    NOT arithmetic -- a power calculation, and it needs an `I_beta` the ruling
    never states. This test recovers the value the figure implies and pins it,
    so the figure is reproducible AS A CONDITIONAL rather than inherited as a
    fact. `V17_R10P_LRT.md` reports what this box measured instead.
    """
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _power_chi2_1(mid, M.CHI2_1_AT_95) < AUTHOR_POWER:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    assert _power_chi2_1(lam, M.CHI2_1_AT_95) == pytest.approx(AUTHOR_POWER, abs=1e-4)
    implied = lam / (AUTHOR_DEPARTURE ** 2 * AUTHOR_N)
    assert implied == pytest.approx(1.390, abs=5e-3), implied
    #: and the two figures are mutually consistent: at that information the 2 %
    #: departure sits BELOW the printed resolution, which is why power < 50 %.
    assert AUTHOR_DEPARTURE < math.sqrt(M.CHI2_1_AT_95 / (AUTHOR_N * implied))


# ----------------------------------------------------------------- what is printed

def test_every_verdict_prints_both_constants_and_its_resolution():
    """RULING 10': "Print both constants alongside the verdict always", and "a
    verdict printed without its minimum detectable departure is a defect"."""
    got = M.beta_lrt(_set_beta(_trained(), [1.1, 0.95]), **_eval_batch())
    text = M.lrt_report(got)
    assert "3.841" in text
    assert "{:.4f}".format(got["ln_n"]) in text
    for row in got["per_parameter"]:
        line = [l for l in text.splitlines() if row["name"] in l]
        assert len(line) == 1, row["name"]
        line = line[0]
        assert row["verdict"] in line
        assert "3.841" in line
        assert "ln n" in line
        assert "|beta-1|_min" in line


def test_a_report_with_no_resolution_still_says_why_rather_than_going_quiet():
    """The untrained case has no `|beta-1|_min` because the information is
    negative. The line must still carry the field with its reason, or the defect
    RULING 10' names -- a verdict without its resolution -- ships silently."""
    got = M.beta_lrt(_set_beta(_model(), [1.0, 1.0]), **_eval_batch())
    text = M.lrt_report(got)
    for row in got["per_parameter"]:
        line = [l for l in text.splitlines() if row["name"] in l][0]
        assert "|beta-1|_min" in line
        assert "undefined" in line


def test_the_joint_statistic_names_its_degrees_of_freedom():
    """The ruling's `Lambda` is written over `beta == 1` -- ALL of them -- and
    called a 1-dof comparison. With one beta per layer those are the same thing
    only at `n_layers == 1`. The joint statistic must carry its own dof so
    nobody reads `3.841` over 2 of them by accident."""
    two = M.beta_lrt(_set_beta(_trained(), [1.2, 0.9]), **_eval_batch())
    assert two["dof_joint"] == 2
    assert two["n_beta"] == 2

    one = _model(num_hidden_layers=1)
    _set_beta(one, [1.3])
    got = M.beta_lrt(one, **_eval_batch())
    assert got["dof_joint"] == 1
    #: at one beta the joint statistic IS the per-parameter one, bitwise
    assert got["lambda_joint"] == got["per_parameter"][0]["lambda"]


# ------------------------------------------------ what RULING 10' retired, and where

def test_beta_summary_calls_the_delta_beta_branch_a_diagnostic():
    """`5*delta_beta` is retired AS THE CRITERION and kept as machinery. The
    numbers must still compute -- `scripts/k_noise_floor.py` keeps measuring
    `delta_beta` and that is correct -- and the module must say what they now
    are."""
    series = [{"step": 0, "name": ["a", "b"], "beta": [1.0, 2.0],
               "grad": [0.1, 0.1], "requires_grad": [True, True]}]
    got = M.beta_summary(series, delta_beta=[1e-3, 1e-3])
    assert got["pinned"] == [True, False]          # machinery intact
    assert got["branch"] == "C"
    assert "diagnostic" in got["criterion"].lower()
    assert "10" in got["criterion"]
    doc = (M.beta_summary.__doc__ or "") + (M._pinning.__doc__ or "")
    assert "RULING 10" in doc
    assert "diagnostic" in doc.lower()


def test_the_card_carries_ruling_10_prime_and_no_longer_states_5_delta_beta_as_the_criterion():
    card = open(os.path.join(ROOT, "MODEL_CARD.md"), encoding="utf-8").read()
    assert "RULING 10" in card
    assert "3.841" in card
    assert "rejected at 0.95, below description-length" in card
    assert "_min" in card and "3.841 / (n" in card       # the power line
    #: the retired sentence, verbatim as the pre-10' card wrote it
    assert "is **PINNED** iff `|β_i,final − 1| ≤ 5·δ_β,i`" not in card
    #: delta_beta is kept, and labelled
    assert "δ_β" in card
    assert "diagnostic" in card.lower()


def test_the_notebook_computes_lambda_at_the_end_of_q3_without_renumbering_anything():
    nb = json.load(open(os.path.join(ROOT, "kaggle", "ceq_v17k.ipynb"),
                        encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 22, len(cells)
    q2 = "".join(cells[15]["source"])
    assert cells[15]["cell_type"] == "markdown"
    assert "Q2 -- DROPPED" in q2 and "RULING 8" in q2
    chunk = "".join(cells[18]["source"])
    assert cells[18]["cell_type"] == "code"
    assert "T.train(" in chunk
    assert "beta_lrt" in chunk and "lrt_report" in chunk
    #: at the END of the cell -- after the training call and its summary prints
    assert chunk.index("T.train(") < chunk.index("beta_lrt")
    assert "final loss" in chunk and chunk.index("final loss") < chunk.index("beta_lrt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this box")
def test_lambdas_passes_are_strict_safe_and_the_hessian_read_is_not():
    """WHERE RULING 10' MEETS RULING 1'S HOLE, measured rather than assumed.

    RULING 1: bitwise for replay + forward, the measured floor for training,
    "the hole is in backward". `Lambda` is two FORWARD passes, so it is inside
    the B2 strict regime and repeats bitwise there. The Hessian-diagonal read is
    a BACKWARD -- autograd differentiates `cumprod` with `cumsum`, and
    `cumsum_cuda_kernel` has no deterministic implementation -- so `beta_lrt`
    RAISES under strict mode.

    CONSEQUENCE, PINNED HERE SO IT CANNOT BE DISCOVERED AT LAUNCH: `Lambda` and
    its verdict are strict-safe, `wald` and `beta_min_detectable` are not, and
    RULING 10' calls a verdict without its resolution a defect. So a `Lambda`
    cell cannot go on RULING 6f's frozen deciding-cell list as a B2 strict cell.
    """
    m = _set_beta(_trained(), [1.2, 0.9]).to("cuda")
    batch = {k: v.to("cuda") for k, v in _eval_batch().items()}
    was = _determinism_state()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            a = float(m(**batch).loss)
            b = float(m(**batch).loss)
        assert a == b, (a, b)               # forward-only: bitwise under strict
        with pytest.raises(RuntimeError, match="cumsum_cuda_kernel"):
            M.beta_lrt(m, **batch)
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    #: THE CHECK THAT WOULD HAVE CAUGHT THIS FILE'S OWN BUG. Restoring from
    #: `are_deterministic_algorithms_enabled()` alone drops `warn_only`, so an
    #: ambient `(True, warn_only=True)` came back as STRICT and every later test
    #: in the session failed on `cumsum_cuda_kernel`. Both halves, or neither.
    assert _determinism_state() == was, (_determinism_state(), was)
    #: and under the round's own regime it runs
    assert M.beta_lrt(m, **batch)["n_beta"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this box")
def test_the_statistic_runs_on_the_certified_device_and_still_restores():
    m = _set_beta(_trained(), [1.2, 0.9]).to("cuda")
    batch = {k: v.to("cuda") for k, v in _eval_batch().items()}
    before = _state(m)
    got = M.beta_lrt(m, **batch)
    for k, v in _state(m).items():
        assert torch.equal(before[k], v), k
    assert got["n_beta"] == 2
    assert got["verdict_joint"] in ("PINNED", "MOVED",
                                    "rejected at 0.95, below description-length")
    assert M.lrt_report(got)
