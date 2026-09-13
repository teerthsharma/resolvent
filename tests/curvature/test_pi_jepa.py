"""RED-FIRST tests for ceqjepa/pi_jepa.py (DR-2 R2: the JEPA).

WHAT THE MODULE UNDER TEST CLAIMS. An online encoder E_theta and an EMA target
E_theta_bar with stop-grad; a predictor that is the family's READ with beta fixed
PER COORDINATE by R1's frozen mask; a loss of latent prediction plus an eps-free
Std term plus a covariance term; action-conditioning scored against a placebo.
Each test below is a way for those claims to be FALSE while numbers still come
out:

  (a) THE PER-COORDINATE READ IS NOT THE FAMILY'S READ. The shipped arm applies
     beta to Z BEFORE the value contraction; a per-coordinate beta can only be
     applied AFTER it. If the two disagree for a scalar beta then the module is
     computing something else and every downstream number is about that other
     thing. The worst difference must be measured at both corners and at one
     interior value, and the loss of bitwise agreement must be stated.
  (b) THE MASK IS DECORATION. A mask of all ones must reproduce the single
     softmax arm and a mask of all zeros the single linear arm. If those two
     identities do not hold the read is not the family's read.
  (c) THE REFUSAL BECOMES A DEFAULT. R1 made its refusal structural precisely so
     a consumer could not silently substitute a corner. A refused coordinate
     must arrive as a refusal carrying its reason, never as a float.
  (d) THE COLLAPSE DETECTOR HAS NEVER SEEN A COLLAPSE. A detector that only ever
     runs on healthy states is a hope. A deliberately collapsed encoder must be
     caught, and a healthy one must not be.
  (e) THE STOP-GRAD DOES NOTHING. Detaching the target is one call and its
     absence is invisible in the loss curve. No gradient may reach the target
     branch, and removing the stop-grad must be SEEN to break that.
  (f) THE EMA DOES NOTHING. Momentum zero must reproduce the online encoder
     exactly, which is the one setting where the answer is known in closed form.
  (g) THE ENCODER IS BORN DEAD. A zero-init output is a stationary point of the
     same shape as the zero-init LoRA death this repository already ate.
  (h) THE BASELINE WAS NEVER RUN. L-SIMPLE: an untrained encoder and a frozen
     random encoder are scored before any claim about training.
  (i) THE THEOREM'S HYPOTHESES WERE NEVER CHECKED ON THE DATA. L-CLASS.
  (j) THE ACTION IS IGNORED. Delta_spec against a placebo, and an action-blind
     predictor must read exactly zero rather than a false win.
  (k) THE NUMBER CAME FROM NO RUN. Every number in the docstrings of BOTH files,
     decimals and bare integers alike, matched as whole TOKENS so a fragment of
     a longer number cannot stand in for one, must appear in the output of a run
     performed here; every published value must equal an expression that
     recomputes it, compared position by position; and every measured number
     stated in prose must be bound to ITS OWN quantity, not merely be present.

NOTHING HERE IS A CLAIM ABOUT A TRAINED MODEL AT SCALE. The bed is synthetic,
the encoder is small, the run is on CPU inside a 300 s bar, and the point of the
exercise is the wiring and the failure modes.

THE IMPORT IS GUARDED ON PURPOSE. A module-level `import ceqjepa.pi_jepa` turns
the RED phase into one collection error with no test names in it. Each test must
be seen to FAIL BY NAME before the module exists.

RUN: python -m pytest tests/curvature/test_pi_jepa.py -v
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    from ceqjepa import pi_jepa as _pi_jepa
except Exception as _exc:                     # noqa: BLE001 -- RED phase carries it
    _pi_jepa = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


def _pj():
    """The module under test, or a named failure instead of a collection error."""
    if _pi_jepa is None:
        pytest.fail("ceqjepa.pi_jepa does not import: %r" % (_IMPORT_ERROR,))
    return _pi_jepa


# ---------------------------------------------------------------------------
# 1. THE READ IS THE FAMILY'S READ
# ---------------------------------------------------------------------------

def test_the_factorised_read_matches_the_shipped_readout_for_a_scalar_beta(capsys):
    """THE OBSTACLE, MEASURED RATHER THAN ASSUMED.

    The shipped arm divides the operator by Z^beta and contracts with V after;
    a per-coordinate beta cannot be applied there, because the operator has no
    output-coordinate axis. The factorisation moves the division past the
    contraction, which is legitimate only because Z_i does not depend on j. That
    is an argument, and an argument is not a measurement: the two are compared
    here at both corners and at one interior value, and the worst difference is
    printed rather than asserted away.
    """
    m = _pj()
    eq = m.report()["equivalence"]
    for row in eq["rows"]:
        print("\n  beta %-6s worst |factorised - shipped| = %.3e   bitwise=%s"
              % (row["beta"], row["worst"], row["bitwise"]))
    by = {row["beta"]: row for row in eq["rows"]}
    assert set(by) == set(m.EQUIV_BETAS), \
        "the equivalence sweep is %r, not %r" % (sorted(by), sorted(m.EQUIV_BETAS))
    assert by[0.0]["bitwise"], \
        "beta=0 is not bitwise: Z^0 is 1 and division by 1 must be exact"
    assert by[0.0]["worst"] == 0.0, "beta=0 worst is %r" % (by[0.0]["worst"],)
    for b in (1.0, m.BETA_INTERIOR):
        assert not by[b]["bitwise"], \
            "beta=%r came back bitwise: the price is being hidden" % (b,)
        assert by[b]["worst"] <= m.EQUIV_TOL, \
            "beta=%r differs by %.3e, beyond %.3e" % (b, by[b]["worst"], m.EQUIV_TOL)
    assert eq["worst_overall"] == max(r["worst"] for r in eq["rows"])
    assert "bitwise" in (m.__doc__ or ""), \
        "the module docstring does not state what bitwise agreement costs"


def test_the_all_ones_mask_is_the_softmax_arm_and_all_zeros_is_the_linear_arm(capsys):
    """IF THESE TWO IDENTITIES FAIL, NOTHING DOWNSTREAM MEANS ANYTHING.

    A per-coordinate mask is only the family's read if its degenerate settings
    are the family's own single-corner arms. Both are checked against the
    SHIPPED readout, not against a re-implementation here, because a
    re-implementation agreeing with itself is not evidence.
    """
    m = _pj()
    c = m.report()["corners"]
    print("\n  all-ones  mask vs shipped softmax arm: worst %.3e (tolerance %.3e)"
          % (c["ones_worst"], m.EQUIV_TOL))
    print("  all-zeros mask vs shipped linear  arm: worst %.3e (bitwise=%s)"
          % (c["zeros_worst"], c["zeros_bitwise"]))
    assert c["ones_worst"] <= m.EQUIV_TOL, \
        "the all-ones mask is not the softmax arm: worst %.3e" % c["ones_worst"]
    assert c["zeros_worst"] == 0.0 and c["zeros_bitwise"], \
        "the all-zeros mask is not bitwise the linear arm: worst %.3e" % c["zeros_worst"]
    assert c["axis_len"] == m.D_LATENT, \
        "the corner identities ran on %d coordinates, not %d" % (c["axis_len"], m.D_LATENT)


# ---------------------------------------------------------------------------
# 2. THE REFUSAL CHANNEL
# ---------------------------------------------------------------------------

def test_a_refused_coordinate_arrives_as_a_refusal_and_never_a_default_corner(capsys):
    """R1 MADE THE REFUSAL STRUCTURAL. UNDOING IT HERE WOULD BE THE WHOLE DEFECT.

    The frozen mask carries None for a coordinate the rule declined. A consumer
    that fills those with a corner has answered a question the rule refused, so
    the read must hand back the refusal itself, carrying the reason, and the
    assigned coordinates must still hand back floats.
    """
    m = _pj()
    rc = m.report()["refusal_channel"]
    print("\n  axis %r length %d, %d assigned, %d refused"
          % (rc["axis"], rc["axis_len"], rc["n_assigned"], len(rc["refused"])))
    for name, code in sorted(rc["refused"].items()):
        print("    REFUSED %-12s %s" % (name, code))
    assert rc["axis"] == "D", "the mask axis is %r" % (rc["axis"],)
    assert rc["axis_len"] == m.D_LATENT
    assert rc["refused"], "no coordinate is refused: the channel is never exercised"
    assert rc["n_assigned"] + len(rc["refused"]) == rc["axis_len"], \
        "the axis was compacted: %d + %d != %d" \
        % (rc["n_assigned"], len(rc["refused"]), rc["axis_len"])
    # Both directions: a refused name yields a Refusal, an assigned name a float.
    for name in rc["refused"]:
        got = m.beta_for(name)
        assert m.is_refusal(got), "%s came back as %r, not a refusal" % (name, got)
        assert got.code == rc["refused"][name]
        assert got not in (0.0, 1.0)
    assert rc["assigned_floats"] == rc["n_assigned"] > 0, \
        "only %d of %d assigned coordinates hand back a float" \
        % (rc["assigned_floats"], rc["n_assigned"])
    assert rc["defaulted"] == 0, \
        "%d refused coordinates were given a corner anyway" % rc["defaulted"]
    # And the read itself refuses rather than producing a column of numbers.
    out = m.read_refused_column()
    assert m.is_refusal(out), "the read produced %r for a refused coordinate" % (out,)


# ---------------------------------------------------------------------------
# 3. COLLAPSE, THE FAILURE MODE THIS ARCHITECTURE ACTUALLY HAS
# ---------------------------------------------------------------------------

def test_a_planted_collapse_is_caught_and_a_healthy_run_is_not(capsys):
    """A COLLAPSE DETECTOR THAT HAS NEVER SEEN A COLLAPSE IS NOT A DETECTOR.

    Both directions, because a detector that fires on everything is as useless
    as one that fires on nothing. And TWO plants, because the two collapses a
    JEPA actually suffers are different failures: a COMPLETE collapse, where
    every sequence maps to the same point, and a DIMENSIONAL collapse, where the
    spread survives but lives on one direction. A per-coordinate variance check
    alone catches the first and passes the second, so the second plant is the
    one that says whether the rank leg is doing any work.
    """
    m = _pj()
    col = m.report()["collapse"]
    for label in m.COLLAPSE_PLANTS:
        row = col[label]
        print("\n  %-9s std_min %.3e  std_med %.3e  erank %.4f of %d  "
              "collapsed=%s  leg=%s"
              % (label, row["std_min"], row["std_med"], row["erank"],
                 row["n_coords"], row["collapsed"], row["leg"]))
    assert set(m.COLLAPSE_PLANTS) == {"healthy", "planted", "rank_one"}

    assert col["planted"]["collapsed"], "the planted collapse was NOT caught"
    assert col["planted"]["leg"] == "std", \
        "the complete collapse fired on %r, not the variance leg" % (col["planted"]["leg"],)
    assert m.is_refusal(col["planted"]["refusal"]), \
        "the detector returned %r rather than a refusal" % (col["planted"]["refusal"],)
    assert col["planted"]["std_min"] < m.COLLAPSE_STD_MIN <= col["healthy"]["std_min"]

    # THE SECOND PLANT, and the reason the rank leg exists at all.
    assert col["rank_one"]["collapsed"], "the dimensional collapse was NOT caught"
    assert col["rank_one"]["leg"] == "rank", \
        "the rank-one plant fired on %r, not the rank leg" % (col["rank_one"]["leg"],)
    assert col["rank_one"]["std_min"] >= m.COLLAPSE_STD_MIN, \
        "the rank-one plant also trips the variance leg, so it does not show the hole"
    assert col["rank_one"]["erank"] < m.COLLAPSE_ERANK_MIN <= col["healthy"]["erank"]
    assert col["rank_one"]["erank"] < col["healthy"]["erank"], \
        "the planted dimensional collapse did not lower the effective rank"

    assert not col["healthy"]["collapsed"], \
        "the detector fires on a healthy representation: std_min %.3e" \
        % col["healthy"]["std_min"]
    assert col["healthy"]["refusal"] is None and col["healthy"]["leg"] is None
    assert col["runs_every_step"], "the detector is not run on every training step"
    assert col["steps_checked"] == m.N_STEPS


def test_the_encoder_initialisation_is_not_degenerate(capsys):
    """NEVER ZERO-INIT THE ENCODER OUTPUT, AND ASSERT IT RATHER THAN INTEND IT."""
    m = _pj()
    ini = m.report()["init"]
    print("\n  init std_min %.4f  erank %.4f of %d  zero_output_layers %d"
          % (ini["std_min"], ini["erank"], ini["n_coords"], ini["zero_layers"]))
    assert ini["zero_layers"] == 0, \
        "%d output layers are zero-initialised" % ini["zero_layers"]
    assert not ini["degenerate"], "the encoder is born collapsed"
    assert ini["std_min"] > m.COLLAPSE_STD_MIN


# ---------------------------------------------------------------------------
# 4. THE WIRING THAT SILENTLY DOES NOTHING
# ---------------------------------------------------------------------------

def test_no_gradient_reaches_the_target_branch_and_removing_the_stop_grad_breaks_it(capsys):
    """ONE LINE, INVISIBLE WHEN ABSENT, SO IT IS SEEN TO FIRE IN BOTH DIRECTIONS."""
    m = _pj()
    sg = m.report()["stopgrad"]
    print("\n  stop-grad ON : %d of %d target parameters carry a gradient"
          % (sg["on_target_grads"], sg["n_target_params"]))
    print("  stop-grad OFF: %d of %d target parameters carry a gradient"
          % (sg["off_target_grads"], sg["n_target_params"]))
    assert sg["n_target_params"] > 0, "there is no target branch to isolate"
    assert sg["on_target_grads"] == 0, \
        "%d target parameters received a gradient with the stop-grad in place" \
        % sg["on_target_grads"]
    assert sg["off_target_grads"] == sg["n_target_params"], \
        "removing the stop-grad changed nothing: %d of %d" \
        % (sg["off_target_grads"], sg["n_target_params"])
    assert sg["isolated"] and not sg["isolated_without_stopgrad"], \
        "the isolation check does not distinguish the two wirings"
    assert not sg["target_requires_grad"], "the target branch requires grad"


def test_ema_at_momentum_zero_reproduces_the_online_encoder_exactly(capsys):
    """THE ONE SETTING WHERE THE ANSWER IS KNOWN IN CLOSED FORM."""
    m = _pj()
    e = m.report()["ema"]
    print("\n  tau=0  worst |target - online| = %.3e over %d tensors (bitwise=%s)"
          % (e["tau0_worst"], e["n_tensors"], e["tau0_bitwise"]))
    print("  tau=%.2f worst |target - online| = %.3e (must be nonzero: the EMA lags)"
          % (m.TAU, e["tau_worst"]))
    assert e["n_tensors"] > 0
    assert e["tau0_bitwise"] and e["tau0_worst"] == 0.0, \
        "momentum 0 did not reproduce the online encoder: worst %.3e" % e["tau0_worst"]
    assert e["tau_worst"] > 0.0, \
        "the EMA at tau=%.2f is identical to the online encoder: it is doing nothing" % m.TAU


# ---------------------------------------------------------------------------
# 5. L-SIMPLE, L-CLASS, AND THE ACTION
# ---------------------------------------------------------------------------

def test_the_simplest_baselines_are_scored_before_any_training_claim(capsys):
    """L-SIMPLE. An untrained encoder and a frozen random encoder come first."""
    m = _pj()
    b = m.report()["baselines"]
    for name in m.BASELINE_ORDER:
        print("\n  %-16s latent MSE %.6f" % (name, b[name]))
    assert set(b) == set(m.BASELINE_ORDER), \
        "the baseline set is %r, not %r" % (sorted(b), sorted(m.BASELINE_ORDER))
    for name in m.BASELINE_ORDER:
        assert b[name] > 0.0, "%s scored exactly zero" % name
    assert "untrained" in b and "frozen_random" in b, \
        "the two required L-SIMPLE baselines are not both present"


def test_the_theorem_hypotheses_are_checked_on_this_bed(capsys):
    """L-CLASS. A win against a theorem whose hypotheses were never checked is
    not a win, so the hypotheses are numbers on THIS bed."""
    m = _pj()
    t = m.report()["theorem"]
    print("\n  %s (%s)" % (t["name"], t["source"]))
    print("  gate zero=%s  logits finite=%s over %d entries"
          % (t["gate_is_zero"], t["logits_finite"], t["n_logits"]))
    print("  row sum at beta=1 %.12f (worst error %.3e); at beta=0 %.6f"
          % (t["row_sum_beta1"], t["row_sum_beta1_err"], t["row_sum_beta0"]))
    assert t["gate_is_zero"] and t["logits_finite"]
    assert t["row_sum_beta1_err"] < m.ROW_SUM_TOL, \
        "the beta=1 corner is not row-stochastic on this bed: %.3e" % t["row_sum_beta1_err"]
    assert t["corners_distinct"], "the two corners coincide on this bed"
    assert "V16Domain.lean" in t["source"]


def test_the_action_is_used_and_a_blind_predictor_reads_exactly_zero(capsys):
    """Delta_spec against a PLACEBO, and the blind control that must read zero."""
    m = _pj()
    a = m.report()["action"]
    print("\n  read       Delta_spec mean %+.6f" % a["read_mean"])
    print("  blind      Delta_spec mean %+.6f (must be exactly 0)" % a["blind_mean"])
    assert a["blind_max_abs"] == 0.0, \
        "an action-blind predictor scored %r: that is a false win" % a["blind_max_abs"]
    assert a["read_max_abs"] > 0.0, \
        "the read ignores its action: Delta_spec is identically zero"
    assert a["placebo_is_different"], \
        "the placebo action equals the real one: the control is vacuous"


# ---------------------------------------------------------------------------
# 6. THE TRAINABLE SURFACE
#
# Everything above reaches the model through zero-argument report functions that
# build it, run it and hand back a dict. That is a fine contract for a
# self-check and a useless one for anything that wants to TRAIN this: there is no
# handle to give an optimiser, no way to run more steps than the demo runs, no
# way to move it to another device, and no way to carry weights off the machine
# that produced them. Each of the five below is pinned separately, because the
# save/load round trip in particular is the one that looks fine while silently
# dropping a buffer -- and here the buffer that would go missing is the mask,
# which is to say the corners the read is taken at.
# ---------------------------------------------------------------------------

def test_the_model_is_constructible_at_a_caller_chosen_size_with_a_caller_mask(capsys):
    """A CLASS, TAKING ITS SHAPES AND ITS MASK AS ARGUMENTS.

    The mask is an ARGUMENT and not a global, because a caller on another machine
    has to be able to pass the frozen vector it loaded. That is checked by
    building at a size and a mask the module's own constants do not describe: if
    the class reaches for beta_vector() internally it cannot honour either.
    """
    import torch

    m = _pj()
    beta = m.beta_vector()
    model = m.PiJepa(beta, x_dim=m.X_DIM, d=m.D_LATENT)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print("\n  default build: %d trainable parameters of %d total, %d columns, "
          "%d refused" % (n_train, n_all, len(model.cols), len(model.refusals)))
    assert n_train > 0 and n_all > n_train, \
        "the target branch is not in parameters(), so it is not in the state_dict"
    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    assert opt.param_groups[0]["params"], "an optimiser cannot consume this model"
    for p in model.trainable_parameters():
        assert p.requires_grad, "a frozen tensor is being handed to the optimiser"

    # A DIFFERENT SIZE AND A DIFFERENT MASK, neither of them the module's own.
    wide = [0.0, 1.0, 0.5, 1.0, 0.0]
    other = m.PiJepa(wide, x_dim=5, d=len(wide), dk=3, hidden=7, n_actions=2)
    print("  caller build:  d=%d dk=%d hidden=%d, axis_len %d, betas %r"
          % (len(wide), 3, 7, other.axis_len, [float(b) for b in wide]))
    assert other.axis_len == len(wide) != m.D_LATENT
    # .tolist(), not list(): a list of 0-dim tensors compared against a list of
    # ints goes through tensor __eq__ and passes on shapes it should not.
    assert other.cols.tolist() == list(range(len(wide))), \
        "a mask with no refusals produced %r columns" % (other.cols.tolist(),)
    for i, want in enumerate(wide):
        assert float(other.beta_assigned[i]) == want, \
            "beta position %d is %r, not the caller's %r" \
            % (i, float(other.beta_assigned[i]), want)
    x = torch.randn(3, 6, 5)
    out = other(x, torch.zeros(3, dtype=torch.long))
    assert tuple(out.out.shape) == (3, 6, len(wide)), \
        "the caller-sized model produced %r" % (tuple(out.out.shape),)

    # And a caller mask carrying a refusal loses exactly that column, declared.
    holed = [1.0, m.Refusal("PLANTED", "a caller's own refusal"), 0.0]
    h = m.PiJepa(holed, x_dim=5, d=3, dk=3, hidden=7, n_actions=2)
    print("  holed build:   axis_len %d, cols %r, refusals %r"
          % (h.axis_len, h.cols.tolist(), sorted(h.refusals)))
    assert h.axis_len == 3 and h.cols.tolist() == [0, 2] \
        and list(h.refusals) == [1]


def test_the_training_loop_takes_its_scale_and_its_data_from_the_caller(capsys):
    """STEPS, BATCH, SEQUENCE LENGTH AND THE DATA SOURCE ARE PARAMETERS.

    The module constants stay as the demo's defaults, so nothing about the
    self-check changes; what is added is that a caller can drive more steps, a
    different batch, and its own data without editing this file.
    """
    import torch

    m = _pj()
    seen = []

    def batches(step):
        """The caller's own data source, and it is SEEN to be the one used."""
        seen.append(step)
        g = torch.Generator().manual_seed(1000 + step)
        x = torch.randn(5, 7 + 2, 6, generator=g)
        a = torch.randint(0, 2, (5,), generator=g)
        return x[:, :7], x[:, 2:2 + 7], a

    beta = [1.0, 0.0, 1.0]
    model = m.PiJepa(beta, x_dim=6, d=3, dk=4, hidden=8, n_actions=2)
    # erank_min is the CALLER'S, because an absolute rank floor is not portable
    # across widths: at d=3 this bed's healthy representation reads under the
    # d=9 floor and would be refused for nothing but its axis length.
    out = m.fit(model, batches=batches, steps=6, lr=1e-2, erank_min=1.1)
    print("\n  caller loop: %d steps, data source called on steps %r"
          % (out["steps_checked"], seen))
    assert out["steps_checked"] == 6 == len(seen) and seen == list(range(6)), \
        "the caller's data source was called %r times" % (len(seen),)
    assert len(out["loss_trace"]) == 6
    assert out["steps_checked"] != m.N_STEPS, \
        "the loop still runs the module's own step count"
    # The demo's defaults are unchanged.
    assert m.train_run()["steps_checked"] == m.N_STEPS


def test_a_state_dict_round_trip_reproduces_the_forward_output_bitwise(capsys):
    """A KAGGLE RUN THAT PRODUCES WEIGHTS NOBODY CAN LOAD PRODUCED NOTHING.

    The failure this is here for is not a missing parameter, which any loss curve
    would show, but a missing BUFFER -- and the buffer at risk here is the mask
    itself. A model that reloads its weights and reads at the wrong corners is
    the exact defect the whole per-coordinate exercise exists to avoid, and it is
    invisible in every metric that does not compare outputs.
    """
    import torch

    m = _pj()
    rt = m.report()["roundtrip"]
    print("\n  round trip worst |reloaded - original| = %.3e over %d state "
          "entries (%d buffers); bitwise=%s"
          % (rt["worst"], rt["n_entries"], rt["n_buffers"], rt["bitwise"]))
    assert rt["bitwise"] and rt["worst"] == 0.0, \
        "a state_dict round trip moved the forward output by %.3e" % rt["worst"]
    assert rt["n_buffers"] > 0, \
        "the state_dict carries no buffers, so the mask is not in it"
    assert rt["mask_in_state"], (
        "the per-coordinate mask is not in the state_dict: a reloaded model reads "
        "at whatever corners its constructor was handed")
    assert rt["mask_bitwise"], "the reloaded mask is not the saved one"

    # And the failure mode, seen: a model reloaded WITHOUT the mask entry reads
    # at different corners, so the round trip is testing something real.
    diff = m.roundtrip_without_mask()
    print("  dropping the mask entry moves the forward output by %.3e" % diff)
    assert diff > 0.0, \
        "dropping the mask from the state_dict changed nothing: the round trip " \
        "would pass with the mask missing"


def test_the_model_builds_on_a_caller_named_device_and_defaults_to_cpu(capsys):
    """DEVICE IS AN ARGUMENT. Nothing assumes CUDA and nothing forbids it."""
    import torch

    m = _pj()
    beta = [1.0, 0.0, 1.0]
    default = m.PiJepa(beta, x_dim=5, d=3, dk=3, hidden=7, n_actions=2)
    named = m.PiJepa(beta, x_dim=5, d=3, dk=3, hidden=7, n_actions=2,
                     device="cpu")
    dev_d = next(default.parameters()).device.type
    dev_n = next(named.parameters()).device.type
    print("\n  default device %r, named device %r, mask device %r"
          % (dev_d, dev_n, named.beta_assigned.device.type))
    assert dev_d == "cpu", "the default device is %r, not cpu" % (dev_d,)
    assert dev_n == "cpu"
    assert named.beta_assigned.device.type == "cpu", \
        "the mask buffer did not follow the model to its device"
    moved = named.to(torch.device("cpu"))
    assert next(moved.parameters()).device.type == "cpu"


# ---------------------------------------------------------------------------
# 7. THE GUARDS
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES, token to token. `"0.5" in out` is true the
#: moment the run prints 10.5 or 0.50, so a substring test passes any fabricated
#: number sharing digits with a real one. The inner (?:\.\d+)* keeps a dotted
#: version string as ONE token; the trailing guard is (?!\w)(?!\.\d) and not
#: (?![\w.]) so a sentence-final period does not hide the number before it.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")

#: Two independently written counters with a floor under each, so that a scope
#: regression reports zero missing and passes. These are the TEST's numbers over
#: BOTH files; the module writes its own over its own scope in
#: docstring_numbers() and all three figures must agree. The floors sit about a
#: tenth under the measured counts.
MIN_DOCSTRINGS = 68
MIN_NUMBERS = 55
MIN_PUBLISHED = 20
MIN_CLAIM_BOUND = 20

_CHILD = "PI_JEPA_GUARD_CHILD"


def _tokens(text):
    """Every number in `text` as a boundary-anchored token, leading + stripped.

    A leading MINUS is kept, because a negative number is a different number and
    matching across the sign would put the defect straight back.
    """
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _own_test_run_output():
    """stdout of THIS file's own tests, so numbers in THEIR docstrings are bound."""
    env = dict(os.environ)
    env[_CHILD] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-s", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.stdout, "the child run printed nothing: %r" % (proc.stderr[-400:],)
    return proc.stdout


def _module_docstrings(mod):
    """(name, docstring) for the module and everything DEFINED in it.

    Filtered on __module__ so torch's and numpy's docstrings do not come along,
    and NOT filtered through __all__, because demo() is the function that prints
    every number and is absent from __all__. This is a SCOPE rule, not an
    exemption list: no number is ever exempted.
    """
    docs = [("module", mod.__doc__ or "")]
    for name, obj in sorted(vars(mod).items()):
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            if getattr(member, "__module__", None) != mod.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


def test_every_measured_number_in_a_docstring_is_printed_by_a_run(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    m = _pj()
    m.demo()
    out = capsys.readouterr().out
    run = _tokens(out) | _tokens(_own_test_run_output())

    # Vacuity controls, both directions and both shapes.
    assert "0.27182818" not in run, "the absent-decimal control was found in the run"
    assert "80317" not in run, "the absent-integer control was found in the run"
    assert str(m.D_LATENT) in run, "a measured integer stopped being matched as a token"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = len({tok for _, doc in docs for tok in _tokens(doc)})
    print("\n  %d distinct numbers across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert len(docs) >= MIN_DOCSTRINGS, \
        "only %d docstrings in scope: the scan has shrunk" % len(docs)
    assert checked >= MIN_NUMBERS, \
        "only %d numbers found: the regex is not biting" % checked

    # The module counts its own, independently, and ALL THREE figures are
    # compared: a floor is not a counter, and n_decimals asserted by nothing at
    # all is how a wrong figure ships green.
    cov = m.docstring_numbers()
    mod_docs = _module_docstrings(m)
    mod_toks = {tok for _, doc in mod_docs for tok in _tokens(doc)}
    mine = (len(mod_docs), len(mod_toks), len([t for t in mod_toks if "." in t]))
    print("  the module counts %d docstrings, %d numbers, %d decimals; "
          "this file recounts %d, %d, %d"
          % (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"],
             mine[0], mine[1], mine[2]))
    assert cov["n_docstrings"] == mine[0], \
        "docstring counters disagree: module %d, test %d" % (cov["n_docstrings"], mine[0])
    assert cov["n_numbers"] == mine[1], \
        "number counters disagree: module %d, test %d" % (cov["n_numbers"], mine[1])
    assert cov["n_decimals"] == mine[2], \
        "decimal counters disagree: module %d, test %d" % (cov["n_decimals"], mine[2])
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


def _recomputed_published(m, r):
    """Every published value, rebuilt from the module's own functions HERE.

    Written independently of report(): each entry re-runs the producing function
    rather than reading the value report() stored, so a literal frozen into the
    published dict fails even when every top-level key beside it is live.
    """
    eq = m.shipped_equivalence()
    co = m.corner_identities()
    cl = m.collapse_must_fire()
    sg = m.stopgrad_must_fire()
    em = m.ema_identity()
    ini = m.init_check()
    tr = m.train_run()
    th = m.theorem_hypotheses()
    ac = m.action_report()
    bl = m.baselines()
    rt = m.roundtrip_check()
    return {
        "equiv_worst_beta0": {row["beta"]: row["worst"] for row in eq["rows"]}[0.0],
        "equiv_worst_beta1": {row["beta"]: row["worst"] for row in eq["rows"]}[1.0],
        "equiv_worst_interior":
            {row["beta"]: row["worst"] for row in eq["rows"]}[m.BETA_INTERIOR],
        "corners_ones_worst": co["ones_worst"],
        "corners_zeros_worst": co["zeros_worst"],
        "collapse_healthy_std_min": cl["healthy"]["std_min"],
        "collapse_healthy_erank": cl["healthy"]["erank"],
        "collapse_planted_std_min": cl["planted"]["std_min"],
        "collapse_planted_erank": cl["planted"]["erank"],
        "collapse_rank_one_std_min": cl["rank_one"]["std_min"],
        "collapse_rank_one_erank": cl["rank_one"]["erank"],
        "stopgrad_on_target_grads": sg["on_target_grads"],
        "stopgrad_off_target_grads": sg["off_target_grads"],
        "ema_tau0_worst": em["tau0_worst"],
        "ema_tau_worst": em["tau_worst"],
        "init_std_min": ini["std_min"],
        "init_erank": ini["erank"],
        "train_loss_first": tr["loss_first"],
        "train_loss_last": tr["loss_last"],
        "train_std_min_last": tr["std_min_last"],
        "theorem_row_sum_beta1_err": th["row_sum_beta1_err"],
        "theorem_row_sum_beta0": th["row_sum_beta0"],
        "action_read_mean": ac["read_mean"],
        "action_blind_max_abs": ac["blind_max_abs"],
        "baseline_untrained": bl["untrained"],
        "baseline_frozen_random": bl["frozen_random"],
        "baseline_trained": bl["trained"],
        "model_trainable_params": rt["n_trainable"],
        "model_total_params": rt["n_params"],
        "roundtrip_worst": rt["worst"],
        "roundtrip_without_mask": m.roundtrip_without_mask(),
    }


def _as_written(text, value):
    """`value` re-formatted in the SHAPE the prose wrote it in.

    A presence guard asks only whether a token appears somewhere in the run, so
    a sentence can be made factually false using a number that is genuinely
    printed but attached to a different quantity. Comparing as strings in the
    prose's own precision is what binds the number to ITS quantity.
    """
    t = text.lstrip("+")
    if "e" in t or "E" in t:
        mant = t.lower().split("e")[0]
        k = len(mant.split(".")[1]) if "." in mant else 0
        return ("%." + str(k) + "e") % float(value)
    if "." in t:
        return ("%." + str(len(t.split(".")[1])) + "f") % float(value)
    return str(int(value))


def _flex(pattern):
    r"""A prose pattern that survives the line wrap it is written across.

    Every literal space becomes \s+, because a docstring sentence breaks wherever
    the column ran out and a pattern anchored to the spaces it happened to have
    when it was written goes red on a reflow rather than on a wrong number. The
    guard is about the NUMBER being right, not about where the line broke.
    """
    return pattern.replace(" ", r"\s+")


def _claims(m, r):
    """(label, regex, live values) for every measured number stated in prose.

    One entry per quantity the module docstring asserts, each compared to the
    live measurement in the prose's own precision. The presence guard cannot do
    this: it only asks whether a token appears in the run, so a sentence rebuilt
    out of a printed token belonging to a different quantity reads as false and
    passes.
    """
    eq = {row["beta"]: row for row in r["equivalence"]["rows"]}
    cl, sg, em, tr = r["collapse"], r["stopgrad"], r["ema"], r["train"]
    co, ac, bl, rt = r["corners"], r["action"], r["baselines"], r["roundtrip"]
    bd, cs = r["baseline_detail"], {row["nu"]: row for row in r["cov_sweep"]}
    br = {row["drift"]: row for row in r["bed_rank"]}
    raw = [
        # -- declared constants of the instrument ---------------------------
        ("the latent axis length", r'axis "D" of length (\d+)', [m.D_LATENT]),
        ("the sequence length and batch",
         r"S = (\d+) positions, B = (\d+) sequences", [m.S_LEN, m.BATCH]),
        ("the training steps", r"(\d+) optimiser steps", [m.N_STEPS]),
        ("the seed", r"seed (\d+)", [m.SEED]),
        ("the EMA momentum", r"tau = (\d+(?:\.\d+)?)", [m.TAU]),
        ("the horizon", r"horizon h = (\d+)", [m.HORIZON]),
        ("the interior beta", r"the interior value (\d+(?:\.\d+)?)",
         [m.BETA_INTERIOR]),
        ("the collapse threshold",
         r"per-coordinate standard deviation below (\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
         [m.COLLAPSE_STD_MIN]),
        ("the rank floor", r"against a floor of (\d+(?:\.\d+)?)",
         [m.COLLAPSE_ERANK_MIN]),
        ("the equivalence tolerance",
         r"tolerance of (\d+(?:\.\d+)?(?:e[+-]?\d+)?) on the read", [m.EQUIV_TOL]),
        ("the assigned and refused counts",
         r"mask (\d+) of (\d+) coordinates are predicted and (\d+) are",
         [len(tr["cols"]), m.D_LATENT, tr["n_refused"]]),
        # -- headline figures ------------------------------------------------
        ("the equivalence at both corners and the interior",
         r"BITWISE EXACT at beta=0, (\d+(?:\.\d+)?e[+-]?\d+) at beta=1 and "
         r"(\d+(?:\.\d+)?e[+-]?\d+) at the interior",
         [eq[1.0]["worst"], eq[m.BETA_INTERIOR]["worst"]]),
        ("the all-ones corner identity",
         r"all-ones mask reproduces the softmax arm to (\d+(?:\.\d+)?e[+-]?\d+)",
         [co["ones_worst"]]),
        ("the collapse detector's two readings",
         r"std_min (\d+(?:\.\d+)?e[+-]?\d+) planted against "
         r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?) healthy",
         [cl["planted"]["std_min"], cl["healthy"]["std_min"]]),
        ("the effective ranks",
         r"effective rank falls from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)",
         [cl["healthy"]["erank"], cl["rank_one"]["erank"]]),
        ("the rank-one plant's surviving spread",
         r"rank-one plant keeps std_min (\d+(?:\.\d+)?) and is missed by the "
         r"variance leg", [cl["rank_one"]["std_min"]]),
        ("the covariance sweep, all three settings",
         r"falls from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) while the variance leg "
         r"reads a",
         [cs[1.0]["erank_first"], cs[1.0]["erank_min"]]),
        ("the step the default weight would be refused at",
         r"would be refused at step (\d+); at nu = (\d+(?:\.\d+)?)",
         [cs[1.0]["collapsed_at"], 25.0]),
        ("the two surviving covariance settings",
         r"the rank bottoms at (\d+(?:\.\d+)?) and at nu = (\d+(?:\.\d+)?) at "
         r"(\d+(?:\.\d+)?)",
         [cs[25.0]["erank_min"], m.NU, cs[m.NU]["erank_min"]]),
        ("the covariance sweep's headline",
         r"at nu = 1.0 the effective rank bottoms at (\d+(?:\.\d+)?) while the "
         r"variance leg reads std_min (\d+(?:\.\d+)?)",
         [cs[1.0]["erank_min"], cs[1.0]["std_min_last"]]),
        ("the bed's own effective rank, both drifts",
         r"drift of (\d+(?:\.\d+)?) the raw observation carries effective rank "
         r"(\d+(?:\.\d+)?) of (\d+)",
         [0.6, br[0.6]["erank"], br[0.6]["n_dims"]]),
        ("the accepted bed", r"at (\d+(?:\.\d+)?) it reads (\d+(?:\.\d+)?)",
         [m.DRIFT, br[m.DRIFT]["erank"]]),
        ("the stop-grad counts, both directions",
         r"(\d+) of (\d+) target parameters carry a gradient with the stop-grad "
         r"in place and (\d+) without it",
         [sg["on_target_grads"], sg["n_target_params"], sg["off_target_grads"]]),
        ("the EMA identity",
         r"tau = 0 reproduces the online encoder to (\d+(?:\.\d+)?) over (\d+) "
         r"tensors", [em["tau0_worst"], em["n_tensors"]]),
        ("the target scales that make the absolute metric a trap",
         r"target RMS runs (\d+(?:\.\d+)?) untrained to (\d+(?:\.\d+)?) trained",
         [bd["untrained"]["target_rms"], bd["trained"]["target_rms"]]),
        ("the three baselines",
         r"untrained (\d+(?:\.\d+)?), frozen random (\d+(?:\.\d+)?), trained "
         r"(\d+(?:\.\d+)?)",
         [bl["untrained"], bl["frozen_random"], bl["trained"]]),
        ("the training loss, first and last",
         r"latent loss runs (\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)",
         [tr["loss_first"], tr["loss_last"]]),
        ("the repriced quantity",
         r"target std_min (\d+(?:\.\d+)?) against the frozen encoder's "
         r"(\d+(?:\.\d+)?)", [tr["std_min_last"], r["init"]["std_min"]]),
        ("the blind control",
         r"action-blind predictor reads exactly (\d+(?:\.\d+)?)",
         [ac["blind_max_abs"]]),
        ("the trainable surface",
         r"weights: (\d+) trainable parameters of (\d+)",
         [rt["n_trainable"], rt["n_params"]]),
        ("the round trip and its own control",
         r"round trip moves the forward output by (\d+(?:\.\d+)?) and dropping "
         r"the mask entry moves it by (\d+(?:\.\d+)?)",
         [rt["worst"], m.roundtrip_without_mask()]),
    ]
    return [(label, _flex(pat), live) for label, pat, live in raw]



def test_every_measured_number_stated_in_prose_is_pinned_to_its_own_quantity(capsys):
    """A WRONG NUMBER BUILT FROM A PRINTED TOKEN PASSES THE PRESENCE GUARD.

    Every token in a sentence can be printed somewhere by the run while the
    sentence itself is false, because presence is all the other guard asks. Each
    such sentence is parsed here and compared to the live measurement in the
    prose's own precision, which binds the number to ITS quantity.
    """
    m = _pj()
    r = m.report()
    doc = m.__doc__ or ""
    bound = set()
    for label, pattern, live in _claims(m, r):
        hit = re.search(pattern, doc)
        assert hit, "%s: no longer in the docstring in a parsable form (%s)" \
            % (label, pattern)
        assert len(hit.groups()) == len(live), \
            "%s: %d groups against %d live values" % (label, len(hit.groups()), len(live))
        for i, (said, value) in enumerate(zip(hit.groups(), live)):
            want = _as_written(said, value)
            assert said.lstrip("+") == want, \
                "%s, position %d: the docstring says %r, the measurement is %r" \
                % (label, i, said, want)
            bound.add(said.lstrip("+"))
    total = _tokens(doc)
    print("\n  %d of %d distinct numbers in the module docstring are pinned to a "
          "named quantity" % (len(bound & total), len(total)))
    print("  the remaining %d are bound by presence in the run only"
          % len(total - bound))
    assert len(bound & total) >= MIN_CLAIM_BOUND, \
        "only %d prose numbers are claim-bound" % len(bound & total)


def test_the_published_values_are_not_rounded_literals(capsys):
    """PRINTED IS NOT MEASURED.

    A literal typed into a print statement satisfies the presence guard
    trivially. Every published value is pinned HERE to an expression that
    recomputes it from the module's own functions in this same run, and every
    vector is compared POSITION BY POSITION so one frozen entry beside live
    neighbours cannot hide inside an aggregate tolerance.

    THE CEILING, STATED RATHER THAN IMPLIED. Equality cannot tell a live
    computation from a constant typed at full precision; what it does bind is
    every ROUNDED literal, which is the shape a copied or stale number takes.
    """
    m = _pj()
    r = m.report()
    m.demo()
    out = capsys.readouterr().out

    published = r["published"]
    assert published, "report() publishes no pinned values"
    mine = _recomputed_published(m, r)
    assert set(mine) == set(published), \
        "published keys %r against the recomputation's %r" \
        % (sorted(set(published) - set(mine)), sorted(set(mine) - set(published)))
    assert len(mine) >= MIN_PUBLISHED, \
        "only %d published values are pinned: the recomputation has shrunk" % len(mine)
    for key in sorted(published):
        assert published[key] == mine[key], \
            "published[%r] = %r, recomputed %r" % (key, published[key], mine[key])

    # Vectors, position by position -- never allclose over a list.
    beta = m.report()["refusal_channel"]["beta"]
    live = m.beta_vector()
    assert len(beta) == len(live) == m.D_LATENT
    for i in range(m.D_LATENT):
        a, b = beta[i], live[i]
        if m.is_refusal(a) or m.is_refusal(b):
            assert m.is_refusal(a) and m.is_refusal(b) and a.code == b.code, \
                "beta position %d: published %r, recomputed %r" % (i, a, b)
        else:
            assert a == b, "beta position %d: published %r, recomputed %r" % (i, a, b)
    for i, (a, b) in enumerate(zip(r["train"]["std_min_trace"],
                                   m.train_run()["std_min_trace"])):
        assert a == b, "std_min trace position %d: %r against %r" % (i, a, b)

    for key, value in published.items():
        assert isinstance(value, (int, float)), "%s is %r" % (key, value)
        txt = str(value) if isinstance(value, int) else None
        assert (txt in out if txt is not None
                else (("%.4f" % value) in out or ("%.6f" % value) in out
                      or ("%.3e" % value) in out or ("%.2e" % value) in out
                      or ("%.2f" % value) in out)), \
            "%s = %r is not printed by the run" % (key, value)
    print("\n  %d published values re-derived against the module in this run"
          % len(published))


def test_the_demo_finishes_inside_the_budget_and_ends_with_the_exact_line(capsys):
    """The bar is 300 s and the last line is fixed by contract.

    The first line is the provenance: the producing commit and the machine id,
    both computed in the run rather than copied into it.
    """
    m = _pj()
    t0 = time.time()
    m.demo()
    dt = time.time() - t0
    out = capsys.readouterr().out.rstrip("\n")
    lines = out.splitlines()
    print("\n  demo() took %.2fs; first line %r" % (dt, lines[0]))
    assert dt < 300.0, "demo() took %.1fs against a 300 s bar" % dt
    assert out.endswith("ALL SELF-CHECKS PASSED"), \
        "the last line is %r" % (lines[-1] if lines else "",)
    prov = m.provenance()
    assert prov["commit"] in lines[0] and prov["machine"] in lines[0], \
        "the first line is not the provenance: %r" % (lines[0],)
    assert re.fullmatch(r"[0-9a-f]{12}", prov["machine"]), \
        "the machine id is not a 12-hex digest: %r" % (prov["machine"],)
    assert prov["commit"] != "UNKNOWN", "the producing commit was not resolved"
    assert "RUN:" in (m.__doc__ or ""), "the module docstring carries no RUN: line"
    assert "NOT" in (m.__doc__ or ""), \
        "the module docstring does not say what this is NOT"
    for name in m.__all__:
        assert hasattr(m, name), "__all__ advertises a missing name %r" % (name,)


def test_the_wall_clock_contributes_no_tokens_to_the_scanned_surface(capsys):
    """A TIMING PRINT IS A LAUNDERING CHANNEL, and it is closed by contract.

    A wall clock printed with a space before its unit enters the "printed by a
    run" token set, which lets an absent-decimal control anywhere in the runtime
    range go red at random and hands a fabricator any plausible small decimal
    for free. Clock readings are printed with NO space before the unit so the
    guard's trailing negative-lookahead swallows them.
    """
    m = _pj()
    m.demo()
    out = capsys.readouterr().out
    timing = [ln for ln in out.splitlines() if "elapsed" in ln and "bar" in ln]
    assert len(timing) == 1, "expected one timing line, found %d" % len(timing)
    toks = _tokens(timing[0])
    print("\n  timing line: %r" % (timing[0].strip(),))
    print("  tokens it contributes: %s" % (sorted(toks) or "none",))
    assert toks == {"300"}, \
        "the timing line contributes %s; only the declared 300 s bar may appear" \
        % (sorted(toks),)
    for sample in ("0.03s", "0.18s", "123.45s", "0.00s"):
        assert _tokens(sample) == set(), "%r still tokenises" % (sample,)
    for sample in ("0.03 s", "0.18 s"):
        assert _tokens(sample), "%r should tokenise: the control is backwards" % (sample,)
