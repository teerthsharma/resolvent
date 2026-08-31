"""RULING 2's code half: `beta` LEARNABLE, INIT 1, LOGGED PER INSTANCE.

WHAT RULING 2 NEEDS FROM A FINISHED RUN, and it is not one number. The card's
trained-model clause must state what was trained and print the final `beta`
distribution; the identity clause cites the `beta = 0` certificate unchanged.
Between those two sits a question a single final value cannot answer: a run
whose `beta` never moved because a bug froze it ends at exactly 1.0, and so does
a run whose `beta` moved and came back. Those two have OPPOSITE meanings for the
card -- the first is a broken experiment, the second is the ruling's "beta pins
at 1" -- so a check that reads only the final value is worthless here.

    test_a_frozen_beta_and_a_beta_that_moved_and_returned_have_the_same_final_value

is the receipt that they are indistinguishable by the final value, and

    test_a_frozen_beta_is_distinguished_from_a_beta_that_moved_and_returned

is the receipt that the logged column plus its gradient tells them apart. Those
two are the load-bearing tests in this file.

GRANULARITY. `ceq/arm_smprime.py::ArmSMPrime.__init__` carries `beta` as ONE
scalar `nn.Parameter` per arm instance, shared across attention heads, and
`ceq/hf/modeling_ceq.py::CEQAttention` carries that same parametrization once
per layer. So "per instance" is PER LAYER and the column is `[n_layers]`.
Widening it to `[n_layers, n_heads]` would be a construction the arm does not
have -- the same argument `V17_ARM_WIRING.md` section 2 makes about the gate
heads -- so it is not built and the granularity is reported instead.

WHICH TESTS PASS BEFORE THE CHANGE, AND WHY THAT IS NOT A DEFECT. The wiring
node already made `beta` an `nn.Parameter` initialised from `SMPRIME_CORNER`, so
the two tests that assert learnability pass at RED and are REGRESSION PINS on
work that already landed rather than new claims. Likewise the three `beta = 0`
corner tests compare against constants measured on the pre-change tree, so they
pass before the change by construction -- a regression guard passes before the
regression. Their planted negatives are separate tests and those were RED.
"""
import hashlib
import json
import math
import os

import pytest
import torch

from ceq import arm_smprime as A
from ceq.hf import modeling_ceq as M
from ceq.hf import train as T
from ceq.hf.configuration_ceq import CEQConfig

SHAPE = dict(vocab_size=32, hidden_size=32, num_hidden_layers=2,
             num_attention_heads=4, max_position_embeddings=16)

#: The (L) bind corner as `ceq/arm_smprime.py`'s docstring states it: `beta = 0`,
#: QK OFF. NOT the corner the LM trains at -- `SMPRIME_CORNER` is `(1, 1, 1)`.
L_CORNER = dict(smp_beta=0.0, smp_qk=0.0, smp_g=1.0)

#: [MEASURED] on the PRE-CHANGE tree (this box, this run): sha256 of the
#: float32 logit bytes at the (L) corner, seed 0 construction, seed 7 input,
#: AT `torch.set_num_threads(1)`. Ruling 2 says the identity clause cites the
#: `beta = 0` certificate UNCHANGED, so this is pinned rather than recomputed.
#:
#: THE THREAD COUNT IS PART OF THE PIN AND WAS FOUND THE HARD WAY. The constant
#: first recorded here was measured at the box default of 20 threads and read
#: `5aff3471...`; the same build at one thread reads `5237c8bd...`. The two
#: differ because a float32 reduction changes order with the thread count, so a
#: BITWISE pin that does not name its thread count is order-dependent on the
#: rest of the suite -- `tests/gate0/test_g03_persist.py` calls
#: `torch.set_num_threads(1)` at MODULE scope, so whether this file's constant
#: matched depended on collection order. One thread is chosen because it is the
#: setting the suite already ends up in and the only one that does not depend on
#: the host's core count. The before/after `torch.equal` proof in
#: `V17_R2_BETA.md` section 4 is unaffected: both sides run in one process.
L_CORNER_LOGITS_SHA = "5237c8bdcb17ca793b00ea5e074eac8bb0fe5ca73a946c57094447bf59823a07"

#: [MEASURED] on the PRE-CHANGE tree: `ceq/arm_smprime.py::label_cell`'s own
#: `beta = 0` cell on `bedm_draw(seed=15, s=8)`. This IS the certificate the
#: card's identity clause cites. `ceq/arm_smprime.py` is not this node's file
#: and was not edited; the pin is what makes that auditable rather than claimed.
LABEL_CELL_RESIDUAL = 1.1102230246251565e-16
LABEL_CELL_BOS_RESIDUAL = 0.0


@pytest.fixture
def one_thread():
    """The bitwise corner pins are taken at ONE thread; see
    `L_CORNER_LOGITS_SHA`. Restored so the loop tests keep the box default."""
    was = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(was)


def _model(**kw):
    torch.manual_seed(0)
    return M.CEQForCausalLM(CEQConfig(operator="smprime", **SHAPE, **kw))


def _ids(seed=7, b=2, s=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, SHAPE["vocab_size"], (b, s), generator=g)


def _logits(model):
    model.eval()
    with torch.no_grad():
        return model(input_ids=_ids()).logits


def _sha(t):
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()


def _betas(model):
    return [layer.self_attn.beta for layer in model.model.layers]


# ------------------------------------------------------- 1. learnable, init 1

def test_beta_is_a_learnable_parameter_at_exactly_one_per_layer():
    """PASSES AT RED. `V17_ARM_WIRING.md` already made the three switches
    `nn.Parameter`s; this pins that they stay so, at exactly the init the
    ruling names."""
    m = _model()
    ident = {id(p) for p in m.parameters()}
    for i, b in enumerate(_betas(m)):
        assert isinstance(b, torch.nn.Parameter), (i, type(b))
        assert b.requires_grad, i
        assert id(b) in ident, "layer {} beta is not in model.parameters()".format(i)
        assert float(b) == 1.0, (i, float(b))
        assert b.shape == (), (i, tuple(b.shape))


def test_beta_receives_a_finite_nonzero_gradient():
    """PASSES AT RED. A parameter with `requires_grad=True` whose gradient is
    identically zero is frozen in every sense that matters to the ruling, so
    learnability is asserted on the GRADIENT and not on the flag."""
    m = _model()
    m.train()
    m(input_ids=_ids(), labels=_ids()).loss.backward()
    for i, b in enumerate(_betas(m)):
        assert b.grad is not None, i
        assert torch.isfinite(b.grad).all(), (i, float(b.grad))
        assert float(b.grad) != 0.0, i


# ------------------------------------------------------------- 2. the column

def test_the_beta_column_is_one_scalar_per_layer_and_tracks_the_parameters():
    m = _model()
    m.train()
    m(input_ids=_ids(), labels=_ids()).loss.backward()
    col = M.beta_column(m)
    n = SHAPE["num_hidden_layers"]
    assert set(col) == {"name", "beta", "grad", "requires_grad"}, sorted(col)
    assert [len(col[k]) for k in col] == [n] * 4, {k: len(v) for k, v in col.items()}
    #: PER-INSTANCE IDENTITY, and it has no head axis: `beta` is one scalar per
    #: layer on the arm, so RULING 2a branch C's "which layers / heads" is
    #: answerable as LAYERS and the name says so.
    assert col["name"] == ["model.layers.{}.self_attn.beta".format(i)
                           for i in range(n)], col["name"]
    assert col["beta"] == [float(b) for b in _betas(m)], col["beta"]
    assert col["grad"] == [float(b.grad) for b in _betas(m)], col["grad"]
    #: it must survive the run record, which is JSON.
    assert json.loads(json.dumps(col)) == col


def test_the_beta_column_is_empty_on_an_operator_that_has_no_beta():
    """Non-degeneracy for the column's length: `[n_layers]` is only a claim
    because a model WITHOUT the arm reads a different length."""
    torch.manual_seed(0)
    m = M.CEQForCausalLM(CEQConfig(operator="sgate", **SHAPE))
    assert M.beta_column(m) == {"name": [], "beta": [], "grad": [],
                                "requires_grad": []}
    assert M.beta_census(m) == []


def test_a_stale_beta_column_is_caught():
    """PLANTED NEGATIVE: a column that caches its first read. The live column
    follows the parameter; the stale one does not."""
    m = _model()
    live_before = M.beta_column(m)["beta"]
    cached = list(live_before)                      # the plant
    with torch.no_grad():
        _betas(m)[0].fill_(0.25)
    live_after = M.beta_column(m)["beta"]
    assert live_after != live_before, live_after
    assert live_after[0] == 0.25, live_after
    assert cached == live_before, "the plant is not stale, so it proves nothing"
    assert cached != live_after


# ------------------------------------- 3. the pinning question, from a series

def _loop(steps, *, lr=0.02, freeze=False, exclude_beta=False, seed=0,
          return_beta_to_init=False, monkeypatch=None, detach=False):
    """A tiny loop whose ONLY output is the logged column series.

    L-LEAN: no loss is returned, no checkpoint written, nothing kept. `lr` is
    large on purpose -- the point is to move a scalar, not to train anything --
    but not larger than the arm survives: at AdamW `lr = 0.5` this shape reads
    `nan` in `beta` by step 3, measured, which is a plumbing fact about the
    probe and NOT a training result.
    """
    if detach:
        monkeypatch.setattr(M.CEQAttention, "_smprime", _smprime_detached_beta)
    torch.manual_seed(seed)
    m = _model()
    m.train()
    if freeze:
        for b in _betas(m):
            b.requires_grad_(False)
    params = [p for n, p in m.named_parameters()
              if not (exclude_beta and n.endswith("self_attn.beta"))]
    opt = torch.optim.AdamW(params, lr=lr)
    ids = _ids()
    series = []
    for _ in range(steps):
        loss = m(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        series.append(M.beta_column(m))
    if return_beta_to_init:
        with torch.no_grad():
            for b in _betas(m):
                b.fill_(1.0)
    series.append(M.beta_column(m))          # the final, unconditional log
    return series


def _smprime_detached_beta(self, q, k, v, x):
    """PLANTED NEGATIVE: `ceq/hf/modeling_ceq.py::CEQAttention._smprime` with
    `beta` detached at the call. `requires_grad` still reads True, the optimizer
    still holds the parameter, and `beta.grad` stays `None` forever."""
    from ceq.arm_smprime import readout
    u = self.m_head(x).squeeze(-1).unsqueeze(-2)
    th = self.theta_head(x).squeeze(-1).unsqueeze(-2)
    return readout(q, k, v, u, th, beta=self.beta.detach(), qk=self.qk,
                   g=self.g).real


def test_the_summary_recovers_the_final_distribution_from_the_series():
    s = _loop(3)
    r = M.beta_summary(s)
    assert r["n_layers"] == SHAPE["num_hidden_layers"], r
    assert r["n_logged"] == 4, r
    assert r["final"] == s[-1]["beta"], r
    assert r["final_min"] == min(r["final"]), r
    assert r["final_max"] == max(r["final"]), r
    assert r["final_min"] <= r["final_median"] <= r["final_max"], r


def test_a_frozen_beta_and_a_beta_that_moved_and_returned_have_the_same_final_value():
    """THE REASON THE OTHER TESTS EXIST. Both runs end at exactly 1.0 in every
    layer, so the final value -- the one number the card would otherwise print
    -- cannot tell a broken experiment from the ruling's `beta` pins at 1."""
    frozen = M.beta_summary(_loop(3, freeze=True))
    returned = M.beta_summary(_loop(3, return_beta_to_init=True))
    assert frozen["final"] == [1.0] * SHAPE["num_hidden_layers"], frozen["final"]
    assert frozen["final"] == returned["final"]
    assert frozen["final_displacement"] == returned["final_displacement"] == 0.0


def test_a_frozen_beta_is_distinguished_from_a_beta_that_moved_and_returned():
    """THE LOAD-BEARING TEST. Two quantities the final value does not carry:
    MOBILITY (did a gradient ever reach `beta`) and DISPLACEMENT (did `beta`'s
    value ever leave its init)."""
    frozen = M.beta_summary(_loop(3, freeze=True))
    returned = M.beta_summary(_loop(3, return_beta_to_init=True))

    assert frozen["mobile"] is False, frozen
    assert frozen["moved"] is False, frozen
    assert frozen["max_abs_grad"] == 0.0, frozen
    assert frozen["max_displacement"] == 0.0, frozen
    assert frozen["state"] == "immobile", frozen

    assert returned["mobile"] is True, returned
    assert returned["moved"] is True, returned
    assert returned["max_abs_grad"] > 0.0, returned
    assert returned["max_displacement"] > 0.0, returned
    assert returned["state"] == "moved", returned


def test_a_detached_beta_gradient_reads_immobile_and_names_itself(monkeypatch):
    """PLANTED NEGATIVE: `beta` reaches the operator detached. `requires_grad`
    is still True and the optimizer still holds it, so the flag alone would say
    the run was fine."""
    r = M.beta_summary(_loop(3, detach=True, monkeypatch=monkeypatch))
    assert r["mobile"] is False, r
    assert r["moved"] is False, r
    assert r["n_missing_grad"] == r["n_logged"] * r["n_layers"], r
    assert r["requires_grad"] is True, "the plant did not clear requires_grad"
    assert r["state"] == "immobile", r


def test_a_beta_the_optimizer_never_updates_is_its_own_state():
    """PLANTED NEGATIVE: `beta` has a live gradient every step and is left out
    of the optimizer. That is neither `immobile` nor `moved`, and reporting it
    as either would hand the card a wrong sentence."""
    r = M.beta_summary(_loop(3, exclude_beta=True))
    assert r["mobile"] is True, r
    assert r["moved"] is False, r
    assert r["max_abs_grad"] > 0.0, r
    assert r["max_displacement"] == 0.0, r
    assert r["state"] == "not_updated", r


def test_the_three_states_are_pairwise_distinct():
    """Non-degeneracy for the discriminator itself: three runs whose final
    `beta` vectors are all `[1.0, 1.0]` land in three different states."""
    runs = {"immobile": _loop(3, freeze=True),
            "not_updated": _loop(3, exclude_beta=True),
            "moved": _loop(3, return_beta_to_init=True)}
    got = {k: M.beta_summary(v) for k, v in runs.items()}
    assert all(g["final"] == [1.0] * SHAPE["num_hidden_layers"]
               for g in got.values()), {k: g["final"] for k, g in got.items()}
    assert {k: g["state"] for k, g in got.items()} == {k: k for k in runs}


def test_beta_actually_moves_on_an_ordinary_run():
    """The must-fire the other three are read against: nothing frozen, nothing
    excluded, nothing detached, and `beta` leaves 1.0. Reports that it moved and
    by how much; reports NO loss, NO cell, NO verdict (L-LEAN)."""
    r = M.beta_summary(_loop(3))
    assert r["mobile"] is True, r
    assert r["moved"] is True, r
    assert r["final_displacement"] > 0.0, r
    assert r["state"] == "moved", r


# --------------------------------------- 3a. RULING 2a: the census and PINNED

def _census_loop(steps, **kw):
    """The census beside the column, from one loop, so the two are read off the
    same run the way `train()` will read them."""
    if kw.pop("detach", False):
        kw["monkeypatch"].setattr(M.CEQAttention, "_smprime", _smprime_detached_beta)
    kw.pop("monkeypatch", None)
    torch.manual_seed(0)
    m = _model()
    m.train()
    if kw.pop("freeze", False):
        for b in _betas(m):
            b.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=0.02)
    ids = _ids()
    census, per_step, series = [], [], []
    for _ in range(steps):
        loss = m(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        per_step.append([0.0 if b.grad is None else abs(float(b.grad))
                         for b in _betas(m)])
        M.beta_census(m, census)             # after backward, before any clip
        opt.step()
        series.append(M.beta_column(m))
    return series, census, per_step


def test_the_census_is_the_integral_of_the_gradient_over_every_step():
    """RULING 2a's GRADIENT CENSUS. Checked against the per-step gradients
    summed by the test itself, not against a second call of the same code."""
    _, census, per_step = _census_loop(3)
    n = SHAPE["num_hidden_layers"]
    assert len(census) == n, census
    for i in range(n):
        assert census[i] == pytest.approx(sum(s[i] for s in per_step), rel=0, abs=0)
        assert census[i] > 0.0, (i, census)


def test_the_census_grows_and_a_one_step_census_is_not_a_three_step_one():
    """Non-degeneracy for the integral: a census that ignored steps would read
    the same at 1 step and at 3."""
    _, c1, _ = _census_loop(1)
    _, c3, _ = _census_loop(3)
    assert all(b > a for a, b in zip(c1, c3)), (c1, c3)


def test_a_detached_beta_has_a_census_of_exactly_zero(monkeypatch):
    """PINNED-WITHOUT-SIGNAL's witness, and the plant that proves it fires: the
    dial was never exercised, so the integral is exactly 0.0."""
    _, census, _ = _census_loop(3, detach=True, monkeypatch=monkeypatch)
    assert census == [0.0] * SHAPE["num_hidden_layers"], census


def test_a_frozen_beta_has_a_census_of_exactly_zero():
    _, census, _ = _census_loop(3, freeze=True)
    assert census == [0.0] * SHAPE["num_hidden_layers"], census


def test_the_column_name_is_the_key_the_floor_node_will_hand_back():
    """THE CROSS-NODE JOIN, pinned. `scripts/k_noise_floor.py::
    per_parameter_deltas` keys `delta_beta` by `named_parameters()` name, and
    RULING 2a's criterion joins the two per parameter. A synthesized name that
    drifted from the real one would join to nothing and every beta would come
    back unpinnable -- silently, since a missing key is a KeyError only if
    somebody indexes rather than `.get`s."""
    m = _model()
    assert M.beta_column(m)["name"] == [
        n for n, _ in m.named_parameters() if n.endswith("self_attn.beta")]


def _series(final):
    """One logged column carrying `final`, enough for the criterion's arithmetic."""
    n = len(final)
    return [{"name": ["model.layers.{}.self_attn.beta".format(i) for i in range(n)],
             "beta": list(final), "grad": [1e-3] * n, "requires_grad": [True] * n}]


def test_the_criterion_is_five_delta_beta_and_the_boundary_is_inclusive():
    """RULING 2a: PINNED iff `|beta - 1| <= 5 * delta_beta`. Checked ON the
    boundary and one step either side, so the constant is the thing measured and
    not a direction."""
    #: POWERS OF TWO, so `5 * delta` and `1 +/- 5 * delta` are exact in binary
    #: and the boundary is the criterion's rather than the decimal literal's.
    #: With `delta = 1e-3` the minus side reads `0.0050000000000000044 > 0.005`
    #: and the test would measure float64 rounding.
    d = [2.0 ** -6, 2.0 ** -6]
    tol = 5 * d[0]
    assert 1.0 + tol - 1.0 == tol and 1.0 - (1.0 - tol) == tol
    r = M.beta_summary(_series([1.0 + tol, 1.0 - tol]), delta_beta=d)
    assert r["pinned"] == [True, True], r
    #: one ulp OF BETA, not of the tolerance: `1.0 + nextafter(tol, 1)` rounds
    #: straight back to `1.0 + tol`, because ulp(1.078) >> ulp(0.078).
    out = math.nextafter(1.0 + tol, 2.0)
    assert out - 1.0 > tol
    r = M.beta_summary(_series([out, 1.0 - tol]), delta_beta=d)
    assert r["pinned"] == [False, True], r
    #: and `k` is a parameter, not a literal buried in a comparison
    assert M.beta_summary(_series([1.0 + tol, 1.0]), delta_beta=d,
                          k=1.0)["pinned"] == [False, True]


def test_the_three_branches_are_reachable_and_the_thresholds_are_not_a_dichotomy():
    """RULING 2a's own correction: two branches were a FALSE DICHOTOMY. All
    three must be reachable, and MIXED must not be swallowed by either."""
    d = [1e-3] * 20
    allp = M.beta_summary(_series([1.0] * 20), delta_beta=d)
    none = M.beta_summary(_series([2.0] * 20), delta_beta=d)
    half = M.beta_summary(_series([1.0] * 10 + [2.0] * 10), delta_beta=d)
    assert (allp["branch"], none["branch"], half["branch"]) == ("A", "B", "C")
    assert half["pinned_fraction"] == 0.5, half
    #: one instance in twenty is 5 %, which is branch B's boundary and not C's
    edge = M.beta_summary(_series([1.0] + [2.0] * 19), delta_beta=d)
    assert (edge["pinned_fraction"], edge["branch"]) == (0.05, "B"), edge


def test_branch_c_names_where_the_moved_betas_live():
    """Branch C's whole content. A flattened distribution cannot answer it."""
    r = M.beta_summary(_series([1.0, 2.0, 1.0, 2.0]), delta_beta=[1e-3] * 4)
    assert r["branch"] == "C", r
    assert r["moved_names"] == ["model.layers.1.self_attn.beta",
                                "model.layers.3.self_attn.beta"], r["moved_names"]


def test_a_bitwise_identical_seed_pair_makes_the_criterion_degenerate():
    """REPORTED, NOT PAPERED OVER. `delta_beta = 0` sets the tolerance to 0, so
    PINNED collapses to `beta` being EXACTLY 1.0 and an ordinary run lands in
    branch B for a reason about the PAIR's determinism, not about training."""
    r = M.beta_summary(_series([1.0 + 1e-9, 1.0]), delta_beta=[0.0, 0.0])
    assert r["n_degenerate_floor"] == 2, r
    assert r["pinned"] == [False, True], r


def test_the_census_word_is_the_weaker_one_unless_the_comparison_group_exists():
    """"Preferred" is COMPARATIVE, so it needs moved betas to compare against.
    With none, the word stays "unused" -- which is branch A's own sentence, so
    this rule can only refuse to upgrade a card sentence, never upgrade one."""
    d = [1e-3] * 4
    allp = M.beta_summary(_series([1.0] * 4), delta_beta=d, census=[9.0] * 4)
    assert allp["branch"] == "A" and allp["word"] == "unused", allp

    #: pinned betas that were exercised harder than the median moved one
    strong = M.beta_summary(_series([1.0, 1.0, 2.0, 2.0]), delta_beta=d,
                            census=[9.0, 9.0, 1.0, 1.0])
    assert strong["word"] == "preferred", strong

    #: one pinned beta that was never exercised drags the word back down
    weak = M.beta_summary(_series([1.0, 1.0, 2.0, 2.0]), delta_beta=d,
                          census=[9.0, 0.0, 1.0, 1.0])
    assert weak["word"] == "unused", weak
    assert weak["signal"] == [True, False, True, True], weak


def test_without_delta_beta_the_summary_refuses_to_branch():
    """The criterion needs a measurement this node does not own. Absent it, the
    summary says so rather than defaulting to a branch."""
    r = M.beta_summary(_series([1.0, 1.0]))
    assert r["branch"] is None and r["pinned"] is None, r
    assert "k_noise_floor" in r["branch_reason"], r["branch_reason"]


# ------------------------------- 3b. the criterion: is beta 1 where it counts

def test_the_substitution_delta_is_exactly_zero_when_beta_is_already_one():
    """MUST-NOT-FIRE. Substituting 1.0 for a `beta` that IS 1.0 rewrites the
    same bytes, so the second forward is bitwise the first and `delta` is
    exactly `0.0` -- not `< eps`, `0.0`."""
    m = _model()
    r = M.beta_substitution(m, input_ids=_ids(), labels=_ids())
    assert r["beta"] == [1.0] * SHAPE["num_hidden_layers"], r
    assert r["delta"] == 0.0, r


def test_the_substitution_delta_is_nonzero_when_beta_left_one():
    """MUST-FIRE, and one layer is enough: the card's sentence is about the
    OBJECT, so a single off-corner layer must show."""
    m = _model()
    with torch.no_grad():
        _betas(m)[1].fill_(0.9)
    r = M.beta_substitution(m, input_ids=_ids(), labels=_ids())
    assert r["beta"] == [1.0, float(torch.tensor(0.9))], r
    assert r["delta"] > 0.0, r


def test_the_substitution_restores_beta_on_the_way_out():
    m = _model()
    with torch.no_grad():
        _betas(m)[0].fill_(0.75)
    M.beta_substitution(m, input_ids=_ids(), labels=_ids())
    assert [float(b) for b in _betas(m)] == [float(torch.tensor(0.75)), 1.0]


def test_the_substitution_is_blind_to_whether_beta_could_move():
    """WHY THE CRITERION IS A CONJUNCTION AND NOT THIS TEST ALONE. A run whose
    `beta` was frozen by a bug ends at 1.0, so the substitution reads `0.0` and
    would call it pinned. Admissibility -- read off the logged column, not off
    the checkpoint -- is what refuses that run."""
    frozen = _loop(3, freeze=True)
    assert M.beta_summary(frozen)["state"] == "immobile"
    m = _model()
    for b in _betas(m):
        b.requires_grad_(False)
    assert M.beta_substitution(m, input_ids=_ids(), labels=_ids())["delta"] == 0.0


# --------------------------------------------- 4. the beta = 0 corner, intact

def test_the_beta_zero_corner_logits_are_the_pre_change_bytes(one_thread):
    """PASSES AT RED by construction -- pinned to a constant measured on the
    pre-change tree. Ruling 2: the identity clause cites the `beta = 0`
    certificate unchanged, so nothing done at `beta = 1` may move it."""
    assert _sha(_logits(_model(**L_CORNER))) == L_CORNER_LOGITS_SHA


def test_the_beta_zero_corner_comparison_is_not_blind_to_beta(one_thread):
    """PLANTED NEGATIVE, and the one that matters. This repository has struck a
    corner certificate read through a quantity in which the certified switch
    ALGEBRAICALLY CANCELS. `Z^0 = 1` for every `Z`, so the suspicion is exact
    here: does the corner's read-out depend on `beta` at all?

    It does. `d/dbeta Z^beta = Z^beta log Z`, which is `log Z != 0` AT
    `beta = 0`, so the corner is a point where `beta` is INACTIVE, not one where
    it is absent -- and moving it by `1e-6` moves the logits."""
    base = _sha(_logits(_model(**L_CORNER)))
    for eps in (1e-6, 1e-4):
        moved = dict(L_CORNER, smp_beta=eps)
        assert _sha(_logits(_model(**moved))) != base, eps


def test_the_beta_zero_corner_comparison_catches_one_ulp(one_thread):
    """PLANTED NEGATIVE: one float32 ulp on `m_head.weight`, the arm's own
    magnitude head, which the (L) corner reads through the path product.

    THE PLANT WAS MOVED ONCE AND THE MEASUREMENT IS IN `V17_R2_BETA.md`
    section 5. `m_head.bias` is NOT on target -- `_init_weights` zeroes every
    `nn.Linear` bias, so one ulp above `0.0` is a denormal and 64 of them do not
    move the fp32 logits. `qkv.weight` is not on target either, and for a
    load-bearing reason: at `qk = 0` the content term is deleted EXACTLY, so the
    q/k projections are dead weights at this corner. `m_head.weight` is caught at
    1 ulp, `max |delta| = 5.960e-08` at one thread."""
    m = _model(**L_CORNER)
    before = _sha(_logits(m))
    assert before == L_CORNER_LOGITS_SHA
    with torch.no_grad():
        w = m.model.layers[0].self_attn.m_head.weight
        w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(float("inf")))
    assert _sha(_logits(m)) != before


def test_the_label_cell_beta_zero_certificate_is_unchanged():
    """The certificate the card's identity clause actually cites lives in
    `ceq/arm_smprime.py::label_cell`, which is NOT this node's file. Pinned to
    the reading taken on the pre-change tree."""
    a, b = A.bedm_draw(seed=15, s=8)
    rec = A.label_cell(a, b)
    assert rec["beta"] == 0.0, rec["beta"]
    assert rec["qk"] == 0.0, rec["qk"]
    assert rec["residual"] == LABEL_CELL_RESIDUAL, repr(rec["residual"])
    assert rec["residual_bos"] == LABEL_CELL_BOS_RESIDUAL, repr(rec["residual_bos"])


def test_the_label_cell_certificate_pin_catches_a_perturbed_draw():
    """Non-degeneracy for the pin above: a draw whose gates are perturbed off
    BED-M's support does not read the certificate's residual."""
    a, b = A.bedm_draw(seed=15, s=8)
    a = a.clone()
    a[2] = a[2] + 1e-3
    assert A.label_cell(a, b)["residual"] != LABEL_CELL_RESIDUAL


# ------------------------------------------- 5. the interface `train()` needs

#: WAS a strict xfail, held open by the beta node because `ceq/hf/train.py`
#: is not its file. The seam is landed there now. The strict marker did its
#: job -- it became a loud XPASS the moment the patch went in, which is how
#: this came off the list rather than by anyone remembering it.
def test_the_run_record_carries_the_beta_column(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("the quick brown fox jumps over the lazy dog. " * 400,
                 encoding="utf-8")
    out = str(tmp_path / "smp")
    torch.manual_seed(0)
    rec = T.train(out_dir=out, steps=3, batch=2, seq=16, hidden_size=32,
                  n_layers=2, n_heads=4, vocab_size=256, device="cpu", lr=1e-3,
                  log_every=1, data_path=str(p), operator="smprime")
    with open(os.path.join(out, "run_record.json"), encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert "beta" in rec, sorted(rec)
    assert "beta_census" in rec, sorted(rec)
    assert on_disk["beta"] == rec["beta"]
    assert on_disk["beta_census"] == rec["beta_census"]
    #: the trajectory is sampled at `log_every`; the census is an INTEGRAL and
    #: must have seen every step, so its length is n_layers and not n_logged.
    assert len(rec["beta_census"]) == 2, rec["beta_census"]
    r = M.beta_summary(rec["beta"], census=rec["beta_census"])
    assert r["n_layers"] == 2, r
    assert r["state"] in ("immobile", "not_updated", "moved"), r
    assert r["branch"] is None, "delta_beta comes from the floor node, not here"


#: WAS a strict xfail, held open by the beta node because `ceq/hf/train.py`
#: is not its file. The seam is landed there now. The strict marker did its
#: job -- it became a loud XPASS the moment the patch went in, which is how
#: this came off the list rather than by anyone remembering it.
def test_build_forwards_the_smprime_switches():
    m = T.build(hidden_size=32, n_layers=2, n_heads=4, seq=16, vocab_size=32,
                operator="smprime", smp_beta=0.0, smp_qk=0.0)
    assert m.config.smp_beta == 0.0, m.config.smp_beta
    assert float(m.model.layers[0].self_attn.beta) == 0.0
