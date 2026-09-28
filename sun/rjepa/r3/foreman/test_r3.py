# A6 unit tests (RED before r3.py exists). python -m pytest test_r3.py -q
import torch
import r3
from r3 import r2


def test_tiebreak_uses_true_posterior_not_learned_distance():
    Pb = torch.tensor([[0.5, 0.5, 0.0]])
    d_learned = torch.tensor([[1.0, 2.0, 3.0]])                 # learned predictor prefers k = 0
    Ed = torch.tensor([[3.0, 1.0, 2.0]])                        # true posterior mean distance prefers k = 1
    assert int((Pb - 1e-9 * d_learned).argmax(1)) == 0         # the r2 rule: defect documented
    assert int(r3.bayes_pick(Pb, Ed)) == 1


def test_tiebreak_restricted_pick():
    Pb = torch.tensor([[0.9, 0.3, 0.3, 0.1]])
    Ed = torch.tensor([[0.1, 2.0, 1.0, 3.0]])
    allowed = torch.tensor([[False, True, True, True]])
    assert int(r3.bayes_pick(Pb, Ed, allowed)) == 2
    assert int(r3.bayes_pick(Pb, Ed)) == 0                      # no tie: argmax untouched


def test_bayes_P3_reproduces_r2_draws():
    dyn = r2.make_dyn()
    b = r2.make_cell(32, 0.5, 0.4, 7, dyn)
    Pb, Ps = r2.bayes_P(b, dyn, 0.4, 0.5, 1.5, M=64, seed=5)
    Pb3, Ps3, Ed = r3.bayes_P3(b, dyn, 0.4, 0.5, 1.5, M=64, seed=5)
    assert torch.equal(Pb, Pb3) and torch.equal(Ps, Ps3) and Ed.shape == Pb.shape and bool((Ed > 0).all())


def test_true_dynamics_lin1_feats_are_noiseless_truth():
    dyn = r2.make_dyn()
    b = r2.make_cell(200, 0.0, 1e-4, 11, dyn)
    d, s = r3.true_feats(b, dyn, 1e-4)
    z = r2.rollout(lambda z, u: r2.true_step(z, u, dyn), b["y"], b["a"])
    assert torch.allclose(d, (z - b["g"][:, None]).norm(dim=-1), atol=1e-5)
    assert r2.hit(d.argmin(1), b) >= 0.99 and bool((s > 0).all())
