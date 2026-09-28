# A7 contract tests T1-T4 (RED before r4.py exists). python -m pytest test_r4.py -q
import json, math, re
import torch
import r4
from r4 import r2


def test_T1_calibration_reproduces_stored_sigmas():
    stored = json.loads((r4.R2OUT / "calib.json").read_text())["sigma"]
    c = r4.calib2(r2.make_dyn())
    assert abs(c["sigma"]["1.0"] - stored["1.0"]) <= 1e-6 and abs(c["sigma"]["3.0"] - stored["3.0"]) <= 1e-6
    assert stored["1.0"] < c["sigma"]["2.0"] < stored["3.0"]


def test_T2_jacobian_matches_finite_differences():
    dyn = r2.make_dyn()
    step = lambda z, u: r2.true_step(z, u, dyn)
    b = r2.make_cell(4, 0.0, 0.3, 3, dyn)
    y, a = b["y"].double(), b["a"][:, :5].double()
    dyn64 = {k: v.double() for k, v in dyn.items()}
    step64 = lambda z, u: r2.true_step(z, u, dyn64)
    z, J = r4.jac(step64, y, a)
    h, fd = 1e-5, torch.zeros_like(J)
    for j in range(r2.D):
        e = torch.zeros(r2.D, dtype=y.dtype, device=y.device); e[j] = h
        yp = (y + e)[:, None].expand(-1, 5, -1); ym = (y - e)[:, None].expand(-1, 5, -1)
        fd[..., j] = (r2.rollout(step64, yp, a) - r2.rollout(step64, ym, a)) / (2 * h)
    assert torch.allclose(z, r2.rollout(step, b["y"][:, None].expand(-1, 5, -1), b["a"][:, :5]).double(), atol=1e-4)
    assert float((J - fd).norm() / fd.norm()) <= 1e-3


def test_T3_lin2_exact_in_linear_limit_and_lin1_is_not():
    torch.manual_seed(0)
    dev = r2.DEV
    Qm, _ = torch.linalg.qr(torch.randn(r2.D, r2.D))
    Mm = (Qm * torch.tensor([2.2, 1.6, 1.0, 1.0, 0.6, 0.4, 0.3, 0.2])) @ Qm.T
    Bm = torch.randn(r2.D, r2.M_ACT) / 2
    Mm, Bm = Mm.to(dev), Bm.to(dev)
    step = lambda z, u: z @ Mm.T + u @ Bm.T                                       # linear: LIN2's Gaussian is exact
    n, Kc, sigma = 10, 40, 0.15
    y = torch.randn(n, r2.D, device=dev)
    a = torch.randn(n, Kc, r2.H, r2.M_ACT, device=dev)
    g = r2.rollout(step, y, a[:, :1])[:, 0] + 0.5 * torch.randn(n, r2.D, device=dev)
    z0 = r2.rollout(step, y, a)
    r = float((z0 - g[:, None]).norm(dim=-1).median())
    mc = torch.zeros(n, Kc, device=dev)
    for _ in range(40):                                                           # 40 x 1024 = 40,960 draws per candidate
        e = torch.randn(n, 1024, Kc, r2.D, device=dev)
        zz = r2.rollout(step, y[:, None, None] - sigma * e, a[:, None].expand(-1, 1024, -1, -1, -1))
        mc += ((zz - g[:, None, None]).norm(dim=-1) < r).float().mean(1) / 40
    _, J = r4.jac(step, y, a)
    p2 = r4.lin2_prob(z0 - g[:, None], J, sigma, r)
    d, s = r2.lin1_feats(step, y, a, g, sigma)
    p1 = r2.lin1_prob(d, s, r)
    assert float((p2 - mc).abs().max()) <= 0.01
    assert float((p1 - mc).abs().max()) >= 0.05                                   # the control discriminates


def test_T4_provenance_has_five_hashes():
    p = r4.prov()
    assert set(p) == {"sha256_r4_py", "sha256_r3_py", "sha256_r2_py", "sha256_bar_r4", "sha256_bar_r2"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in p.values())
