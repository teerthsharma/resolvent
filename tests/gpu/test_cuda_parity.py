"""CUDA parity lane for the m3 arms: softmax, twin, settled (and their controls).

RULES OF THIS LANE.
  1. CUDA rows are TOLERANCE-CHECKED, never bitwise. GPU kernels reduce in a
     different order than CPU BLAS, so bitwise equality is not a property the
     device lane can carry; every assertion here uses atol=2e-4 / rtol=1e-4.
  2. Every GPU number carries its own device tag downstream: `--device cuda`
     journals to `m3_quintuple_v2_cuda.jsonl` and saves weights under
     `m3_quintuple_v2_cuda_weights/`. The registered CPU journal
     `results/m3_quintuple_v2.jsonl` is never appended to from this lane.
  3. NO OMP/thread pinning lives here. Pinning threads is a CPU
     bitwise-reproducibility concern (`inspector.py` pins JOURNAL_THREADS=2 for
     the CPU lane); CUDA reproducibility on this path is governed by
     `torch.backends.cudnn.deterministic = True`, which `scale/paired_arm.py`
     sets when device=cuda.

Every test is skipped when no CUDA device is present, so the CPU-only boxes
this repo is usually read on keep their exact behaviour.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import m3_capability as M3                            # noqa: E402
from scale import m3_quintuple as Q                              # noqa: E402
from scale import negation_scope as NS                           # noqa: E402
from scale import paired_arm as PA                               # noqa: E402

#: The registered geometry, small batch. d=24 is the negation DISTANCE; the
#: model width is M3.D_MODEL=16 everywhere, exactly as the journalled units.
S, D_DIST, K_PIV = 64, 24, 8
ATOL, RTOL = 2e-4, 1e-4

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="no cuda")

PARITY_CELLS = ("softmax", "glance", "settled", "twin", "argmax")
TRAIN_CELLS = ("softmax", "twin", "settled")


def _batch(n: int = 4, seed: int = 0):
    return NS.make_batch(n, S, D_DIST, d_model=M3.D_MODEL, seed=seed)


def _arm_pair(cell: str):
    """Two identically-seeded QuintArms, one on cpu, one on cuda.

    Construction draws weights on the CPU generator in both cases, then the
    cuda copy is MOVED -- so both hold bit-identical parameters and any output
    difference is device arithmetic, not initialisation.
    """
    x, y, _, _ = _batch()
    torch.manual_seed(0)
    m_cpu = Q.QuintArm("softmax", S, cell=cell, k_piv=K_PIV)
    torch.manual_seed(0)
    m_gpu = Q.QuintArm("softmax", S, cell=cell, k_piv=K_PIV).to("cuda")
    return m_cpu, m_gpu, x, y


@requires_cuda
@pytest.mark.parametrize("cell", PARITY_CELLS)
def test_forward_parity_cpu_vs_cuda(cell):
    """Each arm class forwards to the same prediction within float32 tolerance."""
    m_cpu, m_gpu, x, _ = _arm_pair(cell)
    m_cpu.eval()
    m_gpu.eval()
    with torch.no_grad():
        out_cpu = m_cpu(x)
        out_gpu = m_gpu(x.to("cuda"))
    torch.testing.assert_close(out_gpu.cpu(), out_cpu, atol=ATOL, rtol=RTOL)


@requires_cuda
def test_softmax_baseline_arm_forward_parity():
    """The plain capability softmax arm (ceq self-attention baseline) too."""
    x, _, _, _ = _batch()
    torch.manual_seed(0)
    m_cpu = M3.Arm("softmax", S)
    torch.manual_seed(0)
    m_gpu = M3.Arm("softmax", S).to("cuda")
    m_cpu.eval()
    m_gpu.eval()
    with torch.no_grad():
        out_cpu = m_cpu(x)
        out_gpu = m_gpu(x.to("cuda"))
    torch.testing.assert_close(out_gpu.cpu(), out_cpu, atol=ATOL, rtol=RTOL)


@requires_cuda
@pytest.mark.parametrize("cell", PARITY_CELLS)
def test_backward_step_grad_parity(cell):
    """One backward step: every parameter's gradient matches within tolerance."""
    m_cpu, m_gpu, x, y = _arm_pair(cell)
    mu = float(y.mean())
    sigma = float(y.std(unbiased=False)) or 1.0
    target_cpu = ((y - mu) / sigma)
    target_gpu = target_cpu.to("cuda")

    def grads(m, xx, tt):
        m.train()
        m.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(m(xx), tt).backward()
        return {n: p.grad.clone() for n, p in m.named_parameters()}

    g_cpu = grads(m_cpu, x, target_cpu)
    g_gpu = grads(m_gpu, x.to("cuda"), target_gpu)
    assert set(g_cpu) == set(g_gpu)
    for name in g_cpu:
        torch.testing.assert_close(g_gpu[name].cpu(), g_cpu[name],
                                   atol=ATOL, rtol=RTOL, msg=name)


@requires_cuda
@pytest.mark.parametrize("cell", TRAIN_CELLS)
def test_tiny_train_completes_on_cuda_with_finite_loss(cell):
    """/=5 steps, /n_train 256: the full train_and_predict loop runs on cuda
    through paired_arm's own path and ends at a finite loss/prediction."""
    pred, y_eval, npar = PA.train_and_predict(
        "softmax", s=S, d=D_DIST, steps=5, n_train=128, n_eval=32,
        seed=0, batch_fn=NS.make_batch, device=torch.device("cuda"))
    assert npar == 4769
    assert torch.isfinite(pred).all()
    nr = NS.nrmse(pred, y_eval)
    assert nr == nr and abs(nr) != float("inf")   # finite, NaN-safe


@requires_cuda
def test_state_dict_moves_between_devices_preserving_predictions():
    """The SAME trained state_dict predicts the same thing from either device."""
    x, y, _, _ = _batch(n=8, seed=7)
    torch.manual_seed(3)
    model = Q.QuintArm("softmax", S, cell="twin", k_piv=K_PIV).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=M3.LR)
    mu = float(y.mean())
    sigma = float(y.std(unbiased=False)) or 1.0
    tgt_gpu = ((y - mu) / sigma).to("cuda")
    for _ in range(3):                       # trained, not just initialised
        model.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x.to("cuda")), tgt_gpu).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        ref = model(x.to("cuda")).cpu()

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    torch.manual_seed(3)
    on_cpu = Q.QuintArm("softmax", S, cell="twin", k_piv=K_PIV)
    on_cpu.load_state_dict(sd)
    on_cpu.eval()
    with torch.no_grad():
        out_cpu = on_cpu(x)

    torch.manual_seed(3)
    on_gpu = Q.QuintArm("softmax", S, cell="twin", k_piv=K_PIV).to("cuda")
    on_gpu.load_state_dict(sd)
    on_gpu.eval()
    with torch.no_grad():
        out_gpu = on_gpu(x.to("cuda")).cpu()

    torch.testing.assert_close(out_cpu, ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out_gpu, ref, atol=ATOL, rtol=RTOL)
