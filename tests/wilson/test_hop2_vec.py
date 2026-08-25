"""BITWISE bind: the vectorised batched hop-2 against the Python loop it replaces.

The loop under test is `scale/m3_capability.py::Arm.forward`:

    hop2 = torch.stack([pivot_hop2(a[i], select_pivots(k[i], self.k_pivots))
                        for i in range(n)], dim=0)

`pivot_hop2` and `select_pivots` are IMPORTED from `scale/pivot_probe.py`, never
reimplemented here -- a bind that rebuilds both sides proves only that two
reimplementations agree, which is how this project published 89,400.180 against
1.667.

The comparison is `torch.equal`. NOT `allclose`. A tolerance here would hide
exactly the defect class that matters: a faster path that moves a number is a
new arm, not an optimisation.

The operators are the SHIPPED ones -- `bench._causal_sgate_operator` at the
arm's rho=1.5/lam=0.10 (pivot_signed) and `bench._softmax_operator`
(pivot_unsigned) -- not random matrices, so the bind exercises the sparsity and
sign pattern the real path sees.

THREADS. `torch.set_num_threads(1)`. CPU matmul reduction order varies with the
thread pool (`inspector.py:JOURNAL_THREADS` records the same fact for the
journal replay), so a bitwise claim is a claim AT A THREAD COUNT. Pinning it
also keeps this off the cores the n_train=8192 measurement is using.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ceq import bench
from scale.pivot_probe import select_pivots, pivot_hop2
from tests.wilson.hop2_vec import batched_pivot_hop2, batched_select_pivots

torch.set_num_threads(1)

S = 64          # matches the running measurement: --s 64
D_MODEL = 16    # m3_capability.D_MODEL
K_PIVOTS = 8    # m3_capability.K_PIVOTS
SGATE_RHO, SGATE_LAM = 1.5, 0.10


def make_ak(n: int, seed: int, kind: str, s: int = S):
    """(a, k) the way `Arm.forward` produces them: q/k from a linear map on x,
    then the arm's operator. `a` is [n,s,s], `k` is [n,s,D_MODEL]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, s, D_MODEL, generator=g)
    wq = torch.randn(D_MODEL, D_MODEL, generator=g) / D_MODEL ** 0.5
    wk = torch.randn(D_MODEL, D_MODEL, generator=g) / D_MODEL ** 0.5
    q, k = x @ wq, x @ wk
    if kind == "pivot_unsigned":
        a = bench._softmax_operator(q, k)
    else:
        a = bench._causal_sgate_operator(q, k, rho=SGATE_RHO, lam=SGATE_LAM,
                                         window=0)
    return a, k


def loop_hop2(a, k, k_pivots=K_PIVOTS):
    """The exact expression `Arm.forward` runs today."""
    n = a.shape[0]
    return torch.stack(
        [pivot_hop2(a[i], select_pivots(k[i], k_pivots)) for i in range(n)],
        dim=0,
    )


CASES = [(n, seed, kind)
         for n in (1, 3, 16, 64)
         for seed in (0, 1, 7, 12345)
         for kind in ("pivot_signed", "pivot_unsigned")]


@pytest.mark.parametrize("n,seed,kind", CASES)
def test_pivots_bitwise(n, seed, kind):
    """The pivot INDICES must match first. A hop-2 that matches while routing
    through different pivots would be a coincidence, not equivalence."""
    _, k = make_ak(n, seed, kind)
    want = torch.stack([select_pivots(k[i], K_PIVOTS) for i in range(n)], dim=0)
    got = batched_select_pivots(k, K_PIVOTS)
    assert torch.equal(got, want), (
        f"pivot indices differ n={n} seed={seed} {kind}: "
        f"{(got != want).sum().item()} of {want.numel()} entries"
    )


@pytest.mark.parametrize("n,seed,kind", CASES)
def test_hop2_bitwise(n, seed, kind):
    a, k = make_ak(n, seed, kind)
    want = loop_hop2(a, k)
    got = batched_pivot_hop2(a, batched_select_pivots(k, K_PIVOTS))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(got, want), (
        f"NOT BITWISE n={n} seed={seed} {kind}: "
        f"max abs diff = {float((got - want).abs().max()):.6e}, "
        f"{int((got != want).sum())} of {want.numel()} entries differ"
    )


@pytest.mark.parametrize("kind", ["pivot_signed", "pivot_unsigned"])
def test_grad_bitwise(kind):
    """Forward equality is not enough: the loop is inside a TRAINING step, so
    the gradient that reaches `a` must match bitwise too."""
    a, k = make_ak(8, 3, kind)
    a1 = a.clone().requires_grad_(True)
    a2 = a.clone().requires_grad_(True)
    w = torch.randn(a.shape, generator=torch.Generator().manual_seed(99))
    (loop_hop2(a1, k) * w).sum().backward()
    (batched_pivot_hop2(a2, batched_select_pivots(k, K_PIVOTS)) * w).sum().backward()
    assert torch.equal(a2.grad, a1.grad), (
        f"NOT BITWISE (grad) {kind}: "
        f"max abs diff = {float((a2.grad - a1.grad).abs().max()):.6e}"
    )


def test_control_bind_can_fail():
    """MUST-FIRE CONTROL. `torch.equal` between these two tensors must return
    False for a single perturbed entry -- a bind that cannot fail is not a bind
    (instrument #15). Without this, all of the above pass on a broken assert."""
    a, k = make_ak(4, 0, "pivot_signed")
    want = loop_hop2(a, k)
    bad = a.clone()
    bad[0, S - 1, 0] += 1e-6
    got = batched_pivot_hop2(bad, batched_select_pivots(k, K_PIVOTS))
    assert not torch.equal(got, want), "the bind cannot distinguish a perturbed a"
    # and the pivot bind must also be able to fail
    kbad = k.clone()
    kbad[0, 0, :] *= 1e3
    assert not torch.equal(batched_select_pivots(kbad, K_PIVOTS),
                           batched_select_pivots(k, K_PIVOTS))


def _bench(n, kind, reps=3):
    a, k = make_ak(n, 0, kind)
    out = {}
    for name, fn in (("loop", lambda: loop_hop2(a, k)),
                     ("vec", lambda: batched_pivot_hop2(
                         a, batched_select_pivots(k, K_PIVOTS)))):
        fn()                                            # warm
        t = min(_time(fn) for _ in range(reps))
        out[name] = t
    ag = a.clone().requires_grad_(True)
    for name, fn in (("loop_fwdbwd", lambda: loop_hop2(ag, k)),
                     ("vec_fwdbwd", lambda: batched_pivot_hop2(
                         ag, batched_select_pivots(k, K_PIVOTS)))):
        def step(fn=fn):
            if ag.grad is not None:
                ag.grad = None
            fn().sum().backward()
        step()
        out[name] = min(_time(step) for _ in range(reps))
    return out


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  s={S} "
          f"k_pivots={K_PIVOTS}")
    print(f"{'n':>6} {'kind':>15} {'loop fwd':>10} {'vec fwd':>10} {'x':>7} "
          f"{'loop f+b':>10} {'vec f+b':>10} {'x':>7}")
    for kind in ("pivot_signed", "pivot_unsigned"):
        for n in (128, 512, 2048):
            r = _bench(n, kind)
            print(f"{n:>6} {kind:>15} {r['loop']:>10.4f} {r['vec']:>10.4f} "
                  f"{r['loop'] / r['vec']:>6.1f}x {r['loop_fwdbwd']:>10.4f} "
                  f"{r['vec_fwdbwd']:>10.4f} {r['loop_fwdbwd'] / r['vec_fwdbwd']:>6.1f}x")
