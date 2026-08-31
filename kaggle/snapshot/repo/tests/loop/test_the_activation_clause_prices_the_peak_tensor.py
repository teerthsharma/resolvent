"""C-C prices the projections; the peak is the operator, and the ratio is s/d.

THE CLAUSE, from the v-main.3M script, iteration 16:

    "C-C peak activation bytes = 4*n*s*d*heads computed in writing vs card bytes"

`4*n*s*d*heads` is the size of `q` or `k` — a `[n, s, d_model]` projection. But
`m3_capability.py:143-144` computes

    q, k = self.wq(x), self.wk(x)          # [n, s, d_model]
    a = self._operator(q, k)               # [n,s,s]   <- the shape is in the source

and `a` is the peak. `:108-109` build ONE `wq` and ONE `wk`, so `heads = 1` is the
construction rather than a convention.

**The clause under-prices by exactly `s/d_model`.** SATURN measured the
consequence at iteration 16: at batch 8192 with `s = 1024` the clause reads
**512 MiB** and the real tensor is **32,768 MiB** — four times the whole 8188 MiB
card. A clause that passes at 512 MiB is certifying a job that cannot fit.

WHY BIND IT RATHER THAN JUST RECORD IT. A wrong sizing formula is not a wrong
verdict; it is a wrong number that someone reuses. This round already paid for
that once from the other direction: a host-RSS model validated against a held-out
point *inside* its fitted range, then used to price a rung outside it, understated
by 8.2% in the permissive direction (`R10_MECHANISM.md` instance 22). Both errors
quote a smaller requirement than the truth, and both look validated.

THIS TEST IS BEHAVIOURAL. It runs the shipped module at a small shape and reads
the tensor's actual size, rather than asserting the source says `[n,s,s]`. A
comment can be stale; an allocation cannot.

SCOPE. `s` is kept small so the peak is a few MiB — the point is the RATIO, which
is shape-independent, not the absolute number.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")


def shapes_at(n: int, s: int, d_model: int) -> dict[str, tuple]:
    """Actual tensor shapes the shipped forward pass materialises."""
    from scale import m3_capability as MC

    arm = MC.Arm(kind="softmax", s=s, d_model=d_model)
    seen: dict[str, tuple] = {}
    real_op = arm._operator

    def spy(q, k):
        seen["projection"] = tuple(q.shape)
        out = real_op(q, k)
        seen["operator"] = tuple(out.shape)
        return out

    arm._operator = spy
    with torch.no_grad():
        arm(torch.zeros(n, s, d_model))
    return seen


def test_the_spy_actually_observed_both_tensors():
    """Must-fire for this file's own instrument. Empty shapes would make every
    assertion below pass by vacuity."""
    seen = shapes_at(2, 16, 8)
    assert "projection" in seen and "operator" in seen, (
        f"the forward pass did not route through `_operator`: {seen}. This guard "
        "is reading the wrong hook and would pass without measuring anything."
    )


@pytest.mark.parametrize("s,d_model", [(16, 8), (32, 8), (64, 16)])
def test_the_peak_is_the_operator_not_the_projection(s: int, d_model: int):
    """THE DEFECT, as a ratio. GREEN today and it stays green — this records the
    correct formula executably, so a future job sizing reads the peak rather than
    the projection."""
    n = 2
    seen = shapes_at(n, s, d_model)
    proj, op = seen["projection"], seen["operator"]

    assert op == (n, s, s), f"the operator is not [n,s,s]: {op}"
    assert proj == (n, s, d_model), f"the projection is not [n,s,d_model]: {proj}"

    proj_elems = proj[0] * proj[1] * proj[2]
    op_elems = op[0] * op[1] * op[2]
    ratio = op_elems / proj_elems

    assert ratio == pytest.approx(s / d_model), (
        f"the operator/projection element ratio is {ratio}, not s/d_model = "
        f"{s / d_model}. The C-C clause prices `4*n*s*d*heads` (the projection); "
        f"the peak is `4*n*s^2*heads` (the operator). If this ratio has changed, "
        f"the correction recorded at it.16 no longer holds and the clause needs "
        f"re-deriving rather than re-scaling."
    )
    assert op_elems > proj_elems, (
        "the operator is not larger than the projection at this shape, so the "
        "clause's under-pricing does not arise here and this case proves nothing"
    )


def test_the_shipped_shape_underprices_by_sixty_four():
    """The specific number the it.16 finding rests on: at s=1024, d_model=16 the
    clause is 64x low. Asserted arithmetically so the claim in the record is
    executable rather than quoted."""
    from scale.m3_capability import D_MODEL

    s = 1024
    assert D_MODEL == 16, (
        f"D_MODEL is {D_MODEL}, not 16; the 64x figure in the it.16 record was "
        "derived at 16 and must be re-derived"
    )
    assert s / D_MODEL == 64
    # and the absolute figures SATURN reported, at batch 8192
    n, bytes_per = 8192, 4
    clause_mib = n * s * D_MODEL * bytes_per / 2**20
    peak_mib = n * s * s * bytes_per / 2**20
    assert clause_mib == pytest.approx(512.0), clause_mib
    assert peak_mib == pytest.approx(32768.0), peak_mib
    assert peak_mib / clause_mib == pytest.approx(64.0)
