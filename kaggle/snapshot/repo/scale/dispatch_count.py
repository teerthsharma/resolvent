"""Count aten dispatches, so the cost gate stops resting on an inference.

K-F asks whether the settling arm costs more than 1.5x the glance. Two numbers
disagree about that and neither settles it:

    FLOP ratio    1.010420 .. 3.851464   (exact, but ignores per-op overhead)
    clock ratio   1.6546   .. 45.0608    (contended box, not a measurement)

The gap between them was attributed to framework dispatch overhead, on the
strength of fetched literature. That was the right diagnosis and the wrong
evidence class: a citation explains a mechanism, it does not measure this arm.

WHAT THIS MEASURES INSTEAD. `TorchDispatchMode` intercepts every aten call, so
the dispatch count is exact, deterministic, and independent of who else is on the
machine. Unlike wall clock it can be taken on a loaded box; unlike FLOPs it sees
the per-operation cost that a Python settling loop actually pays.

THE ROUTE THE PREVIOUS REPORT NAMED IS UNAVAILABLE HERE. `collect_callgrind`
requires valgrind, which does not exist on Windows, so K-F cannot be closed by
that route on this box at all -- not "not done yet", but not available. The
dispatch count is the platform-independent substitute, and it answers a narrower
question honestly rather than the full question badly.

THE PREDICTION THIS MAKES, WHICH IS WHY IT IS WORTH MEASURING. A settling loop is
a Python loop: it issues O(t*) dispatches while performing O(1) FLOPs in t*. So
the dispatch count must grow LINEARLY in the step count while the FLOP count stays
flat. If that holds, the arm's overhead is amortisable by batching or fusing the
inner loop, which changes no arithmetic. If it does not hold, the diagnosis was
wrong and the overhead is somewhere else.
"""
from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["DispatchCounter", "count_dispatches", "settle_dispatches",
           "glance_dispatches"]


class DispatchCounter(TorchDispatchMode):
    """Counts every aten call made inside the context.

    Exact rather than sampled, and unaffected by machine load — which is the
    whole reason it is used here instead of a timer.
    """

    def __init__(self) -> None:
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def count_dispatches(fn) -> int:
    """Dispatches issued by calling `fn()` once."""
    with DispatchCounter() as c:
        fn()
    return c.n


def _operator(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    q = x @ w
    return torch.softmax(q @ q.transpose(-2, -1) / (x.shape[-1] ** 0.5), -1)


def glance_dispatches(s: int = 64, d: int = 24, seed: int = 0) -> int:
    """A single-pass reading: build the operator, apply it once."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(s, d, generator=g)
    w = torch.randn(d, d, generator=g)
    return count_dispatches(lambda: _operator(x, w) @ x)


def settle_dispatches(steps: int, s: int = 64, d: int = 24, seed: int = 0) -> int:
    """The same operator, then `steps` normalised power iterations.

    The operator build is shared with the glance, so the difference between this
    and `glance_dispatches` is exactly what settling adds.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(s, d, generator=g)
    w = torch.randn(d, d, generator=g)

    def run():
        m = _operator(x, w)
        v = torch.full((s,), 1.0 / s)
        for _ in range(steps):
            v = m @ v
            v = v / v.sum()
        return v

    return count_dispatches(run)
