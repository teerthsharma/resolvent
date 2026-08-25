"""Does a NON-GEOMETRIC dilation schedule repair the severance?

Cameron: severance is a property of the RIGID per-layer power-of-two lattice, not
of bounded row width -- so a schedule that is not a geometric series may keep the
reach result (gradient support exactly `s`, row width exactly 8) while killing
the severance. He named it and did not test it. This tests it.

A REFINEMENT OF MY OWN ITERATION-12 MEASUREMENT, and it matters. That probe
called a position "severed" when `lo == hi` bitwise. But `reach_fraction` in
`dilated.py` counts draws whose gradient is nonzero at BOTH values of `c`, and
those are different questions. Decomposed properly there are THREE states:

    UNREACHED   lo == hi == 0.0     the edge is absent from the graph entirely
    INERT       lo == hi != 0.0     reached, but perturbing c moves nothing
    LIVE        lo != hi            c can actually move the sign

Iteration 12 reported UNREACHED + INERT as one number. That conflation hides
which problem a schedule has: a reach problem and an influence problem want
different repairs, and only the second is what "support is not influence" was
about. Reported separately here.

MATCHED DEPTH IS ENFORCED. Comparing `[1,2,4,8]` against `[1,3,5,7]` at different
depths would confound schedule with depth, and this project has already published
one number that confounded an operator with its parameter point.

PRE-REGISTERED, fixed before the run:
  * severance falls AND support holds  -> the lattice was the defect, route lives
  * both fall                          -> reach was traded away: a different arm,
                                          not a repair, and the round already
                                          refused that trade
  * severance unchanged                -> not the lattice but bounded row width
                                          itself, and the composition family closes
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dilated import composed_draws                       # noqa: E402

#: Matched depth per size. Every schedule at a given `s` has the SAME length.
SCHEDULES = {
    64: [("pow2      ", [1, 2, 4]),
         ("coprime   ", [1, 3, 5]),
         ("coprime-b ", [1, 2, 5]),
         ("contiguous", [1, 1, 1])],
    128: [("pow2      ", [1, 2, 4, 8]),
          ("coprime   ", [1, 3, 5, 7]),
          ("coprime-b ", [1, 2, 3, 5]),
          ("contiguous", [1, 1, 1, 1])],
}


def classify(c: int, *, s: int, i: int, j: int, dilations, n_draws: int = 4,
             window: int = 8, seed: int = 0) -> str:
    """UNREACHED / INERT / LIVE for one position `c`."""
    draws = composed_draws(n_draws=n_draws, s=s, d=16, i=i, j=j, c=c,
                           dilations=dilations, window=window, seed=seed)
    live = any(lo != hi for lo, hi in draws)
    if live:
        return "LIVE"
    reached = any(lo != 0.0 or hi != 0.0 for lo, hi in draws)
    return "INERT" if reached else "UNREACHED"


def sweep(s: int, dilations, n_draws: int = 4, window: int = 8) -> dict:
    i, j = s - 1, max(1, s // 4)
    counts = {"UNREACHED": 0, "INERT": 0, "LIVE": 0}
    for c in range(j + 1, i):
        counts[classify(c, s=s, i=i, j=j, dilations=dilations,
                        n_draws=n_draws, window=window)] += 1
    n = sum(counts.values())
    return dict(n=n, **counts,
                severed=(counts["UNREACHED"] + counts["INERT"]) / n,
                live=counts["LIVE"] / n,
                support=1.0 - counts["UNREACHED"] / n)


def main() -> int:
    torch.set_num_threads(1)          # a measurement is running; do not starve it
    print("DILATION SCHEDULE vs SEVERANCE -- exhaustive over every legal c, matched depth")
    print("UNREACHED lo==hi==0 | INERT lo==hi!=0 | LIVE lo!=hi\n")
    for s, scheds in SCHEDULES.items():
        depth = len(scheds[0][1])
        print(f"  s={s}, depth={depth}, window=8")
        print(f"    {'schedule':<12} {'unreach':>8} {'inert':>7} {'live':>6} "
              f"{'severed':>9} {'support':>9}")
        for name, dil in scheds:
            assert len(dil) == depth, "matched depth is enforced"
            r = sweep(s, dil)
            print(f"    {name:<12} {r['UNREACHED']:>8} {r['INERT']:>7} {r['LIVE']:>6} "
                  f"{r['severed']:>9.4f} {r['support']:>9.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
