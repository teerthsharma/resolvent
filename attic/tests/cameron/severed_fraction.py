"""THE SEVERED FRACTION -- the measurement Cameron named and could not run.

He established that a dilated band composed over log2(s) layers reads the ENTIRE
context (gradient support exactly s, row width exactly 8), refuting
`CHECKLIST.md:50`'s "windowed arms buy flatness only by surrendering reach".

Then he killed his own headline: at the swept geometry `c = i - s/4` every `s` is
a power of two, so `c` sat ON the power-of-two dilation lattice at all five
points. Three positions off, the flip rate fell 0.121094 -> 0.000000 and the raw
gradient pairs were `lo == hi` BITWISE. Off-lattice, perturbing `c` changes the
gradient not slightly but NOT AT ALL: **severed, not diluted.**

What he could not establish, in his own words: *"I showed SOME c positions are
severed, not what share. A random-c sweep is the missing measurement and it
decides whether this route lives."*

THIS IS THAT MEASUREMENT, and it is done EXHAUSTIVELY rather than by sampling.
Severance is a property of c's POSITION against the lattice, not a random event,
so every legal `c` in `(j, i)` is swept and the fraction is exact for that
geometry -- no sampling error to report.

READ IT THIS WAY, fixed before running:
  * **severed fraction near 0** -> the lattice reaches almost everywhere, the
    off-lattice case Cameron found is a rare corner, and composition survives.
  * **severed fraction large** -> the ladder reads the whole context only for the
    positions it happens to land on. "Reach" would then be a statement about the
    SUPPORT of the gradient and not about which tokens can actually MOVE it, and
    the composition route dies on the same blade that killed pivot routing:
    a selection rule that discards the thing being measured.

DECLARED COST: `n_draws` per `c` is small. Severance is checked as `lo == hi`
BITWISE across every draw at that `c`, which is a much cheaper event to detect
than a rate -- one non-identical pair is enough to call a position live, so a
small draw count cannot manufacture severance, only miss liveness. That
asymmetry is stated because it biases the answer toward MORE severance, i.e.
against the route this probe might otherwise be accused of favouring.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dilated import composed_draws, log_schedule   # noqa: E402


def severed_at(c: int, *, s: int, i: int, j: int, dilations, n_draws: int = 4,
               window: int = 8, seed: int = 0) -> bool:
    """True when perturbing token `c` leaves EVERY gradient pair bitwise equal."""
    draws = composed_draws(n_draws=n_draws, s=s, d=16, i=i, j=j, c=c,
                           dilations=dilations, window=window, seed=seed)
    for lo, hi in draws:
        if lo != hi:            # one live pair is enough: the position is not severed
            return False
    return True


def sweep(s: int, *, n_draws: int = 4, window: int = 8, seed: int = 0) -> dict:
    i, j = s - 1, max(1, s // 4)
    dil = log_schedule(s, window=window)
    positions = list(range(j + 1, i))
    severed = [c for c in positions if severed_at(c, s=s, i=i, j=j, dilations=dil,
                                                  n_draws=n_draws, window=window,
                                                  seed=seed)]
    return dict(s=s, i=i, j=j, depth=len(dil), dilations=dil,
                n_positions=len(positions), n_severed=len(severed),
                fraction=len(severed) / len(positions) if positions else float("nan"),
                severed=severed)


def main() -> int:
    torch.set_num_threads(1)          # a measurement is running; do not starve it
    print("SEVERED FRACTION -- exhaustive over every legal c in (j, i)")
    print("severed := perturbing x[c] leaves EVERY gradient pair bitwise equal\n")
    print(f"{'s':>6} {'depth':>6} {'positions':>10} {'severed':>8} {'fraction':>10}  dilations")
    for s in (64, 128):
        r = sweep(s, n_draws=4)
        print(f"{r['s']:>6} {r['depth']:>6} {r['n_positions']:>10} "
              f"{r['n_severed']:>8} {r['fraction']:>10.4f}  {r['dilations']}")
        live = [c for c in range(r["j"] + 1, r["i"]) if c not in set(r["severed"])]
        print(f"       LIVE positions (first 12): {live[:12]}")
        print(f"       offsets i-c for those:     {[r['i'] - c for c in live[:12]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
