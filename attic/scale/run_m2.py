"""M2, per CHECKLIST.md, verbatim. LOCK M2 efadc390c93f.

Draw budget is STATED, not silently capped: 4096 draws at s <= 512, 16384 at
s >= 1024 -- "16384 draws at the tail", which is where the rates are small and
the Clopper-Pearson interval needs them. Timed first: 124.5 ms/draw at s=2048,
so the tail costs 0.57 h per cell rather than being guessed at.

PRE-REGISTERED KILL, frozen with the item:
  slope(c in P) < -0.3
  OR c-not-in-P is ALSO flat  (mechanism story false even if the number is good)
  OR any previously published bench.py number moves  (G2)
"""
from __future__ import annotations
import json, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.pivot_probe import run_arm, clopper_pearson, loglog_slope

SIZES = [8, 32, 128, 512, 1024, 2048]
def budget(s): return 16384 if s >= 1024 else 4096

#: (label, arm, placement). The claim, its own control, and both instrument ends.
CELLS = [
    ("pivot_signed  c IN P",     "pivot_signed",   "in_P"),
    ("pivot_signed  c NOT in P", "pivot_signed",   "not_in_P"),
    ("pivot_signed  c windowed", "pivot_signed",   "windowed"),
    ("dense_signed  (reference)","dense_signed",   "not_in_P"),
    ("softmax FLOOR",            "dense_unsigned", "not_in_P"),
    ("random CEILING",           "random",         "not_in_P"),
]

def main():
    print("PROTOCOL: SCALING   (i=s-1, j=s/4, c drawn from the placement pool)")
    print("M2  LOCK efadc390c93f   k=8 pivots, content-selected by key-norm, "
          "never using c")
    print("draws: 4096 at s<=512, 16384 at s>=1024 (stated, not capped silently)")
    print("KILL: slope(c in P) < -0.3, OR c-not-in-P also flat, OR any published "
          "bench number moves\n")
    hdr = "".join(f"{s:>22}" for s in SIZES)
    print(f"{'cell':>26} {hdr}{'slope':>9}")
    out = {}
    for label, arm, place in CELLS:
        cells, rates = [], []
        for s in SIZES:
            t0 = time.time()
            r = run_arm(arm, s, n_draws=budget(s), k=8, placement=place)
            lo, hi = clopper_pearson(r["k"], max(r["n"], 1))
            rates.append(r["rate"])
            cells.append(f"{r['rate']:.5f}[{lo:.4f},{hi:.4f}]")
            print(f"    .. {label} s={s} -> {r['rate']:.5f} "
                  f"({r['n']} used, {time.time()-t0:.0f}s)", flush=True)
        sl, npts = loglog_slope(SIZES, rates)
        out[label] = dict(rates=rates, slope=sl, nonzero=npts)
        note = "" if npts == len(SIZES) else f" ({npts}/{len(SIZES)} nonzero)"
        print(f"{label:>26} " + "".join(f"{c:>22}" for c in cells)
              + f"{sl:>+9.3f}{note}", flush=True)

    pathlib.Path("results/m2.json").write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT against the frozen kill ===")
    a = out["pivot_signed  c IN P"]["slope"]
    b = out["pivot_signed  c NOT in P"]["slope"]
    k1 = (a < -0.3) if a == a else True
    k2 = (b > -0.3) if b == b else False
    print(f"  slope(c in P)     = {a:+.3f}  -> {'KILL' if k1 else 'survives'}")
    print(f"  slope(c NOT in P) = {b:+.3f}  -> "
          + ("KILL: control is ALSO flat, mechanism story false"
             if k2 else "control decays as the mechanism requires"))
    print(f"\n  M2 = {'RED' if (k1 or k2) else 'GREEN'}")

if __name__ == "__main__":
    main()
