# Foreman K0: bars for the R-DEPTH shortcut hunt. Written 2026-09-23 01:48 IST, BEFORE any construction ran.
# Bed = wald.py make_bed(4096, 16), seed 23 (pinned sha 31271d64...), target = root id of every token.
# "beyond" = tokens with depth > 2^L (the counter's population). Exit non-zero on any failed bar.
import json, math, sys
from pathlib import Path

R = json.loads((Path(__file__).parent / "results.json").read_text())  # produced by shortcut.py
fails = []


def bar(name, ok, got):
    print(("PASS " if ok else "FAIL ") + name + f"  got={got}")
    if not ok:
        fails.append(name)


# T1 foreman.k0.doubling_sim: an explicit pointer-doubling simulation (not the depth formula) reproduces
#    the pinned ceilings 0.0664 / 0.2539 / 0.9856 at L = 4 / 6 / 8 to 4 decimals.
for L, want in ((4, 0.0664), (6, 0.2539), (8, 0.9856)):
    got = R["doubling"][str(L)]["all"]
    bar(f"doubling_sim L={L}", round(got, 4) == want, got)

# T2 foreman.k0.no_local_leak: every position-only / local-window probe scores within 1/16 + 3 SE on
#    tokens deeper than 64 (no leak that bypasses the chain).
for name, p in R["probes"].items():
    N = p["n_beyond"]
    se = math.sqrt((1 / 16) * (15 / 16) / N)
    bar(f"no_local_leak {name}", p["beyond"] <= 1 / 16 + 3 * se, p["beyond"])

# T3 foreman.k0.hybrid_beats_2L: hybrid content+positional-window pointer tripling (k=3, linear-width mux),
#    L = 6, unbounded window: accuracy on depth > 64 exceeds 0.5 (the contract's counter line).
bar("hybrid_beats_2L L=6 W=4096", R["hybrid"]["L6_W4096_k3_anchored"]["beyond"] > 0.5,
    R["hybrid"]["L6_W4096_k3_anchored"]["beyond"])

# T4 foreman.k0.hybrid_at_R1_width: same construction at a window of 26 slots (R1 d_model 320 / 12 bits per
#    pointer, generous: nothing else in the residual) stays at or below 0.5 on depth > 64.
bar("hybrid_at_R1_width L=6 W=26", R["hybrid"]["L6_W26_k3_anchored"]["beyond"] <= 0.5,
    R["hybrid"]["L6_W26_k3_anchored"]["beyond"])

sys.exit(1 if fails else 0)
