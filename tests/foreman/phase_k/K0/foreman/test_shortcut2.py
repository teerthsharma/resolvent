# Foreman K0 round 2 bars. Written 2026-09-23 01:56 IST, BEFORE shortcut2.py existed or ran.
# Hybrid = content fetch + centered positional window, k = 3 (linear-width mux), slots W = floor(d_model / 12).
# D_c = number of depth values d whose accuracy (over all tokens of depth d) is >= 0.5. Exit non-zero on failure.
import json, sys
from pathlib import Path

R = json.loads((Path(__file__).parent / "results2.json").read_text())
fails = []


def bar(name, ok, got):
    print(("PASS " if ok else "FAIL ") + name + f"  got={got}")
    if not ok:
        fails.append(name)


# T5 foreman.k0.rscale_width_confound: on a fresh n=16384, M=16 bed, the doubling control's D_c-vs-2^L log-slope
#    over R0-R2 (L=4/6/8) lies in [0.95, 1.05], and the hybrid at each rung's own width (W=10/26/42) has
#    log-slope > 1.2, i.e. R-SCALE's counter ("D_c grows faster than 2^L with rung") fires by construction.
bar("rscale control slope in [0.95,1.05]", 0.95 <= R["rscale"]["doubling_slope"] <= 1.05, R["rscale"]["doubling_slope"])
bar("rscale hybrid slope > 1.2", R["rscale"]["hybrid_slope"] > 1.2, R["rscale"]["hybrid_slope"])

# T6 foreman.k0.dilated_bed: replacement bed M=64 chains, n=16384 (depth ~256): at L=6, W=26 the hybrid's
#    all-token accuracy is <= 1.25x the doubling ceiling on the same bed (vs 1.96x on the pinned M=16 bed).
bar("dilated bed hybrid/doubling <= 1.25", R["dilated"]["ratio"] <= 1.25, R["dilated"]["ratio"])

sys.exit(1 if fails else 0)
