# Foreman K1 third bar set: the replacement bed owed by the fired kill, and the twin's real positional channel.
# Written 2026-09-23 03:04 IST (clock read 03:03:22), after k1b.json and BEFORE k1c.py exists or runs. Never edited
# after its RED. Reads k1c.json. Exit 1 if any row fails.
#
# Replacement bed ("bed_k, deep"): bed_k's generator unchanged (make_test: 16 lanes, random ids from V = 32,768,
# uncapped); the change is the test length, n >= 64 * 2^L for an L-layer arm, so the beyond-2^L set reaches past
# 4 * 2^L. Fresh seeds default_rng([53, n, k]), k = 0..7. Constructions at R0 pricing (k3 W<=10, k4 W<=6), credited:
# HPD = the oracle C2 with the highest-density window (the strongest oracle construction found, k1b.py);
# local = the no-oracle ALiBi-local window (hunt.py). Line = max(recency, credited doubling, HPD, local).
# foreman.k1.deep_bed_floor : for (L, n) in {(4,1024), (5,2048), (6,4096), (7,8192), (7,16384), (7,32768)}:
#     HPD best <= 0.35 and local best <= 0.25 (a margin >= 0.15 under the 0.5 kill).
# foreman.k1.deep_bed_is_deeper : for every (L, n) above, the fraction of tokens with depth > 4 * 2^L is >= 0.2 (the
#     scored set is not the shallow 2^L..2.2*2^L band that the constant-factor shortcut clears at n = 4096).
#
# Harness-faithful positional channel (K0/wilson/train_ladder.py builds the ALiBi bias in q.dtype, i.e. bfloat16 under
# autocast). Same construction as test_poschan.py (slope 1/256, root sink, readout -ln(o)/m, 4 beds n = 16384 seeds
# [81, k]) but the bias -m*(t - j) is rounded to bfloat16 before the float32 softmax, output rounded to bfloat16.
# foreman.k1.harness_channel_coarse : max pair error over t, t' in [4096, 16384) >= 16 positions (one mean hop) and
#     <= 256 (the channel is coarse, not absent).
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1c.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for key in ("4_1024", "5_2048", "6_4096", "7_8192", "7_16384", "7_32768"):
    c = R["deep"][key]
    check(f"foreman.k1.deep_bed_floor L_n={key} HPD <= 0.35 and local <= 0.25", c["hpd_best"] <= 0.35 and c["local_best"] <= 0.25,
          f"hpd {c['hpd_best']:.4f} [{c['hpd_cfg']}] local {c['local_best']:.4f} line {c['line']:.4f} dbl {c['dbl_cred']:.4f} rec {c['recency']:.4f}")
    check(f"foreman.k1.deep_bed_is_deeper L_n={key} frac(depth > 4*2^L) >= 0.2", c["frac_gt_4x"] >= 0.2, c["frac_gt_4x"])
h = R["harness_channel"]
check("foreman.k1.harness_channel_coarse 16 <= max pair err <= 256", 16 <= h["max_pair_err"] <= 256, h)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
