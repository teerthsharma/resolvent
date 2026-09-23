# Foreman K1 bar: does random ids + ALiBi leave the twin position-blind? Written 2026-09-23 02:59 IST, BEFORE
# poschan.py exists or runs. Never edited after its RED. Reads poschan.json. Exit 1 if any row fails.
#
# The R0 twin (K0/wilson/train_ladder.py, d 128, D_HEAD 64) has 2 heads per layer; get_slopes(2) gives its slopes.
# Construction: ONE head at the smaller slope m, content logit b = 0 on every key except the roots (pid == NULL,
# which the content marks), which get logit 0 too -- i.e. plain ALiBi -- and value 1 on roots, 0 elsewhere. Its
# output o(t) = sum over roots of the ALiBi weight. Readout t_hat = -ln(o(t)) / m. Softmax weights in float32 (as the
# kernel computes them); the head output rounded to the dtype under test. 4 bed_k test beds, n = 16384, seeds
# default_rng([81, k]).
#
# foreman.k1.alibi_position_channel : m equals the R0 twin's smaller slope (1/256), and for every pair of tokens
#     t, t' in [4096, 16384) the relative position is recovered: |(t_hat - t_hat') - (t - t')| <= 1.0 with the output
#     in bfloat16 and <= 0.01 in float32.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "poschan.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


check("foreman.k1.alibi_position_channel slope is the R0 twin's smaller slope 1/256",
      R["slope"] == 1 / 256 and R["slope"] == R["ladder_slopes_R0"][1], (R["slope"], R["ladder_slopes_R0"]))
check("foreman.k1.alibi_position_channel bf16 max pair error <= 1.0", R["bf16_max_pair_err"] <= 1.0, R["bf16_max_pair_err"])
check("foreman.k1.alibi_position_channel fp32 max pair error <= 0.01", R["fp32_max_pair_err"] <= 0.01, R["fp32_max_pair_err"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
