# Foreman K1 item (3): a trained opponent on the far band, given 4x the arms' budget. Written 2026-09-23 04:57 IST
# (clock read 04:57:04), BEFORE any run below exists. Never edited after its RED. Exit 1 on any fail.
#
# Runs: Cameron's rdepth.py (K1/cameron, executed read-only as a subprocess; outputs only in my lane) under the house
# GPU lock: runs/long_aL4_s7 = --arm aL --layers 4; runs/long_aL7_s7 = --arm aL --layers 7 --dff 183; both --bed kpf
# --emb frozen --par_init orth --lr 3e-3 (the recipe his pilot runs/pilot_aL7_frozen_orth_lr3e-3_8k learned with:
# held-out acc 0.995) --seed 7 --steps 16000 --tok 8192 --probe_every 4000: 131M tokens = 4x an arm's 4000 x 8192.
# Eval beds are his K1.F'' beds default_rng([31, n, k]); the band scores are recomputed from ok_<n>.npz.
# Bands (Cameron's far10, c_max 5.0): b4 = depth > 160, b7 = depth > 1280 (n = 16384).
# foreman.k1.far_opp_gate : each run learned what its depth allows: held-out n=1024 accuracy on depth <= min(2^L, 32)
#     >= 0.9 (Cameron's L-TRAINED gate). A failed gate voids that run's rows below (they FAIL, not pass).
# foreman.k1.far_opp_quiet : aL4 on b4 at n in {4096, 8192, 16384} and aL7 on b7 at n = 16384 score <= 0.5.
# foreman.k1.far_opp_chance : the same rows score <= 1/8 + 0.03 = 0.155 (no partial shortcut found by SGD).
# foreman.k1.far_opp_plateau : not a budget artifact: the probe's "band4" (depth > 96, beyond every bound construction
#     at L = 4) for aL4 at n = 4096, and "band7" (depth > 896) for aL7 at n = 16384, rise by <= 0.03 from the step-8000
#     probe to the end.
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


def band(run, n, thr):
    z = np.load(run / f"ok_{n}.npz")
    m = z["depth"] > thr
    return float(z["ok"][m].mean()), int(m.sum())


for arm, L, cells, key, pn in (("aL4", 4, [(4096, 160), (8192, 160), (16384, 160)], "band4", "4096"),
                               ("aL7", 7, [(16384, 1280)], "band7", "16384")):
    run = RUNS / f"long_{arm}_s7"
    res = json.loads((run / "result.json").read_text())
    ev = res["eval"]["1024"]
    le = ev["acc_le16"] if L == 4 else ev["acc_le32"]
    gate = le is not None and le >= 0.9
    check(f"foreman.k1.far_opp_gate {arm} held-out acc(depth <= {min(2 ** L, 32)}) >= 0.9", gate, le)
    for n, thr in cells:
        v, cnt = band(run, n, thr)
        check(f"foreman.k1.far_opp_quiet {arm} n={n} depth>{thr} <= 0.5", gate and v <= 0.5, (v, cnt, "" if gate else "VOID"))
        check(f"foreman.k1.far_opp_chance {arm} n={n} depth>{thr} <= 0.155", gate and v <= 0.155, (v, cnt))
    probes = {r["step"]: r["eval"] for r in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if r.get("t") == "probe"}
    mid, end = probes[8000][pn][key], res["eval"][pn][key]
    check(f"foreman.k1.far_opp_plateau {arm} {key} n={pn} end - step8000 <= 0.03", gate and end - mid <= 0.03, (mid, end))
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
