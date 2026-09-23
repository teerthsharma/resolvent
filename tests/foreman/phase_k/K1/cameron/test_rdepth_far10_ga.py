# R-DEPTH arm (f_R, gamma-annealed) on bed_k' far10 (Dispatcher after Inspector PASS6; to be recorded as amendment K1-b).
# Written 2026-09-23 BEFORE any far_fR_ga run exists. Never edited after its RED.
# Arm far_fR_ga_s{0,1}: rdepth.py --arm fR --layers 4 --hook <Chase resolvent_hook.py sha 3855288d:ResolventAttention>
#   --bed kpf --emb frozen --par_init orth --lr 3e-3 --steps 8000 --probe_every 2000 --eval_ns 1024,4096,8192,16384
#   --gamma_sched "0.5:2000,0.9:4000,0.99:6000,0.999" --seed {0,1}
#   i.e. the far_fR recipe with the resolvent's gamma = 0.5 for steps 1-1999, 0.9 for 2000-3999, 0.99 for 4000-5999 and
#   0.999 (the registered value) from step 6000 to 8000; every evaluation after training runs at gamma = 0.999.
#   Nothing else changes (logit-scale law a_h ln(i+1) + b_h, a = 1, b = 0 at init, as the hook). gamma annealing is
#   resolvent-only, so no twin copy is owed (Dispatcher).
# Runs: runs/far_fR_ga_s<seed>/result.json (config bed "kpf", gamma_sched as above) and ok_<n>.npz.
# GATE L-TRAINED: held-out n=1024 acc on depth <= 16 >= 0.9 (Larm = 4), else UNREAD (failing).
# cameron.k1.rdepth_ga_trained     : the gate, each seed.
# cameron.k1.rdepth_ga_learnable   : contract kill. held-out n=1024 acc (all tokens) >= 0.8, each seed.
# cameron.k1.rdepth_ga_prediction  : far10 bands (as test_rdepth_far10.py): b4 = depth > 160 at n in {4096, 8192, 16384}
#                                    and b7 = depth > 1280 at n = 16384, each >= 0.95, each seed.
# cameron.k1.rdepth_ga_multilen16k : RECORD_K clause 4. acc (all tokens) at n = 16384 >= 0.95, each seed.
# Exit 1 if any row fails.
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
SCHED = "0.5:2000,0.9:4000,0.99:6000,0.999"
fails = []


def check(name, ok, got=""):
    tag = "PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")
    print(tag + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


for s in (0, 1):
    d = HERE / "runs" / f"far_fR_ga_s{s}"
    r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
    if r is not None and not (r["config"].get("bed") == "kpf" and r["config"].get("gamma_sched") == SCHED and r["config"]["arm"] == "fR"):
        r = None
    tr = None if r is None else r["eval"]["1024"]["acc_le16"]
    check(f"cameron.k1.rdepth_ga_trained fR_ga s{s} held-out depth<=16 >= 0.9", False if tr is None else tr >= 0.9, "missing" if tr is None else f"{tr:.4f}")
    v = None if r is None else r["eval"]["1024"]["acc"]
    check(f"cameron.k1.rdepth_ga_learnable fR_ga s{s} held-out acc >= 0.8", False if v is None else v >= 0.8, "missing" if v is None else f"{v:.4f}")
    rd = r if (tr is not None and tr >= 0.9) else None
    for n, thr in (("4096", 160), ("8192", 160), ("16384", 160), ("16384", 1280)):
        if rd is None:
            check(f"cameron.k1.rdepth_ga_prediction fR_ga s{s} n={n} depth>{thr} >= 0.95", None, "unread/missing")
            continue
        z = np.load(d / f"ok_{n}.npz")
        m = z["depth"] > thr
        v = float(z["ok"][m].mean())
        check(f"cameron.k1.rdepth_ga_prediction fR_ga s{s} n={n} depth>{thr} >= 0.95", v >= 0.95, f"{v:.4f} (N {int(m.sum())})")
    v = None if rd is None else rd["eval"]["16384"]["acc"]
    check(f"cameron.k1.rdepth_ga_multilen16k fR_ga s{s} acc n=16384 >= 0.95", None if v is None else v >= 0.95,
          "unread/missing" if v is None else f"{v:.4f}")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
