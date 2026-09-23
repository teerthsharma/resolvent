# Foreman K1 sixth bar: an independent slow reference for the construction that fired the kill. Written 2026-09-23
# 03:11 IST (clock read 03:11:35), BEFORE hpd_ref.py exists or runs. Never edited after its RED. Reads hpd_ref.json.
#
# hpd_ref.py re-implements the HPD-window hybrid per token in plain Python (explicit loops; its own NegBin pmf from
# math.lgamma; its own window search; truth by walking parent pointers to the root; credit by walking back from the
# final pointer to the most recent root) with no import from hunt.py / k1b.py / shortcut.py, on Cameron's beds
# default_rng([12, 4096, k]), k = 0..3, at L = 7, k = 3, W = 10.
# foreman.k1.hpd_reference : 0 final-pointer mismatches against k1b.hpd_hybrid on all 4 beds, and the per-bed credited
#     beyond-2^7 accuracy equals k1b's to 1e-12; the 4-bed mean is > 0.5.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "hpd_ref.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


check("foreman.k1.hpd_reference pointer mismatches == 0", R["mismatches"] == 0, R["mismatches"])
check("foreman.k1.hpd_reference per-bed accuracy equal to 1e-12", R["max_acc_diff"] < 1e-12, R["max_acc_diff"])
check("foreman.k1.hpd_reference 4-bed mean > 0.5", R["ref_mean"] > 0.5, R["ref_mean"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
