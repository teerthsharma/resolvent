# -*- coding: utf-8 -*-
"""P5 (Addendum N sec.5) read from EXISTING trained records, no training.
Arm (a2) in design4x5 is the twin plus a per-head sigmoid output gate after
SDPA (Qiu et al. headwise G1 form). recovery_s = (a - a2)/(a - f) per split
seed s in 0..4, from run_record.json final_eval_loss.
  P5_sink_relief   mean recovery >= 0.5 (the win is sink relief)
  P5_not_sink      mean recovery <  0.2 AND upper 95% t-CI < 0.5 (it is not)
Usage: python test_p5.py RESULTS_JSON NAME -> exit 0 pass, 1 fail."""
import json, sys


def _n(v):
    if v is None:
        raise AssertionError("value is null (stub or not measured)")
    return float(v)


def P5_sink_relief(r):
    m = _n(r["recovery_mean"])
    assert m >= 0.5, "recovery {} < 0.5".format(m)


def P5_not_sink(r):
    m, hi = _n(r["recovery_mean"]), _n(r["recovery_ci"][1])
    assert m < 0.2 and hi < 0.5, "recovery {} (hi {}) not below 0.2/0.5".format(m, hi)


T = dict(P5_sink_relief=P5_sink_relief, P5_not_sink=P5_not_sink)
if __name__ == "__main__":
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    try:
        T[sys.argv[2]](r)
    except (AssertionError, KeyError, TypeError, IndexError) as e:
        print("RED {}: {}".format(sys.argv[2], e)); sys.exit(1)
    print("GREEN {}".format(sys.argv[2])); sys.exit(0)
