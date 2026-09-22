# -*- coding: utf-8 -*-
"""Deeper-cause bars, written after rsink.json (H-SINK struck) and BEFORE
window_measure.py exists. Hypothesis W: the win is locality, not sink relief.
Eval-only hard causal window of w=16 bytes (keys i-15..i), same 64x512 sites.
  W1_f_local    NLL(f, w16)   - NLL(f)   <= 0.01 nats/byte   ((f) lives inside 16 bytes)
  W1_fox_local  NLL(FoX, w16) - NLL(FoX) <= 0.01 nats/byte
  W2_twin_dilution  (NLL(a) - NLL(a, w16)) / (NLL(a) - NLL(f)) >= 0.5
                    (windowing the UNTRAINED-for-it twin recovers half the win:
                     the twin's deficit is long-range dilution).
Usage: python test_window.py RESULTS_JSON NAME -> exit 0 pass, 1 fail."""
import json, sys


def _n(v):
    if v is None:
        raise AssertionError("value is null (stub or not measured)")
    return float(v)


def W1_f_local(r):
    d = _n(r["f_w16"]) - _n(r["f_full"])
    assert d <= 0.01, "(f) window cost {} > 0.01".format(d)


def W1_fox_local(r):
    d = _n(r["fox_w16"]) - _n(r["fox_full"])
    assert d <= 0.01, "FoX window cost {} > 0.01".format(d)


def W2_twin_dilution(r):
    rec = (_n(r["a_full"]) - _n(r["a_w16"])) / (_n(r["a_full"]) - _n(r["f_full"]))
    assert rec >= 0.5, "twin recovery {} < 0.5".format(rec)


T = dict(W1_f_local=W1_f_local, W1_fox_local=W1_fox_local, W2_twin_dilution=W2_twin_dilution)
if __name__ == "__main__":
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    try:
        T[sys.argv[2]](r)
    except (AssertionError, KeyError, TypeError, ZeroDivisionError) as e:
        print("RED {}: {}".format(sys.argv[2], e)); sys.exit(1)
    print("GREEN {}".format(sys.argv[2])); sys.exit(0)
