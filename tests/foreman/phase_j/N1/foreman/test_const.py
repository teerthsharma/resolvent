# -*- coding: utf-8 -*-
"""Is the win a CONSTANT recency prior (ALiBi) or data-dependent forgetting?
Eval-only, existing checkpoints. Written before const_measure.py exists.
(f): every m_t replaced by its layer's geometric mean over the eval sites
     (constant-u is ALiBi by the carry-collapse theorem); theta untouched.
FoX: every log f_t replaced by its (layer, head) mean over the eval sites.
  C_const_suffices  NLL(f_const)-NLL(f) <= 0.02 AND NLL(FoX_const)-NLL(FoX) <= 0.02
                    (a constant decay carries the gated arms' function: the win is
                     a recency prior, not data-dependent gating)
  C_datadep_f       NLL(f_const)-NLL(f)     >= 0.05 (~21% of C_win 0.233)
  C_datadep_fox     NLL(FoX_const)-NLL(FoX) >= 0.05
  (C_datadep_* passing is necessary-at-eval only: the constant is off-distribution.)
Usage: python test_const.py RESULTS_JSON NAME -> exit 0 pass, 1 fail."""
import json, sys


def _n(v):
    if v is None:
        raise AssertionError("value is null (stub or not measured)")
    return float(v)


def C_const_suffices(r):
    df, dx = _n(r["f_const"]) - _n(r["f_full"]), _n(r["fox_const"]) - _n(r["fox_full"])
    assert df <= 0.02 and dx <= 0.02, "const cost f {} fox {} > 0.02".format(df, dx)


def C_datadep_f(r):
    d = _n(r["f_const"]) - _n(r["f_full"])
    assert d >= 0.05, "(f) const cost {} < 0.05".format(d)


def C_datadep_fox(r):
    d = _n(r["fox_const"]) - _n(r["fox_full"])
    assert d >= 0.05, "FoX const cost {} < 0.05".format(d)


T = dict(C_const_suffices=C_const_suffices, C_datadep_f=C_datadep_f, C_datadep_fox=C_datadep_fox)
if __name__ == "__main__":
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    try:
        T[sys.argv[2]](r)
    except (AssertionError, KeyError, TypeError) as e:
        print("RED {}: {}".format(sys.argv[2], e)); sys.exit(1)
    print("GREEN {}".format(sys.argv[2])); sys.exit(0)
