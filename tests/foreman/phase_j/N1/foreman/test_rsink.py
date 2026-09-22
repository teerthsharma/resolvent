# -*- coding: utf-8 -*-
"""R-SINK bars (Addendum N sec.5), written BEFORE rsink_measure.py exists.

Usage: python test_rsink.py RESULTS_JSON NAME   -> exit 0 pass, 1 fail.
Run first against rsink_stub.json (every value null) -> every bar RED.

Pre-registered readings (fixed here, before any number is seen):
  closure        = site-layer pair of (f) with magnitude m < 0.05 (all 3 layers pooled).
  delimiter      = byte 10 (newline; also covers the '\\n\\n' document boundary) or 46.
  P1  lift = P(delim | closure) / P(delim) >= 2, AND AUC(score=-m, label=delim) >= 0.7,
      AND AUC > the 95th pct of a within-document position shuffle of m.
  P2  mean over (closure c, later query i in c's segment) of (f)'s |W_ic| >= 0.3.
  P3  median value norm at closure sites / median value norm over all sites <= 0.5.
  P4  Spearman over sites t in [1, S-65] between (f)'s closure rate (fraction of
      layers closed) and (a)'s received attention sum_{i=t+1..t+64} a_it
      (head- and layer-mean) >= 0.3, AND the 95th pct |rho| of a within-window
      shuffle <= 0.05.
  FT_a_sink    max over layers of (a)'s first-token mass >= 0.2 (uniform ~0.013).
  FT_f_relief  (f)'s mass in (a)'s argmax layer <= 0.5 x (a)'s.
  FT_fox_relief  FoX's mass in that layer <= 0.5 x (a)'s.
  X_colocate   max over layers of Spearman((f)'s m_t, FoX head-mean forget f_t)
               >= 0.3, AND 95th pct shuffle |rho| <= 0.05.
"""
import json
import sys


def _num(v):
    if v is None:
        raise AssertionError("value is null (stub or not measured)")
    return float(v)


def P1(r):
    p = r["P1"]
    assert _num(p["lift"]) >= 2.0, "P1 lift {} < 2".format(p["lift"])
    assert _num(p["auc"]) >= 0.7, "P1 AUC {} < 0.7".format(p["auc"])
    assert _num(p["auc"]) > _num(p["null_auc_p95"]), "P1 AUC not above shuffle null"


def P2(r):
    v = _num(r["P2"]["mean_attn_on_closure"])
    assert v >= 0.3, "P2 {} < 0.3".format(v)


def P3(r):
    v = _num(r["P3"]["median_ratio"])
    assert v <= 0.5, "P3 {} > 0.5".format(v)


def P4(r):
    p = r["P4"]
    assert _num(p["rho"]) >= 0.3, "P4 rho {} < 0.3".format(p["rho"])
    assert _num(p["null_abs_rho_p95"]) <= 0.05, "P4 null too wide"


def _ft(r):
    a = [_num(v) for v in r["FT"]["a"]]
    l = max(range(len(a)), key=lambda i: a[i])
    return a, l


def FT_a_sink(r):
    a, l = _ft(r)
    assert a[l] >= 0.2, "(a) first-token mass max {} < 0.2".format(a[l])


def FT_f_relief(r):
    a, l = _ft(r)
    f = _num(r["FT"]["f"][l])
    assert f <= 0.5 * a[l], "(f) {} > 0.5 x (a) {}".format(f, a[l])


def FT_fox_relief(r):
    a, l = _ft(r)
    f = _num(r["FT"]["fox"][l])
    assert f <= 0.5 * a[l], "FoX {} > 0.5 x (a) {}".format(f, a[l])


def X_colocate(r):
    p = r["X_colocate"]
    rho = max(_num(v) for v in p["rho_per_layer"])
    assert rho >= 0.3, "colocation rho {} < 0.3".format(rho)
    assert _num(p["null_abs_rho_p95"]) <= 0.05, "colocation null too wide"


TESTS = dict(P1=P1, P2=P2, P3=P3, P4=P4, FT_a_sink=FT_a_sink,
             FT_f_relief=FT_f_relief, FT_fox_relief=FT_fox_relief,
             X_colocate=X_colocate)

if __name__ == "__main__":
    path, name = sys.argv[1], sys.argv[2]
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    try:
        TESTS[name](res)
    except (AssertionError, KeyError, TypeError) as e:
        print("RED {}: {}".format(name, e))
        sys.exit(1)
    print("GREEN {}".format(name))
    sys.exit(0)
