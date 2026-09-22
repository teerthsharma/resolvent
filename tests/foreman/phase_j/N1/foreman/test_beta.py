"""Addendum N's convention 'beta = 1' holds on (f)'s trained operator:
max_l |beta_l - 1| <= 0.01. Written before read_beta.py exists.
Usage: python test_beta.py RESULTS_JSON -> exit 0 pass, 1 fail."""
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    b = r["beta"]
    assert b and all(v is not None for v in b), "value is null (stub or not measured)"
    dev = max(abs(float(v) - 1.0) for v in b)
    assert dev <= 0.01, "max |beta-1| = {} > 0.01 (beta = {})".format(dev, b)
except (AssertionError, KeyError, TypeError) as e:
    print("RED N_beta_convention: {}".format(e)); sys.exit(1)
print("GREEN N_beta_convention"); sys.exit(0)
