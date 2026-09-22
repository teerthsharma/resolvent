"""Replacement-route pricing for the R-DIAG' kill. Registered 2026-09-23 00:45,
before measuring:

  (1) THEOREM CHECK: with beta pinned to 1 in every layer (evaluation only, same
      weights), gamma* = 1 / max_i W_ii >= 1 in every (window, layer, head)
      (W_ii = p_ii <= 1). Must be GREEN or the read is wrong.
  (2) PRICE: pinning beta = 1 is FREE iff held-out NLL rises by <= 0.01 nats
      (about 4% of C_win = 1.2757 - 1.0431 = 0.2326). Otherwise beta != 1 is
      load-bearing and a beta = 1 Mode B read describes a different model.
Module from env PIN_IMPL (default beta_pin). Exit 1 on any failure.
"""
import importlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
IMPL = os.environ.get("PIN_IMPL", "beta_pin")
TAG = "scratchpad/phase_j/N2/chase/test_beta_pin.py::{} [" + IMPL + "]"
_R = None
def R():
    global _R
    if _R is None:
        _R = importlib.import_module(IMPL).results()
    return _R
def test_beta1_gstar_at_least_1():
    r = R(); assert r["beta1_gstar_min"] >= 1.0, r["beta1_gstar_min"]
def test_beta_pin_free():
    r = R(); assert r["dnll"] <= 0.01, (r["nll_trained"], r["nll_beta1"], r["dnll"])
TESTS = [test_beta1_gstar_at_least_1, test_beta_pin_free]
def board(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    open(BOARD, "a", encoding="utf-8").write(json.dumps(kw) + "\n")
if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            t(); print("GREEN", t.__name__, flush=True)
            board(t="test", agent="Chase", status="green", name=TAG.format(t.__name__))
        except Exception as e:  # noqa: BLE001
            fails += 1; line = "RED   {} {}: {}".format(t.__name__, type(e).__name__, e)
            print(line, flush=True)
            board(t="test", agent="Chase", status="red", name=TAG.format(t.__name__), detail=line[:600])
    sys.exit(1 if fails else 0)
