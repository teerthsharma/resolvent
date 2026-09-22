"""R-DIAG' tests. Registered bar (set before measuring, 2026-09-23 00:38):

  The Mode B read at gamma = 0.99 is WELL-POSED on the trained family iff
  gamma* = 1 / max_i W_ii >= 0.99 in every (window, layer, head); PAST A POLE
  if gamma* < 0.99 anywhere. Either result binds; the fraction is reported.

Prerequisites (must hold or the read is of the wrong object):
  rebuild fidelity (eval NLL within 2e-3 of the recorded 1.043056, CUDA vs CPU),
  learned beta equal to the state dict, the recomputed attention output equal to
  the model's own self_attn output, diag(W) real and diag(Re W) == diag(W).real.

Module under test from env RDIAG_IMPL (default rdiag_prime). Exit 1 on any failure.
Every result is appended to the board as a test event with "status" and "name".
"""
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
IMPL = os.environ.get("RDIAG_IMPL", "rdiag_prime")
TAG = "scratchpad/phase_j/N2/chase/test_rdiag_prime.py::{} [" + IMPL + "]"

BETA_SD = (1.0605749, 0.9300133, 0.8995796)   # Wilson, from the state dict
EVAL_REC = 1.043056309223175                   # run_record.json final_eval_loss (CUDA)
GAMMA = 0.99

_R = None


def R():
    global _R
    if _R is None:
        _R = importlib.import_module(IMPL).results()
    return _R


def test_rebuild_fidelity():
    r = R()
    assert abs(r["eval_nll"] - EVAL_REC) < 2e-3, (r["eval_nll"], EVAL_REC)


def test_beta_is_learned_beta():
    r = R()
    got = [L["beta"] for L in r["layers"]]
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, BETA_SD)), (got, BETA_SD)


def test_operator_is_model_path():
    r = R()
    assert r["attn_out_max_abs_diff"] <= 1e-6, r["attn_out_max_abs_diff"]


def test_diag_real_and_ReW_diag_equal():
    r = R()
    for L in r["layers"]:
        assert L["diag_imag_nonzero"] == 0, L["diag_imag_nonzero"]
        assert L["ReW_diag_equals_W_diag_bitwise"], L["layer"]


def test_bar_well_posed_gamma_099():
    """THE REGISTERED BAR. RED here means PAST A POLE."""
    r = R()
    bad = {L["layer"]: (L["f32"]["n_pairs_gstar_lt_099"], L["f32"]["n_pairs"],
                        L["f32"]["gstar_min"]) for L in r["layers"]}
    assert all(v[0] == 0 for v in bad.values()), bad


def test_bar_well_posed_gamma_099_ReW():
    r = R()
    bad = {L["layer"]: (L["ReW"]["n_pairs_gstar_lt_099"], L["ReW"]["gstar_min"])
           for L in r["layers"]}
    assert all(v[0] == 0 for v in bad.values()), bad


def test_no_absorber_but_position_0():
    r = R()
    n = {L["layer"]: L["f64"]["absorbers_excl_pos0"] for L in r["layers"]}
    assert all(v == 0 for v in n.values()), n


TESTS = [test_rebuild_fidelity, test_beta_is_learned_beta, test_operator_is_model_path,
         test_diag_real_and_ReW_diag_equal, test_bar_well_posed_gamma_099,
         test_bar_well_posed_gamma_099_ReW, test_no_absorber_but_position_0]


def board(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw) + "\n")


if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            t()
            print("GREEN", t.__name__, flush=True)
            board(t="test", agent="Chase", status="green", name=TAG.format(t.__name__))
        except Exception as e:  # noqa: BLE001 -- any failure is RED, named
            fails += 1
            line = "RED   {} {}: {}".format(t.__name__, type(e).__name__, e)
            print(line, flush=True)
            board(t="test", agent="Chase", status="red", name=TAG.format(t.__name__),
                  detail=line[:600])
    sys.exit(1 if fails else 0)
