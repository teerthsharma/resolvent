"""Addendum N rebuilt from text: one named test per instance / break.
Usage: NIMPL=n_impl_stub python test_n.py   (RED run)   |   python test_n.py
Exits 1 if any assertion fails. Appends one board event per test."""
import importlib, json, os, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
MOD = os.environ.get("NIMPL", "n_impl")
impl = importlib.import_module(MOD)
TOL = 1e-12
REC = {}


def T1():
    r = impl.t1(); REC["T1"] = r
    assert r["max_eig_minus_diag"] <= TOL, r
    assert r["unit_diag"] == [0], r


def T2():
    r = impl.t2(); REC["T2"] = r
    e = r["err"]
    for a, b in zip(e, e[1:]):
        assert 5 <= a / b <= 20, ("not O(1-gamma)", e)
    assert r["resid_change"] <= 1e-2 * r["input_change"], r
    assert r["s2_over_s1"] <= 1e-3, r


def T3():
    r = impl.t3(); REC["T3"] = r
    assert r["sim_resid"] <= TOL and r["eig_shift"] <= TOL, r
    # bitwise was a defect of this test: W (max-shifted softmax) and W_G (unshifted,
    # Z = sum |G| e^s with |G| = 1 +- 1ulp) are different arithmetic; the claim is equality
    assert r["re_diag_max_diff"] <= TOL, r
    o = r["off_v0"]
    assert all(5 <= a / b <= 20 for a, b in zip(o, o[1:])), ("off-v0 not O(1-gamma)", o)


def T4():
    r = impl.t4(); REC["T4"] = r
    assert r["sim_resid"] <= TOL, r


def T5():
    r = impl.t5(); REC["T5"] = r
    assert r["lambda_law_resid"] <= TOL, r
    assert r["unit_diag"] == [0, 16, 40], r
    s = r["sigma"]
    assert s[3] / s[2] < 1e-2, ("rank != #segments", s)


def T6():
    r = impl.t6(); REC["T6"] = r
    for n in ("64", "1024"):
        assert 0 < r[n]["mixed"] <= r[n]["transient"], r


def T7():
    r = impl.t7(); REC["T7"] = r
    for n in ("64", "1024"):
        x = r[n]
        assert x["nilpotent_exact"], x          # N^(D+1) == 0 bitwise
        assert x["err_D"] <= TOL, x
        assert x["err_Dm1"] > TOL, x            # D is tight


def T8():
    r = impl.t8(); REC["T8"] = r
    for n in ("64", "1024"):
        x = r[n]
        assert x["sparse_mismatch"] == 0 and x["sparse_false_mass"] == 0.0, x
        assert abs(x["leak"] - x["leak_law"]) <= TOL, x


def T9():
    r = impl.t9(); REC["T9"] = r
    assert r["fwd_vs_dense"] <= TOL, r


def T10():
    r = impl.t10(); REC["T10"] = r
    assert r["q2_as_written"] > 5000, r
    assert r["corrected_quats"] == 120 and r["corrected_rots"] == 60 and r["has_minus_one"], r


def X1():
    r = impl.x1(); REC["X1"] = r
    assert r["sim_resid"] <= TOL and r["eig_vs_diag"] <= TOL, r
    assert r["absorbers"] == [0, 16, 40], r


def X2a():
    r = impl.x2a(); REC["X2a"] = r
    assert r["change_direct"] <= TOL and r["change_scan"] <= TOL, r


def X2b():
    r = impl.x2b(); REC["X2b"] = r
    # author's reported ordering: global prefix MORE accurate than segmented
    assert r["global"] < r["segmented"], r


def X3a():
    r = impl.x3a(); REC["X3a"] = r
    assert r["max_err_deg"] < 1e-5, r


def X3b():
    r = impl.x3b(); REC["X3b"] = r
    a = r["acc"]
    assert a[0] < a[1] < a[2] and a[0] < 1.0, r


def X4():
    r = impl.x4(); REC["X4"] = r
    assert r["H0"] == [4, 3, 0, 1], r
    assert [round(g, 3) for g in r["gap"]] == [2.0, 2.0, 0.012, 0.047], r


# ---- breaks -------------------------------------------------------------
def B1_D1_unit_count_beta1():
    r = impl.b1(); REC["B1"] = r
    x = r["1.0"]
    assert x["unit"] == x["structural"], x


def B1_D1_unit_count_trained_beta():
    r = REC.get("B1") or impl.b1()
    for b in ("1.09", "0.96", "0.99"):
        assert r[b]["unit"] == r[b]["structural"], (b, r[b])


def B2_D3_as_written_complex_gate():
    r = impl.b2(); REC["B2"] = r
    assert r["as_written"] <= TOL, r


def B2r_D3_modulus_rho():
    r = REC.get("B2") or impl.b2()
    assert r["modulus_rho"] <= TOL, r


def B3_D5_dense_depth():
    r = impl.b3(); REC["B3"] = r
    for n in ("64", "1024"):
        assert r[n]["D"] == int(n) - 1, r


def B4_RDIAG_gate_route():
    r = impl.b4(); REC["B4"] = r
    x = r["gate"]
    assert x["diag_bitwise"] and x["count_equal"], x


def B4_RDIAG_logit_route_diag():
    r = REC.get("B4") or impl.b4()
    assert r["logit"]["diag_bitwise"], r["logit"]


def B4_RDIAG_can_fire_at_trained_beta():
    r = REC.get("B4") or impl.b4()
    for b in ("1.09", "0.96", "0.99"):
        assert r["beta"][b]["count"] > 0, (b, r["beta"][b])


def B5_no_pole_inside_unit_gamma():
    r = REC.get("B1") or impl.b1()
    for b in ("1.09", "0.96", "0.99"):
        assert r[b]["max_diag"] <= 1.0, (b, r[b])


def KR():
    r = impl.kr(); REC["KR"] = r
    for d in ("0.05", "0.2", "0.5"):
        for sl in ("1.0", "2.0", "4.0"):
            assert r[d][sl] == 1.0, (d, sl, r[d])


def PF():
    r = impl.pf(); REC["PF"] = r
    author = [0.0299, 0.0980, 0.2509, 0.3305, 0.3330, 0.3333, 0.3333]
    for a, b in zip(r["cost"], author):
        assert abs(a - b) <= 5e-4, (r["cost"], author)


TESTS = [KR, PF, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, X1, X2a, X2b, X3a, X3b, X4,
         B1_D1_unit_count_beta1, B1_D1_unit_count_trained_beta,
         B2_D3_as_written_complex_gate, B2r_D3_modulus_rho, B3_D5_dense_depth,
         B4_RDIAG_gate_route, B4_RDIAG_logit_route_diag,
         B4_RDIAG_can_fire_at_trained_beta, B5_no_pole_inside_unit_gamma]

if __name__ == "__main__":
    only = sys.argv[1:]
    bad = 0
    with open(BOARD, "a", encoding="utf-8") as fb:
        for t in TESTS:
            if only and t.__name__ not in only:
                continue
            try:
                t(); st, why = "green", ""
            except BaseException as e:  # noqa: BLE001 - report every failure
                st, why = "red", f"{type(e).__name__}: {str(e)[:300]}"
                bad += 1
            print(f"{st.upper():5s} {t.__name__} {why}", flush=True)
            fb.write(json.dumps({"t": "test", "agent": "Chase", "status": st,
                                 "name": f"scratchpad/phase_j/N1/chase/test_n.py::{t.__name__}"
                                         + ("" if MOD == "n_impl" else f" [{MOD}]")}) + "\n")
    with open(os.path.join(HERE, f"results_{MOD}.json"), "w") as f:
        json.dump(REC, f, indent=1, default=str)
    sys.exit(1 if bad else 0)
