"""M0: the engine must be trusted before any candidate.

    python sun/daedalus/engine/m0.py

Writes results/m0_redteam.json (20 planted cheats + honest controls + power checks)
and results/known_verdicts.json (5 verdicts; 1-2 read from Cameron's re-run file).
Kill (registry/bars.json m0_kill): any cheat ACCEPTed, < 4/5 verdicts, or no honest
control through V0, V1, V4, V2 -> NOT TRUSTED.
"""
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import verifier as V  # noqa: E402

RES = os.path.join(D, "results")


def main():
    t0 = time.time()
    bars_sha = hashlib.sha256(open(os.path.join(D, "registry", "bars.json"), "rb").read()).hexdigest()
    cheats = json.load(open(os.path.join(D, "cheats", "CHEATS.json")))
    rows = []
    for c in cheats:
        r = V.verify(os.path.join(D, c["dir"]))
        r.update(id=c["id"], expected_stage=c["expected_stage"], record_cite=c["record_cite"],
                 caught=r["verdict"] == "REJECT")
        rows.append(r)
        print(c["id"], r["verdict"], r["stage"], r["reason"][:90], flush=True)
    attacks = {}
    for a in sorted(os.listdir(os.path.join(D, "attacks"))):  # the lead's own post-plant attacks (not in the 20)
        attacks[a] = V.verify(os.path.join(D, "attacks", a))
        print("attack", a, attacks[a]["verdict"], attacks[a]["stage"], attacks[a]["reason"][:90], flush=True)
    controls = {}
    for name in ("gru", "alibi_twin", "a5_far_band_bed", "bed_kp_far"):
        controls[name] = V.verify(os.path.join(D, "controls", name))
        print("control", name, controls[name]["verdict"], controls[name]["stage"], controls[name]["reason"][:90], flush=True)
    # power check: V3 and V5 can pass -- the GRU against a pool that lacks the GRU
    power = V.verify(os.path.join(D, "controls", "gru"), pool=["alibi_twin"])
    print("power gru vs [alibi_twin]", power["verdict"], power["stage"], power["reason"][:90], flush=True)
    g = controls["gru"]
    honest_ok = g.get("stage") in ("V3", "V5") and g.get("band_acc") is not None
    caught = sum(r["caught"] for r in rows)
    red = {"when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bars_sha256": bars_sha,
           "machine": platform.node(), "python": platform.python_version(),
           "cheats_rejected": f"{caught}/{len(rows)}",
           "by_stage": {s: sum(1 for r in rows if r["caught"] and r["stage"] == s) for s in sorted({r["stage"] for r in rows})},
           "expected_stage_matches": sum(1 for r in rows if r["stage"] == r["expected_stage"]),
           "honest_control_passes_V0_V1_V4_V2": honest_ok,
           "power_check_V3_V5": {"verdict": power["verdict"], "stage": power["stage"], "reason": power["reason"],
                                 "v3": power.get("v3"), "v5": power.get("v5")},
           "controls": controls, "attacks": attacks, "errors": sum(r["verdict"] == "ERROR" for r in rows), "cheats": rows, "wall_s": round(time.time() - t0, 1)}
    json.dump(red, open(os.path.join(RES, "m0_redteam.json"), "w"), indent=2, default=float)

    # ---------------- known verdicts
    kv = {}
    lm = os.path.join(RES, "verdicts_12.json")
    if os.path.exists(lm):
        v12 = json.load(open(lm))
        for k in ("V1_alibi_beats_family", "V2_fox_recovers_cwin"):
            kv[k] = dict(v12[k], source="results/verdicts_12.json (Cameron)")
    else:
        for k in ("V1_alibi_beats_family", "V2_fox_recovers_cwin"):
            kv[k] = {"reproduced": False, "method": "missing", "why": "results/verdicts_12.json not present"}
    sys.path.insert(0, V.beds.N1_CHASE)
    import n_impl
    t10 = n_impl.t10()
    kv["V3_q2_closure"] = {"method": "rerun", "cmd": "n_impl.t10() (tests/foreman/phase_j/N1/chase/n_impl.py:367)", **t10,
                           "reproduced": t10["corrected_quats"] == 120 and t10["corrected_rots"] == 60
                           and t10["corrected_closed"] and t10["has_minus_one"] and not t10["q2_closed"]}
    x3 = next(r for r in rows if r["id"] == "c11")
    kv["V4_x3_recency"] = {"method": "rerun (engine V2 on the X3 bed proposal, cheat c11)", "floors": x3.get("floors"),
                           "engine_verdict": x3["reason"],
                           "reproduced": x3["stage"] == "V2" and x3.get("floors", {}).get("recency_1.0", 0) >= 0.99}
    bk = next(r for r in rows if r["id"] == "c10")
    kp = V.verify(os.path.join(D, "controls", "bed_kp_beyond"))
    far = controls["bed_kp_far"]
    kv["V5_phase_k_window"] = {"method": "rerun (engine V2 shortcut library)",
                               "bed_k_beyond": {"stage": bk["stage"], "reason": bk["reason"]},
                               "bed_kp_beyond": {"stage": kp["stage"], "reason": kp["reason"]},
                               "bed_kp_far_control": {"verdict": far["verdict"], "reason": far["reason"]},
                               "reproduced": bk["stage"] == "V2" and kp["stage"] == "V2" and far["verdict"] == "ACCEPT"}
    n_rep = sum(bool(v.get("reproduced")) for v in kv.values())
    out = {"when_utc": red["when_utc"], "bars_sha256": bars_sha, "reproduced": f"{n_rep}/5", "verdicts": kv}
    json.dump(out, open(os.path.join(RES, "known_verdicts.json"), "w"), indent=2, default=float)
    trusted = caught == len(rows) == 20 and n_rep >= 4 and honest_ok
    print(f"M0: cheats {caught}/{len(rows)} rejected, verdicts {n_rep}/5, honest control {honest_ok} -> "
          f"{'TRUSTED' if trusted else 'NOT TRUSTED'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
