"""Ranker suite (bars_r2 r2_b): 4 planted ranker cheats + 2 honest controls.

    python daedalus/engine/r2_rankers.py <out.json>

A cheat counts as caught only if REJECTed at its must_catch_at stage; a control is admissible iff it is not
rejected at an integrity stage (V0, V0.sandbox, V0.init, V1, V4, V2.ceiling, V2.order). ERROR is neither.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import verifier as V  # noqa: E402

INTEGRITY = {"V0", "V0.sandbox", "V0.init", "V1", "V4", "V2.ceiling", "V2.order", "infra"}


def main(out_path):
    t0 = time.time()
    rows, ctl = [], {}
    for c in json.load(open(os.path.join(D, "cheats", "RANKER_CHEATS.json"))):
        r = V.verify(os.path.join(D, c["dir"]))
        r.update(id=c["id"], must_catch_at=c["must_catch_at"], caught=r["verdict"] == "REJECT" and r["stage"] == c["must_catch_at"])
        rows.append(r)
        print(c["id"], r["verdict"], r["stage"], r["reason"][:110], flush=True)
    for name in ("r_latent_distance", "r_deepsets"):
        r = V.verify(os.path.join(D, "controls", name))
        r["admissible"] = r["verdict"] != "ERROR" and not (r["verdict"] == "REJECT" and r["stage"] in INTEGRITY)
        ctl[name] = r
        print("control", name, r["verdict"], r["stage"], r["reason"][:110], flush=True)
    s = {"cheats_caught_at_named_stage": f"{sum(r['caught'] for r in rows)}/{len(rows)}",
         "controls_admissible": f"{sum(r['admissible'] for r in ctl.values())}/{len(ctl)}",
         "errors": sum(r["verdict"] == "ERROR" for r in rows + list(ctl.values())),
         "cheats": rows, "controls": ctl, "wall_s": round(time.time() - t0, 1)}
    json.dump(s, open(out_path, "w"), indent=2, default=float)
    print(s["cheats_caught_at_named_stage"], "caught;", s["controls_admissible"], "admissible;", s["errors"], "errors;", s["wall_s"], "s")


if __name__ == "__main__":
    main(sys.argv[1])
