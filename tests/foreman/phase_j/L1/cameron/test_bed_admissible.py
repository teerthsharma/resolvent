"""Bed admissibility for contract 5.3 (+0.10 margin over the best control).
A bed is admissible iff, at its eval region (ply >= 40 / prefix >= 32):
  A1 a commuting-control ceiling is measured (multiset Bayes, plug-in upper bound) and <= 0.90
  A2 a plain softmax twin was trained on the bed: loss plateaued, and it LEARNED the task where
     the task is short (acc at the shortest reported prefix >= 0.90) -- else its long-range
     number is a broken run, not a floor
  A3 max(every measured floor, twin) + 0.10 <= 1.0  -- the margin is arithmetically possible
  A4 the SU(2) state carries the label: an exact construction on the arms' own scan
     (fq_construction.json beside the input) scores >= best control + 0.10
Usage: python test_bed_admissible.py <result.json>   (board_bed_result.json or shell_bed_floors.json)
Exit 1 with a RED line naming every failed clause; exit 0 with GREEN naming the admissible beds.
"""
import json, sys


def board_clauses(d):
    fl = d["floors"]["ply>=40"]
    best = max(v["per_square_acc"] for v in fl.values())
    who = max(fl, key=lambda k: fl[k]["per_square_acc"])
    fails = ["A1: no commuting-control ceiling measured",
             "A2: no softmax twin trained on the bed",
             "A4: no SU(2) construction on the bed"]
    if best + 0.10 > 1.0:
        fails.append("A3: best floor %s = %.4f, +0.10 = %.4f > 1.0" % (who, best, best + 0.10))
    return {"board ply>=40": fails}


def shell_clauses(d):
    out = {}
    for name, b in d["beds"].items():
        fails = []
        ceil = b["F1"]["R256"]["mean_ge32"]["plugin"]
        if not ceil <= 0.90:
            fails.append("A1: multiset ceiling %.4f > 0.90" % ceil)
        f3 = b.get("F3")
        if not f3 or "mean_ge32_acc" not in f3:
            fails.append("A2: no softmax twin trained on the bed")
            twin = 0.0
        else:
            twin = f3["mean_ge32_acc"]
            short = f3["per_position_acc"][str(min(int(k) for k in f3["per_position_acc"]))]
            short = short["acc"] if isinstance(short, dict) else short
            if not f3.get("plateau"):
                fails.append("A2: twin loss did not plateau")
            if not short >= 0.90:
                fails.append("A2: twin never learned the task (short-prefix acc %.4f < 0.90)" % short)
        floors = {"chance": b["F0"]["chance"], "multiset": ceil,
                  "window8": b["F2"]["w8"]["mean_ge32"], "twin": twin}
        who = max(floors, key=floors.get)
        if floors[who] + 0.10 > 1.0:
            fails.append("A3: best control %s = %.4f, +0.10 > 1.0" % (who, floors[who]))
        fq = CONS.get(name)
        if not fq:
            fails.append("A4: no SU(2) construction on the bed")
        elif not fq["acc_mean_ge32"] >= floors[who] + 0.10:
            fails.append("A4: construction %.4f < best control + 0.10" % fq["acc_mean_ge32"])
        out[name + " prefix>=32 (best control %s %.4f)" % (who, floors[who])] = fails
    return out


CONS = {}

if __name__ == "__main__":
    import os
    d = json.load(open(sys.argv[1]))
    cp = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "fq_construction.json")
    if os.path.exists(cp):
        CONS.update(json.load(open(cp)))
    res = board_clauses(d) if "floors" in d else shell_clauses(d)
    ok = [k for k, v in res.items() if not v]
    for k, v in res.items():
        print(("ADMISSIBLE " if not v else "INADMISSIBLE ") + k + ("" if not v else ": " + " | ".join(v)))
    if not ok:
        print("RED: no admissible bed in " + sys.argv[1])
        sys.exit(1)
    print("GREEN: admissible " + ", ".join(ok))
