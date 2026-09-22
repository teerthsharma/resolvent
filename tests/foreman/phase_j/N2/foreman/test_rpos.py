"""R-POS registered bar (RECORD_N.md, "Kill row (R-POS)"); not edited after results.

Recovery per seed = (loss(a) - loss(arm)) / (loss(a) - loss(f)), grid losses read
from tests/foreman/design4x5/design4x5_results.jsonl.
PASS    : a_rope recovers >= 0.8 at all three seeds (the leap binds).
KILL    : a_rope AND a_alibi recover < 0.5 at all three seeds.
NEITHER : otherwise.
This test asserts PASS; it exits non-zero on KILL or NEITHER and prints which.
--stub : no losses (NaN) with grid digests, to prove the assertion fires.
"""
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = r"C:\Users\seal\Desktop\New folder (32)\tests\foreman\design4x5\design4x5_results.jsonl"
ROWS = os.path.join(HERE, "rpos_results.jsonl")
SEEDS = (0, 1, 2)
ARMS = ("a_rope", "a_alibi")


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def verdict(rec):
    rope = [rec[("a_rope", s)] for s in SEEDS]
    alibi = [rec[("a_alibi", s)] for s in SEEDS]
    if all(x >= 0.8 for x in rope):
        return "PASS"
    if all(x < 0.5 for x in rope) and all(x < 0.5 for x in alibi):
        return "KILL"
    return "NEITHER"


def main(stub):
    grid = {(r["arm"], r["split_seed"]): r for r in load(GRID) if r.get("stage") == "cell"}
    if stub:
        rows = {(a, s): dict(final_eval_loss=float("nan"), crn_digest_recomputed=grid[("a", s)]["crn_digest"])
                for a in ARMS for s in SEEDS}
    else:
        rows = {(r["arm"], r["split_seed"]): r for r in load(ROWS)}
    rec = {}
    for a in ARMS:
        for s in SEEDS:
            assert (a, s) in rows, "missing cell {} ss{}".format(a, s)
            assert rows[(a, s)]["crn_digest_recomputed"] == grid[("a", s)]["crn_digest"] == grid[("f", s)]["crn_digest"], \
                "VOID: CRN digest mismatch {} ss{}".format(a, s)
            la, lf = grid[("a", s)]["final_eval_loss"], grid[("f", s)]["final_eval_loss"]
            rec[(a, s)] = (la - rows[(a, s)]["final_eval_loss"]) / (la - lf)
    for a in ARMS:
        v = [rec[(a, s)] for s in SEEDS]
        print("{:8s} recovery per seed {}  mean {:.4f}".format(
            a, ["{:.4f}".format(x) for x in v], sum(v) / 3), flush=True)
    vd = verdict(rec)
    fmt = lambda a: "[" + ", ".join("{:.4f}".format(rec[(a, s)]) for s in SEEDS) + "]"
    assert vd == "PASS", "R-POS verdict {}, not PASS: a_rope {} a_alibi {}".format(vd, fmt("a_rope"), fmt("a_alibi"))
    print("R-POS verdict PASS", flush=True)


if __name__ == "__main__":
    main("--stub" in sys.argv)
