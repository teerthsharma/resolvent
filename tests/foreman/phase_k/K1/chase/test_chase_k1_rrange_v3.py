"""Chase K1 R-RANGE v3. Registered 2026-09-23 ~06:25 after rrange_fR_depth_R0_v2 / rrange_fR_span_R0_v2 came back GREEN
with range = inf on both heads of far_fR_s0, a checkpoint that did not learn its bed (its own result.json: acc 0.2026 at
n 1024). A head with near-uniform attention has unbounded range, so v2 cannot tell a learned long-range channel from a
diffuse head. Written before the far_fR_s0 single-hop masses (res_hop_m) were looked at. RED first with CHASE_STUB=1.
Reads the rows already written by the instrument (ckpt_rows_v2.jsonl, last record per checkpoint path); no GPU.
 rrange_fR_dressed_R0  L-TRAINED premise: the Cameron (f_R) R0 checkpoint's result.json eval["1024"]["acc"] >= 0.8 (the
                       contract's learnability line). Bar: >= 1 checkpoint meets the premise, and on each one, >= 1
                       resolvent-layer head at S 16384 has bare range (single-hop band mass of W, 1/res_hop_m) < the bed's
                       max chain depth AND dressed range (1/res_m) >= that depth: the resolvent, not diffuseness, supplies
                       the range. Checkpoints failing the premise are listed, not scored.
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_chase_k1_ckpt_v2 as T


def rng(m):
    return 1.0 / m if m > 0 else float("inf")


@T.bar("rrange_fR_dressed_R0")
def t_dressed():
    T._stub()
    last = {}
    for l in open(os.path.join(HERE, "ckpt_rows_v2.jsonl")):
        r = json.loads(l)
        last[r["path"]] = r
    out, n_ok, ok = {}, 0, True
    for p, r in last.items():
        if r["lane"] != "cameron" or r["arm"] != "fR":
            continue
        acc = json.load(open(os.path.join(os.path.dirname(p), "result.json")))["eval"]["1024"]["acc"]
        row = r["S"]["16384"]
        bare = [rng(m) for m in row["res_hop_m"]]
        dressed = row["res_range"]
        need = row["bed_max_depth"]
        name = os.path.basename(os.path.dirname(p))
        if acc < 0.8:
            out[name] = {"premise": False, "acc1024": acc, "bare": bare, "dressed": dressed, "need": need}
            continue
        n_ok += 1
        good = any(b < need <= d for b, d in zip(bare, dressed))
        ok &= good
        out[name] = {"premise": True, "acc1024": acc, "bare": bare, "dressed": dressed, "need": need, "ok": good}
    return ok and n_ok >= 1, {"n_premise": n_ok, **out}


if __name__ == "__main__":
    t_dressed()
    print("SUMMARY", json.dumps(T.results))
    sys.exit(1 if "red" in T.results.values() else 0)
