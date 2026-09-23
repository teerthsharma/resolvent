"""Builds SESSION_DIR/mfu_table.json from its logs: python mfu_table_k1.py SESSION_DIR. MFU = (6N + 6*L*ctx*d)*tok_s/(peak_pre median)."""
import json, os, statistics, sys
RUNGS = {"R0": (4, 128), "R1": (6, 320), "R2": (8, 512), "R3": (12, 768)}
sd = sys.argv[1]
ses = json.load(open(os.path.join(sd, "session.json")))
med = json.load(open(os.path.join(sd, "peak_pre.json")))["median"]
rows = []
for j in [json.loads(l) for l in open(os.path.join(sd, "jobs.jsonl")) if l.strip()]:
    if not j["name"].startswith("probe_"):
        continue
    recs = [json.loads(l) for l in open(os.path.join(sd, "runs", j["name"], "log.jsonl"))]
    cfg = recs[0]
    steps = [r for r in recs if r["t"] == "step"]
    row = {"name": j["name"], "attn": cfg["attn"], "rung": cfg["rung"], "ctx": cfg["ctx"], "batch": cfg["batch"], "params": cfg["params"]}
    if j["rc"] == 3 or any(r["t"] == "abort" for r in recs):
        row.update(status="no_fit", tok_s=None, mfu=None, peak_mem_gib=None,
                   abort_peak_mem_gib=round(max(r["peak_mem_gib"] for r in recs if "peak_mem_gib" in r), 2))
    elif j["rc"] != 0 or len(steps) <= 10:
        row.update(status="error", tok_s=None, mfu=None, peak_mem_gib=None)
    else:
        L, d = RUNGS[cfg["rung"]]
        L, d = cfg.get("layers") or L, cfg.get("width") or d
        tok = statistics.mean([s["tok_s"] for s in steps[10:]])
        row.update(status="fit", tok_s=round(tok), mfu=round((6 * cfg["params"] + 6 * L * cfg["ctx"] * d) * tok / (med * 1e12), 3),
                   peak_mem_gib=round(max(s["peak_mem_gib"] for s in steps), 2), mfu_vs_26_77=round((6 * cfg["params"] + 6 * L * cfg["ctx"] * d) * tok / 26.77e12, 3))
    rows.append(row)
json.dump({"session": ses["session"], "peak_tflops": round(med, 2), "rows": rows}, open(os.path.join(sd, "mfu_table.json"), "w"), indent=1)
print(json.dumps({"peak_tflops": round(med, 2)}))
for r in rows:
    print(json.dumps(r))
