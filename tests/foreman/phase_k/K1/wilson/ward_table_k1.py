"""Builds runs/ward_table.json (and ward_curves.json) from runs/ward_jobs.jsonl + logs. python ward_table_k1.py"""
import json, os, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
rd = os.path.join(HERE, "runs")
rows, curves, done = [], {}, {}
for j in [json.loads(l) for l in open(os.path.join(rd, "ward_jobs.jsonl")) if l.strip()]:
    n = j["name"]
    arm, seed = n.rsplit("_s", 1)
    recs = [json.loads(l) for l in open(os.path.join(rd, n, "log.jsonl"))]
    cfg = [r for r in recs if r["t"] == "config"][-1]
    steps = [r for r in recs if r["t"] == "step"]
    ev = [(r["step"], r["val_loss"]) for r in recs if r["t"] == "eval"]
    hook = os.path.join(HERE, "train_ladder_k1.py") if cfg["attn"] == "alibi" else cfg["attn"].rsplit(":", 1)[0]
    sh = cfg["harness_sha256"] if cfg["attn"] == "alibi" else cfg["attn_sha256"]
    row = {"name": n, "arm": arm, "seed": int(seed), "attn": cfg["attn"], "params": cfg["params"], "hook_file": hook, "hook_sha256": sh,
           "ckpt": os.path.join(rd, n, "ckpt.pt"), "job_start": j["start"], "job_end": j["end"], "rc": j["rc"]}
    if j["rc"] != 0 or any(r["t"] == "abort" for r in recs):
        row.update(status="failed", val_loss=None, tok_s=None, hours=round(j["wall_s"] / 3600, 3), peak_mem_gib=None)
    else:
        row.update(status="done", val_loss=round(ev[-1][1], 4), val_step=ev[-1][0], tok_s=round(statistics.mean([s["tok_s"] for s in steps[10:]])),
                   hours=round(j["wall_s"] / 3600, 3), train_hours=round(steps[-1]["wall_s"] / 3600, 3),
                   peak_mem_gib=round(max(s["peak_mem_gib"] for s in steps), 2), tokens=steps[-1]["tokens"],
                   tok_s_first1000=round(statistics.mean([s["tok_s"] for s in steps[10:1010]])), tok_s_last1000=round(statistics.mean([s["tok_s"] for s in steps[-1000:]])))
        done[(arm, int(seed))] = row["val_loss"]
    curves[n] = ev
    rows.append(row)
deltas = [{"seed": s, "f_R": done[("f_R", s)], "a_L": done[("a_L", s)], "delta": round(done[("f_R", s)] - done[("a_L", s)], 4)}
          for (a, s) in sorted(done) if a == "a_L" and ("f_R", s) in done]
json.dump({"rows": rows, "deltas": deltas}, open(os.path.join(rd, "ward_table.json"), "w"), indent=1)
json.dump(curves, open(os.path.join(rd, "ward_curves.json"), "w"))
for r in rows:
    print(json.dumps({k: r.get(k) for k in ("name", "status", "val_loss", "tok_s", "hours", "train_hours", "peak_mem_gib", "params", "hook_sha256", "tok_s_first1000", "tok_s_last1000")}))
print(json.dumps(deltas))
