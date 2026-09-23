"""wilson.k1.ward -- the R0 language-model runs table is what the logs, the job records and the checkpoints say.
python test_ward_k1.py RUNS_DIR   -> checks RUNS_DIR/ward_table.json against RUNS_DIR/ward_jobs.jsonl, RUNS_DIR/<name>/log.jsonl, ckpts
python test_ward_k1.py --stub     -> the same checks on a built-bad directory (must be RED; prints which checks fired)
Exit 1 if any row fails. Stated digits: val loss 4 decimals, tok/s integer, hours 3 decimals, memory 2 decimals.

 row_present / row_orphan  every ward job has exactly one row; every row names a ward job
 failed_reported  a job with rc != 0 or an abort record is reported status "failed" with val_loss null
 done_rc0         a "done" row has rc 0 and no abort record
 done_tokens      config --tokens == 144,552,960 (20 x 7,227,648) and the last step == tokens // (batch*accum*ctx) == number of step records
 done_seeds       config seed == row seed, eval_seed == 0, deterministic off
 done_val         reported val loss == the eval at the last step (5e-5)
 done_curve       an eval record at every multiple of eval_every and at the last step
 done_tok_s       reported tok/s == mean logged tok/s over steps 11+ (0.5)
 done_hours       reported hours == job wall_s / 3600 (5e-4)
 done_mem         logged peak memory < device total memory; reported to 0.005 GiB
 done_params      reported params == config params
 done_ckpt        the checkpoint at the reported path loads and its step == the last step
 done_sha         reported hook sha == the sha in the config == sha256 of that file on disk now
                  (alibi: config harness_sha256 and train_ladder_k1.py; hooks: config attn_sha256 and the --attn file)
 done_spill       no 500-step window's median tok/s is below 0.5x the run's median (a WDDM spill or contention)
 done_clean       no foreign GPU process was seen during the run
 delta            each reported delta == (f_R val loss) - (a_L val loss) of the same seed's done rows (1e-4)"""
import hashlib, json, os, statistics, sys, tempfile

TOKENS = 20 * 7227648
ALL = ["row_present", "row_orphan", "failed_reported", "done_rc0", "done_tokens", "done_seeds", "done_val", "done_curve",
       "done_tok_s", "done_hours", "done_mem", "done_params", "done_ckpt", "done_sha", "done_spill", "done_clean", "delta"]
HERE = os.path.dirname(os.path.abspath(__file__))


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def ckpt_step(p):
    import torch
    try:
        return torch.load(p, map_location="cpu", weights_only=False)["step"]
    except Exception as e:
        return repr(e)[:80]


def check(rd):
    fails = []
    F = lambda n, r, m: fails.append((n, r, m))
    tab = json.load(open(os.path.join(rd, "ward_table.json")))
    jobs = jl(os.path.join(rd, "ward_jobs.jsonl"))
    rows = {}
    for r in tab["rows"]:
        rows.setdefault(r["name"], []).append(r)
    for n in rows:
        if n not in {j["name"] for j in jobs}: F("row_orphan", n, "row names no ward job")
    done = {}
    for j in jobs:
        n = j["name"]
        if len(rows.get(n, [])) != 1:
            F("row_present", n, f"{len(rows.get(n, []))} rows"); continue
        r = rows[n][0]
        recs = jl(os.path.join(rd, n, "log.jsonl"))
        cfg = [x for x in recs if x["t"] == "config"][-1]
        steps = [x for x in recs if x["t"] == "step"]
        evals = {x["step"]: x["val_loss"] for x in recs if x["t"] == "eval"}
        aborts = [x for x in recs if x["t"] == "abort"]
        if (j["rc"] != 0 or aborts) and not (r["status"] == "failed" and r["val_loss"] is None):
            F("failed_reported", n, f"rc {j['rc']} aborts {len(aborts)} reported {r['status']}")
        if r["status"] != "done":
            continue
        if j["rc"] != 0 or aborts: F("done_rc0", n, f"rc {j['rc']} aborts {len(aborts)}")
        tps = cfg["batch"] * cfg["accum"] * cfg["ctx"]
        last = steps[-1]["step"] if steps else -1
        if not (cfg["tokens"] == TOKENS and last == TOKENS // tps == len(steps)): F("done_tokens", n, f"tokens {cfg['tokens']} last {last} n {len(steps)} want {TOKENS // tps}")
        if not (cfg["seed"] == r["seed"] and cfg.get("eval_seed") == 0 and not cfg["deterministic"]): F("done_seeds", n, f"seed {cfg['seed']} eval_seed {cfg.get('eval_seed')} det {cfg['deterministic']}")
        if r["val_loss"] is None or last not in evals or abs(r["val_loss"] - evals[last]) > 5e-5: F("done_val", n, f"reported {r['val_loss']} eval@last {evals.get(last)}")
        want = set(range(cfg["eval_every"], last + 1, cfg["eval_every"])) | {last}
        if not want <= set(evals): F("done_curve", n, f"{len(want - set(evals))} evals missing")
        toks = [s["tok_s"] for s in steps[10:]] or [0.0]
        if r["tok_s"] is None or abs(r["tok_s"] - statistics.mean(toks)) > 0.5: F("done_tok_s", n, f"reported {r['tok_s']} logged {statistics.mean(toks):.2f}")
        if r["hours"] is None or abs(r["hours"] - j["wall_s"] / 3600) > 5e-4: F("done_hours", n, f"reported {r['hours']} job {j['wall_s'] / 3600:.4f}")
        pm = max((s["peak_mem_gib"] for s in steps), default=99.0)
        if not (pm < (cfg.get("device_total_gib") or 0)) or r["peak_mem_gib"] is None or abs(r["peak_mem_gib"] - pm) > 0.005:
            F("done_mem", n, f"logged {pm} device {cfg.get('device_total_gib')} reported {r['peak_mem_gib']}")
        if r["params"] != cfg["params"]: F("done_params", n, f"{r['params']} vs {cfg['params']}")
        st = ckpt_step(r["ckpt"]) if r.get("ckpt") else None
        if st != last: F("done_ckpt", n, f"ckpt step {st} last {last}")
        if cfg["attn"] == "alibi":
            cs, fp = cfg.get("harness_sha256"), os.path.join(HERE, "train_ladder_k1.py")
        else:
            cs, fp = cfg.get("attn_sha256"), cfg["attn"].rsplit(":", 1)[0]
        if not (r["hook_sha256"] == cs == sha(fp)): F("done_sha", n, f"reported {r['hook_sha256']} config {cs} disk {sha(fp)}")
        allt = [s["tok_s"] for s in steps]
        med = statistics.median(allt) if allt else 0
        wins = [statistics.median(allt[i:i + 500]) for i in range(0, max(1, len(allt) - 499), 500)] if len(allt) >= 500 else [0]
        if min(wins) < 0.5 * med: F("done_spill", n, f"min window median {min(wins):.0f} vs run median {med:.0f}")
        if j.get("foreign_gpu_pids_seen"): F("done_clean", n, f"foreign {j['foreign_gpu_pids_seen']}")
        done[(r["arm"], r["seed"])] = r["val_loss"]
    for dl in tab.get("deltas", []):
        a, f = done.get(("a_L", dl["seed"])), done.get(("f_R", dl["seed"]))
        if a is None or f is None or dl["delta"] is None or abs(dl["delta"] - (f - a)) > 1e-4: F("delta", dl["seed"], f"reported {dl['delta']} f_R {f} a_L {a}")
    return fails


def build_stub():
    import torch
    d = tempfile.mkdtemp()
    tps = 8 * 1024
    last = TOKENS // tps

    def run(n, seed, attn, rc, n_steps, tok, mem, evals, abort=False, ckstep=None, sd=0):
        os.makedirs(os.path.join(d, n))
        cfg = {"t": "config", "attn": attn, "batch": 8, "accum": 1, "ctx": 1024, "tokens": TOKENS, "seed": seed, "eval_seed": sd,
               "deterministic": False, "eval_every": 250, "params": 7227648, "device_total_gib": 7.996, "harness_sha256": "x", "attn_sha256": None}
        with open(os.path.join(d, n, "log.jsonl"), "w") as g:
            g.write(json.dumps(cfg) + "\n")
            for s in range(1, n_steps + 1):
                g.write(json.dumps({"t": "step", "step": s, "tok_s": tok(s), "peak_mem_gib": mem}) + "\n")
                if s in evals: g.write(json.dumps({"t": "eval", "step": s, "val_loss": 4.0}) + "\n")
            if abort: g.write(json.dumps({"t": "abort", "reason": "allocated > device memory"}) + "\n")
        if ckstep is not None: torch.save({"step": ckstep}, os.path.join(d, n, "ckpt.pt"))
        return {"name": n, "rc": rc, "wall_s": 1800.0, "foreign_gpu_pids_seen": [99] if n == "a_L_s2" else []}

    full = set(range(250, last + 1, 250)) | {last}
    jobs = [run("a_L_s1", 1, "alibi", 0, last, lambda s: 94000.0, 5.6, full, ckstep=last),
            run("a_L_s2", 2, "alibi", 0, last - 5, lambda s: 94000.0 if s < 3000 else 900.0, 9.1, {250}, ckstep=3, sd=7),
            run("f_R_s1", 1, "alibi", 3, 1, lambda s: 1.0, 11.0, set(), abort=True),
            run("f_R_s2", 2, "alibi", 0, last, lambda s: 94000.0, 5.6, full, ckstep=last),
            run("x_s1", 1, "alibi", 0, last, lambda s: 94000.0, 5.6, full, ckstep=last)]
    with open(os.path.join(d, "ward_jobs.jsonl"), "w") as f:
        for j in jobs: f.write(json.dumps(j) + "\n")
    R = lambda n, arm, seed, st, vl, tok, h, mem, p, sh: {"name": n, "arm": arm, "seed": seed, "status": st, "val_loss": vl, "tok_s": tok, "hours": h,
                                                           "peak_mem_gib": mem, "params": p, "ckpt": os.path.join(d, n, "ckpt.pt"), "hook_sha256": sh}
    rows = [R("a_L_s1", "a_L", 1, "done", 4.0, 94000, 0.5, 5.6, 7227648, "x"),
            R("a_L_s2", "a_L", 3, "done", 3.9, 1, 0.9, 5.6, 1, "y"),
            R("f_R_s1", "f_R", 1, "done", 4.0, 94000, 0.5, 5.6, 7227648, "x"),
            R("f_R_s2", "f_R", 2, "done", 4.0, 94000, 0.5, 5.6, 7227648, "x"),
            R("orphan_s9", "a_L", 9, "done", 4.0, 1, 0.5, 5.6, 1, "x")]
    json.dump({"rows": rows, "deltas": [{"seed": 1, "delta": 0.0}, {"seed": 2, "delta": 0.1}]}, open(os.path.join(d, "ward_table.json"), "w"))
    return d


if __name__ == "__main__":
    stub = sys.argv[1] == "--stub"
    fails = check(build_stub() if stub else sys.argv[1])
    for n, r, m in fails:
        print(f"FAIL {n} [{r}] {m}")
    fired = sorted({f[0] for f in fails})
    print(json.dumps({"test": "wilson.k1.ward", "status": "red" if fails else "green", "n_fail": len(fails),
                      "checks_fired": f"{len(fired)}/{len(ALL)}", "never_fired": [c for c in ALL if c not in fired]}))
    sys.exit(1 if fails else 0)
