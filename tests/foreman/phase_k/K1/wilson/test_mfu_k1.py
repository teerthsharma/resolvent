"""wilson.k1.mfu -- clause 8 of RECORD_K.md, bar written before the sweep.
python test_mfu_k1.py SESSION_DIR   -> checks the reported MFU table of one session; exit 1 if any row fails.
python test_mfu_k1.py --stub        -> same checks on a built-bad session (must be RED; prints which checks fired).

Session dir: session.json {session, t_start}; peak_pre.json (peak_k1.py); jobs.jsonl (gpujob.py record);
runs/<job name>/log.jsonl for every job named probe_*; mfu_table.json {session, peak_tflops, rows[]}.

Asserts (stated digits: tok/s to the integer, MFU to 3 decimals, peak TF and memory to 2 decimals):
 peak_device      the peak was measured on the 4060
 peak_reps        5 repetitions
 peak_median      median(reps) == stored median (rel 1e-9) and the table's peak == that median to 0.005 TF
 peak_session     peak, session file and table carry the same session tag
 peak_before_sweep session start <= peak start and peak end <= the first probe's start (same session, before the sweep)
 row_present      every probe job has exactly one table row;  row_orphan: every row names a probe job
 nofit_reported   a probe whose process exited 3 or whose log has an abort record is reported no_fit with tok_s = mfu = null
 nofit_real       a row reported no_fit has rc 3 and a memory abort record in its log
 fit_rc0          a row reported fit has rc 0 and no abort record
 fit_steps        step records == the requested --steps
 fit_step_consistency every step's tok_s == batch*accum*ctx/dt_s (rel 1e-9)
 fit_tok_s        |reported tok/s - mean logged tok/s over steps 11+| <= 0.5
 fit_mfu          |reported MFU - (6N + 6*L*ctx*d) * tok_s / (session median * 1e12)| <= 5e-4
 fit_mem          logged peak memory < the device's total memory, and reported to 0.005 GiB
 fit_cv           coefficient of variation of logged tok/s over steps 11+ < 0.10
 fit_clean        no foreign GPU process was seen during the probe"""
import json, os, statistics, sys, tempfile

RUNGS = {"R0": (4, 128), "R1": (6, 320), "R2": (8, 512), "R3": (12, 768)}
ALL = ["peak_device", "peak_reps", "peak_median", "peak_session", "peak_before_sweep", "row_present", "row_orphan",
       "nofit_reported", "nofit_real", "fit_rc0", "fit_steps", "fit_step_consistency", "fit_tok_s", "fit_mfu",
       "fit_mem", "fit_cv", "fit_clean"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def check(sd):
    fails = []
    F = lambda name, row, msg: fails.append((name, row, msg))
    ses = json.load(open(os.path.join(sd, "session.json")))
    pk = json.load(open(os.path.join(sd, "peak_pre.json")))
    tab = json.load(open(os.path.join(sd, "mfu_table.json")))
    jobs = [j for j in jl(os.path.join(sd, "jobs.jsonl")) if j["name"].startswith("probe_")]
    med = pk["median"]
    if "4060" not in pk["device"]: F("peak_device", "-", pk["device"])
    if len(pk["tflops"]) != 5: F("peak_reps", "-", len(pk["tflops"]))
    if abs(statistics.median(pk["tflops"]) - med) > 1e-9 * med or abs(tab["peak_tflops"] - med) > 0.005:
        F("peak_median", "-", f"median {statistics.median(pk['tflops'])} stored {med} table {tab['peak_tflops']}")
    if not (pk["session"] == ses["session"] == tab["session"]): F("peak_session", "-", f"{pk['session']} {ses['session']} {tab['session']}")
    first = min((j["t_start"] for j in jobs), default=float("inf"))
    if not (ses["t_start"] <= pk["t_start"] and pk["t_end"] <= first): F("peak_before_sweep", "-", f"peak {pk['t_start']}-{pk['t_end']} first probe {first}")
    rows = {}
    for r in tab["rows"]:
        rows.setdefault(r["name"], []).append(r)
    for n in rows:
        if n not in {j["name"] for j in jobs}: F("row_orphan", n, "row names no probe job")
    for j in jobs:
        n = j["name"]
        if len(rows.get(n, [])) != 1:
            F("row_present", n, f"{len(rows.get(n, []))} rows"); continue
        r = rows[n][0]
        recs = jl(os.path.join(sd, "runs", n, "log.jsonl"))
        cfg = [x for x in recs if x["t"] == "config"][0]
        steps = [x for x in recs if x["t"] == "step"]
        aborts = [x for x in recs if x["t"] == "abort"]
        if (j["rc"] == 3 or aborts) and not (r["status"] == "no_fit" and r["tok_s"] is None and r["mfu"] is None):
            F("nofit_reported", n, f"rc {j['rc']} aborts {len(aborts)} reported {r['status']}")
        if r["status"] == "no_fit":
            if not (j["rc"] == 3 and any("memory" in a.get("reason", "") for a in aborts)): F("nofit_real", n, f"rc {j['rc']} aborts {len(aborts)}")
            continue
        if j["rc"] != 0 or aborts: F("fit_rc0", n, f"rc {j['rc']} aborts {len(aborts)}")
        if len(steps) != cfg["steps"]: F("fit_steps", n, f"{len(steps)} of {cfg['steps']}")
        tps = cfg["batch"] * cfg["accum"] * cfg["ctx"]
        if any(abs(s["tok_s"] - tps / s["dt_s"]) > 1e-9 * s["tok_s"] for s in steps): F("fit_step_consistency", n, "tok_s != tps/dt_s")
        toks = [s["tok_s"] for s in steps[10:]] or [0.0]
        mean = statistics.mean(toks)
        if r["tok_s"] is None or abs(r["tok_s"] - mean) > 0.5: F("fit_tok_s", n, f"reported {r['tok_s']} logged {mean:.3f}")
        L, d = RUNGS[cfg["rung"]]
        L, d = cfg.get("layers") or L, cfg.get("width") or d
        mfu = (6 * cfg["params"] + 6 * L * cfg["ctx"] * d) * mean / (med * 1e12)
        if r["mfu"] is None or abs(r["mfu"] - mfu) > 5e-4: F("fit_mfu", n, f"reported {r['mfu']} recomputed {mfu:.5f} at {med:.4f} TF")
        pm = max((s["peak_mem_gib"] for s in steps), default=99.0)
        if not (pm < cfg["device_total_gib"]) or r["peak_mem_gib"] is None or abs(r["peak_mem_gib"] - pm) > 0.005:
            F("fit_mem", n, f"logged {pm:.4f} device {cfg['device_total_gib']} reported {r['peak_mem_gib']}")
        cv = statistics.pstdev(toks) / mean if len(toks) > 1 and mean > 0 else 9.9
        if cv >= 0.10: F("fit_cv", n, f"cv {cv:.4f}")
        if j.get("foreign_gpu_pids_seen"): F("fit_clean", n, f"foreign {j['foreign_gpu_pids_seen']}")
    return fails


def build_stub():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "runs"))
    json.dump({"session": "S", "t_start": 100.0}, open(os.path.join(d, "session.json"), "w"))
    json.dump({"session": "T", "device": "stub", "tflops": [30.0, 30.0, 30.0, 30.0], "median": 30.0, "t_start": 90.0, "t_end": 250.0},
              open(os.path.join(d, "peak_pre.json"), "w"))
    cfg = {"t": "config", "rung": "R0", "layers": None, "width": None, "ctx": 1024, "batch": 8, "accum": 1, "steps": 20,
           "params": 7227648, "device_total_gib": 7.996}
    good = [{"t": "step", "step": s, "dt_s": 0.1, "tok_s": 81920.0, "peak_mem_gib": 5.6} for s in range(1, 21)]
    bad = [{"t": "step", "step": s, "dt_s": 0.1, "tok_s": 81920.0 * (1 + (s % 2)), "peak_mem_gib": 9.0} for s in range(1, 16)]
    runs = {"probe_mfu26": (0, good), "probe_abort_as_fit": (3, good[:1] + [{"t": "abort", "reason": "allocated > device memory"}]),
            "probe_fit_as_nofit": (0, good), "probe_no_row": (0, good), "probe_all_bad": (0, bad)}
    with open(os.path.join(d, "jobs.jsonl"), "w") as f:
        for i, (n, (rc, recs)) in enumerate(runs.items()):
            os.makedirs(os.path.join(d, "runs", n))
            with open(os.path.join(d, "runs", n, "log.jsonl"), "w") as g:
                for x in [cfg] + recs:
                    g.write(json.dumps(x) + "\n")
            f.write(json.dumps({"name": n, "rc": rc, "t_start": 200.0 + i, "foreign_gpu_pids_seen": [4242] if n == "probe_all_bad" else []}) + "\n")
    fpt = 6 * 7227648 + 6 * 4 * 1024 * 128
    rows = [{"name": "probe_mfu26", "status": "fit", "tok_s": 81920, "mfu": round(fpt * 81920 / 26.77e12, 3), "peak_mem_gib": 5.6},
            {"name": "probe_abort_as_fit", "status": "fit", "tok_s": 81920, "mfu": 0.2, "peak_mem_gib": 5.6},
            {"name": "probe_fit_as_nofit", "status": "no_fit", "tok_s": None, "mfu": None, "peak_mem_gib": None},
            {"name": "probe_all_bad", "status": "fit", "tok_s": 1, "mfu": 0.9, "peak_mem_gib": 5.6},
            {"name": "probe_orphan", "status": "fit", "tok_s": 1, "mfu": 0.1, "peak_mem_gib": 1.0}]
    json.dump({"session": "S", "peak_tflops": 26.77, "rows": rows}, open(os.path.join(d, "mfu_table.json"), "w"))
    return d


if __name__ == "__main__":
    stub = sys.argv[1] == "--stub"
    fails = check(build_stub() if stub else sys.argv[1])
    for name, row, msg in fails:
        print(f"FAIL {name} [{row}] {msg}")
    fired = sorted({f[0] for f in fails})
    print(json.dumps({"test": "wilson.k1.mfu", "status": "red" if fails else "green", "n_fail": len(fails),
                      "checks_fired": f"{len(fired)}/{len(ALL)}", "never_fired": [c for c in ALL if c not in fired]}))
    sys.exit(1 if fails else 0)
