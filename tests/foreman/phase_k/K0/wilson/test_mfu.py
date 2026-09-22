"""K0_mfu_measured: a run's logged MFU is well-formed and reproducible from its own step records.
python test_mfu.py RUN_DIR MIN_STEPS   -> prints GREEN/RED + measured values, exit 0/1.
python test_mfu.py --stub              -> builds a malformed 5-step log and runs the same assertion (must be RED).
Asserts: config names the 4060; >= MIN_STEPS steps; every step's tok_s == tokens_per_step/dt_s and
mfu == (6*params + 6*L*ctx*d)*tok_s/(peak*1e12) to rel 1e-9; end.mean_mfu == mean(step mfu[10:]);
peak memory < 8 GiB; tok_s coefficient of variation over steps 11+ < 0.25 (a stable measurement)."""
import json, os, statistics, sys, tempfile

RUNGS = {"R0": (4, 128), "R1": (6, 320), "R2": (8, 512), "R3": (12, 768)}


def check(run_dir, min_steps):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    cfg = [r for r in recs if r["t"] == "config"][0]
    steps = [r for r in recs if r["t"] == "step"]
    end = [r for r in recs if r["t"] == "end"]
    L, d = RUNGS[cfg["rung"]]
    fpt = 6 * cfg["params"] + 6 * L * cfg["ctx"] * d
    tps = cfg["batch"] * cfg["accum"] * cfg["ctx"]
    fails = []
    if "4060" not in cfg["device_name"]:
        fails.append(f"device {cfg['device_name']}")
    if len(steps) < min_steps:
        fails.append(f"{len(steps)} steps < {min_steps}")
    for r in steps:
        if abs(r["tok_s"] - tps / r["dt_s"]) > 1e-9 * r["tok_s"] or abs(r["mfu"] - fpt * r["tok_s"] / (cfg["peak_tflops"] * 1e12)) > 1e-9 * r["mfu"]:
            fails.append(f"step {r['step']} tok_s/mfu not reproducible"); break
    tail = [r["mfu"] for r in steps[10:]] or [0.0]
    mean_mfu = sum(tail) / len(tail)
    if not end or abs(end[0]["mean_mfu_excl_first_10"] - mean_mfu) > 1e-9 * max(mean_mfu, 1e-30):
        fails.append("end record missing or mean_mfu mismatch")
    peak = max((r["peak_mem_gib"] for r in steps), default=99)
    if peak >= 8.0:
        fails.append(f"peak mem {peak:.2f} GiB")
    toks = [r["tok_s"] for r in steps[10:]]
    cv = statistics.pstdev(toks) / statistics.mean(toks) if len(toks) > 1 else 9.9
    if cv >= 0.25:
        fails.append(f"tok_s cv {cv:.3f}")
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "steps": len(steps), "tok_s_mean": statistics.mean(toks) if toks else 0,
           "mfu_mean": mean_mfu, "tok_s_cv": cv, "peak_mem_gib": peak, "params": cfg["params"], "flops_per_token": fpt}
    return (not fails), fails, out


if __name__ == "__main__":
    if sys.argv[1] == "--stub":
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "log.jsonl"), "w") as f:
            f.write(json.dumps({"t": "config", "rung": "R0", "ctx": 1024, "batch": 8, "accum": 1, "params": 7227648,
                                "peak_tflops": 26.77, "device_name": "stub"}) + "\n")
            for s in range(1, 6):
                f.write(json.dumps({"t": "step", "step": s, "dt_s": 0.1, "tok_s": 81920.0, "mfu": 0.5, "peak_mem_gib": 1.0}) + "\n")
        ok, fails, out = check(d, 300)
    else:
        ok, fails, out = check(sys.argv[1], int(sys.argv[2]))
    print(json.dumps({"status": "green" if ok else "red", "fails": fails, **out}))
    sys.exit(0 if ok else 1)
