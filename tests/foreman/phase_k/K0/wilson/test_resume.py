"""K0_harness_resume_gpu: a run split by checkpoint+resume in a fresh process reproduces the straight run.
python test_resume.py DIR_STRAIGHT DIR_RESUMED   -> GREEN iff per-step losses and eval losses are bitwise equal
                                                  and the resumed log records resumed_from_step.
python test_resume.py --stub                      -> two fabricated logs that differ at one step (must be RED)."""
import json, os, sys, tempfile


def losses(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    st = {r["step"]: r["loss"] for r in recs if r["t"] == "step"}
    ev = {r["step"]: r["val_loss"] for r in recs if r["t"] == "eval"}
    res = [r.get("resumed_from_step") for r in recs if r["t"] == "config"]
    return st, ev, res


def check(a, b):
    sa, ea, _ = losses(a)
    sb, eb, rb = losses(b)
    diff = [s for s in sa if sa[s] != sb.get(s)]
    ediff = [s for s in ea if s in eb and ea[s] != eb[s]]
    maxdev = max((abs(sa[s] - sb[s]) for s in sa if s in sb), default=float("nan"))
    fails = []
    if not sa or set(sa) != set(sb):
        fails.append(f"step sets differ ({len(sa)} vs {len(sb)})")
    if diff:
        fails.append(f"{len(diff)} steps differ, first {diff[0]}, max |dloss| {maxdev:.3e}")
    if ediff:
        fails.append(f"eval differs at {ediff}")
    if not any(r for r in rb if r):
        fails.append("resumed log has no resumed_from_step")
    return not fails, fails, {"steps": len(sa), "max_abs_dloss": maxdev, "resumed_from": [r for r in rb if r],
                              "final_loss": sa[max(sa)] if sa else None, "evals": ea}


if __name__ == "__main__":
    if sys.argv[1] == "--stub":
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        for d, bump in ((a, 0.0), (b, 1e-3)):
            with open(os.path.join(d, "log.jsonl"), "w") as f:
                f.write(json.dumps({"t": "config", "resumed_from_step": None}) + "\n")
                for s in range(1, 5):
                    f.write(json.dumps({"t": "step", "step": s, "loss": 10.0 - s + (bump if s == 3 else 0)}) + "\n")
        ok, fails, out = check(a, b)
    else:
        ok, fails, out = check(sys.argv[1], sys.argv[2])
    print(json.dumps({"status": "green" if ok else "red", "fails": fails, **out}))
    sys.exit(0 if ok else 1)
