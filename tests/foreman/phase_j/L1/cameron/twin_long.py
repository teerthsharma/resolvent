"""Longer softmax-twin runs on the S4 and S3 shell beds (same model, eval set and
plateau rule as shell_bed.compute_f3); writes shell_bed_floors_long.json with F3 replaced."""
import json, sys, time
import numpy as np
sys.path.insert(0, r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad")
import design4x5
import shell_bed as sb

ok = design4x5.poll_until_free(timeout_s=float(sys.argv[1]) if len(sys.argv) > 1 else 900, interval_s=10)
print("FREE", ok, flush=True)
if not ok:
    sys.exit("GPU never free; no twin run")
d = json.load(open("shell_bed_floors.json"))
for name, budget in (("S4", 240.0), ("S3", 150.0)):
    spec = sb.BEDS[name]
    rng = np.random.default_rng(sb.EVAL_SEED)
    tok = sb.gen_sequences(spec["n_tok"], sb.EVAL_N, sb.L, rng)
    lab = sb.labels_from_tokens(tok, spec["table"])
    f3 = sb.compute_f3(name, spec, tok, lab, "cuda", budget)
    d["beds"][name]["F3"] = f3
    print(name, f3["per_position_acc"], f3["mean_ge32_acc"], f3["plateau"], f3["steps"], flush=True)
    json.dump(d, open("shell_bed_floors_long.json", "w"), indent=1)
print("done", time.strftime("%H:%M:%S"), flush=True)
