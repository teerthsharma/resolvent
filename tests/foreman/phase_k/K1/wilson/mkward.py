"""python mkward.py ARM SEED ATTN -> runs/spec_<ARM>_s<SEED>.json: one R0 LM run (ctx 1024, b8, 20 x 7,227,648 tokens, eval every 250 x 40 batches, eval_seed 0)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
arm, seed, attn = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n = f"{arm}_s{seed}"
cmd = [sys.executable, HERE + "/train_ladder_k1.py", "--rung", "R0", "--ctx", "1024", "--batch", "8", "--tokens", str(20 * 7227648),
       "--attn", attn, "--data_dir", "C:/Users/seal/datasets/fineweb_edu", "--out_dir", f"{HERE}/runs/{n}",
       "--eval_every", "250", "--eval_batches", "40", "--seed", str(seed), "--eval_seed", "0"] + sys.argv[4:]
spec = {"tag": "ward_" + n, "session": "K1-wilson-ward", "record": HERE + "/runs/ward_jobs.jsonl",
        "jobs": [{"name": n, "cmd": cmd, "out": f"{HERE}/runs/{n}.out"}]}
json.dump(spec, open(f"{HERE}/runs/spec_{n}.json", "w"), indent=1)
print(f"{HERE}/runs/spec_{n}.json")
