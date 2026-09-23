"""python mkmfu.py ARM ATTN -> mfu/<ARM>/{spec.json, session.json}: peak_pre, 6 R0 probes (c1024 b8/b4/b16, c4096 b2/b1/b4, 300 steps), peak_post, one lock hold.
(mfu/aL was built by the same recipe inline before this file existed.)"""
import json, os, sys, time, datetime
W = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
arm, attn = sys.argv[1], sys.argv[2]
sd = f"{W}/mfu/{arm}"
os.makedirs(sd + "/runs", exist_ok=True)
ses = f"K1-wilson-mfu-{arm}-{datetime.date.today():%Y%m%d}"
py = sys.executable
jobs = [{"name": "peak_pre", "cmd": [py, W + "/peak_k1.py", ses, "pre", sd + "/peak_pre.json"], "out": sd + "/peak_pre.out"}]
for ctx, b in [(1024, 8), (1024, 4), (1024, 16), (4096, 2), (4096, 1), (4096, 4)]:
    n = f"probe_{arm}_c{ctx}_b{b}"
    jobs.append({"name": n, "cmd": [py, W + "/train_ladder_k1.py", "--rung", "R0", "--ctx", str(ctx), "--batch", str(b), "--steps", "300",
                 "--attn", attn, "--data_dir", "C:/Users/seal/datasets/fineweb_edu", "--out_dir", sd + "/runs/" + n,
                 "--eval_every", "1000000", "--eval_batches", "1", "--seed", "1"] + sys.argv[3:], "out": sd + "/runs/" + n + ".out"})
jobs.append({"name": "peak_post", "cmd": [py, W + "/peak_k1.py", ses, "post", sd + "/peak_post.json"], "out": sd + "/peak_post.out"})
json.dump({"tag": "mfu_" + arm, "session": ses, "record": sd + "/jobs.jsonl", "jobs": jobs}, open(sd + "/spec.json", "w"), indent=1)
json.dump({"session": ses, "t_start": time.time()}, open(sd + "/session.json", "w"))
print(sd + "/spec.json")
