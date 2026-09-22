"""Read (a), (a2), (f) final_eval_loss for split seeds 0..4; write p5.json."""
import json, os
import numpy as np
from scipy.stats import t as T
SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
L = lambda arm, s: json.load(open(os.path.join(SP, "d45_ckpt_{}_ss{}".format(arm, s), "run_record.json")))["final_eval_loss"]
rows = [dict(ss=s, a=L("a", s), a2=L("a2", s), f=L("f", s)) for s in range(5)]
rec = np.array([(r["a"] - r["a2"]) / (r["a"] - r["f"]) for r in rows])
h = T.ppf(0.975, len(rec) - 1) * rec.std(ddof=1) / np.sqrt(len(rec))
res = dict(rows=rows, recovery=rec.tolist(), recovery_mean=float(rec.mean()),
           recovery_ci=[float(rec.mean() - h), float(rec.mean() + h)],
           C_win=[r["a"] - r["f"] for r in rows], C_a2=[r["a"] - r["a2"] for r in rows])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p5.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
