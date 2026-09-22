"""R-NEVER-LEN' (replacement route for the struck R-NEVER-LEN P2): threshold TRANSFER.
The oracle picks each softmax-family arm's tau per (n, seed). Here tau is chosen ONCE,
on n = 256 (mean exact-set accuracy over the seeds), frozen, and applied at n = 1024, 4096.
f_N uses tau = 0 everywhere (no choice to make).
python test_never_len_transfer.py [--stub]   (--stub: f_N replaced by a softmax stub; seed 0 only)
Asserts: f_N acc - max over (a_s) arms of transferred-tau acc >= 0.10 at n = 4096;
counter: within 0.02 -> exactness is a certificate only even without an oracle tau."""
import sys, json, numpy as np
import never_len as NL
NL.STUB_FN = "--stub" in sys.argv
seeds = range(1) if NL.STUB_FN else range(5)
rows = NL.main(ns=(256, 1024, 4096), seeds=seeds,
               out="never_len_transfer_stub.jsonl" if NL.STUB_FN else "never_len_transfer.jsonl")
by = {n: [r for r in rows if r["meta"]["n"] == n] for n in (256, 1024, 4096)}
arms = [k for k in rows[0] if isinstance(rows[0][k], dict) and "acc_curve" in rows[0][k]]
res = {}
for a in arms:
    curve256 = np.mean([r[a]["acc_curve"] for r in by[256]], 0)
    j = int(np.argmax(curve256)) if a != "f_N" else 0
    res[a] = dict(tau=float(NL.TAUS[j]), acc256=float(curve256[j]),
                  **{f"acc{n}": [r[a]["acc_curve"][j] for r in by[n]] for n in (1024, 4096)})
    print(f"{a:22s} tau*={res[a]['tau']:.3e} acc256={res[a]['acc256']:.4f} "
          f"acc1024={np.mean(res[a]['acc1024']):.4f} acc4096={np.mean(res[a]['acc4096']):.4f} per-seed4096={np.round(res[a]['acc4096'], 4).tolist()}")
json.dump(res, open("never_len_transfer_summary%s.json" % ("_stub" if NL.STUB_FN else ""), "w"), indent=1)
fN = np.mean(res["f_N"]["acc4096"])
best = max(np.mean(res[a]["acc4096"]) for a in arms if a.startswith("a_s["))
print(f"n=4096 f_N {fN:.4f}  best a_s with transferred tau {best:.4f}  gap {fN - best:.4f}")
assert fN - best > 0.02, f"R-NEVER-LEN' COUNTER HOLDS: transferred-tau (a_s) within 0.02 ({fN - best:.4f})"
assert fN - best >= 0.10, f"R-NEVER-LEN' FAIL: gap {fN - best:.4f} < 0.10"
