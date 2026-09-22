"""R-NEVER-LEN bar (contract §5). Exit non-zero on failure.
python test_never_len.py [--stub] [--quick]
--stub : f_N replaced by a softmax stub (the RED run, before the sparsemax arm is used)
--quick: n = 256, seed 0 only.
Asserts, in order:
 P1  f_N false-influence mass == 0 exactly at every n and seed;
 P2  f_N exact-set accuracy beats every (a_s) arm's best-threshold accuracy by >= 0.10 at n = 4096 (seed mean);
 C   counter: best (a_s)+threshold within 0.02 of f_N at n = 4096 -> NEVER is a certificate only (assert fails)."""
import sys, json, numpy as np
import never_len as NL
NL.STUB_FN = "--stub" in sys.argv
quick = "--quick" in sys.argv
rows = NL.main(ns=(256,) if quick else (256, 1024, 4096), seeds=range(1) if quick else range(5),
               out="never_len_rows_stub.jsonl" if NL.STUB_FN else "never_len_rows.jsonl")
fm = [r["f_N"]["false_mass"] for r in rows]
print("P1 f_N false mass per (n,seed):", fm)
assert all(x == 0.0 for x in fm), f"R-NEVER-LEN P1 FAIL: f_N false mass max {max(fm):.3e} > 0"
big = [r for r in rows if r["meta"]["n"] == 4096]
assert big, "R-NEVER-LEN: no n=4096 rows (quick mode cannot certify P2)"
fN = np.mean([r["f_N"]["acc_strict"] for r in big])
as_best = max(np.mean([r[k]["acc_best"] for r in big]) for k in big[0] if k.startswith("a_s["))
print(f"P2 n=4096: f_N acc {fN:.4f}  best (a_s)+threshold acc {as_best:.4f}  gap {fN - as_best:.4f}")
assert fN - as_best > 0.02, f"R-NEVER-LEN COUNTER HOLDS: (a_s)+threshold within 0.02 ({fN - as_best:.4f}); NEVER is a certificate only"
assert fN - as_best >= 0.10, f"R-NEVER-LEN P2 FAIL: gap {fN - as_best:.4f} < 0.10"
