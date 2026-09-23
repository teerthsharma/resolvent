# R-DEPTH on bed_k': the twin-under-line bar against the max of BOTH bound floor tables (Dispatcher after Inspector
# PASS2). Written 2026-09-23 before any R-DEPTH arm trained on bed_k'. Never edited after its RED. Supersedes the
# twin_under_line rows of test_rdepth_kp.py (which read only K1.F''s 0.4085 at (4096, L7), below Foreman's bound 0.4398).
# Line per cell (n in {4096, 8192, 16384}, L in {4, 7}) = max(k1fp.json rows[n][L].line.beyond,
#   Foreman's K1/foreman/kp_attack.json cells["n_L"].best), both means of the same 8 beds default_rng([22, n, k]).
# cameron.k1.rdepth_kp_twin_under_maxline : aL4 (beyond 2^4) and aL7 (beyond 2^7), each seed, each test n, read only if
#   L-TRAINED (held-out n=1024 acc on depth <= min(2^L, 32) >= 0.9; else UNREAD, counted failing), score <= line + 3 SE,
#   SE = sqrt(line (1 - line) / N_beyond). Runs: runs/kp_<arm>_s<seed>/result.json with config bed == "kp".
# Exit 1 if any row fails.
import json, math, sys
from pathlib import Path
HERE = Path(__file__).parent
FORE = HERE.parent / "foreman" / "kp_attack.json"
fails = []


def check(name, ok, got=""):
    tag = "PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")
    print(tag + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


K1FP = json.loads((HERE / "k1fp.json").read_text())
KPA = json.loads(FORE.read_text())
for a, L in (("aL4", 4), ("aL7", 7)):
    for s in (0, 1):
        p = HERE / "runs" / f"kp_{a}_s{s}" / "result.json"
        r = json.loads(p.read_text()) if p.exists() else None
        if r is not None and r["config"].get("bed") != "kp":
            r = None
        tr = None if r is None else r["eval"]["1024"]["acc_le16" if L == 4 else "acc_le32"]
        for n in ("4096", "8192", "16384"):
            line = max(K1FP["rows"][n][f"L{L}"]["line"]["beyond"], KPA["cells"][f"{n}_{L}"]["best"])
            if tr is None or tr < 0.9:
                check(f"cameron.k1.rdepth_kp_twin_under_maxline {a} s{s} n={n}", None, f"unread (L-TRAINED {tr}); line {line:.4f}")
                continue
            v = r["eval"][n]["beyond16" if L == 4 else "beyond128"]
            N = r["eval"][n]["n_beyond16" if L == 4 else "n_beyond128"]
            bar = line + 3 * math.sqrt(line * (1 - line) / N)
            check(f"cameron.k1.rdepth_kp_twin_under_maxline {a} s{s} n={n}", v <= bar, f"{v:.4f} vs {bar:.4f} (line {line:.4f})")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
