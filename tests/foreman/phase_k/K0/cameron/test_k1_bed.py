# K0_k1_bed. Written 2026-09-23 01:49 before bed_k.make_train/make_test/evaluate existed.
# The K1 R-DEPTH bed obeys its stated conventions, and the floor machinery agrees with a slow reference.
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import make_train, make_test, evaluate, floor_recency, ceiling_credit, make_bed, NULL, V

fails = []
def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got)); (None if ok else fails.append(name))

def content_ok(b):
    n = len(b["ids"]); nr = b["parent"] < 0; i = np.arange(n)
    return (len(np.unique(b["ids"])) == n and b["ids"].max() < V and (b["parent"][~nr] < i[~nr]).all()
            and (b["pids"][nr] == NULL).all() and (b["pids"][~nr] == b["ids"][b["parent"][~nr]]).all()
            and (b["target"] == b["ids"][b["root"]]).all())

rng = np.random.default_rng(7)
for _ in range(50):
    b = make_train(rng)
    if not (len(b["ids"]) == 1024 and b["depth"].max() <= 32 and content_ok(b)):
        check("train bed invariants", False, b["depth"].max()); break
else:
    check("train bed: n=1024, depth<=32, distinct ids, parent before child, pids/targets consistent (50 draws)", True)
for n in (4096, 8192, 16384):
    b = make_test(np.random.default_rng(n + 1), n)
    check(f"test bed n={n}: 16 roots, content consistent", len(np.unique(b["root"])) == 16 and content_ok(b), b["depth"].max())
b = make_test(np.random.default_rng(16385), 16384)
check("test bed 16k reaches depth >= 900", b["depth"].max() >= 900, b["depth"].max())
r = evaluate(b["target"], b, Ls=(4, 7)); check("oracle scores 1.0 everywhere", r["acc"] == 1.0 and r["beyond"]["7"] == 1.0, r["acc"])
f, pred = floor_recency(b["parent"], b["root"])
check("evaluate(recency preds) == floor_recency", abs(evaluate(b["ids"][pred], b)["acc"] - f) < 1e-12, f)

# slow reference for the chance-credited ceiling: walk 2^L parents one by one, then most recent root before it
p, d, rt = make_bed(600, 16, np.random.default_rng(3))
for L in (2, 3, 5):
    ok = 0
    for i in range(600):
        if d[i] <= 2 ** L: ok += 1; continue
        a = i
        for _ in range(2 ** L): a = p[a]
        lr = max(j for j in range(a + 1) if p[j] < 0)
        ok += rt[i] == lr
    check(f"ceiling_credit L={L} == slow reference", abs(ceiling_credit(p, d, rt, L) - ok / 600) < 1e-12, ok / 600)
sys.exit(1 if fails else 0)
