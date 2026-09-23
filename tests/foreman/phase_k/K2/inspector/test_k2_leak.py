# K2.0, the leakage bar (RECORD_K "K2, as registered here", item 1), plus Cameron's registered rows below it.
# Written 2026-09-23 BEFORE any K2 cell exists. Never edited after its RED.
# Checkpoint: K1/cameron/runs/far_fR_ga_s0/model.pt (the one K1-b arm that learned). Evaluation only, no training.
# Producer (written after this file): k2_leak.py <runs dir>. Layout it must write under <runs dir>:
#   k<kappa>/result.json + ok_4096.npz + ok_16384.npz   for kappa in {1, 1.5, 2, 3, 5}, eval gamma = 0.999 (the hook's)
#   g0.9999/result.json + ok_4096.npz + ok_16384.npz    at the best kappa (rule below), eval gamma = 0.9999
#   numerics.json                                        {"kappa": best, "relerr": {"0.999": e, "0.9999": e}}
#   result.json: {"kappa", "gamma", "ckpt_sha256", "a", "b", "eval": rdepth.evaluate(ns=(1024, 4096, 16384))}
# kappa scales the trained per-head logit scale s_i = a_h ln(i+1) + b_h of the resolvent layer: a -> kappa a,
# b -> kappa b. Beds: the far10 beds (bed_kp.make_test, default_rng([31, n, k]), k = 0..7), the same beds
# test_rdepth_far10_ga.py reads; the n = 1024 gate reads the 64 held-out train beds default_rng([21, k]).
# Best kappa: argmax over kappa of acc on depth > 160 at n = 4096; ties go to the smaller kappa.
#
# K2.0 rows (the registered bar, verbatim in substance):
#   cameron.k2.leak_prediction    : some kappa lifts depth > 160 at n = 4096 to >= 0.9.
#   cameron.k2.leak_counter_quiet : the counter is "no kappa lifts it above 0.5" (leakage is not the wall);
#                                   the row PASSES when max over kappa > 0.5 and FAILS when the counter fires.
#   cameron.k2.leak_cell          : every (kappa, cell) read against the far10 bar 0.95; cells are depth > 160 at
#                                   n = 4096 and 16384, and depth > 1280 at 16384.
# Machinery and control:
#   cameron.k2.leak_machinery     : each kappa dir: ckpt sha256 = the K1 file's; kappa, gamma 0.999; a, b = kappa x
#                                   the checkpoint's; depth arrays equal K1's ok_<n>.npz (same beds).
#   cameron.k2.leak_control_k1    : kappa = 1 reproduces K1: |acc - K1| <= 0.0005 on each cell (K1 recomputed from
#                                   K1's ok_<n>.npz: 0.2855 / 0.0784 / 0.0322) and on the n = 1024 gate (0.9963).
#   cameron.k2.leak_gate_at_best  : the best kappa still passes the learned-the-task gate (n = 1024, depth <= 16,
#                                   >= 0.9): the lift must not cost the training depth.
# Error profile (registered row asked for by the Dispatcher; the K1 report's unexplained "513+ at 0.55-0.69"):
#   cameron.k2.err_profile        : at the best kappa (and kappa = 1 beside it), accuracy and N per depth bucket
#                                   (rdepth EDGES) at n = 4096 and 16384; PASS iff the buckets partition every token.
# Gamma-horizon rows (Cameron's hypothesis for what walls the band beyond leakage, registered here, before any
#   number): with an exact parent pointer and an absorbing root, the read x = (1-g)(I-gW)^-1 V puts weight g^d on the
#   root at depth d (0.999^693 = 0.5, 0.999^1280 = 0.28), so gamma alone caps the far band. Raising gamma at
#   evaluation to 0.9999 moves that weight back onto the root without touching the pointer.
#   cameron.k2.gamma_numerics     : at the best kappa, bed k = 0, n = 4096: the fp32 GPU read (fs5c) against a float64
#                                   dense solve on the same captured qs, k, v: max |dx| / max |v| <= 1e-3, at
#                                   gamma = 0.999 and 0.9999.
#   cameron.k2.gamma_machinery    : g0.9999 dir: kappa = best, gamma 0.9999, sha and depth arrays as above.
#   cameron.k2.gamma_explains_513 : bucket 513-1024 at n = 16384: acc(0.9999) - acc(0.999) >= 0.10 at the best kappa.
#   cameron.k2.gamma_lift_1280    : depth > 1280 at n = 16384: acc(0.9999) - acc(0.999) >= 0.10 at the best kappa.
#   cameron.k2.gamma_counter_quiet: counter "the lift on depth > 1280 is <= 0.02" (gamma is not the wall); PASS iff
#                                   lift > 0.02.
#   cameron.k2.gamma_cell         : the three cells at gamma 0.9999 against the far10 bar 0.95.
#   cameron.k2.gamma_gate         : gamma 0.9999 at the best kappa still passes the gate (>= 0.9).
# Usage: python test_k2_leak.py [runs dir, default runs]. Exit 1 if any row fails.
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
K1R = HERE.parent.parent / "K1" / "cameron" / "runs" / "far_fR_ga_s0"
RUNS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
KAPPAS = ("1", "1.5", "2", "3", "5")
CELLS = (("4096", 160), ("16384", 160), ("16384", 1280))
EDGES = [0, 17, 33, 65, 129, 257, 513, 1025, 10 ** 9]
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok is True else "FAIL ") + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


def cell(z, thr):
    m = z["depth"] > thr
    return (float(z["ok"][m].mean()) if m.any() else float("nan")), int(m.sum())


def fmt(v, N):
    return f"{v:.4f} (N {N}, SE {math.sqrt(max(v * (1 - v), 0) / N):.4f})" if N and v == v else "nan"


def load(d):
    try:
        r = json.loads((d / "result.json").read_text())
        return r, {n: np.load(d / f"ok_{n}.npz") for n in ("4096", "16384")}
    except (OSError, ValueError, KeyError):
        return None, None


sha = hashlib.sha256((K1R / "model.pt").read_bytes()).hexdigest()
sd = torch.load(K1R / "model.pt", map_location="cpu")
a0, b0 = sd["blocks.3.attn.a"].double().numpy(), sd["blocks.3.attn.b"].double().numpy()
K1 = {n: np.load(K1R / f"ok_{n}.npz") for n in ("4096", "16384")}
K1gate = json.loads((K1R / "result.json").read_text())["eval"]["1024"]["acc_le16"]


def machinery(tag, r, z, kappa, gamma):
    if r is None:
        return False, "missing"
    bad = []
    if r.get("ckpt_sha256") != sha: bad.append("sha")
    if not math.isclose(float(r.get("kappa", -1)), kappa, rel_tol=0, abs_tol=1e-12): bad.append("kappa")
    if not math.isclose(float(r.get("gamma", -1)), gamma, rel_tol=0, abs_tol=1e-12): bad.append("gamma")
    if not (np.allclose(r.get("a", [0, 0]), kappa * a0, atol=1e-6) and np.allclose(r.get("b", [0, 0]), kappa * b0, atol=1e-6)): bad.append("a,b")
    for n in ("4096", "16384"):
        if z[n]["depth"].shape != K1[n]["depth"].shape or not np.array_equal(z[n]["depth"], K1[n]["depth"]): bad.append(f"beds {n}")
    return (not bad), ("ok" if not bad else "bad: " + ", ".join(bad))


# ---- the kappa grid
R, Z = {}, {}
for t in KAPPAS:
    R[t], Z[t] = load(RUNS / f"k{t}")
    ok, got = machinery(t, R[t], Z[t], float(t), 0.999)
    check(f"cameron.k2.leak_machinery kappa={t}", ok, got)

r1, z1 = R["1"], Z["1"]
for n, thr in CELLS:
    ref, N = cell(K1[n], thr)
    if z1 is None:
        check(f"cameron.k2.leak_control_k1 n={n} depth>{thr} == K1 {ref:.4f} +- 0.0005", False, "missing")
        continue
    v, _ = cell(z1[n], thr)
    agree = float((z1[n]["ok"] == K1[n]["ok"]).mean())
    check(f"cameron.k2.leak_control_k1 n={n} depth>{thr} == K1 {ref:.4f} +- 0.0005", abs(v - ref) <= 0.0005,
          f"{v:.4f} (N {N}; per-token agreement with K1 {agree:.6f})")
g1 = None if r1 is None else r1["eval"]["1024"]["acc_le16"]
check(f"cameron.k2.leak_control_k1 n=1024 gate == K1 {K1gate:.4f} +- 0.0005", g1 is not None and abs(g1 - K1gate) <= 0.0005,
      "missing" if g1 is None else f"{g1:.4f}")

for t in KAPPAS:
    for n, thr in CELLS:
        if Z[t] is None:
            check(f"cameron.k2.leak_cell kappa={t} n={n} depth>{thr} >= 0.95", False, "missing")
            continue
        v, N = cell(Z[t][n], thr)
        check(f"cameron.k2.leak_cell kappa={t} n={n} depth>{thr} >= 0.95", v >= 0.95, fmt(v, N))

b4 = {t: cell(Z[t]["4096"], 160)[0] for t in KAPPAS if Z[t] is not None}
b4 = {t: v for t, v in b4.items() if v == v}
best = max(b4, key=lambda t: (b4[t], -float(t))) if len(b4) == len(KAPPAS) else None
top = None if best is None else b4[best]
check("cameron.k2.leak_prediction max_kappa acc(depth>160, n=4096) >= 0.9", top is not None and top >= 0.9,
      "missing" if top is None else f"{top:.4f} at kappa={best}; grid " + " / ".join(f"x{t} {b4[t]:.4f}" for t in KAPPAS))
check("cameron.k2.leak_counter_quiet max_kappa acc(depth>160, n=4096) > 0.5 (FAIL = counter fires)",
      top is not None and top > 0.5, "missing" if top is None else f"{top:.4f} at kappa={best}")
gb = None if best is None else R[best]["eval"]["1024"]["acc_le16"]
check("cameron.k2.leak_gate_at_best n=1024 depth<=16 >= 0.9", gb is not None and gb >= 0.9,
      "missing" if gb is None else f"{gb:.4f} at kappa={best}")


def profile(z):
    out, tot = [], 0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (z["depth"] >= lo) & (z["depth"] < hi)
        if m.any():
            out.append((f"{lo}-{hi - 1 if hi < 10 ** 9 else 'max'}", float(z["ok"][m].mean()), int(m.sum())))
            tot += int(m.sum())
    return out, tot == z["ok"].size


for n in ("4096", "16384"):
    if best is None:
        check(f"cameron.k2.err_profile n={n} at best kappa", False, "missing")
        continue
    pb, okb = profile(Z[best][n])
    p1, ok1 = profile(Z["1"][n])
    check(f"cameron.k2.err_profile n={n} at best kappa={best} (x1 beside it) buckets partition the tokens", okb and ok1,
          "; ".join(f"{b} {v:.4f} (x1 {v1:.4f}, N {N})" for (b, v, N), (_, v1, _) in zip(pb, p1)))

# ---- gamma horizon, at the best kappa
try:
    nm = json.loads((RUNS / "numerics.json").read_text())
except (OSError, ValueError):
    nm = None
for g in ("0.999", "0.9999"):
    e = None if nm is None or best is None or str(nm.get("kappa")) != best else nm["relerr"].get(g)
    check(f"cameron.k2.gamma_numerics gamma={g} fs5c fp32 vs float64 dense, max|dx|/max|v| <= 1e-3", e is not None and e <= 1e-3,
          "missing" if e is None else f"{e:.3e} at kappa={best}")

rg, zg = load(RUNS / "g0.9999")
okm, got = (False, "missing best kappa") if best is None else machinery("g", rg, zg, float(best), 0.9999)
check("cameron.k2.gamma_machinery", okm, got)
if okm:
    m9 = lambda z: (lambda m: (float(z["ok"][m].mean()), int(m.sum())))((z["depth"] >= 513) & (z["depth"] < 1025))
    (va, N), (vb, _) = m9(zg["16384"]), m9(Z[best]["16384"])
    check("cameron.k2.gamma_explains_513 n=16384 depth 513-1024 lift >= 0.10", va - vb >= 0.10,
          f"{vb:.4f} -> {va:.4f} (lift {va - vb:+.4f}, N {N}) at kappa={best}")
    (va, N), (vb, _) = cell(zg["16384"], 1280), cell(Z[best]["16384"], 1280)
    check("cameron.k2.gamma_lift_1280 n=16384 depth>1280 lift >= 0.10", va - vb >= 0.10, f"{vb:.4f} -> {va:.4f} (lift {va - vb:+.4f}, N {N})")
    check("cameron.k2.gamma_counter_quiet n=16384 depth>1280 lift > 0.02 (FAIL = counter fires)", va - vb > 0.02, f"lift {va - vb:+.4f}")
    for n, thr in CELLS:
        v, N = cell(zg[n], thr)
        check(f"cameron.k2.gamma_cell gamma=0.9999 kappa={best} n={n} depth>{thr} >= 0.95", v >= 0.95, fmt(v, N))
    gg = rg["eval"]["1024"]["acc_le16"]
    check("cameron.k2.gamma_gate n=1024 depth<=16 >= 0.9", gg >= 0.9, f"{gg:.4f}")
    pg, okg = profile(zg["16384"])
    print("  field (err_profile at gamma 0.9999, n=16384): " + "; ".join(f"{b} {v:.4f} (N {N})" for b, v, N in pg))
else:
    for nm_ in ("gamma_explains_513", "gamma_lift_1280", "gamma_counter_quiet", "gamma_cell", "gamma_gate"):
        check(f"cameron.k2.{nm_}", False, "unread (gamma machinery)")

print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
