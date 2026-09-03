"""IT.11 MERCURY -- read the two it.11 journals and print the cells. No writes."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
F1 = ROOT / "results" / "v20_r15_it11_mercury_smprime.jsonl"
F4 = ROOT / "results" / "v20_r15_it11_mercury_draw4.jsonl"
BANK = {"results/v17k_r4_retake.jsonl": range(0, 8),
        "results/v20_r15_it6_seeds8_15.jsonl": range(8, 16)}
FLOOR = 0.7071067811865476

def rows(p):
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []

R = rows(F1)
hdr = [r for r in R if r.get("t") == "header"][-1] if R else {}
sc = {(r["seed"], r["eval_seed"]): r for r in R if r.get("t") == "rescore"}
wall = [r for r in R if r.get("t") == "wall"]

banked = {}
for path, seeds in BANK.items():
    for l in (ROOT / path).open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("t") == "cell" and r.get("kind") == "arm_smprime" and r.get("seed") in seeds:
            banked[r["seed"]] = r["eval_nrmse"]

print("== CONTROL: draw 12345 vs frozen journals, bitwise ==")
ok = bad = 0
for s in sorted(banked):
    r = sc.get((s, 12345))
    if r is None:
        print(f"seed {s:>2}  NOT RUN"); continue
    hit = r["eval_nrmse"] == banked[s]
    ok += hit; bad += (not hit)
    print(f"seed {s:>2}  {banked[s]!r:22}  {r['eval_nrmse']!r:22}  {'yes' if hit else 'NO'}")
print(f"bitwise {ok}/{ok+bad}")

ES = hdr.get("eval_seeds", [12345, 12346, 20260902])
print(f"\n== EXPERIMENT A: arm_smprime, floor_1={FLOOR!r} ==")
print("seed | " + " / ".join(str(e) for e in ES) + " | crossings | lambda_hat | lam_live | annih | a_max | unit_root | stable")
nflip = 0
for s in range(16):
    rs = [sc.get((s, e)) for e in ES]
    if any(r is None for r in rs):
        print(f"{s:>4} | NOT RUN"); continue
    n = [f"{r['eval_nrmse']:.6f}" for r in rs]
    c = [r["crosses_cell"] for r in rs]
    stable = len(set(c)) == 1
    nflip += (not stable)
    r0 = rs[0]
    print(f"{s:>4} | {' / '.join(n)} | {''.join('T' if x else 'F' for x in c)} | "
          f"{r0['lambda_hat']} | {r0['lambda_hat_live']:+.6f} | {r0['frac_gate_annihilated']:.6f} | "
          f"{r0['a_hat_max']:.6f} | {r0['unit_root']} | {'yes' if stable else 'FLIP'}")
done = sum(1 for s in range(16) if all(sc.get((s, e)) for e in ES))
print(f"cells complete {done}/16, scorings {len(sc)}, verdict flips {nflip}")
cross = {s for s in range(16) if all((sc.get((s,e)) or {}).get('crosses_cell') for e in ES)}
anyx = {s for s in range(16) for e in ES if (sc.get((s,e)) or {}).get('crosses_cell')}
print(f"cross on ALL draws: {sorted(cross)}   cross on ANY draw: {sorted(anyx)}")
for w in wall:
    est = w.get("price_estimate_secs")
    print(f"WALL {w['secs']:.2f}s vs ESTIMATE {est}s  delta {100*(w['secs']-est)/est:+.1f}%")

R4 = rows(F4)
if R4:
    s4 = {(r["seed"], r["eval_seed"]): r for r in R4 if r.get("t") == "rescore"}
    IT10 = ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl"
    prev = {}
    for l in IT10.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("t") == "rescore" and r.get("kind") == "arm_pl":
            prev.setdefault(r["seed"], {})[r["eval_seed"]] = r
    RANK = [(15,0.999),(12,0.589),(9,0.493),(14,0.266),(10,0.180),(13,0.180),(0,0.165),(8,0.158),(11,0.105)]
    print("\n== EXPERIMENT B: fourth eval draw 20260903 on arm_pl ==")
    print("rank | seed | VENUS ratio | 3-draw verdicts | draw4 nrmse | draw4 cross | flipped")
    flips = []
    for i, (s, ratio) in enumerate(RANK, 1):
        r = s4.get((s, 20260903))
        if r is None:
            print(f"{i:>4} | {s:>4} | {ratio} | NOT RUN"); continue
        old = [prev[s][e]["crosses_cell"] for e in (12345, 12346, 20260902)]
        f = (r["crosses_cell"] != old[0])
        if f: flips.append(s)
        print(f"{i:>4} | {s:>4} | {ratio:.3f} | {''.join('T' if x else 'F' for x in old)} | "
              f"{r['eval_nrmse']:.6f} | {r['crosses_cell']} | {'FLIP' if f else 'no'}")
        c = s4.get((s, 12345))
        if c is not None:
            print(f"       control 12345: {c['eval_nrmse']!r} vs it.10 {prev[s][12345]['eval_nrmse']!r} "
                  f"-> {'bitwise' if c['eval_nrmse']==prev[s][12345]['eval_nrmse'] else 'DIFFERS'}")
    print(f"cells flipped on draw 4: {flips if flips else 'none'}")
    for w in [r for r in R4 if r.get("t") == "wall"]:
        est = w.get("price_estimate_secs")
        print(f"WALL {w['secs']:.2f}s vs ESTIMATE {est}s  delta {100*(w['secs']-est)/est:+.1f}%")

print("\n== REGIME DIFF: it.10 header vs it.11 header ==")
h10 = [json.loads(l) for l in (ROOT/"results"/"v20_r15_it10_mercury_rescore.jsonl").open(encoding="utf-8") if '"header"' in l][0]
h11 = hdr
keys = sorted(set(h10) | set(h11))
same = 0
for k in keys:
    a, b = h10.get(k, "<absent>"), h11.get(k, "<absent>")
    if a == b:
        same += 1
    else:
        print(f"DIFF {k:<26} {json.dumps(a, default=str):<34} {json.dumps(b, default=str)}")
print(f"-- header keys: {len(keys)} union, {same} identical, {len(keys)-same} differ")
