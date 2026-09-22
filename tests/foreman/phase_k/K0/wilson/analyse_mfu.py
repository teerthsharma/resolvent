"""MFU table + hours per full run (20 tokens/param) from the sweep logs. Hours = 20*N / measured tok/s (peak-free);
also the contract's Y4 at the assumed MFU 0.35 for comparison."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mfu import check, RUNGS

HERE = os.path.dirname(os.path.abspath(__file__))
MIN = {"R0": 300, "R1": 100, "R2": 40, "R3": 30}
rows = {}
for d in sorted(glob.glob(f"{HERE}/runs/mfu_*/")):
    name = os.path.basename(d.rstrip("/\\"))
    rung = name.split("_")[1]
    try:
        ok, fails, out = check(d, MIN[rung])
    except Exception as e:
        print(json.dumps({"run": name, "status": "error", "err": repr(e)})); continue
    cfg = json.loads(open(os.path.join(d, "log.jsonl")).readline())
    out.update(status="green" if ok else "red", fails=fails, rung=rung, ctx=cfg["ctx"], batch=cfg["batch"])
    print(json.dumps(out))
    if ok:
        rows[(rung, cfg["ctx"], -cfg["batch"])] = out
print("\n| rung | ctx | batch | params (measured) | Np = 12Ld^2+50304d | tok/s | MFU vs 26.77 | peak mem GiB | hours @20N (measured N) | hours @20Np | Y4 hours @MFU 0.35 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (rung, ctx, _), o in sorted(rows.items()):
    L, d = RUNGS[rung]
    Np = 12 * L * d * d + 50304 * d
    y4 = (6 * Np + 6 * L * ctx * d) * 20 * Np / (26.77e12 * 0.35) / 3600
    print(f"| {rung} | {ctx} | {o['batch']} | {o['params']:,} | {Np:,} | {o['tok_s_mean']:,.0f} | {o['mfu_mean']:.4f} | {o['peak_mem_gib']:.2f} | "
          f"{20 * o['params'] / o['tok_s_mean'] / 3600:.2f} | {20 * Np / o['tok_s_mean'] / 3600:.2f} | {y4:.2f} |")
