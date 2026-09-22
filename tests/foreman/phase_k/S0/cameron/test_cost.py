"""COST BAR (registered): at hidden 384, 6 layers, 6 heads, seq 1024, batch 8, fp32, an f-lite model
trains at >= 65,359 / 1.5 = 43,573 tok/s with peak memory <= 4 GiB.
`--stub` measures the dense family (ceq/arm_smprime.py) instead; it must exit non-zero."""
import sys, json, subprocess
ARM = "family" if "--stub" in sys.argv else "flite"
TWIN, BAR_TOK, BAR_GIB = 65359, 65359 / 1.5, 4.0
out = subprocess.run([sys.executable, "measure.py", ARM, "384", "6", "6", "1024", "8"], capture_output=True, text=True)
row = json.loads(out.stdout.strip().splitlines()[-1])
ok = not row.get("oom") and row["tok_per_s"] >= BAR_TOK and row["peak_gib"] <= BAR_GIB
desc = "OOM" if row.get("oom") else f"{row['tok_per_s']} tok/s ({row['tok_per_s'] / TWIN:.3f}x twin), peak {row['peak_gib']} GiB"
print(json.dumps(row))
print(f"{'GREEN' if ok else 'RED'} cost@11.2M: {ARM} {desc} vs bar >= {BAR_TOK:.0f} tok/s, <= {BAR_GIB} GiB")
sys.exit(0 if ok else 1)
