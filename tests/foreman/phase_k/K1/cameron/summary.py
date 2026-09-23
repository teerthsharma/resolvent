# Assembles the per-run table for CAMERON_REPORT.md (fields only; the bars live in the test files).
import json, sys
from pathlib import Path
import numpy as np
RUNS = ["far_fR_s0", "far_fR_s1", "far_fR_ga_s0", "far_fR_ga_s1", "far_aL4_s0", "far_aL4_s1", "far_aL7_s0", "far_aL7_s1",
        "far_ass4_s0", "far_ass4_s1", "far_aloop5_s0", "far_aloop5_s1", "pilot_fR_ganneal_s0", "pilot_aL7_frozen_orth_lr3e-3_8k"]
print("| run | params (non-emb) | train s / wall s | gate | held-out acc | depth>160 at 4k / 8k / 16k | depth>1280 at 16k | acc at 16k |")
print("|---|---|---|---|---|---|---|---|")
for r in RUNS:
    p = Path("runs") / r / "result.json"
    if not p.exists():
        print(f"| {r} | not run yet | | | | | | |"); continue
    x = json.loads(p.read_text()); e = x["eval"]
    gate = e["1024"]["acc_le16"] if x["Larm"] == 4 else e["1024"]["acc_le32"]
    cells, b7 = [], "--"
    for n in (4096, 8192, 16384):
        f = Path("runs") / r / f"ok_{n}.npz"
        if f.exists():
            z = np.load(f); d, ok = z["depth"], z["ok"]
            cells.append(f"{ok[d > 160].mean():.4f}")
            if n == 16384:
                b7 = f"{ok[d > 1280].mean():.4f}"
        else:
            cells.append("--")
    a16 = f"{e['16384']['acc']:.4f}" if "16384" in e else "--"
    print(f"| {r} | {x['config']['params']:,} ({x['config']['params_nonemb']:,}) | {x['train_s']:.0f} / {x['wall_s']:.0f} | "
          f"{gate:.4f} {'pass' if gate >= 0.9 else 'FAIL'} | {e['1024']['acc']:.4f} | {' / '.join(cells)} | {b7} | {a16} |")
