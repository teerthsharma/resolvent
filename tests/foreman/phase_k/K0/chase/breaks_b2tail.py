"""B2t: the tail-mass instrument (rrange.head_mass_tail, validated on Y2 by bar y2_tail_instrument) on the same
ALiBi + random-content heads as breaks.py B2 (seed 7, SSMax a=1 b=0, n=4096), next to the log-kernel instrument.
Also the Toeplitz (content-free) twin. Rows -> breaks_rows.jsonl (appended)."""
import json, math, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrange, resolvent as R
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "breaks_rows.jsonl"), "a")
n = 4096
slopes = [2.0 ** (-h) for h in range(1, 9)]
rows = np.arange(3500, 4001, 100); r = np.arange(30, 400)
g0 = torch.Generator().manual_seed(7)
q = torch.randn(1, 8, n, 64, generator=g0, dtype=torch.float64); k = torch.randn(1, 8, n, 64, generator=g0, dtype=torch.float64)
qs = R.ssmax_q(q, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
i_ = torch.arange(n, dtype=torch.float64)
for h in range(8):
    z = R._logits(qs[:, h:h + 1], k[:, h:h + 1])[0, 0] - slopes[h] * (i_[:, None] - i_[None, :]).clamp(min=0)
    W = torch.softmax(z, -1).numpy(); del z
    for g in (0.99, 0.999):
        d = dict(t="B2t", g=g, slope=slopes[h], ssmax_a=1.0,
                 m_logkernel=rrange.head_mass(W, g, rows, r), m_tail=rrange.head_mass_tail(W, g, rows, r),
                 m_y2_bare_slope=math.log(g + (1 - g) * math.exp(slopes[h])))
        d["range_tail"] = 1 / d["m_tail"] if d["m_tail"] > 0 else float("inf")
        print(json.dumps(d), flush=True); out.write(json.dumps(d) + "\n")
out.close()
