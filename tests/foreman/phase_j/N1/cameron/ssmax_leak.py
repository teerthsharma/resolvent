"""SSMax one-hop leak law on the planted bed: logits x s*log(i+1) turn e^DELTA into
(i+1)^(s*DELTA), so leak_i = (i+1-p)/(p (i+1)^(s DELTA) + i+1-p): it decays with i iff s*DELTA > 1."""
import numpy as np, never_len as NL
b = NL.plant(4096, 0); S = b["S"]; idx = np.arange(4096); T = np.flatnonzero(b["level"] > 0)
for s in (0.168, 0.5, 1.0, 2.0):
    W = NL.softmax_rows(S * (s * np.log(idx + 1.0))[:, None])
    p = np.array([len(b["parents"][i]) for i in T])
    meas = np.array([1 - W[i, b["parents"][i]].sum() for i in T])
    law = (T + 1 - p) / (p * (T + 1.0) ** (s * NL.DELTA) + T + 1 - p)
    print(f"s={s} s*DELTA={s*NL.DELTA:.3f} mean leak measured {meas.mean():.4e} law {law.mean():.4e} max|dev| {np.abs(meas-law).max():.1e}  leak at last transient row {meas[-1]:.3e}")
