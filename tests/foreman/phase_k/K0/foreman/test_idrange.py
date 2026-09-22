# Foreman K0 T8 bar, written before running: under wald.py's convention (id = position), R-DEPTH's split
# (train n = 1024, test n = 16384) puts > 0.9 of test tokens' own ids AND parent ids outside the id range
# ever seen in training (0..1023), so a learned id embedding is untrained on them for every arm. Exit 1 on failure.
import sys
import numpy as np
from shortcut import fresh_bed
p, d, r = fresh_bed(4000, n=16384, m=16)
own = (np.arange(16384) > 1023).mean()
par = (p[p >= 0] > 1023).mean()
ok = own > 0.9 and par > 0.9
print(("PASS" if ok else "FAIL") + f" id_range_gap own={own:.4f} parent={par:.4f}")
sys.exit(0 if ok else 1)
