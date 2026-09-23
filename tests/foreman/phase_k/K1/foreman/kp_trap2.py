# Foreman K1: sc-HPD W62, 2-dim heads, period 4096 + twin ALiBi 2^-5, flat-top multiplexer, n = 16384. Bar: test_kp_trap2.py.
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kp_trap as T  # noqa: E402
out = {"slope": T.get_slopes(64)[39], "period": 4096.0, "beta": 3e7}
for L, kappa in ((7, 10.0), (4, 8.5)):
    out[str(L)] = T.chain(16384, 2, L, 62, np.array([4096.0]), 3e7, out["slope"], kappa)
    print(L, out[str(L)], flush=True)
(HERE / "kp_trap2.json").write_text(json.dumps(out, indent=1))
