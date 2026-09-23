# Foreman K1: sc-HPD k3 on bed_k' at (4096, L7) for W = 11..20 (outside R0 pricing). Bar: test_kp_pricing.py.
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kp_attack as K  # noqa: E402
beds = [K.bed_kp.make_test(np.random.default_rng([22, 4096, k]), 4096) for k in range(8)]
cal = [K.bed_kp.make_test(np.random.default_rng([23, 4096, k]), 4096) for k in range(8)]
curve = {W: K.run(beds, 7, W, 0, K.calibrate(cal, 7, W, 0))[0] for W in range(11, 21)}
cross = [W for W, v in curve.items() if v > 0.5]
out = {"crossing_W": min(cross) if cross else None, "curve": {str(W): v for W, v in curve.items()}}
(HERE / "kp_pricing.json").write_text(json.dumps(out, indent=1))
print(out)
