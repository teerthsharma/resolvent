"""V2 shortcut library check: bed_k' is VOID beyond 2^L and NOT void on the far band (L=7). Run:
python sun/daedalus/engine/test_shortcuts.py"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shortcuts import void_check  # noqa: E402

near = void_check("bed_kp", 4096, 7, 22, "beyond")
assert near["void"] and abs(near["scores"]["schpd_14slot"] - 0.5107702349869452) < 1e-12, near
far = void_check("bed_kp", 16384, 7, 31, "far")
assert not far["void"] and max(far["scores"].values()) <= 1 / 8 + 0.03, far
try:                                   # n=4096 has no token past 896: an empty band must not read as "not void"
    void_check("bed_kp", 4096, 7, 22, "far")
    raise AssertionError("empty far band was scored")
except ValueError:
    pass
print(f"PASS near VOID ({near['best']} {near['score']:.4f}); far not void (max {far['score']:.4f} on {far['band_tokens']} tokens)")
