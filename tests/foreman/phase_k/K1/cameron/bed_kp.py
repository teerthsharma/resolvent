"""bed_k' (Cameron, it.K1): the replacement R-DEPTH bed after bed_k was voided at (L = 7, n = 4096) by Foreman's HPD
window (0.5642 > 0.5, Dispatcher ruling). The generator is bed_k's, unchanged (bed_k.make_bed + _pack: each token picks
one of M lanes i.i.d. uniform, parent = the lane's previous token, n distinct ids from V = 32,768, a root's parent id
is NULL = V, target = the root's id). Only the lane count changes, and it differs by role:
  train: M = 32 lanes, depth cap 32 (a lane restarts as a new root), n in {256, 512, 1024}
  test : M = 8 lanes, no cap, n in {4096, 8192, 16384}
The property that defeats the window: a positional window lengthens a pointer's reach by a constant factor c > 1 over
2^L (it catches the next hop when the hop's positional offset falls in W slots), so the beyond-2^L set must sit mostly
past c * 2^L. Test beds buy that with depth (8 lanes: max depth ~ n/8, i.e. 4.2 * 2^7 at n = 4096 against ~2.2 with
16 lanes); training beds, whose depth is capped at 32 = 2 * 2^4 and cannot buy depth, buy it with sparse gaps
(32 lanes: mean gap 32 >> W = 10, so the window rarely catches). K1.F' (test_k1fp.py) binds both.
"""
import sys
sys.dont_write_bytecode = True
from bed_k import make_bed, _pack, _last_root, V, NULL  # noqa: F401  (bed_k copied unchanged, sha aa60e6d9...)

M_TRAIN, M_TEST, CAP = 32, 8, 32


def make_train(rng, n=1024):
    return _pack(*make_bed(n, M_TRAIN, rng, cap=CAP), rng)


def make_test(rng, n):
    return _pack(*make_bed(n, M_TEST, rng), rng)
