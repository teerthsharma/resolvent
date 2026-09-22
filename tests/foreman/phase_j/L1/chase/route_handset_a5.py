"""Route object for the leap: a hand-set f_Q read (no training) on the A5 word problem.
Token k carries the fixed SU(2) generator of its letter; BOS carries value 1 and gate 1;
the query attends to BOS with any positive weight p; out = p * Pi_L, read out by the
nearest of the 120 binary-icosahedral elements, scored against the SO(3) group table."""
import sys; sys.dont_write_bytecode = True
import numpy as np, indep
from indep import M, Q, dag
phi = (1 + 5 ** .5) / 2
a2, b2 = indep.su2((0, 1, phi), 72), indep.su2((1, 1, 1), 120)
gens = [a2, b2, dag(a2), dag(b2)]
els = np.stack([Q(m) for m in indep.closure_su2([a2, b2])])            # 120 x 4
def truth(word):                                                       # SO(3) product, independent path
    a3, b3 = indep.rot((0, 1, phi), 72), indep.rot((1, 1, 1), 120)
    R = np.eye(3)
    for w in word: R = R @ [a3, b3, a3.T, b3.T][w]
    return R
def so3_of(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def run(L, n, dtype, seed=0):
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n):
        word = rng.integers(0, 4, L)
        Pi = np.eye(2, dtype=dtype)
        for w in word[::-1]: Pi = (gens[w].astype(dtype) @ Pi).astype(dtype)   # Pi_L = g_1 ... g_L applied right-to-left as a scan would
        p = 0.37                                                                 # any positive attention weight on BOS
        out = Q(p * Pi.astype(np.complex128)); out /= np.linalg.norm(out)
        best = els[np.argmax(np.abs(els @ out))]                                 # nearest element up to sign
        ok += np.allclose(so3_of(best), truth(word), atol=1e-6)
    return ok / n
if __name__ == "__main__":
    for L, n, dt in [(64, 500, np.complex128), (64, 500, np.complex64), (4096, 50, np.complex64)]:
        print(L, n, dt.__name__, run(L, n, dt))
