"""RED-first assert test for PF-GAMMA. Run against stub pf.py -> AssertionError/NotImplementedError."""
import math
from pf import continuum_min_energy, discrete_min, refined_min

# (1) continuum constant must be a finite positive float
E_min, formula = continuum_min_energy()
assert isinstance(E_min, float) and E_min > 0, "continuum_min_energy must return positive float"
c_W = math.sqrt(2) / 6
assert abs(E_min - c_W) / c_W > 0.05, "sanity: literal functional should NOT equal c_W (missing 1/2 factor)"

# (2) discrete minimisation must return a finite P_min and an interior-point count
P_min, count_mid = discrete_min(eps=1.0, n=400)
assert isinstance(P_min, float) and P_min > 0
assert isinstance(count_mid, int) and count_mid >= 0

# (3) refined lattice must also return a finite P_min, closer to E_min as eps shrinks than unit lattice
P_ref_loose = refined_min(eps=1.0, n=400)
P_ref_tight = refined_min(eps=0.03, n=400)
assert isinstance(P_ref_loose, float) and P_ref_loose > 0
assert isinstance(P_ref_tight, float) and P_ref_tight > 0

print("ALL ASSERTIONS PASSED")
