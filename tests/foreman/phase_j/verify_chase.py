"""Verify Chase row measurements"""
import numpy as np
import chase_q2q3 as impl

# Q2 test
print("Q2: Stated generators exceed cap")
r = impl.q2_closure()
print(f"  closure_su2: {r['closure_su2']}")
print(f"  closure_so3: {r['closure_so3']}")
print(f"  exceeds_cap: {r['exceeds_cap']}")
print(f"  commutator_norm: {impl.q2_commutator_norm()}")
print(f"  gate_commutes: {impl.q2_gate_commutes()}")
print()

# Q2-REROUTE test
print("Q2-REROUTE: A5 icosahedron closes at bar")
a5, a3 = impl.a5_icosahedron_pair()
r2 = impl.q2_closure(gens=(a5, a3))
print(f"  closure_su2: {r2['closure_su2']}")
print(f"  closure_so3: {r2['closure_so3']}")
print(f"  exceeds_cap: {r2['exceeds_cap']}")
cayley_ok = impl.q2_cayley_check(gens=(a5, a3))
print(f"  cayley_check: {cayley_ok}")
print()

# Q3 test
print("Q3: Reads and gate diffs")
d = impl.q3_diff()
print(f"  diff_out_AB_minus_out_BA_norm: {d}")
g_scalar, g_phase = impl.q3_gate_diffs()
print(f"  scalar_gate_diff: {g_scalar}")
print(f"  u1_phase_gate_diff: {g_phase}")
