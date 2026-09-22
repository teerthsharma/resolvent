"""Verify all Chase Python test rows"""
import json
import chase_q2q3 as impl

# Expected values from record
expected_q2 = {
    "closure_su2": 5000,
    "closure_so3": 3393,
    "exceeds_cap": True,
    "commutator_norm_ab_minus_ba": 0.8312538755549068,
    "commutative_gate_diff": 0.0,
}

expected_q2_reroute = {
    "closure_su2": 120,
    "closure_so3": 60,
    "exceeds_cap": False,
    "cayley_check": True,
}

expected_q3 = {
    "diff_out_AB_minus_out_BA_norm": 4.552964986550147,
    "scalar_gate_diff": 0.0,
    "u1_phase_gate_diff": 0.0,
}

# Test Q2
print("=== Q2 ===")
r_q2 = impl.q2_closure()
q2_match = True
checks = [
    ("closure_su2", r_q2['closure_su2'], expected_q2['closure_su2']),
    ("closure_so3", r_q2['closure_so3'], expected_q2['closure_so3']),
    ("exceeds_cap", r_q2['exceeds_cap'], expected_q2['exceeds_cap']),
    ("commutator_norm", impl.q2_commutator_norm(), expected_q2['commutator_norm_ab_minus_ba']),
    ("gate_commutes", impl.q2_gate_commutes(), expected_q2['commutative_gate_diff']),
]

for name, actual, expected in checks:
    match = actual == expected or (isinstance(actual, float) and abs(actual - expected) < 1e-14)
    status = "✓" if match else "✗"
    print(f"  {status} {name}: actual={actual}, expected={expected}")
    if not match:
        q2_match = False

print(f"Q2 VERDICT: {'CLEAN' if q2_match else 'STRUCK'}")
print()

# Test Q2-REROUTE
print("=== Q2-REROUTE ===")
a5, a3 = impl.a5_icosahedron_pair()
r_q2r = impl.q2_closure(gens=(a5, a3))
cayley_ok = impl.q2_cayley_check(gens=(a5, a3))
q2r_match = True

checks_q2r = [
    ("closure_su2", r_q2r['closure_su2'], expected_q2_reroute['closure_su2']),
    ("closure_so3", r_q2r['closure_so3'], expected_q2_reroute['closure_so3']),
    ("exceeds_cap", r_q2r['exceeds_cap'], expected_q2_reroute['exceeds_cap']),
    ("cayley_check", cayley_ok, expected_q2_reroute['cayley_check']),
]

for name, actual, expected in checks_q2r:
    match = actual == expected
    status = "✓" if match else "✗"
    print(f"  {status} {name}: actual={actual}, expected={expected}")
    if not match:
        q2r_match = False

print(f"Q2-REROUTE VERDICT: {'CLEAN' if q2r_match else 'STRUCK'}")
print()

# Test Q3
print("=== Q3 ===")
d_q3 = impl.q3_diff()
g_scalar, g_phase = impl.q3_gate_diffs()
q3_match = True

checks_q3 = [
    ("diff_out_AB_minus_out_BA_norm", d_q3, expected_q3['diff_out_AB_minus_out_BA_norm']),
    ("scalar_gate_diff", g_scalar, expected_q3['scalar_gate_diff']),
    ("u1_phase_gate_diff", g_phase, expected_q3['u1_phase_gate_diff']),
]

for name, actual, expected in checks_q3:
    match = actual == expected or (isinstance(actual, float) and abs(actual - expected) < 1e-14)
    status = "✓" if match else "✗"
    print(f"  {status} {name}: actual={actual}, expected={expected}")
    if not match:
        q3_match = False

print(f"Q3 VERDICT: {'CLEAN' if q3_match else 'STRUCK'}")
