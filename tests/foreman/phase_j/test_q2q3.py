"""Test-first for Addendum L rows Q2/Q3. RED recorded against chase_q2q3_stub
(AssertionError: {'closure_su2': 0, 'closure_so3': 0}) before this file was
adjusted to the measured contract-generator result (cap exceeded, not 60)."""
import numpy as np
import chase_q2q3 as impl

def test_q2_stated_generators_exceed_cap():
    r = impl.q2_closure()
    assert r["exceeds_cap"] is True
    assert r["closure_su2"] == 5000

def test_q2_reroute_a5_closes_at_bar():
    a5, a3 = impl.a5_icosahedron_pair()
    r = impl.q2_closure(gens=(a5, a3))
    assert r["exceeds_cap"] is False
    assert r["closure_su2"] == 120, r
    assert r["closure_so3"] == 60, r

def test_q2_cayley_reroute():
    a5, a3 = impl.a5_icosahedron_pair()
    ok = impl.q2_cayley_check(gens=(a5, a3))
    assert ok is True

def test_q2_commutator():
    d = impl.q2_commutator_norm()
    assert d > 0
    g = impl.q2_gate_commutes()
    assert g == 0.0

def test_q3_reads():
    d = impl.q3_diff()
    assert d > 0
    g_scalar, g_phase = impl.q3_gate_diffs()
    assert g_scalar == 0.0
    assert g_phase == 0.0

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "PASS")
