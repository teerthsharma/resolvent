import time, json
import numpy as np
import chase_q2q3 as C
t=time.time(); r=C.q2_closure(); t1=time.time()-t
print(json.dumps({"stated_closure_su2":r["closure_su2"],"stated_closure_so3":r["closure_so3"],"exceeds_cap":r["exceeds_cap"],"seconds":t1}))
t=time.time(); r2=C.q2_closure(C.a5_icosahedron_pair()); t2=time.time()-t
print(json.dumps({"reroute_closure_su2":r2["closure_su2"],"reroute_closure_so3":r2["closure_so3"],"exceeds_cap":r2["exceeds_cap"],"seconds":t2}))
print(json.dumps({"cayley_check_reroute":C.q2_cayley_check(C.a5_icosahedron_pair())}))
print(json.dumps({"commutator":C.q2_commutator_norm(),"q3_diff":C.q3_diff(),"q3_gate_diffs":C.q3_gate_diffs()}))
