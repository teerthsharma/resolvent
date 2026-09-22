# diagnostic (not a bar): the gradcheck bars must be able to fail. Mutants of the blocked adjoint:
# (1) dv * 1.01; (2) the right-looking push acc += W^T u removed; (3) dense adjoint uses the untransposed solve.
import os, sys, inspect, torch
os.environ["CHASE_BOARD"] = "0"; sys.path.insert(0, ".")
import resolvent as R, test_chase_k0 as T
orig = R.backward_blocked
R.backward_blocked = lambda qs, k, x, gx, g, c=256: (lambda r: (r[0], r[1], r[2] * 1.01))(orig(qs, k, x, gx, g, c))
print("mutant dv*1.01:"); T.t_gc_blocked()
src = inspect.getsource(orig).replace("acc[:, :, :lo] += P[..., :lo].transpose(-1, -2) @ uJ", "pass")
ns = dict(R.__dict__); exec(src, ns); R.backward_blocked = ns["backward_blocked"]
print("mutant no push:"); T.t_gc_blocked()
R.backward_blocked = orig
tri = torch.linalg.solve_triangular
torch.linalg.solve_triangular = lambda A, B, upper, **kw: tri(A.transpose(-1, -2), B, upper=not upper) if upper else tri(A, B, upper=upper)
print("mutant untransposed dense adjoint:"); T.t_gc_soft()
