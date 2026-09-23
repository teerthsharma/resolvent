"""Chase K1 instrument bar for clause6_v2's float64 reference. Registered before it runs; RED first with CHASE_STUB=1.
 x64_blocked_matches_dense  test_chase_k1_clause6_v2.x64_blocked (row blocks c = 128) vs the hook's dense float64 forward
                            (hook._dense_fwd, K0 math), CPU float64, B1 H2 S 512 D 64, q, k, v ~ N(0,1) seed 3, SSMax a = 1,
                            g = .999: max|a - b| / max|b| <= 1e-12."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import test_chase_k1_ckpt_v2 as T


@T.bar("x64_blocked_matches_dense")
def t_x64():
    T._stub()
    import torch, test_chase_k1_clause6_v2 as C6, instr2 as I
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, 2, 512, 64, generator=g, dtype=torch.float64) for _ in range(3))
    qs = I.R.ssmax_q(q, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    a, b = C6.x64_blocked(qs, k, v, 0.999, c=128), I.hk._dense_fwd(qs, k, v, 0.999)
    e = float((a - b).abs().max() / b.abs().max())
    return e <= 1e-12, {"rel": e}


if __name__ == "__main__":
    t_x64()
    print("SUMMARY", json.dumps(T.results)); sys.exit(1 if "red" in T.results.values() else 0)
