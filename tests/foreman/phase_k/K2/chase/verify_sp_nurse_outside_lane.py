import sys, math, torch
sys.dont_write_bytecode = True
torch.manual_seed(0)

K0 = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0/chase"
sys.path.insert(0, K0)
import resolvent as R  # K0 dense reference

K2 = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K2/chase"
sys.path.insert(0, K2)
import resolvent_sp as SP  # file under review

torch.set_default_dtype(torch.float64)

def mk(B, H, S, D, seed):
    g = torch.Generator().manual_seed(seed)
    qs = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    k = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    v = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    return qs, k, v

def clone_leaf(t):
    return t.detach().clone().requires_grad_(True)

def run_case(B, H, S, D, c, g_gamma, seed):
    qs0, k0, v0 = mk(B, H, S, D, seed)

    # K0 densec path (pivot=True): x = (1-g)(I-gW)^-1 v, W sparsemax
    qsA, kA, vA = clone_leaf(qs0), clone_leaf(k0), clone_leaf(v0)
    xA = R._Dense.apply(qsA, kA, vA, g_gamma, "sparsemax", True)

    # K2 blocked path
    qsB, kB, vB = clone_leaf(qs0), clone_leaf(k0), clone_leaf(v0)
    xB = SP.read(qsB, kB, vB, g_gamma, c)

    fwd_diff = (xA - xB).abs().max().item()

    gx = torch.randn_like(xA)
    xA.backward(gx)
    xB.backward(gx)

    dq_diff = (qsA.grad - qsB.grad).abs().max().item()
    dk_diff = (kA.grad - kB.grad).abs().max().item()
    dv_diff = (vA.grad - vB.grad).abs().max().item()

    # Also check W itself (support / exact zeros) via weights_blocked vs R.weights
    WA = R.weights(qs0, k0, "sparsemax")
    WB = SP.weights_blocked(qs0, k0, c)
    w_diff = (WA - WB).abs().max().item()
    supp_diff = ((WA > 0) != (WB > 0)).sum().item()

    print(f"B{B} H{H} S{S} D{D} c{c} g{g_gamma}: fwd={fwd_diff:.3e} dq={dq_diff:.3e} dk={dk_diff:.3e} dv={dv_diff:.3e} W={w_diff:.3e} supp_mismatch={supp_diff}")
    return fwd_diff, dq_diff, dk_diff, dv_diff, w_diff, supp_diff

cases = [
    (1, 1, 7, 4, 3, 0.9, 1),    # partial last block (7 not mult of 3)
    (2, 3, 10, 5, 4, 0.999, 2), # partial last block (10 not mult of 4)
    (1, 2, 9, 4, 3, 0.999, 3),  # exact multiple blocks (9 = 3*3)
    (2, 2, 1, 4, 3, 0.5, 4),    # S < c, single partial block
    (1, 1, 6, 4, 2, 0.99, 5),   # multiple exact blocks
]

worst = 0.0
for args in cases:
    res = run_case(*args)
    worst = max(worst, max(res[:5]))

print("worst_diff", worst)
print("ALL OK" if worst < 1e-8 else "MISMATCH DETECTED")
