"""EXACTNESS BAR (registered): f-lite output and grads (q, k, v, u, beta) match the dense
ceq/arm_smprime.py readout at theta = 0 (phase_route "gate", hard-concrete magnitude) to
<= 1e-5 relative (max|a-b| / max|b|) in fp32 at S=256, H=4, d_head=32, with exact gate zeros,
for beta in {0.9, 1.0, 1.06}. `--stub` swaps f-lite for plain causal SDPA; it must exit non-zero."""
import sys, json, torch, torch.nn.functional as F
REPO = r"C:/Users/seal/Desktop/New folder (32)"
SP = r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
sys.path.insert(0, REPO); sys.path.insert(0, SP)
from ceq import arm_smprime
import r1_gate as R
import flite

BAR = 1e-5
STUB = "--stub" in sys.argv
B, H, S, D = 2, 4, 256, 32


def draw(dtype):
    g = torch.Generator().manual_seed(20260923)
    q, k, v = (torch.randn(B, H, S, D, generator=g) for _ in range(3))
    u = torch.randn(B, S, generator=g) * 1.5 + 1.0
    pick = torch.rand(B, S, generator=g)
    u = torch.where(pick < 0.05, torch.full_like(u, -5.0), u)   # hard-concrete exact zeros
    u = torch.where(pick > 0.95, torch.full_like(u, 5.0), u)    # exact ones
    w = torch.randn(B, H, S, D, generator=g)
    return [t.to("cuda", dtype) for t in (q, k, v, u, w)]


def run(route, beta, dtype):
    q, k, v, u, w = [t.clone().requires_grad_(i < 4) for i, t in enumerate(draw(dtype))]
    b = torch.tensor(beta, device="cuda", dtype=dtype, requires_grad=True)
    qk = torch.tensor(1.0, device="cuda", dtype=dtype, requires_grad=True)
    gg = torch.tensor(1.0, device="cuda", dtype=dtype, requires_grad=True)
    if route == "dense":
        orig = arm_smprime.magnitude
        arm_smprime.magnitude = R.FORMS["hard_concrete"]
        try:
            uu = u.unsqueeze(-2)
            out = arm_smprime.readout(q, k, v, uu, torch.zeros_like(uu), beta=b, qk=qk, g=gg,
                                      phase_route="gate").real
        finally:
            arm_smprime.magnitude = orig
    elif STUB:
        out = F.scaled_dot_product_attention(q * qk, k, v, is_causal=True) + 0 * (b + gg + u.sum())
    else:
        out = flite.flite_attention(q, k, v, u, b, qk, gg)
    (out * w).sum().backward()
    zeros = int((R.FORMS["hard_concrete"](u.detach()) == 0).sum())
    return dict(out=out.detach(), q=q.grad, k=k.grad, v=v.grad, u=u.grad, beta=b.grad,
                qk=qk.grad, g=gg.grad), zeros


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def main():
    assert torch.equal(flite.hard_concrete(torch.linspace(-6, 6, 1001)),
                       R.FORMS["hard_concrete"](torch.linspace(-6, 6, 1001))), "hard-concrete copy drifted"
    rows, ok = [], True
    for beta in (0.9, 1.0, 1.06):
        ref, zeros = run("dense", beta, torch.float32)
        got, _ = run("flite", beta, torch.float32)
        o64, _ = run("dense", beta, torch.float64)
        row = dict(cfg_beta=beta, exact_zero_gates=zeros)
        for name in ref:
            row[name] = rel(got[name], ref[name])
            row[name + "_vs_fp64:flite"] = rel(got[name], o64[name])
            row[name + "_vs_fp64:dense32"] = rel(ref[name], o64[name])
        barred = ("out", "q", "k", "v", "u", "beta")
        row["max_barred"] = max(row[n] for n in barred)
        row["pass"] = row["max_barred"] <= BAR and zeros > 0
        ok &= row["pass"]
        rows.append(row)
        print(json.dumps({k2: (f"{v2:.3e}" if isinstance(v2, float) else v2) for k2, v2 in row.items()}))
    if not STUB:
        json.dump(rows, open("exactness.json", "w"), indent=1)
    worst = max(r["max_barred"] for r in rows)
    line = f"{'GREEN' if ok else 'RED'} exactness: worst barred rel {worst:.3e} vs bar {BAR:.0e} ({'stub' if STUB else 'flite'})"
    print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
