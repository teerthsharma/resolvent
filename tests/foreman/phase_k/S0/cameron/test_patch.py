"""PATCH CHECK: CEQAttention.forward = flite.flite_forward on an R.build_repaired smprime model gives the
same loss and parameter grads as the unpatched dense model with theta_head zeroed (phase off, theta == 0
exactly), hard-concrete magnitude, to <= 1e-5 relative. `--stub` patches the softmax twin forward instead."""
import sys, torch
SP = r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
REPO = r"C:/Users/seal/Desktop/New folder (32)"
sys.path.insert(0, SP); sys.path.insert(0, REPO)
import r1_gate as R, q2_certificate as Q2
from ceq import arm_smprime
from ceq.hf.modeling_ceq import CEQAttention
import flite

ORIG = CEQAttention.forward
arm_smprime.magnitude = R.FORMS["hard_concrete"]


def build_and_grad(fwd):
    CEQAttention.forward = fwd
    try:
        torch.manual_seed(0)
        m = R.build_repaired(hidden_size=128, n_layers=3, n_heads=4, seq=256, vocab_size=256, operator="smprime").cuda()
        with torch.no_grad():
            for layer in m.model.layers:
                layer.self_attn.theta_head.weight.zero_(); layer.self_attn.theta_head.bias.zero_()
                layer.self_attn.beta.fill_(1.06)
        x = torch.randint(0, 256, (2, 256), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        return loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None and "theta_head" not in n}
    finally:
        CEQAttention.forward = ORIG


l_ref, g_ref = build_and_grad(ORIG)
l_got, g_got = build_and_grad(Q2.softmax_forward if "--stub" in sys.argv else flite.flite_forward)
worst = abs(float(l_got - l_ref)) / abs(float(l_ref))
for n, g in g_ref.items():
    got = g_got.get(n)
    worst = max(worst, 1.0 if got is None else float((got - g).abs().max() / g.abs().max().clamp_min(1e-30)))
ok = worst <= 1e-5
print(f"{'GREEN' if ok else 'RED'} patch: worst rel (loss + {len(g_ref)} param grads) {worst:.3e} vs bar 1e-05")
sys.exit(0 if ok else 1)
