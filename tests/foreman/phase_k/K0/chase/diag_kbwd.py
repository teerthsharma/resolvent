# diagnostic (not a bar): precision of the fp32 efficient-attention backward itself vs float64, plain SDPA, S=1024
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "."); from test_chase_k0 import _qkv, rel
q, k, v = _qkv(1024, seed=2); g = _qkv(1024, seed=3)[0]
t32 = [x.clone().requires_grad_() for x in (q, k, v)]
F.scaled_dot_product_attention(*t32, is_causal=True).backward(g)
t64 = [x.double().clone().requires_grad_() for x in (q, k, v)]
S = 1024; z = (t64[0] @ t64[1].transpose(-1, -2)) / 8.0
z = z.masked_fill(torch.ones(S, S, dtype=bool, device="cuda").triu(1), float("-inf"))
(torch.softmax(z, -1) @ t64[2]).backward(g.double())
print({n: rel(a.grad, b.grad) for n, a, b in zip("qkv", t32, t64)})
