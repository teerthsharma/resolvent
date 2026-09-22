"""Can the as-built I-AUT (f_Q) attention (l1_iaut.QuatVectorArm q/k + iaut _rope, non-causal,
pooled at the last query) be optimised to attend key 0 from the pooled query? Not a finding test:
it decides whether the I-AUT anchor is reachable (A5's is not)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l1_iaut as L
torch.manual_seed(0)
m = L.QuatVectorArm(64)
opt = torch.optim.Adam(list(m.q_proj.parameters()) + list(m.k_proj.parameters()) + list(m.sym_emb.parameters()), lr=1e-2)
def lp0(sym):
    B, S = sym.shape; e = m.sym_emb(sym); pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    q, k = L.IR._rope(m.q_proj(e), pos), L.IR._rope(m.k_proj(e), pos)
    return torch.log_softmax(q @ k.transpose(-1, -2) / 8.0, -1)[:, -1, 0]
g = torch.Generator().manual_seed(1)
for _ in range(800):
    s = torch.randint(0, 2, (256, 14), generator=g); opt.zero_grad(); (-lp0(s).mean()).backward(); opt.step()
with torch.no_grad():
    p = lp0(torch.randint(0, 2, (1024, 14), generator=g)).exp()
print(f"I-AUT pooled query -> key 0: mean p {p.mean():.4f}, min p {p.min():.4f}")
