"""RED if the as-built A5 layer-1 attention (a5_bed.Layer qkv + a5_bed._rope, causal) can be
OPTIMISED to put >= 0.9 weight on key 0 for every query position 1..63 on held-out sequences.
Only Q/K/embedding/LN are trained, directly on -log p(i->0); nothing else in the way."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a5_bed as A
torch.manual_seed(0)
layer, emb = A.Layer(A.D, A.N_HEADS, "fq"), torch.nn.Embedding(4, A.D)
ps = list(layer.qkv.parameters()) + list(emb.parameters()) + list(layer.ln1.parameters())
opt = torch.optim.Adam(ps, lr=1e-2)
def p0(tok):
    b, s = tok.shape
    x = layer.ln1(emb(tok)); q, k, _ = layer.qkv(x).chunk(3, -1)
    sh = lambda t: t.view(b, s, A.N_HEADS, A.D_HEAD).transpose(1, 2)
    pos = torch.arange(s).view(1, 1, s).expand(b, A.N_HEADS, s)
    Q, K = A._rope(sh(q), pos), A._rope(sh(k), pos)
    sc = (Q @ K.transpose(-1, -2)) / A.D_HEAD ** 0.5
    sc = sc.masked_fill(torch.triu(torch.ones(s, s, dtype=torch.bool), 1), float("-inf"))
    return torch.log_softmax(sc, -1)[..., 0]            # [B,H,S] log p(i->0)
g = torch.Generator().manual_seed(1)
for step in range(800):
    tok = torch.randint(0, 4, (64, A.L), generator=g)
    opt.zero_grad(); loss = -p0(tok)[..., 1:].mean(); loss.backward(); opt.step()
with torch.no_grad():
    lp = p0(torch.randint(0, 4, (256, A.L), generator=g)).exp()   # [B,H,S]
    best_head = lp[..., 1:].mean((0, 2)).argmax()
    per_q = lp[:, best_head, :].mean(0)                           # mean over held-out seqs
    worst = float(per_q[1:].min()); worst_i = int(per_q[1:].argmin()) + 1
print(f"final loss {loss.item():.4f}; best head {int(best_head)}; per-query p(i->0) at 13,25,38,63: "
      f"{[round(float(per_q[i]), 4) for i in (13, 25, 38, 63)]}; worst {worst:.4f} at i={worst_i}")
assert worst >= 0.9, f"RED: optimised as-built layer-1 attention reaches key 0 with p={worst:.4f} at query {worst_i} (bar 0.9)"
print("GREEN")
