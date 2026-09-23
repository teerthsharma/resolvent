# CPU pilot (unbarred): does the twin config learn at all? Seed 999, B=8, 600 steps. Not a measurement of any bar.
import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
import twin
from train_ladder import get_lr
torch.set_num_threads(24)
torch.manual_seed(999); g = torch.Generator(); g.manual_seed(999)
L, B, n, steps, lr = int(sys.argv[1]), 8, 1024, 600, 1e-3
m = twin.Twin(L)
dec = [p for nm, p in m.named_parameters() if p.dim() == 2 and not nm.startswith("E.")]
rest = [p for nm, p in m.named_parameters() if not (p.dim() == 2 and not nm.startswith("E."))]
opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
tr = [twin.bed_k.make_train(np.random.default_rng([991, k])) for k in range(8)]
def ev():
    with torch.no_grad():
        ids = torch.tensor(np.stack([b["ids"] for b in tr])); pids = torch.tensor(np.stack([b["pids"] for b in tr]))
        ok = (torch.gather(ids, 1, m(ids, pids).argmax(-1)).numpy() == np.stack([b["target"] for b in tr])).ravel()
        d = np.stack([b["depth"] for b in tr]).ravel()
        return {k: round(float(v), 4) for k, v in dict(d0_2=ok[d <= 2].mean(), d0_4=ok[d <= 4].mean(), d0_16=ok[d <= 16].mean(), d17_32=ok[d >= 17].mean()).items()}
t0 = time.time()
for s in range(1, steps + 1):
    for gr in opt.param_groups: gr["lr"] = get_lr(s, steps, 100, lr)
    ids, pids, depth, root = twin.batch(g, B, n, 32, "cpu")
    loss = F.cross_entropy(m(ids, pids).view(-1, n), root.view(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if s % 100 == 0:
        print(s, round(loss.item(), 4), ev(), f"{time.time()-t0:.0f}s", flush=True)
