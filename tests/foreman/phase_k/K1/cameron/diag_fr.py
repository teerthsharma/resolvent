# Unbarred diagnostic of a trained fR checkpoint: where does the resolvent layer's W put its mass? (CPU, CUDA hidden)
import sys, json, math
import numpy as np, torch
sys.dont_write_bytecode = True; sys.path.insert(0, ".")
import rdepth, bed_kp
run = sys.argv[1]
cfg = json.loads(open(f"runs/{run}/result.json").read())["config"]
hook, _ = rdepth.load_hook(cfg["hook"])
m = rdepth.Chain("fR", 4, 512, 5, hook, cfg["emb"], cfg["par_init"])
m.load_state_dict(torch.load(f"runs/{run}/model.pt", map_location="cpu")); m.eval()
att = m.blocks[-1].attn
print("a", att.a.data.tolist(), "b", att.b.data.tolist())
out = {}
for k in range(4):
    b = bed_kp.make_train(np.random.default_rng([21, k]))
    att.capture = []
    with torch.no_grad():
        m(torch.from_numpy(b["ids"].astype(np.int64))[None], torch.from_numpy(b["pids"].astype(np.int64))[None])
    c = att.capture[0]; qs, kk = c["qs"][0], c["k"][0]               # H, S, D
    S = qs.shape[1]; D = qs.shape[2]
    lg = qs @ kk.transpose(-1, -2) / math.sqrt(D)
    lg = lg.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    W = torch.softmax(lg, -1).numpy()                                  # H, S, S
    p = b["parent"]; i = np.arange(S); tgt = np.where(p < 0, i, p)     # parent, or self for a root
    for h in range(W.shape[0]):
        mass = W[h, i, tgt]; am = W[h].argmax(-1)
        out.setdefault(h, []).append((float(mass[p >= 0].mean()), float(mass[p < 0].mean()), float((am == tgt)[p >= 0].mean()),
                                      float((am == tgt)[p < 0].mean()), float((am == i)[p >= 0].mean())))
for h, v in out.items():
    v = np.array(v).mean(0)
    print(f"head {h}: mass on parent (non-roots) {v[0]:.4f}, mass on self (roots) {v[1]:.4f}, argmax=parent {v[2]:.4f}, "
          f"argmax=self at roots {v[3]:.4f}, argmax=self at non-roots {v[4]:.4f}")
