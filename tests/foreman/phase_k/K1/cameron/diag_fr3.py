# Unbarred: what does the trained resolvent layer's W point at? Hop distance (in the chain) of each row's argmax key.
import sys, json, math
import numpy as np, torch
sys.dont_write_bytecode = True; sys.path.insert(0, ".")
import rdepth, bed_kp
run, n = sys.argv[1], int(sys.argv[2])
cfg = json.loads(open(f"runs/{run}/result.json").read())["config"]
hook, _ = rdepth.load_hook(cfg["hook"])
m = rdepth.Chain("fR", 4, 512, 5, hook, cfg["emb"], cfg["par_init"])
m.load_state_dict(torch.load(f"runs/{run}/model.pt", map_location="cpu")); m.eval()
att = m.blocks[-1].attn
b = bed_kp.make_test(np.random.default_rng([31, n, 0]), n)
att.capture = []
with torch.no_grad():
    m(torch.from_numpy(b["ids"].astype(np.int64))[None], torch.from_numpy(b["pids"].astype(np.int64))[None])
c = att.capture[0]; qs, kk = c["qs"][0], c["k"][0]
S, D = qs.shape[1], qs.shape[2]
p, d, r = b["parent"], b["depth"], b["root"]
for h in range(qs.shape[0]):
    am = np.zeros(S, int); mx = np.zeros(S)
    for t0 in range(0, S, 2048):
        t1 = min(S, t0 + 2048)
        lg = qs[h, t0:t1] @ kk[h, :t1].T / math.sqrt(D)
        lg = lg.masked_fill(torch.arange(t1)[None, :] > torch.arange(t0, t1)[:, None], float("-inf"))
        W = torch.softmax(lg, -1); v, a = W.max(-1); am[t0:t1] = a.numpy(); mx[t0:t1] = v.numpy()
    same = r[am] == r; hop = np.where(same, d - d[am], -1)
    print(f"head {h}: max weight mean {mx.mean():.3f}; argmax in own chain {same.mean():.3f}; argmax = root {(am == r).mean():.3f}; "
          f"argmax = self {(am == np.arange(S)).mean():.3f}; hop histogram (own chain) {np.bincount(np.clip(hop[same], 0, 40))[:41].tolist()}")
    for lo, hi in ((1, 33), (33, 129), (129, 257), (257, 10 ** 6)):
        s = (d >= lo) & (d < hi)
        print(f"   depth {lo}-{hi - 1}: argmax root {(am[s] == r[s]).mean():.3f}, own chain {same[s].mean():.3f}, max weight {mx[s].mean():.3f}")
