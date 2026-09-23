# Unbarred: resolvent-layer pointer quality of an fR checkpoint on bed_kp test beds at length n (CPU, CUDA hidden).
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
print("a", [round(x, 3) for x in att.a.data.tolist()], "b", [round(x, 3) for x in att.b.data.tolist()])
b = bed_kp.make_train(np.random.default_rng([21, 0])) if n == 1024 else bed_kp.make_test(np.random.default_rng([31, n, 0]), n)
att.capture = []
with torch.no_grad():
    m(torch.from_numpy(b["ids"].astype(np.int64))[None], torch.from_numpy(b["pids"].astype(np.int64))[None])
c = att.capture[0]; qs, kk = c["qs"][0], c["k"][0]
S, D = qs.shape[1], qs.shape[2]
p = b["parent"]; i = np.arange(S); tgt = np.where(p < 0, i, p)
for h in range(qs.shape[0]):
    mass = np.zeros(S); am = np.zeros(S, int)
    for t0 in range(0, S, 2048):
        t1 = min(S, t0 + 2048)
        lg = qs[h, t0:t1] @ kk[h, :t1].T / math.sqrt(D)
        lg = lg.masked_fill(torch.arange(t1)[None, :] > torch.arange(t0, t1)[:, None], float("-inf"))
        W = torch.softmax(lg, -1)
        mass[t0:t1] = W[torch.arange(t1 - t0), torch.from_numpy(tgt[t0:t1])].numpy(); am[t0:t1] = W.argmax(-1).numpy()
    q = lambda lo, hi: (lambda s: f"{mass[s].mean():.3f}/{(am[s] == tgt[s]).mean():.3f}")(((i >= lo) & (i < hi)))
    print(f"n={n} head {h}: target mass/argmax-hit by position: [0,1k) {q(0,1024)}  [1k,4k) {q(1024,4096)}  [4k,16k) {q(4096,16384)}  roots {mass[p<0].mean():.3f}")
