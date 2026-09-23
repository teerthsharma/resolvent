# Unbarred probe: sharpen the trained resolvent's softmax at eval (logit scale x k, no retraining); accuracy by depth.
import sys, json
import numpy as np, torch
sys.dont_write_bytecode = True; sys.path.insert(0, ".")
import rdepth, bed_kp
run, n = sys.argv[1], int(sys.argv[2])
cfg = json.loads(open(f"runs/{run}/result.json").read())["config"]
hook, _ = rdepth.load_hook(cfg["hook"])
m = rdepth.Chain("fR", 4, 512, 5, hook, cfg["emb"], cfg["par_init"])
m.load_state_dict(torch.load(f"runs/{run}/model.pt", map_location="cpu")); m.eval()
att = m.blocks[-1].attn
a0, b0 = att.a.data.clone(), att.b.data.clone()
beds = [bed_kp.make_test(np.random.default_rng([31, n, k]), n) for k in range(2)]
for k in (1.0, 1.5, 2.0, 3.0, 5.0):
    att.a.data, att.b.data = a0 * k, b0 * k
    oks, ds = [], []
    for b in beds:
        oks.append(rdepth.predict(m, b, torch.device("cpu"), False) == b["target"]); ds.append(b["depth"])
    ok, d = np.concatenate(oks), np.concatenate(ds)
    f = lambda lo, hi: ok[(d >= lo) & (d < hi)].mean()
    print(f"scale x{k}: acc {ok.mean():.4f} | 0-128 {f(0,129):.3f} 129-256 {f(129,257):.3f} 257-512 {f(257,513):.3f} 513+ {f(513,10**6):.3f} | band>160 {ok[d>160].mean():.4f}", flush=True)
