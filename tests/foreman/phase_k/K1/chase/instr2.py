"""instr.py with one change: Cameron checkpoints are read on their OWN evaluation beds, as rdepth.py builds them now
(BED[cfg bed] = (module, held-out train seed s_tr, test seed s_te); S 1024: module.make_train(default_rng([s_tr, 0]));
S 16384: module.make_test(default_rng([s_te, 16384, 0]), 16384)), and the model is built with the run's emb/par_init.
Reason: after instr.py was registered, Cameron moved R-DEPTH to bed_k' (bed_kp.py) and rdepth.py no longer exports
make_train/make_test, so instr.capture cannot read his checkpoints."""
import importlib.util, os, sys
import numpy as np
import torch

_s = importlib.util.spec_from_file_location("instr_base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "instr.py"))
base = importlib.util.module_from_spec(_s)
_s.loader.exec_module(base)
globals().update({k: v for k, v in vars(base).items() if not k.startswith("__")})


@torch.no_grad()
def capture(ck, lengths):
    if ck["lane"] != "cameron":
        return base.capture(ck, lengths)
    dev = torch.device("cuda")
    rd = base.load(base.K1 + "/cameron/rdepth.py", "cam_rdepth_v2")
    cfg = ck["cfg"]
    hook_cls = rd.load_hook(cfg["hook"])[0] if cfg["arm"] == "fR" else None
    model = rd.Chain(cfg["arm"], cfg["layers"], cfg["dff"], cfg["loops"], hook_cls,
                     emb=cfg.get("emb", "learned"), par_init=cfg.get("par_init", "normal"))
    model.load_state_dict(torch.load(ck["path"], map_location="cpu"))
    model = model.to(dev).eval()
    mod, s_tr, s_te = rd.BED[cfg.get("bed", "k")]
    res = {}
    for S in lengths:
        bed = mod.make_train(np.random.default_rng([s_tr, 0])) if S == 1024 else \
            mod.make_test(np.random.default_rng([s_te, S, 0]), S)
        ids = torch.from_numpy(bed["ids"].astype(np.int64))[None].to(dev)
        pids = torch.from_numpy(bed["pids"].astype(np.int64))[None].to(dev)
        res[S] = base._run(model, lambda: model(ids, pids))
        res[S]["bed"] = bed
        res[S]["bed_name"] = cfg.get("bed", "k")
        rd._BIAS_CACHE.clear()
        torch.cuda.empty_cache()
    return res
