"""Chase K2 bars, second registration (2026-09-23, after test_chase_k2.py's RED and before these run). RED first on
resolvent_sp_stub2.py (the stub plus an FRSP factory). Helpers, board line and exit rule as test_chase_k2.py.

Why: sp_support_S1024's mutant (_tau with support size + 1) cannot change the support set. With support size k and
tau = (cs_k - 1)/k, the mutant gives tau' = (k tau + z_(k+1)) / (k+1) with z_(k+1) <= tau, so z_(k+1) <= tau' <= tau:
every old support entry stays in and z_(k+1) stays out. That bar stays registered as written; this is its replacement.

BARS
 sp_support_S1024_v2   as sp_support_S1024 (weights_blocked vs R.weights(..., "sparsemax") float64, in1 and in2: {W > 0}
                       equal exactly; row sums within 1e-12 of 1; some row has support >= 2; some causal entry is 0.0),
                       with the mutant: _tau returns the smallest support logit z_(k) (so that entry's weight is exactly 0);
                       it must make the sets differ on in1.
 sp_train_ladder_R0_50 K0 train_ladder.py --rung R0 --attn $CHASE_HOOK:FRSP --steps 50 --ctx 1024 --batch 8 --eval_batches 2
                       --seed 1337 --no_poll (bf16 autocast default, FineWeb-Edu shards): rc 0; 50 step losses, all finite;
                       every eval val_loss finite (at least one); mean(loss steps 41-50) < mean(loss steps 1-10); config
                       attn_sha256 = sha256 of $CHASE_HOOK; ckpt.pt model has attn.a / attn.b only under blocks.3 and
                       max|a - 1| + max|b| > 0.
"""
import hashlib, json, math, os, shutil, subprocess, sys
sys.dont_write_bytecode = True
os.environ.setdefault("CHASE_HOOK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolvent_sp.py"))
import test_chase_k2 as T
import torch

TLPATH = T.REPO + "/tests/foreman/phase_k/K0/wilson/train_ladder.py"


def mut_tau_kth(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    kk = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    ksz = ((1 + kk * zs) > zs.cumsum(-1)).sum(-1, keepdim=True)
    return zs.gather(-1, ksz - 1)


@T.bar("sp_support_S1024_v2")
def t_support_v2():
    hk = T.load(T.HOOK, "hook_under_test")
    out, ok = {}, True
    tri = T.R._mask(1024, "cpu")
    for nm, (seed, a, b) in T.INS.items():
        qs, k, _, _ = T.inputs(seed, a, b, 1024)
        Wb = hk.weights_blocked(qs, k, 256)
        same = bool(torch.equal(Wb > 0, T.R.weights(qs, k, "sparsemax") > 0))
        rs = float((Wb.sum(-1) - 1).abs().max())
        nsupp = (Wb > 0).sum(-1)
        cz = int(((Wb == 0) & ~tri).sum())
        out[nm] = {"sets_equal": same, "rowsum_err": rs, "max_support": int(nsupp.max()), "causal_exact_zeros": cz}
        ok &= same and rs <= 1e-12 and int(nsupp.max()) >= 2 and cz > 0
    qs, k, _, _ = T.inputs(0, (1.0, 1.0), (0.0, 0.0), 1024)
    orig = hk._tau
    hk._tau = mut_tau_kth
    try:
        out["mutant_tau_kth_sets_equal"] = bool(torch.equal(hk.weights_blocked(qs, k, 256) > 0, T.R.weights(qs, k, "sparsemax") > 0))
    finally:
        hk._tau = orig
    ok &= not out["mutant_tau_kth_sets_equal"]
    return ok, out


@T.bar("sp_train_ladder_R0_50")
def t_tl():
    d = os.path.join(T.HERE, "runs", f"{T.TAG}_tl_R0_50")
    shutil.rmtree(d, ignore_errors=True)
    cmd = [sys.executable, TLPATH, "--rung", "R0", "--attn", T.HOOK + ":FRSP", "--steps", "50", "--ctx", "1024", "--batch", "8",
           "--eval_batches", "2", "--seed", "1337", "--no_poll", "--out_dir", d]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    os.makedirs(d, exist_ok=True)
    with open(d + ".stdout.log", "w") as f:
        f.write(p.stdout)
    with open(d + ".stderr.log", "w") as f:
        f.write(p.stderr)
    lp = os.path.join(d, "log.jsonl")
    recs = [json.loads(l) for l in open(lp)] if os.path.exists(lp) else []
    steps = [r["loss"] for r in recs if r.get("t") == "step"]
    ev = [r["val_loss"] for r in recs if r.get("t") == "eval"]
    cfg = [r for r in recs if r.get("t") == "config"]
    val = {"rc": p.returncode, "n_steps": len(steps), "first10": sum(steps[:10]) / 10 if len(steps) >= 10 else None,
           "last10": sum(steps[-10:]) / 10 if len(steps) >= 10 else None, "eval": ev,
           "sha_ok": bool(cfg) and cfg[0].get("attn_sha256") == hashlib.sha256(open(T.HOOK, "rb").read()).hexdigest()}
    ok = p.returncode == 0 and len(steps) == 50 and all(math.isfinite(s) for s in steps) and len(ev) >= 1 and \
        all(math.isfinite(e) for e in ev) and val["last10"] < val["first10"] and val["sha_ok"]
    cp = os.path.join(d, "ckpt.pt")
    if os.path.exists(cp):
        sd = torch.load(cp, map_location="cpu", weights_only=False)["model"]
        ka = sorted(x for x in sd if x.endswith(".attn.a"))
        kb = sorted(x for x in sd if x.endswith(".attn.b"))
        val["a_keys"], val["b_keys"] = ka, kb
        if ka == ["blocks.3.attn.a"] and kb == ["blocks.3.attn.b"]:
            val["moved"] = float((sd[ka[0]] - 1).abs().max() + sd[kb[0]].abs().max())
            ok &= val["moved"] > 0
        else:
            ok = False
    else:
        ok = False
    return ok, val


if __name__ == "__main__":
    print(json.dumps({"hook": T.HOOK, "sha256": hashlib.sha256(open(T.HOOK, "rb").read()).hexdigest()}), flush=True)
    t_support_v2()
    t_tl()
    print(json.dumps(T.results), flush=True)
    sys.exit(0 if all(s == "green" for s in T.results.values()) else 1)
