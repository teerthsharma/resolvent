# R-JEPA / Cameron round 4. Bar: BAR.md in this folder (registered before any seed-14/15/16 value existed).
#   python r4.py fixes -> fixes.json (record fixes 1-2: spent r3 caches, read-only)
#   python r4.py sharp -> sharp.json (claims B, B-route, I: T = 32, seeds 14/15/16, 1,024 fresh members)
#   python r4.py rev   -> rev.json   (claims R0, RM1, RM2: T 36/40/44/48/56, seeds 14/15/16, member split)
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r3"))
import r3                                        # stubs torch, imports r2 and rj (numpy-only)
from r3 import r2, rj

K, M, R = rj.K, r2.M, rj.R
SEEDS, LEADS, B = [14, 15, 16], [36, 40, 44, 48, 56], 656
CURVE = [640, 656, 704, 768, 896, 1024]
SPLIT = (slice(0, 3), slice(3, 128), slice(128, 256))       # ens3 | ceiling pick + class | reference
I_ARMS = ["b4_656", "b4_640", "b2p", "half", "ens80", "lin1", "ens3", "floor"]
FULL = lambda T: K * (M + 1) * T
CFG = json.loads((HERE.parent / "r3" / "tune.json").read_text())["chosen"]


def cache_path(seed, T):
    return HERE / "cache" / f"ev_s{seed}_T{T}.npz"


r3.cache_path = cache_path                        # r3.cached_batch now reads/writes r4/cache only


def regime(s):
    """0 = S (s < R/2), 1 = F (R/2 <= s < 2 pi), 2 = X (s >= 2 pi)."""
    return (s >= R / 2).astype(int) + (s >= 2 * np.pi)


def zstat(succ, H, pa, pb):
    """Realised vs expected difference of picks pa, pb, and the sd of the realised one (one posterior draw per row)."""
    I = np.arange(len(pa)); ha, hb = H[I, pa], H[I, pb]
    Pa, Pb, Pab = ha.mean(-1), hb.mean(-1), (ha & hb).mean(-1)
    V = np.where(pa == pb, 0.0, Pa + Pb - 2 * Pab - (Pa - Pb) ** 2)
    return float((succ[I, pa].astype(float) - succ[I, pb]).mean()), float((Pa - Pb).mean()), float(np.sqrt(np.clip(V, 0, None).sum()) / len(pa))


def p_hits(ev, seed, Mref=1024, chunk=25):
    """r3.p_ref's draws (stream [seed, T, 3], same chunking), keeping every member hit (N, K, Mref)."""
    rng = np.random.default_rng([seed, ev["T"], 3]); y, u = ev["y"], ev["u"]; N = len(y); H = np.empty((N, K, Mref), bool)
    for i in range(0, N, chunk):
        xm = y[i:i + chunk, None] + ev["delta"] * rng.standard_normal((min(chunk, N - i), Mref, 2))
        H[i:i + chunk] = rj.tdist(rj.roll(xm[:, None] + u[i:i + chunk, :, None], ev["T"]), rj.G) < R
    return H


def sharp_seed(seed, T=32):
    t0 = time.time(); ev = r3.cached_batch(seed, T); H = p_hits(ev, seed); P = H.mean(-1); h, d = ev["hits"], ev["d"]
    picks = dict(bayes=r2.pick_bayes(ev), b2p=r3.run_cfg(ev, CFG)[0], half=rj._tb(h[..., :128].mean(-1), d),
                 ens80=rj._tb(h[..., :80].mean(-1), d), lin1=rj._tb(ev["lin1"], d), ens3=rj._tb(h[..., :3].mean(-1), d),
                 floor=rj.pick_floor(ev))
    calls = {}
    for b in CURVE:
        picks[f"b4_{b}"], calls[b] = r3.race_budget(ev, CFG["c"], CFG["b"], CFG["eps"], b)
    bl, by = P.mean(), r3.escore(P, picks["bayes"]); ns = lambda a: float((r3.escore(P, picks[a]) - bl) / (by - bl))
    row = dict(T=T, blind_ref=float(bl), bayes_ref=float(by), floor_ref=float(r3.escore(P, picks["floor"])),
               p90ratio_b4_656=FULL(T) / float(np.percentile(calls[B], 90)), meanratio_b4_656=FULL(T) / float(calls[B].mean()),
               curve={b: dict(ns=ns(f"b4_{b}"), p90ratio=FULL(T) / float(np.percentile(calls[b], 90)),
                              meanratio=FULL(T) / float(calls[b].mean())) for b in CURVE}, z={}, zparts={})
    sb = rj.score(ev, picks["bayes"])
    for a in picks:
        row["ns_" + a] = ns(a)
        row["ns_realised_" + a] = float(r2.ns(rj.score(ev, picks[a]), ev["succ"].mean(), sb))
        row["agree_" + a] = float((picks[a] == picks["bayes"]).mean())
    for a in I_ARMS:
        Dr, De, sd = zstat(ev["succ"], H, picks[a], picks["bayes"])
        row["z"][a] = (Dr - De) / sd; row["zparts"][a] = dict(D_real=Dr, D_exp=De, sd=sd, sdns=sd / float(by - bl))
    row["sdns_b4_656"] = row["zparts"]["b4_656"]["sdns"]; row["wall_s"] = round(time.time() - t0, 1)
    print("sharp", seed, {a: round(row["ns_" + a], 5) for a in picks}, "p90", round(row["p90ratio_b4_656"], 3),
          "z", {a: round(v, 2) for a, v in row["z"].items()}, "sdns", round(row["sdns_b4_656"], 4), row["wall_s"], flush=True)
    return row


def rev_seed(seed):
    out = []
    for T in LEADS:
        t0 = time.time(); ev = r3.cached_batch(seed, T); h, d = ev["hits"], ev["d"]; I = np.arange(len(d)); e, c, r = SPLIT
        Pr = h[..., r].mean(-1); top3 = h[..., e].mean(-1)
        picks = dict(ceil=rj._tb(h[..., c].mean(-1), d), lin1=rj._tb(ev["lin1"], d), ens3=rj._tb(top3, d),
                     floor=rj.pick_floor(ev), bayes=r2.pick_bayes(ev))
        bl, ce = Pr.mean(), r3.escore(Pr, picks["ceil"]); den = ce - bl
        cls = regime(ev["stretch"][I, picks["ceil"], -1]); dP = (Pr[I, picks["lin1"]] - Pr[I, picks["ens3"]]) / den
        sb = rj.score(ev, picks["bayes"])
        row = dict(T=T, N=len(d), blind_ref=float(bl), ceil_ref=float(ce), floor_ref=float(r3.escore(Pr, picks["floor"])),
                   n=np.bincount(cls, minlength=3).tolist(), gsum=np.bincount(cls, weights=dP, minlength=3).tolist(),
                   ens3_tie_rate=float(((top3 == top3.max(1, keepdims=True)).sum(1) >= 2).mean()),
                   ens3_allzero_rate=float((top3.max(1) == 0).mean()))
        for a in ("lin1", "ens3", "floor", "bayes"):
            row["ns_" + a] = float((r3.escore(Pr, picks[a]) - bl) / den)
            row["ns_realised_" + a] = float(r2.ns(rj.score(ev, picks[a]), ev["succ"].mean(), sb))
        row["wall_s"] = round(time.time() - t0, 1); out.append(row); del ev
        print("rev", seed, T, "gap", round(row["ns_lin1"] - row["ns_ens3"], 4), "lin1", round(row["ns_lin1"], 4),
              "ens3", round(row["ns_ens3"], 4), "n", row["n"], "g", [round(g / max(n, 1), 4) for g, n in zip(row["gsum"], row["n"])],
              "tie", round(row["ens3_tie_rate"], 3), row["wall_s"], flush=True)
    return out


def fixes():
    """Record fixes 1-2 from spent r3 files, read-only."""
    S = json.loads((HERE.parent / "r3" / "sharp.json").read_text())["seeds"]
    half = {s: x["ns_half"] for s, x in S.items()}
    rows = {}
    for s in [6, 7, 8, 11, 12, 13]:
        z = np.load(HERE.parent / "r3" / "cache" / f"ev_s{s}_T32.npz")
        h = np.unpackbits(z["hits_p"], -1)[..., :M].astype(bool); d, succ = z["d"], z["succ"]
        ev = dict(succ=succ); bl = succ.mean(); sb = rj.score(ev, rj._tb(h.mean(-1), d))
        rows[str(s)] = {k: float(r2.ns(rj.score(ev, rj._tb(h[..., sl].mean(-1), d)), bl, sb))
                        for k, sl in (("half_1_128", slice(0, 128)), ("half_129_256", slice(128, 256)))}
        del h
    below = sum(v < 0.99 for r in rows.values() for v in r.values())
    return dict(fix1_ns_prime_half=dict(min=min(half.values()), max=max(half.values()), per_seed=half),
                fix2_realised_half_T32=rows, fix2_below_099=f"{below}/{2 * len(rows)}")


if __name__ == "__main__":
    t0 = time.time(); mode = sys.argv[1]
    meta = dict(bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256((HERE / "r4.py").read_bytes()).hexdigest())
    f = HERE / f"{mode}.json"
    if mode == "fixes":
        res = dict(meta, **fixes()); print(json.dumps(res, indent=1))
    else:
        res = json.loads(f.read_text()) if f.exists() else dict(meta, seeds={}, **(dict(B=B, M_ref=1024, cfg=CFG) if mode == "sharp" else dict(split="1-3 ens3 | 4-128 ceil+class | 129-256 ref")))
        for s in SEEDS:
            if str(s) not in res["seeds"]:
                res["seeds"][str(s)] = sharp_seed(s) if mode == "sharp" else rev_seed(s)
                res["script_sha256_" + str(s)] = meta["script_sha256"]; res["bar_sha256_" + str(s)] = meta["bar_sha256"]
                f.write_text(json.dumps(res, indent=1, default=float))
    if mode == "fixes":
        f.write_text(json.dumps(res, indent=1, default=float))
    print("wall", round(time.time() - t0, 1))
