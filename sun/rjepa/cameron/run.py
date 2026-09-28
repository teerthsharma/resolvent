# python run.py floors   -> floors.json (blind / floor / bayes / oracle, bed validity, deciding band)
# python run.py arms     -> arms.json   (needs floors.json; reads the band from it)
import hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import rj

HERE = Path(__file__).parent
LEADS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
SEEDS = [0, 1, 2]
NTR, NEV = 20000, 10000


def rng_for(seed, T, tag):
    return np.random.default_rng([seed, T, {"eval": 0, "train": 1, "arm": 2}[tag]])


def floors_one(seed):
    rows = []
    for T in LEADS:
        b = rj.make_batch(rng_for(seed, T, "eval"), NEV, T)
        rows.append(dict(T=T, blind=float(b["succ"].mean()), floor=float(rj.score(b, rj.pick_floor(b))),
                         bayes=float(rj.score(b, rj.pick_bayes(b))), oracle=float(b["succ"].any(1).mean())))
    return rows


def ns(s, r):
    return (s - r["blind"]) / (r["bayes"] - r["blind"])


def arms_one(seed):
    torch.set_num_threads(2)
    out = []
    for T in LEADS:
        ra = rng_for(seed, T, "arm")
        ev = rj.make_batch(rng_for(seed, T, "eval"), NEV, T)
        tr = rj.make_batch(rng_for(seed, T, "train"), NTR, T, need_post=(T == 2))
        row = dict(T=T, blind=float(ev["succ"].mean()), floor=float(rj.score(ev, rj.pick_floor(ev))),
                   bayes=float(rj.score(ev, rj.pick_bayes(ev))), ens3=float(rj.score(ev, rj.pick_ens3(ev))))
        t0 = time.perf_counter(); p_lin = rj.lin_prob(ev, ra); pk = rj._tb(p_lin, ev["d"]); t_lin = time.perf_counter() - t0
        row["lin"] = float(rj.score(ev, pk))
        pc, ok = rj.pick_cert(ev, ra); row["cert"] = float(rj.score(ev, pc)); row["cert_abstain"] = float(1 - ok.mean())
        # label-shuffle null (bed validity 3)
        perm = ra.permuted(np.tile(np.arange(rj.K), (NEV, 1)), axis=1)
        sh = np.take_along_axis(ev["succ"], perm, 1)
        row["null_max_dev"] = float(max(abs(sh[np.arange(NEV), p].mean() - sh.mean())
                                        for p in (rj.pick_floor(ev), rj.pick_bayes(ev), pk, pc)))
        p_lin_tr = rj.lin_prob(tr, ra)
        wall = dict(lin_s=t_lin)
        for name, variant, steps, bs in (("dj", "DJ", 1000, 8), ("djl", "DJ", 1000, 256), ("djs", "DJs", 1000, 256),
                                         ("lindj", "LINDJ", 1000, 256)):
            v, base = rj.tokens(tr, variant, p_lin_tr)
            t0 = time.perf_counter(); net = rj.train_dj(v, base, tr["succ"], steps, bs, seed); wall[name + "_train_s"] = time.perf_counter() - t0
            ve, be = rj.tokens(ev, variant, p_lin)
            t0 = time.perf_counter(); pk_dj = rj.pick_dj(net, ve, be); wall[name + "_infer_s"] = time.perf_counter() - t0
            row[name] = float(rj.score(ev, pk_dj))
            if T == 2:
                row[name + "_train_ns"] = float(ns(rj.score(tr, rj.pick_dj(net, v, base)),
                                                   dict(blind=tr["succ"].mean(), bayes=rj.score(tr, rj.pick_bayes(tr)))))
        row["params_dj"] = sum(p.numel() for p in net.parameters())
        # TOPO diagnostic: H0 of the K-cloud of predicted endpoints, torus metric
        pf, pb = rj.pick_floor(ev), rj.pick_bayes(ev)
        event = (~ev["succ"][np.arange(NEV), pf]) & ev["succ"][np.arange(NEV), pb]
        bars = np.stack([rj.h0_bars(z) for z in ev["zh"]])
        spread = np.log(rj.sigma(ev)[np.arange(NEV), pf])
        row["event_rate"] = float(event.mean())
        row["auc"] = {k: rj.auc(x, event) for k, x in (("h0_longest", bars[:, -1]), ("h0_shortest", bars[:, 0]),
                                                       ("h0_total", bars.sum(1)), ("spread", spread))}
        row["wall"] = wall
        out.append(row)
        print(seed, T, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("wall",)}, flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time(); mode = sys.argv[1]
    meta = dict(bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(),
                script_sha256={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("run.py", "rj.py")})
    with Pool(3) as p:
        runs = p.map(floors_one if mode == "floors" else arms_one, SEEDS)
    m = lambda k, i: float(np.mean([r[i][k] for r in runs]))
    if mode == "floors":
        mean = [{k: m(k, i) for k in runs[0][i]} for i in range(len(LEADS))]
        band = [r["T"] for r in mean if r["bayes"] - r["floor"] >= 0.03 and r["bayes"] - r["blind"] >= 0.10]
        valid = dict(ceiling=mean[0]["bayes"] >= 0.98 * mean[0]["oracle"],
                     gap=sum(r["bayes"] - r["floor"] >= 0.03 for r in mean) >= 2)
        res = dict(meta, per_seed=runs, mean=mean, band=band, valid=valid, wall_s=round(time.time() - t0, 1))
        for r in mean:
            print(r)
        print("band", band, "valid", valid)
    else:
        res = dict(meta, per_seed=runs, wall_s=round(time.time() - t0, 1))
    (HERE / f"{mode}.json").write_text(json.dumps(res, indent=1))
    print("wall", res["wall_s"])
