# R-JEPA round 3 (Foreman): BAR.md Amendment A6 (sun/rjepa/r2/BAR.md). r2.py and its results are read-only here.
# python r3.py audit F Q [F Q ...]  -> results/audit_f{F}_q{Q}_s{seed}.json  (A6.1 tie-break audit + A6.3 true-dyn arms)
# python r3.py cell F [F ...]        -> results/res_f{F}_q3.0_s{seed}.json   (A6.2 gap-onset cells)
import hashlib, json, math, sys, time
from pathlib import Path
import torch

R2DIR = Path(__file__).resolve().parents[2] / "r2"
sys.path.insert(0, str(R2DIR))
import r2                                                                   # sets torch threads = 2
from r2 import DEV, K, D, H, MB, true_step, make_cell, hit, succ, lin1_feats, lin1_pick, rank01, tokens, train_head, base_for, Pred

HERE = Path(__file__).parent
OUT = HERE / "results"
R2OUT = R2DIR / "results"
BAR = R2DIR / "BAR.md"


@torch.no_grad()
def bayes_P3(b, dyn, sigma, f, r, M=2048, seed=0, chunk=16):
    """r2.bayes_P with the same draws, plus Ed = posterior-mean TRUE distance E|z_k - g| (the A6.1 tie-breaker)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Pb, Ps, Ed = [], [], []
    for i in range(0, len(b["y"]), chunk):
        y, a, goal = (b[k][i:i + chunk].to(DEV) for k in ("y", "a", "g"))
        c = len(y)
        e = math.sqrt(f) * torch.randn(c, M, 1, D, generator=gen, device=DEV) \
            + math.sqrt(1 - f) * torch.randn(c, M, K, D, generator=gen, device=DEV)
        z = y[:, None, None] - sigma * e
        for t in range(H):
            z = true_step(z, a[:, None, :, t], dyn)
        dist = (z - goal[:, None, None]).norm(dim=-1)
        Pb.append(torch.nn.functional.one_hot(dist.argmin(-1), K).float().mean(1))
        Ps.append((dist < r).float().mean(1))
        Ed.append(dist.mean(1))
    dev = b["y"].device
    return torch.cat(Pb).to(dev), torch.cat(Ps).to(dev), torch.cat(Ed).to(dev)


def bayes_pick(Pb, Ed, allowed=None):
    """argmax P(k best) over allowed k; exact ties go to the smallest true posterior-mean distance."""
    P = Pb if allowed is None else Pb.masked_fill(~allowed, -1.0)
    top = P == P.max(1, keepdim=True).values
    return Ed.masked_fill(~top, float("inf")).argmin(1)


def true_feats(b, dyn, sigma):
    return lin1_feats(lambda z, u: true_step(z, u, dyn), b["y"], b["a"], b["g"], sigma)


def load_pred(seed):
    net = Pred().to(DEV)
    net.load_state_dict(torch.load(R2OUT / f"pred_s{seed}.pt"))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def cid(f, q):
    return round(f * 100) * 100 + round(q * 10)


def ceiling(ev, dyn, sigma, f, r, seed, de):
    """Bayes picks under both tie-breaks + true-dynamics arms. Returns picks dict, tie rate, Ps."""
    Pb, Ps, Ed = bayes_P3(ev, dyn, sigma, f, r, M=MB, seed=seed)
    re_ = rank01(de)
    reach = re_ <= re_.min(1, keepdim=True).values + 0.4
    dt, st = true_feats(ev, dyn, sigma)
    picks = {"bayes": bayes_pick(Pb, Ed), "bayes_r2tie": (Pb - 1e-9 * de).argmax(1),
             "bayes_eps02": bayes_pick(Pb, Ed, reach),
             "bayes_eps02_r2tie": (Pb - 1e-9 * de).masked_fill(~reach, -1).argmax(1),
             "bayes_succ": Ps.argmax(1), "dist_true": dt.argmin(1), "lin1_true": lin1_pick(dt, st, r)}
    ties = float(((Pb == Pb.max(1, keepdim=True).values).sum(1) > 1).float().mean())
    return picks, ties


def ns_of(hits, hb):
    return {k: (h - 1 / K) / (hb - 1 / K) for k, h in hits.items()}


def audit(f, q, seed, dyn, calib):
    """A6.1 + A6.3 on a regenerated r2 eval set; old tie-break must reproduce r2's stored Bayes pick exactly."""
    t0 = time.time()
    res2 = json.loads((R2OUT / f"res_f{f}_q{q}_s{seed}.json").read_text())
    pk2 = torch.load(R2OUT / f"picks_f{f}_q{q}_s{seed}.pt")
    sigma, r = calib["sigma"][str(q)], res2["r"]
    ev = make_cell(res2["n_eval"], f, sigma, 10_000 * seed + cid(f, q) + 2, dyn)
    assert torch.equal(ev["best"].cpu().to(torch.int8), pk2["best"]), "eval set not reproduced"
    de, se = lin1_feats(load_pred(seed), ev["y"], ev["a"], ev["g"], sigma)
    picks, ties = ceiling(ev, dyn, sigma, f, r, seed, de)
    old_ok = bool(torch.equal(picks["bayes_r2tie"].cpu().to(torch.int8), pk2["bayes"]))
    arms2 = {k: v.to(DEV).long() for k, v in pk2.items() if k not in ("best", "bayes", "bayes_eps02", "bayes_succ")}
    hits = {k: hit(p, ev) for k, p in (picks | arms2).items()}
    hb_new, hb_old = hits["bayes"], hits["bayes_r2tie"]
    ns_new, ns_old = ns_of(hits, hb_new), ns_of(hits, hb_old)
    out = dict(f=f, q=q, seed=seed, r=r, sigma=sigma, n_eval=res2["n_eval"], old_tiebreak_reproduces_r2=old_ok,
               r2_bayes_hit=res2["hit"]["bayes"], tie_rate=ties, hit=hits, ns_new=ns_new, ns_old=ns_old,
               ns_move={k: ns_new[k] - ns_old[k] for k in ns_new},
               agree_old_new=float((picks["bayes"] == picks["bayes_r2tie"]).float().mean()),
               bar_sha256=hashlib.sha256(BAR.read_bytes()).hexdigest(), wall_s=round(time.time() - t0, 1))
    (OUT / f"audit_f{f}_q{q}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("f", "q", "seed", "old_tiebreak_reproduces_r2", "tie_rate", "agree_old_new", "wall_s")}
                     | {"hb_new": round(hb_new, 4), "hb_old": round(hb_old, 4)}), flush=True)


KINDS = ["hop1", "dj02", "dj0.5", "dj4L", "null"]


def run_cell(f, q, seed, dyn, calib, n_train=60_000, n_eval=20_000):
    """A6.2: r2.run_cell's bed, nets and training; Bayes with the A6.1 tie-break; true-dyn arms reported."""
    t0 = time.time()
    sigma = calib["sigma"][str(q)]
    net = load_pred(seed)
    tr = make_cell(n_train, f, sigma, 10_000 * seed + cid(f, q) + 1, dyn)
    ev = make_cell(n_eval, f, sigma, 10_000 * seed + cid(f, q) + 2, dyn)
    r = float(tr["dist"].min(1).values.median())
    vt, rt, lt, _, _ = tokens(net, tr, sigma, r)
    ve, re_, le, de, se = tokens(net, ev, sigma, r)
    del tr["a"], tr["y"], tr["g"], tr["dist"]
    picks, ties = ceiling(ev, dyn, sigma, f, r, seed, de)
    picks |= {"dist": de.argmin(1), "lin1": lin1_pick(de, se, r)}
    dsat, nparams = {}, {}
    for kd in KINDS:
        lab = tr["best"]
        if kd == "null":
            lab = lab[torch.randperm(len(lab), generator=torch.Generator().manual_seed(seed)).to(lab.device)]
        k2 = "point" if kd == "null" else kd
        h = train_head(k2, vt, base_for(k2, rt, lt), lab, seed=seed)
        nparams[kd] = sum(p.numel() for p in h.parameters())
        with torch.no_grad():
            bse, out, ds_ = base_for(k2, re_, le), [], []
            for i in range(0, n_eval, 1000):
                out.append(h(ve[i:i + 1000], bse[i:i + 1000]).argmax(1))
                if k2.startswith("dj"):
                    ds_.append((h.last_delta.abs() / h.eps).mean())
            picks[kd] = torch.cat(out)
            if ds_:
                dsat[kd] = float(torch.stack(ds_).mean())
        del h; torch.cuda.empty_cache()
    hits = {k: hit(p, ev) for k, p in picks.items()}
    res = dict(f=f, q=q, seed=seed, sigma=sigma, r=r, n_train=n_train, n_eval=n_eval, M=MB, tie_rate=ties,
               bar_sha256=hashlib.sha256(BAR.read_bytes()).hexdigest(), nparams=nparams, hit=hits,
               succ={k: succ(p, ev, r) for k, p in picks.items()}, delta_over_eps=dsat,
               ns=ns_of(hits, hits["bayes"]), ns_r2tie=ns_of(hits, hits["bayes_r2tie"]))
    res["gap_hit"] = hits["bayes"] - hits["dist"]
    res["wall_s"] = round(time.time() - t0, 1)
    torch.save({k: p.cpu().to(torch.int8) for k, p in picks.items()} | {"best": ev["best"].cpu().to(torch.int8)},
               OUT / f"picks_f{f}_q{q}_s{seed}.pt")
    (OUT / f"res_f{f}_q{q}_s{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("f", "q", "seed", "wall_s", "tie_rate", "gap_hit")}
                     | {"ns": {k: round(v, 3) for k, v in res["ns"].items()}}), flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dyn, calib = r2.make_dyn(), json.loads((R2OUT / "calib.json").read_text())
    a = [float(x) for x in sys.argv[2:]]
    if sys.argv[1] == "table":                                                 # prints only, writes table.json
        rows = []
        for p in sorted(OUT.glob("res_f*.json")) + sorted(OUT.glob("audit_f*.json")):
            r = json.loads(p.read_text())
            ns = r.get("ns") or r["ns_new"]
            row = dict(file=p.name, f=r["f"], q=r["q"], seed=r["seed"], tie_rate=round(r["tie_rate"], 4),
                       bayes_hit=round(r["hit"]["bayes"], 4), gap_hit=round(r["hit"]["bayes"] - r["hit"]["dist"], 4),
                       ns={k: round(v, 3) for k, v in ns.items()})
            if p.name.startswith("res_"):                                          # A6.2 reported quantities
                row["closable_ns"] = round(1 - ns["dist"], 3)
                row["frac"] = round((ns["dj4L"] - ns["dj02"]) / (1 - ns["dist"]), 3)
                pk = torch.load(OUT / p.name.replace("res_", "picks_").replace(".json", ".pt"))
                dv = (pk["dj4L"] == pk["best"]).float() - (pk["dj02"] == pk["best"]).float()
                row["edge_ns"] = round(ns["dj4L"] - ns["dj02"], 3)                # paired SE, denominator held fixed
                row["edge_se_ns"] = round(float(dv.std() / len(dv) ** 0.5) / (r["hit"]["bayes"] - 1 / K), 3)
            if "ns_move" in r:
                row["ns_move_max"] = round(max(abs(v) for v in r["ns_move"].values()), 4)
                row["reproduces_r2"] = r["old_tiebreak_reproduces_r2"]
            rows.append(row); print(json.dumps(row))
        (OUT / "table.json").write_text(json.dumps(rows, indent=1))
    elif sys.argv[1] == "audit":
        for f, q in zip(a[::2], a[1::2]):
            for sd in (0, 1, 2):
                if not (OUT / f"audit_f{f}_q{q}_s{sd}.json").exists():
                    audit(f, q, sd, dyn, calib)
    else:
        for f in a:
            for sd in (0, 1, 2):
                if not (OUT / f"res_f{f}_q3.0_s{sd}.json").exists():
                    run_cell(f, 3.0, sd, dyn, calib)
