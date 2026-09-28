# R-JEPA round 4 (Foreman): Amendment A7 (r4/foreman/BAR.md). r2.py, r3.py and their results are imported read-only.
# python r4.py lin2              -> results/lin2_f0.0_q1.0_s{seed}.json   (A7.2)
# python r4.py cell F [F ...]    -> results/res_f{F}_q2.0_s{seed}.json    (A7.1, n_eval 60,000)
# python r4.py table             -> results/table.json
import hashlib, json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
R3DIR = HERE.parents[1] / "r3" / "foreman"
sys.path.insert(0, str(R3DIR))
import r3                                                                    # imports r2 (torch threads = 2)
from r3 import r2
from r2 import DEV, K, D, H

R2OUT = r3.R2OUT
OUT = HERE / "results"
BAR4, BAR2 = HERE / "BAR.md", r3.BAR


def prov():
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(sha256_r4_py=h(__file__), sha256_r3_py=h(r3.__file__), sha256_r2_py=h(r2.__file__),
                sha256_bar_r4=h(BAR4), sha256_bar_r2=h(BAR2))


def calib2(dyn):
    return r2.calibrate(dyn, ratios=(1.0, 2.0, 3.0))


def jac(step, y, a, chunk=250):
    """Noiseless rollout z (n, K, D) and full Jacobian J[n, k, i, j] = dz_i / dy_j (D VJPs per chunk)."""
    zs, Js = [], []
    for i in range(0, len(y), chunk):
        yy = y[i:i + chunk].to(DEV)[:, None].expand(-1, a.shape[1], -1).clone().requires_grad_(True)
        with torch.enable_grad():
            z = r2.rollout(step, yy, a[i:i + chunk].to(DEV))
            rows = [torch.autograd.grad(z[..., k].sum(), yy, retain_graph=k < D - 1)[0] for k in range(D)]
        zs.append(z.detach()); Js.append(torch.stack(rows, -2))
    return torch.cat(zs), torch.cat(Js)


@torch.no_grad()
def lin2_prob(mu, J, sigma, r, M2=16384, seed=4242, chunk=256):
    """P(|mu + sigma J w| < r), w ~ N(0, I_D); one common draw set for every candidate (CRN)."""
    W = torch.randn(M2, D, generator=torch.Generator(device=DEV).manual_seed(seed), device=DEV, dtype=mu.dtype)
    m, Jf = mu.reshape(-1, D).to(DEV), J.reshape(-1, D, D).to(DEV)
    out = []
    for i in range(0, len(m), chunk):
        x = m[i:i + chunk, None] + sigma * W @ Jf[i:i + chunk].transpose(1, 2)         # (c, M2, D)
        out.append((x.pow(2).sum(-1) < r * r).float().mean(1))
    return torch.cat(out).reshape(mu.shape[:-1])


def lin2_pick(P, d):
    return (P - 1e-9 * d).argmax(1)                                           # ties to the smaller distance


def lin2(seed, dyn, calib):
    """A7.2 at (0, 1) on the regenerated r2 eval set; must reproduce r3's stored audit hits."""
    t0 = time.time()
    f, q = 0.0, 1.0
    a3 = json.loads((R3DIR / "results" / f"audit_f{f}_q{q}_s{seed}.json").read_text())
    sigma, r = calib["sigma"][str(q)], a3["r"]
    ev = r2.make_cell(a3["n_eval"], f, sigma, 10_000 * seed + r3.cid(f, q) + 2, dyn)
    net = r3.load_pred(seed)
    de, _ = r2.lin1_feats(net, ev["y"], ev["a"], ev["g"], sigma)
    picks, ties = r3.ceiling(ev, dyn, sigma, f, r, seed, de)
    for name, step in (("lin2_true", lambda z, u: r2.true_step(z, u, dyn)), ("lin2", net)):
        z, J = jac(step, ev["y"], ev["a"])
        mu = z - ev["g"][:, None]
        P = lin2_prob(mu, J, sigma, r)
        picks[name] = lin2_pick(P, mu.norm(dim=-1))
        del z, J, mu, P; torch.cuda.empty_cache()
    hits = {k: r2.hit(p, ev) for k, p in picks.items()}
    ns = r3.ns_of(hits, hits["bayes"])
    rep = all(abs(hits[k] - a3["hit"][k]) < 1e-9 for k in ("bayes", "bayes_succ", "lin1_true", "dist_true"))
    bv = {k: (p.to(ev["best"].device) == ev["best"]).float() for k, p in picks.items()}
    den = hits["bayes"] - 1 / K
    se = {k: float((bv[k] - bv["lin1_true"]).std() / len(bv[k]) ** 0.5 / den) for k in ("lin2_true", "bayes_succ", "lin2")}
    out = dict(f=f, q=q, seed=seed, r=r, sigma=sigma, n_eval=a3["n_eval"], M2=16384, reproduces_r3=rep, tie_rate=ties,
               hit=hits, succ={k: r2.succ(p, ev, r) for k, p in picks.items()}, ns=ns, se_vs_lin1_true_ns=se,
               residue=ns["bayes_succ"] - ns["lin1_true"], closed=ns["lin2_true"] - ns["lin1_true"],
               wall_s=round(time.time() - t0, 1)) | prov()
    (OUT / f"lin2_f{f}_q{q}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("seed", "reproduces_r3", "residue", "closed", "wall_s")}), flush=True)


def cell(f, seed, dyn, calib):
    """A7.1: r3.run_cell verbatim (bed, arms, training, A6.1 Bayes) at sigma/rho = 2, n_eval 60,000; output here."""
    r3.OUT = OUT
    r3.run_cell(f, 2.0, seed, dyn, calib, n_train=60_000, n_eval=60_000)
    p = OUT / f"res_f{f}_q2.0_s{seed}.json"
    res = json.loads(p.read_text())
    res |= prov() | dict(calib_rho=calib["rho"], calib_sigma=calib["sigma"])
    p.write_text(json.dumps(res, indent=1))


def table():
    rows, now = [], prov()["sha256_r4_py"]
    for p in sorted(OUT.glob("res_f*.json")) + sorted(OUT.glob("lin2_f*.json")):
        r = json.loads(p.read_text())
        row = dict(file=p.name, seed=r["seed"], stale=r.get("sha256_r4_py") != now, bayes_hit=round(r["hit"]["bayes"], 4),
                   ns={k: round(v, 4) for k, v in r["ns"].items()})
        if p.name.startswith("res_"):
            pk = torch.load(OUT / p.name.replace("res_", "picks_").replace(".json", ".pt"))
            dv = (pk["dj4L"] == pk["best"]).float() - (pk["dj02"] == pk["best"]).float()
            den = r["hit"]["bayes"] - 1 / K
            row |= dict(closable_ns=round(1 - r["ns"]["dist"], 4), edge_ns=round(r["ns"]["dj4L"] - r["ns"]["dj02"], 4),
                        edge_se_ns=round(float(dv.std() / len(dv) ** 0.5) / den, 4), gap_hit=round(r["gap_hit"], 4),
                        null_ok=r["hit"]["null"] <= r["hit"]["dist"] + 3 * (r["hit"]["dist"] * (1 - r["hit"]["dist"]) / r["n_eval"]) ** 0.5)
        else:
            row |= dict(residue=round(r["residue"], 4), closed=round(r["closed"], 4), se=r["se_vs_lin1_true_ns"],
                        reproduces_r3=r["reproduces_r3"])
        rows.append(row); print(json.dumps(row))
    (OUT / "table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "table":
        table(); sys.exit()
    dyn = r2.make_dyn()
    calib = json.loads((R2OUT / "calib.json").read_text())
    if sys.argv[1] == "lin2":
        for sd in (0, 1, 2):
            if not (OUT / f"lin2_f0.0_q1.0_s{sd}.json").exists():
                lin2(sd, dyn, calib)
    else:
        c2 = calib2(dyn)
        assert all(abs(c2["sigma"][k] - calib["sigma"][k]) <= 1e-6 for k in ("1.0", "3.0")), "A7.1: calibration void"
        for f in [float(x) for x in sys.argv[2:]]:
            for sd in (0, 1, 2):
                if not (OUT / f"res_f{f}_q2.0_s{sd}.json").exists():
                    cell(f, sd, dyn, c2)
