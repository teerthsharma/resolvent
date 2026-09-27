# P2-DJ: Bayes ceiling on ranking K candidate actions on Lorenz-63 vs lead time t/T*.
# Bar: sun/canon/p2/BAR_DJ.md incl. Amendment A1 (sha256 recorded in the output). Perfect model, no learning.
import hashlib, json, time
from collections import deque
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
LAM, DT, K, M, MB, N, NB = 0.9059, 0.01, 4, 64, 64, 2000, 4000
GRID = 25                                             # absolute grid: every 0.25 time units
REL_LEADS = np.arange(1, 19) * 0.25                   # t / T*, out to 4.5
CONS = ("point", "win", "cum")


def lorenz(x, F):
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([10 * (b - a), a * (28 - c) - b, a * b - 8 / 3 * c], -1) + F


def rk4(x, F):
    k1 = lorenz(x, F); k2 = lorenz(x + DT / 2 * k1, F); k3 = lorenz(x + DT / 2 * k2, F); k4 = lorenz(x + DT * k3, F)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def clim_states(rng, n):
    # ponytail: vectorised burn-in of perturbed copies instead of S2's sequential spacing; 50+ time units decorrelate them
    x = np.array([1.0, 1.0, 1.0]) + 1e-3 * rng.standard_normal((n, 3))
    for _ in range(5000 + int(rng.integers(200, 600))):
        x = rk4(x, 0.0)
    return x


def actions(family, rng, n):
    if family == "impulse":
        u = rng.standard_normal((n, K, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
        return 1.0 * u, np.zeros((K, 3))
    F = np.zeros((K, 3)); F[:, 0] = (np.arange(K) - 1.5) * 2
    return np.zeros((n, K, 3)), F


def vote(Jm):                                         # Jm (N,K,members) -> top-1 Bayes pick
    win = Jm.argmax(1)                                # (N,members)
    return np.stack([(win == k).mean(1) for k in range(K)], 1).argmax(1)


def spearman(a, b):
    ra = a.argsort(1).argsort(1).astype(float); rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return float(((ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))).mean())


def run(cfg):
    family, delta, seed = cfg
    rng = np.random.default_rng([seed, int(-np.log10(delta)), family == "forcing"])
    Tstar = np.log(1 / delta) / LAM
    n_grid = int(np.ceil(4.5 * Tstar / DT / GRID))
    w = max(1, int(round(0.5 * Tstar / DT / GRID)))   # J^win window in grid units
    x0 = clim_states(rng, N)
    y = x0 + delta * rng.standard_normal((N, 3))
    off, F = actions(family, rng, N)
    X = x0[:, None] + off                                                      # truth (N,K,3)
    E = y[:, None, None] + delta * rng.standard_normal((N, 1, M, 3)) + off[:, :, None]   # posterior, CRN over k
    pool = clim_states(rng, NB)
    B = pool[rng.integers(0, NB, (N, MB))][:, None] + off[:, :, None]         # blind ensemble (N,K,MB,3)
    Fx, Fe = F[None], F[None, :, None]
    cs = {n: np.zeros(a.shape[:-1]) for n, a in (("X", X), ("E", E), ("B", B))}   # running sums of x
    hist = deque([{k: v.astype(np.float32) for k, v in cs.items()}], maxlen=w + 1)   # ponytail: f32 window snapshots for RAM
    rows = []
    for g in range(1, n_grid + 1):
        for _ in range(GRID):
            X, E, B = rk4(X, Fx), rk4(E, Fe), rk4(B, Fe)
            cs["X"] += X[..., 0]; cs["E"] += E[..., 0]; cs["B"] += B[..., 0]
        hist.append({k: v.astype(np.float32) for k, v in cs.items()})
        cm = B.mean((0, 2))                                                    # each action's clim mean (K,3)
        row = dict(t=g * GRID * DT, t_over_Tstar=g * GRID * DT / Tstar,
                   state_skill=float(1 - ((E.mean(2) - X) ** 2).sum(-1).mean() / ((X - cm) ** 2).sum(-1).mean()))
        for c in CONS:
            if c == "point":
                J = {"X": X[..., 0], "E": E[..., 0], "B": B[..., 0]}
            elif c == "cum":
                J = {k: cs[k] / (g * GRID) for k in cs}
            else:
                old = hist[0]; span = (len(hist) - 1) * GRID
                J = {k: (cs[k] - old[k]) / span for k in cs}
            best = J["X"].argmax(1)
            hb, hbm, hbl = (vote(J["E"]) == best).mean(), (J["E"].mean(-1).argmax(1) == best).mean(), (vote(J["B"]) == best).mean()
            row[c] = dict(hit_bayes=float(hb), hit_bayes_meanargmax=float(hbm), hit_blind=float(hbl),
                          excess=float(hb - hbl), spearman=spearman(J["E"].mean(-1), J["X"]))
        rows.append(row)
    return dict(family=family, delta=delta, seed=seed, Tstar=float(Tstar), win_time=w * GRID * DT, rows=rows)


def at(rows, r):                                      # nearest absolute grid point to t/T* = r
    return min(rows, key=lambda q: abs(q["t_over_Tstar"] - r))


def read_cell(rows, Tstar):
    S = [(r, at(rows, r)["state_skill"]) for r in REL_LEADS]
    tS = next((r for r, s in S if s < 0.1), None)
    out = dict(t_S_over_Tstar=tS)
    for c in ("point", "win", "cum"):
        band = [at(rows, r)[c]["excess"] for r in REL_LEADS if tS is not None and r >= 1.5 * tS]
        void = tS is None or len(band) < 4
        E0 = at(rows, 0.25)[c]["excess"]
        tR = None                                     # linear interpolation on the absolute grid from 0.25 T*
        pts = [q for q in rows if q["t_over_Tstar"] >= 0.25 - 1e-9]
        for a, b in zip(pts, pts[1:]):
            ea, eb = a[c]["excess"], b[c]["excess"]
            if ea >= 0.1 * E0 > eb:
                tR = a["t"] + (ea - 0.1 * E0) / (ea - eb) * (b["t"] - a["t"]); break
        out[c] = dict(void=void, n_band=len(band), late_excess=None if void else float(np.mean(band)), t_R=tR,
                      E_first=E0)
    return out


if __name__ == "__main__":
    t0 = time.time()
    cfgs = [(f, d, s) for f in ("impulse", "forcing") for d in (1e-2, 1e-4) for s in (0, 1, 2)]
    with Pool(3) as p:                                # RAM-bound on this box, not core-bound
        runs = p.map(run, cfgs)
    summary = {}
    for f in ("impulse", "forcing"):
        for d in (1e-2, 1e-4):
            rs = [r for r in runs if r["family"] == f and r["delta"] == d]
            def mean(key_fn):
                return float(np.mean([key_fn(r) for r in rs]))
            mrows = []
            for i in range(len(rs[0]["rows"])):
                q = dict(t=rs[0]["rows"][i]["t"], t_over_Tstar=rs[0]["rows"][i]["t_over_Tstar"],
                         state_skill=mean(lambda r: r["rows"][i]["state_skill"]))
                for c in CONS:
                    q[c] = {k: mean(lambda r: r["rows"][i][c][k]) for k in rs[0]["rows"][i][c]}
                mrows.append(q)
            cell = read_cell(mrows, rs[0]["Tstar"])
            cell["per_seed"] = [read_cell(r["rows"], r["Tstar"]) for r in rs]
            summary[f"{f}_{d:.0e}"] = dict(Tstar=rs[0]["Tstar"], win_time=rs[0]["win_time"], rows=mrows, **cell)
    keys = list(summary)
    verdicts = {}
    for c in ("point", "win"):
        late = {k: summary[k][c]["late_excess"] for k in keys}
        pred = all(v is not None and v <= 0.03 for v in late.values())
        counter = any(all(late[f"{f}_{d:.0e}"] is not None and late[f"{f}_{d:.0e}"] >= 0.08 for d in (1e-2, 1e-4))
                      for f in ("impulse", "forcing"))
        verdicts[c] = dict(late_excess=late, prediction=pred, counter=counter,
                           verdict="PREDICTION" if pred and not counter else "COUNTER" if counter and not pred else "OPEN")
    null_hit = [at(summary[f"impulse_{d:.0e}"]["rows"], 3.0)["point"]["hit_bayes"] for d in (1e-2, 1e-4)]
    ceil_hit = {k: at(summary[k]["rows"], 0.25)["point"]["hit_bayes"] for k in keys}
    null_ok = all(abs(h - 0.25) <= 0.03 for h in null_hit)
    ceil_ok = all(h >= 0.95 for h in ceil_hit.values())
    tr = [summary[f"impulse_{d:.0e}"]["point"]["t_R"] for d in (1e-2, 1e-4)]
    law = float(np.log(100) / LAM)
    scaling = None if None in tr else dict(t_R=tr, diff=tr[1] - tr[0], law=law, holds=bool(abs((tr[1] - tr[0]) / law - 1) <= 0.3))
    out = dict(bar_sha256=hashlib.sha256((HERE / "BAR_DJ.md").read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               N=N, M=M, MB=MB, K=K, NB=NB, seeds=[0, 1, 2], verdicts=verdicts, null_hit_impulse_3Tstar=null_hit,
               null_ok=null_ok, ceiling_hit_0p25Tstar=ceil_hit, ceiling_ok=ceil_ok, horizon_scaling=scaling,
               bed_valid=null_ok and ceil_ok, summary=summary, wall_clock_s=round(time.time() - t0, 1))
    (HERE / "results_dj.json").write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(k, "T*=%.2f win=%.2f t_S/T*=%s" % (v["Tstar"], v["win_time"], v["t_S_over_Tstar"]),
              {c: (v[c]["late_excess"], v[c]["n_band"], v[c]["t_R"]) for c in CONS})
        for r in REL_LEADS:
            q = at(v["rows"], r)
            print("  t/T*=%.2f S=%.3f " % (r, q["state_skill"]) + " ".join(
                "%s hit=%.3f blind=%.3f E=%+.3f rho=%.2f |" % (c, q[c]["hit_bayes"], q[c]["hit_blind"], q[c]["excess"], q[c]["spearman"]) for c in CONS))
    print("verdicts", verdicts)
    print("null", null_hit, null_ok, "ceiling", ceil_hit, ceil_ok, "scaling", scaling, "wall", out["wall_clock_s"])
