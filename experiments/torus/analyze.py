# python analyze.py -> verdict.json: reads floors.json, arms.json (+ lin1.json if present) against BAR.md
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
F = json.loads((HERE / "floors.json").read_text()); A = json.loads((HERE / "arms.json").read_text())
band = F["band"]
ARMS = ["floor", "ens3", "lin", "cert", "dj", "djl", "djs", "lindj"]
L1 = json.loads((HERE / "lin1.json").read_text()) if (HERE / "lin1.json").exists() else None


def ns(r, a):
    return (r[a] - r["blind"]) / (r["bayes"] - r["blind"])


tab = {}
for i, T in enumerate(r["T"] for r in A["per_seed"][0]):
    rows = [s[i] for s in A["per_seed"]]
    if L1:
        for r, l in zip(rows, (s[i] for s in L1)):
            r["lin1"] = l["lin1"]
    arms = ARMS + (["lin1"] if L1 else [])
    tab[T] = {a: dict(s=float(np.mean([r[a] for r in rows])), ns=float(np.mean([ns(r, a) for r in rows])),
                      ns_sd=float(np.std([ns(r, a) for r in rows]))) for a in arms}
    tab[T]["bayes"] = float(np.mean([r["bayes"] for r in rows])); tab[T]["blind"] = float(np.mean([r["blind"] for r in rows]))
    tab[T]["cert_abstain"] = float(np.mean([r["cert_abstain"] for r in rows]))
    tab[T]["null_max_dev"] = float(max(r["null_max_dev"] for r in rows))
    tab[T]["event_rate"] = float(np.mean([r["event_rate"] for r in rows]))
    tab[T]["auc"] = {k: float(np.mean([r["auc"][k] for r in rows])) for k in rows[0]["auc"]}
    tab[T]["wall"] = {k: float(np.mean([r["wall"][k] for r in rows])) for k in rows[0]["wall"]}

gate = {a: [s[0][a + "_train_ns"] for s in A["per_seed"]] for a in ("dj", "djl", "djs", "lindj")}
bm = lambda a: float(np.mean([tab[T][a]["ns"] for T in band]))
best_dj = {T: max(tab[T]["dj"]["ns"], tab[T]["djl"]["ns"]) for T in band}
ori = lambda x: max(x, 1 - x)
v = dict(
    band=band, learn_gate=gate, learn_ok={a: all(g >= 0.9 for g in gs) for a, gs in gate.items()},
    null_ok=all(tab[T]["null_max_dev"] <= 0.01 for T in tab),
    band_mean_ns={a: bm(a) for a in ARMS + (["lin1"] if L1 else [])},
    lin_win=all(tab[T]["lin"]["ns"] >= best_dj[T] - 0.01 for T in band) and bm("lin") >= bm("floor") + 0.05,
    lin_kill=any(tab[T]["lin"]["ns"] < best_dj[T] - 0.01 for T in band),
    lin_beats_ens3=bm("lin") > bm("ens3") + 0.02,
    cert_never_hurts=all(tab[T]["cert"]["s"] >= tab[T]["floor"]["s"] - 0.01 for T in tab),
    topo={T: {k: ori(x) for k, x in tab[T]["auc"].items()} for T in band},
)
v["topo_pass"] = all(max(v["topo"][T]["h0_longest"], v["topo"][T]["h0_shortest"]) >= 0.65 and
                     max(v["topo"][T]["h0_longest"], v["topo"][T]["h0_shortest"]) >= v["topo"][T]["spread"] - 0.02 for T in band)
if L1:
    v["lin1_survives"] = bm("lin1") >= bm("lin") - 0.02
out = dict(bar_sha256_floors=F["bar_sha256"], bar_sha256_arms=A["bar_sha256"], verdict=v, table=tab)
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
print(json.dumps(v, indent=1))
for T in tab:
    print(T, "bayes %.4f blind %.4f" % (tab[T]["bayes"], tab[T]["blind"]),
          " ".join("%s %.4f(%.3f)" % (a, tab[T][a]["s"], tab[T][a]["ns"]) for a in ARMS + (["lin1"] if L1 else [])),
          "abst %.2f" % tab[T]["cert_abstain"])
