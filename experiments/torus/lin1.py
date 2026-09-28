# python lin1.py -> lin1.json: LIN1 (one-VJP chance rule) on the same eval batches as run.py (Amendment A2)
import json, time
from multiprocessing import Pool
import numpy as np
import rj
from run import HERE, LEADS, NEV, SEEDS, rng_for


def one(seed):
    out = []
    for T in LEADS:
        ev = rj.make_batch(rng_for(seed, T, "eval"), NEV, T)
        t0 = time.perf_counter(); pk = rj._tb(rj.lin1_prob(ev), ev["d"]); dt = time.perf_counter() - t0
        pb = rj.pick_bayes(ev); rk = rj.rank01(ev["d"])[np.arange(NEV), pb]
        bounded = np.where(rk <= 0.4, pb, rj.pick_floor(ev))    # best a 2eps = 0.4 bounded reorder can reach
        out.append(dict(T=T, lin1=float(rj.score(ev, pk)), bayes=float(rj.score(ev, pb)), lin1_s=dt,
                        bayes_outside_eps=float((rk > 0.4).mean()), bounded_ceiling=float(rj.score(ev, bounded))))
    return out


if __name__ == "__main__":
    with Pool(3) as p:
        runs = p.map(one, SEEDS)
    (HERE / "lin1.json").write_text(json.dumps(runs, indent=1))
    print(runs)
