"""Wilson: the CHANCE FLOOR of in-sample resolution, measured by shuffle.

Any B-bin partition fitted in-sample buys free resolution. Null: permute the
bin labels against the realised outcomes, refit the bin rates, recompute RES.

L-REPRO: seed random.Random(0) for the bed (identical stream to
wil_bar_can_fire.py, same 600 games, same one-position-per-game draw),
np.random.default_rng(1) for the 1000 permutations, float64, CPU.
Convention identical to wil_bar_can_fire.murphy. command: python wil_res_null.py
"""
import json, random, sys
import numpy as np
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
from ceqjepa.beds.chess import generate_selfplay_game, _outcome_onehot

SEED, B, NPERM = 0, 8, 1000


def res_of(bins, Y):
    obar = Y.mean(0); r = 0.0
    for k in np.unique(bins):
        m = bins == k
        r += (m.mean()) * float(((Y[m].mean(0) - obar) ** 2).sum())
    return r


def run(n_games, max_plies):
    rng = random.Random(SEED)
    Y, plies = [], []
    for _ in range(n_games):
        g = generate_selfplay_game(rng, max_plies=max_plies)
        Y.append(_outcome_onehot(g.headers.get("Result", "*")))
        n_ply = sum(1 for _ in g.mainline_moves())
        plies.append(rng.randrange(max(n_ply, 1)))
    Y = np.asarray(Y, float); plies = np.asarray(plies)
    edges = np.quantile(plies, np.linspace(0, 1, B + 1)[1:-1])
    bins = np.digitize(plies, edges)
    obs = res_of(bins, Y)
    g = np.random.default_rng(1)
    null = np.array([res_of(bins, Y[g.permutation(len(Y))]) for _ in range(NPERM)])
    unc = float((Y.mean(0) * (1 - Y.mean(0))).sum())
    return dict(max_plies=max_plies, n_games=n_games, B=B, UNC=unc,
                RES_observed=obs, null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)),
                analytic_null_mean=(B - 1) / n_games * unc,
                ratio_obs_over_null_mean=obs / float(null.mean()),
                p_value=float((null >= obs).mean()),
                clears_null_p95=bool(obs > np.percentile(null, 95)))


if __name__ == "__main__":
    out = [run(600, 80), run(600, 400)]
    print(json.dumps(out, indent=2))
    with open(r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\wil_res_null.json", "w") as f:
        json.dump(out, f, indent=2)
