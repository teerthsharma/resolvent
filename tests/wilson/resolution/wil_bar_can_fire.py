"""Wilson: does the proposed bar FIRE on degenerate arms? Run it and see.

L-REPRO
  seed        : random.Random(0) for the bed; np.random.default_rng(0) for the bootstrap
  generator   : ceqjepa.beds.chess.generate_selfplay_game (uniform random legal move)
  convention  : multiclass Brier BS = mean_i sum_c (p_ic - y_ic)^2
                REL = sum_b (n_b/N) sum_c (p_bc - obar_bc)^2
                RES = sum_b (n_b/N) sum_c (obar_bc - obar_c)^2
                UNC = sum_c obar_c (1 - obar_c)
                partition = the arm's own distinct forecast vectors (exact, no re-binning)
  arms        : A0 constant marginal predictor (RES == 0 by construction)
                A1 ply-bucket base-rate predictor (architecture-free, but discriminating)
                both are recalibrator-invariant by construction: each arm's forecast
                inside a bin already EQUALS that bin's realised rate on its own split,
                i.e. REL is at its floor before any temperature/isotonic step.
  dtype       : float64, CPU, no torch
  command     : python wil_bar_can_fire.py
"""
import json, random, sys
import numpy as np
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
from ceqjepa.beds.chess import generate_selfplay_game, _outcome_onehot, OUTCOME_NAMES

SEED = 0


def murphy(p, y):
    """p [N,4] forecasts, y [N,4] realised one-hot. Returns REL, RES, UNC, BS_raw, residual."""
    keys = [tuple(np.round(r, 12)) for r in p]
    obar = y.mean(0)
    rel = res = 0.0
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        nb = m.sum() / len(y)
        ob = y[m].mean(0)
        pb = np.asarray(k)
        rel += nb * float(((pb - ob) ** 2).sum())
        res += nb * float(((ob - obar) ** 2).sum())
    unc = float((obar * (1 - obar)).sum())
    bs = float(((p - y) ** 2).sum(1).mean())
    return dict(REL=rel, RES=res, UNC=unc, BS_raw=bs,
                identity_residual=bs - (rel - res + unc), n_bins=len(set(keys)))


def run(n_games, max_plies, n_buckets=8):
    rng = random.Random(SEED)
    Y, plies = [], []
    for _ in range(n_games):
        g = generate_selfplay_game(rng, max_plies=max_plies)
        oh = _outcome_onehot(g.headers.get("Result", "*"))
        n_ply = sum(1 for _ in g.mainline_moves())
        j = rng.randrange(max(n_ply, 1))          # one position per game
        Y.append(oh); plies.append(j)
    Y = np.asarray(Y, dtype=np.float64); plies = np.asarray(plies)

    A0 = np.tile(Y.mean(0), (len(Y), 1))          # constant marginal arm
    edges = np.quantile(plies, np.linspace(0, 1, n_buckets + 1)[1:-1])
    b = np.digitize(plies, edges)
    A1 = np.zeros_like(Y)
    for k in np.unique(b):
        A1[b == k] = Y[b == k].mean(0)            # in-sample bucket base rate

    m0, m1 = murphy(A0, Y), murphy(A1, Y)
    gap = abs(m1["RES"] - m0["RES"])
    bar_ground = 0.05 * min(m1["RES"], m0["RES"])       # the bar as written in the ground file
    return dict(max_plies=max_plies, n_games=n_games,
                marginal={k: float(v) for k, v in zip(OUTCOME_NAMES, Y.mean(0))},
                arm_constant=m0, arm_plybucket=m1,
                RES_gap=gap, bar_as_written=bar_ground,
                bar_as_written_fires=bool(gap > bar_ground))


if __name__ == "__main__":
    out = [run(600, 80), run(600, 400)]
    print(json.dumps(out, indent=2))
    with open(r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\wil_bar_can_fire.json", "w") as f:
        json.dump(out, f, indent=2)
