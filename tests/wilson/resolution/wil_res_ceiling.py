"""Wilson: the RESOLUTION CEILING of ceqjepa/beds/chess.py, measured.

L-REPRO
  seed         : random.Random(SEED), SEED=0, one stream, consumed in order
  generator    : ceqjepa.beds.chess.generate_selfplay_game (uniform random
                 legal move, python-chess 1.11.2) -- imported, not reimplemented
  outcome map  : ceqjepa.beds.chess._outcome_onehot  ("1-0"->0,"1/2-1/2"->1,
                 "0-1"->2, anything else incl. "*"/ply-cap ->3 SINK)
  convention   : multiclass Brier BS = mean_i sum_c (p_ic - y_ic)^2
                 UNC = sum_c pbar_c (1 - pbar_c)
                 RES_ceiling = sum_c Var_pos(p_true_c), the resolution of the
                 ORACLE forecaster q(pos) = P(Y | pos, policy). No forecaster
                 on this bed can exceed it.
  estimator    : R rollouts per position give p_hat with
                 E[Var_pos(p_hat)] = Var_pos(p_true) + E[p(1-p)]/R, and
                 E[p_hat(1-p_hat)] = (1-1/R) E[p(1-p)], so
                 Var_hat(p_true) = Var_pos(p_hat) - mean(p_hat(1-p_hat))/(R-1).
  dtype        : float64 (numpy default), CPU only, no torch, no grad
  iterations   : N_GAMES games, one position per game (chess_do.py's own
                 sampling rule), R rollouts per position
  command      : python wil_res_ceiling.py
"""
import json, random, sys
import numpy as np
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
import chess
from ceqjepa.beds.chess import generate_selfplay_game, _outcome_onehot, OUTCOME_NAMES

SEED = 0
R = 16


def rollout_result(board, rng, max_plies):
    b = board.copy()
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        b.push(rng.choice(list(b.legal_moves)))
    if b.ply() >= max_plies and not b.is_game_over(claim_draw=True):
        return "*"                      # ply cap -> SINK, exactly the bed's rule
    return b.result(claim_draw=True)


def boards_of(game):
    b = game.board()
    out = [b.copy()]
    for mv in game.mainline_moves():
        b.push(mv)
        out.append(b.copy())
    return out[:-1]                      # positions BEFORE a move, as chess.py rows


def measure(n_games, max_plies, tag):
    rng = random.Random(SEED)
    marg, phat, plies = [], [], []
    for _ in range(n_games):
        g = generate_selfplay_game(rng, max_plies=max_plies)
        marg.append(_outcome_onehot(g.headers.get("Result", "*")))
        bs = boards_of(g)
        b = bs[rng.randrange(len(bs))]   # one position per game, uniform over plies
        plies.append(b.ply())
        acc = np.zeros(4)
        for _r in range(R):
            acc += _outcome_onehot(rollout_result(b, rng, max_plies))
        phat.append(acc / R)
    marg = np.asarray(marg, dtype=np.float64)
    phat = np.asarray(phat, dtype=np.float64)

    pbar = marg.mean(0)
    unc = float((pbar * (1 - pbar)).sum())
    var_phat = phat.var(0, ddof=1)
    noise = (phat * (1 - phat)).mean(0) / (R - 1)
    res_c = var_phat - noise
    res = float(res_c.sum())

    # bootstrap over positions, 2000 resamples, same rng stream
    bs_rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        idx = bs_rng.integers(0, len(phat), len(phat))
        s = phat[idx]
        boot.append(float((s.var(0, ddof=1) - (s * (1 - s)).mean(0) / (R - 1)).sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return dict(tag=tag, n_games=n_games, max_plies=max_plies, R=R,
                marginal={k: float(v) for k, v in zip(OUTCOME_NAMES, pbar)},
                uncertainty=unc, res_ceiling=res,
                res_ceiling_ci95=[float(lo), float(hi)],
                res_per_class={k: float(v) for k, v in zip(OUTCOME_NAMES, res_c)},
                res_over_unc=(res / unc if unc > 0 else float("nan")),
                mean_sampled_ply=float(np.mean(plies)))


if __name__ == "__main__":
    out = [measure(300, 80, "ship_default_cap80"),
           measure(150, 400, "control_cap400")]
    for o in out:
        print(json.dumps(o, indent=2))
    with open(r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\wil_res_ceiling.json", "w") as f:
        json.dump(out, f, indent=2)
