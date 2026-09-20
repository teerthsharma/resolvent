"""ceqjepa/positive_control.py -- CAN THIS PIPELINE DETECT AN INTERVENTION AT ALL?

bed_headroom.py measures how much causal signal the chess_do bed contains. If
that is ~0, run_causal_test.py's NULL is unattributable: it could mean the
operator cannot represent do(a), or it could mean nothing on that bed could.
This file removes the ambiguity from the other side, by PLANTING an effect and
checking the identical pipeline finds it.

THE PLANTED EFFECT. Same real chess positions, same ChessDoBed, same TinyCEQ,
same L_do, same causal_eval scoring path -- only the do-label is replaced:

    k_do(move) = (7 * from_square + 13 * to_square) mod 4

a DETERMINISTIC function of the forced move and of nothing else. The
observational label is the same function applied to the move actually played.
So:
  * the ignore-the-intervention bar predicts q(do a) = q(obs), and q(obs) is a
    function of the POSITION only -- it cannot see which move was forced, so it
    is stuck at chance no matter how well it is trained;
  * a do-read that genuinely carries the forced move into the committor can in
    principle reach PPL 1.0.

If the model beats the bar here it establishes that the operator's do(a) arm,
the Sherman-Morrison read, and the whole scoring harness DO transmit and detect
an intervention -- so a NULL on real chess is a fact about the chess bed's
causal signal, not a broken pipeline. If the model does NOT beat the bar even
here, the failure is in the architecture or the training, and the chess NULL
cannot be blamed on the bed.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import ceqjepa.causal_eval as ce
from ceqjepa.beds.chess_do import build_intervention_dataset
from ceqjepa.train import ChessDoBed
from ceqjepa.run_causal_test import FROZEN, const, read_arms, train_one

PC = dict(train_games=400, train_seed=0, train_m=8, eval_games=300, eval_seed=12345,
          eval_m=1, R=1, max_plies=400, steps=4000, batch_size=32,
          model_seeds=(0,), n_boot=2000, boot_seed=7)


PLANTED = 'mod4'


def planted(moves):
    """[..., 2] (from_square, to_square) -> planted outcome class in 0..3.

    'mod4' = (7*from + 13*to) mod 4: deterministic, but a HIGH-FREQUENCY function
      of the square indices -- adjacent squares get unrelated classes, so an
      8-dim square embedding read through one linear map has to memorise 64
      values per factor. Measured: the model never descended on it (L_do flat at
      ~2.0-2.4 over 4000 steps, PPL_do 5.06 against chance 4.00).
    'rank' = to_square // 16: which quarter of the board the move LANDS on.
      Equally deterministic and equally invisible to the observational arm, but
      SMOOTH in the square index -- four contiguous blocks of 16. This is the
      fair capacity test; mod4 is the adversarial one.
    """
    if PLANTED == 'rank':
        return moves[..., 1] // 16
    return (7 * moves[..., 0] + 13 * moves[..., 1]) % 4


def repaint(bed):
    """Overwrite the bed's do-labels (and the observational label) with the
    planted function of the move. Nothing else about the bed changes."""
    k = planted(bed.moves)                                     # [N, m]
    bed.do_tgt = torch.nn.functional.one_hot(k, 4).float()     # [N, m, 4]
    # the observational label is the planted class of the move actually played;
    # ChessDoBed does not keep obs_uci, so it is carried in alongside.
    bed.q_star = torch.nn.functional.one_hot(bed.obs_k, 4).float()
    return bed


def build(n_games, seed, m):
    s = build_intervention_dataset(n_games=n_games, seed=seed, m_candidates=m,
                                   R=PC['R'], max_plies=PC['max_plies'])
    bed = ChessDoBed(s, m)
    import chess
    obs = torch.tensor([[chess.Move.from_uci(x.obs_uci).from_square,
                         chess.Move.from_uci(x.obs_uci).to_square] for x in s])
    bed.obs_k = planted(obs)
    return repaint(bed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--planted', choices=['mod4', 'rank'], default='mod4')
    a = ap.parse_args()
    globals()['PLANTED'] = a.planted
    if a.smoke:
        PC.update(train_games=30, eval_games=30, steps=50, n_boot=200)
    print("=== POSITIVE CONTROL: a PLANTED intervention, --planted %s ===" % a.planted)
    for k, v in PC.items():
        print("  %s = %s" % (k, v))
    FROZEN['steps'] = PC['steps']

    t0 = time.time()
    train_bed = build(PC['train_games'], PC['train_seed'], PC['train_m'])
    eval_bed = build(PC['eval_games'], PC['eval_seed'], PC['eval_m'])
    print("[bed] %d train / %d eval positions in %.0fs"
          % (len(train_bed.x), len(eval_bed.x), time.time() - t0), flush=True)
    K = 4
    k_obs = eval_bed.q_star.argmax(-1)
    k_do = eval_bed.do_tgt[:, 0].argmax(-1)
    print("[bed] chance_level = %.4f; planted do-label counts %s"
          % (ce.chance_level(K), torch.bincount(k_do, minlength=K).tolist()))

    for seed in PC['model_seeds']:
        model, _ = train_one(train_bed, seed, PC['steps'], PC['batch_size'])
        arms = read_arms(model, eval_bed)
        ok = arms['ok']
        print("[eval] refused %d/%d" % (int((~ok).sum()), ok.numel()))
        q_obs, q_do, q_nc = arms['q_obs'][ok], arms['q_do'][ok], arms['q_noclamp'][ok]
        bed = dict(k_star_obs=k_obs[ok], k_star_do=k_do[ok])
        bed_do = dict(k_star_obs=k_do[ok], k_star_do=k_do[ok])
        kw = dict(n_boot=PC['n_boot'], seed=PC['boot_seed'])
        ce.causal_gap(const(q_obs), const(q_do), bed, label="MODEL do-read s%d" % seed, **kw)
        ce.ignore_intervention_gap(const(q_obs), bed, **kw)
        ce.causal_gap(const(q_obs), const(q_nc), bed, label="NO-CLAMP same-row s%d" % seed, **kw)
        h = ce.causal_gap(const(q_obs), const(q_do), bed_do,
                          label="MODEL-minus-BAR s%d" % seed, **kw)
        hn = ce.causal_gap(const(q_nc), const(q_do), bed_do,
                           label="MODEL-minus-NOCLAMP s%d" % seed, **kw)
        print("[POSITIVE CONTROL seed %d] MODEL PPL_do minus BAR PPL_do = %+.4f +- %.4f "
              "(NEGATIVE = the planted intervention was detected)" % (seed, h['gap'], h['se_gap']))
        print("[POSITIVE CONTROL seed %d] MODEL PPL_do minus NO-CLAMP PPL_do = %+.4f +- %.4f"
              % (seed, hn['gap'], hn['se_gap']))


if __name__ == '__main__':
    main()
