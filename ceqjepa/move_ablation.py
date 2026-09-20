"""ceqjepa/move_ablation.py -- DOES THE do-READ ACTUALLY USE THE FORCED MOVE?

run_causal_test.py's seed-0 trace shows the observational read q_alpha
DEGRADING on held-out data as training proceeds (PPL_obs 2.96 at step 500 ->
8.35 at step 4000, against chance 4.00) while the do-read q_do stays near 3.3.
A head-to-head win over the ignore-the-intervention bar under those conditions
has TWO possible causes:

  (a) the do-read carries the forced move into the committor -- the causal
      claim; or
  (b) the rank-1 clamp pulls the row toward build_operator's teleport target
      and so REGULARISES an over-confident operator, which would beat a
      degraded bar with no reference to the move at all.

This file separates them by the only test that can: score the SAME trained
model's do-read twice, once with each position's real forced move and once
with the moves PERMUTED across positions (same marginal distribution of moves,
same code path, the move-to-position pairing destroyed).

  |PPL_do(real moves) - PPL_do(permuted moves)| ~ 0  =>  cause (b). The do-read
      is not reading the move; the win is calibration, not causality.
  PPL_do(real) clearly better than PPL_do(permuted)  =>  cause (a).

The beds are rebuilt from the same seeds run_causal_test froze, so the seed-0
model here is the same model bitwise; they are cached to the scratchpad
because the 800-game interventional bed costs ~19 minutes to play out.
"""
from __future__ import annotations

import os
import pickle
import time

import torch

import ceqjepa.causal_eval as ce
from ceqjepa.beds.chess_do import build_intervention_dataset
from ceqjepa.train import ChessDoBed
from ceqjepa.run_causal_test import FROZEN, const, read_arms, train_one

CACHE = os.environ.get('CEQ_CACHE', '.')


def cached(tag, fn):
    p = os.path.join(CACHE, tag + '.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    t0 = time.time()
    v = fn()
    with open(p, 'wb') as f:
        pickle.dump(v, f)
    print("[bed] built %s in %.0fs" % (tag, time.time() - t0), flush=True)
    return v


def main():
    tr = cached('ma_train_800_s0', lambda: build_intervention_dataset(
        n_games=FROZEN['train_games'], seed=FROZEN['train_seed'],
        m_candidates=FROZEN['train_m'], R=FROZEN['train_R'], max_plies=FROZEN['max_plies']))
    ev = cached('ma_eval_600_s12345', lambda: build_intervention_dataset(
        n_games=FROZEN['eval_games'], seed=FROZEN['eval_seed'],
        m_candidates=FROZEN['eval_m'], R=FROZEN['eval_R'], max_plies=FROZEN['max_plies']))
    train_bed = ChessDoBed(tr, FROZEN['train_m'])
    eval_bed = ChessDoBed(ev, FROZEN['eval_m'])
    k_obs = eval_bed.q_star.argmax(-1)
    k_do = eval_bed.do_tgt[:, 0].argmax(-1)
    kw = dict(n_boot=FROZEN['n_boot'], seed=FROZEN['boot_seed'])

    seed = 0
    model, _ = train_one(train_bed, seed, FROZEN['steps'], FROZEN['batch_size'])
    arms = read_arms(model, eval_bed)
    ok = arms['ok']

    # the SAME model, the SAME code path, the move-to-position pairing destroyed.
    perm_bed = ChessDoBed(ev, FROZEN['eval_m'])
    g = torch.Generator().manual_seed(4242)
    perm = torch.randperm(perm_bed.moves.shape[0], generator=g)
    perm_bed.moves = perm_bed.moves[perm]
    assert not bool((perm_bed.moves == eval_bed.moves).all()), "permutation was a no-op"
    arms_p = read_arms(model, perm_bed)
    ok2 = ok & arms_p['ok']

    q_obs, q_do = arms['q_obs'][ok2], arms['q_do'][ok2]
    q_nc, q_perm = arms['q_noclamp'][ok2], arms_p['q_do'][ok2]
    print("[eval] refused: real moves %d, permuted moves %d, scored on the %d positions "
          "both arms accepted" % (int((~ok).sum()), int((~arms_p['ok']).sum()), int(ok2.sum())))
    bed = dict(k_star_obs=k_obs[ok2], k_star_do=k_do[ok2])
    bed_do = dict(k_star_obs=k_do[ok2], k_star_do=k_do[ok2])
    ce.causal_gap(const(q_obs), const(q_do), bed, label="MODEL real moves", **kw)
    ce.causal_gap(const(q_obs), const(q_perm), bed, label="MODEL PERMUTED moves", **kw)
    ce.ignore_intervention_gap(const(q_obs), bed, **kw)
    ce.causal_gap(const(q_obs), const(q_nc), bed, label="NO-CLAMP same-row", **kw)
    r = ce.causal_gap(const(q_perm), const(q_do), bed_do,
                      label="REAL-minus-PERMUTED", **kw)
    print("[MOVE ABLATION] PPL_do(real moves) - PPL_do(permuted moves) = %+.4f +- %.4f "
          "(%+.2f SE). NEGATIVE and outside its SE = the do-read is reading the move. "
          "Zero = the win over the bar is calibration, not causality."
          % (r['gap'], r['se_gap'], r['gap'] / r['se_gap'] if r['se_gap'] else float('nan')))
    print("[MOVE ABLATION] max|q_do(real) - q_do(permuted)| over the eval set = %.6e"
          % float((q_do - q_perm).abs().max()))
    for nm, q, kk in (("obs q_alpha", q_obs, k_obs[ok2]), ("do  q_do   ", q_do, k_do[ok2]),
                      ("no-clamp   ", q_nc, k_do[ok2])):
        rep = ce.calibration_report(q, kk)
        top = max((b for b in rep['bins'] if b['count']), key=lambda b: b['count'])
        print("  ECE(%s) = %.4f ; busiest bin [%.1f,%.1f) n=%d mean_pred=%.4f empirical=%.4f"
              % (nm, rep['ece'], top['lo'], top['hi'], top['count'], top['mean_pred'],
                 top['empirical_freq']))


if __name__ == '__main__':
    main()
