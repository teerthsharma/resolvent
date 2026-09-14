"""ceqjepa/bed_headroom.py -- HOW MUCH CAUSAL SIGNAL IS IN THE BED AT ALL.

run_causal_test.py asks whether DCM-1's do(a) read beats the
ignore-the-intervention bar. A NULL there has two completely different causes
and the model card must not confuse them:

  (a) the operator does not represent the intervention, or
  (b) forcing one ply of a uniformly-random self-play game barely moves the
      outcome distribution, so NO predictor -- not even one handed the true
      interventional distribution -- could beat that bar on this bed.

This file measures (b) directly, with rollouts and no model anywhere.

THE ORACLE. On the SAME held-out positions run_causal_test scores
(build_intervention_dataset regenerates them bitwise from the same seed), it
estimates by brute force, R rollouts each under the identical uniform-random
policy:
    p_do   = P(outcome | do(forced candidate move))
    p_obs  = P(outcome | do(the move actually played))
    p_do2  = p_do again, from INDEPENDENT rollouts -- the noise floor
Both p_do and p_obs are add-one smoothed ((count+1)/(R+K)) so a class unseen
in R draws costs a finite number of nats rather than the eps clamp; the same
smoothing is applied to both arms, so the comparison is not tilted.

THE THREE NUMBERS THAT DECIDE WHETHER THE MAIN TEST IS RESOLVABLE:
  ORACLE HEADROOM = PPL(p_obs -> k_do) - PPL(p_do -> k_do), scored against the
    single realized forced outcome the eval bed carries. This is the LARGEST
    head-to-head advantage over the ignore-the-intervention bar that any
    predictor on this bed could show. If it is within its own paired bootstrap
    SE of zero, the main test cannot resolve anything and a model NULL says
    nothing about the architecture.
  TV(p_do, p_obs) -- how far the intervention actually moves the distribution.
  TV(p_do, p_do2) -- the SAME distribution estimated twice. The planted
    negative: whatever TV(p_do, p_obs) reads, this is what pure rollout noise
    reads, and only the excess is a real causal effect.
"""
from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np
import torch

import ceqjepa.causal_eval as ce
from ceqjepa.beds.chess import N_OUTCOMES, OUTCOME_NAMES, _outcome_onehot
from ceqjepa.beds.chess_do import build_intervention_dataset, _rollout_outcome

FROZEN = dict(eval_games=600, eval_seed=12345, eval_m=1, eval_R=1, max_plies=400,
              oracle_R=8, oracle_seed=999, n_boot=2000, boot_seed=7)


def rollouts(board, uci, rng, R, max_plies):
    b = board.copy(stack=False)
    b.push(chess.Move.from_uci(uci))
    acc = np.zeros(N_OUTCOMES, dtype=np.float64)
    for _ in range(R):
        acc += _rollout_outcome(b, rng, max_plies)
    return acc


def smooth(counts, R):
    return (counts + 1.0) / (R + N_OUTCOMES)   # add-one, identical on both arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        FROZEN.update(eval_games=25, oracle_R=3, n_boot=200)
    print("=== bed_headroom FROZEN ===")
    for k, v in FROZEN.items():
        print("  %s = %s" % (k, v))

    t0 = time.time()
    ev = build_intervention_dataset(n_games=FROZEN['eval_games'], seed=FROZEN['eval_seed'],
                                    m_candidates=FROZEN['eval_m'], R=FROZEN['eval_R'],
                                    max_plies=FROZEN['max_plies'])
    print("[bed] regenerated the SAME %d held-out positions in %.0fs"
          % (len(ev), time.time() - t0), flush=True)

    rng = random.Random(FROZEN['oracle_seed'])
    R, mp = FROZEN['oracle_R'], FROZEN['max_plies']
    P_do, P_obs, P_do2, k_do, k_obs = [], [], [], [], []
    t0 = time.time()
    for j, s in enumerate(ev):
        board = chess.Board(s.fen)
        P_do.append(smooth(rollouts(board, s.candidate_ucis[0], rng, R, mp), R))
        P_obs.append(smooth(rollouts(board, s.obs_uci, rng, R, mp), R))
        P_do2.append(smooth(rollouts(board, s.candidate_ucis[0], rng, R, mp), R))
        k_do.append(int(s.do_outcome_mean[0].argmax()))
        k_obs.append(int(s.obs_outcome.argmax()))
        if (j + 1) % 100 == 0:
            print("  oracle %d/%d (%.0fs)" % (j + 1, len(ev), time.time() - t0), flush=True)
    P_do = torch.tensor(np.stack(P_do))
    P_obs = torch.tensor(np.stack(P_obs))
    P_do2 = torch.tensor(np.stack(P_do2))
    k_do = torch.tensor(k_do)
    k_obs = torch.tensor(k_obs)

    tv_causal = float((P_do - P_obs).abs().sum(-1).mean() / 2)
    tv_noise = float((P_do - P_do2).abs().sum(-1).mean() / 2)
    print("\n[MEASURED] mean TV(p_do, p_obs)  = %.4f   <- forced move vs played move" % tv_causal)
    print("[MEASURED] mean TV(p_do, p_do2) = %.4f   <- SAME distribution, independent "
          "rollouts: the noise floor at R=%d" % (tv_noise, R))
    print("[MEASURED] excess over noise    = %+.4f" % (tv_causal - tv_noise))

    print("\n[MEASURED] chance level = %.4f over K=%d %s (1.00 perfect)"
          % (ce.chance_level(N_OUTCOMES), N_OUTCOMES, OUTCOME_NAMES))
    bed = dict(k_star_obs=k_do, k_star_do=k_do)   # both arms scored on the FORCED outcome
    kw = dict(n_boot=FROZEN['n_boot'], seed=FROZEN['boot_seed'])
    r = ce.causal_gap(lambda _b: P_obs, lambda _b: P_do, bed, label="ORACLE HEADROOM", **kw)
    print("    PPL_obs above = the ORACLE ignore-the-intervention bar p(outcome | played move); "
          "PPL_do = the oracle interventional distribution p(outcome | forced move).")
    print("    [HEADROOM] the best possible head-to-head advantage on this bed = %+.4f +- %.4f "
          "(%+.2f SE; NEGATIVE = knowing the intervention helps)"
          % (r['gap'], r['se_gap'], r['gap'] / r['se_gap'] if r['se_gap'] else float('nan')))
    # planted negative: the same oracle against ITSELF must show a gap of exactly 0.
    r0 = ce.causal_gap(lambda _b: P_do, lambda _b: P_do, bed, label="ORACLE vs ITSELF", **kw)
    assert r0['gap'] == 0.0 and r0['se_gap'] == 0.0, "paired bootstrap is not paired"
    print("    [PLANTED NEGATIVE] oracle scored against itself: gap=%+.6f se=%.6f (must be "
          "exactly 0/0, and is)" % (r0['gap'], r0['se_gap']))
    # and the independent-rollout copy: an oracle that DOES know the intervention but is
    # estimated from different draws. Its gap against p_do is pure estimator noise.
    r1 = ce.causal_gap(lambda _b: P_do2, lambda _b: P_do, bed, label="ORACLE vs ITS COPY", **kw)
    print("    [NOISE FLOOR] the same oracle re-estimated from independent rollouts: "
          "gap=%+.4f +- %.4f -- any headroom smaller than this is estimator noise, not signal."
          % (r1['gap'], r1['se_gap']))
    print("\n[MEASURED] forcing the candidate changed the REALIZED outcome in %.1f%% of "
          "positions (k_obs vs k_do, single draws each)"
          % (100 * float((k_obs != k_do).float().mean())))


if __name__ == '__main__':
    main()
