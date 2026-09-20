"""ceqjepa/run_causal_test.py -- THE CAUSAL TEST.

Trains DCM-1 on --bed chess_do, then scores it with ceqjepa.causal_eval on a
HELD-OUT interventional bed (different games, different seed) and asks the one
question the architecture exists to answer:

    does the operator's do(a) read predict the outcome of a FORCED move better
    than the observational read does, by more than the paired bootstrap SE?

WHAT THE ARMS ARE.
  obs arm : k_star_obs = the outcome the REAL game actually reached, exact.
            prediction = q_alpha, the alpha-mixed committor L_q supervises.
  do arm  : the eval bed is built with R=1 and m=1, so `do_outcome_mean[0]` IS
            a one-hot -- the realized outcome of ONE forced playout, a single
            honest draw from the true post-intervention distribution, and
            index-aligned with the obs arm (same position, same game).
            prediction = q(do a)[i_star], the row-i read L_do supervises.

THE FOUR PREDICTORS, SCORED THROUGH THE IDENTICAL causal_gap CODE PATH:
  MODEL do-read       q_do[i_star]     the operator's Sherman-Morrison read
  IGNORE-INTERVENTION q_alpha          THE BAR: predict q(do a) = q(obs)
  NO-CLAMP (same row) q_field[i_star]  the TIGHTER bar: same read POSITION,
                                       same operator, intervention removed.
                                       Beating IGNORE could be an artifact of
                                       reading at a row instead of at the mix;
                                       beating NO-CLAMP cannot -- the only
                                       difference is the rank-1 do edit.
  MARGINAL            training-set outcome frequency; has learned nothing.

HEAD-TO-HEAD. gap = PPL_do - PPL_obs, and PPL_obs is IDENTICAL across the rows
above (same predict_obs), so gap(MODEL) - gap(BAR) = PPL_do(MODEL) -
PPL_do(BAR). That difference needs its OWN paired bootstrap, not a subtraction
of two separately-estimated SEs, so it is obtained by handing causal_gap a bed
whose BOTH k_star fields are k_star_do and the two do-arm predictions as the
two arms. No new scoring code: same _logp, same paired _bootstrap.

BUDGET AND SEEDS ARE FROZEN BELOW, fixed before any number from this harness
was read.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import ceqjepa.causal_eval as ce
from ceqjepa.beds.chess import OUTCOME_NAMES
from ceqjepa.beds.chess_do import build_intervention_dataset
from ceqjepa.train import ChessDoBed, TinyCEQ, compute_loss, topo_blocks

# --- FROZEN: decided before any number from this harness was read. ----------
FROZEN = dict(
    train_games=800, train_seed=0, train_m=8, train_R=4, max_plies=400,
    eval_games=600, eval_seed=12345, eval_m=1, eval_R=1,
    # steps raised 1500 -> 4000 AFTER the --smoke crash test but BEFORE any real
    # number was produced, purely on wall-clock grounds: training is ~0.05 s/step
    # against a ~22 min bed build, so 1500 steps was leaving the machine idle and
    # an undertrained NULL would be uninformative. No causal number was seen.
    steps=4000, batch_size=32, lr=3e-4, n=16, d_enc=16, rank=12, g=0.9,
    lambda_do=1.0, lambda_z=0.0, lambda_topo=0.0,
    model_seeds=(0, 1, 2), n_boot=2000, boot_seed=7,
)


def _args(seed):
    return argparse.Namespace(lambda_z=FROZEN['lambda_z'], lambda_do=FROZEN['lambda_do'],
                              lambda_topo=FROZEN['lambda_topo'], topo_blocks='phase',
                              seed=seed)


def train_one(bed, seed, steps, batch_size, trace=None, verbose_every=250):
    torch.manual_seed(seed)
    model = TinyCEQ(n=FROZEN['n'], nA=bed.nA, d_enc=FROZEN['d_enc'], x_dim=bed.x_dim,
                    z_dim_state=6, g=FROZEN['g'], rank=FROZEN['rank'],
                    absorbing_idx=torch.arange(bed.nA))
    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN['lr'], weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    args = _args(seed)
    stats = dict(attempts=0, fails=0, last_error=None)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, x_nx, q_star, _, moves, do_tgt, do_mask = bed.batch_do(gen, batch_size)
        model.train()
        out = model(x)
        blocks = topo_blocks(args, q_star, 0)
        loss, l_q, l_z, l_do, l_topo = compute_loss(
            model, out, q_star, x_nx, moves, do_tgt, do_mask, blocks, args, stats)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % verbose_every == 0 or step == steps:
            print("  [seed %d] step %5d  L_q=%.4f L_do=%.4f p_spread=%.3e refuse=%d/%d (%.0fs)"
                  % (seed, step, l_q, l_do, float(out['P'].std(dim=0).max()),
                     stats['fails'], stats['attempts'], time.time() - t0), flush=True)
        # Held-out trace, RECORDED ONLY. Nothing selects on it -- no early stop,
        # no checkpoint picking. It exists so "did it overfit the 800 positions"
        # is answerable from the log instead of being a hole in the report.
        if trace is not None and (step % 500 == 0 or step == steps):
            trace(model, seed, step)
    return model, stats


@torch.no_grad()
def read_arms(model, eb):
    """Every prediction this test scores, from ONE forward pass over the whole
    held-out bed. Returns [N,K] tensors plus the per-item refusal mask."""
    model.eval()
    out = model(eb.x)
    stats = dict(attempts=0, fails=0, last_error=None)
    field, ok, i_star = model.do_read(out, eb.moves, stats, return_field=True)  # [N,m,n,K]
    b = torch.arange(eb.x.shape[0])
    return dict(
        q_obs=out['q_alpha'],                       # the alpha-mixed observational read
        q_do=field[b, 0, i_star, :],                # do(a) at the intervened row
        q_noclamp=out['q_field'][b, i_star, :],     # SAME row, no intervention
        ok=ok, i_star=i_star, stats=stats)


def build_beds():
    t0 = time.time()
    tr = build_intervention_dataset(n_games=FROZEN['train_games'], seed=FROZEN['train_seed'],
                                    m_candidates=FROZEN['train_m'], R=FROZEN['train_R'],
                                    max_plies=FROZEN['max_plies'])
    print("[bed] train: %d paired positions in %.0fs" % (len(tr), time.time() - t0), flush=True)
    t0 = time.time()
    ev = build_intervention_dataset(n_games=FROZEN['eval_games'], seed=FROZEN['eval_seed'],
                                    m_candidates=FROZEN['eval_m'], R=FROZEN['eval_R'],
                                    max_plies=FROZEN['max_plies'])
    print("[bed] eval : %d paired positions in %.0fs" % (len(ev), time.time() - t0), flush=True)
    train_bed = ChessDoBed(tr, FROZEN['train_m'])
    eval_bed = ChessDoBed(ev, FROZEN['eval_m'])
    # the eval bed's do-label must be a ONE-HOT (R=1), not an average: assert it,
    # because everything downstream reads its argmax as "what actually happened".
    assert eval_bed.do_tgt.shape[1] == 1
    s = eval_bed.do_tgt[:, 0]
    assert bool(((s == 0) | (s == 1)).all()), "eval do-label is not one-hot; R must be 1"
    forced_is_played = float(np.mean([x.candidate_ucis[0] == x.obs_uci for x in ev]))
    return train_bed, eval_bed, tr, ev, forced_is_played


def const(q):
    """A causal_gap arm that returns a fixed [N,K] block of predictions."""
    return lambda _bed: q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='tiny budget, crash test only')
    a = ap.parse_args()
    if a.smoke:
        FROZEN.update(train_games=25, eval_games=25, steps=20, model_seeds=(0,), n_boot=200)

    print("=== FROZEN BUDGET (fixed before any number from this harness was read) ===")
    for k, v in FROZEN.items():
        print("  %s = %s" % (k, v))

    train_bed, eval_bed, tr_samples, ev_samples, forced_is_played = build_beds()
    K = eval_bed.nA
    print("[bed] K = %d absorbing sets %s; chance_level = %.4f "
          "(1.00 perfect, %.2f = no information)"
          % (K, OUTCOME_NAMES, ce.chance_level(K), ce.chance_level(K)))
    k_obs = eval_bed.q_star.argmax(-1)
    k_do = eval_bed.do_tgt[:, 0].argmax(-1)
    print("[bed] eval obs outcome counts %s" % torch.bincount(k_obs, minlength=K).tolist())
    print("[bed] eval do  outcome counts %s" % torch.bincount(k_do, minlength=K).tolist())
    print("[bed] forced move == played move in %.1f%% of eval positions (a uniform draw "
          "from the legal moves, NOT filtered out)" % (100 * forced_is_played))
    print("[bed] forcing that move CHANGED the realized outcome in %.1f%% of eval positions"
          % (100 * float((k_obs != k_do).float().mean())))
    k_train = train_bed.q_star.argmax(-1)
    marg = ce.marginal_predictor(k_train, K)

    rows = []
    for seed in FROZEN['model_seeds']:
        print("\n=== TRAIN seed %d (%d steps, batch %d) ==="
              % (seed, FROZEN['steps'], FROZEN['batch_size']), flush=True)
        def trace(m, sd, st):
            a = read_arms(m, eval_bed)
            o = a['ok']
            print("    [trace s%d step %d] held-out PPL_obs(q_alpha)=%.4f  "
                  "PPL_do(q_do)=%.4f  PPL_do(q_alpha, THE BAR)=%.4f  refused=%d"
                  % (sd, st, ce.outcome_ppl(a['q_obs'][o], k_obs[o]),
                     ce.outcome_ppl(a['q_do'][o], k_do[o]),
                     ce.outcome_ppl(a['q_obs'][o], k_do[o]), int((~o).sum())), flush=True)

        model, tstats = train_one(train_bed, seed, FROZEN['steps'], FROZEN['batch_size'],
                                  trace=trace)
        arms = read_arms(model, eval_bed)
        ok = arms['ok']
        n_ref = int((~ok).sum())
        print("[eval] Sherman-Morrison refused %d/%d held-out positions (last: %s)"
              % (n_ref, ok.numel(), arms['stats']['last_error']))
        # drop refusals from EVERY arm identically, so the pairing survives
        q_obs, q_do, q_nc = arms['q_obs'][ok], arms['q_do'][ok], arms['q_noclamp'][ok]
        bed = dict(k_star_obs=k_obs[ok], k_star_do=k_do[ok])
        bed_do_only = dict(k_star_obs=k_do[ok], k_star_do=k_do[ok])  # head-to-head, do arm only
        kw = dict(n_boot=FROZEN['n_boot'], seed=FROZEN['boot_seed'])
        f_obs, f_do, f_nc = const(q_obs), const(q_do), const(q_nc)
        print("--- seed %d: the four predictors, identical scoring path ---" % seed)
        r_model = ce.causal_gap(f_obs, f_do, bed, label="MODEL do-read s%d" % seed, **kw)
        r_bar = ce.ignore_intervention_gap(f_obs, bed, **kw)
        r_nc = ce.causal_gap(f_obs, f_nc, bed, label="NO-CLAMP same-row s%d" % seed, **kw)
        r_marg = ce.causal_gap(marg, marg, bed, label="MARGINAL s%d" % seed, **kw)
        print("--- seed %d: HEAD-TO-HEAD (paired bootstrap on the DIFFERENCE) ---" % seed)
        h_bar = ce.causal_gap(f_obs, f_do, bed_do_only,
                              label="MODEL-minus-BAR s%d" % seed, **kw)
        h_nc = ce.causal_gap(f_nc, f_do, bed_do_only,
                             label="MODEL-minus-NOCLAMP s%d" % seed, **kw)
        verdict = ("BEATS the bar" if h_bar['gap'] < -abs(h_bar['se_gap'])
                   else "does NOT beat the bar")
        print("    [VERDICT seed %d] MODEL PPL_do minus BAR PPL_do = %+.4f +- %.4f "
              "(%+.2f SE; NEGATIVE = the do-read is better) -> %s"
              % (seed, h_bar['gap'], h_bar['se_gap'],
                 h_bar['gap'] / h_bar['se_gap'] if h_bar['se_gap'] else float('nan'), verdict))
        print("    [VERDICT seed %d] MODEL PPL_do minus NO-CLAMP PPL_do = %+.4f +- %.4f (%+.2f SE)"
              % (seed, h_nc['gap'], h_nc['se_gap'],
                 h_nc['gap'] / h_nc['se_gap'] if h_nc['se_gap'] else float('nan')))
        if seed == FROZEN['model_seeds'][0]:
            print("--- seed %d: RELIABILITY BUCKETS, BOTH ARMS (one-vs-rest, K=%d, "
                  "10 equal-width bins) ---" % (seed, K))
            for name, q, kk in (("obs arm  q_alpha", q_obs, k_obs[ok]),
                                ("do  arm  q_do   ", q_do, k_do[ok])):
                rep = ce.calibration_report(q, kk)
                print("  %s: ECE=%.4f over %d (item,class) points"
                      % (name, rep['ece'], rep['n_points']))
                for bb in rep['bins']:
                    if bb['count']:
                        print("    [%.1f,%.1f)  n=%6d  mean_pred=%.4f  empirical=%.4f"
                              % (bb['lo'], bb['hi'], bb['count'], bb['mean_pred'],
                                 bb['empirical_freq']))
        rows.append((seed, r_model, r_bar, r_nc, r_marg, h_bar, h_nc, n_ref))

    print("\n=== SUMMARY over %d seeds ===" % len(rows))
    for seed, r_model, r_bar, r_nc, r_marg, h_bar, h_nc, n_ref in rows:
        print("  seed %d: gap(MODEL)=%+.4f+-%.4f  gap(BAR)=%+.4f+-%.4f  "
              "gap(NO-CLAMP)=%+.4f+-%.4f  gap(MARGINAL)=%+.4f+-%.4f  "
              "MODEL-minus-BAR=%+.4f+-%.4f  refused=%d"
              % (seed, r_model['gap'], r_model['se_gap'], r_bar['gap'], r_bar['se_gap'],
                 r_nc['gap'], r_nc['se_gap'], r_marg['gap'], r_marg['se_gap'],
                 h_bar['gap'], h_bar['se_gap'], n_ref))
    wins = sum(1 for r in rows if r[5]['gap'] < -abs(r[5]['se_gap']))
    d = [r[5]['gap'] for r in rows]
    print("\n[ANSWER] the operator's do-read beat the IGNORE-THE-INTERVENTION bar by more "
          "than one paired bootstrap SE in %d/%d seeds; mean difference %+.4f "
          "(per-seed %s)." % (wins, len(rows), float(np.mean(d)), ["%+.4f" % x for x in d]))


if __name__ == '__main__':
    main()
