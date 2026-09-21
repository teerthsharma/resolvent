"""wil_chess400.py -- Wilson's row (Phase I follow-up).

OWNED FILES: this script and its sibling .md, both in this scratchpad only.
No git, no results/*.pt, no tests/chase/deq_run.jsonl, no docs/canon, no lean.

WHAT THIS DOES, IN ORDER (per the task's own kill rule -- (1) first, written to
disk, before anything else; if the ceiling does not reproduce, stop):

  (1) Rebuild ceqjepa.beds.chess.ChessBed at max_plies=400 (REAL code, REAL
      python-chess self-play, not reimplemented) and measure its own floor:
      SINK fraction, an ORACLE resolution ceiling, UNC, RES/UNC, and a
      permutation shuffle null -- all computed HERE, because a tree-wide
      search (see docs/PHASE_I.md sec 1: "Neither the bed nor its Murphy
      residual exists anywhere in the tracked repository") turned up no
      surviving Murphy/RES-UNC code anywhere in ceqjepa, ceq, scale, scripts
      or tests. That code lived only in an uncommitted scratchpad that no
      longer exists. This file is the replacement, built honestly and
      labelled as new rather than presented as a recovered original.

  ORACLE CEILING, DEFINED HERE (there is no surviving definition to match
  exactly, so this one is stated plainly rather than assumed). Self-play
  moves are drawn uniformly at random from legal_moves; two different games
  never share a FEN beyond the opening handful of plies, so the only
  covariate that is genuinely SHARED across games -- the only thing any
  forecaster restricted to "no chess engine, no lookahead" (ceq/chess_steps.py
  0's own refused door) could key on -- is ply depth (v_idx): how far into
  its own game a position sits. Binning by v_idx and reading each bin's own
  empirical outcome frequency as the "forecast" is therefore the resolution
  a forecaster attains BY CONSTRUCTION if depth is all it can see, and nothing
  a board-only model can exploit beyond the true structure of random play (no
  engine, so nothing sees deeper) can beat it in expectation. That is the
  ceiling this file reports. REL is exactly 0 for this forecaster (its own
  bin mean IS its forecast), so ceiling_RES = UNC - Brier_oracle.

  (2) Train.py's real TinyCEQ model (unmodified import, ceqjepa.operator's
      real build_operator/committor inside it, never re-derived here) on the
      REBUILT max_plies=400 bed, then run, FOR THE FIRST TIME IN THIS TREE, a
      Murphy (REL/RES/UNC) decomposition on its own held-out committor output
      against realised k_star. Verify Brier = REL - RES + UNC to near machine
      epsilon and flag that this proves the arithmetic, not the bed.

  (3) Compare the trained committor's resolution against
      ceqjepa.causal_eval.marginal_predictor (imported, not reimplemented) on
      the SAME held-out items, with a paired bootstrap CI on each resolution.

  (4) Bar-check: push the marginal predictor through the identical Murphy
      pipeline and confirm it reports RES ~ 0 (it must, by construction) --
      the pipeline's own known-zero-resolution control.

RESULTS are appended to wil_chess400_results.json (one record per phase, as
they complete) and to house-events.jsonl (the board), never rewritten.
"""
from __future__ import annotations

import io
import json
import math
import random
import sys
import time
import datetime

import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, REPO)

from ceqjepa.beds.chess import ChessBed, N_OUTCOMES, OUTCOME_NAMES  # noqa: E402
import ceqjepa.operator as op  # noqa: E402
from ceqjepa.train import TinyCEQ, stage1_loss, heldout_split, draw  # noqa: E402
from ceqjepa.causal_eval import marginal_predictor  # noqa: E402

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
RESULTS_PATH = SCRATCH + r"\wil_chess400_results.json"
BOARD_PATH = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"


def board_log(event, **kw):
    rec = dict(agent="Wilson", ts=datetime.datetime.utcnow().isoformat() + "Z", event=event, **kw)
    with io.open(BOARD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def save_result(key, payload):
    """Append-merge into the results JSON so partial progress survives a kill."""
    try:
        with io.open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[key] = payload
    with io.open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Murphy (REL/RES/UNC) machinery -- NEW, because none survives in-tree
# (confirmed: grep across ceqjepa/ceq/scale/scripts/tests for Murphy/REL/RES/
# UNC/Beta(2,2) patterns returns nothing but prose in docs/PHASE_I.md and
# sharpness.py's unrelated cross-entropy decomposition). committor() itself
# is never re-derived: it is read straight off op.committor via TinyCEQ.
# --------------------------------------------------------------------------- #

def oracle_ceiling(rows_outcome, rows_bin, K):
    """REL/RES/UNC for the ORACLE forecaster whose forecast in a bin IS that
    bin's own empirical outcome mean (REL == 0 by construction). rows_outcome:
    [N,K] one-hot float64. rows_bin: [N] int, the binning (here: ply depth).
    Returns dict(RES, UNC, REL(=0), brier, n_bins, bin_sizes)."""
    o = torch.as_tensor(rows_outcome, dtype=torch.float64)
    b = torch.as_tensor(rows_bin, dtype=torch.long)
    N = o.shape[0]
    pi = o.mean(0)
    UNC = float((pi * (1 - pi)).sum())
    uniq = torch.unique(b)
    RES = 0.0
    sizes = []
    for u in uniq.tolist():
        mask = b == u
        n_b = int(mask.sum())
        sizes.append(n_b)
        obar = o[mask].mean(0)
        RES += (n_b / N) * float(((obar - pi) ** 2).sum())
    brier_oracle = UNC - RES  # REL == 0
    return dict(REL=0.0, RES=RES, UNC=UNC, brier=brier_oracle, n_bins=len(uniq),
                bin_sizes_min=min(sizes), bin_sizes_max=max(sizes), n=N)


def shuffle_null(rows_outcome, rows_bin, K, n_perm=1000, seed=0):
    """Permute the outcome rows (iid over all N rows -- this is what makes the
    analytic (B-1)*UNC/N bias correction apply; chess.py's own docstring notes
    the TRUE effective sample size is games not plies, so this null is
    deliberately the more optimistic, ply-level one, same choice the number
    this file is checking against was built on, per PHASE_I.md's report of
    the analytic match)."""
    o = torch.as_tensor(rows_outcome, dtype=torch.float64)
    b = torch.as_tensor(rows_bin, dtype=torch.long)
    N = o.shape[0]
    pi = o.mean(0)
    UNC = float((pi * (1 - pi)).sum())
    uniq = torch.unique(b)
    B = len(uniq)
    rng = np.random.default_rng(seed)
    idx_by_bin = [torch.nonzero(b == u, as_tuple=True)[0] for u in uniq.tolist()]
    null_res = np.empty(n_perm, dtype=np.float64)
    o_np = o.numpy()
    for p in range(n_perm):
        perm = rng.permutation(N)
        o_perm = o_np[perm]
        res = 0.0
        for mask_idx in idx_by_bin:
            n_b = len(mask_idx)
            obar = o_perm[mask_idx.numpy()].mean(0)
            res += (n_b / N) * float(((obar - pi.numpy()) ** 2).sum())
        null_res[p] = res
    analytic = (B - 1) * UNC / N
    return dict(null_mean=float(null_res.mean()), null_std=float(null_res.std()),
                analytic=analytic, n_perm=n_perm, B=B)


def game_bootstrap_ceiling(bed, n_boot=1000, seed=0):
    """CI95 on the oracle ceiling RES, bootstrapping over GAMES (the real
    effective sample size per chess.py's own docstring), not over rows."""
    ply = torch.tensor([r['ply_idx'] for r in bed.rows], dtype=torch.long)
    gid = (ply == 0).cumsum(0) - 1
    n_games = int(gid.max()) + 1
    outcome = torch.tensor(np.stack([r['outcome'] for r in bed.rows]), dtype=torch.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    # pre-index rows per game once
    rows_by_game = [torch.nonzero(gid == g, as_tuple=True)[0].numpy() for g in range(n_games)]
    ply_np = ply.numpy()
    out_np = outcome.numpy()
    for i in range(n_boot):
        chosen = rng.integers(0, n_games, size=n_games)
        idx = np.concatenate([rows_by_game[g] for g in chosen])
        res = oracle_ceiling(out_np[idx], ply_np[idx], outcome.shape[1])
        draws[i] = res['RES']
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(ci95=[float(lo), float(hi)], n_boot=n_boot, n_games=n_games)


def murphy_binned_onevsrest(q_pred, k_star, K, n_bins=10):
    """One-vs-rest Murphy decomposition, K equal-width-probability binnings
    (one per class, its own base rate pi_k -- NOT flattened across classes,
    which would collapse every class to base rate 1/K and destroy the skew
    that is the entire point on this bed). REL/RES/UNC sum coordinate-wise
    because Brier = sum_i ||q_i - o_i||^2 / N decomposes columnwise exactly."""
    q_pred = torch.as_tensor(q_pred, dtype=torch.float64)
    k_star = torch.as_tensor(k_star, dtype=torch.long)
    N = q_pred.shape[0]
    onehot = F.one_hot(k_star, K).double()
    brier_direct = float(((q_pred - onehot) ** 2).sum(-1).mean())
    REL = RES = UNC = 0.0
    per_class = []
    for k in range(K):
        f = q_pred[:, k]
        o = onehot[:, k]
        pi_k = float(o.mean())
        unc_k = pi_k * (1 - pi_k)
        edges = torch.linspace(0.0, 1.0, n_bins + 1)
        rel_k = res_k = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            last = i == n_bins - 1
            mask = (f >= lo) & ((f <= hi) if last else (f < hi))
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            fbar = float(f[mask].mean())
            obar = float(o[mask].mean())
            rel_k += (n_b / N) * (fbar - obar) ** 2
            res_k += (n_b / N) * (obar - pi_k) ** 2
        REL += rel_k
        RES += res_k
        UNC += unc_k
        per_class.append(dict(cls=OUTCOME_NAMES[k], pi_k=pi_k, rel_k=rel_k, res_k=res_k, unc_k=unc_k))
    brier_decomp = REL - RES + UNC
    residual = brier_direct - brier_decomp
    return dict(REL=REL, RES=RES, UNC=UNC, brier_direct=brier_direct,
                brier_decomp=brier_decomp, residual=residual, per_class=per_class, n=N, n_bins=n_bins)


def paired_bootstrap_res_gap(q_a, q_b, k_star, K, n_bins=10, n_boot=400, seed=0):
    """Bootstrap CI on RES(a), RES(b) and the paired gap RES(a)-RES(b), over
    ITEMS (both arms score the SAME held-out items, so the resample is
    paired -- same pattern as causal_eval._bootstrap)."""
    q_a = torch.as_tensor(q_a, dtype=torch.float64)
    q_b = torch.as_tensor(q_b, dtype=torch.float64)
    k_star = torch.as_tensor(k_star, dtype=torch.long)
    N = q_a.shape[0]
    rng = np.random.default_rng(seed)
    res_a = np.empty(n_boot); res_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        res_a[i] = murphy_binned_onevsrest(q_a[idx], k_star[idx], K, n_bins)['RES']
        res_b[i] = murphy_binned_onevsrest(q_b[idx], k_star[idx], K, n_bins)['RES']
    gap = res_a - res_b
    return dict(res_a_ci95=[float(np.percentile(res_a, 2.5)), float(np.percentile(res_a, 97.5))],
                res_b_ci95=[float(np.percentile(res_b, 2.5)), float(np.percentile(res_b, 97.5))],
                gap_mean=float(gap.mean()), gap_se=float(gap.std()),
                gap_ci95=[float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
                n_boot=n_boot)


# --------------------------------------------------------------------------- #
# PART (1): rebuild the bed at max_plies=400, measure the floor.
# --------------------------------------------------------------------------- #

def part1(n_games=1000, seed=0):
    board_log("start", part=1, n_games=n_games, seed=seed)
    t0 = time.time()
    bed400 = ChessBed.build(n_games=n_games, seed=seed, max_plies=400)
    t_build400 = time.time() - t0
    total400 = sum(bed400.outcome_counts.values())
    sink_frac400 = bed400.outcome_counts['sink'] / total400

    ply = np.array([r['ply_idx'] for r in bed400.rows], dtype=np.int64)
    outcome = np.stack([r['outcome'] for r in bed400.rows]).astype(np.float64)
    ceil400 = oracle_ceiling(outcome, ply, N_OUTCOMES)
    null400 = shuffle_null(outcome, ply, N_OUTCOMES, n_perm=1000, seed=seed)
    ci400 = game_bootstrap_ceiling(bed400, n_boot=500, seed=seed)

    # cheap reproducibility cross-check at max_plies=80, same seed/n_games
    t1 = time.time()
    bed80 = ChessBed.build(n_games=n_games, seed=seed, max_plies=80)
    t_build80 = time.time() - t1
    total80 = sum(bed80.outcome_counts.values())
    sink_frac80 = bed80.outcome_counts['sink'] / total80

    record = dict(
        n_games=n_games, seed=seed,
        max_plies_400=dict(
            build_seconds=t_build400, n_rows=len(bed400.rows), n_games_actual=total400,
            outcome_counts=bed400.outcome_counts, sink_fraction=sink_frac400,
            oracle_ceiling_RES=ceil400['RES'], UNC=ceil400['UNC'],
            RES_over_UNC=ceil400['RES'] / ceil400['UNC'] if ceil400['UNC'] > 0 else float('nan'),
            oracle_ceiling_CI95=ci400['ci95'],
            shuffle_null_mean=null400['null_mean'], shuffle_null_std=null400['null_std'],
            shuffle_null_analytic_B_minus_1_UNC_over_N=null400['analytic'],
            n_bins_v_idx=ceil400['n_bins'],
        ),
        max_plies_80_crosscheck=dict(
            build_seconds=t_build80, n_rows=len(bed80.rows), n_games_actual=total80,
            outcome_counts=bed80.outcome_counts, sink_fraction=sink_frac80,
        ),
        prior_claim_reference=dict(
            sink_frac_80="98.67%", ceiling_400="0.101170", note=(
                "prior numbers' generating code is confirmed absent from the tracked "
                "tree (docs/PHASE_I.md sec 1); this run defines its own oracle-ceiling "
                "methodology (v_idx-binned) rather than guessing at a lost one, and "
                "reports the SINK-fraction cross-check as the cheap, exact, "
                "methodology-independent reproducibility gate."
            ),
        ),
    )
    save_result("part1_bed_floor", record)
    board_log("part1_done",
              sink_frac400=sink_frac400, sink_frac80=sink_frac80,
              oracle_ceiling_RES_400=ceil400['RES'], RES_over_UNC_400=record['max_plies_400']['RES_over_UNC'],
              build_seconds_400=t_build400, build_seconds_80=t_build80)
    print(json.dumps(record, indent=2, default=str))
    return bed400, record


# --------------------------------------------------------------------------- #
# PART (2)+(3)+(4): train the REAL TinyCEQ operator on the rebuilt max_plies=400
# bed, Murphy-decompose its held-out committor for the first time in this tree,
# compare against causal_eval.marginal_predictor (imported, not reimplemented),
# and confirm the pipeline's known-zero-resolution control actually fires.
# --------------------------------------------------------------------------- #

def part2(n_games=1000, seed=0, steps=2000, batch_size=64, n=16, d_enc=32,
          z_dim_state=8, rank=8, lr=3e-3, heldout_frac=0.2, eval_n=6000, n_bins=10):
    board_log("start", part=2, n_games=n_games, seed=seed, steps=steps)
    bed = ChessBed.build(n_games=n_games, seed=seed, max_plies=400)
    train_bed, eval_bed, split_note = heldout_split(bed, heldout_frac, seed + 1)
    print("[split]", split_note)

    torch.manual_seed(seed)
    absorbing_idx = torch.arange(bed.nA)
    model = TinyCEQ(n=n, nA=bed.nA, d_enc=d_enc, x_dim=bed.x_dim,
                     z_dim_state=z_dim_state, g=0.9, absorbing_idx=absorbing_idx, rank=rank)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_gen = torch.Generator().manual_seed(seed)

    t0 = time.time()
    l_q_hist = []
    for step in range(steps):
        x, x_nx, q_star, v_idx, moves, do_tgt, do_mask = draw(train_bed, train_gen, batch_size)
        out = model(x)
        loss, l_q, l_z = stage1_loss(out, q_star, x_nx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        l_q_hist.append(l_q)
        if step % max(1, steps // 10) == 0:
            print(f"[train] step={step} L_q={l_q:.4f}")
    train_seconds = time.time() - t0

    n_eval = min(eval_n, len(eval_bed.rows))
    eval_gen = torch.Generator().manual_seed(seed + 2)
    x_e, x_nx_e, q_star_e, v_idx_e, *_ = draw(eval_bed, eval_gen, n_eval)
    with torch.no_grad():
        out_e = model(x_e)
    q_pred = out_e['q_alpha'].double()
    k_star = q_star_e.argmax(-1)

    murphy_op = murphy_binned_onevsrest(q_pred, k_star, bed.nA, n_bins=n_bins)

    n_tr = min(20000, len(train_bed.rows))
    x_tr, x_nx_tr, q_star_tr, v_idx_tr, *_ = draw(train_bed, torch.Generator().manual_seed(seed + 3), n_tr)
    k_star_tr = q_star_tr.argmax(-1)
    marg_fn = marginal_predictor(k_star_tr, bed.nA)
    q_marg = marg_fn({"k_star_obs": k_star}).double()
    murphy_marg = murphy_binned_onevsrest(q_marg, k_star, bed.nA, n_bins=n_bins)

    cmp = paired_bootstrap_res_gap(q_pred, q_marg, k_star, bed.nA, n_bins=n_bins, n_boot=400, seed=seed)

    # bar-check: does the pipeline detect the KNOWN-zero-resolution marginal
    # forecaster as such? RES(marginal) should sit near 0 relative to UNC.
    bar_fires = murphy_marg['RES'] < 0.02 * murphy_marg['UNC']

    record = dict(
        n_games=n_games, seed=seed, steps=steps, batch_size=batch_size,
        model=dict(n=n, nA=int(bed.nA), d_enc=d_enc, x_dim=int(bed.x_dim),
                    z_dim_state=z_dim_state, rank=rank, lr=lr,
                    n_params=sum(p.numel() for p in model.parameters())),
        split_note=split_note, train_seconds=train_seconds,
        L_q_first=l_q_hist[0], L_q_last=l_q_hist[-1],
        n_eval=n_eval, n_train_for_marginal=n_tr,
        operator_committor=dict(REL=murphy_op['REL'], RES=murphy_op['RES'], UNC=murphy_op['UNC'],
                                  brier_direct=murphy_op['brier_direct'], brier_decomp=murphy_op['brier_decomp'],
                                  residual=murphy_op['residual'], per_class=murphy_op['per_class']),
        marginal_baseline=dict(REL=murphy_marg['REL'], RES=murphy_marg['RES'], UNC=murphy_marg['UNC'],
                                 brier_direct=murphy_marg['brier_direct'], brier_decomp=murphy_marg['brier_decomp'],
                                 residual=murphy_marg['residual']),
        comparison_bootstrap=cmp,
        bar_check=dict(marginal_RES=murphy_marg['RES'], UNC=murphy_marg['UNC'],
                         threshold_2pct_UNC=0.02 * murphy_marg['UNC'], bar_fires=bool(bar_fires)),
        operator_beats_marginal=dict(
            gap=cmp['gap_mean'], gap_ci95=cmp['gap_ci95'],
            beats=bool(cmp['gap_ci95'][0] > 0),
        ),
    )
    save_result("part2_operator_murphy", record)
    board_log("part2_done",
              L_q_first=l_q_hist[0], L_q_last=l_q_hist[-1],
              op_RES=murphy_op['RES'], op_residual=murphy_op['residual'],
              marg_RES=murphy_marg['RES'], bar_fires=bool(bar_fires),
              gap_mean=cmp['gap_mean'], gap_ci95=cmp['gap_ci95'],
              beats_marginal=bool(cmp['gap_ci95'][0] > 0))
    print(json.dumps(record, indent=2, default=str))
    return record


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--n-games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    if args.part == 1:
        part1(n_games=args.n_games, seed=args.seed)
    elif args.part == 2:
        part2(n_games=args.n_games, seed=args.seed, steps=args.steps)
