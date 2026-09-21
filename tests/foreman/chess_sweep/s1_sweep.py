"""S1 -- THE ROW: RES/ceiling size sweep, operator vs G-identically-1 softmax
twin, isotonic-recalibrated, both against the piece-count-histogram and
base-rate floors, oracle ceiling as the top line. max_plies=400 throughout.

REUSES, not reimplemented: ceqjepa.beds.chess.ChessBed/fen_to_vec,
ceqjepa.train.TinyCEQ/stage1_loss/heldout_split/draw,
ceqjepa.causal_eval.marginal_predictor, wil_chess400.murphy_binned_onevsrest,
wil_recal_race.encode/game_paired_bootstrap patterns. wil_res_ceiling.py's own
measured ceiling (0.10116955630126778, CI95 [0.0708, 0.1329], n_games=150,
max_plies=400, R=16 rollouts, game-bootstrapped) is the TOP LINE reused
verbatim rather than re-measured, per L-REPRO provenance below.

NEW here (nothing built elsewhere to reuse):
  - SoftmaxTwin: same enc (Linear-GELU-Linear) as TinyCEQ, head Linear(d_enc,
    nA)+softmax, NO resolvent/committor solve at all -- "G identically 1"
    read as: the gate that turns causal attention into a resolvent is pinned
    off, so the twin is a plain softmax classifier on the same 769-float
    board input. Parameter count matched to the operator's numel() by a
    d_enc grid search (both dominated by the fixed x_dim=769 encoder/head
    linear, so this is a one-line closest-match search, not an optimisation).
  - piece_count_histogram_arm: 8 quantile bins of total piece count, parsed
    directly from each row's own FEN (board field alpha-character count) --
    the same summary statistic the ceiling row used, strictly less info than
    the operator's 769-float input.
  - isotonic_arm: sklearn.isotonic.IsotonicRegression, ONE PER CLASS (fit on
    calib, applied to test), fit independently per class -- valid because
    murphy_binned_onevsrest scores each class as an independent one-vs-rest
    binary problem (REL/RES/UNC sum coordinatewise), so per-class isotonic
    output need not renormalise to a simplex row for the score to be exact.

L-REPRO
  seed        : --seed (bed build, split offsets, torch init, isotonic n/a)
  bed         : ChessBed.build(n_games=N, seed, max_plies=400)
  splits      : heldout_split twice, BY GAME (60 train / 20 calib / 20 test)
  dtype       : float64 scoring, float32 net, CPU only (matches every sibling
                script in tests/wilson/resolution/; no CUDA touched here)
  ceiling     : REUSED from wil_res_ceiling.py's own run, not re-measured
                (0.10116955630126778, CI95 [0.070828, 0.132868], n_games=150,
                R=16, max_plies=400) -- see CEILING_PROVENANCE below.
  command     : python s1_sweep.py --tier 0 --n-games 300 --steps 1500 --seed 0

KNOWN SCOPE CUT (kill-adjacent structural finding, reported flatly, not
buried): TinyCEQ's encoder and readout are both Linear(*, x_dim=769) /
Linear(769, *), and x_dim is a bed-level constant (chess.py:71). The SMALLEST
config this architecture can build is ~8.3k parameters (n=8,d_enc=2,z=2,
rank=2: measured 8,289) -- 2,000 parameters is not reachable without changing
the encoder, which this task forbids rewriting. TIERS below substitute the
smallest achievable size for the pre-registered "2k" and say so in the record
rather than silently relabelling 8.3k as 2k.
"""
from __future__ import annotations
import argparse, io, json, sys, time, datetime, itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sys.path.insert(0, REPO)
sys.path.insert(0, SCRATCH)

from ceqjepa.beds.chess import ChessBed, N_OUTCOMES, OUTCOME_NAMES, fen_to_vec  # noqa: E402
from ceqjepa.train import TinyCEQ, stage1_loss, heldout_split, draw  # noqa: E402
from ceqjepa.causal_eval import marginal_predictor  # noqa: E402
from wil_chess400 import murphy_binned_onevsrest  # noqa: E402

BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
OUT = SCRATCH + r"\s1_results.jsonl"

# --- CEILING_PROVENANCE: verbatim from wil_res_ceiling.py's own recorded run
# (tag=control_cap400, this repo, this box) -- reused per the task's own
# instruction that wil_res_ceiling.py is validated and not to be reimplemented.
CEILING = 0.10116955630126778
CEILING_CI95 = [0.070828, 0.132868]
CEILING_PROVENANCE = dict(source="wil_res_ceiling.py:measure(150,400,'control_cap400')",
                           n_games=150, R=16, max_plies=400, seed=0)

TIERS = [
    dict(label="tier0_smallest_achievable_not_2k", n=8, d_enc=2, z=2, rank=2),
    dict(label="tier1_20k", n=16, d_enc=8, z=8, rank=8),
    dict(label="tier2_200k", n=40, d_enc=80, z=40, rank=16),
]


def board_log(event, **kw):
    rec = dict(agent="Foreman", ts=datetime.datetime.utcnow().isoformat() + "Z", event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def append_result(rec):
    with io.open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


# --------------------------------------------------------------------------- #
# G-identically-1 softmax twin: same enc, no resolvent/committor solve at all.
# --------------------------------------------------------------------------- #
class SoftmaxTwin(nn.Module):
    def __init__(self, x_dim, d_enc, nA):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, d_enc), nn.GELU(), nn.Linear(d_enc, d_enc))
        self.head = nn.Linear(d_enc, nA)

    def forward(self, x):
        e = self.enc(x)
        q_alpha = torch.softmax(self.head(e), dim=-1)
        return dict(q_alpha=q_alpha)


def twin_matched_to(target_params, x_dim, nA, d_grid=range(1, 400)):
    best = None
    for d in d_grid:
        m = SoftmaxTwin(x_dim, d, nA)
        n = sum(p.numel() for p in m.parameters())
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (d, n)
        if n > target_params * 1.5:
            break
    d, n = best
    return SoftmaxTwin(x_dim, d, nA), n


def q_loss(q_alpha, q_star, eps=1e-6):
    qc = q_alpha.clamp(eps, 1 - eps)
    return -(q_star * qc.log() + (1 - q_star) * (1 - qc).log()).sum(-1).mean()


def train_to_convergence(model, train_bed, gen, steps, batch, patience_frac=0.2, min_delta=1e-4):
    """Adam on q_loss (TinyCEQ: full stage1_loss incl. its own L_z=0 term via
    forward's x_hat, but only L_q is scored here -- SAME loss surface for both
    arms since the twin has no x_hat). Early-stops on a plateau of the
    trailing-mean training loss so 'to convergence' is a checked claim, not an
    assertion; if the budget runs out first that is reported, not hidden."""
    is_operator = hasattr(model, "L0")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    hist = []
    window = max(20, steps // 20)
    stopped_at = steps
    for step in range(steps):
        x, x_nx, q_star, v_idx, *_ = draw(train_bed, gen, batch)
        out = model(x)
        if is_operator:
            loss, l_q, l_z = stage1_loss(out, q_star, x_nx)
        else:
            l_q = float(q_loss(out["q_alpha"], q_star))
            loss = q_loss(out["q_alpha"], q_star)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(l_q)
        if len(hist) >= 2 * window and step > steps * (1 - patience_frac):
            prev = float(np.mean(hist[-2 * window:-window]))
            cur = float(np.mean(hist[-window:]))
            if prev - cur < min_delta:
                stopped_at = step + 1
                break
    return dict(L_q_first=hist[0], L_q_last=hist[-1], steps_run=stopped_at,
                steps_budget=steps, converged=stopped_at < steps,
                trailing_mean_window=window)


def encode(bed, cap=None):
    rows = bed.rows if cap is None else bed.rows[:cap]
    x = torch.from_numpy(np.stack([fen_to_vec(r["fen_before"]) for r in rows]))
    k = torch.tensor([int(r["outcome"].argmax()) for r in rows], dtype=torch.long)
    ply = torch.tensor([r["ply_idx"] for r in rows], dtype=torch.long)
    piece = torch.tensor([sum(1 for c in r["fen_before"].split(" ")[0] if c.isalpha())
                           for r in rows], dtype=torch.long)
    gid = torch.cumsum((ply == 0).long(), 0) - 1
    return x, k, ply, piece, gid


def histogram_arm(fit_vals, k_fit, apply_vals, K, n_buckets=8):
    """8 quantile buckets on `fit_vals` (piece count or nothing else), rates
    fit on TRAIN, applied to TEST/CALIB -- architecture-free by construction."""
    fv = fit_vals.numpy().astype(np.float64)
    edges = np.quantile(fv, np.linspace(0, 1, n_buckets + 1)[1:-1])
    b_fit = np.digitize(fv, edges)
    b_app = np.digitize(apply_vals.numpy().astype(np.float64), edges)
    oh = F.one_hot(k_fit, K).double().numpy()
    glob = oh.mean(0)
    rates = {int(u): oh[b_fit == u].mean(0) for u in np.unique(b_fit)}
    return torch.tensor(np.stack([rates.get(int(b), glob) for b in b_app]), dtype=torch.float64)


def isotonic_recal(q_cal, k_cal, q_apply, K):
    """One IsotonicRegression PER CLASS, fit on calib, applied elsewhere.
    No renormalisation: murphy_binned_onevsrest scores each class
    independently, so this is exact for that scoring rule as-is."""
    out = np.empty(q_apply.shape, dtype=np.float64)
    k_oh = F.one_hot(k_cal, K).double().numpy()
    q_cal_np = q_cal.double().numpy()
    q_apply_np = q_apply.double().numpy()
    for c in range(K):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(q_cal_np[:, c], k_oh[:, c])
        out[:, c] = ir.predict(q_apply_np[:, c])
    return torch.tensor(out, dtype=torch.float64)


def game_paired_bootstrap(arms, k_star, gid, K, n_bins=10, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = torch.unique(gid)
    idx_of = {int(g): (gid == g).nonzero(as_tuple=True)[0] for g in uniq.tolist()}
    names = list(arms)
    res = {a: np.empty(n_boot) for a in names}
    for i in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = torch.cat([idx_of[int(uniq[p])] for p in pick])
        kk = k_star[idx]
        for a in names:
            res[a][i] = murphy_binned_onevsrest(arms[a][idx], kk, K, n_bins)["RES"]
    out = {a: dict(res_ci95=[float(np.percentile(res[a], 2.5)), float(np.percentile(res[a], 97.5))],
                   res_mean=float(res[a].mean())) for a in names}
    return out, res


def run_tier(tier, n_games, seed, steps, n_boot, eval_cap, batch):
    t_all = time.time()
    board_log("s1_tier_start", tier=tier["label"], n_games=n_games, seed=seed, steps=steps)

    bed = ChessBed.build(n_games=n_games, seed=seed, max_plies=400)
    train_bed, rest_bed, note1 = heldout_split(bed, 0.4, seed + 1)
    calib_bed, test_bed, note2 = heldout_split(rest_bed, 0.5, seed + 2)

    absorb = torch.arange(N_OUTCOMES)
    torch.manual_seed(seed)
    operator = TinyCEQ(n=tier["n"], nA=bed.nA, d_enc=tier["d_enc"], x_dim=bed.x_dim,
                        z_dim_state=tier["z"], g=0.9, absorbing_idx=absorb, rank=tier["rank"])
    n_params_op = sum(p.numel() for p in operator.parameters())
    torch.manual_seed(seed + 777)
    twin, n_params_twin = twin_matched_to(n_params_op, bed.x_dim, bed.nA)

    gen_op = torch.Generator().manual_seed(seed)
    gen_tw = torch.Generator().manual_seed(seed)
    t0 = time.time()
    conv_op = train_to_convergence(operator, train_bed, gen_op, steps, batch)
    t_train_op = time.time() - t0
    t0 = time.time()
    conv_tw = train_to_convergence(twin, train_bed, gen_tw, steps, batch)
    t_train_tw = time.time() - t0
    board_log("s1_trained", tier=tier["label"], n_params_op=n_params_op, n_params_twin=n_params_twin,
              op_L_q_last=conv_op["L_q_last"], twin_L_q_last=conv_tw["L_q_last"],
              op_converged=conv_op["converged"], twin_converged=conv_tw["converged"])

    x_cal, k_cal, ply_cal, piece_cal, _ = encode(calib_bed, cap=eval_cap)
    x_te, k_te, ply_te, piece_te, gid_te = encode(test_bed, cap=eval_cap)
    x_tr, k_tr, ply_tr, piece_tr, _ = encode(train_bed, cap=eval_cap)

    with torch.no_grad():
        q_op_cal = operator(x_cal)["q_alpha"].double()
        q_op_te = operator(x_te)["q_alpha"].double()
        q_tw_cal = twin(x_cal)["q_alpha"].double()
        q_tw_te = twin(x_te)["q_alpha"].double()

    q_op_te_iso = isotonic_recal(q_op_cal, k_cal, q_op_te, bed.nA)
    q_tw_te_iso = isotonic_recal(q_tw_cal, k_cal, q_tw_te, bed.nA)

    marg_fn = marginal_predictor(k_tr, bed.nA)
    q_marg_te = marg_fn({"k_star_obs": k_te}).double()
    q_marg_cal = marg_fn({"k_star_obs": k_cal}).double()
    q_marg_te_iso = isotonic_recal(q_marg_cal, k_cal, q_marg_te, bed.nA)

    q_piece_te = histogram_arm(piece_tr, k_tr, piece_te, bed.nA)
    q_piece_cal = histogram_arm(piece_tr, k_tr, piece_cal, bed.nA)
    q_piece_te_iso = isotonic_recal(q_piece_cal, k_cal, q_piece_te, bed.nA)

    arms = dict(operator=q_op_te_iso, twin=q_tw_te_iso, base_rate=q_marg_te_iso,
                piece_count_hist=q_piece_te_iso)
    murphy = {a: murphy_binned_onevsrest(q, k_te, bed.nA, 10) for a, q in arms.items()}

    boot, raw = game_paired_bootstrap(arms, k_te, gid_te, bed.nA, n_boot=n_boot, seed=seed)
    d = raw["operator"] - raw["twin"]
    op_v_twin = dict(gap_point=murphy["operator"]["RES"] - murphy["twin"]["RES"],
                      gap_ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0))

    rec = dict(
        tier=tier["label"], config=tier, seed=seed, n_games_requested=n_games,
        n_games_used_train=int(len(torch.unique((torch.tensor([r["ply_idx"] for r in train_bed.rows]) == 0).cumsum(0) - 1))),
        steps_budget=steps, batch=batch, n_boot=n_boot,
        n_params_operator=n_params_op, n_params_twin=n_params_twin,
        n_test_games=int(len(torch.unique(gid_te))), n_test_rows=int(len(k_te)),
        train_seconds=dict(operator=t_train_op, twin=t_train_tw),
        convergence=dict(operator=conv_op, twin=conv_tw),
        murphy={a: {k: v for k, v in m.items() if k != "per_class"} for a, m in murphy.items()},
        bootstrap_game_paired=boot,
        ceiling=CEILING, ceiling_ci95=CEILING_CI95, ceiling_provenance=CEILING_PROVENANCE,
        res_over_ceiling={a: murphy[a]["RES"] / CEILING for a in arms},
        operator_vs_twin=op_v_twin,
        kill_check_200k=(tier["label"] == "tier2_200k" and murphy["operator"]["RES"] / CEILING < 0.05),
        total_seconds=time.time() - t_all)
    append_result(rec)
    board_log("s1_tier_done", tier=tier["label"], op_res_over_ceiling=rec["res_over_ceiling"]["operator"],
              twin_res_over_ceiling=rec["res_over_ceiling"]["twin"],
              piece_res_over_ceiling=rec["res_over_ceiling"]["piece_count_hist"],
              base_res_over_ceiling=rec["res_over_ceiling"]["base_rate"],
              op_vs_twin_excludes_zero=op_v_twin["excludes_zero"], seconds=rec["total_seconds"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, help="index into TIERS")
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--eval-cap", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    rec = run_tier(TIERS[a.tier], a.n_games, a.seed, a.steps, a.n_boot, a.eval_cap, a.batch)
    print(json.dumps({k: v for k, v in rec.items()}, indent=2, default=float))
