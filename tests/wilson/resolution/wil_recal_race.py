"""Wilson row-2, steps (3) and (4): the RECALIBRATED resolution race on the
rebuilt max_plies=400 chess bed, against TWO baselines, not one.

WHAT THIS ADDS OVER wil_chess400.py (row 1), which is imported, not reimplemented:
  1. RECALIBRATION. Row 1 compared an uncalibrated operator (REL=0.2456) against a
     marginal predictor sitting at its reliability floor (REL=0.0052). The task's own
     rule says recalibrate BOTH so only resolution can differ. Temperature is fit on a
     CALIBRATION split of games disjoint from both train and test.
  2. A NON-DEGENERATE MATCHED BASELINE. The marginal predictor has RES == 0 by
     construction; beating it is the weakest possible win. The ply-bucket base-rate arm
     (8 quantile buckets on ply_idx, rates fit on TRAIN) is architecture-free, reads NO
     board -- only game depth -- and already measured RES = 0.027236 in
     wil_bar_can_fire.py. It is the control that LACKS the tested property
     (board-conditional forecasting) while keeping everything else.
  3. GAME-PAIRED BOOTSTRAP. Row 1 resampled ITEMS. chess.py's own docstring says the
     effective n is GAMES, not plies -- every ply of a game carries the same label.
     Resampling plies understates the CI. This resamples games.
  4. TWO-SIDED BAR-CAN-FIRE. Known-zero arm (marginal) must report RES ~ 0; known-
     informative arm (ply-bucket) must clear the bar. A pipeline that only passes the
     first half can still be blind.

L-REPRO
  bed        : ceqjepa.beds.chess.ChessBed.build(n_games=N, seed=0, max_plies=400)
  splits     : ceqjepa.train.heldout_split, twice, on GAMES (60 train / 20 calib / 20 test)
  model      : ceqjepa.train.TinyCEQ, stage1_loss, draw -- unmodified
  committor  : model(x)['q_alpha'], the operator.committor read
  decomp     : wil_chess400.murphy_binned_onevsrest (10 equal-width prob bins, one-vs-rest)
  bar        : 0.00506 = 5% of the cap-400 oracle ceiling 0.101170, pre-registered in
               wil_resolution_ruling.md BEFORE any arm was scored
  dtype      : float64 for all scoring, float32 for the net, CPU only
  command    : python wil_recal_race.py --n-games 1000 --steps 4000
"""
from __future__ import annotations
import argparse, io, json, sys, time, datetime
import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sys.path.insert(0, REPO)
sys.path.insert(0, SCRATCH)

from ceqjepa.beds.chess import ChessBed, N_OUTCOMES, OUTCOME_NAMES, fen_to_vec
from ceqjepa.train import TinyCEQ, stage1_loss, heldout_split, draw
from ceqjepa.causal_eval import marginal_predictor
from wil_chess400 import murphy_binned_onevsrest

BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
OUT = SCRATCH + r"\wil_recal_race.json"
BAR = 0.00506          # 5% of ceiling 0.101170, pre-registered in wil_resolution_ruling.md
CEILING = 0.10116955630126778


def board_log(event, **kw):
    rec = dict(agent="Wilson", ts=datetime.datetime.utcnow().isoformat() + "Z", event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def encode(bed, cap=None):
    """Deterministic full-corpus encode: x, k_star, ply_idx, game_id. No draw()."""
    rows = bed.rows if cap is None else bed.rows[:cap]
    x = torch.from_numpy(np.stack([fen_to_vec(r["fen_before"]) for r in rows]))
    k = torch.tensor([int(r["outcome"].argmax()) for r in rows], dtype=torch.long)
    ply = torch.tensor([r["ply_idx"] for r in rows], dtype=torch.long)
    gid = torch.cumsum((ply == 0).long(), 0) - 1
    return x, k, ply, gid


def fit_temperature(q_cal, k_cal):
    """Scalar T minimising multiclass NLL on the calibration split. logits = log q;
    q_T = softmax(log q / T). T=1 is the identity, so the recalibrator can always
    decline to move. Grid then refine -- two lines, no scipy."""
    logits = torch.log(torch.clamp(q_cal.double(), min=1e-12))

    def nll(T):
        return float(F.cross_entropy(logits / float(T), k_cal))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121))
    T = float(min(grid, key=nll))
    fine = np.linspace(max(T * 0.7, 1e-3), T * 1.3, 61)
    T = float(min(fine, key=nll))
    return T, nll(T), nll(1.0)


def apply_T(q, T):
    return torch.softmax(torch.log(torch.clamp(q.double(), min=1e-12)) / float(T), dim=-1)


def ply_bucket_arm(ply_fit, k_fit, ply_apply, K, n_buckets=8):
    """Base rate per ply-depth quantile bucket. Reads depth only, never the board."""
    edges = np.quantile(ply_fit.numpy(), np.linspace(0, 1, n_buckets + 1)[1:-1])
    b_fit = np.digitize(ply_fit.numpy(), edges)
    b_app = np.digitize(ply_apply.numpy(), edges)
    oh = F.one_hot(k_fit, K).double().numpy()
    glob = oh.mean(0)
    rates = {int(u): oh[b_fit == u].mean(0) for u in np.unique(b_fit)}
    return torch.tensor(np.stack([rates.get(int(b), glob) for b in b_app]), dtype=torch.float64)


def game_paired_bootstrap(arms, k_star, gid, K, n_bins=10, n_boot=300, seed=0):
    """Resample GAMES with replacement (chess.py: effective n is games, not plies).
    Every arm is scored on the SAME resampled rows, so gaps are paired."""
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
                   res_mean=float(res[a].mean()), res_se=float(res[a].std())) for a in names}
    return out, res


def main(n_games, seed, steps, n_boot, eval_cap):
    t_all = time.time()
    board_log("start", row="recal_race", n_games=n_games, seed=seed, steps=steps, bar=BAR)

    t0 = time.time()
    bed = ChessBed.build(n_games=n_games, seed=seed, max_plies=400)
    t_build = time.time() - t0
    train_bed, rest_bed, note1 = heldout_split(bed, 0.4, seed + 1)
    calib_bed, test_bed, note2 = heldout_split(rest_bed, 0.5, seed + 2)
    print("[split1]", note1, flush=True)
    print("[split2]", note2, flush=True)

    torch.manual_seed(seed)
    model = TinyCEQ(n=16, nA=bed.nA, d_enc=32, x_dim=bed.x_dim, z_dim_state=8,
                    g=0.9, absorbing_idx=torch.arange(bed.nA), rank=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    lhist = []
    for step in range(steps):
        x, x_nx, q_star, v_idx, *_ = draw(train_bed, gen, 64)
        loss, l_q, l_z = stage1_loss(model(x), q_star, x_nx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lhist.append(l_q)
        if step % max(1, steps // 8) == 0:
            print(f"[train] step={step} L_q={l_q:.4f}", flush=True)
    t_train = time.time() - t0
    board_log("trained", L_q_first=lhist[0], L_q_last=lhist[-1], seconds=t_train)

    x_cal, k_cal, ply_cal, _ = encode(calib_bed, cap=eval_cap)
    x_te, k_te, ply_te, gid_te = encode(test_bed, cap=eval_cap)
    x_tr, k_tr, ply_tr, _ = encode(train_bed, cap=eval_cap)
    with torch.no_grad():
        q_cal = model(x_cal)["q_alpha"].double()
        q_te = model(x_te)["q_alpha"].double()
    T, nll_T, nll_1 = fit_temperature(q_cal, k_cal)
    board_log("temperature", T=T, nll_at_T=nll_T, nll_at_1=nll_1,
              n_calib=int(len(k_cal)), n_test=int(len(k_te)))
    print(f"[recal] T={T:.4f} NLL {nll_1:.4f} -> {nll_T:.4f}", flush=True)

    marg_fn = marginal_predictor(k_tr, bed.nA)
    q_marg = marg_fn({"k_star_obs": k_te}).double()
    q_ply = ply_bucket_arm(ply_tr, k_tr, ply_te, bed.nA)

    # recalibrate EVERY arm through the identical step, including the controls
    T_marg, _, _ = fit_temperature(marg_fn({"k_star_obs": k_cal}).double(), k_cal)
    T_ply, _, _ = fit_temperature(ply_bucket_arm(ply_tr, k_tr, ply_cal, bed.nA), k_cal)

    arms = {
        "operator_raw": q_te,
        "operator_recal": apply_T(q_te, T),
        "marginal_recal": apply_T(q_marg, T_marg),
        "plybucket_recal": apply_T(q_ply, T_ply),
    }
    murphy = {a: murphy_binned_onevsrest(q, k_te, bed.nA, 10) for a, q in arms.items()}
    for a, m in murphy.items():
        print(f"[murphy] {a:16s} REL={m['REL']:.6f} RES={m['RES']:.6f} UNC={m['UNC']:.6f} "
              f"BS={m['brier_direct']:.6f} resid={m['residual']:.2e}", flush=True)
    board_log("murphy", **{a: dict(REL=m["REL"], RES=m["RES"], BS=m["brier_direct"])
                           for a, m in murphy.items()})

    boot, raw = game_paired_bootstrap(arms, k_te, gid_te, bed.nA, n_boot=n_boot, seed=seed)
    gaps = {}
    for a, b in [("operator_recal", "marginal_recal"),
                 ("operator_recal", "plybucket_recal"),
                 ("plybucket_recal", "marginal_recal")]:
        d = raw[a] - raw[b]
        point = murphy[a]["RES"] - murphy[b]["RES"]
        gaps[f"{a}__minus__{b}"] = dict(
            gap_point=point, gap_mean=float(d.mean()), gap_se=float(d.std()),
            gap_ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
            clears_bar=bool(abs(point) > BAR), frac_of_bar=point / BAR)

    bar_fire = dict(
        known_zero_arm="marginal_recal",
        known_zero_RES=murphy["marginal_recal"]["RES"],
        known_zero_reports_near_zero=bool(murphy["marginal_recal"]["RES"] < 1e-6),
        known_informative_arm="plybucket_recal",
        known_informative_RES=murphy["plybucket_recal"]["RES"],
        known_informative_clears_bar=bool(
            murphy["plybucket_recal"]["RES"] - murphy["marginal_recal"]["RES"] > BAR),
        pipeline_two_sided_ok=bool(
            murphy["marginal_recal"]["RES"] < 1e-6 and
            murphy["plybucket_recal"]["RES"] - murphy["marginal_recal"]["RES"] > BAR))

    rec = dict(
        n_games=n_games, seed=seed, steps=steps, build_seconds=t_build, train_seconds=t_train,
        split_train=note1, split_calib_test=note2,
        n_params=sum(p.numel() for p in model.parameters()),
        L_q_first=lhist[0], L_q_last=lhist[-1],
        n_calib=int(len(k_cal)), n_test=int(len(k_te)),
        n_test_games=int(len(torch.unique(gid_te))),
        temperature=dict(operator=T, marginal=T_marg, plybucket=T_ply,
                         nll_at_1=nll_1, nll_at_T=nll_T),
        murphy={a: {k: v for k, v in m.items() if k != "per_class"} for a, m in murphy.items()},
        murphy_per_class={a: m["per_class"] for a, m in murphy.items()},
        bootstrap_game_paired=boot, n_boot=n_boot, gaps=gaps,
        bar=dict(value=BAR,
                 basis="5% of cap-400 oracle ceiling 0.101170, pre-registered in wil_resolution_ruling.md"),
        ceiling=CEILING,
        operator_frac_of_ceiling=murphy["operator_recal"]["RES"] / CEILING,
        plybucket_frac_of_ceiling=murphy["plybucket_recal"]["RES"] / CEILING,
        bar_can_fire=bar_fire,
        total_seconds=time.time() - t_all)
    json.dump(rec, open(OUT, "w"), indent=2, default=float)
    board_log("recal_race_done", op_RES=murphy["operator_recal"]["RES"],
              ply_RES=murphy["plybucket_recal"]["RES"], marg_RES=murphy["marginal_recal"]["RES"],
              op_vs_marg=gaps["operator_recal__minus__marginal_recal"],
              op_vs_ply=gaps["operator_recal__minus__plybucket_recal"],
              bar_two_sided_ok=bar_fire["pipeline_two_sided_ok"], seconds=rec["total_seconds"])
    print(json.dumps(rec["gaps"], indent=2, default=float))
    print(json.dumps(rec["bar_can_fire"], indent=2, default=float))
    print(f"[done] {rec['total_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--eval-cap", type=int, default=40000)
    a = ap.parse_args()
    main(a.n_games, a.seed, a.steps, a.n_boot, a.eval_cap)
