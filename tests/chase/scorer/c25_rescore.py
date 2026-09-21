"""Chase: C25 (2) -- re-score a fresh operator-vs-marginal chess row under all
three bin schemes, using c25_scorer.py (imported, not reimplemented) and the
SAME real code paths wil_chess400.py part2 uses (ChessBed, TinyCEQ,
stage1_loss, heldout_split, draw, marginal_predictor -- all imported, none
re-derived). A reduced scale (300 games, 800 steps) vs wil_chess400's
1000 games / 4000 steps, because this is a fresh training run inside a
scratchpad-only, no-full-test-suite task -- flagged explicitly, not
presented as the published 0.001469 number, which this run does not
reproduce (different n_games/steps/seed budget) and is not claimed to.

L-REPRO
  seed        : 0 for bed build, split, model init, train draws; +2 offset
                for eval draws, +3 for the marginal predictor's training slice
                -- identical offsets to wil_chess400.part2.
  bed         : ChessBed.build(n_games=300, seed=0, max_plies=400)
  model       : TinyCEQ(n=16, d_enc=32, z_dim_state=8, rank=8), 800 Adam
                steps, lr=3e-3, batch=64 -- same architecture as
                wil_chess400.part2, fewer steps.
  dtype       : float64 for the scorer, float32 for the model (unchanged).
  command     : python c25_rescore.py
"""
import io, json, random, sys, datetime
import numpy as np
import torch

REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + r"\tests\wilson\resolution")

from ceqjepa.beds.chess import ChessBed, N_OUTCOMES, OUTCOME_NAMES  # noqa: E402
from ceqjepa.train import TinyCEQ, stage1_loss, heldout_split, draw  # noqa: E402
from ceqjepa.causal_eval import marginal_predictor  # noqa: E402

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sys.path.insert(0, SCRATCH)
from c25_scorer import score_all_schemes, score_forecasts, SCHEMES  # noqa: E402

BOARD_PATH = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"


def board_log(event, **kw):
    rec = dict(agent="Chase", ts=datetime.datetime.utcnow().isoformat() + "Z", event=event, **kw)
    with io.open(BOARD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def reflector_audit(bed, model, seed, n_train, steps, out_e_shape):
    """L-REFLECTOR: the fixed structure, printed before any arm is scored."""
    rows = [
        ("initializer", "torch default (kaiming/uniform per-layer), no custom init"),
        ("parameterization", f"TinyCEQ n=16 d_enc=32 z_dim_state=8 rank=8, "
                              f"{sum(p.numel() for p in model.parameters())} params"),
        ("corpus_regime", f"ChessBed self-play, uniform-random legal moves, "
                           f"n_games=300 seed={seed} max_plies=400"),
        ("scorer_functional", "one-vs-rest Murphy REL/RES/UNC, c25_scorer.score_forecasts"),
        ("bin_scheme", "fixed_width_10 / equal_count_200 / equal_count_50 (all three, per C25)"),
        ("dtype_path", "model forward float32; scorer accumulation float64"),
        ("torch_build", torch.__version__ + (" cuda" if torch.cuda.is_available() else " cpu-only")),
        ("eval_subsample_size", str(out_e_shape)),
    ]
    print("=== L-REFLECTOR: fixed structure audit ===")
    for name, val in rows:
        print(f"  {name:20s} = {val}")
    return rows


def main(n_games=300, seed=0, steps=800, batch_size=64):
    board_log("start", task="c25_rescore", n_games=n_games, seed=seed, steps=steps)
    bed = ChessBed.build(n_games=n_games, seed=seed, max_plies=400)
    train_bed, eval_bed, split_note = heldout_split(bed, 0.2, seed + 1)

    torch.manual_seed(seed)
    absorbing_idx = torch.arange(bed.nA)
    model = TinyCEQ(n=16, nA=bed.nA, d_enc=32, x_dim=bed.x_dim,
                     z_dim_state=8, g=0.9, absorbing_idx=absorbing_idx, rank=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    train_gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x, x_nx, q_star, v_idx, moves, do_tgt, do_mask = draw(train_bed, train_gen, batch_size)
        out = model(x)
        loss, l_q, l_z = stage1_loss(out, q_star, x_nx)
        opt.zero_grad(); loss.backward(); opt.step()

    n_eval = min(6000, len(eval_bed.rows))
    eval_gen = torch.Generator().manual_seed(seed + 2)
    x_e, x_nx_e, q_star_e, v_idx_e, *_ = draw(eval_bed, eval_gen, n_eval)
    with torch.no_grad():
        out_e = model(x_e)
    q_pred = out_e["q_alpha"].double()
    k_star = q_star_e.argmax(-1)

    reflector_audit(bed, model, seed, len(train_bed.rows), steps, tuple(q_pred.shape))

    n_tr = min(20000, len(train_bed.rows))
    x_tr, x_nx_tr, q_star_tr, v_idx_tr, *_ = draw(train_bed, torch.Generator().manual_seed(seed + 3), n_tr)
    k_star_tr = q_star_tr.argmax(-1)
    marg_fn = marginal_predictor(k_star_tr, bed.nA)
    q_marg = marg_fn({"k_star_obs": k_star}).double()

    out = score_all_schemes({"operator": q_pred, "marginal": q_marg}, k_star, bed.nA,
                             class_names=OUTCOME_NAMES)

    record = dict(n_games=n_games, seed=seed, steps=steps, n_eval=n_eval,
                   split_note=split_note, orderings=out["orderings"], verdict=out["verdict"],
                   per_scheme={s: {arm: {k: v for k, v in row.items() if k != "per_class"}
                                    for arm, row in out["per_scheme"][s]["rows"].items()}
                               for s in SCHEMES},
                   per_class={s: {arm: out["per_scheme"][s]["rows"][arm]["per_class"]
                                   for arm in ("operator", "marginal")}
                              for s in SCHEMES})
    with open(SCRATCH + r"\c25_rescore_results.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    print("\n=== C25 (2): ordering per scheme ===")
    for s in SCHEMES:
        r = out["per_scheme"][s]["rows"]
        print(f"  {s:16s} operator.RES={r['operator']['RES']:.6f}  "
              f"marginal.RES={r['marginal']['RES']:.6f}  order={out['orderings'][s]}")
    print("verdict:", out["verdict"])

    flagged = [(s, row["cls"]) for s in SCHEMES for row in out["per_scheme"][s]["rows"]["operator"]["per_class"]
               if row["flagged_single_bin"]]
    print("flagged single-bin (scheme, class) pairs, operator arm:", flagged)

    board_log("c25_rescore_done", n_games=n_games, steps=steps,
              orderings={s: list(out["orderings"][s]) for s in SCHEMES}, verdict=out["verdict"],
              operator_RES=out["per_scheme"]["fixed_width_10"]["rows"]["operator"]["RES"],
              marginal_RES=out["per_scheme"]["fixed_width_10"]["rows"]["marginal"]["RES"],
              flagged=[f"{s}:{c}" for s, c in flagged])
    return record


if __name__ == "__main__":
    main()
