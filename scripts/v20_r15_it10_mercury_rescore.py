"""IT.10 MERCURY -- second and third eval draw, and the missing softmax pair.

`scripts/v15_r1.py` is NOT edited and NOT imported by name; it is loaded as a
module so `train_one`, `gate_columns` and `probe` are the runner's own, byte for
byte, and the eval draw is the only thing that moves.

WHY THIS RETRAINS. `scripts/v15_r1.py:879` writes `models[(kind, seed)] = model`
and that key is never read again -- there is no `torch.save`, no state_dict, no
checkpoint anywhere under `scripts/`, `ceq/` or `scale/`. The dict dies with the
process. "Models are kept" is true INSIDE a run and false between runs, so a
second eval seed costs a retrain, not a forward pass. Training is deterministic
in the cell seed, so the retrain must reproduce the journalled cell bitwise at
eval seed 12345 -- that reproduction is this file's control and it aborts if it
fails on the control cell.
"""
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("_v15", ROOT / "scripts" / "v15_r1.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
EVAL_SEEDS = [12345, 12346, 20260902]
ARMS = {"arm_pl": [0, 8, 9, 10, 11, 12, 13, 14, 15], "softmax": [0, 8, 9, 10, 11, 12, 13, 14, 15]}
N_TRAIN, N_EVAL, STEPS = 2048, 4096, 150

torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(8)
S, D, DM, T_STAR = V.S, V.D, V.D_MODEL, V.T_STAR
floor1 = math.sqrt((T_STAR - 1) / T_STAR)
head = S - 1 - T_STAR
live = list(range(head + 1, S))
batch_fn = V.M3_TASKS[f"e3_t{T_STAR}"][0]

out = (ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl").open("a", encoding="utf-8")
def emit(rec):
    out.write(json.dumps(rec, default=float) + "\n"); out.flush()

emit(dict(t="header", tag="v20_r15_it10_mercury_rescore", task=f"e3_t{T_STAR}",
          t_star=T_STAR, s=S, d=D, d_model=DM, n_train=N_TRAIN, n_eval=N_EVAL,
          steps=STEPS, arms=list(ARMS), seeds=ARMS["arm_pl"], eval_seeds=EVAL_SEEDS,
          lr=V.LR, device=DEV, floor_1=floor1,
          deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
          deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
          cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
          threads=torch.get_num_threads(), torch=torch.__version__,
          instrument_hash=V.INSTRUMENT_MANIFEST["hash"],
          why="re-score kept-nothing models on 3 eval draws; softmax paired on 8..15",
          when=time.strftime("%Y-%m-%d %H:%M:%S")))

EV = {}
for es in EVAL_SEEDS:
    xe, ye, _, _ = batch_fn(N_EVAL, S, D, d_model=DM, seed=es, device=DEV)
    EV[es] = (xe, ye, xe[:, live, V.CH_DRIVE].reshape(-1))
    emit(dict(t="eval_draw", eval_seed=es, device=DEV,
              std_y=float(ye.std(unbiased=False)), mean_a=float(EV[es][2].mean())))

t0 = time.time()
for kind, seeds in ARMS.items():
    for seed in seeds:
        x_tr, y_tr, _, _ = batch_fn(N_TRAIN, S, D, d_model=DM, seed=seed, device=DEV)
        a_tr = x_tr[:, live, V.CH_DRIVE].reshape(-1)
        xe0, ye0, _ = EV[12345]
        model, r0 = V.train_one(kind, x_tr, y_tr, xe0, ye0, s=S, steps=STEPS,
                                seed=seed, device=DEV)
        mu = float(y_tr.mean()); sigma = float(y_tr.std(unbiased=False)) or 1.0
        model.eval()
        for es in EVAL_SEEDS:
            xe, ye, a_ev = EV[es]
            with torch.no_grad():
                pe = model(xe) * sigma + mu
            row = dict(t="rescore", kind=kind, seed=seed, eval_seed=es, steps=STEPS,
                       device=DEV, floor_1=floor1, train_secs=r0["secs"],
                       eval_nrmse=float(V.nrmse(pe, ye)), red_ok=bool(r0["red_ok"]))
            row["crosses_cell"] = bool(row["eval_nrmse"] < floor1)
            row["dist_to_floor"] = row["eval_nrmse"] - floor1
            row.update(V.gate_columns(kind, model, xe, live))
            ftr = V.gate_features(model, x_tr, live)
            if ftr is not None:
                fev = V.gate_features(model, xe, live)
                row["sign_r2"], row["sign_acc"], _ = V.probe(
                    ftr, torch.sign(a_tr), fev, torch.sign(a_ev))
                row["gate_r2"], _, _ = V.probe(V.recovered_gate(kind, ftr), a_tr,
                                               V.recovered_gate(kind, fev), a_ev)
            emit(row)
            print(f"[{kind:>8} seed={seed:>2} eval={es}] nrmse={row['eval_nrmse']:.10f} "
                  f"cross={row['crosses_cell']} lam={row['lambda_hat']:+.4f} "
                  f"amax={row['a_hat_max']:.4f} gr2={row.get('gate_r2')}")
        del x_tr, y_tr, model
emit(dict(t="wall", secs=time.time() - t0, device=DEV,
          n_cells=sum(len(v) for v in ARMS.values()), n_eval_seeds=len(EVAL_SEEDS)))
print(f"TOTAL {time.time() - t0:.2f}s")
