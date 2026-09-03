"""IT.11 MERCURY, EXPERIMENT B -- a FOURTH eval draw on the nine arm_pl cells.

it.10 covered `arm_pl` and `softmax` only. The round therefore holds 18
draw-checked W3/skyline cells against 0 for W1, and seed 2's crossing cell
`eval_nrmse = 0.20391993939877656` -- the whole of W1's remaining mass -- has
never been scored on a second eval draw. This closes that, and nothing else.

`scripts/v15_r1.py` is NOT edited and NOT imported by name; it is loaded with
importlib so `train_one`, `gate_columns`, `gate_features`, `recovered_gate`,
`probe` and `nrmse` are the runner's own and `instrument_hash` is unmoved.

SEED ORDER IS 2 FIRST, DELIBERATELY. Seed 2 is simultaneously the round's
highest-value cell and one of its own bitwise controls, so a run cut off by the
wall clock still answers the question the round most needs answered.
"""
import importlib.util, json, math, os, pathlib, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("_v15", ROOT / "scripts" / "v15_r1.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
EVAL_SEEDS = [12345, 20260903]
SEEDS = [15, 12, 9, 14, 10, 13, 0, 8, 11]
N_TRAIN, N_EVAL, STEPS = 2048, 4096, 150
#: DERIVED price estimate, stated BEFORE the run: sum of the 16 banked
#: `secs` (129.287 retake 0-7 + 146.399 it.6 8-15 = 275.686) + 26 s of
#: module load, three eval-draw builds and 48 scorings with gate columns.
PRICE_ESTIMATE = 26.0

torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(8)
S, D, DM, T_STAR = V.S, V.D, V.D_MODEL, V.T_STAR
floor1 = math.sqrt((T_STAR - 1) / T_STAR)
head = S - 1 - T_STAR
live = list(range(head + 1, S))
batch_fn = V.M3_TASKS[f"e3_t{T_STAR}"][0]

out = (ROOT / "results" / "v20_r15_it11_mercury_draw4.jsonl").open("a", encoding="utf-8")
def emit(rec):
    out.write(json.dumps(rec, default=float) + "\n"); out.flush()

emit(dict(t="header", tag="v20_r15_it11_mercury_draw4", task=f"e3_t{T_STAR}",
          t_star=T_STAR, s=S, d=D, d_model=DM, n_train=N_TRAIN, n_eval=N_EVAL,
          steps=STEPS, arms=["arm_pl"], seeds=SEEDS, eval_seeds=EVAL_SEEDS,
          lr=V.LR, device=DEV, floor_1=floor1,
          deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
          deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
          cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
          threads=torch.get_num_threads(), torch=torch.__version__,
          instrument_hash=V.INSTRUMENT_MANIFEST["hash"],
          price_estimate_secs=PRICE_ESTIMATE,
          why="EXPERIMENT B: fourth eval draw 20260903 on the nine arm_pl cells; tests VENUS it.10 2.2 pre-registration; 12345 re-scored as the bitwise control",
          when=time.strftime("%Y-%m-%d %H:%M:%S")))

EV = {}
for es in EVAL_SEEDS:
    xe, ye, _, _ = batch_fn(N_EVAL, S, D, d_model=DM, seed=es, device=DEV)
    EV[es] = (xe, ye, xe[:, live, V.CH_DRIVE].reshape(-1))
    emit(dict(t="eval_draw", eval_seed=es, device=DEV,
              std_y=float(ye.std(unbiased=False)), mean_a=float(EV[es][2].mean())))

kind = "arm_pl"
t0 = time.time()
for seed in SEEDS:
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
        print(f"[smprime seed={seed:>2} eval={es}] nrmse={row['eval_nrmse']:.12f} "
              f"cross={row['crosses_cell']} lam={row['lambda_hat']} "
              f"lamlive={row['lambda_hat_live']} amax={row['a_hat_max']:.6f} "
              f"annih={row['frac_gate_annihilated']:.6f} ur={row['unit_root']}",
              flush=True)
    del x_tr, y_tr, model
emit(dict(t="wall", secs=time.time() - t0, device=DEV, n_cells=len(SEEDS),
          n_eval_seeds=len(EVAL_SEEDS), price_estimate_secs=PRICE_ESTIMATE))
print(f"TOTAL {time.time() - t0:.2f}s  ESTIMATE {PRICE_ESTIMATE:.1f}s", flush=True)
