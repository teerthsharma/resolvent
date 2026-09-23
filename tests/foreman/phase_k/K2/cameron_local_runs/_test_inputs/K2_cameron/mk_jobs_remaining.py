# Writes jobs_remaining.json for Wilson's Kaggle kernels (Dispatcher order). Reads grid_id_jobs.json (the ward's
# registered argv, unchanged), the landed runs, dependency shas, and bed fingerprints (numpy stream check).
import hashlib, json, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent.resolve()
PK = HERE.parent.parent                          # SP/phase_k
K1C = PK / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import bed_kp
REPO = Path("C:/Users/seal/Desktop/New folder (32)")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
J = json.loads((HERE / "grid_id_jobs.json").read_text())
out_of = lambda j: Path(j["argv"][j["argv"].index("--out") + 1])
landed = [j["name"] for j in J if (out_of(j) / "result.json").exists() and (not j["fR"] or (out_of(j) / "walk" / "meta.json").exists())]
py = "python"
RD = str(K1C / "rdepth.py").replace("\\", "/")
TRAIN_BACK = ["result.json", "log.jsonl", "model.pt", "ok_4096.npz", "ok_8192.npz", "ok_16384.npz"]
WALK_BACK = ["walk/ptr_4096.npz", "walk/ptr_16384.npz", "walk/meta.json"]
jobs = []
c = next(j for j in J if j["name"] == "id_aL4_s0")
jobs.append({"order": 0, "name": "id_aL4_s0", "kind": "control", "cwd": str(HERE).replace("\\", "/"),
             "argv": [py, RD] + c["argv"],
             "note": "hardware check: id_aL4_s0 re-run with its registered argv unchanged; it ran locally (RTX 4060, landed 12:03:19). "
                     "Bring its outputs back to runs_kaggle_control/id_aL4_s0/, NEVER into runs/ (runs/id_aL4_s0 is the local landed run "
                     "that test_k2_grid_kaggle.py reads). No bar compares the two yet; one must be registered before the control is read.",
             "bring_back": TRAIN_BACK, "bring_back_to": "runs_kaggle_control/id_aL4_s0"})
k = 1
for j in J:
    if j["name"] in landed:
        continue
    jobs.append({"order": k, "name": j["name"], "kind": "train", "cwd": str(HERE).replace("\\", "/"),
                 "argv": [py, RD] + j["argv"], "bring_back": TRAIN_BACK, "bring_back_to": f"runs/{j['name']}"})
    k += 1
    if j["fR"]:
        jobs.append({"order": k, "name": f"{j['name']}_walk", "kind": "walk", "after": j["name"], "cwd": str(HERE).replace("\\", "/"),
                     "argv": [py, "k2_ptr.py", "walk", str(out_of(j)).replace("\\", "/")],
                     "bring_back": WALK_BACK, "bring_back_to": f"runs/{j['name']}"})
        k += 1


def fp(beds):
    h = hashlib.sha256()
    for b in beds:
        for key in ("ids", "pids", "target", "depth", "root"):
            h.update(np.ascontiguousarray(b[key], dtype=np.int64).tobytes())
    return h.hexdigest()


fps = {f"far10_n{n}": fp([bed_kp.make_test(np.random.default_rng([31, n, kk]), n) for kk in range(8)]) for n in (4096, 8192, 16384)}
fps["heldout_1024"] = fp([bed_kp.make_train(np.random.default_rng([21, kk])) for kk in range(64)])
ns = (256, 512, 1024, 2048, 4096)
for s in (0, 1, 2):
    bs = []
    for step in range(1, 6):
        rng = np.random.default_rng([1000 + s, step]); n = ns[step % 5]
        bs += [bed_kp.make_train(rng, n=n) for _ in range(8192 // n)]
    fps[f"train_seed{s}_steps1-5"] = fp(bs)
deps = {"K1/cameron/rdepth.py": K1C / "rdepth.py", "K1/cameron/train_ladder.py": K1C / "train_ladder.py",
        "K1/cameron/bed_k.py": K1C / "bed_k.py", "K1/cameron/bed_kp.py": K1C / "bed_kp.py",
        "K2/cameron/k2_ptr.py": HERE / "k2_ptr.py", "K1/chase/resolvent_hook.py": PK / "K1" / "chase" / "resolvent_hook.py",
        "K2/chase/resolvent_sp.py": PK / "K2" / "chase" / "resolvent_sp.py",
        "REPO/tests/foreman/phase_k/K0/chase/resolvent.py": REPO / "tests/foreman/phase_k/K0/chase/resolvent.py",
        "REPO/tests/foreman/phase_k/K0/wilson/train_ladder.py": REPO / "tests/foreman/phase_k/K0/wilson/train_ladder.py"}
walls = {n: round(json.loads((out_of(j) / "result.json").read_text())["wall_s"]) for j in J for n in [j["name"]] if n in landed}
doc = {
    "what": "Identity grid (test_k2_grid.py id, sha a622cd95), the registered jobs NOT landed locally, in ward order, plus one control. "
            "Read on return by test_k2_grid_kaggle.py id (sha 3ff92f9f), once, after all 18 grid runs are in runs/.",
    "landed_locally": landed, "local_run_wall_s": walls,
    "argv_paths": "argv are the exact registered argv, with this machine's absolute paths. Map SP = "
                  + str(PK.parent).replace("\\", "/") + " to the kernel's copy of the scratchpad and keep the tree below it "
                  "(phase_k/K1/cameron, phase_k/K1/chase, phase_k/K2/cameron, phase_k/K2/chase). Only paths change: every flag and value stays.",
    "hooks": {"fR_ga": {"file": "phase_k/K1/chase/resolvent_hook.py:ResolventAttention", "sha256": sha(deps["K1/chase/resolvent_hook.py"])},
              "fR_sp": {"file": "phase_k/K2/chase/resolvent_sp.py:ResolventAttention", "sha256": sha(deps["K2/chase/resolvent_sp.py"])}},
    "hook_bytes_must_not_change": "test_k2_grid_kaggle.py pins each run's recorded hook_sha256. resolvent_hook.py hard-codes "
                  "REPO = 'C:/Users/seal/Desktop/New folder (32)' (sys.path to tests/foreman/phase_k/K0/chase, and K0/wilson/train_ladder.py); "
                  "resolvent_sp.py hard-codes TL_PATH to the same K0/wilson/train_ladder.py. Editing either file changes its sha and fails "
                  "config_hw. On Linux those strings are relative paths: create the directory 'C:/Users/seal/Desktop/New folder (32)/tests/"
                  "foreman/phase_k/K0/{chase,wilson}' under the kernel's working directory (cwd) with the two K0 files, and leave the hooks byte-identical.",
    "cwd": "run every job with cwd = the copy of phase_k/K2/cameron (lockjob.py is not needed on Kaggle).",
    "dependency_sha256": {k2: sha(p) for k2, p in deps.items()},
    "bed_fingerprints": {"how": "sha256 over int64 bytes of ids, pids, target, depth, root of each bed, in order; beds from bed_kp "
                                "(far10: make_test(default_rng([31, n, k]), n), k 0..7; held-out: make_train(default_rng([21, k])), k 0..63; "
                                "train: rdepth.batch's beds for steps 1-5, train_ns 256,512,1024,2048,4096). A mismatch means a different numpy "
                                "stream: do not launch; the walk rows would fail on the beds, not the result.", **fps},
    "portability_risks": ["rdepth.py trains under torch.autocast(bfloat16) on CUDA; the local runs used an RTX 4060 (sm_89). Kaggle T4/P100 "
                          "are pre-Ampere: check bf16 support before launch; a dtype change would be a recipe change, not a path change.",
                          "The resolvent hooks capture CUDA graphs and use side streams (fs5c and the sparsemax path); --compile is barred.",
                          "Record config 'device' (rdepth does this: torch.cuda.get_device_name(0)); test_k2_grid_kaggle.py fails a run without it.",
                          "id_fR_ga_s0 took far longer locally than the ALiBi arms at train_ns up to 4096 (see local_run_wall_s)."],
    "jobs": jobs}
(HERE / "jobs_remaining.json").write_text(json.dumps(doc, indent=1))
print(len(jobs), [j["name"] for j in jobs])
print("landed", landed)
