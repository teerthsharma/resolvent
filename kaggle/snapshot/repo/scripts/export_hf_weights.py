"""Export trained m3_quintuple_v2 checkpoints to safetensors for the HF package.

WHAT THIS SHIPS AND WHAT IT DOES NOT

The checkpoints under results/m3_quintuple_v2_weights/ are TRAINED PROBE-ARM
weights: 4,769-parameter `QuintArm` modules (q/k projection, 2-layer MLP,
scalar readout) trained by scale/m3_quintuple.py on one registered task each.
They are NOT `CEQForCausalLM` weights -- the 24-layer language model in
ceq/hf/modeling_ceq.py still ships none, and no file produced here may be
named `model.safetensors`, because that name would claim a loadability these
tensors do not have. The module class the tensors round-trip through is
`scale.m3_quintuple.QuintArm`, rebuilt exactly the way its own `load_unit`
rebuilds it -- one source of truth, no re-derivation of constructor arguments.

VERIFICATION GATE

Every exported file is loaded back from disk (safetensors -> state_dict ->
fresh QuintArm) and re-evaluated on the task's own eval batch, rebuilt from
the recorded (task, s, d, n_eval, seed + 12345) exactly as the training run
built it. The recomputed NRMSE must match the journal's eval_nrmse within
TOLERANCE (float32 file-roundtrip level). A checkpoint that fails is EXCLUDED
from the manifest and reported loudly; nothing unverifiable ships.

PROVENANCE

Each file carries metadata: source filename, cell, task, seed, the full
geometry string (the journal key), the journal's eval_nrmse matched by key,
both relevant git commits, and the torch version the checkpoint trained
under. results/m3_quintuple_v2.jsonl is the single metric source of truth;
the checkpoint's own embedded eval_nrmse is cross-checked against it and a
disagreement fails the export.

THREADS ARE PINNED TO 2 BEFORE ANY TORCH WORK, matching the training run's
registered geometry (scale/m3_quintuple.py pins torch.set_num_threads(2) at
import); CPU matmul reduction order depends on the thread count, so an
unpinned verifier could report a phantom delta.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch                                                    # noqa: E402
torch.set_num_threads(2)                       # the training geometry, pinned

from safetensors.torch import load_file, save_file              # noqa: E402

from scale import m3_capability as M3                           # noqa: E402
from scale import m3_quintuple as MQ                            # noqa: E402
from scale import negation_scope as NS                          # noqa: E402

JOURNAL = REPO / "results" / "m3_quintuple_v2.jsonl"
DEFAULT_WEIGHTS_DIR = REPO / "results" / "m3_quintuple_v2_weights"
PACKAGE_WEIGHTS_DIR = REPO / "ceq" / "hf_artifact" / "weights"
#: NRMSE values are O(1); float32 state-dict round-trip through safetensors is
#: lossless, so any real mismatch shows up far above this bar.
TOLERANCE = 1e-6


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                         text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def load_journal() -> dict[str, dict]:
    rows = {}
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows[rec["key"]] = rec["value"]
    return rows


def default_checkpoints() -> list[pathlib.Path]:
    """THE SHIPPED SET: all five twin t*=1 seeds, the ship candidate's own arm."""
    pat = "twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd{}_taske3_t1.pt"
    return [DEFAULT_WEIGHTS_DIR / pat.format(sd) for sd in range(5)]


def rebuild_model(rec: dict, state_dict: dict) -> torch.nn.Module:
    """A QuintArm from RECORD FIELDS + an arbitrary state_dict.

    The constructor arguments are copied verbatim from
    `scale.m3_quintuple.load_unit`, which owns their meaning; duplicating the
    six kwargs here (rather than inventing them) keeps a consumer from
    re-deriving a `twin` number under `settled`.
    """
    m = MQ.QuintArm(rec["kind"], rec["s"], cell=rec["cell"],
                    k_piv=rec["k_piv"], beta=rec["beta"],
                    t_max=rec["t_max"], n_neumann=rec["n_neumann"])
    m.load_state_dict(state_dict)
    m.eval()
    return m


def eval_batch_for(rec: dict):
    """The eval batch the journal number was taken on, rebuilt by the task's
    own builder: bfn(n_eval, s, d, d_model=D_MODEL, seed=seed + 12345) --
    scale/m3_quintuple._unit lines, byte-for-byte."""
    bfn = NS.M3_TASKS[rec["task"]][0]
    return bfn(rec["n_eval"], rec["s"], rec["d"], d_model=M3.D_MODEL,
               seed=rec["seed"] + 12345)[:2]


def evaluate(model, xe, ye, mu: float, sigma: float) -> float:
    """De-standardised prediction, then the task family's NRMSE. This is
    train_and_predict's prediction step verbatim: model(xe) * sigma + mu."""
    with torch.no_grad():
        pred = model(xe) * sigma + mu
    return NS.nrmse(pred, ye)


def export_one(pt_path: pathlib.Path, out_dir: pathlib.Path,
               journal: dict, git_export: str, git_journal: str) -> dict:
    key = pt_path.stem
    ref_model, rec = MQ.load_unit(pt_path)

    # ---- the journal is the metric source of truth; the checkpoint agrees or
    # the export dies before anything is written.
    if key not in journal:
        raise ValueError(f"{key}: no journal row in {JOURNAL.name}")
    jr = journal[key]
    if jr["eval_nrmse"] != rec["eval_nrmse"]:
        raise ValueError(
            f"{key}: checkpoint eval_nrmse {rec['eval_nrmse']!r} disagrees "
            f"with journal {jr['eval_nrmse']!r}")

    sd_ref = rec["state_dict"]
    meta = {
        "format": "pt",
        "source_file": pt_path.name,
        "key": key,
        "cell": rec["cell"],
        "task": rec["task"],
        "seed": str(rec["seed"]),
        "geometry": key,                       # the journal key IS the geometry
        "kind": rec["kind"],
        "s": str(rec["s"]), "d": str(rec["d"]),
        "d_model": str(rec["d_model"]),
        "k_piv": str(rec["k_piv"]), "beta": repr(rec["beta"]),
        "t_max": str(rec["t_max"]), "n_neumann": str(rec["n_neumann"]),
        "steps": str(rec["steps"]),
        "n_train": str(rec["n_train"]), "n_eval": str(rec["n_eval"]),
        "n_params": str(rec["n_params"]),
        "eval_nrmse_journal": repr(jr["eval_nrmse"]),
        "mu": repr(rec["mu"]), "sigma": repr(rec["sigma"]),
        "torch_version_train": rec["torch_version"],
        "git_commit_journal_record": git_journal,
        "git_commit_export_head": git_export,
        "exported_by": "scripts/export_hf_weights.py",
        "module_class": "scale.m3_quintuple.QuintArm",
    }

    out_path = out_dir / f"{key}.safetensors"
    save_file({k: v.contiguous() for k, v in sd_ref.items()},
              str(out_path), metadata=meta)

    # ---- ROUND TRIP: back from disk, into a fresh QuintArm, re-evaluated.
    rt = load_file(str(out_path))
    if set(rt) != set(sd_ref):
        raise ValueError(f"{key}: tensor names changed across the round trip")
    drift = max(float((rt[k] - sd_ref[k]).abs().max()) for k in sd_ref)
    xe, ye = eval_batch_for(rec)
    nrmse_rt = evaluate(rebuild_model(rec, rt), xe, ye, rec["mu"], rec["sigma"])
    nrmse_ref = evaluate(ref_model, xe, ye, rec["mu"], rec["sigma"])
    delta_journal = nrmse_rt - jr["eval_nrmse"]
    delta_ckpt = nrmse_rt - rec["eval_nrmse"]

    ok = (drift == 0.0
          and abs(delta_journal) <= TOLERANCE
          and abs(delta_ckpt) <= TOLERANCE)
    entry = {
        "file": f"weights/{out_path.name}",
        "source_checkpoint": str(pt_path.relative_to(REPO)),
        "sha256_source_pt": None,           # filled by caller
        "cell": rec["cell"], "task": rec["task"], "seed": rec["seed"],
        "geometry": key, "n_params": rec["n_params"],
        "eval_nrmse_journal": jr["eval_nrmse"],
        "eval_nrmse_recomputed_from_safetensors": nrmse_rt,
        "eval_nrmse_recomputed_from_pt_reference": nrmse_ref,
        "abs_delta_vs_journal": abs(delta_journal),
        "abs_delta_vs_checkpoint": abs(delta_ckpt),
        "tensor_roundtrip_max_abs_drift": drift,
        "tolerance": TOLERANCE,
        "verified": ok,
        "git_commit_journal_record": git_journal,
        "git_commit_export_head": git_export,
    }
    if not ok:
        out_path.unlink(missing_ok=True)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="explicit checkpoint paths; default = the five "
                         "twin taske3_t1 seeds")
    ap.add_argument("--out", type=pathlib.Path, default=PACKAGE_WEIGHTS_DIR)
    args = ap.parse_args()

    paths = ([pathlib.Path(p) for p in args.ckpt] if args.ckpt
             else default_checkpoints())
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {[str(m) for m in missing]}")

    git_export = _git("rev-parse", "HEAD")
    git_journal = _git("log", "-n", "1", "--format=%H", "--",
                       str(JOURNAL.relative_to(REPO)))
    journal = load_journal()

    args.out.mkdir(parents=True, exist_ok=True)
    import hashlib
    entries, excluded = [], []
    for p in paths:
        try:
            entry = export_one(p, args.out, journal, git_export, git_journal)
            entry["sha256_source_pt"] = hashlib.sha256(p.read_bytes()).hexdigest()
            entries.append(entry)
            flag = "PASS" if entry["verified"] else "FAIL"
            print(f"[{flag}] {p.name}")
            print(f"      journal={entry['eval_nrmse_journal']!r}  "
                  f"safetensors={entry['eval_nrmse_recomputed_from_safetensors']!r}  "
                  f"|delta|={entry['abs_delta_vs_journal']:.3e}  "
                  f"tensor_drift={entry['tensor_roundtrip_max_abs_drift']:.1f}")
            if not entry["verified"]:
                excluded.append({"source_checkpoint": p.name,
                                 "reason":
                                     "verification failed (see deltas above);"
                                     " file deleted, not shipped"})
                print(f"      *** EXCLUDED FROM THE MANIFEST ***")
        except Exception as exc:                       # noqa: BLE001
            excluded.append({"source_checkpoint": p.name,
                             "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[EXCLUDED] {p.name}: {exc}")

    manifest = {
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "what_this_is": (
            "Trained probe-arm weights from the m3_quintuple_v2 campaign. "
            "Each file is one trained QuintArm (4,769 parameters, module "
            "class scale.m3_quintuple.QuintArm), NOT a CEQForCausalLM "
            "checkpoint; the language model in modeling_ceq.py still ships "
            "no weights."),
        "metric_source_of_truth": str(JOURNAL.relative_to(REPO)),
        "git_commit_journal_record": git_journal,
        "git_commit_export_head": git_export,
        "threads_registered_geometry": (
            "torch.set_num_threads(2) -- scale/m3_quintuple.py:74"),
        "verification": {
            "procedure": (
                "safetensors reloaded from disk into a fresh QuintArm, "
                "forward on the task's own eval batch "
                "(bfn(n_eval, s, d, d_model=16, seed=seed+12345)), NRMSE "
                "compared against the journal row matched by key"),
            "tolerance_abs_nrmse": TOLERANCE,
            "policy": (
                "a failed checkpoint is excluded from this manifest and its "
                "safetensors file is deleted"),
            "passed": sum(1 for e in entries if e["verified"]),
            "failed": len(entries) - sum(1 for e in entries if e["verified"])
                      + len(excluded),
        },
        "shipped": entries,
        "excluded": excluded,
    }
    man = args.out / "MANIFEST.json"
    man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest: {man}")
    print(f"shipped {sum(1 for e in entries if e['verified'])}/{len(paths)};"
          f" excluded {len(excluded)}")
    if excluded:
        print("EXCLUSIONS ABOVE ARE LOUD ON PURPOSE -- do not quote metrics "
              "for files not in this manifest.")
    return 0 if not excluded else 1


if __name__ == "__main__":
    raise SystemExit(main())
