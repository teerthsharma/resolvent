"""step0_rowmass.py -- Q2 row-mass surgery, step (1) only: find the arm (f)
checkpoint and print its fixed structure before any scored quantity (row
mass, exact-zero fraction, W5) is computed.

CPU only. No training. Reads results/ and ceq/ only.

Run: python step0_rowmass.py
"""
import io
import json
import os

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
ARM_F_DIRS = [os.path.join(SCRATCH, "q2_ckpt_hardconcrete_seed{}".format(s)) for s in range(5)]

# Weight-file extensions a real checkpoint would carry.
WEIGHT_EXTS = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")


def fixed_structure():
    """(1) Print path, shape, step count, arm -- for every seed -- before
    anything else runs. This is metadata read off run_record.json, not a
    scored quantity."""
    rows = []
    for d in ARM_F_DIRS:
        rec_path = os.path.join(d, "run_record.json")
        print("=== {} ===".format(d))
        if not os.path.isdir(d):
            print("  MISSING DIRECTORY")
            rows.append(dict(dir=d, exists=False))
            continue
        entries = os.listdir(d)
        weight_files = [f for f in entries if f.lower().endswith(WEIGHT_EXTS)]
        print("  files:", entries)
        print("  weight-shaped files (.pt/.pth/.bin/.safetensors/.ckpt):", weight_files)
        if not os.path.isfile(rec_path):
            print("  NO run_record.json")
            rows.append(dict(dir=d, exists=True, has_run_record=False, weight_files=weight_files))
            continue
        with io.open(rec_path, encoding="utf-8") as fh:
            rec = json.load(fh)
        print("  arm (operator field) :", rec.get("operator"))
        print("  steps                :", rec.get("steps"))
        print("  seed                 :", rec.get("seed"))
        print("  hidden/layers/heads/d_head/seq/batch :",
              rec.get("hidden_size"), rec.get("n_layers"), rec.get("n_heads"),
              rec.get("d_head"), rec.get("seq"), rec.get("batch"))
        print("  n_params             :", rec.get("n_params"))
        print("  final_eval_loss      :", rec.get("final_eval_loss"))
        print("  run_record.json keys :", sorted(rec.keys()))
        rows.append(dict(dir=d, exists=True, has_run_record=True,
                          weight_files=weight_files, operator=rec.get("operator"),
                          steps=rec.get("steps"), seed=rec.get("seed"),
                          n_params=rec.get("n_params"), keys=sorted(rec.keys())))
    return rows


def trace_save_path():
    """Show, from the source, that q2_certificate.py's train_arm_f route
    never calls torch.save -- so the absence of weight files above is not
    an accident of listing, it is how the row that made these directories
    was written."""
    q2 = os.path.join(REPO, "tests", "foreman", "q2", "q2_certificate.py")
    r3 = os.path.join(REPO, "tests", "foreman", "eval", "r3_eval.py")
    print("\n=== save-path trace ===")
    print("q2_certificate.py:train_arm_f calls RE.train_with_eval(out_dir=..., ...)")
    print("  ({})".format(q2))
    print("tests/foreman/eval/r3_eval.py:train_with_eval docstring says:")
    print('  "`ceq.hf.train.train()`, minus checkpoint resume/save_every ' \
          '(not needed for these short scratchpad runs...)"')
    print("  ({})".format(r3))
    with io.open(r3, encoding="utf-8") as fh:
        src = fh.read()
    has_save = "torch.save" in src or "state_dict()" in src
    print("  'torch.save' or 'state_dict()' found in r3_eval.py:", has_save)
    print("  -> only json.dump(record, ...) is written to run_record.json; ")
    print("     the trained nn.Module is discarded when train_with_eval returns.")
    return has_save


def main():
    rows = fixed_structure()
    has_save = trace_save_path()

    any_weights = any(r.get("weight_files") for r in rows)
    all_have_records = all(r.get("has_run_record") for r in rows if r.get("exists"))

    print("\n=== VERDICT ===")
    if any_weights or has_save:
        print("Weight files or a save path were found -- proceed to step (2).")
        return 0

    print("NO arm (f) MODEL CHECKPOINT SURVIVED.")
    print("All 5 q2_ckpt_hardconcrete_seed{0..4} directories contain only")
    print("run_record.json (training-curve metadata: losses, grad_norms,")
    print("eval_losses, final_eval_loss, param/shape counts). None contains")
    print("a state_dict, .pt/.pth/.bin/.safetensors file, or any other")
    print("serialized tensor. r3_eval.train_with_eval() -- the function that")
    print("produced these directories -- never calls torch.save or")
    print("model.state_dict(); its own docstring says it is train.train()")
    print("'minus checkpoint resume/save_every'. The trained nn.Module for")
    print("every arm (f) seed was discarded in-process when training ended.")
    print("")
    print("Steps (2)-(6) (trained beta, row mass, exact-zero fraction, W5,")
    print("the pre-registered prediction) all require forward-passing a")
    print("live trained model on a held-out batch and reading G/theta_head/")
    print("beta out of it. That data does not exist on disk for arm (f) at")
    print("any seed. This row is a checkpoint READ, training is out of")
    print("scope and a rebuild would not be reading the checkpoint the row")
    print("was asked to read. STOPPING HERE per instruction: 'the row cannot")
    print("proceed on a rebuild and must not fake one.'")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
