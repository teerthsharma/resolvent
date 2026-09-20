"""Builds ceq_jepa_kaggle.ipynb next to this script. Run locally to regenerate;
not itself uploaded to Kaggle. ponytail: hand-rolled nbformat dict, stdlib json
only -- no nbformat dependency for a one-shot generator.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "ceq_jepa_kaggle.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""\
# CEQ-JEPA / DCM-1 stage-1 training -- self-contained Kaggle notebook

torch + numpy only. No pip installs, no dataset dependency: the
`ceqjepa.operator` and `ceqjepa.train` modules are inlined verbatim below
(read directly off disk by `build_notebook.py` at generation time). The
`ceqjepa` package is UNTRACKED (`git ls-files ceqjepa` is empty) -- there is
no commit that names this code's provenance. Do not cite a commit SHA as
this code's provenance; state it as an untracked working tree instead.

Runs the **stage-1** objective only (`L = L_q + 0.5*L_z`), per the DCM-1
build spec: the committor head, the constant-predictor control (C1, `q_bar`),
the collapse-floor diagnostic (CFD, `||q - q_floor||_inf`), the encoder
VAR-across-examples collapse detector, and the teleport conditioning bound
(`kappa_bound`), all logged every eval. Stage 2 (decision head / hinge loss)
is gated behind the R-1 kill and is not built here.

`build_operator` carries the teleport fix (`teleport=0.0125` by default):
every transient row sends a fraction `c` of its mass onto the causally
visible absorbing states before the boundary overwrite, giving the exact
bound `||(I-Q)^-1||_inf <= 80*(1+6e-6)` in float32 (`op.KAPPA_DESIGN_BOUND_F32`)
whenever index 0 is absorbing -- which it always is here (`torch.arange(NA)`).
That is a bound, not a promise the run stays comfortably under it: a long
soak (measured on a prior run) reached kappa 75.6954 by step 1180 and was
STILL CLIMBING, with float32 rounding able to push a further ~1e-4 past 80.
**D8 fix:** `evaluate()` therefore only WARNS when `kappa_bound` exceeds
`KAPPA_CEILING` -- it never asserts or aborts. A T4 quota should never be
spent watching a hard assert kill a healthy run mid-flight. `committor()`
and `q_floor_closed_form()` still raise `SingularTransientBlockError` on a
genuine defect (e.g. NaN logits, or an armed `kappa_max` at teleport=0);
nothing in this notebook catches that exception -- it is a bug to surface,
not a per-step event to paper over. (The kernel has no `.git` -- there is
nothing to print at runtime; the honesty fix is in this cell's own prose,
not a live git call.)

**D7 fix:** the training loop below writes an atomic checkpoint to
`/kaggle/working` every `CKPT_EVERY` steps (`save_checkpoint()`, inlined from
`train.py`: write to `<out>.tmp`, then `os.replace()` onto `<out>` -- a
9-hour Kaggle session timeout, or any kill, leaves the last periodic
checkpoint on disk under `/kaggle/working`, which Kaggle persists as kernel
output, instead of leaving nothing (the failure mode of the run this fixes:
a prior soak reached step 1840 healthy and left zero checkpoints because the
only `torch.save` ran after the loop, and the process was killed first)."""))

cells.append(code("""\
import time, json, platform
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""))

cells.append(md("## `ceqjepa.operator` (inlined verbatim)"))

cells.append(code(Path(Path(__file__).parent.parent / "operator.py").read_text()
                   .split('if __name__ == "__main__":')[0].rstrip() + "\n"))

cells.append(md("## `ceqjepa.train` building blocks (inlined, `op.` calls point at the cell above)"))

train_src = Path(Path(__file__).parent.parent / "train.py").read_text()
# Drop the module docstring (everything up to and including the closing
# `"""`), drop the import block, stop before the argparse-based main() --
# only the Bed / TinyCEQ / stage1_loss / evaluate building blocks survive.
_, _, after_docstring = train_src.partition('"""\n')
train_body = after_docstring.split("def main():")[0]
train_body = "import os\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" + \
    "EPS_Q = 1e-6  # BCE input clamp, MANDATORY per build spec: q attains 0 and 1 exactly\n" + \
    train_body.split("EPS_Q = 1e-6", 1)[1]
# op.X -> X (the functions now live in this same notebook namespace, not a package)
import re
train_body = re.sub(r"\bop\.", "", train_body)
assert "class Bed" in train_body and "class TinyCEQ" in train_body and "def evaluate" in train_body
assert "import ceqjepa" not in train_body and '"""ceqjepa/train.py' not in train_body
cells.append(code(train_body.rstrip() + "\n"))

cells.append(md("""\
## Geometry -- DESIGN (matches `train.py --geometry design`: N=256, d_enc=128,
the frozen DCM-1 N and d; other dims stay at `train.py`'s own flag defaults).

`TinyCEQ`'s per-example `delta_logits` is the D6 low-rank factorization
(rank `RANK`, default 8: two `[d_enc -> n*rank]` linears, `delta_logits =
einsum('bnr,bmr->bnm', a, b)`, cost `O(d_enc*n*rank)`), matching `train.py`'s
own `TinyCEQ`. It still does not hit the ~325,792-param target exactly at
this N, d_enc, rank -- `n_params` below reports the actual count rather than
fudging it; a smaller `RANK` lands closer, left as a knob."""))

cells.append(code("""\
N            = 256      # chart positions -- frozen DCM-1 design N
NA           = 4        # absorbing positions (train.py default, matches build spec)
D_ENC        = 128      # frozen DCM-1 design d
X_DIM        = 8        # train.py default
Z_DIM        = 4        # bed latent width, NOT known to the model (train.py default)
Z_DIM_STATE  = 6        # train.py default
G            = 0.9      # state-channel discount, frozen per build spec
RANK         = 8        # D6: delta_logits factorization rank (DCM-1 spec r=8)
BATCH_SIZE   = 64       # bumped from train.py's CPU default (16) since this runs on a T4
STEPS        = 20000
EVAL_EVERY   = 500
EVAL_N       = 512
LR           = 3e-4
SEED         = 0
KAPPA_CEILING = KAPPA_DESIGN_BOUND_F32  # D8: WARN above this, never assert/abort
CKPT_EVERY    = 500       # D7: atomic checkpoint every this many steps
CKPT_PATH     = "/kaggle/working/ceqjepa_dcm1_design.pt"  # Kaggle persists this dir as kernel output
"""))

cells.append(code("""\
torch.manual_seed(SEED)

bed = Bed(N, NA, X_DIM, Z_DIM, seed=SEED)
train_gen = torch.Generator().manual_seed(SEED)
heldout_gen = torch.Generator().manual_seed(SEED + 1_000_000)  # disjoint stream, per build spec

# q_bar: the constant predictor's whole content -- training-set channel mean (control C1).
_, _, q_pool, _ = bed.batch(train_gen, max(2048, BATCH_SIZE))
q_bar = q_pool.mean(0).to(device)
q_floor_table = bed.q_floor().to(device)

model = TinyCEQ(n=N, nA=NA, d_enc=D_ENC, x_dim=X_DIM, z_dim_state=Z_DIM_STATE, g=G,
                rank=RANK, absorbing_idx=torch.arange(NA)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

n_params = sum(p.numel() for p in model.parameters())
print(f"[ceqjepa] N={N} NA={NA} d_enc={D_ENC} x_dim={X_DIM} z_dim={Z_DIM} "
      f"z_dim_state={Z_DIM_STATE} g={G} rank={RANK} params={n_params} device={device}")
print(f"[ceqjepa] q_bar={[round(v, 4) for v in q_bar.tolist()]}")
print(f"[ceqjepa] target_params~325792 actual_params={n_params} "
      f"({'hit' if n_params == 325792 else 'did NOT hit'} the target exactly, reporting actual)")

# save_checkpoint() (inlined above from train.py) takes `args` and does vars(args) --
# a SimpleNamespace of the geometry config stands in for the argparse Namespace train.py has.
import types
_ckpt_args = types.SimpleNamespace(N=N, NA=NA, D_ENC=D_ENC, X_DIM=X_DIM, Z_DIM=Z_DIM,
                                    Z_DIM_STATE=Z_DIM_STATE, G=G, RANK=RANK,
                                    BATCH_SIZE=BATCH_SIZE, STEPS=STEPS, LR=LR, SEED=SEED)
"""))

cells.append(md("""\
## Training loop

Mirrors `train.py main()`'s loop, plus D7/D8: a periodic atomic checkpoint
to `CKPT_PATH` (`/kaggle/working`, so a 9-hour session timeout or any kill
leaves usable weights), and `kappa_bound` checked against `KAPPA_CEILING`
inside `evaluate()` as a printed WARNING, never an assert.
`SingularTransientBlockError` is not caught here: it means a genuine defect
(NaN logits, or an armed `kappa_max` at teleport=0) -- a bug to surface, not
a per-step event to paper over."""))

cells.append(code("""\
history = []
t0 = time.time()

for step in range(1, STEPS + 1):
    x, x_nx, q_star, _ = bed.batch(train_gen, BATCH_SIZE)
    x, x_nx, q_star = x.to(device), x_nx.to(device), q_star.to(device)
    model.train()
    out = model(x)
    loss, l_q, l_z = stage1_loss(out, q_star, x_nx)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % EVAL_EVERY == 0 or step == STEPS:
        model.eval()
        ev = evaluate(model, bed, q_bar, q_floor_table, heldout_gen, EVAL_N,
                       kappa_ceiling=KAPPA_CEILING)
        row = dict(step=step, train_loss=loss.item(), wall_s=time.time() - t0, **ev)
        history.append(row)
        print(f"[eval] step={step:6d} train_loss={loss.item():.4f} "
              f"S(vs const)={ev['S']:+.4f} mse_model={ev['mse_model']:.4e} "
              f"mse_bar={ev['mse_bar']:.4e} collapse_floor={ev['collapse_floor']:.4e} "
              f"var_across_examples={ev['var_across_examples']:.4e} "
              f"kappa_bound={ev['kappa_bound']:.4f}")

    # D7: periodic atomic checkpoint -- a killed run (9h Kaggle timeout, OOM, manual
    # stop) leaves the last one of these on disk under /kaggle/working instead of
    # nothing (the prior soak reached step 1840 healthy and left zero checkpoints).
    if CKPT_EVERY and step % CKPT_EVERY == 0:
        save_checkpoint(CKPT_PATH, model, _ckpt_args, n_params, q_bar, history,
                         time.time() - t0)
        print(f"[ceqjepa] wrote checkpoint {CKPT_PATH} at step={step}")

save_checkpoint(CKPT_PATH, model, _ckpt_args, n_params, q_bar, history, time.time() - t0)
print(f"[ceqjepa] wrote final checkpoint {CKPT_PATH}")
print(f"done in {time.time() - t0:.1f}s")
"""))

cells.append(md("## Manifest"))

cells.append(code("""\
manifest = dict(
    geometry=dict(N=N, NA=NA, d_enc=D_ENC, x_dim=X_DIM, z_dim=Z_DIM,
                  z_dim_state=Z_DIM_STATE, g=G, rank=RANK, batch_size=BATCH_SIZE,
                  steps=STEPS, lr=LR, seed=SEED),
    n_params=n_params,
    device=str(device),
    torch_version=torch.__version__,
    python_version=platform.python_version(),
    q_bar=q_bar.tolist(),
    wall_s=time.time() - t0,
    history=history,
    checkpoint_path=CKPT_PATH,
)
Path = __import__("pathlib").Path
Path("/kaggle/working/ceqjepa_kaggle_manifest.json").write_text(json.dumps(manifest, indent=2))
print("wrote /kaggle/working/ceqjepa_kaggle_manifest.json")
print("final S(vs const) =", history[-1]["S"] if history else None)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
