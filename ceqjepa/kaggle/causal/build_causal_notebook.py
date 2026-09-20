"""Builds ceq_jepa_causal_t4.ipynb (and, with --extract, a flat .py of every
code cell so the body can be run on CPU before it costs a T4 minute).

DESIGN NOTE, and the reason this does not look like build_notebook.py: that
generator INLINED train.py's body into a cell and regex-rewrote `op.` -> ``.
This one WRITES THE PACKAGE VERBATIM to /kaggle/working/pkg/ and puts that on
sys.path, so every module imports under its real name. No import surgery, no
rewrite that can silently drift from the file it copied, and the kernel runs
byte-identical source to the working tree.

REWIRED 2026-09-09. The first version of this file carried its OWN training
loop and imported four symbols from ceqjepa/train.py, so none of train.py's
guards reached the GPU and a 4 h 43 m T4 run shipped its WORST checkpoint.
train_seed() below now imports and CALLS train.py's own heldout_split(),
evaluate(), diverging() and save_checkpoint(); it keeps a loop body only
because train.py's stop/keep/shout logic lives inline in main(), behind
argparse, and main() itself cannot run on a GPU (it never moves the batch --
see THE DEVICE TRAP). The bed default moved to chess_policy at T=0.25 with the
uniform bed kept as a named control arm.

ponytail: hand-rolled nbformat dict + json.dumps of a {path: source} map.
stdlib only, no nbformat dependency for a one-shot generator.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent            # .../New folder (32)
OUT = HERE / "ceq_jepa_causal_guarded_t4.ipynb"   # NEW file; the old kernel keeps its own

# Everything the kernel imports. ceqjepa/beds/__init__.py imports english and
# markets, so they ship even though this kernel never builds those beds; and
# ceqjepa/beds/chess.py imports ceq.kdata at module level, so ceq ships too.
PKG_GLOBS = ["ceqjepa/*.py", "ceqjepa/beds/*.py", "ceq/__init__.py", "ceq/kdata.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def collect_files():
    files = {}
    for pat in PKG_GLOBS:
        for p in sorted(REPO.glob(pat)):
            files[p.relative_to(REPO).as_posix()] = p.read_text(encoding="utf-8")
    assert "ceqjepa/train.py" in files and "ceqjepa/beds/chess_do.py" in files
    assert "ceq/kdata.py" in files, "ceqjepa/beds/chess.py imports ceq.kdata"
    return files


# ---------------------------------------------------------------------------
CELLS = []

CELLS.append(md(r"""# CEQ-JEPA / DCM-1 — THE CAUSAL ARM, GUARDED (T4)

`--bed chess_policy` at `T = 0.25` with the uniform bed kept as a named control
arm, the interventional term `L_do` ON, `lambda_z = 0`, rank 16, `n = 256`.
Trains three seeds on the default arm and one on the control, and then asks the
one question the architecture exists to answer, with a paired bootstrap SE
attached to every number:

> does the operator's `do(a)` read predict the outcome of a FORCED move better
> than the observational read does, by more than the paired bootstrap SE?

## Read this before reading any number below

A CPU run at n=16 / 800 train positions already answered that question
**yes, 3/3 seeds, up to −7.13 SE — and the answer was void**, for three
measured reasons that this kernel re-measures at scale rather than papering
over:

1. **the move-permutation ablation.** Destroying the move→position pairing
   moved `PPL_do` by `−0.0360 ± 0.0469` (−0.77 SE). The do-read's whole
   advantage survived randomising *which* intervention was performed.
2. **the bed's oracle headroom.** An oracle handed the true interventional
   distribution by brute force scored `gap = +0.0935 ± 0.0464` — *positive*,
   i.e. knowing the intervention does not help. `TV(p_do, p_obs) = 0.1501`
   against a rollout-noise floor `TV(p_do, p_do2) = 0.1468`: **excess over
   noise +0.0033.** Forcing one ply of a uniformly-random self-play game barely
   moves the outcome distribution.
3. **the observational arm was worse than chance.** `PPL_obs` degraded
   `2.96 → 8.35` (chance 4.00) while train `L_q` fell `1.74 → 0.05`. The
   head-to-head "win" was the bar collapsing, not the do-read improving.

So the controls are not decoration here. Both the move-permutation ablation
and the oracle headroom run **in this same kernel, on this same bed**, and
their numbers are printed next to the headline.

## What the previous kernel measured, and what changed because of it

`ceq-jepa-dcm-1-causal-arm-t4` ran 4 h 43 m, 1,285,771 parameters, 3 seeds x
12,000 steps, and **wasted 5 of every 6 steps with nothing able to see it**.
Its own log: held-out `PPL_obs` `3.1796` at step 2000 → `10.3872` at step
12000, **rising in 15 of 15 intervals across three seeds**, while training
`L_q` fell to `0.0443` — 257 parameters per training label. The checkpoint it
shipped was step 12000. Four changes, all of them `ceqjepa/train.py`'s own code
called from here rather than a second copy of it:

1. **the split is on GAMES** (`train.heldout_split`), asserted at run time —
   positions from one game share an outcome, so a position split leaks the label;
2. **early stopping is ON** (patience 5 evals) and `--out` holds the **BEST**
   checkpoint, which is loaded back before anything is scored;
3. **the `[MEMORISATION]` guard** (`train.diverging`, `DIVERGENCE_REL = 0.05`,
   K = 3) — replayed on that series it fires at step 8000, about **1 h 40 m of
   T4 time before the run actually ended**;
4. **the sharpness margin prints on every eval line**. On that run it was
   `0.1196 − (1.3967 + 0.0025) = −1.2796`, three times more negative than a
   deliberately temperature-sabotaged control, and no number printed during the
   run could show it.

Scale attacks (3) directly. It cannot fix (2), which is a fact about the bed —
which is why the bed itself changed: `chess_policy` at `T = 0.25` carries
`I(X;Y) = 0.3084` nats against the uniform bed's `0.1363`, and the uniform bed
still runs, as the control arm, beside every number.

## `L_do` at init is exactly 2.249340 for any model

With `L0 = 0`, the LoRA-zeroed delta and the zero-init move head, `P` is the
uniform causal chain and all four absorbing sets are symmetric, so
`q(do a)[i] = 0.25` in every coordinate and the BCE against any target summing
to 1 is `−(ln 0.25 + 3 ln 0.75) = 2.2493`. **The starting value of `L_do`
carries no information; only its descent does.**

## Timeout and quota safety

`train_seed()` writes an atomic checkpoint to `/kaggle/working` every
`CKPT_EVERY` steps and resumes from it, and the bed itself is cached there
after it is built. A 9-hour session timeout or a quota pause therefore costs
**one checkpoint interval plus zero bed rebuilds**, not the run. There is also
a hard `DEADLINE_S`: training stops early and the evaluation still runs, so
the kernel always reports what it actually did rather than dying mid-flight.

## The device trap, which is the reason this notebook exists as a new file

`ChessDoBed.batch_do()` builds **seven** CPU tensors (`x, x_nx, q_star, v_idx,
moves, do_tgt, do_mask`) and the model is moved with `.to(device)`. Without
moving the batch too, the first forward raises a CPU/CUDA mismatch **at step
1**, and no CPU smoke test can see it. The training cell moves the batch
explicitly, on its own line, marked `# THE DEVICE TRAP`."""))

CELLS.append(md("## Cell 1 — environment"))

CELLS.append(code(r'''import os, sys, json, time, math, random, subprocess, platform
T_START = time.time()

import numpy as np
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("device:", DEVICE, "| python", platform.python_version())

# python-chess: preinstalled in the Kaggle image on every run this notebook has
# seen, but the bed is worthless without it, so fall back to pip rather than
# discovering the ImportError 70 minutes into a bed build.
try:
    import chess
    print("python-chess", chess.__version__, "(preinstalled)")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-chess"])
    import chess
    print("python-chess", chess.__version__, "(pip installed)")

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("CEQ_WORK", "./ceq_work")
os.makedirs(WORK, exist_ok=True)
SMOKE = os.environ.get("CEQ_SMOKE") == "1"     # local CPU crash test; never set on Kaggle
print("WORK =", WORK, "| SMOKE =", SMOKE)
'''))

CELLS.append(md("""## Cell 2 — the `ceqjepa` + `ceq` package, written verbatim

Not inlined and not rewritten: the files below are byte-identical to the
working tree and are written to disk, then imported under their real names.
The package is **untracked** (`git ls-files ceqjepa` is empty) — do not cite a
commit SHA as this code's provenance; it is an untracked working tree."""))

CELLS.append(code("PKG_FILES = __FILES__\n" + r'''
PKG_ROOT = os.path.join(WORK, "pkg")
for rel, src in PKG_FILES.items():
    dst = os.path.join(PKG_ROOT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(src)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import ceqjepa.causal_eval as ce
# THE BED. chess_policy is a drop-in for chess_do (same InterventionSample, same
# fen_to_vec, same signature plus a trailing `temperature`); `UNIFORM` reproduces
# chess_do's rollout policy verbatim, which is what the control arm runs.
from ceqjepa.beds.chess_policy import (OUTCOME_NAMES, N_OUTCOMES, UNIFORM,
                                       build_intervention_dataset, _rollout_outcome)
# THE GUARDS, IMPORTED, NOT REIMPLEMENTED. Everything that decides what is held
# out, what is kept, when to stop and when to shout comes from train.py itself:
#   heldout_split    game-disjoint split, ASSERTED at run time (assert_disjoint)
#   evaluate         the eval row, sharpness margin included (sharpness.decompose)
#   diverging        the MEMORISATION vote; DIVERGENCE_REL is its 5% threshold
#   save_checkpoint  the atomic writer, used for BOTH the best and the last file
from ceqjepa.train import (ChessDoBed, TinyCEQ, compute_loss, topo_blocks, draw,
                           heldout_split, evaluate, diverging, save_checkpoint,
                           DIVERGENCE_REL)

print("wrote %d files under %s" % (len(PKG_FILES), PKG_ROOT))
print("sha1 of the package, for the manifest:")
import hashlib
PKG_SHA = hashlib.sha1("".join(PKG_FILES[k] for k in sorted(PKG_FILES)).encode()).hexdigest()
print("  ", PKG_SHA)
'''))

CELLS.append(md(r"""## Cell 3 — FROZEN budget

Fixed here, before any number from this kernel was read. `n = 256` is the
frozen DCM-1 design chart width and is 16x the n=16 the CPU causal test ran
at; `rank = 16` is above the task's floor of 12. `lambda_z = 0` because the
encoder-side auxiliary is MEASURED adverse for the committor read (L_q alone
+0.5056 vs L_q+0.5·L_z +0.4070, 3/3 seeds). `lambda_topo = 0` because
`--topo-blocks phase` is provably identically 0.0 inside a single-phase run.

`max_plies = 400`, not `chess_do`'s own default of 80: at 80, **93% of
positions absorb into SINK**, `q_bar` is nearly one-hot and the eval metric
reads `nan`. Measured outcome counts (white, draw, black, sink) at 60 games:
`max_plies=80 -> [3,0,1,56]`, `200 -> [2,0,2,56]`, `400 -> [6,34,3,17]`.

The eval bed is built at `R=1, m=1` so its do-label is a **one-hot**: the
realized outcome of ONE forced playout, an honest single draw from the true
post-intervention distribution, index-aligned with the observational arm."""))

CELLS.append(code(r'''FROZEN = dict(
    # --- bed. THE ARMS: (name, temperature, model seeds). temperature None means
    # chess_policy.UNIFORM = math.inf, i.e. chess_do's own uniform-random rollout
    # policy reproduced verbatim -- the CONTROL, the "what zero information scores"
    # number every headline below is printed beside. None rather than inf because
    # inf is not JSON and the manifest has to round-trip. ---
    arms=(("policy_T0.25", 0.25, (0, 1, 2)), ("uniform_control", None, (0,))),
    train_games=5000, train_seed=0, train_m=8, train_R=4, max_plies=400,
    eval_games=1500, eval_seed=12345, eval_m=1, eval_R=1,
    # --- the held-out split, from ceqjepa.train.heldout_split: GROUPS (games) ---
    heldout_frac=0.2, split_seed=1234,
    # --- model: THE SCALE-UP. n 16 -> 256, d_enc 16 -> 128, rank 12 -> 16 ---
    n=256, d_enc=128, rank=16, z_dim_state=6, g=0.9,
    # --- objective ---
    lambda_do=1.0, lambda_z=0.0, lambda_topo=0.0, topo_blocks="phase",
    # --- optimisation. steps is now a CAP, not a plan: early stopping decides. ---
    steps=12000, batch_size=32, lr=3e-4,
    eval_every=250, eval_n=256, early_stop_patience=5, divergence_k=3,
    # --- scoring ---
    n_boot=2000, boot_seed=7,
    # --- controls that decide whether the headline means anything ---
    oracle_positions=600, oracle_R=8, oracle_seed=999, perm_seed=4242,
    # --- timeout safety ---
    ckpt_every=500, trace_every=2000, eval_chunk=256, deadline_s=8.0 * 3600,
)
if SMOKE:
    FROZEN.update(arms=(("policy_T0.25", 0.25, (0,)), ("uniform_control", None, (0,))),
                  train_games=12, eval_games=10, max_plies=120,
                  n=24, d_enc=16, rank=6,
                  steps=int(os.environ.get("CEQ_STEPS", 6)), batch_size=4,
                  n_boot=50, eval_every=2, eval_n=4,
                  # the two guards are the point of this kernel, so the CPU smoke can
                  # turn them down far enough to make them FIRE on a 40-step run
                  early_stop_patience=int(os.environ.get("CEQ_PATIENCE", 5)),
                  divergence_k=int(os.environ.get("CEQ_DIVK", 3)),
                  oracle_positions=4, oracle_R=2, ckpt_every=3, trace_every=3,
                  eval_chunk=4,
                  deadline_s=900.0)

print("=== FROZEN (fixed before any number from this kernel was read) ===")
for k, v in FROZEN.items():
    print("  %-16s = %s" % (k, v))
'''))

CELLS.append(md(r"""## Cell 4 — the two bed arms, and the GAME-DISJOINT split

**The default bed is now `chess_policy` at `T = 0.25`, and `uniform` is a named
control arm rather than the only arm.** `chess_policy` is a drop-in for
`chess_do` — same `InterventionSample`, same `fen_to_vec`, one extra
`temperature` — that replaces the rollout policy `rng.choice(legal_moves)` with
a softmax over a cheap material-and-mobility score. MEASURED by that module's
own `demo()`, 120 positions x 20 rollouts, 400-permutation shuffle null:

| arm | `I(X;Y)` | vs null | labels |
|---|---|---|---|
| `uniform` (the old bed, the CONTROL) | 0.1363 nats | 28.3 sigma | 92% draw-or-ply-cap |
| `T = 0.25` (the DEFAULT) | 0.3084 nats | 54.9 sigma | 79% decisive, SINK 35.2% -> 5.0% |

`T = 0.25` is the floor of the swept range, not an extrapolation: below it the
policy goes near-deterministic, games repeat into draws and `I(X;Y)` FALLS
(`T=0.05 -> 0.2551`). The control arm is not decoration — a headline that the
uniform arm reproduces is a headline about the estimator, not the architecture.

**The split is `ceqjepa.train.heldout_split`, on GAMES.** Every position of a
self-play game carries that game's single outcome, so a position-level split
hands the evaluator a label it trained on; `assert_disjoint` re-checks the
result at run time and raises, naming the offending groups. The transposition
overlap a game split cannot remove is printed, not silenced.

The train bed is the expensive object, so each arm is `torch.save`d the moment
it exists and the cache key includes the temperature. A resumed session reloads
it instead of rebuilding a dataset a fixed seed makes byte-identical anyway."""))

CELLS.append(code(r'''BED_ARMS = {}          # name -> the whole arm, built once, reused on rebind

def _temp(t):
    """None in FROZEN means chess_policy.UNIFORM (= math.inf), which is not JSON."""
    return UNIFORM if t is None else t

def _bed_key(temperature):
    return tuple(FROZEN[k] for k in ("train_games", "train_seed", "train_m", "train_R",
                                     "max_plies", "eval_games", "eval_seed",
                                     "eval_m", "eval_R")) + (repr(temperature),)

def build_or_load_beds(name, temperature):
    path = os.path.join(WORK, "beds_causal_%s.pt" % name)
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        if d.get("frozen_key") == _bed_key(temperature):
            print("[bed %s] RESUMED from cache %s (%d train, %d eval)"
                  % (name, path, len(d["tr"]), len(d["ev"])), flush=True)
            return d["tr"], d["ev"]
        print("[bed %s] cache present but its budget key differs; rebuilding" % name, flush=True)
    t0 = time.time()
    tr = build_intervention_dataset(n_games=FROZEN["train_games"], seed=FROZEN["train_seed"],
                                    m_candidates=FROZEN["train_m"], R=FROZEN["train_R"],
                                    max_plies=FROZEN["max_plies"], temperature=temperature)
    print("[bed %s] train: %d paired positions in %.0fs" % (name, len(tr), time.time() - t0),
          flush=True)
    t0 = time.time()
    ev = build_intervention_dataset(n_games=FROZEN["eval_games"], seed=FROZEN["eval_seed"],
                                    m_candidates=FROZEN["eval_m"], R=FROZEN["eval_R"],
                                    max_plies=FROZEN["max_plies"], temperature=temperature)
    print("[bed %s] eval : %d paired positions in %.0fs" % (name, len(ev), time.time() - t0),
          flush=True)
    tmp = path + ".tmp"
    torch.save(dict(tr=tr, ev=ev, frozen_key=_bed_key(temperature)), tmp)
    os.replace(tmp, path)              # atomic: a kill mid-write cannot leave a half bed
    return tr, ev

def build_arm(name, temperature):
    if name in BED_ARMS:
        return BED_ARMS[name]
    T = _temp(temperature)
    print("\n=== BED ARM %s (temperature=%s) ===" % (name, T), flush=True)
    tr_s, ev_s = build_or_load_beds(name, T)
    full = ChessDoBed(tr_s, FROZEN["train_m"])
    # THE SPLIT, from train.py. Partitions GAMES and asserts the result; raises if
    # a single group lands on both sides. The eval bed below is a different corpus
    # (its own game seed) and is what the CAUSAL question is scored on; this split
    # is what early stopping, the best checkpoint and the memorisation guard read.
    fit, ho, note = heldout_split(full, FROZEN["heldout_frac"], FROZEN["split_seed"])
    print("[split %s] %s" % (name, note), flush=True)
    ev = ChessDoBed(ev_s, FROZEN["eval_m"])
    # the eval do-label MUST be a one-hot; everything downstream reads its argmax
    # as "what actually happened" when the move was forced.
    assert ev.do_tgt.shape[1] == 1
    _s = ev.do_tgt[:, 0]
    assert bool(((_s == 0) | (_s == 1)).all()), "eval do-label is not one-hot; R must be 1"
    k_o, k_d = ev.q_star.argmax(-1), ev.do_tgt[:, 0].argmax(-1)
    k_fit = fit.q_star.argmax(-1)      # the marginal control sees TRAINING rows only
    a = dict(name=name, temperature=T, tr_samples=tr_s, ev_samples=ev_s,
             full=full, fit=fit, ho=ho, eval_bed=ev, K=ev.nA,
             k_obs=k_o, k_do=k_d, marg=ce.marginal_predictor(k_fit, ev.nA),
             eval_x=ev.x.to(DEVICE), eval_moves=ev.moves.to(DEVICE))
    print("[bed %s] K = %d absorbing sets %s; chance_level = %.4f (1.00 perfect, %.2f = no "
          "information)" % (name, ev.nA, OUTCOME_NAMES, ce.chance_level(ev.nA),
                            ce.chance_level(ev.nA)))
    print("[bed %s] train candidate pad rate = %.4f (positions with < m legal moves, "
          "padded+masked, NOT dropped)" % (name, fit.pad_rate))
    print("[bed %s] fit outcome counts %s (held-out %s)"
          % (name, torch.bincount(k_fit, minlength=ev.nA).tolist(),
             torch.bincount(ho.q_star.argmax(-1), minlength=ev.nA).tolist()))
    print("[bed %s] eval  obs outcome counts %s" % (name, torch.bincount(k_o, minlength=ev.nA).tolist()))
    print("[bed %s] eval  do  outcome counts %s" % (name, torch.bincount(k_d, minlength=ev.nA).tolist()))
    _fip = float(np.mean([s.candidate_ucis[0] == s.obs_uci for s in ev_s]))
    print("[bed %s] forced move == played move in %.1f%% of eval positions (a uniform draw from "
          "the legal moves, NOT filtered out)" % (name, 100 * _fip))
    print("[bed %s] forcing that move CHANGED the realized outcome in %.1f%% of eval positions "
          "-- and the oracle cell below says how much of that is rollout NOISE"
          % (name, 100 * float((k_o != k_d).float().mean())), flush=True)
    BED_ARMS[name] = a
    return a

ARM = None

def use_arm(name, temperature):
    """Rebind the names every cell below reads. The scoring cells are written
    against ONE arm at a time, so this is what makes the uniform control the SAME
    code path rather than a second copy of it."""
    global ARM, fit_bed, ho_bed, eval_bed, tr_samples, ev_samples
    global K, k_obs, k_do, marg, EVAL_X, EVAL_MOVES
    ARM = build_arm(name, temperature)
    fit_bed, ho_bed, eval_bed = ARM["fit"], ARM["ho"], ARM["eval_bed"]
    tr_samples, ev_samples = ARM["tr_samples"], ARM["ev_samples"]
    K, k_obs, k_do, marg = ARM["K"], ARM["k_obs"], ARM["k_do"], ARM["marg"]
    EVAL_X, EVAL_MOVES = ARM["eval_x"], ARM["eval_moves"]
    return ARM

PRIMARY = FROZEN["arms"][0]
use_arm(PRIMARY[0], PRIMARY[1])        # build the default arm now, not 70 minutes in
'''))

CELLS.append(md(r"""## Cell 5 — training: train.py's guards, the device trap, the resume

**What this loop no longer owns.** The held-out split, the eval row, the
memorisation vote and the checkpoint writer are `ceqjepa.train`'s, called here:

| what | where it comes from |
|---|---|
| game-disjoint split, asserted at run time | `train.heldout_split` / `assert_disjoint` |
| the eval row, **sharpness margin included** | `train.evaluate` -> `sharpness.decompose` |
| the memorisation vote (`DIVERGENCE_REL = 0.05`) | `train.diverging` |
| atomic checkpoint, best AND last | `train.save_checkpoint` |
| the batch draw | `train.draw` |

**What it still owns, and why.** train.py's stop/keep/shout *sequencing* — the
stale counter, the patience test, the `K`-eval streak, the `[SUMMARY]` line —
lives inline inside `main()`, behind `argparse`, and `main()` is not importable
as a loop. It also cannot run here at all: `main()` moves the model with
`.to(device)` and hands it a **CPU** batch straight from `draw()`, so
`--device cuda` raises at step 1. train.py is off-limits in this task, so the
sequencing is re-expressed below around the imported predicates, and the eval
line is printed in train.py's exact format.

Consequences, stated rather than discovered later: `--out` holds the **BEST**
checkpoint by held-out `L_q` and `--out.last` the rolling last one (which is
what `--resume` reads); after training, **the best checkpoint is loaded back
into the model** before anything is scored, so the causal question is asked of
the checkpoint the guard chose, not of step 12000; and `steps` is now a CAP.

`bed.batch_do()` returns seven CPU tensors. The line marked `# THE DEVICE
TRAP` is the whole fix; without it the first forward raises
`Expected all tensors to be on the same device` at **step 1** on a GPU, and a
CPU smoke test cannot see it because on CPU both sides already agree. The same
trap is on the eval path — `train.evaluate` draws from the bed and hands the
result straight to the model — so the held-out bed is wrapped in `_OnDevice`,
which is that same one-line move applied where train.py would otherwise do it.

`do_read` refusals are counted, never silenced: `SingularTransientBlockError`
means a candidate row drove `P'[i,i]` toward 1 and `(I − Q')` toward singular.
A refusal SKIPS that example's term (it does not contribute a fabricated
zero), and a window rate above 1% prints as a `[FINDING]`.

DERIVED, and the reason the refusal counter is expected to read 0 during
training: `den = (1 − P'_ii)/(1 − P_ii)`, and `A_tel[i][i] = 0` for a transient
`i`, so `P'_ii <= 1 − c` and `den >= c = 0.0125` against
`den_min = sqrt(eps_f32) = 3.45e-04` — a **36x margin**. The guard is provably
live only at `teleport = 0`. A nonzero refusal rate here would itself be the
finding."""))

CELLS.append(code(r'''import argparse

class _OnDevice:
    """train.evaluate() draws its batch from the bed and hands it straight to the
    model. The bed's tensors are CPU, the model is on DEVICE, and train.py may not
    be edited -- so THE DEVICE TRAP's one-line fix lives here for the eval path."""
    def __init__(self, bed):
        self.bed = bed
    def batch_do(self, gen, B):
        return tuple(t.to(DEVICE) for t in self.bed.batch_do(gen, B))

def _args(seed):
    # save_checkpoint stores geometry=vars(args), so anything set here rides along
    # into the checkpoint and comes back on resume -- which is how the cumulative
    # refusal counters survive a restart without a second checkpoint format.
    return argparse.Namespace(lambda_z=FROZEN["lambda_z"], lambda_do=FROZEN["lambda_do"],
                              lambda_topo=FROZEN["lambda_topo"],
                              topo_blocks=FROZEN["topo_blocks"], seed=seed,
                              n=FROZEN["n"], nA=fit_bed.nA, d_enc=FROZEN["d_enc"],
                              x_dim=fit_bed.x_dim, rank=FROZEN["rank"],
                              early_stop_patience=FROZEN["early_stop_patience"],
                              divergence_k=FROZEN["divergence_k"], pkg_sha=PKG_SHA)

def new_model(seed):
    torch.manual_seed(seed)
    return TinyCEQ(n=FROZEN["n"], nA=fit_bed.nA, d_enc=FROZEN["d_enc"],
                   x_dim=fit_bed.x_dim, z_dim_state=FROZEN["z_dim_state"],
                   g=FROZEN["g"], rank=FROZEN["rank"],
                   absorbing_idx=torch.arange(fit_bed.nA))

def ckpt_paths(arm, seed):
    """(best, last) -- train.py's --out / --out.last convention, per arm per seed."""
    p = os.path.join(WORK, "ceqjepa_causal_%s_seed%d.pt" % (arm, seed))
    return p, p + ".last"

def train_seed(seed, trace=None):
    arm = ARM["name"]
    best_path, last_path = ckpt_paths(arm, seed)
    model = new_model(seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN["lr"], weight_decay=0.01)
    stats = dict(attempts=0, fails=0, last_error=None, win_attempts=0, win_fails=0)
    args = _args(seed)
    args.arm, args.out = arm, best_path
    train_gen = torch.Generator().manual_seed(seed)              # CPU gens: they draw indices
    heldout_gen = torch.Generator().manual_seed(seed + 1_000_000)  # disjoint stream, as main()
    # q_bar, the constant-predictor control: the training split's channel mean.
    # Drawn BEFORE any resume restores train_gen, and from a freshly seeded gen, so
    # a resumed run reproduces the same q_bar and then continues the same stream.
    _, _, q_pool, _ = fit_bed.batch(train_gen, max(64, FROZEN["batch_size"]))
    q_bar = q_pool.mean(0).to(DEVICE)
    history, start_step, wall_prev = [], 0, 0.0
    if os.path.exists(last_path):
        # map_location='cpu': torch_rng_state must be a CPU ByteTensor, and
        # load_state_dict copies into the already-placed model/optimizer anyway.
        d = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(d["model_state_dict"])
        opt.load_state_dict(d["optimizer_state_dict"])
        torch.set_rng_state(d["torch_rng_state"])
        train_gen.set_state(d["train_gen_state"]); heldout_gen.set_state(d["heldout_gen_state"])
        start_step, history = d["step"], d.get("history", [])
        wall_prev = d.get("wall_s", 0.0)
        stats.update(d.get("geometry", {}).get("do_stats") or {})
        print("  [%s seed %d] RESUMED from step %d (%.0fs of prior wall, %d eval rows)"
              % (arm, seed, start_step, wall_prev, len(history)), flush=True)
    args.do_stats = stats
    n_params = sum(q.numel() for q in model.parameters())
    print("  [%s seed %d] n_params = %d  (n=%d d_enc=%d rank=%d x_dim=%d); best -> %s, "
          "last -> %s" % (arm, seed, n_params, FROZEN["n"], FROZEN["d_enc"], FROZEN["rank"],
                          fit_bed.x_dim, os.path.basename(best_path),
                          os.path.basename(last_path)), flush=True)

    # BEST, NOT LAST -- seeded from history so a resume cannot re-crown a worse
    # checkpoint over a good one. Same rule as train.py main().
    prior = [r["L_q"] for r in history if r.get("phase") == arm and r["L_q"] == r["L_q"]]
    best_score = min(prior) if prior else float("inf")
    best_step = (min((r for r in history if r.get("phase") == arm and r["L_q"] == best_score),
                     key=lambda r: r["step"])["step"] if prior else start_step)
    best_train_loss = float("inf")
    stale = diverge_streak = 0
    stopped_early = False
    last_eval = None
    ho_dev = _OnDevice(ho_bed)

    t0 = time.time()
    win = dict(attempts=stats["attempts"], fails=stats["fails"])
    step = start_step
    for step in range(start_step + 1, FROZEN["steps"] + 1):
        batch = draw(fit_bed, train_gen, FROZEN["batch_size"])
        # ------------------------------------------------------------------
        x, x_nx, q_star, v_idx, moves, do_tgt, do_mask = [t.to(DEVICE) for t in batch]
        # ^^^ THE DEVICE TRAP. batch_do() builds all seven on CPU; the model is
        # on DEVICE. Drop this line and the first forward raises at step 1 on a
        # GPU, and no CPU smoke test can see it.
        # ------------------------------------------------------------------
        model.train()
        out = model(x)
        blocks = topo_blocks(args, q_star, 0)
        loss, l_q, l_z, l_do, l_topo = compute_loss(
            model, out, q_star, x_nx, moves, do_tgt, do_mask, blocks, args, stats)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % FROZEN["eval_every"] == 0 or step == FROZEN["steps"] or step == 1:
            # THE EVAL ROW IS train.py's, held-out arm and all: L_q on the
            # game-disjoint split, S against the constant predictor, kappa, p_spread,
            # the refusal rates, and margin(Iq-J-KL) from ceqjepa.sharpness.
            ev = evaluate(model, ho_dev, q_bar, None, heldout_gen, FROZEN["eval_n"],
                          args=args, phase_index=0, train_stats=stats)
            row = dict(step=step, phase=arm, train_loss=loss.item(),
                       train_L_do=l_do, train_L_topo=l_topo, **ev)
            history.append(row)
            score, tr_loss, last_eval = ev["L_q"], loss.item(), row
            improved = score < best_score
            if improved:
                best_score, best_step, best_train_loss, stale = score, step, tr_loss, 0
            else:
                stale += 1
            tail = (" NEW BEST" if improved else
                    " (stale %d/%d" % (stale, FROZEN["early_stop_patience"])
                    if FROZEN["early_stop_patience"] else " (stale %d, early stop OFF" % stale)
            print(f"[eval] step={step:4d} train_loss={loss.item():.4f} "
                  f"heldout_L_q={score:.4f} best_heldout_L_q={best_score:.4f}@step{best_step}"
                  f"{tail if improved else tail + ')'} "
                  f"S(vs const)={ev['S']:+.4f} mse_model={ev['mse_model']:.4e} "
                  f"mse_bar={ev['mse_bar']:.4e} collapse_floor={ev['collapse_floor']:.4e} "
                  f"var_across_examples(enc_output,batch_dim)={ev['var_across_examples']:.4e} "
                  f"kappa_bound={ev['kappa_bound']:.4f} p_spread={ev['p_spread']:.3e} "
                  f"margin(Iq-J-KL)={ev['sharp_margin']:+.4f}"
                  f"{'' if ev['beats_marginal'] else ' OVERCONFIDENT'} "
                  f"L_do(train)={l_do:.4f} L_do(heldout)={ev['L_do']:.4f} "
                  f"L_topo={ev['L_topo']:.4f} "
                  f"do_refuse[window]={ev['do_refuse_window']:.4f} "
                  f"do_refuse[cum]={ev['do_refuse_cum']:.4f}", flush=True)

            if improved:                       # --out holds the BEST, written the moment it is
                save_checkpoint(best_path, model, opt, args, n_params, q_bar, history,
                                wall_prev + time.time() - t0, step, train_gen, heldout_gen,
                                arm, [arm])

            # THE MEMORISATION GUARD. train.diverging() is the vote (held-out
            # materially worse than the BEST while training loss is materially better
            # than it was there, DIVERGENCE_REL=0.05); K consecutive votes fire it.
            # Replayed on the old T4 series this fires at step 8000 -- 1 h 40 m early.
            if not improved and diverging(score, best_score, tr_loss, best_train_loss):
                diverge_streak += 1
            else:
                diverge_streak = 0
            if FROZEN["divergence_k"] and diverge_streak > FROZEN["divergence_k"]:
                print("[MEMORISATION] still diverging: %d evals, held-out L_q %.4f vs best "
                      "%.4f@%d, train_loss %.4f vs %.4f there"
                      % (diverge_streak, score, best_score, best_step, tr_loss,
                         best_train_loss), flush=True)
            elif FROZEN["divergence_k"] and diverge_streak == FROZEN["divergence_k"]:
                print("[MEMORISATION] held-out L_q has been WORSE than its best for %d "
                      "consecutive evals while the training loss kept IMPROVING past the best "
                      "checkpoint's: held-out L_q %.4f (step %d) -> %.4f (step %d), %+.4f; "
                      "training loss %.4f -> %.4f, %+.4f. The model is fitting the training "
                      "rows, not the task. The best checkpoint is %d steps back and every step "
                      "since has bought training loss with held-out loss."
                      % (diverge_streak, best_score, best_step, score, step, score - best_score,
                         best_train_loss, tr_loss, tr_loss - best_train_loss, step - best_step),
                      flush=True)

            if FROZEN["early_stop_patience"] and stale >= FROZEN["early_stop_patience"]:
                print("[EARLY STOP] %d consecutive evals with no new best held-out L_q "
                      "(patience=%d). Best %.4f at step %d; stopping at step %d instead of "
                      "running to %d." % (stale, FROZEN["early_stop_patience"], best_score,
                                          best_step, step, FROZEN["steps"]), flush=True)
                stopped_early = True

        if step % FROZEN["ckpt_every"] == 0 or step == FROZEN["steps"]:
            save_checkpoint(last_path, model, opt, args, n_params, q_bar, history,
                            wall_prev + time.time() - t0, step, train_gen, heldout_gen,
                            arm, [arm])
        if step % FROZEN["trace_every"] == 0 or step == FROZEN["steps"] or step == 1:
            da = stats["attempts"] - win["attempts"]
            df = stats["fails"] - win["fails"]
            rate = df / da if da else float("nan")
            print("  [%s seed %d] step %5d  L_q=%.4f L_do=%.4f p_spread=%.3e "
                  "refuse[win]=%d/%d (%.4f) refuse[cum]=%d/%d  %.1fs (%.3f s/step)"
                  % (arm, seed, step, l_q, l_do, float(out["P"].std(dim=0).max()), df, da, rate,
                     stats["fails"], stats["attempts"], time.time() - t0,
                     (time.time() - t0) / max(1, step - start_step)), flush=True)
            if da and rate > 0.01:
                print("  [FINDING] intervene REFUSED %d/%d (%.2f%%) since the last trace -- "
                      "above the 1%% line. Last: %s" % (df, da, 100 * rate, stats["last_error"]),
                      flush=True)
            win = dict(attempts=stats["attempts"], fails=stats["fails"])
            if trace is not None:
                trace(model, seed, step)
        if stopped_early:
            break
        if time.time() - T_START > FROZEN["deadline_s"]:
            print("  [%s seed %d] DEADLINE at step %d -- checkpointed, stopping early so the "
                  "evaluation still runs" % (arm, seed, step), flush=True)
            break
    wall_s = wall_prev + time.time() - t0
    save_checkpoint(last_path, model, opt, args, n_params, q_bar, history, wall_s, step,
                    train_gen, heldout_gen, arm, [arm])

    # THE RATIO, train.py's [SUMMARY]. On the old T4 causal arm (best 2000, ran to
    # 12000) it reads 10000/2000 = 5.00 -- and nothing printed it, so nobody saw it.
    final = last_eval["L_q"] if last_eval else float("nan")
    wasted, useful = step - best_step, max(1, best_step)
    print("[SUMMARY] %s seed %d: best_step=%d best_heldout_L_q=%.4f | final_step=%d "
          "final_heldout_L_q=%.4f | delta_final_minus_best=%+.4f"
          % (arm, seed, best_step, best_score, step, final, final - best_score))
    print("[SUMMARY] %s seed %d: wasted/useful = %d/%d = %.2f (%d steps ran after the best "
          "checkpoint and made it no better; %d steps produced it)"
          % (arm, seed, wasted, useful, wasted / useful, wasted, useful))

    # BEST, NOT LAST, WHERE IT COUNTS: the causal question below is asked of the
    # checkpoint the guard chose. The old run scored step 12000 and reported it.
    if os.path.exists(best_path):
        d = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(d["model_state_dict"])
        print("[SUMMARY] %s seed %d: loaded the BEST checkpoint (step %d) back into the model; "
              "every number below is scored on it, not on step %d"
              % (arm, seed, d["step"], step), flush=True)
    return dict(model=model, stats=stats, last_step=step, wall_s=wall_s, n_params=n_params,
                best_step=best_step, best_heldout_L_q=best_score, final_heldout_L_q=final,
                wasted=wasted, useful=useful, stopped_early=stopped_early, history=history)
'''))

CELLS.append(md(r"""## Cell 6 — the four predictors, read from ONE forward pass

| arm | prediction | what it is |
|---|---|---|
| **MODEL do-read** | `q_do[i*]` | the operator's Sherman-Morrison read |
| **IGNORE-INTERVENTION** | `q_alpha` | **THE BAR**: predict `q(do a) = q(obs)`. What a correlational model does — the input did not change when the move was forced, so neither did its read. |
| **NO-CLAMP (same row)** | `q_field[i*]` | **the tighter bar**: same read *position*, same operator, intervention removed. Beating IGNORE could be an artifact of reading at a row instead of at the alpha-mix; beating NO-CLAMP cannot. Measured at init: `max\|q_do − q_noclamp\| = 0.0` exactly, so any value it takes is learned. |
| **MARGINAL** | training-set outcome frequency | has learned nothing |

`gap = PPL_do − PPL_obs`, and `PPL_obs` is identical across those rows, so
`gap(MODEL) − gap(BAR) = PPL_do(MODEL) − PPL_do(BAR)`. That difference needs
its **own paired bootstrap**, not a subtraction of two separately-estimated
SEs — the two arms are the same positions. It is obtained by handing
`causal_gap` a bed whose *both* `k_star` fields are `k_star_do`. No new
scoring code: same `_logp`, same paired `_bootstrap`."""))

CELLS.append(code(r'''# EVAL_X / EVAL_MOVES are set by use_arm() above -- moved ONCE per arm, same trap,
# eval side. They are read here, never assigned, so the control arm runs this code.

@torch.no_grad()
def read_arms(model, moves=None):
    """Every prediction this test scores, from ONE forward pass over the whole
    held-out bed. Returns CPU float64 [N,K] tensors plus the refusal mask.

    Chunked: P is [N, n, n] float32 = 393 MB at N=1500, n=256, and
    build_operator / committor / state_solve each hold several tensors that
    size. An OOM here would kill the run at a TRACE, i.e. mid-training, so the
    eval forward is chunked rather than trusted to fit."""
    model.eval()
    mv = EVAL_MOVES if moves is None else moves
    stats = dict(attempts=0, fails=0, last_error=None)
    cpu = lambda t: t.detach().double().cpu()
    acc = {k: [] for k in ("q_obs", "q_do", "q_noclamp", "ok", "i_star")}
    for lo in range(0, EVAL_X.shape[0], FROZEN["eval_chunk"]):
        hi = min(lo + FROZEN["eval_chunk"], EVAL_X.shape[0])
        out = model(EVAL_X[lo:hi])
        field, ok, i_star = model.do_read(out, mv[lo:hi], stats, return_field=True)
        b = torch.arange(hi - lo, device=DEVICE)
        acc["q_obs"].append(cpu(out["q_alpha"]))
        acc["q_do"].append(cpu(field[b, 0, i_star, :]))
        acc["q_noclamp"].append(cpu(out["q_field"][b, i_star, :]))
        acc["ok"].append(ok.cpu())
        acc["i_star"].append(i_star.cpu())
    r = {k: torch.cat(v) for k, v in acc.items()}
    r["stats"] = stats
    return r

def const(q):
    return lambda _bed: q

def score_seed(seed, arms):
    ok = arms["ok"]
    n_ref = int((~ok).sum())
    print("[eval] Sherman-Morrison refused %d/%d held-out positions (last: %s)"
          % (n_ref, ok.numel(), arms["stats"]["last_error"]))
    q_obs, q_do, q_nc = arms["q_obs"][ok], arms["q_do"][ok], arms["q_noclamp"][ok]
    bed = dict(k_star_obs=k_obs[ok], k_star_do=k_do[ok])
    bed_do_only = dict(k_star_obs=k_do[ok], k_star_do=k_do[ok])
    kw = dict(n_boot=FROZEN["n_boot"], seed=FROZEN["boot_seed"])
    f_obs, f_do, f_nc = const(q_obs), const(q_do), const(q_nc)
    print("--- seed %d: the four predictors, identical scoring path ---" % seed)
    r_model = ce.causal_gap(f_obs, f_do, bed, label="MODEL do-read s%d" % seed, **kw)
    r_bar = ce.ignore_intervention_gap(f_obs, bed, **kw)
    r_nc = ce.causal_gap(f_obs, f_nc, bed, label="NO-CLAMP same-row s%d" % seed, **kw)
    r_marg = ce.causal_gap(marg, marg, bed, label="MARGINAL s%d" % seed, **kw)
    print("--- seed %d: HEAD-TO-HEAD (paired bootstrap on the DIFFERENCE) ---" % seed)
    h_bar = ce.causal_gap(f_obs, f_do, bed_do_only, label="MODEL-minus-BAR s%d" % seed, **kw)
    h_nc = ce.causal_gap(f_nc, f_do, bed_do_only, label="MODEL-minus-NOCLAMP s%d" % seed, **kw)
    sig = lambda r: r["gap"] / r["se_gap"] if r["se_gap"] else float("nan")
    verdict = "BEATS the bar" if h_bar["gap"] < -abs(h_bar["se_gap"]) else "does NOT beat the bar"
    print("    [VERDICT seed %d] MODEL PPL_do minus BAR PPL_do = %+.4f +- %.4f (%+.2f SE; "
          "NEGATIVE = the do-read is better) -> %s"
          % (seed, h_bar["gap"], h_bar["se_gap"], sig(h_bar), verdict))
    print("    [VERDICT seed %d] MODEL PPL_do minus NO-CLAMP PPL_do = %+.4f +- %.4f (%+.2f SE)"
          % (seed, h_nc["gap"], h_nc["se_gap"], sig(h_nc)))
    return dict(seed=seed, n_refused=n_ref, model=r_model, bar=r_bar, noclamp=r_nc,
                marginal=r_marg, h_bar=h_bar, h_noclamp=h_nc)

def calibration(seed, arms):
    ok = arms["ok"]
    print("--- seed %d: RELIABILITY BUCKETS, BOTH ARMS (one-vs-rest, K=%d, 10 equal-width bins) ---"
          % (seed, K))
    outs = {}
    for name, q, kk in (("obs arm  q_alpha", arms["q_obs"][ok], k_obs[ok]),
                        ("do  arm  q_do   ", arms["q_do"][ok], k_do[ok])):
        rep = ce.calibration_report(q, kk)
        outs[name.strip()] = rep["ece"]
        print("  %s: ECE=%.4f over %d (item,class) points" % (name, rep["ece"], rep["n_points"]))
        for bb in rep["bins"]:
            if bb["count"]:
                print("    [%.1f,%.1f)  n=%6d  mean_pred=%.4f  empirical=%.4f"
                      % (bb["lo"], bb["hi"], bb["count"], bb["mean_pred"], bb["empirical_freq"]))
    return outs
'''))

CELLS.append(md(r"""## Cell 7 — run the arms, then the seeds

Both arms run the **same** code path; only `temperature` differs, so
`uniform_control` is literally chess_do's bed and its numbers are what zero
information scores. It runs one seed, the default arm three.

The `PPL` trace on the causal eval bed is **recorded only** — nothing selects on
it. Selection now happens on `heldout_L_q` over the game-disjoint split of the
TRAINING corpus, which is what `[eval]`, `[MEMORISATION]`, `[EARLY STOP]` and
`[SUMMARY]` above read. Two different held-out sets, on purpose: one chooses the
checkpoint, the other answers the causal question."""))

CELLS.append(code(r'''RESULTS = []
FIRST_MODEL = FIRST_ARMS = None
for arm_name, arm_T, arm_seeds in FROZEN["arms"]:
    use_arm(arm_name, arm_T)
    for seed in arm_seeds:
        print("\n=== TRAIN arm %s seed %d (cap %d steps, batch %d, n=%d rank=%d) ==="
              % (arm_name, seed, FROZEN["steps"], FROZEN["batch_size"], FROZEN["n"],
                 FROZEN["rank"]), flush=True)

        def trace(m, sd, st):
            a = read_arms(m)
            o = a["ok"]
            print("    [trace %s s%d step %d] causal-bed PPL_obs(q_alpha)=%.4f  "
                  "PPL_do(q_do)=%.4f  PPL_do(q_alpha, THE BAR)=%.4f  PPL_do(q_noclamp)=%.4f  "
                  "refused=%d"
                  % (arm_name, sd, st, ce.outcome_ppl(a["q_obs"][o], k_obs[o]),
                     ce.outcome_ppl(a["q_do"][o], k_do[o]),
                     ce.outcome_ppl(a["q_obs"][o], k_do[o]),
                     ce.outcome_ppl(a["q_noclamp"][o], k_do[o]), int((~o).sum())), flush=True)

        tr_out = train_seed(seed, trace=trace)
        model, tstats = tr_out["model"], tr_out["stats"]
        arms = read_arms(model)
        row = score_seed(seed, arms)
        row.update(arm=arm_name, temperature=float(ARM["temperature"]),
                   steps_done=tr_out["last_step"], wall_s=tr_out["wall_s"],
                   n_params=tr_out["n_params"], best_step=tr_out["best_step"],
                   best_heldout_L_q=tr_out["best_heldout_L_q"],
                   final_heldout_L_q=tr_out["final_heldout_L_q"],
                   wasted=tr_out["wasted"], useful=tr_out["useful"],
                   stopped_early=tr_out["stopped_early"], history=tr_out["history"],
                   train_refused=tstats["fails"], train_attempts=tstats["attempts"])
        if arm_name == PRIMARY[0] and seed == arm_seeds[0]:
            row["ece"] = calibration(seed, arms)
            FIRST_MODEL, FIRST_ARMS = model, arms
        RESULTS.append(row)
        print("  [%s seed %d] %d steps in %.0fs = %.3f s/step; train refusals %d/%d"
              % (arm_name, seed, tr_out["last_step"], tr_out["wall_s"],
                 tr_out["wall_s"] / max(1, tr_out["last_step"]),
                 tstats["fails"], tstats["attempts"]), flush=True)

use_arm(PRIMARY[0], PRIMARY[1])    # the two control cells below are the DEFAULT arm's
'''))

CELLS.append(md(r"""## Cell 8 — CONTROL 1: the move-permutation ablation

Same trained model, same code path, the move→position pairing destroyed by a
permutation. If the do-read's advantage survives randomising *which*
intervention was performed, the advantage is not causal — whatever it
exploited, it was not the move.

Two numbers, and both are needed: `PPL_do(real) − PPL_do(permuted)` says
whether the moves carry information, and `max|q_do(real) − q_do(permuted)|`
says whether the read is even *responding* to the move (a dead read would give
a small difference for the trivial reason)."""))

CELLS.append(code(r'''g_perm = torch.Generator().manual_seed(FROZEN["perm_seed"])
perm = torch.randperm(EVAL_MOVES.shape[0], generator=g_perm)
arms_perm = read_arms(FIRST_MODEL, moves=EVAL_MOVES[perm.to(DEVICE)])

ok_both = FIRST_ARMS["ok"] & arms_perm["ok"]
print("[ablation] refused: real %d, permuted %d; scored on the %d positions both arms accepted"
      % (int((~FIRST_ARMS["ok"]).sum()), int((~arms_perm["ok"]).sum()), int(ok_both.sum())))
bed_do_only = dict(k_star_obs=k_do[ok_both], k_star_do=k_do[ok_both])
kw = dict(n_boot=FROZEN["n_boot"], seed=FROZEN["boot_seed"])
r_real = ce.causal_gap(const(FIRST_ARMS["q_obs"][ok_both]), const(FIRST_ARMS["q_do"][ok_both]),
                       dict(k_star_obs=k_obs[ok_both], k_star_do=k_do[ok_both]),
                       label="MODEL real moves", **kw)
r_perm = ce.causal_gap(const(FIRST_ARMS["q_obs"][ok_both]), const(arms_perm["q_do"][ok_both]),
                       dict(k_star_obs=k_obs[ok_both], k_star_do=k_do[ok_both]),
                       label="MODEL PERMUTED moves", **kw)
h_perm = ce.causal_gap(const(arms_perm["q_do"][ok_both]), const(FIRST_ARMS["q_do"][ok_both]),
                       bed_do_only, label="REAL-minus-PERMUTED", **kw)
q_move = float((FIRST_ARMS["q_do"][ok_both] - arms_perm["q_do"][ok_both]).abs().max())
print("[MOVE ABLATION] PPL_do(real moves) - PPL_do(permuted moves) = %+.4f +- %.4f (%+.2f SE)"
      % (h_perm["gap"], h_perm["se_gap"],
         h_perm["gap"] / h_perm["se_gap"] if h_perm["se_gap"] else float("nan")))
print("[MOVE ABLATION] max|q_do(real) - q_do(permuted)| over the eval set = %.6e  "
      "(large = the read DOES respond to the move; the SE above says whether that response "
      "carries outcome information)" % q_move)
ABLATION = dict(real=r_real, permuted=r_perm, head_to_head=h_perm, max_abs_dq=q_move,
                n=int(ok_both.sum()))
'''))

CELLS.append(md(r"""## Cell 9 — CONTROL 2: the bed's oracle headroom, no model anywhere

A NULL in the headline has two completely different causes and the model card
must not confuse them:

* **(a)** the operator does not represent the intervention, or
* **(b)** forcing one ply barely moves the outcome distribution, so *no*
  predictor — not even one handed the true interventional distribution — could
  beat the bar on this bed. That was measured TRUE on the uniform bed
  (`TV(p_do, p_obs) = 0.1501` against a rollout-noise floor of `0.1468`, excess
  `+0.0033`), and it is re-measured here **on the arm actually being scored**:
  the rollouts below run the same `temperature` the bed was built with.

This cell measures **(b)** directly by brute-force rollouts. Both arms are
add-one smoothed identically, so the comparison is not tilted. `ORACLE vs
ITSELF` is a planted negative: it must read exactly `gap = 0.000000,
se = 0.000000` or the bootstrap is not paired. `ORACLE vs ITS COPY` is the
noise floor — **any headroom smaller than that is estimator noise, not
signal.**"""))

CELLS.append(code(r'''def _rollouts(board, uci, rng, R, mp, T):
    b = board.copy(stack=False)
    b.push(chess.Move.from_uci(uci))
    acc = np.zeros(N_OUTCOMES, dtype=np.float64)
    for _ in range(R):
        acc += _rollout_outcome(b, rng, mp, T)    # chess_policy's: same policy the bed used
    return acc

def _smooth(counts, R):
    return (counts + 1.0) / (R + N_OUTCOMES)      # add-one, identical on both arms

rng = random.Random(FROZEN["oracle_seed"])
R_or, mp, T_or = FROZEN["oracle_R"], FROZEN["max_plies"], ARM["temperature"]
print("[oracle] arm %s, temperature %s -- the rollouts below use the SAME policy the bed was "
      "built with, or the headroom is a number about a different bed" % (ARM["name"], T_or))
sub = ev_samples[:FROZEN["oracle_positions"]]     # subsample: the SE below is for THIS n, not 1500
P_do, P_obs, P_do2 = [], [], []
t0 = time.time()
for j, s in enumerate(sub):
    board = chess.Board(s.fen)
    P_do.append(_smooth(_rollouts(board, s.candidate_ucis[0], rng, R_or, mp, T_or), R_or))
    P_obs.append(_smooth(_rollouts(board, s.obs_uci, rng, R_or, mp, T_or), R_or))
    P_do2.append(_smooth(_rollouts(board, s.candidate_ucis[0], rng, R_or, mp, T_or), R_or))
    if (j + 1) % 100 == 0:
        print("  oracle %d/%d (%.0fs)" % (j + 1, len(sub), time.time() - t0), flush=True)
P_do, P_obs, P_do2 = (torch.tensor(np.stack(z)) for z in (P_do, P_obs, P_do2))
ko, kd = k_obs[:len(sub)], k_do[:len(sub)]

tv_causal = float((P_do - P_obs).abs().sum(-1).mean() / 2)
tv_noise = float((P_do - P_do2).abs().sum(-1).mean() / 2)
print("\n[MEASURED] mean TV(p_do, p_obs)  = %.4f   <- forced move vs played move" % tv_causal)
print("[MEASURED] mean TV(p_do, p_do2) = %.4f   <- SAME distribution, independent rollouts: "
      "the noise floor at R=%d" % (tv_noise, R_or))
print("[MEASURED] excess over noise    = %+.4f" % (tv_causal - tv_noise))

bed_o = dict(k_star_obs=kd, k_star_do=kd)          # both arms scored on the FORCED outcome
r_head = ce.causal_gap(const(P_obs), const(P_do), bed_o, label="ORACLE HEADROOM", **kw)
print("    [HEADROOM] the best possible head-to-head advantage on this bed = %+.4f +- %.4f "
      "(%+.2f SE; NEGATIVE = knowing the intervention helps)"
      % (r_head["gap"], r_head["se_gap"],
         r_head["gap"] / r_head["se_gap"] if r_head["se_gap"] else float("nan")))
r_self = ce.causal_gap(const(P_do), const(P_do), bed_o, label="ORACLE vs ITSELF", **kw)
assert r_self["gap"] == 0.0 and r_self["se_gap"] == 0.0, "paired bootstrap is not paired"
print("    [PLANTED NEGATIVE] oracle scored against itself: gap=%+.6f se=%.6f (must be exactly "
      "0/0, and is)" % (r_self["gap"], r_self["se_gap"]))
r_copy = ce.causal_gap(const(P_do2), const(P_do), bed_o, label="ORACLE vs ITS COPY", **kw)
print("    [NOISE FLOOR] the same oracle re-estimated from independent rollouts: gap=%+.4f "
      "+- %.4f -- any headroom smaller than this is estimator noise, not signal."
      % (r_copy["gap"], r_copy["se_gap"]))
HEADROOM = dict(headroom=r_head, self=r_self, copy=r_copy, tv_causal=tv_causal,
                tv_noise=tv_noise, n=len(sub), R=R_or)
'''))

CELLS.append(md("## Cell 10 — the answer, and the manifest"))

CELLS.append(code(r'''print("\n=== SUMMARY over %d runs (%d arms) ===" % (len(RESULTS), len(FROZEN["arms"])))
for r in RESULTS:
    print("  %s seed %d: ran %d steps (best %d, wasted/useful %d/%d = %.2f%s) %.0fs | "
          "gap(MODEL)=%+.4f+-%.4f gap(BAR)=%+.4f+-%.4f gap(NO-CLAMP)=%+.4f+-%.4f "
          "gap(MARGINAL)=%+.4f+-%.4f | MODEL-minus-BAR=%+.4f+-%.4f "
          "MODEL-minus-NOCLAMP=%+.4f+-%.4f | refused=%d"
          % (r["arm"], r["seed"], r["steps_done"], r["best_step"], r["wasted"], r["useful"],
             r["wasted"] / r["useful"], ", EARLY STOP" if r["stopped_early"] else "",
             r["wall_s"], r["model"]["gap"], r["model"]["se_gap"],
             r["bar"]["gap"], r["bar"]["se_gap"], r["noclamp"]["gap"], r["noclamp"]["se_gap"],
             r["marginal"]["gap"], r["marginal"]["se_gap"], r["h_bar"]["gap"],
             r["h_bar"]["se_gap"], r["h_noclamp"]["gap"], r["h_noclamp"]["se_gap"],
             r["n_refused"]))

PRIM = [r for r in RESULTS if r["arm"] == PRIMARY[0]]
CTRL = [r for r in RESULTS if r["arm"] != PRIMARY[0]]
wins = sum(1 for r in PRIM if r["h_bar"]["gap"] < -abs(r["h_bar"]["se_gap"]))
d = [r["h_bar"]["gap"] for r in PRIM]
print("\n[ANSWER, LITERAL] on arm %s the do-read beat the IGNORE-THE-INTERVENTION bar by more "
      "than one paired bootstrap SE in %d/%d seeds; mean difference %+.4f (per-seed %s)."
      % (PRIMARY[0], wins, len(PRIM), float(np.mean(d)), ["%+.4f" % x for x in d]))
for r in CTRL:
    print("[CONTROL ARM %s] the SAME code path on the uniform bed -- chess_do's own rollout "
          "policy, MEASURED I(X;Y)=0.1363 nats against 0.3084 at T=0.25 -- scored "
          "MODEL-minus-BAR=%+.4f+-%.4f, PPL_obs=%.4f (chance %.2f), best_step=%d, "
          "wasted/useful=%d/%d. A headline the control reproduces is a headline about the "
          "estimator, not the architecture."
          % (r["arm"], r["h_bar"]["gap"], r["h_bar"]["se_gap"], r["model"]["ppl_obs"],
             r["model"]["chance"], r["best_step"], r["wasted"], r["useful"]))
print("[ANSWER, SCIENTIFIC] that number is causal evidence ONLY IF both controls agree:")
print("   move-permutation  PPL_do(real) - PPL_do(permuted) = %+.4f +- %.4f  "
      "(a value inside its own SE means the advantage does not depend on WHICH move was forced)"
      % (ABLATION["head_to_head"]["gap"], ABLATION["head_to_head"]["se_gap"]))
print("   oracle headroom   = %+.4f +- %.4f, noise floor %+.4f +- %.4f, TV excess %+.4f  "
      "(a headroom that is positive, or smaller than the noise floor, means NO predictor could "
      "beat the bar on this bed and a model NULL says nothing about the architecture)"
      % (HEADROOM["headroom"]["gap"], HEADROOM["headroom"]["se_gap"],
         HEADROOM["copy"]["gap"], HEADROOM["copy"]["se_gap"],
         HEADROOM["tv_causal"] - HEADROOM["tv_noise"]))
print("   observational arm PPL_obs = %.4f against chance %.2f  (an arm WORSE than chance means "
      "any head-to-head win is the bar collapsing, not the do-read improving)"
      % (PRIM[0]["model"]["ppl_obs"], PRIM[0]["model"]["chance"]))
print("   sharpness margin  last eval Iq-J-KL = %+.4f%s  (NEGATIVE means the read LOSES to a "
      "predictor that never looked at the input, however good its cross-entropy looks)"
      % (PRIM[0]["history"][-1]["sharp_margin"],
         "" if PRIM[0]["history"][-1]["beats_marginal"] else " OVERCONFIDENT"))

MANIFEST = dict(
    frozen={k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()},
    torch=torch.__version__, device=str(DEVICE),
    gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    pkg_sha1=PKG_SHA, wall_total_s=time.time() - T_START,
    n_train=len(tr_samples), n_eval=len(ev_samples),
    train_pad_rate=fit_bed.pad_rate, n_fit=len(fit_bed.x), n_heldout=len(ho_bed.x),
    arms_run=sorted({r["arm"] for r in RESULTS}),
    seeds=RESULTS, ablation=ABLATION, headroom=HEADROOM,
    wins_over_bar=wins, mean_model_minus_bar=float(np.mean(d)),
)
with open(os.path.join(WORK, "causal_manifest.json"), "w") as fh:
    json.dump(MANIFEST, fh, indent=2, default=float)
print("\nwrote", os.path.join(WORK, "causal_manifest.json"), "| total wall %.0fs"
      % (time.time() - T_START))
print("checkpoints:", [f for f in sorted(os.listdir(WORK))
                       if f.endswith(".pt") or f.endswith(".last")])
'''))


def build():
    files = collect_files()
    cells = []
    for c in CELLS:
        src = "".join(c["source"])
        if "__FILES__" in src:
            src = src.replace("__FILES__", json.dumps(files, indent=0))
            c = code(src)
        cells.append(c)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote %s (%d cells, %.0f KB, %d package files inlined)"
          % (OUT, len(cells), OUT.stat().st_size / 1024, len(files)))
    if "--extract" in sys.argv:
        flat = HERE / "causal_extracted.py"
        body = "\n\n# %s CELL %d %s\n" % ("=" * 22, 0, "=" * 22)
        parts = []
        i = 0
        for c in cells:
            if c["cell_type"] != "code":
                continue
            i += 1
            parts.append("# %s CELL %d %s\n%s" % ("=" * 22, i, "=" * 22, "".join(c["source"])))
        flat.write_text("\n\n".join(parts), encoding="utf-8")
        print("wrote %s (%d code cells) -- run it with CEQ_SMOKE=1 to prove the body executes"
              % (flat, i))


if __name__ == "__main__":
    build()
