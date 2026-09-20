"""Builds ceq_jepa_kaggle_curriculum.ipynb next to this script, from the
CURRENT working tree (ceqjepa/*.py, ceqjepa/beds/*.py, ceq/kdata.py's
label_plies/iter_games). Run locally to regenerate; not itself uploaded to
Kaggle.

Same house pattern as build_notebook.py (the stage-1 generator): read the
real source files off disk at generation time and inline them into cells,
rather than hand-retyping a second copy that can drift. ponytail: stdlib
json + regex only, no nbformat dependency.

WHY FULL INLINING (no `sys.path` + subprocess to `python -m ceqjepa.train`,
the way ceqjepa/curriculum.py drives phases locally): Kaggle's kernel
filesystem doesn't carry ceqjepa/ as an installed package, and the stage-1
notebook already established "read the source at generation time, paste it
into cells" as this repo's house pattern for that gap. Extending it (rather
than switching to a different packaging strategy for just the curriculum
notebook) keeps the two notebooks consistent and is less code than writing
a package tree to /kaggle/working and shelling out to it.

WHAT GETS PATCHED, and why (ceqjepa/train.py itself is never edited):
  - `bed.batch(...)` results are moved to `device` at both call sites
    (the shared `evaluate()` and the per-step training loop) -- train.py
    itself never does this (every local run in FOUNDATION/CHAIN RESULTS was
    --device cpu); the task's own device-handling note ("batches are moved
    to device after bed.batch()") describes the STAGE-1 NOTEBOOK's own
    training-loop cell, not train.py, and this generator reproduces that
    same fix at the two spots the curriculum path needs it (train loop +
    evaluate(), the latter shared by the per-phase eval and the forgetting
    matrix's eval_bed()).
  - `q_bar` / `q_floor_table` are moved to `device` wherever train.py's
    own logic computes or restores them, for the same reason.
  - `from ceqjepa.beds.xxx import XxxBed` lines inside `make_bed` are
    dropped -- the bed classes are already in this notebook's global
    namespace, pasted in above.
  - `from ceq.kdata import iter_games, label_plies` (chess.py) is dropped
    the same way -- both functions are pasted in from ceq/kdata.py above.
  - `import ceqjepa.operator as op` is dropped and every `op.` prefix
    stripped (english.py, markets.py, train.py's own building blocks) --
    same technique build_notebook.py already used for train.py.
  - english.py's `_ROOT = pathlib.Path(__file__)...` (no `__file__` in a
    notebook) becomes `_ROOT = pathlib.Path(os.environ.get("CEQJEPA_DATA_ROOT",
    "/kaggle/working"))`, and the notebook writes an embedded slice of the
    real tinystories corpus to `_ROOT/"data"/"tinystories_20k.txt"` before
    EnglishBed is ever constructed. The corpus is a REAL 600,000-character
    PREFIX of data/tinystories_20k.txt (18,167,706 chars total), embedded
    directly rather than attached as a Kaggle dataset -- there is no
    existing Kaggle dataset id for this exact file to cite honestly, and
    600,000 real chars is already >> LOOKAHEAD(2048)+n, so nothing about
    the bed's design is degraded by not shipping the full 18 MB.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent          # ceqjepa/
REPO = ROOT.parent          # repo root
OUT_DIR = HERE / "curriculum"   # own directory + own kernel-metadata.json,
                                 # so pushing this kernel never touches the
                                 # stage-1 kernel's directory (ceqjepa/kaggle/).
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "ceq_jepa_kaggle_curriculum.ipynb"
FAILED_MARKER = OUT.with_name(OUT.name + ".FAILED")
CORPUS_CHARS = 600_000      # real prefix of data/tinystories_20k.txt, see docstring

# THE SILENT-FAILURE FIX. Every cell-building step below is an assert against
# the CURRENT ceqjepa/train.py -- when train.py's shape changes underneath
# one of them (as it did when evaluate() started drawing batches through the
# draw() helper instead of a literal bed.batch() call), the assert fires,
# nothing after it runs, and OUT is never touched -- so the OLD notebook sits
# there looking exactly as fresh as a real rebuild would have left it. This
# hook is the difference: on ANY uncaught exception it drops a marker file
# next to OUT (and re-prints the real traceback, unchanged) so a failed
# build always leaves something on disk that says so, instead of nothing.
FAILED_MARKER.unlink(missing_ok=True)   # clear a stale marker before this attempt


def _mark_failure_on_disk(exc_type, exc_value, exc_tb):
    import traceback
    FAILED_MARKER.write_text(
        f"build_curriculum_notebook.py did NOT finish -- {OUT.name} on disk, if "
        "present, is STALE (from an earlier successful build), not from this run.\n\n"
        + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    )
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _mark_failure_on_disk


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def strip_module_docstring(src):
    m = re.match(r'^""".*?"""\n', src, re.S)
    return src[m.end():] if m else src


def strip_op_prefix(src):
    """`op.build_operator(...)` -> `build_operator(...)` -- the functions
    live in this same notebook namespace now (operator.py's cell), not a
    `ceqjepa.operator` import."""
    return re.sub(r"\bop\.", "", src)


def move_batch_to_device(src, bed_expr, gen_expr, count_expr):
    """Insert a device-move line right after the unique call that draws a
    batch off `bed_expr` in `src`, matching whatever indentation that line
    already has.

    train.py used to draw every batch as a literal
    `x, x_nx, q_star[, v_idx] = bed_expr.batch(gen_expr, count_expr)`. It now
    goes through the `draw(bed, gen, B)` helper (train.py's own indirection,
    added so beds with an interventional arm can return the extra
    moves/do_tgt/do_mask fields), so the call site reads
    `<names> = draw(bed_expr, gen_expr, count_expr)` instead. Both forms are
    matched here -- the OLD one so a future revert of the helper doesn't
    silently stop being patched. `draw()` returns None for moves/do_tgt/
    do_mask on beds without an interventional arm, so the inserted line only
    calls `.to(device)` on the elements that are actually tensors."""
    pat = re.compile(
        r"(\n( *)([\w, ]*)= (?:draw\(" + re.escape(bed_expr) + r", " + re.escape(gen_expr) +
        r", " + re.escape(count_expr) + r"\)|" + re.escape(bed_expr) + r"\.batch\(" +
        re.escape(gen_expr) + r", " + re.escape(count_expr) + r"\))\n)"
    )
    m = pat.search(src)
    assert m, (f"move_batch_to_device: no match for draw({bed_expr}, {gen_expr}, {count_expr}) "
               f"or {bed_expr}.batch({gen_expr}, {count_expr}) in source")
    names = [n.strip() for n in m.group(3).split(",") if n.strip() and n.strip() != "_"]
    indent = m.group(2)
    moved = ", ".join(f"{n}.to(device) if torch.is_tensor({n}) else {n}" for n in names)
    move_line = f"{indent}{', '.join(names)} = ({moved})\n"
    return src[:m.end()] + move_line + src[m.end():]


cells = []

cells.append(md("""\
# CEQ-JEPA curriculum -- chess -> english -> markets, self-contained Kaggle notebook

torch + numpy (+ `python-chess`, pip-installed below) only. Every module below
is inlined VERBATIM, read off `ceqjepa/*.py`, `ceqjepa/beds/*.py` and
`ceq/kdata.py` at notebook-generation time by `build_curriculum_notebook.py`
-- regenerate with `python build_curriculum_notebook.py` after editing any of
those files; do this before every push, so the kernel never drifts from
what's actually in the repo. `ceqjepa/` and `ceq/` are UNTRACKED
(`git ls-files ceqjepa ceq` is empty) -- there is no commit SHA to cite as
this code's provenance; it is generated from the current working tree.

This is the sibling of the existing stage-1 kernel
(`melowdramtic/ceq-jepa-dcm1-stage1`, `ceq_jepa_kaggle.ipynb`, already ran
green on a T4 in 89.6s) -- that kernel stays as the stage-1 record; this one
is a NEW kernel (`melowdramtic/ceq-jepa-curriculum`) that runs the author's
three-phase curriculum: **phase 1 CHESS -> phase 2 ENGLISH -> phase 3
MARKETS**, each phase resuming from the previous phase's checkpoint
(sequential curriculum learning, not simultaneous multi-task training --
see the repo's own architecture notes on why that distinction matters for
what block-diagonal-P objection does and doesn't transfer).

**PAUSABLE, by construction, not by promise.** Every phase writes an atomic
checkpoint to `/kaggle/working/curriculum_<phase>.pt` every `CKPT_EVERY`
steps (train.py's own `--ckpt-every` / `--resume`, unmodified, exercised
here through this notebook rather than train.py's own CLI). On (re)start,
this notebook's orchestration cell reads whatever checkpoints already exist
under `/kaggle/working`, works out which phase is in progress and how many
steps that phase still needs, and resumes -- so a 9-hour Kaggle session
timeout or a quota pause loses at most one `CKPT_EVERY`-step interval, never
a whole phase. The **forgetting matrix** (S on every bed seen so far,
BUILD SPEC's own mandatory instrument -- forgetting is the EXPECTED failure
mode here, not a bug to hide) prints after every phase, using the checkpoint
files exactly as written, so it re-derives correctly across a restart too.

**kappa is a printed warning, never an assert** (train.py's own D8 fix,
unmodified): `evaluate()` only warns when `kappa_bound` exceeds
`--kappa-ceiling` (configurable below, default the exact float32 teleport
bound `KAPPA_DESIGN_BOUND_F32 = 80*(1+6e-6)`); it never aborts a healthy,
still-climbing run.

**Known, un-fixed gaps in train.py this notebook does not attempt to fix**
(out of scope -- this file only builds the Kaggle notebook): `--lr` is
silently ignored on a same-phase resume (the checkpoint's optimizer state
overwrites it) -- worked around here by using ONE constant `LR` for the
whole curriculum, never changed across a resume, so the bug is inert; there
is no bed-identity fingerprint in the checkpoint, so resuming with different
`N_GAMES`/`--seed` than the run that wrote the checkpoint would silently
continue on a different corpus with no warning -- this notebook keeps those
constants fixed for the life of one curriculum run."""))

cells.append(code("""\
import time, json, os, re, platform, types
import numpy as np
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""))

cells.append(md("## `python-chess` (pip-installed; not in the base Kaggle image)"))
cells.append(code("""\
try:
    import chess, chess.pgn
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess"], check=True)
    import chess, chess.pgn
print("python-chess", chess.__version__)
"""))

# ---------------------------------------------------------------------------
# ceqjepa.operator -- verbatim, same technique as build_notebook.py
# ---------------------------------------------------------------------------
cells.append(md("## `ceqjepa.operator` (inlined verbatim)"))
op_src = (ROOT / "operator.py").read_text()
op_src = op_src.split('if __name__ == "__main__":')[0].rstrip() + "\n"
cells.append(code(op_src))

# ---------------------------------------------------------------------------
# ceq.kdata slice -- only what ChessBed's default (self-play) and from_pgn
# paths need: iter_games + label_plies, verbatim.
# ---------------------------------------------------------------------------
cells.append(md("""\
## `ceq.kdata` slice (`iter_games`, `label_plies`, inlined verbatim)

Only these two functions -- ChessBed's oracle. `label_plies` replays a game
on a real `python-chess` Board and recomputes `fen_before`/`uci`/`fen_after`/
`legal` straight from the board, never from a stored column -- the same
oracle the repo's own gate-0 hygiene rule 5 requires. `iter_games` is the
scale path (`ChessBed.from_pgn`, e.g. an attached PGN dataset); not called
by the default self-play path this notebook actually runs, kept so
`from_pgn` is real code here too, not a promise."""))

kdata_src = (REPO / "ceq" / "kdata.py").read_text()


def extract_def(src, name):
    start = src.index(f"def {name}(")
    m = re.search(r"\n(?:def |class )", src[start + 1:])
    end = start + 1 + m.start() if m else len(src)
    return src[start:end].rstrip() + "\n"


kdata_cell = "import pathlib\n\n\n" + extract_def(kdata_src, "iter_games") + "\n\n" + \
    extract_def(kdata_src, "label_plies")
cells.append(code(kdata_cell))

# ---------------------------------------------------------------------------
# ceqjepa/beds/chess.py
# ---------------------------------------------------------------------------
cells.append(md("## `ceqjepa.beds.chess.ChessBed` (inlined, adapted)"))
chess_src = strip_module_docstring((ROOT / "beds" / "chess.py").read_text())
chess_src = chess_src.split("def demo()")[0]
chess_src = chess_src.replace("from ceq.kdata import iter_games, label_plies\n", "")
# ponytail: `from __future__ import annotations` must be a cell/file's first
# statement; dropping it is safe here -- every forward-ref annotation in
# this file is already quoted, and bare `list[dict]` is valid at runtime on
# Python >= 3.9 without it.
chess_src = chess_src.replace("from __future__ import annotations\n\n", "")
cells.append(code(chess_src.rstrip() + "\n"))

# ---------------------------------------------------------------------------
# ceqjepa/beds/english.py
# ---------------------------------------------------------------------------
cells.append(md("""\
## `ceqjepa.beds.english.EnglishBed` (inlined, adapted)

`_ROOT` (originally `pathlib.Path(__file__).resolve().parents[2]` -- no
`__file__` in a notebook) becomes `CEQJEPA_DATA_ROOT` (default
`/kaggle/working`, overridable for a local smoke test), and the corpus is
written to `_ROOT/"data"/"tinystories_20k.txt"` by the cell below, from an
embedded real prefix of the actual file (see the top markdown cell for why
a prefix, not the full 18 MB, and not a Kaggle dataset)."""))
english_src = strip_module_docstring((ROOT / "beds" / "english.py").read_text())
english_src = english_src.replace("from __future__ import annotations\n\n", "")
english_src = english_src.replace("import ceqjepa.operator as op\n", "")
english_src = strip_op_prefix(english_src)
english_src = english_src.replace(
    "_ROOT = pathlib.Path(__file__).resolve().parents[2]\n",
    '_ROOT = pathlib.Path(os.environ.get("CEQJEPA_DATA_ROOT", "/kaggle/working"))\n',
)
english_src = english_src.split('if __name__ == "__main__":')[0]
cells.append(code(english_src.rstrip() + "\n"))

cells.append(md("## Embedded corpus (real prefix of `data/tinystories_20k.txt`, written to disk)"))
corpus_full = (REPO / "data" / "tinystories_20k.txt").read_text(encoding="utf-8", errors="replace")
assert len(corpus_full) >= CORPUS_CHARS, "data/tinystories_20k.txt is shorter than the requested prefix"
corpus_slice = corpus_full[:CORPUS_CHARS]
corpus_cell = (
    "_CORPUS_PATH = pathlib.Path(os.environ.get(\"CEQJEPA_DATA_ROOT\", \"/kaggle/working\")) / \"data\" / \"tinystories_20k.txt\"\n"
    "_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
    f"_CORPUS_TEXT = {corpus_slice!r}\n"
    "_CORPUS_PATH.write_text(_CORPUS_TEXT, encoding='utf-8')\n"
    f"print(f'wrote {{len(_CORPUS_TEXT)}} real chars (prefix of the {len(corpus_full)}-char corpus) to {{_CORPUS_PATH}}')\n"
)
cells.append(code(corpus_cell))

# ---------------------------------------------------------------------------
# ceqjepa/beds/markets.py
# ---------------------------------------------------------------------------
cells.append(md("## `ceqjepa.beds.markets.MarketsBed` (inlined, adapted)"))
markets_src = strip_module_docstring((ROOT / "beds" / "markets.py").read_text())
markets_src = markets_src.replace("import ceqjepa.operator as op\n", "")
markets_src = strip_op_prefix(markets_src)
markets_src = markets_src.split('if __name__ == "__main__":')[0]
cells.append(code(markets_src.rstrip() + "\n"))

# ---------------------------------------------------------------------------
# ceqjepa/train.py building blocks -- same extraction as build_notebook.py,
# plus stripping the `from ceqjepa.beds.xxx import` lines inside make_bed
# (the bed classes are already in-notebook), plus device-move patches.
# ---------------------------------------------------------------------------
cells.append(md("""\
## `ceqjepa.train` building blocks (`Bed`, `make_bed`, `TinyCEQ`, `stage1_loss`,
`evaluate`, `save_checkpoint`) -- inlined, `op.` calls point at the operator
cell above. `evaluate()` is patched to move `bed.batch(...)`'s output onto
`next(model.parameters()).device` -- train.py itself never does this (every
run in this repo's own record was `--device cpu`); this is the same
device-handling fix the stage-1 notebook's training-loop cell already made
for its own loop, applied here at evaluate()'s single call site so both the
per-step eval and the forgetting matrix's `eval_bed()` inherit it for free."""))

train_src = (ROOT / "train.py").read_text()
_, _, after_docstring = train_src.partition('"""\n')
train_body = after_docstring.split("def main():")[0]
train_body = "import os\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" + \
    "EPS_Q = 1e-6  # BCE input clamp, MANDATORY per build spec: q attains 0 and 1 exactly\n" + \
    train_body.split("EPS_Q = 1e-6", 1)[1]
train_body = re.sub(r"^\s*from ceqjepa\.beds\.\w+ import .+\n", "", train_body, flags=re.M)
train_body = strip_op_prefix(train_body)
train_body = move_batch_to_device(train_body, "bed", "gen", "n")  # evaluate()'s draw(bed, gen, n)
assert "class Bed" in train_body and "class TinyCEQ" in train_body and "def evaluate" in train_body
assert "import ceqjepa" not in train_body and '"""ceqjepa/train.py' not in train_body
assert "from ceqjepa.beds" not in train_body
cells.append(code(train_body.rstrip() + "\n"))

# ---------------------------------------------------------------------------
# ceqjepa/curriculum.py helpers (_splice, eval_bed, print_matrix, _bed_ns) --
# reused verbatim rather than reinventing the forgetting-matrix machinery.
# ---------------------------------------------------------------------------
cells.append(md("""\
## `ceqjepa.curriculum` helpers (`_bed_ns`, `_splice`, `eval_bed`, `print_matrix`)
-- inlined verbatim from `ceqjepa/curriculum.py`, the already-tested
forgetting-matrix machinery (splice the CURRENT shared trunk with each bed's
OWN last-trained head, isolating trunk drift from the unavoidable head
reset when nA/x_dim change across a phase boundary)."""))
curriculum_src = (ROOT / "curriculum.py").read_text()
# Keep only _bed_ns through print_matrix -- drop the imports/REPO_ROOT/PHASES
# preamble, the subprocess-driving run_phase() (superseded by the in-process
# run_phase(args) extracted from train.py below), and main() (superseded by
# the orchestration cell below).
curr_body = "def _bed_ns(" + curriculum_src.split("def _bed_ns(", 1)[1]
curr_body = curr_body.split("def main():")[0]
curr_body = "import argparse\n\n\n" + curr_body  # _bed_ns builds an argparse.Namespace
assert all(f"def {name}(" in curr_body for name in ("_bed_ns", "_splice", "eval_bed", "print_matrix"))
cells.append(code(curr_body.rstrip() + "\n"))

# ---------------------------------------------------------------------------
# run_phase(args) -- train.py's own main(), post-argparse, extracted
# verbatim and patched only for device placement (see module docstring).
# ---------------------------------------------------------------------------
cells.append(md("""\
## `run_phase(args)` -- `ceqjepa/train.py`'s own `main()`, everything after
`args = ap.parse_args()`, extracted verbatim into a callable (so the
orchestration cell below can build an `args` object per phase instead of a
CLI invocation) and patched ONLY to move `bed.batch(...)`'s output, `q_bar`
and `q_floor_table` onto `device` -- the resume / warm-start / checkpoint /
kappa-warning logic itself is byte-for-byte train.py's, untouched."""))

main_src = train_src.split("def main():", 1)[1]
_, _, after_args = main_src.partition("args = ap.parse_args()\n")
body = after_args.split("\nif __name__ == '__main__':")[0]
run_phase_src = "def run_phase(args):\n" + body.rstrip() + "\n"
run_phase_src = re.sub(
    r"(\n( *)q_bar = q_pool\.mean\(0\)\n)",
    lambda m: m.group(1)[:-1].replace("q_bar = q_pool.mean(0)", "q_bar = q_pool.mean(0).to(device)") + "\n",
    run_phase_src,
)
run_phase_src = run_phase_src.replace(
    "q_bar = ckpt['q_bar']\n", "q_bar = ckpt['q_bar'].to(device)\n"
)
run_phase_src = run_phase_src.replace(
    "q_floor_table = bed.q_floor() if hasattr(bed, 'q_floor') else None\n",
    "q_floor_table = bed.q_floor().to(device) if hasattr(bed, 'q_floor') else None\n",
)
run_phase_src = move_batch_to_device(run_phase_src, "train_bed", "train_gen", "args.batch_size")
assert "def run_phase(args):" in run_phase_src and "save_checkpoint(args.out" in run_phase_src
cells.append(code(run_phase_src.rstrip() + "\n"))

# ---------------------------------------------------------------------------
# Orchestration: config, resume-on-restart detection, phase loop, forgetting
# matrix after every phase.
# ---------------------------------------------------------------------------
cells.append(md("""\
## Curriculum config

All step/size knobs are environment-variable overridable (`CEQJEPA_*`) so a
CPU smoke test (extract the cells to a script, set tiny values) exercises
the exact same code path as the real Kaggle GPU run, just smaller and
faster -- not a second, hand-simplified copy of the loop below."""))

config_cell = """\
from pathlib import Path

PHASES = ('chess', 'english', 'markets')
CKPT_DIR = Path(os.environ.get("CEQJEPA_CKPT_DIR", "/kaggle/working"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ckpt_path = {p: CKPT_DIR / f"curriculum_{p}.pt" for p in PHASES}

STEPS_PER_PHASE_DEFAULT = int(os.environ.get("CEQJEPA_STEPS_PER_PHASE", "6000"))
STEPS_PER_PHASE = {p: int(os.environ.get(f"CEQJEPA_STEPS_{p.upper()}", STEPS_PER_PHASE_DEFAULT))
                    for p in PHASES}
CKPT_EVERY      = int(os.environ.get("CEQJEPA_CKPT_EVERY", "250"))    # D7: pause loses <= this many steps
EVAL_EVERY      = int(os.environ.get("CEQJEPA_EVAL_EVERY", "250"))
EVAL_N          = int(os.environ.get("CEQJEPA_EVAL_N", "128"))
MATRIX_EVAL_N   = int(os.environ.get("CEQJEPA_MATRIX_EVAL_N", "256"))  # needs to be large: chess
                                                                        # self-play is ~98% sink
N_GAMES         = int(os.environ.get("CEQJEPA_N_GAMES", "800"))
KAPPA_CEILING   = float(os.environ.get("CEQJEPA_KAPPA_CEILING", str(KAPPA_DESIGN_BOUND_F32)))
LR              = float(os.environ.get("CEQJEPA_LR", "3e-4"))  # ONE constant lr for the whole
                                                                 # curriculum -- see top markdown
                                                                 # cell on the --lr-on-resume gap

# Shared architecture across all 3 phases -- this is what makes the trunk
# (L0/Vt/delta_a/delta_b/chart/enc.2) shape-compatible and splice-able
# across phases; changing these mid-curriculum defeats that.
common = dict(n=int(os.environ.get("CEQJEPA_N", "64")),
              d_enc=int(os.environ.get("CEQJEPA_D_ENC", "64")),
              rank=int(os.environ.get("CEQJEPA_RANK", "8")),
              z_dim_state=6, g=0.9,
              batch_size=int(os.environ.get("CEQJEPA_BATCH_SIZE", "64")),
              lr=LR, x_dim=769)  # x_dim forced equal for chess+english (chess is fixed at
                                 # 769; english is configurable) -- a TRUE bitwise warm start,
                                 # matching ceqjepa/curriculum.py's own choice.

print(f"[curriculum] PHASES={PHASES} steps_per_phase={STEPS_PER_PHASE} ckpt_every={CKPT_EVERY} "
      f"n_games={N_GAMES} kappa_ceiling={KAPPA_CEILING:.6f} common={common}")
"""
cells.append(code(config_cell))

cells.append(md("""\
## The phase loop

Resume-on-restart: for each phase, if `/kaggle/working/curriculum_<phase>.pt`
already exists AND its own `phase` field matches, this is a phase already
in progress -- resume it and run only the REMAINING steps (train.py's
`--steps` already means "new steps this call", so `remaining = target -
already_step`). If it doesn't exist yet but the PREVIOUS phase's checkpoint
does, warm-start from that instead (the cross-phase splice train.py's
`run_phase` already implements). The forgetting matrix is recomputed and
reprinted after every phase, including a phase that needed zero additional
training this call (a restart right after a phase finished but before the
matrix printed)."""))

loop_cell = """\
head_state = {}
matrix_rows = []
prior_ckpt = None

for phase in PHASES:
    path = ckpt_path[phase]
    target_steps = STEPS_PER_PHASE[phase]
    resume_from, already_step = None, 0

    if path.exists():
        probe = torch.load(path, map_location='cpu', weights_only=False)
        if probe.get('phase') == phase:
            resume_from, already_step = path, probe['step']
            print(f"[curriculum] found existing {phase!r} checkpoint at step={already_step} "
                  f"(target {target_steps})")
    elif prior_ckpt is not None:
        resume_from = prior_ckpt
        print(f"[curriculum] no {phase!r} checkpoint yet -- warm-starting from {prior_ckpt}")

    remaining = max(0, target_steps - already_step)
    if remaining == 0 and path.exists():
        print(f"[curriculum] phase={phase!r} already at/above target step {target_steps}, "
              f"skipping training this call")
    else:
        args = types.SimpleNamespace(
            geometry='tiny', steps=remaining, seed=0, device=str(device), out=str(path),
            max_seconds=None, n=common['n'], nA=4, d_enc=common['d_enc'], x_dim=common['x_dim'],
            z_dim=4, z_dim_state=common['z_dim_state'], g=common['g'], rank=common['rank'],
            batch_size=common['batch_size'], eval_every=EVAL_EVERY, eval_n=EVAL_N,
            lr=common['lr'], ckpt_every=CKPT_EVERY, lambda_z=0.5, kappa_ceiling=KAPPA_CEILING,
            resume=str(resume_from) if resume_from else None, phase=phase, bed=phase,
            n_games=N_GAMES,
        )
        print(f"[curriculum] === phase={phase} steps_this_call={remaining} "
              f"resume={args.resume} ===")
        t0 = time.time()
        run_phase(args)
        print(f"[curriculum] phase={phase} done in {time.time() - t0:.1f}s -> {path}")

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    trunk_state = ckpt['model_state_dict']
    head_state[phase] = trunk_state   # this phase's own head is exactly this checkpoint
    prior_ckpt = path

    seen = [p for p in PHASES if PHASES.index(p) <= PHASES.index(phase)]
    scores = {}
    for bed_name in seen:
        ev = eval_bed(bed_name, trunk_state, head_state[bed_name], common,
                      eval_seed=999_000 + PHASES.index(bed_name), n_games=N_GAMES,
                      eval_n=MATRIX_EVAL_N)
        scores[bed_name] = ev['S']
        tag = '(current)' if bed_name == phase else '(forgetting probe)'
        print(f"[curriculum] after phase={phase}: eval on bed={bed_name} S={ev['S']:+.4f} "
              f"mse_model={ev['mse_model']:.4e} mse_bar={ev['mse_bar']:.4e} {tag}")
    matrix_rows.append((f'after {phase}', scores))

    print(f"\\n[curriculum] FORGETTING MATRIX after phase={phase} "
          f"(S = 1 - mse_model/mse_bar; blank cell = bed not reached yet):")
    print_matrix(matrix_rows, list(PHASES))
    print()

print("[curriculum] all phases complete.")
"""
cells.append(code(loop_cell))

cells.append(md("## Manifest"))
cells.append(code("""\
manifest = dict(
    phases=list(PHASES),
    steps_per_phase=STEPS_PER_PHASE,
    ckpt_every=CKPT_EVERY,
    n_games=N_GAMES,
    kappa_ceiling=KAPPA_CEILING,
    common=common,
    device=str(device),
    torch_version=torch.__version__,
    python_version=platform.python_version(),
    forgetting_matrix=[(label, scores) for label, scores in matrix_rows],
    checkpoint_paths={p: str(ckpt_path[p]) for p in PHASES},
)
(CKPT_DIR / "ceqjepa_curriculum_manifest.json").write_text(json.dumps(manifest, indent=2))
print("wrote", CKPT_DIR / "ceqjepa_curriculum_manifest.json")
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
print("wrote", OUT, f"({OUT.stat().st_size} bytes, {len(cells)} cells)")
