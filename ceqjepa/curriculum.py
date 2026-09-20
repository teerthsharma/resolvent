"""ceqjepa/curriculum.py -- drives the chess -> english -> markets curriculum
(BUILD SPEC: three sequential phases, each resuming from the previous
phase's checkpoint), plus the MANDATORY forgetting matrix.

TRAINING is three ceqjepa/train.py invocations chained by subprocess +
--resume -- reusing the actual, already-tested resume machinery rather than
a second training loop. Nothing here retrains anything: it drives the CLI.

WHY chess AND english SHARE --x-dim. ChessBed's x_dim=769 is fixed (baked
into its board encoding); EnglishBed's x_dim is a free constructor arg. This
script forces english's --x-dim to chess's 769, so phase 1 -> phase 2 is a
TRUE full bitwise warm start (nA is already equal: both beds are 4-outcome).
MarketsBed's nA=2 and x_dim=3 are both fixed by the lattice and cannot be
matched -- phase 2 -> phase 3 genuinely resets a task head (train.py prints
this at the transition: "cross-phase warm start ... RESET (shape mismatch,
new task head) [...]").

THE FORGETTING MATRIX -- what it actually measures. After phase p, evaluate
p's checkpoint on every bed seen so far (BUILD SPEC: this is the whole point
of sequencing -- the lambda_z ablation already showed an encoder-side
auxiliary costs -0.0985 S on the committor read, so forgetting is the
EXPECTED failure mode and must be visible, not inferred). But a later
phase's checkpoint may no longer HOLD an earlier bed's task head at all once
shapes changed (markets overwrote the 4-outcome readout with a 2-outcome
one -- the old tensor is gone, not just stale). So evaluating phase p's
checkpoint on an earlier bed q splices: the CURRENT shared trunk (L0, Vt,
delta_a/delta_b, chart, enc.2 -- the actual committor machinery, the thing
that can be "forgotten") from p's checkpoint, with bed q's OWN
last-trained task head (enc.0, readout, absorbing_idx -- the input/output
width, an architecture fact tied to q's geometry, not a forgetting signal)
from the checkpoint q's own phase wrote. This isolates trunk drift from the
unavoidable head reset. Keys are matched generically by comparing shapes
between the two checkpoints -- nothing here hardcodes TinyCEQ's parameter
names twice.

UPLOAD NOTHING, COMMIT NOTHING: this script only runs `python -m
ceqjepa.train` as a subprocess and writes checkpoints under --out-dir. No
kaggle/huggingface/git command is ever invoked.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from ceqjepa.train import TinyCEQ, evaluate, make_bed

REPO_ROOT = Path(__file__).resolve().parent.parent
# Order set by the author: the phase you most want PRESERVED trains LAST,
# because forgetting runs forward. The measured lambda_z ablation (L_q only
# S=+0.5056 sd 0.1039 vs L_q+0.5*L_z S=+0.4070 sd 0.0962, delta -0.0985 on all
# three seeds) says the encoder-side objective damages the committor read, so
# English goes FIRST where chess and markets can specialise on top of it.
PHASES = ('english', 'chess', 'markets')


def run_phase(phase, resume_path, out_path, common, seed, steps, max_seconds, n_games):
    cmd = [sys.executable, '-m', 'ceqjepa.train',
           '--bed', phase, '--phase', phase,
           '--steps', str(steps), '--seed', str(seed), '--out', str(out_path),
           '--n', str(common['n']), '--d-enc', str(common['d_enc']),
           '--rank', str(common['rank']), '--z-dim-state', str(common['z_dim_state']),
           '--g', str(common['g']), '--batch-size', str(common['batch_size']),
           '--lr', str(common['lr']), '--eval-every', str(common['eval_every']),
           '--eval-n', str(common['eval_n']), '--x-dim', str(common['x_dim'])]
    if phase == 'chess':
        cmd += ['--n-games', str(n_games)]
    if max_seconds is not None:
        cmd += ['--max-seconds', str(max_seconds)]
    if resume_path is not None:
        cmd += ['--resume', str(resume_path)]
    print(f"[curriculum] === phase={phase} ===\n[curriculum] $ {' '.join(cmd)}")
    t0 = time.time()
    import subprocess
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"[curriculum] phase={phase} failed (exit {proc.returncode})")
    print(f"[curriculum] phase={phase} done in {time.time() - t0:.1f}s -> {out_path}")


def _bed_ns(bed, common, seed, n_games):
    """The minimal argparse.Namespace make_bed() needs to construct a bed."""
    return argparse.Namespace(bed=bed, n=common['n'], x_dim=common['x_dim'],
                               z_dim=common.get('z_dim', 4), seed=seed, n_games=n_games)


def _splice(trunk_state, head_state):
    """Keys whose shape DIFFERS between the two states are task-specific
    (width tied to a bed's own nA/x_dim) -- take those from head_state, the
    bed's own last-trained head. Keys whose shape MATCHES are the shared
    committor machinery -- take those from trunk_state, so the eval reflects
    what training THROUGH the current phase left in the shared trunk."""
    out = {}
    for k, v in head_state.items():
        t = trunk_state.get(k)
        out[k] = t if (t is not None and t.shape == v.shape) else v
    return out


def eval_bed(bed_name, trunk_state, head_state, common, eval_seed, n_games, eval_n):
    """Build bed_name's own model geometry, splice in the current trunk, and
    score it -- returns the evaluate() dict (S is the number that matters)."""
    ns = _bed_ns(bed_name, common, eval_seed, n_games)
    bed = make_bed(ns)  # overwrites ns.nA / ns.x_dim to bed_name's real geometry
    q_floor_table = bed.q_floor() if hasattr(bed, 'q_floor') else None
    model = TinyCEQ(n=common['n'], nA=ns.nA, d_enc=common['d_enc'], x_dim=ns.x_dim,
                     z_dim_state=common['z_dim_state'], g=common['g'], rank=common['rank'],
                     absorbing_idx=torch.arange(ns.nA))
    model.load_state_dict(_splice(trunk_state, head_state))
    model.eval()
    gen = torch.Generator().manual_seed(eval_seed)
    _, _, q_pool, _ = bed.batch(gen, max(eval_n, common['batch_size']))  # fresh q_bar for THIS bed
    q_bar = q_pool.mean(0)
    return evaluate(model, bed, q_bar, q_floor_table, gen, eval_n)


def print_matrix(rows, cols):
    """rows: list of (phase_label, {bed: S or None}). cols: bed names, in order."""
    header = "phase".ljust(10) + "".join(c.rjust(12) for c in cols)
    print(header)
    for label, scores in rows:
        line = label.ljust(10)
        for c in cols:
            s = scores.get(c)
            line += (f"{s:+.4f}".rjust(12) if s is not None else "n/a".rjust(12))
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'ceqjepa' / 'artifacts' / 'curriculum'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=300, help='per-phase step budget, all 3 phases')
    ap.add_argument('--steps-chess', type=int, default=None)
    ap.add_argument('--steps-english', type=int, default=None)
    ap.add_argument('--steps-markets', type=int, default=None)
    ap.add_argument('--max-seconds', type=float, default=None, help='per-phase wall-clock cap, all 3 phases')
    ap.add_argument('--n-games', type=int, default=300, help='chess: self-play games')
    ap.add_argument('--matrix-eval-n', type=int, default=256,
                     help='eval batch size for the forgetting-matrix probes specifically '
                          '(separate from --eval-n, which is the per-step training diagnostic). '
                          'Needs to be large: chess self-play is ~98% sink-outcome (measured), so '
                          'a small eval batch can land all-sink and make mse_bar exactly 0 -- a '
                          'real, reported degeneracy (S undefined, 0/0), not a bug -- rather than a '
                          'meaningful constant-predictor control.')
    # shared model architecture across all 3 phases -- this is what makes
    # the trunk (L0/Vt/delta_a/delta_b/chart/enc.2) shape-compatible and so
    # splice-able across phases; changing these mid-curriculum defeats that.
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--d-enc', type=int, default=16)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--z-dim-state', type=int, default=6)
    ap.add_argument('--g', type=float, default=0.9)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--eval-n', type=int, default=64)
    ap.add_argument('--x-dim', type=int, default=769,
                     help='forced equal for chess+english (chess is fixed at 769; '
                          'english is configurable and matches it here so phase '
                          '1->2 is a full warm start, no head reset)')
    args = ap.parse_args()

    common = dict(n=args.n, d_enc=args.d_enc, rank=args.rank, z_dim_state=args.z_dim_state,
                  g=args.g, batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                  eval_n=args.eval_n, x_dim=args.x_dim)
    steps = dict(chess=args.steps_chess or args.steps,
                 english=args.steps_english or args.steps,
                 markets=args.steps_markets or args.steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = {p: out_dir / f'{p}.pt' for p in PHASES}
    head_state = {}   # phase -> that phase's OWN last-trained model_state_dict
    matrix_rows = []

    resume = None
    for phase in PHASES:
        run_phase(phase, resume, ckpt_path[phase], common, args.seed, steps[phase],
                  args.max_seconds, args.n_games)
        resume = ckpt_path[phase]

        ckpt = torch.load(ckpt_path[phase], map_location='cpu', weights_only=False)
        trunk_state = ckpt['model_state_dict']
        head_state[phase] = trunk_state  # this phase's own head is exactly this checkpoint's

        seen = [p for p in PHASES if PHASES.index(p) <= PHASES.index(phase)]
        scores = {}
        for bed_name in seen:
            ev = eval_bed(bed_name, trunk_state, head_state[bed_name], common,
                          eval_seed=args.seed + 999_000 + PHASES.index(bed_name),
                          n_games=args.n_games, eval_n=args.matrix_eval_n)
            scores[bed_name] = ev['S']
            tag = '(current)' if bed_name == phase else '(forgetting probe: current trunk + its own frozen head)'
            print(f"[curriculum] after phase={phase}: eval on bed={bed_name} S={ev['S']:+.4f} "
                  f"mse_model={ev['mse_model']:.4e} mse_bar={ev['mse_bar']:.4e} {tag}")
        matrix_rows.append((f'after {phase}', scores))

    print("\n[curriculum] FORGETTING MATRIX (S = 1 - mse_model/mse_bar; rows = phase just "
          "finished, cols = bed evaluated; blank cell = bed not reached yet):")
    print_matrix(matrix_rows, list(PHASES))


if __name__ == '__main__':
    main()
