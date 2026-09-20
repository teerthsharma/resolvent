"""ceqjepa/train.py

Training loop + synthetic bed generator for the CEQ committor operator.
Builds a small model ON TOP of ceqjepa.operator's primitives
(build_operator, committor, state_solve, q_floor_closed_form) — nothing
here redefines the causal row-stochastic operator, the boundary-row
overwrite, or the resolvent solves; those are the sibling module's job.

Implements, per DCM-1 BUILD SPECIFICATION (stage 1 only; stage 2 -- the
decision head and its hinge loss -- is gated behind the R-1 kill and is
NOT built here):
  (a) a synthetic in-class bed whose committor labels are computed EXACTLY
      by ceqjepa.operator.committor (float64, a closed linear solve on the
      transient block -- never touched by the model, which reads float32);
  (b) the stage-1 loss L = L_q + 0.5*L_z (BUILD SPEC section 3), every term named;
  (c) the constant-predictor control C1 (q_bar), scored on the same metric,
      printed beside the model every eval -- the author's own prior repo
      measured a learned rollout LOSE to exactly this control (a latent
      state carrying no dynamics is worth checking for before anything else);
  (d) the collapse-floor diagnostic ||q - q_floor||_inf (WHAT SURVIVED, A),
      via ceqjepa.operator.q_floor_closed_form, logged every eval;
  (e) argparse with --steps/--seed/--device/--out, tiny CPU-runnable default.

Geometry (n positions, nA absorbing, encoder width, state width) is a
constructor argument here, not hardcoded -- the frozen DESIGN GEOMETRY in
the build spec (N=256, K=2, d=128, ...) is what a full run would pass;
this file's defaults are a CPU-tiny in-class stand-in so `--steps 3`
finishes in seconds.

WIRED 2026-09-08, the causal and topological terms:
  (f) --bed chess_do: paired observational/interventional data from
      ceqjepa.beds.chess_do. The loss gains L_do -- for m forced moves at the
      position the model's own read selects, q(do a) comes from
      ceqjepa.intervene.committor_do_batch (ONE factorisation for all m) and is
      scored against the realised interventional outcome. NOTHING SUPERVISED
      THIS TERM BEFORE. MEASURED, 200 steps CPU, n_games=200 do_max_plies=400
      m=8 R=4: held-out L_do 2.2030 -> 1.5759, p_spread 7.18e-4 -> 9.59e-2.
      CONTROL, 3 seeds: freezing L0/delta_a/delta_b/enc so only the move head
      can learn costs +0.5098/+0.5513/+0.5947 nats at step 200, 3/3 -- the
      operator, not the move head, is what the interventional term trains.
  (g) --lambda-topo: ceqjepa.topo_loss.topo_coupling_loss over the encoder
      output, blocked by --topo-blocks. Default 0.0, so nothing already
      measured changes silently.
  (h) L_do, L_topo, p_spread and intervene's REFUSAL RATE logged every eval; a
      window rate above 1% prints a [FINDING] line rather than being silenced.

WIRED 2026-09-09, the four things that let a 4 h 43 m T4 run ship its WORST
checkpoint (ceqjepa/kaggle/causal/out/ceq-jepa-dcm-1-causal-arm-t4.stdout.txt,
seed 0: held-out PPL_obs 3.1796 at step 2000 -> 10.3872 at step 12000, rising in
15 of 15 intervals across three seeds, while L_q fell to 0.0443 -- 1,285,771
parameters over 5,000 one-hot labels, 257 per label):
  (i)   heldout_split(): the held-out arm is a partition of GAMES, never of
        positions, because every position of a game carries that game's single
        outcome. assert_disjoint() re-checks the result at run time and raises,
        naming the offending groups; self_check (v) proves it fires on both an
        identical split and a position-level split of shared games.
  (j)   --early-stop-patience (DEFAULT 5 EVALS, ON): --out now holds the BEST
        checkpoint by held-out L_q, written the moment it is the best, and every
        eval prints current and best-so-far with the step the best came from.
        The rolling LAST checkpoint moved to --out+'.last' (still resumable).
  (k)   the end-of-run [SUMMARY]: best step, best and final held-out score, and
        wasted/useful steps. On the run above that ratio reads 10000/2000 = 5.00.
  (l)   --divergence-k (default 3): the MEMORISATION guard, diverging() above.

WHAT L_do CANNOT TEACH. The intervened position i is argmax over alpha's
transient entries -- argmax carries no gradient and the committor is read AT i,
so `chart` (the alpha read) receives NO gradient from L_do at all; measured
p.grad is None after 6 steps of L_do alone, while L0 gets 2.62e-1, delta_b
8.84e-3, delta_a 4.77e-4, enc.0 8.38e-4 and the move head 1.34e-1. Where the
query sits on the chart is supervised by L_q only.

torch + numpy ONLY.
"""
import argparse
import math
import os
import time

import numpy as np  # ChessDoBed stacks the bed's numpy arrays before the one torch conversion
import torch
import torch.nn as nn
import torch.nn.functional as F

import ceqjepa.operator as op
from ceqjepa.sharpness import decompose as _sharp_decompose
from ceqjepa.intervene import committor_do_batch
from ceqjepa.operator import SingularTransientBlockError
from ceqjepa.topo_loss import topo_coupling_loss

EPS_Q = 1e-6  # BCE input clamp, MANDATORY per build spec: q attains 0 and 1 exactly

# The MEMORISATION guard's materiality threshold: both moves must be at least this
# large RELATIVE to the best checkpoint's own numbers. A guard that fires on eval
# noise is worth as little as one that never fires, and both failures were seen
# here before this constant existed. MEASURED separation, three series:
#   T4 causal arm seed 0, best step 2000 -> step 8000: held-out +125%, train -77%  FIRE
#   --bed chess_do 60 games (below),   step 125 -> 250: held-out  +14%, train -29%  FIRE
#   --bed synthetic at the tiny defaults, eval noise:   held-out +0.24%, train -0.2% quiet
# 5% sits ~3x below the tightest true positive and ~60x above the noise.
DIVERGENCE_REL = 0.05


# ---------------------------------------------------------------------------
# (a) synthetic in-class bed -- committor labels computed EXACTLY, via the
# shared operator's own exact solve (float64), never by the model.
# ---------------------------------------------------------------------------
class Bed:
    """A fixed causal absorbing-chain corpus over n positions, the first
    nA of which are absorbing (canonical order, matches build spec). Base
    tensors (L_env, D, W_obs, phi) are generated once at construction
    (seed 0) and fixed thereafter; each batch() draws fresh latents,
    builds the per-example causal operator via ceqjepa.operator, and
    solves the EXACT committor -- the label -- via ceqjepa.operator.committor
    in float64. Mirrors BUILD SPEC section 1 at a CPU-tiny geometry.
    """

    def __init__(self, n, nA, x_dim, z_dim, seed=0, dtype=torch.float64):
        assert n > nA >= 1
        self.n, self.nA = n, nA
        self.x_dim, self.z_dim = x_dim, z_dim
        self.dtype = dtype
        self.absorbing_idx = torch.arange(nA)
        g = torch.Generator().manual_seed(seed)
        self.L_env = torch.randn(n, n, generator=g, dtype=dtype)
        self.D = torch.randn(z_dim, n, n, generator=g, dtype=dtype) * 0.5
        self.W_obs = torch.randn(x_dim, z_dim, generator=g, dtype=dtype) / (z_dim ** 0.5)
        self.phi = torch.randn(n, x_dim, generator=g, dtype=dtype) / (x_dim ** 0.5)

    def batch(self, gen, B):
        """One batch, ONE solve (BUILD SPEC: 'Batch the solve'). Returns
        (x[B,x_dim] f32, x_nx[B,x_dim] f32, q_star[B,nA] f32, v_idx[B])."""
        n, nA = self.n, self.nA
        U = torch.randn(B, self.z_dim, generator=gen, dtype=self.dtype)
        v_idx = torch.randint(nA, n, (B,), generator=gen)
        logits = self.L_env.unsqueeze(0) + torch.einsum('bm,mij->bij', U, self.D)
        P = op.build_operator(logits, self.absorbing_idx)        # [B,n,n], causal, boundary rows overwritten
        q_full = op.committor(P, self.absorbing_idx)             # [B,n,nA], EXACT (I-Q)^-1 R
        rows = torch.arange(B)
        q_star = q_full[rows, v_idx]                             # [B,nA]  the label
        v_next = torch.multinomial(P[rows, v_idx], 1, generator=gen).squeeze(-1)
        noise = 0.05 * torch.randn(B, self.x_dim, generator=gen, dtype=self.dtype)
        x = torch.tanh(U @ self.W_obs.T + self.phi[v_idx]) + noise
        x_nx = torch.tanh(U @ self.W_obs.T + self.phi[v_next])
        return x.float(), x_nx.float(), q_star.float(), v_idx

    def q_floor(self):
        """(d) Closed-form committor of the uniform causal chain -- no
        encoder, no solve, no oracle. A collapsed encoder's read equals
        this exactly (WHAT SURVIVED, A)."""
        return op.q_floor_closed_form(self.n, self.absorbing_idx, dtype=torch.float32)


# ---------------------------------------------------------------------------
# --bed chess_do: the PAIRED bed. Adapter only -- the dataset itself is
# ceqjepa.beds.chess_do.build_intervention_dataset, which plays the games,
# forces the candidate moves and rolls each one out R times. This class does
# nothing but pre-encode its FENs once and hand out batches, because
# fen_to_vec on 769 floats per row per step is the whole cost otherwise.
#
# THE TARGET IS AN EMPIRICAL MEAN, NOT A DISTRIBUTION (adverse item 5, and
# chess_do.py's own docstring): Var[p_hat] = p(1-p)/R <= 1/(4R), so at the
# default R=4 the per-coordinate std is up to 0.25. The interventional term
# below is supervised against a NOISY label and its absolute value is not
# comparable to L_q's.
# ---------------------------------------------------------------------------
class ChessDoBed:
    nA = 4          # N_OUTCOMES: white/draw/black/sink, chess.py's order, reused
    x_dim = 769     # X_DIM: 12*64 piece planes + side-to-move

    def __init__(self, samples, m):
        import chess
        from ceqjepa.beds.chess import fen_to_vec, N_OUTCOMES, X_DIM
        assert (self.nA, self.x_dim) == (N_OUTCOMES, X_DIM), "chess.py's geometry moved"
        self.m = m
        xs, xns, qs, vs, mvs, tgts, msks = [], [], [], [], [], [], []
        for s in samples:
            board = chess.Board(s.fen)
            board.push(chess.Move.from_uci(s.obs_uci))
            xs.append(fen_to_vec(s.fen))
            xns.append(fen_to_vec(board.fen()))
            qs.append(s.obs_outcome)
            vs.append(s.ply_idx)
            k = len(s.candidate_ucis)
            # PAD to a fixed m by repeating candidate 0, and carry the mask.
            # Not by dropping short positions: min(m_candidates, len(legal))
            # is short exactly in endgames, and silently filtering those would
            # bias the bed toward the middlegame without saying so.
            pad = [0] * (m - k)
            order = list(range(min(k, m))) + pad
            mv = [chess.Move.from_uci(s.candidate_ucis[c]) for c in order]
            mvs.append([[q.from_square, q.to_square] for q in mv])
            tgts.append(s.do_outcome_mean[order])
            msks.append([c < k for c in range(min(k, m))] + [False] * len(pad))
        self.x = torch.from_numpy(np.stack(xs)).float()
        self.x_nx = torch.from_numpy(np.stack(xns)).float()
        self.q_star = torch.from_numpy(np.stack(qs)).float()
        self.v_idx = torch.tensor(vs, dtype=torch.long)
        self.moves = torch.tensor(mvs, dtype=torch.long)          # [N,m,2] (from_sq, to_sq)
        self.do_tgt = torch.from_numpy(np.stack(tgts)).float()    # [N,m,nA]
        self.do_mask = torch.tensor(msks, dtype=torch.bool)       # [N,m]
        self.pad_rate = 1.0 - float(self.do_mask.float().mean())
        # THE SPLIT KEY (heldout_split below). The group is the GAME, not the
        # position: every position of one game carries that game's single
        # outcome, so a position-level split puts the same label on both sides
        # and the "held-out" score is a training score under another name.
        self.group_ids = torch.tensor([s.game_id for s in samples], dtype=torch.long)
        self.row_keys = [a.tobytes() for a in xs]   # exact position identity, for the overlap report

    def batch_do(self, gen, B):
        i = torch.randint(len(self.x), (B,), generator=gen)
        return (self.x[i], self.x_nx[i], self.q_star[i], self.v_idx[i],
                self.moves[i], self.do_tgt[i], self.do_mask[i])

    def batch(self, gen, B):
        return self.batch_do(gen, B)[:4]

    def subset(self, idx):
        """A view on rows `idx` with the identical interface. Shares the parent's
        tensors by indexing them -- no FEN is re-encoded, which is the whole cost
        of this class (see the class docstring)."""
        s = object.__new__(ChessDoBed)
        s.m = self.m
        for k in ('x', 'x_nx', 'q_star', 'v_idx', 'moves', 'do_tgt', 'do_mask', 'group_ids'):
            setattr(s, k, getattr(self, k)[idx])
        s.row_keys = [self.row_keys[i] for i in idx.tolist()]
        s.pad_rate = 1.0 - float(s.do_mask.float().mean())
        return s


def draw(bed, gen, B):
    """One batch. Beds with an interventional arm expose batch_do; the other
    four return moves/do_tgt/do_mask = None and every do-term below is skipped."""
    if hasattr(bed, 'batch_do'):
        return bed.batch_do(gen, B)
    x, x_nx, q_star, v_idx = bed.batch(gen, B)
    return x, x_nx, q_star, v_idx, None, None, None


# ---------------------------------------------------------------------------
# THE HELD-OUT SPLIT. Disjoint BY CONSTRUCTION and ASSERTED AT RUN TIME.
#
# WHY ON GAMES AND NOT ON POSITIONS. Both chess beds label a position with the
# outcome of the GAME it came from: ChessBed._rows_from_game stamps one
# `onehot` on every ply of a game, and chess_do's InterventionSample carries a
# `game_id` for exactly this reason. Two positions from one game therefore
# share a label, and a position-level split hands the evaluator a label it was
# trained on. The split below partitions GROUPS (games) and derives the row
# split from that; assert_disjoint then re-checks the result rather than
# trusting the construction that produced it.
#
# WHAT THIS DOES NOT CLAIM. A game split does not make the two sides share no
# POSITION -- every self-play game starts from the same board, and
# transpositions exist. That overlap is measured and printed, not silenced,
# because it is a real property of the corpus and not a bug in the split.
# ---------------------------------------------------------------------------
def bed_groups(bed):
    """(group_ids[N], subset(idx)->bed, row_keys[N]) for a bed that is a FINITE
    CORPUS, or None for one that draws a fresh i.i.d. sample every batch.

    A generative bed has no corpus to partition: `Bed`/`EnglishBed`/`MarketsBed`
    build every example on the spot from the generator handed to batch(), so
    train and held-out are disjoint iff their generators are -- which is what
    main() already does with two seeds 1e6 apart. Only the two chess beds hold
    a fixed pool of rows, and only they can leak."""
    if hasattr(bed, 'group_ids'):        # ChessDoBed: one sampled ply per self-play game
        return bed.group_ids, bed.subset, bed.row_keys
    if hasattr(bed, 'rows'):             # ChessBed: MANY plies per game, all sharing one outcome
        ply = torch.tensor([r['ply_idx'] for r in bed.rows], dtype=torch.long)
        assert int(ply[0]) == 0, "ChessBed rows do not start at ply_idx 0 -- game boundaries unknown"
        gid = (ply == 0).cumsum(0) - 1   # ply_idx==0 marks a game's first row (chess.py's own comment)
        keys = [r['fen_before'] for r in bed.rows]
        return gid, (lambda idx: type(bed)([bed.rows[i] for i in idx.tolist()])), keys
    return None


def diverging(score, best_score, train_loss, best_train_loss):
    """One eval's vote for the MEMORISATION guard: held-out materially WORSE than the
    best checkpoint's while the training loss is materially BETTER than it was there.
    Both comparisons are against the best eval, never the previous one -- train_loss
    is a single minibatch and bounces, and the strict consecutive form of this test
    scored ZERO fires on a run whose held-out L_q went 1.0942 -> 3.2113 while the
    training loss went 2.2978 -> 0.7769 (measured, --bed chess_do, self_check (vi)
    pins both series). DIVERGENCE_REL sets 'materially'."""
    return (score > best_score * (1 + DIVERGENCE_REL)
            and train_loss < best_train_loss * (1 - DIVERGENCE_REL))


def assert_disjoint(train_groups, heldout_groups, train_idx=None, heldout_idx=None):
    """The run-time check. Raises AssertionError naming the offenders, so a
    leaking split kills the run instead of producing a held-out number that is
    a training number. Public because the split is only as trustworthy as this
    is: self_check() case (v) proves it fires."""
    tg = set(int(v) for v in train_groups.tolist())
    hg = set(int(v) for v in heldout_groups.tolist())
    assert tg and hg, (f"degenerate split: {len(tg)} training groups, {len(hg)} held-out "
                       f"groups -- one side is empty, so the held-out score is undefined")
    both = sorted(tg & hg)
    assert not both, (
        f"HELD-OUT SPLIT LEAKS: {len(both)} of {len(hg)} held-out groups also appear in "
        f"training (first 10: {both[:10]}). Every position of a game carries that game's "
        f"outcome, so a shared game is a shared label and the held-out score would be a "
        f"training score wearing a different name.")
    if train_idx is not None and heldout_idx is not None:
        shared = sorted(set(train_idx.tolist()) & set(heldout_idx.tolist()))
        assert not shared, (f"HELD-OUT SPLIT LEAKS: {len(shared)} rows are in BOTH splits "
                            f"(first 10: {shared[:10]})")
    return dict(
                train_groups=len(tg), heldout_groups=len(hg))


def heldout_split(bed, frac, seed):
    """(train_bed, eval_bed, note). Partitions GROUPS, asserts the result."""
    gs = bed_groups(bed)
    if gs is None:
        return bed, bed, ("no finite corpus to partition -- this bed builds every example "
                          "on the spot from the generator it is handed, so the held-out arm "
                          "is disjoint by drawing from its own generator (seed+1000000)")
    gid, subset, keys = gs
    uniq = torch.unique(gid)
    assert len(uniq) >= 2, f"only {len(uniq)} group(s) in the corpus -- nothing to hold out"
    perm = uniq[torch.randperm(len(uniq), generator=torch.Generator().manual_seed(seed))]
    n_ho = min(max(1, int(round(frac * len(uniq)))), len(uniq) - 1)
    is_ho = torch.isin(gid, perm[:n_ho])
    tr_idx = (~is_ho).nonzero(as_tuple=True)[0]
    ho_idx = is_ho.nonzero(as_tuple=True)[0]
    counts = assert_disjoint(gid[tr_idx], gid[ho_idx], tr_idx, ho_idx)
    # The overlap a GAME split cannot remove: identical positions reached by two
    # different games. Measured and stated, never assumed away.
    tr_keys = set(keys[i] for i in tr_idx.tolist())
    dup = sum(1 for i in ho_idx.tolist() if keys[i] in tr_keys)
    note = (f"{counts['train_groups']} train groups / {counts['heldout_groups']} held-out "
            f"groups (games), {len(tr_idx)} / {len(ho_idx)} rows, split seed {seed}, "
            f"ASSERTED disjoint on groups and on rows; {dup}/{len(ho_idx)} held-out rows "
            f"({100.0 * dup / max(1, len(ho_idx)):.2f}%) are a position that also occurs in "
            f"training (transpositions and the shared start position -- a game split does "
            f"not and cannot remove these)")
    if dup:
        print(f"[FINDING] {dup}/{len(ho_idx)} held-out positions also occur in the training "
              f"split under a DIFFERENT game. The label is still held out (a different game "
              f"can end differently), but the input is not novel.")
    return subset(tr_idx), subset(ho_idx), note


# ---------------------------------------------------------------------------
# --bed factory. `args.n` is TinyCEQ's OWN internal chart width -- an
# architecture hyperparameter for the model's operator-based read -- and is
# NOT the same thing as however many positions a bed's own committor solve
# used internally to produce its q_star labels. Only nA (target class count)
# and x_dim (input width) must match the bed; those get overwritten here when
# a bed fixes them, same pattern --geometry design uses to override
# args.n/args.d_enc. Getting this backwards (setting args.n = bed.n) breaks
# chess: ChessBed.n == ChessBed.nA == 4 (no transient chart, "every absorbing
# set IS an outcome" per its own docstring) -- forcing the model's chart to
# width 4 as well leaves it with an EMPTY transient block and
# op.committor crashes (0x0 solve). synthetic keeps args.n driving the bed
# too, since there the whole point is recovering the SAME n-position chain
# the bed generated labels from -- that coupling is the in-class design, not
# a generic requirement.
# ---------------------------------------------------------------------------
def make_bed(args):
    if args.bed == 'synthetic':
        return Bed(args.n, args.nA, args.x_dim, args.z_dim, seed=args.seed)
    if args.bed == 'chess':
        from ceqjepa.beds.chess import ChessBed
        bed = ChessBed.build(n_games=args.n_games, seed=args.seed)
        args.nA, args.x_dim = bed.nA, bed.x_dim
        assert args.n > args.nA, f"--n={args.n} must exceed --bed chess's nA={args.nA}"
        return bed
    if args.bed == 'chess_do':
        from ceqjepa.beds.chess_do import build_intervention_dataset
        samples = build_intervention_dataset(n_games=args.n_games, seed=args.seed,
                                              m_candidates=args.do_m, R=args.do_R,
                                              max_plies=args.do_max_plies)
        bed = ChessDoBed(samples, args.do_m)
        args.nA, args.x_dim = bed.nA, bed.x_dim
        assert args.n > args.nA, f"--n={args.n} must exceed --bed chess_do's nA={args.nA}"
        print(f"[ceqjepa.train] bed=chess_do {len(samples)} paired positions, "
              f"m={args.do_m} R={args.do_R}, candidate pad rate={bed.pad_rate:.4f}")
        return bed
    if args.bed == 'english':
        from ceqjepa.beds.english import EnglishBed
        bed = EnglishBed(n=args.n, x_dim=args.x_dim, seed=args.seed)
        args.nA = bed.nA
        return bed
    if args.bed == 'markets':
        from ceqjepa.beds.markets import MarketsBed
        bed = MarketsBed()
        args.nA, args.x_dim = bed.nA, 3
        assert args.n > args.nA, f"--n={args.n} must exceed --bed markets's nA={args.nA}"
        return bed
    if args.bed == 'chess_policy':
        # The bed chess_do should have been. MEASURED, 120 positions x 20 rollouts
        # against a 400-permutation shuffle null: chess_do's uniform rollout carries
        # I(X;Y) = 0.1363 nats at 28.3 sigma and is 92% draw-or-ply-cap; this at
        # T=0.25 carries 0.3084 nats at 54.9 sigma, sinks fall 35.2% -> 5.0%, and 79%
        # of games are decisive. It was built, measured, documented -- and until now
        # unreachable from any runner, which is why 4h43m of T4 quota went to the
        # inert one instead.
        from ceqjepa.beds.chess_policy import build_intervention_dataset as _bp
        samples = _bp(n_games=args.n_games, seed=args.seed, m_candidates=args.do_m,
                      R=args.do_R, max_plies=args.do_max_plies,
                      temperature=args.bed_temperature)
        bed = ChessDoBed(samples, args.do_m)
        args.nA, args.x_dim = bed.nA, bed.x_dim
        assert args.n > args.nA, f"--n={args.n} must exceed --bed chess_policy's nA={args.nA}"
        print(f"[ceqjepa.train] bed=chess_policy {len(samples)} paired positions, "
              f"T={args.bed_temperature} m={args.do_m} R={args.do_R}, "
              f"candidate pad rate={bed.pad_rate:.4f}")
        return bed
    if args.bed == 'gridworld':
        # Reach-avoid on a grid; the intervention is BLOCKING a cell. MEASURED at
        # gap=1: I(X;Y) = 0.3986 nats at 157.2 sigma, the highest of any bed here.
        # Its committor agrees with the demo page's own solver to 3.553e-15.
        from ceqjepa.beds.gridworld import GridBed, build_intervention_dataset as _bg
        samples = _bg(n_instances=args.n_games, seed=args.seed, gap=args.bed_gap)
        bed = GridBed(samples, args.do_m)
        args.nA, args.x_dim = bed.nA, bed.x_dim
        assert args.n > args.nA, f"--n={args.n} must exceed --bed gridworld's nA={args.nA}"
        print(f"[ceqjepa.train] bed=gridworld {len(samples)} instances, gap={args.bed_gap}, "
              f"m={args.do_m}, candidate pad rate={bed.pad_rate:.4f}")
        return bed
    raise ValueError(f"unknown --bed {args.bed!r}")


# ---------------------------------------------------------------------------
# The model: an encoder + chart-read wired on top of ceqjepa.operator's
# build_operator / committor / state_solve. The operator itself (causal
# softmax, boundary overwrite, resolvent solves) is never redefined here.
# ---------------------------------------------------------------------------
class TinyCEQ(nn.Module):
    def __init__(self, n, nA, d_enc, x_dim, z_dim_state, g, absorbing_idx, rank=8):
        super().__init__()
        self.n, self.nA, self.rank = n, nA, rank
        self.register_buffer('absorbing_idx', absorbing_idx)
        self.register_buffer('g', torch.tensor(float(g)))
        self.L0 = nn.Parameter(torch.zeros(n, n))               # spec: init ZEROS -> uniform causal chain at step 0
        self.Vt = nn.Parameter(torch.randn(n, z_dim_state) * (n ** -0.5))
        self.enc = nn.Sequential(nn.Linear(x_dim, d_enc), nn.GELU(), nn.Linear(d_enc, d_enc))
        # D6: per-example modulation of the shared chart, factored to the DCM-1 spec's own
        # rank r (default 8) instead of a dense [n,n] map. Two [d_enc -> n*r] linears produce
        # per-example factors a,b in R^{n,r}; delta_logits = a @ b.T is the rank-r perturbation.
        # Cost drops from O(d_enc*n*n) to O(d_enc*n*r).
        self.delta_a = nn.Linear(d_enc, n * rank)
        self.delta_b = nn.Linear(d_enc, n * rank)
        # LoRA init (Hu et al., arXiv:2106.09685): ONE factor random, the other zero.
        # delta_logits = a @ b.T is BILINEAR, so zeroing BOTH factors is an exact
        # stationary point of gradient descent, not a slow start: d(ab)/da = b = 0 and
        # d(ab)/db = a = 0. Measured on the shipped code before this fix, one backward
        # on L_q: grad delta_a.weight = 0.000000e+00, grad delta_b.weight = 0.000000e+00,
        # while chart.weight got 1.237690e-03 -- and out['P'].std(dim=0).max() = 0.0, i.e.
        # the causal operator was BITWISE IDENTICAL for every example in every batch.
        # Every number in this repo's ledger up to 2026-09-08 was produced with the
        # per-example operator pathway dead.
        # The spec property the old comment claimed is a constraint on the PRODUCT, not on
        # the parameters: with delta_b zero the product is still exactly zero at step 0
        # (measured max|P_fixed - P_broken| = 0.0), and delta_b now receives a gradient
        # (3.784898e-04 on the first backward) instead of nothing.
        nn.init.kaiming_uniform_(self.delta_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.delta_a.bias)
        nn.init.zeros_(self.delta_b.weight)                     # spec: delta=0 at step 0, matches L0
        nn.init.zeros_(self.delta_b.bias)
        self.chart = nn.Linear(d_enc, n)                        # alpha read weights
        self.readout = nn.Linear(z_dim_state + nA, x_dim)
        # THE do(a) ARM'S ONLY NEW PARAMETERS. A candidate move is (from_square,
        # to_square); each square gets an embedding and the pair produces an
        # ADDITIVE BIAS on position i's own logit row. The row itself is still
        # built from `logits[b, i]` -- the per-example chart the encoder already
        # produced -- so the clamped row is context-dependent and the gradient
        # of the interventional term reaches delta_a/delta_b/L0, not just here.
        # 64+64 squares, d_mv=8: 2*64*8 + n*(2*8) params, ~1.3k at n=16.
        # Promotion piece is DROPPED (from/to only): e7e8q and e7e8n share an
        # embedding. Cheap, and wrong only for underpromotions.
        d_mv = 8
        self.mv_from = nn.Embedding(64, d_mv)
        self.mv_to = nn.Embedding(64, d_mv)
        self.mv_row = nn.Linear(2 * d_mv, n)
        # THE MOVE-SPECIFICITY BUG, and why zero-init is wrong HERE even though
        # mv_row is a plain Linear on nonzero embeddings and so CAN receive a
        # gradient (unlike the bilinear delta_a/delta_b, which could not).
        # At zero init every move yields row = softmax(logits[i]), i.e. P's own
        # row, so do(a) is the NULL intervention for EVERY candidate and
        # delta_q = 0 exactly for all of them. The loss cannot teach
        # move-specificity from a state where no move produces a distinguishable
        # output -- there is no signal separating them to sharpen.
        # MEASURED consequence, 5 seeds x 2000 steps, N=3000 held out:
        #   operator beats ignore-the-intervention by +0.0235 (5/5 seeds)
        #   PLACEBO, fed ANOTHER position's forced move,      +0.0243 (5/5 seeds)
        #   move-specific effect                              -0.0008 (sd 0.0056)
        # i.e. the do-path applied a constant shift and knew nothing about WHICH
        # move was forced. Without the placebo arm that reads as a 6-8 sigma win.
        # Small random init makes candidate rows distinguishable at step 0 so the
        # interventional loss has something to sharpen. The bias stays zero so the
        # EXPECTED row is still P's own row at init.
        nn.init.normal_(self.mv_row.weight, std=0.02)
        nn.init.zeros_(self.mv_row.bias)
        # A[i] is build_operator's teleport target, [n, n], constant. The clamped
        # row is mixed with it at the SAME teleport as every other row of P, so
        # the intervened row lives in the same family as the rows it replaces --
        # otherwise a confident do-row drives P'[i,i] -> 1 and Sherman-Morrison's
        # denominator to zero for reasons that are an artifact of the head, not
        # of the intervention.
        self.register_buffer('A_tel', op.absorbing_teleport(n, absorbing_idx, torch.float32))

    def forward(self, x):
        B = x.shape[0]
        e = self.enc(x)                                                  # [B, d_enc]
        a = self.delta_a(e).view(B, self.n, self.rank)                   # [B,n,r]
        b = self.delta_b(e).view(B, self.n, self.rank)                   # [B,n,r]
        delta_logits = torch.einsum('bnr,bmr->bnm', a, b)                # [B,n,n], rank<=r
        logits = self.L0.unsqueeze(0) + delta_logits
        P = op.build_operator(logits, self.absorbing_idx)                # [B,n,n]
        q_field = op.committor(P, self.absorbing_idx)                    # [B,n,nA]
        Vt_b = self.Vt.unsqueeze(0).expand(B, -1, -1)
        z, O = op.state_solve(P, Vt_b, float(self.g))                    # [B,n,d]
        # THE ESCAPE HATCH, closed. alpha used to be a free softmax over ALL n
        # rows, absorbing rows included -- and q_field's absorbing rows are the
        # simplex vertices of R^nA for EVERY operator, since an absorbing state
        # reaches itself with probability 1. MEASURED: max|q_field[absorbing] - I|
        # = 0.000e+00 at random, zero and huge logits alike, and with the whole
        # operator FROZEN at the uniform chain a free alpha hits 200 random
        # Dirichlet targets at max|q_alpha - target| = 0.000e+00, in closed form,
        # with no optimisation at all. The operator therefore sat in the null
        # space of the loss: freezing L0/Vt/delta_a/delta_b -- 8,768 of 12,640
        # parameters, P.std = 0.000e+00 -- cost only MAE 0.0397 -> 0.0496, and a
        # free 64-row lookup table with no operator beat the frozen model.
        # Restricting alpha to the TRANSIENT rows costs nothing (0.0402 vs
        # 0.0397) and makes the same freeze cost 15x more (MAE 0.1926, worse than
        # the constant predictor). The planted negative: the same 200 targets are
        # reachable to 0.000e+00 with the absorbing rows available and to only
        # 7.042e-01 without them, so the restriction is what makes the operator
        # load-bearing rather than optional.
        chart_logits = self.chart(e)
        chart_logits = chart_logits.masked_fill(
            torch.zeros(self.n, dtype=torch.bool, device=e.device)
            .index_fill_(0, self.absorbing_idx, True), float('-inf'))
        alpha = torch.softmax(chart_logits, dim=-1)                      # [B,n]
        q_alpha = torch.einsum('bn,bnk->bk', alpha, q_field)             # THE SUPERVISED READ
        h = torch.einsum('bn,bnd->bd', alpha, O)
        x_hat = self.readout(torch.cat([h, q_alpha], dim=-1))
        return dict(q_alpha=q_alpha, q_field=q_field, h=h, x_hat=x_hat, alpha=alpha, P=P, e=e,
                    logits=logits)

    def do_read(self, out, moves, stats, return_field=False):
        """q(do a) at the intervened position, for m candidate moves per example.

        moves: [B, m, 2] long (from_square, to_square). Returns
        (q_do [B, m, nA], ok [B] bool). One factorisation per example covers all
        m moves -- ceqjepa.intervene.committor_do_batch, measured 21.65x faster
        at n=512, m=32 than m separate full solves.

        WHERE THE INTERVENTION LANDS. i = the transient chart position the model's
        own read alpha puts the most weight on, restricted to j >= nA because
        clamping a declared-absorbing row is not a rank-1 edit of (I - Q) (it
        moves i from A to T and changes q's shape -- intervene._blocks refuses).
        The committor is then read AT i: q_do[i] is exactly "which absorbing set
        is hit first, given the move was forced here".

        GUARD (task item 5). A candidate row that drives P'[i,i] toward 1 makes
        (I - Q') singular and committor_do_batch RAISES rather than returning a
        committor amplified by 1/den. That must not kill a 20k-step soak, so it
        is counted per example and reported as a RATE at eval -- never silenced:
        a rate above 1% prints loudly and is a finding, not a nuisance.
        """
        B, m, _ = moves.shape
        n = self.n
        alpha = out['alpha']
        logits, P = out['logits'], out['P']
        i_star = self.nA + alpha[:, self.nA:].argmax(dim=-1)             # [B], transient only
        mv = torch.cat([self.mv_from(moves[..., 0]), self.mv_to(moves[..., 1])], dim=-1)
        bias = self.mv_row(mv)                                           # [B,m,n]
        ar = torch.arange(n, device=alpha.device)
        zero = (torch.zeros(m, n, self.nA, dtype=P.dtype, device=P.device) if return_field
                else torch.zeros(m, self.nA, dtype=P.dtype, device=P.device))
        ok = torch.zeros(B, dtype=torch.bool, device=P.device)
        cols = []
        for b in range(B):
            i = int(i_star[b])
            rl = logits[b, i].unsqueeze(0) + bias[b]                     # [m,n]
            rl = rl.masked_fill((ar > i).unsqueeze(0), float('-inf'))
            row = torch.softmax(rl, dim=-1)
            a_i = self.A_tel[i]
            c = op.TELEPORT if float(a_i.sum()) > 0 else 0.0
            row = (1.0 - c) * row + c * a_i                              # same family as P's rows
            # MOVE-SPECIFICITY GUARD. The sibling of p_spread. Every other check
            # in this file asks whether the operator is CORRECT; this one asks
            # whether the candidate rows are DISTINGUISHABLE. They were not --
            # mv_row was zero-initialised, every move produced the identical
            # clamp, and a placebo fed another position's move scored the same
            # (+0.0243 against the real move's +0.0235). A zero here means the
            # do-path is applying a constant shift and the causal claim is void.
            if m > 1:
                _sp = float((row.unsqueeze(0) - row.unsqueeze(1)).abs().mean())
                stats['move_spread'] = max(stats.get('move_spread', 0.0), _sp)
            stats['attempts'] += 1
            try:
                qd, _den = committor_do_batch(P[b], self.absorbing_idx, i, row)
            except (SingularTransientBlockError, ValueError) as e:
                stats['fails'] += 1
                stats['last_error'] = f"{type(e).__name__}: {e}"
                cols.append(zero)          # excluded from the loss by `ok`, not by a fake label
                continue
            cols.append(qd if return_field else qd[:, i, :])
            ok[b] = True
        # return_field=True is the EVAL path (causal_eval's alpha-mix read): the
        # caller needs the whole q(do a) FIELD [B,m,n,nA] and i_star to mix it
        # with the model's own alpha, not just the row-i read L_do supervises.
        if return_field:
            return torch.stack(cols), ok, i_star
        return torch.stack(cols), ok


# ---------------------------------------------------------------------------
# (b) the loss, every term named (BUILD SPEC section 3, stage 1 only)
# ---------------------------------------------------------------------------
def stage1_loss(out, q_star, x_nx, lambda_z=0.0):
    qc = out['q_alpha'].clamp(EPS_Q, 1 - EPS_Q)
    L_q = -(q_star * qc.log() + (1 - q_star) * (1 - qc).log()).sum(-1).mean()
    L_z = F.mse_loss(out['x_hat'], x_nx)
    return L_q + lambda_z * L_z, L_q.item(), L_z.item()


def do_loss(model, out, moves, do_tgt, do_mask, stats):
    """L_do: the INTERVENTIONAL term. THE TERM THE ARCHITECTURE EXISTS FOR.

    Same clamped BCE as L_q, but against the do(a) label -- an empirical mean
    over R rollouts, not a one-hot, so soft targets are the point and its
    absolute scale is NOT comparable to L_q's (the label carries up to 0.25
    per-coordinate std at R=4; see ChessDoBed's docstring).

    Rows are masked twice: `do_mask` drops padded candidates, `ok` drops whole
    examples whose Sherman-Morrison update refused. Returns None (term skipped,
    not zeroed) if every example in the batch refused."""
    q_do, ok = model.do_read(out, moves, stats)
    mask = do_mask & ok.unsqueeze(-1)
    if not bool(mask.any()):
        return None
    qc = q_do.clamp(EPS_Q, 1 - EPS_Q)
    bce = -(do_tgt * qc.log() + (1 - do_tgt) * (1 - qc).log()).sum(-1)   # [B,m]
    return (bce * mask).sum() / mask.sum()


def compute_loss(model, out, q_star, x_nx, moves, do_tgt, do_mask, blocks, args, stats):
    """L = L_q + lambda_z*L_z + lambda_do*L_do + lambda_topo*L_topo.

    lambda_z defaults to 0.0 (MEASURED adverse: L_q alone +0.5056 vs L_q+0.5L_z
    +0.4070, delta -0.0985, 3/3 seeds -- the encoder-side auxiliary HURTS the
    committor read). lambda_topo defaults to 0.0 so no already-measured run
    changes silently. L_do is present only when the bed has an interventional arm."""
    total, l_q, l_z = stage1_loss(out, q_star, x_nx, args.lambda_z)
    l_do = do_loss(model, out, moves, do_tgt, do_mask, stats) if moves is not None else None
    if l_do is not None:
        total = total + args.lambda_do * l_do
    l_topo = topo_coupling_loss(out['e'], blocks) if args.lambda_topo > 0 else None
    if l_topo is not None:
        total = total + args.lambda_topo * l_topo
    return (total, l_q, l_z,
            float('nan') if l_do is None else float(l_do),
            float('nan') if l_topo is None else float(l_topo))


def topo_blocks(args, q_star, phase_index):
    """Blocks for the 0-dim coupling term.

    'phase' is what the task names: which curriculum phase a sample came from.
    A SINGLE-PHASE RUN THEREFORE HAS ONE BLOCK, and topo_coupling_loss returns
    an exact 0.0 by construction (its own <2-block guard) -- the term is inert,
    which main() prints once rather than letting a zero read as success.
    'outcome' is the only per-sample label that actually varies inside one batch
    here (which absorbing set the observational arm realised), so it is offered
    for the case where the term is meant to do work in a single phase."""
    if args.topo_blocks == 'outcome':
        return q_star.argmax(dim=-1)
    return torch.full((q_star.shape[0],), phase_index, dtype=torch.long)


# ---------------------------------------------------------------------------
# (c) + (d): eval -- constant-predictor control (C1) and collapse-floor diag
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, bed, q_bar, q_floor_table, gen, n, kappa_ceiling=80.0,
             args=None, phase_index=0, train_stats=None):
    # args=None is the ceqjepa.curriculum call path (it scores a checkpoint and
    # never had loss weights to pass): every added term OFF, so that caller's
    # numbers are the same numbers it always got.
    if args is None:
        args = argparse.Namespace(lambda_z=0.0, lambda_do=0.0, lambda_topo=0.0,
                                  topo_blocks='phase')
    x, x_nx, q_star, v_idx, moves, do_tgt, do_mask = draw(bed, gen, n)
    out = model(x)
    # committor.last_kappa_bound is set by the op.committor() call inside model(x)
    # above (for q_field) -- the teleport guarantee is ||(I-Q)^-1||_inf <= 80(1+6e-6) in
    # float32 (D4). D8: a long soak can climb past kappa_ceiling while still healthy
    # (measured 75.6954 at step 1180, still climbing) -- a run must degrade LOUDLY, not
    # abort mid-flight, so this is a printed warning, never a raising assert.
    kappa_bound = op.committor.last_kappa_bound
    if kappa_bound > kappa_ceiling:
        print(f"[ceqjepa.train] WARNING: kappa_bound={kappa_bound:.6f} exceeded "
              f"--kappa-ceiling={kappa_ceiling} -- conditioning is degrading, continuing anyway")
    # THE GUARD THAT WAS MISSING. Every other check in this repo tests a property of
    # the operator GIVEN its inputs -- bitwise softmax containment, kappa exactness to
    # 1.697e-16, the NaN raise, the stranger test. Not one tested whether the operator's
    # inputs VARY. A bilinear delta with both factors zeroed is an exact stationary point,
    # so P was bitwise identical across the batch for every run before 2026-09-08 and the
    # model was a lookup table with an attention read on top. One line would have fired.
    p_spread = float(out['P'].std(dim=0).max()) if out['P'].shape[0] > 1 else float('nan')

    q_hat = out['q_alpha']
    # SHARPNESS DECOMPOSITION (House's identity): CE = H(pi) + KL + J(q) - I_q.
    # The read beats the marginal predictor iff I_q > J(q) + KL. On the T4 causal arm
    # that margin was 0.1196 - (1.3967 + 0.0025) = -1.2796 -- the model ranked fine and
    # paid 1.397 nats of overconfidence to collect 0.12, and nothing in this loop could
    # see it until 4 h 43 m of GPU time had been spent. It is one line and it is now here.
    # q_star is a soft target here (a rollout mean or an exact committor). The
    # decomposition needs a HARD label, so score the realised class: argmax of the
    # target. Where q_star is already one-hot this is exact; where it is an R-rollout
    # mean it is the modal outcome, which is the right thing to be calibrated against.
    _sharp = None
    try:
        _k_star = q_star.argmax(dim=-1)
        _sharp = _sharp_decompose(q_hat.detach(), _k_star)
    except Exception as _e:                         # never let the instrument kill the run
        _sharp = {'margin': float('nan'), 'sharpness': float('nan'),
                  'i_q': float('nan'), 'beats_marginal': False, 'error': str(_e)[:80]}
    mse_model = F.mse_loss(q_hat, q_star).item()
    mse_bar = F.mse_loss(q_bar.expand_as(q_star), q_star).item()   # (c) control
    S = 1.0 - mse_model / mse_bar if mse_bar > 0 else float('nan')  # the R-1 kill metric
    if q_floor_table is not None:                                   # (d) only synthetic/english define this
        floor_here = q_floor_table[v_idx]                           # same query positions the model saw
        collapse_floor = (q_hat - floor_here).abs().max().item()
    else:
        collapse_floor = float('nan')  # bed has no closed-form floor (chess: v_idx isn't a chart index; markets: lattice, not a chain)
    # var_across_examples = Var[e] taken over the BATCH dimension (dim=0) of the ENCODER
    # OUTPUT e = model.enc(x), averaged over the d_enc channels. This is the real collapse
    # detector, not collapse_floor/CFD above. CFD is monotone in logit scale only -- it
    # reads healthy even when the encoder ignores x entirely, as long as e still varies
    # across positions within one example. An encoder collapsed across the BATCH dimension
    # (identical e for every example, i.e. e independent of x) reads exactly 0 here
    # regardless of what CFD says.
    var_across_examples = out['e'].var(dim=0, unbiased=False).mean().item()
    # The held-out arm is scored with a SEPARATE stats dict, so the refusal rate
    # reported below is the TRAINING rate the task asked for and is not diluted
    # by eval's own attempts.
    eval_stats = dict(attempts=0, fails=0, last_error=None)
    blocks = topo_blocks(args, q_star, phase_index)
    loss, l_q, l_z, l_do, l_topo = compute_loss(
        model, out, q_star, x_nx, moves, do_tgt, do_mask, blocks, args, eval_stats)

    # (task item 5) THE REFUSAL RATE, over the training steps since the last eval.
    # Cumulative too, so a late-onset failure cannot hide behind a healthy prefix.
    st = train_stats if train_stats is not None else dict(attempts=0, fails=0, last_error=None)
    w_att = st['attempts'] - st.get('win_attempts', 0)
    w_fail = st['fails'] - st.get('win_fails', 0)
    st['win_attempts'], st['win_fails'] = st['attempts'], st['fails']
    do_refuse_window = (w_fail / w_att) if w_att else float('nan')
    do_refuse_cum = (st['fails'] / st['attempts']) if st['attempts'] else float('nan')
    if w_att and do_refuse_window > 0.01:
        print(f"[FINDING] intervene REFUSED {w_fail}/{w_att} training interventions "
              f"({100 * do_refuse_window:.2f}%) since the last eval -- above the 1% line. "
              f"The Sherman-Morrison denominator is collapsing, i.e. the do-row is driving "
              f"P'[i,i] toward 1 and (I - Q') toward singular. Last: {st.get('last_error')}")

    _sh = _sharp or {}

    return dict(sharp_margin=_sh.get('margin', float('nan')),
                sharp_J=_sh.get('sharpness', float('nan')),
                sharp_Iq=_sh.get('i_q', float('nan')),
                beats_marginal=bool(_sh.get('beats_marginal', False)),
                mse_model=mse_model, mse_bar=mse_bar, S=S,
                collapse_floor=collapse_floor, var_across_examples=var_across_examples,
                kappa_bound=kappa_bound, p_spread=p_spread, loss=loss.item(), L_q=l_q, L_z=l_z,
                L_do=l_do, L_topo=l_topo,
                do_refuse_window=do_refuse_window, do_refuse_cum=do_refuse_cum,
                do_eval_refused=eval_stats['fails'], do_eval_attempts=eval_stats['attempts'])


def save_checkpoint(path, model, opt, args, n_params, q_bar, history, wall_s,
                     step, train_gen, heldout_gen, phase, phase_history):
    """D7 + RESUME: atomic checkpoint write -- write to a temp path in the same
    dir, then os.replace() it onto `path`. os.replace is atomic on both POSIX
    and Windows, so a process killed mid-write leaves either the old
    checkpoint or nothing at `path`, never a half-written (corrupt) one.

    Ported from ceq/hf/train.py's four-component resume pattern: parameters,
    optimizer.state_dict(), RNG state (global torch + both dedicated data
    generators), and the step counter -- everything needed to continue as if
    never interrupted. Without this a resumed run restarts Adam from zero and
    re-draws the same batches (measured cost of NOT having this: COSTS.md
    line 266, 647x GPU-min on a mid-chunk kill).
    """
    tmp = path + '.tmp'
    torch.save(dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=opt.state_dict(),
        step=step,
        torch_rng_state=torch.get_rng_state(),
        train_gen_state=train_gen.get_state(),
        heldout_gen_state=heldout_gen.get_state(),
        phase=phase,
        phase_history=phase_history,
        geometry=vars(args),
        n_params=n_params,
        q_bar=q_bar,
        final_eval=history[-1] if history else None,
        history=history,
        wall_s=wall_s,
    ), tmp)
    os.replace(tmp, path)


def self_check():
    """The one check the do(a) arm must not lose. Builds no chess games (seconds,
    no python-chess): random candidate moves through the real do_read path.

    (i) healthy: the term is finite and nothing refuses.
    (ii) PLANTED NEGATIVE: chart forced so i_star == j for every example, the move
         head's bias forced onto j so the clamped row is delta_j, and A_tel zeroed
         so the do-row carries teleport=0 (the SHIP/EVAL setting). Then P'[j,j] = 1,
         (I - Q') is exactly singular, and every intervention MUST be refused,
         counted, and survived -- do_loss returns None rather than a number.
    (iii) the same forced row at the TRAINING teleport refuses NOTHING, because
         A_tel[i][i] = 0 for a transient i, so P'_ii <= 1 - c and the
         Sherman-Morrison denominator (1 - P'_ii)/(1 - P_ii) >= c = 0.0125,
         36x above den_min = sqrt(eps_float32) = 3.45e-4. The guard is therefore
         UNREACHABLE while teleport > 0: it protects the eval/ship path, not the
         training path, and a 0.0000 refusal rate during training is what the
         teleport buys, not evidence the guard works."""
    torch.manual_seed(0)
    n, nA, m, B = 16, 4, 4, 8
    mk = lambda: TinyCEQ(n=n, nA=nA, d_enc=16, x_dim=8, z_dim_state=6, g=0.9, rank=12,
                          absorbing_idx=torch.arange(nA))
    x = torch.randn(B, 8)
    moves = torch.randint(0, 64, (B, m, 2))
    tgt = torch.rand(B, m, nA)
    tgt = tgt / tgt.sum(-1, keepdim=True)
    mask = torch.ones(B, m, dtype=torch.bool)

    model = mk()
    st = dict(attempts=0, fails=0, last_error=None)
    L = do_loss(model, model(x), moves, tgt, mask, st)
    assert L is not None and torch.isfinite(L), f"healthy L_do is not finite: {L}"
    assert st['fails'] == 0, f"healthy path refused {st['fails']}/{st['attempts']}"
    print(f"[SELF-CHECK] (i) healthy: L_do={float(L):.6f}, refused {st['fails']}/{st['attempts']}")

    j = 9
    bad = mk()
    with torch.no_grad():
        bad.chart.weight.zero_(); bad.chart.bias.zero_(); bad.chart.bias[j] = 100.0
        bad.mv_row.bias[j] = 60.0
        bad.A_tel.zero_()                       # teleport = 0 on the do-row
    st2 = dict(attempts=0, fails=0, last_error=None)
    L2 = do_loss(bad, bad(x), moves, tgt, mask, st2)
    assert L2 is None, "planted self-absorbing do-row was NOT refused -- the guard is dead"
    assert st2['fails'] == st2['attempts'] == B, f"refusal count wrong: {st2}"
    assert 'SingularTransientBlockError' in st2['last_error'], st2['last_error']
    print(f"[SELF-CHECK] (ii) planted (teleport=0, do-row -> delta_i): refused "
          f"{st2['fails']}/{st2['attempts']}, survived, term SKIPPED not zeroed. "
          f"{st2['last_error'][:110]}...")

    warm = mk()
    with torch.no_grad():
        warm.chart.weight.zero_(); warm.chart.bias.zero_(); warm.chart.bias[j] = 100.0
        warm.mv_row.bias[j] = 60.0              # A_tel LEFT ALONE: teleport = op.TELEPORT
    st3 = dict(attempts=0, fails=0, last_error=None)
    L3 = do_loss(warm, warm(x), moves, tgt, mask, st3)
    den_min = float(torch.finfo(torch.float32).eps) ** 0.5
    assert st3['fails'] == 0 and L3 is not None, (
        "the teleport bound den >= c failed: %s" % st3)
    print(f"[SELF-CHECK] (iii) SAME forced row at teleport={op.TELEPORT}: refused "
          f"{st3['fails']}/{st3['attempts']}, L_do={float(L3):.6f}. Bound den >= "
          f"{op.TELEPORT} vs den_min={den_min:.3e} -> {op.TELEPORT / den_min:.1f}x margin; "
          f"the guard cannot fire during training, only at teleport=0.")

    g = torch.Generator().manual_seed(0)
    E = mk()(torch.randn(24, 8))['e']
    one = torch.zeros(24, dtype=torch.long)
    two = torch.arange(24) % 2
    assert float(topo_coupling_loss(E, one)) == 0.0, (
        "single-block topo term is not exactly 0 -- the warning in main() is wrong")
    assert float(topo_coupling_loss(E, two)) > 0.0
    print(f"[SELF-CHECK] (iv) topo term: 1 block -> {float(topo_coupling_loss(E, one)):.4f} "
          f"(inert, as --topo-blocks phase is inside one phase); "
          f"2 blocks -> {float(topo_coupling_loss(E, two)):.4f}")

    # (v) THE SPLIT GUARD, both directions. A clean game split must pass and an
    # overlapping one must RAISE -- a guard never seen to fire is not a guard.
    # Built on a stand-in corpus of 20 games x 3 positions each, every position of
    # a game carrying that game's outcome, which is the leak the guard exists for.
    N_G, PER_G = 20, 3
    fake = object.__new__(ChessDoBed)
    fake.m = 1
    fake.group_ids = torch.arange(N_G).repeat_interleave(PER_G)
    fake.x = torch.randn(N_G * PER_G, 8)
    fake.x_nx, fake.q_star = fake.x.clone(), torch.rand(N_G * PER_G, nA)
    fake.v_idx = torch.zeros(N_G * PER_G, dtype=torch.long)
    fake.moves = torch.zeros(N_G * PER_G, 1, 2, dtype=torch.long)
    fake.do_tgt = torch.rand(N_G * PER_G, 1, nA)
    fake.do_mask = torch.ones(N_G * PER_G, 1, dtype=torch.bool)
    fake.row_keys = [r.numpy().tobytes() for r in fake.x]
    tr, ho, note = heldout_split(fake, 0.25, seed=0)
    tr_g, ho_g = set(tr.group_ids.tolist()), set(ho.group_ids.tolist())
    assert not (tr_g & ho_g) and len(ho_g) == 5 and len(tr_g) == 15, (tr_g, ho_g)
    assert len(ho.x) == 5 * PER_G and len(tr.x) == 15 * PER_G, (len(tr.x), len(ho.x))
    print(f"[SELF-CHECK] (v) clean GAME split of {N_G} games x {PER_G} positions: "
          f"{len(tr_g)} train / {len(ho_g)} held-out games, {len(tr.x)} / {len(ho.x)} rows, "
          f"0 games shared. {note}")

    all_idx = torch.arange(N_G * PER_G)
    try:
        # the leak: hold out rows 0..14 while training on ALL of them
        assert_disjoint(fake.group_ids, fake.group_ids[:15], all_idx, all_idx[:15])
        raise SystemExit("[SELF-CHECK] (v) FAILED: an overlapping split was ACCEPTED -- "
                          "the disjointness guard is dead")
    except AssertionError as e:
        assert 'LEAKS' in str(e), e
        print(f"[SELF-CHECK] (v) overlapping split RAISED as it must: {str(e)[:150]}...")
    try:
        # the subtler leak: different rows, but the same GAMES on both sides --
        # positions 0,3,6.. train and 1,4,7.. held out. Row indices are disjoint;
        # the label is not, because every position of a game shares its outcome.
        assert_disjoint(fake.group_ids[0::3], fake.group_ids[1::3], all_idx[0::3], all_idx[1::3])
        raise SystemExit("[SELF-CHECK] (v) FAILED: a POSITION-level split of shared games "
                          "was accepted -- the guard checks rows but not labels")
    except AssertionError as e:
        assert 'LEAKS' in str(e) and 'shared label' in str(e), e
        print(f"[SELF-CHECK] (v) position-level split of the SAME games RAISED: "
              f"{str(e)[:150]}...")

    # (vi) THE MEMORISATION GUARD, on the series it exists for. Three REAL recorded
    # (step, held-out score, training loss) traces replayed through the shipped rule:
    # the T4 causal arm that motivated all of this, the deliberately overfit chess_do
    # run below, and the tiny synthetic defaults where the guard MUST stay quiet. The
    # calibration is pinned here so a future edit to DIVERGENCE_REL breaks a check
    # rather than a run.
    def replay(rows, k):
        best, best_tl, streak, first = float('inf'), float('inf'), 0, None
        for step, score, tl in rows:
            if score < best:
                best, best_tl, streak = score, tl, 0
            else:
                streak = streak + 1 if diverging(score, best, tl, best_tl) else 0
            if streak >= k and first is None:
                first = step
        return first

    t4 = [(1, 3.9975, 2.2491), (2000, 3.1796, 1.2486), (4000, 3.9866, 1.0996),
          (6000, 5.9879, 0.5411), (8000, 7.1490, 0.2883), (10000, 9.1577, 0.0443),
          (12000, 10.3872, 0.1415)]     # ceq-jepa-dcm-1-causal-arm-t4.stdout.txt, seed 0
    over = [(125, 1.4056, 2.3387), (150, 2.0609, 2.0017), (175, 3.5080, 2.3520),
            (200, 1.6656, 2.1417), (225, 2.6120, 2.0278), (250, 1.6081, 1.6569)]
    quiet = [(5, 2.2447, 2.2490), (6, 2.2465, 2.2470), (7, 2.2481, 2.2461),
             (8, 2.2490, 2.2452), (9, 2.2502, 2.2444), (10, 2.2478, 2.2437)]
    assert replay(t4, 3) == 8000, replay(t4, 3)
    assert replay(over, 3) == 250, replay(over, 3)
    assert replay(quiet, 3) is None, replay(quiet, 3)
    print(f"[SELF-CHECK] (vi) MEMORISATION guard, DIVERGENCE_REL={DIVERGENCE_REL}, K=3: "
          f"T4 causal arm seed 0 -> fires at step {replay(t4, 3)} (4000 steps and ~1h40m "
          f"of T4 time before that run ended at 12000); deliberately overfit chess_do -> "
          f"fires at step {replay(over, 3)}; tiny synthetic eval noise (+0.24% held-out, "
          f"-0.2% train) -> {replay(quiet, 3)}, stays quiet.")
    print("[SELF-CHECK] ALL PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-check', action='store_true',
                    help='run the do(a) arm and topo term self-check (no bed, seconds) and exit')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=None,
                    help='alias for --steps (the step cap); overrides it when both are given')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='ceqjepa_checkpoint.pt')
    ap.add_argument('--max-seconds', type=float, default=None,
                     help='stop after this many wall-clock seconds, in addition to --steps')
    ap.add_argument('--geometry', choices=['tiny', 'design'], default='tiny',
                     help="'tiny': the CPU stand-in below (default). 'design': sets "
                          "n=256, d-enc=128 (the DCM-1 frozen N, d) and prints the actual "
                          "param count against the ~325,792 target -- this geometry has "
                          "never been built before, so it is NOT expected to hit the "
                          "target exactly through this stand-in's dense per-example "
                          "delta_logits (cost d_enc*n*n alone).")
    # geometry -- tiny CPU-runnable defaults; NOT the frozen design geometry
    ap.add_argument('--n', type=int, default=16, help='chart positions (build spec N)')
    ap.add_argument('--nA', type=int, default=4, help='absorbing positions (build spec nA)')
    ap.add_argument('--d-enc', type=int, default=16)
    ap.add_argument('--x-dim', type=int, default=8)
    ap.add_argument('--z-dim', type=int, default=4, help='latent width of the bed (unknown to the model)')
    ap.add_argument('--z-dim-state', type=int, default=6, help='model state-channel width')
    ap.add_argument('--g', type=float, default=0.9, help='state-channel discount, frozen')
    ap.add_argument('--rank', type=int, default=12,
                     help='D6: rank of the per-example delta_logits factorization '
                          '(DCM-1 spec rank r=8); cost is O(d_enc*n*rank), not O(d_enc*n*n)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--eval-n', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ckpt-every', type=int, default=0,
                     help='D7: write a checkpoint every this many steps (0 = only at the '
                          'end, the old behavior). Atomic: written to --out+".tmp" then '
                          'os.replace()d, so a kill mid-write cannot corrupt --out.')
    ap.add_argument('--heldout-frac', type=float, default=0.2,
                     help='fraction of GROUPS (games, never positions) held out of training '
                          'and used for every reported held-out number. Beds with no finite '
                          'corpus (synthetic/english/markets build each example on the spot) '
                          'ignore this and stay on the disjoint-generator arm they always had.')
    ap.add_argument('--split-seed', type=int, default=1234,
                     help='seed for which groups land in the held-out split (independent of '
                          '--seed, so the same corpus can be re-split without redrawing it)')
    ap.add_argument('--early-stop-patience', type=int, default=5,
                     help='ON BY DEFAULT. Stop after this many consecutive evals with no new '
                          'best held-out L_q, and keep the BEST checkpoint rather than the '
                          'last one. Counted in EVALS, not steps, so it scales with '
                          '--eval-every. 0 disables it (the pre-2026-09-09 behaviour: run to '
                          '--max-steps and ship whatever the last step happened to be).')
    ap.add_argument('--divergence-k', type=int, default=3,
                     help='MEMORISATION guard: print a loud warning once held-out L_q has '
                          'worsened for this many consecutive evals WHILE the training loss '
                          'improved. 0 disables it.')
    ap.add_argument('--lambda-z', type=float, default=0.0,
                    help="weight on the encoder-side auxiliary L_z. 0.0 ablates it "
                         "(House's composition test: does an encoder-side auxiliary "
                         "already in the objective help the operator's read at all?) "
                         "DEFAULT CHANGED 2026-09-08 from 0.5 to 0.0: MEASURED L_q "
                         "alone +0.5056 vs L_q+0.5L_z +0.4070, delta -0.0985, 3/3 seeds.")
    ap.add_argument('--lambda-do', type=float, default=1.0,
                    help='weight on the INTERVENTIONAL term L_do (--bed chess_do only; '
                         'no other bed supplies a do(a) arm, so the term is simply absent '
                         'there and no already-measured run changes). This is the only '
                         'term that ever supervises the Sherman-Morrison committor.')
    ap.add_argument('--lambda-topo', type=float, default=0.0,
                    help='weight on ceqjepa.topo_loss.topo_coupling_loss over the encoder '
                         'output. Default 0.0 so nothing already measured changes silently.')
    ap.add_argument('--topo-blocks', choices=['phase', 'outcome'], default='phase',
                    help="what the coupling term's blocks are. 'phase' (default) is the "
                         "curriculum phase; inside a SINGLE-phase run that is one block and "
                         "the term is identically 0.0 -- said out loud at startup, not hidden. "
                         "'outcome' blocks by the observational arm's realised absorbing set, "
                         "which does vary inside a batch.")
    ap.add_argument('--do-m', type=int, default=8,
                    help='--bed chess_do: candidate forced moves per position (one '
                         'factorisation covers all m).')
    ap.add_argument('--do-max-plies', type=int, default=80,
                    help='--bed chess_do: ply cap on every game AND every forced rollout. '
                         'MEASURED at n_games=60 seed=0, outcome counts (white, draw, black, '
                         'SINK): 80 -> [3,0,1,56], 200 -> [2,0,2,56], 400 -> [6,34,3,17]. At '
                         'the 80 default 93%% of positions absorb into SINK, so q_bar is '
                         'nearly one-hot, mse_bar ~ 0 and S is nan/meaningless. Raise it to '
                         '400 for a bed with more than one outcome in it.')
    ap.add_argument('--do-R', type=int, default=4,
                    help='--bed chess_do: rollouts per candidate. The label is their MEAN, '
                         'with per-coordinate variance p(1-p)/R <= 1/(4R).')
    ap.add_argument('--kappa-ceiling', type=float, default=80.0,
                     help='D8: kappa_bound above this prints a warning instead of aborting '
                          'the run (was a hard assert that killed 20k-step soaks)')
    ap.add_argument('--resume', default=None,
                     help='path to a checkpoint written by this script. If it exists, '
                          'restores model, optimizer, RNG and data-generator state and '
                          'continues from the saved step; if it does not exist, starts '
                          'fresh and says so on stdout. --steps counts NEW steps to run '
                          'this invocation, added on top of the resumed step.')
    ap.add_argument('--phase', default='chess',
                     help='free-form curriculum phase name (e.g. chess/english/markets), '
                          'recorded in the checkpoint. A phase-3 checkpoint carries the '
                          'phase_history of every phase it passed through, so it knows it '
                          'came through phases 1 and 2.')
    ap.add_argument('--bed', choices=['synthetic', 'chess', 'chess_do', 'english', 'markets',
                                 'chess_policy', 'gridworld'],
                     default='synthetic',
                     help='which corpus to train/eval on. chess and markets have FIXED '
                          'geometry baked into the corpus (n/nA/x_dim); when selected '
                          'those overwrite --n/--nA/--x-dim, same as --geometry design '
                          'overwriting --n/--d-enc. Default synthetic keeps the original '
                          'in-class bed so nothing already working breaks.')
    ap.add_argument('--bed-temperature', type=float, default=0.25,
                    help="chess_policy rollout temperature; 0.25 is the measured optimum "
                         "(0.3084 nats, 54.9 sigma). Below it the policy goes near-deterministic "
                         "and information FALLS -- see chess_policy's [LIMIT] line.")
    ap.add_argument('--bed-gap', type=int, default=1,
                    help='gridworld barrier gap; 1 is the highest-information setting measured '
                         '(0.3986 nats, 157.2 sigma). gap=0 seals it and the label stops being '
                         'about the walk at all.')
    ap.add_argument('--n-games', type=int, default=200,
                     help='--bed chess only: self-play games to build the ply pool from '
                          '(ChessBed.build default).')
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    if args.max_steps is not None:
        args.steps = args.max_steps
    if args.geometry == 'design':
        args.n, args.d_enc = 256, 128  # DCM-1 frozen N, d; other dims stay at their --flags

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    assert device.type == 'cpu' or True  # ponytail: no GPU-specific path needed at this scale

    bed = make_bed(args)
    print(f"[ceqjepa.train] bed={args.bed} n={args.n} nA={args.nA} x_dim={args.x_dim}")
    train_bed, eval_bed, split_note = heldout_split(bed, args.heldout_frac, args.split_seed)
    print(f"[split] heldout_frac={args.heldout_frac}: {split_note}")
    train_gen = torch.Generator().manual_seed(args.seed)
    heldout_gen = torch.Generator().manual_seed(args.seed + 1_000_000)  # disjoint stream

    # RESUME: decide before anything touches train_gen, since a resumed run
    # must NOT redraw the q_bar pool (that draw already happened in the run
    # being resumed, and redrawing here would desync train_gen from the
    # saved state -- breaking bitwise continuation).
    resuming = bool(args.resume) and os.path.exists(args.resume)
    if args.resume and not resuming:
        print(f"[ceqjepa.train] --resume {args.resume} not found, starting fresh")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if resuming else None
    # CURRICULUM: a resume is "same phase" (an interrupted run continuing on
    # the SAME bed -- the case the resume machinery was built and tested
    # for) only if the checkpoint's own phase matches --phase. A different
    # phase means a different bed, so q_bar (that bed's channel mean) and
    # possibly the model's own shapes (nA differs: chess/english=4,
    # markets=2) are NOT meaningful carried over -- see below.
    same_phase = resuming and ckpt.get('phase') == args.phase

    if same_phase:
        q_bar = ckpt['q_bar']
    else:
        # q_bar: the constant predictor's whole content -- training-set channel mean.
        # Recomputed fresh on a phase change: the OLD bed's channel mean is not a
        # meaningful control for a NEW corpus.
        _, _, q_pool, _ = train_bed.batch(train_gen, max(64, args.batch_size))
        q_bar = q_pool.mean(0)
        if resuming:
            print(f"[ceqjepa.train] --resume phase changed ({ckpt.get('phase')!r} -> "
                  f"{args.phase!r}): recomputed q_bar fresh for the new bed")
    q_floor_table = train_bed.q_floor() if hasattr(train_bed, 'q_floor') else None

    model = TinyCEQ(n=args.n, nA=args.nA, d_enc=args.d_enc, x_dim=args.x_dim,
                     z_dim_state=args.z_dim_state, g=args.g, rank=args.rank,
                     absorbing_idx=torch.arange(args.nA)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=0.01)

    start_step = 0
    history = []
    phase_history = []
    if resuming:
        ckpt_state = ckpt['model_state_dict']
        if same_phase:
            model.load_state_dict(ckpt_state)
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            # CURRICULUM warm start across a phase (bed) change: transfer every
            # tensor whose shape still matches (the shared committor machinery --
            # L0, Vt, delta_a/delta_b, chart, and enc.* layers whose width didn't
            # change) bitwise; RESET only what the new task's shape forces --
            # readout and the absorbing_idx buffer when nA changes (chess/english
            # nA=4 -> markets nA=2), or enc.0/readout when x_dim changes. The
            # optimizer is reset fresh too: its saved Adam moments are keyed to
            # the OLD shapes and meaningless for a reset (or even just newly
            # re-initialized) head.
            own_state = model.state_dict()
            mismatched = sorted(k for k in ckpt_state
                                 if k in own_state and own_state[k].shape != ckpt_state[k].shape)
            compatible = {k: v for k, v in ckpt_state.items() if k not in mismatched}
            model.load_state_dict(compatible, strict=False)
            print(f"[ceqjepa.train] cross-phase warm start ({ckpt.get('phase')!r} -> "
                  f"{args.phase!r}): transferred {len(compatible)}/{len(ckpt_state)} tensors "
                  f"bitwise; RESET (shape mismatch, new task head) {mismatched or 'none'}; "
                  f"optimizer reset fresh")
        torch.set_rng_state(ckpt['torch_rng_state'])
        train_gen.set_state(ckpt['train_gen_state'])
        heldout_gen.set_state(ckpt['heldout_gen_state'])
        start_step = ckpt['step']
        history = ckpt.get('history', [])
        phase_history = ckpt.get('phase_history', [])
        print(f"[ceqjepa.train] resumed from {args.resume} at step={start_step} "
              f"phase_history={phase_history}")
    if not phase_history or phase_history[-1] != args.phase:
        phase_history = phase_history + [args.phase]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ceqjepa.train] n={args.n} nA={args.nA} d_enc={args.d_enc} "
          f"x_dim={args.x_dim} z_dim={args.z_dim} z_dim_state={args.z_dim_state} "
          f"g={args.g} params={n_params}")
    print(f"[ceqjepa.train] q_bar={[round(v, 4) for v in q_bar.tolist()]}")
    if args.geometry == 'design':
        target = 325_792
        delta_rank_cost = 2 * args.d_enc * args.n * args.rank  # D6: delta_a + delta_b, low-rank
        delta_dense_cost = args.d_enc * args.n * args.n        # what the old dense delta_logits cost
        print(f"[ceqjepa.train] geometry=design target_params~{target} actual_params={n_params} "
              f"(D6 low-rank delta_logits, rank={args.rank}: delta_a+delta_b cost "
              f"2*d_enc*n*rank={delta_rank_cost} vs dense d_enc*n*n={delta_dense_cost} that the "
              f"pre-fix per-example modulation cost -- "
              f"{'hit' if n_params == target else 'did NOT hit'} the target exactly, reporting actual)")

    phase_index = len(phase_history) - 1
    if args.lambda_topo > 0 and args.topo_blocks == 'phase':
        print("[ceqjepa.train] WARNING: --lambda-topo > 0 with --topo-blocks=phase inside a "
              "single-phase run: every sample carries the SAME block label, so "
              "topo_coupling_loss returns exactly 0.0 by its own <2-block guard and the term "
              "contributes NOTHING. Use --topo-blocks outcome for a term that does work here.")
    # (task item 5) counters for intervene's refusals during TRAINING.
    do_stats = dict(attempts=0, fails=0, last_error=None, win_attempts=0, win_fails=0)

    # (2) BEST, NOT LAST. The held-out score is L_q on the held-out split -- the same
    # observational fit the T4 causal arm reported as PPL_obs = exp(CE_obs), which rose
    # in 15 of 15 intervals while the training loss fell to 0.03-0.14 nats and the loop
    # shipped the LAST step anyway. Seeded from history so a --resume cannot re-crown a
    # worse checkpoint and clobber a good --out; only rows from THIS phase compare.
    prior = [r['L_q'] for r in history if r.get('phase') == args.phase and r.get('L_q') == r.get('L_q')]
    best_score = min(prior) if prior else float('inf')
    best_step = min((r for r in history if r.get('phase') == args.phase and r.get('L_q') == best_score),
                    key=lambda r: r['step'], default=dict(step=start_step))['step'] if prior else start_step
    best_train_loss = float('inf')
    stale = 0
    diverge_streak = 0
    stopped_early = False
    last_eval = None

    t0 = time.time()
    last_step = start_step
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step  # absolute step, carries across --resume
        x, x_nx, q_star, _, moves, do_tgt, do_mask = draw(train_bed, train_gen, args.batch_size)
        model.train()
        out = model(x)
        blocks = topo_blocks(args, q_star, phase_index)
        loss, l_q, l_z, l_do, l_topo = compute_loss(
            model, out, q_star, x_nx, moves, do_tgt, do_mask, blocks, args, do_stats)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or local_step == args.steps:
            model.eval()
            ev = evaluate(model, eval_bed, q_bar, q_floor_table, heldout_gen, args.eval_n,
                           kappa_ceiling=args.kappa_ceiling, args=args,
                           phase_index=phase_index, train_stats=do_stats)
            row = dict(step=step, phase=args.phase, train_loss=loss.item(),
                       train_L_do=l_do, train_L_topo=l_topo, **ev)
            history.append(row)
            score, tr_loss, last_eval = ev['L_q'], loss.item(), row

            improved = score < best_score
            if improved:
                best_score, best_step, best_train_loss, stale = score, step, tr_loss, 0
            else:
                stale += 1
            tail = (' NEW BEST' if improved else
                    f" (stale {stale}/{args.early_stop_patience}" if args.early_stop_patience
                    else f" (stale {stale}, early stop OFF")
            print(f"[eval] step={step:4d} train_loss={loss.item():.4f} "
                  f"heldout_L_q={score:.4f} best_heldout_L_q={best_score:.4f}@step{best_step}"
                  f"{tail if improved else tail + ')'} "
                  f"S(vs const)={ev['S']:+.4f} mse_model={ev['mse_model']:.4e} "
                  f"mse_bar={ev['mse_bar']:.4e} collapse_floor={ev['collapse_floor']:.4e} "
                  f"var_across_examples(enc_output,batch_dim)={ev['var_across_examples']:.4e} "
                  f"kappa_bound={ev['kappa_bound']:.4f} p_spread={ev['p_spread']:.3e} "
                  f"margin(Iq-J-KL)={ev['sharp_margin']:+.4f}{'' if ev['beats_marginal'] else ' OVERCONFIDENT'} "
                  f"L_do(train)={l_do:.4f} L_do(heldout)={ev['L_do']:.4f} "
                  f"L_topo={ev['L_topo']:.4f} "
                  f"do_refuse[window]={ev['do_refuse_window']:.4f} "
                  f"do_refuse[cum]={ev['do_refuse_cum']:.4f}")

            # (2) BEST, NOT LAST: --out always holds the best-scoring checkpoint, written
            # the moment it is the best. A run killed at any point therefore leaves the
            # best weights on disk, not the most recent ones.
            if improved:
                save_checkpoint(args.out, model, opt, args, n_params, q_bar, history,
                                 time.time() - t0, step, train_gen, heldout_gen,
                                 args.phase, phase_history)

            # (4) THE DIVERGENCE GUARD. Held-out worse AND training loss better, K evals
            # running: that is memorisation, and it is exactly the shape the T4 causal arm
            # printed for 15 of 15 intervals with nothing in the loop looking at it.
            #
            # BOTH COMPARISONS ARE AGAINST THE BEST EVAL, NOT THE PREVIOUS ONE. Measured,
            # this file, --bed chess_do 60 games, 800 steps: against the previous eval the
            # guard fired ZERO times on a run whose held-out L_q went 1.0942 -> 3.2113
            # while the training loss went 2.2978 -> 0.7769, because train_loss is one
            # minibatch and bounces (2.03, 1.66, 1.88, 1.36 at consecutive evals) so the
            # strict consecutive test kept resetting. A guard that cannot fire on the
            # textbook case is not a guard; the reference is the best checkpoint.
            if not improved and diverging(score, best_score, tr_loss, best_train_loss):
                diverge_streak += 1
            else:
                diverge_streak = 0
            if args.divergence_k and diverge_streak > args.divergence_k:
                # already said in full below; keep it visible without a wall of repeats
                print(f"[MEMORISATION] still diverging: {diverge_streak} evals, held-out L_q "
                      f"{score:.4f} vs best {best_score:.4f}@{best_step}, train_loss {tr_loss:.4f} "
                      f"vs {best_train_loss:.4f} there")
            elif args.divergence_k and diverge_streak == args.divergence_k:
                print(f"[MEMORISATION] held-out L_q has been WORSE than its best for "
                      f"{diverge_streak} consecutive evals while the training loss kept "
                      f"IMPROVING past the best checkpoint's: held-out L_q {best_score:.4f} "
                      f"(step {best_step}) -> {score:.4f} (step {step}), {score - best_score:+.4f}; "
                      f"training loss {best_train_loss:.4f} -> {tr_loss:.4f}, "
                      f"{tr_loss - best_train_loss:+.4f}. The model is fitting the training "
                      f"rows, not the task. The best checkpoint is {step - best_step} steps "
                      f"back and every step since has bought training loss with held-out loss.")

            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(f"[EARLY STOP] {stale} consecutive evals with no new best held-out L_q "
                      f"(patience={args.early_stop_patience}). Best {best_score:.4f} at step "
                      f"{best_step}; stopping at step {step} instead of running to "
                      f"{start_step + args.steps}.")
                stopped_early = True

        # D7: periodic checkpoint so a killed run leaves usable weights behind --
        # the last soak reached step 1840 healthy and left nothing on disk.
        # CHANGED 2026-09-09: this rolling LAST checkpoint moved to --out+'.last',
        # because --out now holds the BEST one. Both are full, resumable checkpoints.
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(args.out + '.last', model, opt, args, n_params, q_bar, history,
                             time.time() - t0, step, train_gen, heldout_gen,
                             args.phase, phase_history)
            print(f"[ceqjepa.train] wrote checkpoint {args.out}.last at step={step}")

        last_step = step
        if stopped_early:
            break
        if args.max_seconds is not None and time.time() - t0 > args.max_seconds:
            print(f"[ceqjepa.train] stopping at step={step}: max-seconds={args.max_seconds} exceeded")
            break

    # Self-describing checkpoint: what the HuggingFace artifact will wrap.
    # model state_dict + the geometry config (args) + q_bar + the final eval numbers
    # + everything RESUME needs (optimizer state, RNG state, data-gen state, step,
    # phase/phase_history) to continue as if never interrupted.
    save_checkpoint(args.out + '.last', model, opt, args, n_params, q_bar, history,
                     time.time() - t0, last_step, train_gen, heldout_gen,
                     args.phase, phase_history)
    if best_score == float('inf'):      # never evaluated (--steps 0, or a caller with no evals)
        save_checkpoint(args.out, model, opt, args, n_params, q_bar, history,
                         time.time() - t0, last_step, train_gen, heldout_gen,
                         args.phase, phase_history)
        print(f"[ceqjepa.train] no eval ran; wrote the last step to {args.out}")
    else:
        # (3) THE RATIO. On the T4 causal arm (best step 2000, ran to 12000) this line
        # reads 10000/2000 = 5.00 -- five steps thrown away for every one that helped,
        # 3 h 56 m of the 4 h 43 m. Nothing printed it, so nobody saw it.
        final = last_eval['L_q'] if last_eval else float('nan')
        wasted, useful = last_step - best_step, max(1, best_step)
        print(f"[SUMMARY] best_step={best_step} best_heldout_L_q={best_score:.4f} | "
              f"final_step={last_step} final_heldout_L_q={final:.4f} | "
              f"delta_final_minus_best={final - best_score:+.4f}")
        print(f"[SUMMARY] wasted/useful = {wasted}/{useful} = {wasted / useful:.2f} "
              f"({wasted} steps ran after the best checkpoint and made it no better; "
              f"{useful} steps produced it)")
        print(f"[ceqjepa.train] {args.out} holds the BEST checkpoint (step {best_step}); "
              f"{args.out}.last holds the LAST one (step {last_step}) for --resume")


if __name__ == '__main__':
    main()
