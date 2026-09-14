#!/usr/bin/env python
"""The resolvent family against the softmax head it contains, both trained.

WHAT THIS SETTLES AND WHAT IT CANNOT
------------------------------------
`lean/CEQ/V16Domain.lean:433 three_corners_containment` proves the family CAN be
softmax attention (`beta = 1`) and CAN be linear attention (`beta = 0`).
Containment is a statement about EXPRESSIVITY. It is silent on whether the
family, TRAINED, reaches what softmax reaches. This file trains both and prints
the parameter count of each arm beside its score.

THE BED
-------
Next-move prediction on real Lichess games. A token is the ply actually played,
`(from square, to square)`; the label at context position `i` is the human's
move at ply `i + 1`. Nothing synthetic, and the label is a decision a person
made rather than a target the encoder also produces -- which is the defect that
made the pi-JEPA bed unable to settle a training question at all.

The input carries NO BOARD. The model sees only the move sequence, so the
position must be reconstructed by mixing across the prefix: the mixing operator
is what is being scored, not an encoder in front of it. `no-mix` is the arm that
holds that claim up -- it deletes the mixing and keeps everything else, and if it
ties the mixing arms the bed has no attention headroom and every other row here
is void.

NO NORMALISER SITS AFTER THE READ, DELIBERATELY
-----------------------------------------------
With the gate at its default the beta switch is exactly a per-row gain
`Z_i^(1 - beta)` on the softmax row (`tests/curvature/test_beta_axis_is_a_row_gain.py`),
and an RMS or Layer norm after the read removes a per-row gain exactly. A bed
with one would score every beta identically by construction. This bed has none,
so the beta axis is given the most favourable arithmetic it can have.

WHAT EACH ARM VARIES AND WHAT IT PINS -- L-NULL, in the code
------------------------------------------------------------
Every arm shares, bitwise: the corpus, the by-game split, the parameter
initialisation (same seed, same module order), the batch sequence, the
optimiser and its constants, the step count, the evaluation protocol, the dtype
(float64 throughout, so the shipped operator's complex128 path is not compared
against a float32 softmax). `ARMS` below names, per arm, the single thing that
moves.

DTYPE, AND WHY THE SOFTMAX ARM IS ALSO IN float64
--------------------------------------------------
`ceq/arm_smprime.py` is a float64/complex128 module: its hop is a cumulative
product, and `path_product` is the repair that a prefix scan cannot make. Racing
it against a float32 softmax would price the dtype, not the operator. Both arms
are float64. The wall clock reported for the resolvent arms therefore includes
the complex128 contraction the family pays for carrying a phase -- that is a
real cost of the family and it is reported, not netted out.
"""

import argparse
import json
import math
import pathlib
import platform
import subprocess
import sys
import time

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ceq.arm_smprime as arm                                    # noqa: E402

S_LEN = 32                 #: context plies; the label sits at ply i + 1
D_MODEL = 64
LR = 3e-3
BATCH = 256
STEPS = 1500
N_SQ = 64
#: OFF by default, so every run above is unaffected. The head is otherwise AFFINE
#: from the mix to the output -- embed, mix, residual, linear, and no nonlinearity
#: anywhere -- which is enough for the move bed's largely positional statistic and
#: is NOT enough to represent a committor. `CEQ_H2H_MLP=1` inserts the one
#: transformer-block piece that was missing, identically in every arm.
USE_MLP = os.environ.get("CEQ_H2H_MLP") == "1"


# ---------------------------------------------------------------- the corpus

def load_corpus(path):
    z = np.load(path)
    moves, split = z["moves"], z["split"]
    tr = torch.from_numpy(moves[split == 0].astype(np.int64))
    ho = torch.from_numpy(moves[split == 1].astype(np.int64))
    return tr, ho                                    # [G, S_LEN + 1, 2] each


# ----------------------------------------------------------------- the arms
#: name -> (mixer kind, pinned beta or None for learned, beta init, gate on)
ARMS = {
    "no-mix":        dict(mix="none",     beta=None, beta0=None, gate=False,
                          varies="the mixing operator is DELETED (o_i = v_i)",
                          pins="embedding, residual, head, optimiser, seed, batches"),
    "softmax":       dict(mix="torch",    beta=None, beta0=None, gate=False,
                          varies="nothing -- this is the operator being contained",
                          pins="everything"),
    "arm-beta1":     dict(mix="resolvent", beta=1.0, beta0=None, gate=False,
                          varies="the IMPLEMENTATION only: ceq/arm_smprime.readout "
                                 "at the proved softmax corner",
                          pins="beta = 1, gate off, and every shared module"),
    "arm-beta0":     dict(mix="resolvent", beta=0.0, beta0=None, gate=False,
                          varies="beta, pinned at the proved linear corner",
                          pins="gate off, and every shared module"),
    "arm-beta-free": dict(mix="resolvent", beta=None, beta0=1.0, gate=False,
                          varies="beta, LEARNED, started at the softmax corner",
                          pins="gate off, and every shared module"),
    "arm-beta-free-half": dict(mix="resolvent", beta=None, beta0=0.5, gate=False,
                          varies="beta, LEARNED, started HALFWAY between the corners",
                          pins="gate off, and every shared module"),
    "arm-gate":      dict(mix="resolvent", beta=1.0, beta0=None, gate="identity",
                          varies="the gate heads (m, theta) at ceq/arm_smprime.py"
                                 ":534 identity_heads' OWN init, m = 1, theta = 0",
                          pins="beta = 1, and every shared module"),
    "arm-gate-live": dict(mix="resolvent", beta=1.0, beta0=None, gate="live",
                          varies="the gate heads, started 1e-3 OFF the two "
                                 "critical points identity_heads sits on",
                          pins="beta = 1, and every shared module"),
    "arm-full":      dict(mix="resolvent", beta=None, beta0=1.0, gate="identity",
                          varies="beta AND the gate heads at the shipped init",
                          pins="every shared module"),
    "arm-full-live": dict(mix="resolvent", beta=None, beta0=1.0, gate="live",
                          varies="beta AND the gate heads at the live init -- the "
                                 "whole family, every axis able to move",
                          pins="every shared module"),
}

GATE_OFF = 1e-3       #: how far the live init steps off each critical point


class Head(nn.Module):
    """One mixing layer over the ply sequence, then a from/to read-out.

    The shared modules are built in a FIXED ORDER before any arm-specific
    parameter, so two arms drawn from the same seed hold bitwise-identical
    shared weights however many extra parameters one of them carries.
    """

    def __init__(self, spec, d=D_MODEL, s=S_LEN, n_a=N_SQ, n_out=2 * N_SQ):
        super().__init__()
        self.spec, self.s, self.d, self.n_out = spec, s, d, n_out
        self.emb_from = nn.Embedding(n_a, d // 2)
        self.emb_to = nn.Embedding(N_SQ, d // 2)
        self.pos = nn.Embedding(s, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.mlp = (nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                    if USE_MLP else None)
        self.out = nn.Linear(d, n_out)
        #: arm-specific, after the shared block
        if spec["beta"] is None and spec["mix"] == "resolvent":
            self.beta = nn.Parameter(torch.tensor(float(spec["beta0"])))
        if spec["gate"]:
            self.m_head = nn.Linear(d, 1)
            self.theta_head = nn.Linear(d, 1)
            with torch.no_grad():                 # arm_smprime.identity_heads
                for h in (self.m_head, self.theta_head):
                    h.weight.zero_()
                    h.bias.zero_()
                off = GATE_OFF if spec["gate"] == "live" else 0.0
                self.m_head.bias.fill_(1.0 - off)
                self.theta_head.bias.fill_(off)
        self.double()

    def mix(self, x):
        kind = self.spec["mix"]
        v = self.wv(x)
        if kind == "none":
            return v
        q, k = self.wq(x), self.wk(x)
        if kind == "torch":
            w = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            up = torch.ones(self.s, self.s, dtype=torch.bool,
                            device=x.device).triu(1)
            return F.softmax(w.masked_fill(up, float("-inf")), dim=-1) @ v
        beta = self.beta if self.spec["beta"] is None else self.spec["beta"]
        if self.spec["gate"]:
            u = self.m_head(x).squeeze(-1)
            th = self.theta_head(x).squeeze(-1)
            g = 1.0
        else:
            u, th, g = None, None, 0.0
        return arm.readout(q, k, v, u, th, beta=beta, qk=1.0, g=g).real

    def forward(self, idx):
        x = torch.cat([self.emb_from(idx[..., 0]), self.emb_to(idx[..., 1])], -1)
        x = x + self.pos(torch.arange(self.s, device=idx.device))
        h = x + self.mix(x)
        if self.mlp is not None:
            h = h + self.mlp(h)
        o = self.out(h)
        if self.n_out == 1:
            return o.squeeze(-1)
        return o[..., :N_SQ], o[..., N_SQ:]


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------------- train / score

def _loss(model, batch):
    lf, lt = model(batch[:, :-1])
    tgt = batch[:, 1:]
    return (F.cross_entropy(lf.reshape(-1, N_SQ), tgt[..., 0].reshape(-1))
            + F.cross_entropy(lt.reshape(-1, N_SQ), tgt[..., 1].reshape(-1)))


@torch.no_grad()
def score(model, held, chunk=512, per_game=None):
    """`per_game`, if a list, collects the per-GAME exact-correct count.

    Positions inside one game are not independent -- consecutive plies differ by
    one move -- so the binomial error bar over 22,272 decisions is wrong by the
    intra-game correlation. The cluster unit is the game and this is what the
    paired bootstrap in `paired_delta` resamples.
    """
    model.eval()
    nf = nt = ne = n = 0
    ce = 0.0
    for i in range(0, held.shape[0], chunk):
        b = held[i:i + chunk]
        lf, lt = model(b[:, :-1])
        tgt = b[:, 1:]
        pf, pt = lf.argmax(-1), lt.argmax(-1)
        gf, gt = pf == tgt[..., 0], pt == tgt[..., 1]
        nf += int(gf.sum()); nt += int(gt.sum()); ne += int((gf & gt).sum())
        if per_game is not None:
            pg_ce = (F.cross_entropy(lf.reshape(-1, N_SQ), tgt[..., 0].reshape(-1),
                                     reduction="none").view(tgt.shape[:2])
                     + F.cross_entropy(lt.reshape(-1, N_SQ), tgt[..., 1].reshape(-1),
                                       reduction="none").view(tgt.shape[:2]))
            per_game.append(torch.stack([(gf & gt).sum(-1).to(torch.float64),
                                         pg_ce.sum(-1).to(torch.float64),
                                         torch.full((b.shape[0],), float(tgt.shape[1]),
                                                    dtype=torch.float64)], -1))
        n += int(tgt[..., 0].numel())
        ce += float(F.cross_entropy(lf.reshape(-1, N_SQ), tgt[..., 0].reshape(-1),
                                    reduction="sum")
                    + F.cross_entropy(lt.reshape(-1, N_SQ), tgt[..., 1].reshape(-1),
                                      reduction="sum"))
    model.train()
    return dict(n_decisions=n, acc_from=nf / n, acc_to=nt / n, acc_exact=ne / n,
                ce_per_decision=ce / n)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Game-level paired bootstrap of `a - b` in exact moves per game.

    VARIES: which held-out GAMES are in the resample. PINS: the two trained
    models, the plies inside each game, the ply count per game (32, by
    construction of the corpus), and the pairing -- arm a and arm b are scored on
    the SAME resampled games in the same draw, so the shared difficulty of a game
    cancels instead of entering the spread.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, a.shape[0], (n_boot, a.shape[0]), generator=g)
    out = {}
    for col, name in ((0, "exact"), (1, "ce")):
        d, w = (a[:, col] - b[:, col]), a[:, 2]
        m = d[idx].sum(-1) / w[idx].sum(-1)
        lo, hi = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64))
        out["delta_" + name] = float(d.sum() / w.sum())
        out["ci95_" + name] = [float(lo), float(hi)]
        out["p_" + name] = float(2 * min((m <= 0).to(torch.float64).mean(),
                                         (m >= 0).to(torch.float64).mean()))
    out.update(n_games=int(a.shape[0]), n_boot=n_boot)
    return out


def run_arm(name, train, held, seed, steps=STEPS, batch=BATCH, lr=LR, dump=None):
    spec = ARMS[name]
    torch.manual_seed(seed)
    model = Head(spec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)       # the SAME batch sequence
    t0 = time.time()
    nonfinite = 0
    curve = []
    for step in range(steps):
        if step % 500 == 0 and step:
            curve.append(dict(step=step, **{k: round(v, 6) for k, v in
                                            score(model, held).items() if k != "n_decisions"}))
        idx = torch.randint(0, train.shape[0], (batch,), generator=g)
        loss = _loss(model, train[idx])
        if not torch.isfinite(loss):                  # L-SURFACE: asserted, not assumed
            nonfinite += 1
            raise FloatingPointError(
                "%s went non-finite at step %d: loss=%r" % (name, step, float(loss)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = time.time() - t0
    pg = [] if dump is not None else None
    out = score(model, held, per_game=pg)
    if pg is not None:
        dump[name] = torch.cat(pg)
    out.update(arm=name, params=n_params(model), seconds=round(wall, 2),
               seed=seed, steps=steps, batch=batch, lr=lr,
               varies=spec["varies"], pins=spec["pins"], nonfinite_steps=nonfinite,
               curve=curve)
    if hasattr(model, "beta"):
        out["beta_final"] = float(model.beta.detach())
        out["beta_init"] = float(spec["beta0"])
    elif spec["mix"] == "resolvent":
        out["beta_final"] = float(spec["beta"])
    if spec["gate"]:
        with torch.no_grad():
            x = held[:256, :-1]
            e = torch.cat([model.emb_from(x[..., 0]), model.emb_to(x[..., 1])], -1)
            e = e + model.pos(torch.arange(S_LEN))
            m = arm.magnitude(model.m_head(e).squeeze(-1))
            th = model.theta_head(e).squeeze(-1)
            out.update(gate_init=spec["gate"],
                       gate_m_mean=float(m.mean()), gate_m_min=float(m.min()),
                       gate_theta_absmax=float(th.abs().max()),
                       m_head_weight_absmax=float(model.m_head.weight.abs().max()),
                       theta_head_weight_absmax=float(model.theta_head.weight.abs().max()))
    return out


# ------------------------------------------------------- the committor bed
#: `CEQ_V20_R15_CONTRACT.md:42-45`, quoted at `docs/canon/CHARTER.md:50`, rules
#: next-token prediction out as the target: "predicting the NEXT STATE toward
#: equilibrium, not the next token". This bed's label is the committor -- the
#: probability a uniform-random walk reaches checkmate before a draw -- solved
#: exactly over all 368,452 KQK positions by `ceqjepa/chess_steps.py`. The input
#: is an EVENT STREAM (piece, destination square), never the board, so the state
#: is recoverable only by mixing over the prefix and the operator is what is
#: being scored rather than an encoder in front of it.
C_S_LEN = 33
C_PIECES = 4


def load_committor(path):
    #: unscored positions carry NaN, and NaN * 0 is NaN in the BACKWARD even when
    #: the forward masks them away -- the whole gradient goes non-finite at step
    #: 0. They are zeroed here; `scored` is what keeps them out of the loss.
    z = np.load(path)
    y = np.nan_to_num(z["y"]).astype(np.float64)
    hold, scored = z["hold"], z["scored"]
    #: STANDARDISE ON THE TRAIN SPLIT ONLY. q has mean 0.0157 and std 0.0533, so
    #: an un-scaled MSE is 3e-3 at the mean-predictor and Adam at lr 3e-3 steps
    #: further than the whole target range every step -- measured, the held-out
    #: R2 oscillated -0.031, -0.037, -0.454, -0.055, -0.109 over 2,500 steps and
    #: never went positive. R2 is invariant under this affine map, so the number
    #: reported is the same number; only the optimiser's footing changes.
    tr = scored & ~hold
    mu, sd = float(y[tr].mean()), float(y[tr].std())
    y = (y - mu) / sd
    return (torch.from_numpy(z["tok"].astype(np.int64)), torch.from_numpy(y),
            torch.from_numpy(hold), torch.from_numpy(scored))


def c_build(spec, seed):
    torch.manual_seed(seed)
    return Head(spec, s=C_S_LEN, n_a=C_PIECES, n_out=1)


def c_loss(model, tok, y, mask):
    return ((model(tok) - y) ** 2)[mask].mean()


@torch.no_grad()
def c_score(model, tok, y, mask, chunk=2048, per_traj=None):
    model.eval()
    se = n = 0.0
    for i in range(0, tok.shape[0], chunk):
        p, yy, m = model(tok[i:i + chunk]), y[i:i + chunk], mask[i:i + chunk]
        d = (p - yy) ** 2 * m
        se += float(d.sum())
        n += float(m.sum())
        if per_traj is not None:
            per_traj.append(torch.stack([d.sum(-1), m.sum(-1).to(d.dtype)], -1))
    model.train()
    var = float(((y[mask] - y[mask].mean()) ** 2).mean())
    return dict(n_scored=int(n), mse=se / n, rmse=(se / n) ** 0.5,
                r2=1.0 - (se / n) / var, var_holdout=var)


def c_run_arm(name, data, seed, steps=STEPS, batch=BATCH, lr=LR, dump=None):
    tok, y, hold, scored = data
    spec = ARMS[name]
    model = c_build(spec, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    tr_mask, ho_mask = scored & ~hold, scored & hold
    t0, curve = time.time(), []
    for step in range(steps):
        if step % 500 == 0 and step:
            curve.append(dict(step=step,
                              **{k: round(v, 8) for k, v
                                 in c_score(model, tok, y, ho_mask).items()
                                 if k != "n_scored"}))
        i = torch.randint(0, tok.shape[0], (batch,), generator=g)
        loss = c_loss(model, tok[i], y[i], tr_mask[i])
        if not torch.isfinite(loss):                  # L-SURFACE
            raise FloatingPointError("%s non-finite at step %d" % (name, step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = time.time() - t0
    pt = [] if dump is not None else None
    out = c_score(model, tok, y, ho_mask, per_traj=pt)
    if pt is not None:
        dump[name] = torch.cat(pt)
    out.update(arm=name, params=n_params(model), seconds=round(wall, 2), seed=seed,
               steps=steps, batch=batch, lr=lr, varies=spec["varies"],
               pins=spec["pins"], curve=curve, bed="committor")
    if hasattr(model, "beta"):
        out["beta_final"] = float(model.beta.detach())
        out["beta_init"] = float(spec["beta0"])
    elif spec["mix"] == "resolvent":
        out["beta_final"] = float(spec["beta"])
    if spec["gate"]:
        with torch.no_grad():
            e = torch.cat([model.emb_from(tok[:256, :, 0]),
                           model.emb_to(tok[:256, :, 1])], -1)
            e = e + model.pos(torch.arange(C_S_LEN))
            m = arm.magnitude(model.m_head(e).squeeze(-1))
            out.update(gate_init=spec["gate"], gate_m_mean=float(m.mean()),
                       gate_m_min=float(m.min()),
                       m_head_weight_absmax=float(model.m_head.weight.abs().max()),
                       theta_head_weight_absmax=float(
                           model.theta_head.weight.abs().max()))
    return out


def c_paired(a, b, n_boot=10000, seed=0):
    """Paired bootstrap over held-out TRAJECTORIES of the MSE difference a - b.

    VARIES: which trajectories enter the resample. PINS: the two trained models,
    the positions inside each trajectory, the holdout state set, and the pairing
    -- both arms are scored on the SAME resampled trajectories in one draw, so a
    trajectory's own difficulty cancels instead of entering the spread.
    """
    keep = a[:, 1] > 0
    d, w = (a[keep, 0] - b[keep, 0]), a[keep, 1]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
    m = d[idx].sum(-1) / w[idx].sum(-1)
    lo, hi = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return dict(delta_mse=float(d.sum() / w.sum()), ci95=[float(lo), float(hi)],
                n_traj=int(d.numel()), n_boot=n_boot,
                p_two_sided=float(2 * min((m <= 0).to(torch.float64).mean(),
                                          (m >= 0).to(torch.float64).mean())))


def c_main(a):
    data = load_committor(a.corpus)
    rows, dump = [], ({} if a.dump else None)
    for name in a.arms.split(","):
        try:
            r = c_run_arm(name, data, a.seed, steps=a.steps, batch=a.batch,
                          lr=a.lr, dump=dump)
        except FloatingPointError as e:
            r = dict(arm=name, refused="NON-FINITE", why=str(e), seed=a.seed)
        rows.append(r)
        print(json.dumps(r), flush=True)
    doc = dict(provenance=provenance(), bed="committor", seed=a.seed, steps=a.steps,
               batch=a.batch, s_len=C_S_LEN, d_model=D_MODEL, rows=rows)
    if dump:
        np.savez(a.dump, **{k: v.numpy() for k, v in dump.items()})
        doc["paired_vs_softmax"] = {k: c_paired(v, dump["softmax"])
                                    for k, v in dump.items() if k != "softmax"}
        print(json.dumps(doc["paired_vs_softmax"], indent=1), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
    return doc


def provenance():
    root = pathlib.Path(__file__).resolve().parents[1]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return dict(commit=sha, machine=platform.node(), python=platform.python_version(),
                torch=torch.__version__, threads=torch.get_num_threads())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=5501)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="npz of per-game exact counts")
    ap.add_argument("--bed", default="moves", choices=("moves", "committor"))
    a = ap.parse_args()
    if a.bed == "committor":
        return c_main(a)
    train, held = load_corpus(a.corpus)
    rows, dump = [], ({} if a.dump else None)
    for name in a.arms.split(","):
        try:
            r = run_arm(name, train, held, a.seed, steps=a.steps, batch=a.batch,
                        lr=a.lr, dump=dump)
        except FloatingPointError as e:              # a refusal, carrying its cause
            r = dict(arm=name, refused="NON-FINITE", why=str(e), seed=a.seed)
        rows.append(r)
        print(json.dumps(r), flush=True)
    doc = dict(provenance=provenance(), seed=a.seed, steps=a.steps, batch=a.batch,
               n_train_games=int(train.shape[0]), n_holdout_games=int(held.shape[0]),
               s_len=S_LEN, d_model=D_MODEL, rows=rows)
    if dump:
        np.savez(a.dump, **{k: v.numpy() for k, v in dump.items()})
        ref = "softmax"
        doc["paired_vs_softmax"] = {k: paired_bootstrap(v, dump[ref])
                                    for k, v in dump.items() if k != ref}
        print(json.dumps(doc["paired_vs_softmax"], indent=1), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    main()
