"""modeling_dcm1.py -- self-contained DCM-1 (Dirichlet Chart Machine), the
CEQ-JEPA causal-operator model. ONE file, plain torch, NO transformers and
NO dependency on the `ceqjepa` package.

WHAT THIS MODEL IS. An encoder reads a state x and emits per-example logits
for a CAUSAL (lower-triangular) row-stochastic operator P over n chart
positions. nA of those positions are declared ABSORBING and are written into
P as identity rows. Two exact linear-algebra reads come off P:

    q = committor(P, A) = (I - Q)^{-1} R        [n, nA]
        a genuine probability: for each chart position, which absorbing set
        is hit FIRST. Attains 0 and 1 exactly. ONE lower-triangular solve.

    q(do a) = the same committor after position i's outgoing row is CLAMPED
        to a row built from a candidate action. Computed by Sherman-Morrison
        as a rank-1 edit of (I - Q): ONE factorisation covers m candidate
        actions. Its denominator equals (1 - P'_ii)/(1 - P_ii); when that
        collapses the code RAISES rather than returning an amplified solve.

The difference q(do a) - q is the model's stated CONSEQUENCE of forcing
action a. Read the adverse findings in README.md before believing it: on the
chess bed this artifact was trained on, that difference was measured to carry
NO outcome information (move-permutation ablation -0.0044 +- 0.0138).

Vendored verbatim from ceqjepa/operator.py and ceqjepa/intervene.py at
package sha1 6a1be293647ee6391583c20cda72a987e9f4c726 -- the exact code that
produced these weights -- so this directory alone is loadable.
"""
import json
import os

import torch
import torch.nn as nn

__all__ = ["DCM1Model", "SingularTransientBlockError", "committor",
           "build_operator", "committor_do_batch", "fen_to_vec", "square"]

TELEPORT = 0.0125   # training teleport floor: bounds ||(I-Q)^-1||_inf <= 1/c = 80


class SingularTransientBlockError(RuntimeError):
    """(I - Q) singular, or so ill-conditioned the solve would return garbage."""


# ---------------------------------------------------------------------------
# vendored from ceqjepa/operator.py
# ---------------------------------------------------------------------------
def causal_mask(n, device=None):
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))


def absorbing_teleport(n, absorbing_idx, dtype, device=None):
    """[n,n] row-stochastic teleport target: uniform over the absorbing states
    CAUSALLY VISIBLE from each row (a <= i). Rows with none get a zero row."""
    idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=device)
    A = torch.zeros(n, n, dtype=dtype, device=device)
    if idx.numel() == 0:
        return A
    rows = torch.arange(n, device=device).unsqueeze(1)
    visible = (idx.unsqueeze(0) <= rows)
    A[rows.expand(-1, idx.numel()), idx.unsqueeze(0).expand(n, -1)] = visible.to(dtype)
    counts = A.sum(-1, keepdim=True)
    return torch.where(counts > 0, A / counts.clamp_min(1), A)


def build_operator(logits, absorbing_idx, teleport=TELEPORT):
    """Causal row-stochastic P from logits, boundary rows overwritten with I.

    At teleport=0 the g=0 read is the causal softmax BITWISE (measured max abs
    diff 0.000e+00). teleport>0 is the training scaffold: it bounds
    ||Q||_inf <= 1 - c so no setting of the logits can drive P[i,i] to 1.
    """
    if not (0.0 <= teleport < 1.0):
        raise ValueError("teleport must be in [0, 1), got %r" % (teleport,))
    if not torch.isfinite(logits).all():
        raise ValueError("build_operator: logits contain %d non-finite entries; "
                         "refusing, because softmax would propagate NaN into q and "
                         "into the conditioning number, where no threshold check "
                         "catches it ((x < tol) is False for NaN)"
                         % int((~torch.isfinite(logits)).sum()))
    n = logits.shape[-1]
    masked = logits.masked_fill(~causal_mask(n, device=logits.device), float("-inf"))
    P = torch.softmax(masked, dim=-1)
    absorbing_idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=logits.device)
    if teleport > 0.0 and absorbing_idx.numel() > 0:
        A = absorbing_teleport(n, absorbing_idx, P.dtype, P.device)
        c = (A.sum(-1, keepdim=True) > 0).to(P.dtype) * teleport
        P = (1.0 - c) * P + c * A
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    P = P.clone()
    if absorbing_idx.numel() > 0:
        P[..., absorbing_idx, :] = eye[absorbing_idx]
    return P


def state_solve(P, V, g):
    """z = (I - g P)^{-1} V by ONE triangular forward substitution; O = (1-g) P z.
    Needs no guard: diag(I - gP)_ii = 1 - g P_ii >= 1 - g > 0 for g < 1."""
    n = P.shape[-1]
    M = torch.eye(n, dtype=P.dtype, device=P.device) - g * P
    z = torch.linalg.solve_triangular(M, V, upper=False)
    return z, (1 - g) * (P @ z)


def committor(P, absorbing_idx, tol=None, kappa_max=None):
    """q = (I - Q)^{-1} R embedded back to [..., n, k].

    Journals committor.last_kappa_bound = ||(I-Q)^{-1}||_inf, EXACT (one extra
    triangular solve against the ones vector), not the 1/min(1-Q_ii) LOWER
    bound that callers used to assert against as if it were an upper bound.
    """
    if not torch.isfinite(P).all():
        raise ValueError("committor: P contains %d non-finite entries; refusing, "
                         "because (diag.abs() < tol).any() is False for NaN"
                         % int((~torch.isfinite(P)).sum()))
    n = P.shape[-1]
    device = P.device
    absorbing_idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=device)
    is_absorbing = torch.zeros(n, dtype=torch.bool, device=device)
    is_absorbing[absorbing_idx] = True
    transient_idx = torch.nonzero(~is_absorbing, as_tuple=True)[0]
    k = absorbing_idx.numel()

    Q = P[..., transient_idx, :][..., :, transient_idx]
    R = P[..., transient_idx, :][..., :, absorbing_idx]

    diag = 1.0 - torch.diagonal(Q, dim1=-2, dim2=-1)
    if tol is None:
        # dtype-aware: a fixed 1e-10 is BELOW float32 eps (1.192e-07), and a
        # saturated row once passed that guard and returned a q summing to 1.1868.
        tol = max(1e-10, transient_idx.numel() * torch.finfo(P.dtype).eps)
    if (diag.abs() < tol).any():
        raise SingularTransientBlockError(
            "transient block singular (I - Q has a diagonal entry below %.3e in %s)"
            % (tol, P.dtype))

    M = torch.eye(transient_idx.numel(), dtype=P.dtype, device=device) - Q
    q_T = torch.linalg.solve_triangular(M, R, upper=False)

    with torch.no_grad():
        Md = M.detach()
        ones = torch.ones(*Md.shape[:-1], 1, dtype=P.dtype, device=device)
        kappa = float(torch.linalg.solve_triangular(Md, ones, upper=False).abs().max())
    committor.last_kappa_bound = kappa
    if kappa_max is not None and kappa > kappa_max:
        raise SingularTransientBlockError(
            "transient block ill-conditioned: ||(I-Q)^{-1}||_inf = %.6g exceeds %.6g"
            % (kappa, kappa_max))

    q_full = torch.zeros(*P.shape[:-1], k, dtype=P.dtype, device=device)
    q_full[..., transient_idx, :] = q_T
    q_full[..., absorbing_idx, :] = torch.eye(k, dtype=P.dtype, device=device)
    return q_full


# ---------------------------------------------------------------------------
# vendored from ceqjepa/intervene.py -- THE do(a) ARM
# ---------------------------------------------------------------------------
def _check_rows(rows, i, n):
    """Raises unless every row is a causal probability row for position i:
    finite, nonneg, sums to 1, and ZERO on every j > i."""
    if rows.shape[-1] != n:
        raise ValueError("row has length %d, operator has n = %d" % (rows.shape[-1], n))
    if not torch.isfinite(rows).all():
        raise ValueError("clamped row contains non-finite entries")
    tail = rows[..., i + 1:]
    if tail.numel() and float(tail.abs().max()) != 0.0:
        raise ValueError(
            "clamped row puts mass %.6g on j > i = %d: that is an ACAUSAL edit, P "
            "would stop being lower triangular, and every downstream claim (one "
            "triangular solve, exact committor, Sherman-Morrison) depends on it"
            % (float(tail.abs().max()), i))
    if float(rows.min()) < 0.0:
        raise ValueError("clamped row has a negative entry %.6g" % float(rows.min()))
    tol = 10 * n * torch.finfo(rows.dtype).eps
    err = float((rows.sum(-1) - 1.0).abs().max())
    if err > tol:
        raise ValueError("clamped row is not stochastic: max |sum - 1| = %.3e > %.3e"
                         % (err, tol))


def _blocks(P, absorbing_idx, i):
    if P.dim() != 2 or P.shape[-1] != P.shape[-2]:
        raise ValueError("intervene: P must be a single square [n, n] operator, got %s"
                         % (tuple(P.shape),))
    if not torch.isfinite(P).all():
        raise ValueError("intervene: P contains non-finite entries")
    n = P.shape[-1]
    idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=P.device)
    is_abs = torch.zeros(n, dtype=torch.bool, device=P.device)
    is_abs[idx] = True
    if not (0 <= i < n):
        raise ValueError("intervened index %d out of range [0, %d)" % (i, n))
    if bool(is_abs[i]):
        raise ValueError(
            "index %d is DECLARED ABSORBING: clamping it is not a rank-1 edit of "
            "(I - Q) -- it moves i from A to T, changes the block partition, and "
            "changes the shape of q. Intervene on a transient position." % i)
    T = torch.nonzero(~is_abs, as_tuple=True)[0]
    t = int(torch.nonzero(T == i, as_tuple=True)[0])
    Q, R = P[T][:, T], P[T][:, idx]
    M = torch.eye(T.numel(), dtype=P.dtype, device=P.device) - Q
    return idx, T, t, Q, R, M


def _embed(q_T, T, idx, n, batch=None):
    k = idx.numel()
    shape = (n, k) if batch is None else (batch, n, k)
    q = torch.zeros(*shape, dtype=q_T.dtype, device=q_T.device)
    q[..., T, :] = q_T
    q[..., idx, :] = torch.eye(k, dtype=q_T.dtype, device=q_T.device)
    return q


def committor_do_batch(P, absorbing_idx, i, rows, den_min=None):
    """m candidate actions clamped at position i, ONE factorisation.

    rows: [m, n]. Returns (q_do [m, n, k], den [m]). den equals
    (1 - P'_ii)/(1 - P_ii); below den_min = sqrt(eps) this RAISES instead of
    returning a committor amplified by 1/den. Measured 21.65x faster than m
    separate full solves at n=512, m=32.
    """
    rows = torch.as_tensor(rows, dtype=P.dtype, device=P.device)
    if rows.dim() != 2:
        raise ValueError("rows must be [m, n], got %s" % (tuple(rows.shape),))
    idx, T, t, Q, R, M = _blocks(P, absorbing_idx, i)
    _check_rows(rows, i, P.shape[-1])
    if den_min is None:
        den_min = float(torch.finfo(P.dtype).eps) ** 0.5   # f32 -> 3.453e-04

    D = rows[:, T] - Q[t]
    DR = rows[:, idx] - R[t]
    q_T = torch.linalg.solve_triangular(M, R, upper=False)          # shared
    e_t = torch.zeros(T.numel(), 1, dtype=P.dtype, device=P.device)
    e_t[t, 0] = 1.0
    c = torch.linalg.solve_triangular(M, e_t, upper=False)          # shared
    W = torch.linalg.solve_triangular(M.transpose(-1, -2), D.transpose(0, 1), upper=True)
    den = 1.0 - W[t]
    if bool((den.abs() < den_min).any()):
        bad = torch.nonzero(den.abs() < den_min, as_tuple=True)[0]
        raise SingularTransientBlockError(
            "Sherman-Morrison denominator below %.3e for %d of %d candidate rows "
            "(first: row %d, den = %.3e): those actions make position %d "
            "self-absorbing. Refusing the whole batch."
            % (den_min, bad.numel(), rows.shape[0], int(bad[0]), float(den[bad[0]]), i))
    upd = (W.transpose(0, 1) @ R) + DR
    q_do_T = q_T.unsqueeze(0) + (c.squeeze(-1)[None, :, None] * upd[:, None, :]) / den[:, None, None]
    return _embed(q_do_T, T, idx, P.shape[-1], batch=rows.shape[0]), den


# ---------------------------------------------------------------------------
# input encoding -- 769 floats, reproducible without python-chess
# ---------------------------------------------------------------------------
_PLANE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}   # matches ceqjepa/beds/chess.py


def square(name):
    """'e2' -> 12. a1=0 .. h8=63, python-chess ordering."""
    return (int(name[1]) - 1) * 8 + (ord(name[0]) - ord('a'))


def fen_to_vec(fen):
    """769-float board vector: 12 piece planes x 64 squares, then side-to-move
    (1.0 = white). Same layout as ceqjepa/beds/chess.py:fen_to_vec, rewritten
    against the FEN string so this file needs no python-chess."""
    parts = fen.split()
    placement, turn = parts[0], parts[1]
    v = torch.zeros(769)
    for rank_i, rank in enumerate(placement.split('/')):        # FEN goes rank 8 -> 1
        f = 0
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            else:
                v[_PLANE[ch] * 64 + (7 - rank_i) * 8 + f] = 1.0
                f += 1
    v[-1] = 1.0 if turn == 'w' else 0.0
    return v


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class DCM1Model(nn.Module):
    """Same module as ceqjepa.train.TinyCEQ, renamed for the artifact.

    n=256 chart positions, nA=4 absorbing outcomes (white_win, draw,
    black_win, sink), rank-16 per-example modulation of the shared chart L0.
    """

    def __init__(self, n, nA, d_enc, x_dim, z_dim_state, g, absorbing_idx,
                 rank=16, d_mv=8, teleport=TELEPORT):
        super().__init__()
        self.n, self.nA, self.rank, self.teleport = n, nA, rank, teleport
        self.register_buffer('absorbing_idx', torch.as_tensor(absorbing_idx, dtype=torch.long))
        self.register_buffer('g', torch.tensor(float(g)))
        self.L0 = nn.Parameter(torch.zeros(n, n))
        self.Vt = nn.Parameter(torch.zeros(n, z_dim_state))
        self.enc = nn.Sequential(nn.Linear(x_dim, d_enc), nn.GELU(), nn.Linear(d_enc, d_enc))
        # rank-r per-example modulation: delta_logits = a @ b.T, factored to the
        # DCM-1 spec's rank instead of a dense [n,n] map.
        self.delta_a = nn.Linear(d_enc, n * rank)
        self.delta_b = nn.Linear(d_enc, n * rank)
        self.chart = nn.Linear(d_enc, n)                    # alpha read weights
        self.readout = nn.Linear(z_dim_state + nA, x_dim)
        # the do(a) arm's only parameters: (from_square, to_square) -> additive
        # bias on position i's own logit row. Promotion piece is dropped, so
        # e7e8q and e7e8n share an embedding.
        self.mv_from = nn.Embedding(64, d_mv)
        self.mv_to = nn.Embedding(64, d_mv)
        self.mv_row = nn.Linear(2 * d_mv, n)
        self.register_buffer('A_tel', absorbing_teleport(n, self.absorbing_idx, torch.float32))

    def forward(self, x):
        B = x.shape[0]
        e = self.enc(x)
        a = self.delta_a(e).view(B, self.n, self.rank)
        b = self.delta_b(e).view(B, self.n, self.rank)
        logits = self.L0.unsqueeze(0) + torch.einsum('bnr,bmr->bnm', a, b)
        P = build_operator(logits, self.absorbing_idx, teleport=self.teleport)
        q_field = committor(P, self.absorbing_idx)                       # [B,n,nA]
        Vt_b = self.Vt.unsqueeze(0).expand(B, -1, -1)
        z, O = state_solve(P, Vt_b, float(self.g))
        alpha = torch.softmax(self.chart(e), dim=-1)
        q_alpha = torch.einsum('bn,bnk->bk', alpha, q_field)             # THE SUPERVISED READ
        h = torch.einsum('bn,bnd->bd', alpha, O)
        x_hat = self.readout(torch.cat([h, q_alpha], dim=-1))
        return dict(q_alpha=q_alpha, q_field=q_field, h=h, x_hat=x_hat,
                    alpha=alpha, P=P, e=e, logits=logits)

    def do_read(self, out, moves):
        """q(do a) at the intervened position, for m candidate moves per example.

        moves: [B, m, 2] long (from_square, to_square).
        Returns (q_do [B, m, nA], q_base [B, nA], i_star [B], ok [B] bool).

        i_star = the transient chart position the model's own read alpha puts
        the most weight on (restricted to j >= nA: clamping a declared-absorbing
        row is not a rank-1 edit of (I - Q)). q_base is the UNINTERVENED
        committor at that same position, so q_do - q_base is the model's stated
        consequence of forcing the move.

        ok[b] is False where Sherman-Morrison REFUSED (denominator collapsed);
        those rows of q_do are zero and must be dropped, never read as a value.
        """
        B, m, _ = moves.shape
        n, nA = self.n, self.nA
        alpha, logits, P = out['alpha'], out['logits'], out['P']
        i_star = nA + alpha[:, nA:].argmax(dim=-1)
        mv = torch.cat([self.mv_from(moves[..., 0]), self.mv_to(moves[..., 1])], dim=-1)
        bias = self.mv_row(mv)                                           # [B,m,n]
        ar = torch.arange(n, device=alpha.device)
        ok = torch.zeros(B, dtype=torch.bool, device=P.device)
        q_do, q_base = [], []
        for b in range(B):
            i = int(i_star[b])
            rl = (logits[b, i].unsqueeze(0) + bias[b]).masked_fill(
                (ar > i).unsqueeze(0), float('-inf'))
            row = torch.softmax(rl, dim=-1)
            a_i = self.A_tel[i]
            c = self.teleport if float(a_i.sum()) > 0 else 0.0
            row = (1.0 - c) * row + c * a_i          # same family as P's own rows
            q_base.append(out['q_field'][b, i])
            try:
                qd, _den = committor_do_batch(P[b], self.absorbing_idx, i, row)
            except (SingularTransientBlockError, ValueError):
                q_do.append(torch.zeros(m, nA, dtype=P.dtype, device=P.device))
                continue
            q_do.append(qd[:, i, :])
            ok[b] = True
        return torch.stack(q_do), torch.stack(q_base), i_star, ok

    @classmethod
    def from_pretrained(cls, path):
        """config.json + model.safetensors -> a model in eval() mode.
        Plain json + safetensors. No transformers, no ceqjepa."""
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        model = cls(n=cfg["n"], nA=cfg["nA"], d_enc=cfg["d_enc"], x_dim=cfg["x_dim"],
                    z_dim_state=cfg["z_dim_state"], g=cfg["g"],
                    absorbing_idx=cfg["absorbing_idx"], rank=cfg["rank"],
                    d_mv=cfg["d_mv"], teleport=cfg["teleport"])
        st = os.path.join(path, "model.safetensors")
        if not os.path.exists(st):
            raise FileNotFoundError("no model.safetensors in %s" % path)
        from safetensors.torch import load_file
        missing, unexpected = model.load_state_dict(load_file(st), strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.eval()
        return model
