"""Does a resolvent read generalise across a kappa shift the operator was never trained on?

THE CLAIM ON TRIAL. Under an intervention applied to the TRANSITION OPERATOR -- not
the embedding -- a resolvent read's committor error is invariant to kappa, while a
learned direct read's error grows with the kappa-shift. The mechanism is that
q = Qq + R holds exactly for EVERY (Q,R), including ones outside the training
support; a learned map s -> q is bounded only on its training distribution.

WHAT THIS IS NOT. It is NOT Theorem 4. Theorem 4 (depth L errs by Q^{L+1}(I-Q)^{-1}R,
so L >= kappa ln(1/eps)) was retired as a separation argument, because it pins every
Neumann coefficient at 1: with FITTED coefficients a degree-8 polynomial drops from
112.6157% of the committor's range to 1.448e-02% at kappa=94.08, and Cayley-Hamilton
makes degree n-1 exact outright. The separation under test here is about
GENERALISATION, not representation. No representation claim is made or needed.

WHAT THE KNOB ACTUALLY DOES, confirmed before anything was built on it (RUN, n=64,
k=3, the 512 held-out test seeds):

    self_loop  0.00 -> 0.98      kappa 1.94 -> ~94      max|q(0.98) - q(0.00)| ~ 4e-15

The self_loop knob is a TIME CHANGE. It rescales how long the chain lingers in each
transient state and leaves the embedded jump chain -- hence the committor -- bitwise
fixed. So "out of distribution" here means EXACTLY ONE THING: the input operator moves
far outside its training support while the target does not move at all. That is the
cleanest possible isolation of the kappa shift, and it is also why the constant
predictor is a serious baseline rather than a formality: a predictor that ignores the
input is trivially kappa-invariant, so "invariant" is worthless unless it is also
BETTER than the mean.

THE INPUT, and why it is fair. Both readers see the identical tensor:

    x = concat( tril(Q) including the diagonal ,  vec(R) )      dim 2272 at n=64,k=3

Raw operator entries, no projection, no standardisation (a standardiser fitted on the
training self_loops would itself be a kappa-dependent transform and could manufacture
the effect). Neither arm sees anything the other does not. Neither arm is supervised on
Q: BOTH are trained on the committor alone, so the operator arm gets no extra label.
Both arms carry the same matched output prior -- rows of q sum to 1, so both end in a
softmax over k. The arms differ in ONE place: what sits between the trunk and the loss.

    (a) OPERATOR READ   trunk -> (Q_hat, R_hat) -> q_hat = (I - Q_hat)^{-1} R_hat
                        The solve is exact and is NOT learned.
    (b) DIRECT READ     trunk -> q_hat.  No solve.
    (c) TRUNCATED READ  identical to (a) with the solve replaced by sum_{h<=L} Q^h R.
                        This is the control that separates "the solve generalises" from
                        "the Q-head generalises". If (c) tracks (a), the solve is not
                        what is doing the work.

Parameter counts are matched by bisecting the direct arm's width, the discipline v2.py
used to reach ratio 1.005, and the achieved ratio is printed rather than asserted.

KILL CONDITION, fixed before any number was seen and not moved afterwards: if the
direct read's OOD residual comes within +0.020 (the repo's frozen bar) of the operator
read's on 5/5 seeds, the claim is DEAD.

THE CLAIM IS DEAD. It fired 5/5, and not narrowly -- the direct read is BETTER out of
distribution on every seed. MAE over all n*k committor entries, 5 model seeds, 512
held-out instance seeds, 3000 steps, CPU (RUN, 459.5 s):

    arm                     in-dist            out-of-dist        ood - id
    constant (train-mean)   0.0774             0.0774             +0.0000
    operator (solve)        0.0620 (sd .0003)  0.0741 (sd .0017)  +0.0121
    direct (mlp)            0.0416 (sd .0003)  0.0718 (sd .0001)  +0.0302
    operator (trunc L=8)    0.0618 (sd .0003)  0.0735 (sd .0011)  +0.0118
    operator SHUFFLED       0.0777             0.0776
    direct SHUFFLED         0.0778             0.0777

    per-seed (direct_ood - operator_ood): -0.0007 -0.0025 -0.0051 -0.0009 -0.0026

Both halves of the claim fail. The operator read's error is NOT invariant to kappa
(+0.0121), and the direct read still wins OOD despite degrading more, because it starts
far lower. Out of distribution both arms have collapsed most of the way to the mean:
0.0741 and 0.0718 against a constant baseline of 0.0774.

WHY, and this is the part that generalises past this bed. The exactness of q = Qq + R
is real, but it lives in the SOLVE, and the model still has to PRODUCE (Q,R). That
production is a learned map bounded only on its training distribution -- the same bound
the direct read carries. Routing through a resolvent MOVES the generalisation burden
from the read to the Q-head; it does not remove it. Measured on the trained head:

    input   kappa(Q_hat) median   kappa(Q_true) median   mean diag(Q_hat) vs true
    id           2.44                   1.93                 0.020  vs  0.000
    ood          1.34                  93.51                 0.003  vs  0.965

The head never learns to emit a self-loop. At OOD inputs it returns a near-descent chain
of kappa 1.34 while the true operator sits at 93.51. So the solve is never asked to do
anything a short truncation could not, which is exactly why the truncation control
matches it to 0.0006 at BOTH ends. The control separates "the solve generalises" from
"the Q-head generalises" and answers: neither, and the solve contributes nothing.

Not a budget artefact. At 3x the steps (9000, seed 0) the operator arm's in-distribution
error improves 0.0620 -> 0.0548 while OOD stays pinned at 0.0740. More training buys
in-distribution accuracy only, and the arm remains worse than the direct read at both
ends.

THE NEAREST SURVIVING CLAIM, measured here and not dependent on this bed: an exact solve
placed downstream of a learned operator head does not inherit the solve's exactness,
because the head is the distribution-bounded component. Any separation argument of the
form "the resolvent is exact for every (Q,R)" must first show the head recovers (Q,R)
out of support -- which it does not here, by a factor of 70 in kappa. Note also that
kappa-invariance is FREE on a time-change bed: the constant predictor is exactly
invariant (0.0774 = 0.0774) while predicting nothing. Invariance alone is not evidence.
"""

import time

import torch
import torch.nn as nn

from ceqjepa.beds.kappa_bed import build

__all__ = ["make_split", "OperatorRead", "DirectRead", "match_width", "run"]

N, K = 64, 3
TRAIN_LOOPS = (0.0, 0.5)
OOD_LOOP = 0.98
SEED_ROOT = 7
DEPTH = 8                 # the truncation control's depth
#: Keeps I - Q_hat away from singular. 1e-3 caps the representable kappa at ~1000,
#: comfortably ABOVE the OOD bed's ~94, so the operator arm is not hobbled by it.
#: (op.TELEPORT = 0.0125 caps at 80, BELOW the OOD kappa, and is deliberately not used.)
TELEPORT = 1e-3
FROZEN_BAR = 0.020        # the repo's existing threshold, not invented here

_TRIL = torch.tril_indices(N, N, 0)


def x_of(b):
    """The shared input: raw lower-triangular Q entries (diagonal included) then R."""
    return torch.cat([b.Q[_TRIL.unbind()], b.R.reshape(-1)]).float()


def make_split(n_train=2000, n_val=256, n_test=512):
    """Instance seeds are disjoint across train/val/test; only self_loop differs OOD.

    ID and OOD test share the SAME instance seeds, so their committor targets are
    identical and the only thing that moves between them is the operator.
    """
    off = SEED_ROOT * 10000
    tr = range(off, off + n_train)
    va = range(off + n_train, off + n_train + n_val)
    te = range(off + n_train + n_val, off + n_train + n_val + n_test)

    def pack(seeds, loops):
        xs, qs, kap = [], [], []
        for sl in loops:
            for s in seeds:
                b = build(n=N, k=K, self_loop=sl, seed=s)
                xs.append(x_of(b))
                qs.append(b.q.reshape(-1).float())
                kap.append(b.kappa)
        return torch.stack(xs), torch.stack(qs), torch.tensor(kap)

    return {"train": pack(tr, TRAIN_LOOPS), "val": pack(va, TRAIN_LOOPS),
            "id": pack(te, TRAIN_LOOPS), "ood": pack(te, (OOD_LOOP,))}


def _trunk(x_dim, h):
    return nn.Sequential(nn.Linear(x_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU())


class OperatorRead(nn.Module):
    """q_hat = (I - Q_hat)^{-1} R_hat, or its depth-L truncation when mode='trunc'."""

    def __init__(self, x_dim, h=64, mode="solve"):
        super().__init__()
        self.mode = mode
        self.trunk = _trunk(x_dim, h)
        self.head = nn.Linear(h, N * N + N * K)
        self.register_buffer("_mask", torch.ones(N, N, dtype=torch.bool).tril())
        self.register_buffer("_eye", torch.eye(N))

    def forward(self, x):
        z = self.head(self.trunk(x))
        ql = z[:, :N * N].view(-1, N, N).masked_fill(~self._mask, float("-inf"))
        rl = z[:, N * N:].view(-1, N, K)
        both = torch.cat([ql, rl], -1).softmax(-1)
        c = TELEPORT
        Q = (1 - c) * both[..., :N]
        R = (1 - c) * both[..., N:] + c / K
        if self.mode == "solve":
            q = torch.linalg.solve_triangular(self._eye - Q, R, upper=False)
        else:
            q, M = R.clone(), R
            for _ in range(DEPTH):
                M = Q @ M
                q = q + M
        return q.reshape(x.shape[0], -1)


class DirectRead(nn.Module):
    """q_hat straight out of an MLP. Same input, same output prior, no solve."""

    def __init__(self, x_dim, h):
        super().__init__()
        self.trunk = _trunk(x_dim, h)
        self.head = nn.Linear(h, N * K)

    def forward(self, x):
        return self.head(self.trunk(x)).view(-1, N, K).softmax(-1).reshape(x.shape[0], -1)


def _nparam(m):
    return sum(p.numel() for p in m.parameters())


def match_width(target, x_dim, lo=4, hi=4096):
    """Bisect the direct arm's width to the operator arm's parameter count (v2.py's rule)."""
    best = None
    while lo <= hi:
        w = (lo + hi) // 2
        c = _nparam(DirectRead(x_dim, w))
        if best is None or abs(c - target) < abs(best[1] - target):
            best = (w, c)
        if c < target:
            lo = w + 1
        else:
            hi = w - 1
    return best


@torch.no_grad()
def mae(model, split):
    return float((model(split[0]) - split[1]).abs().mean())


def train(model, data, seed, steps=3000, bs=256, lr=1e-3, shuffle_labels=False, dev="cpu"):
    """Model selection is on the ID validation set ONLY. OOD is never looked at here."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    model.to(dev)
    X, Y = data["train"][0].to(dev), data["train"][1].to(dev)
    if shuffle_labels:
        Y = Y[torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(seed))]
    val = (data["val"][0].to(dev), data["val"][1].to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed + 1)
    best, best_state = float("inf"), None
    for t in range(steps):
        i = torch.randint(0, X.shape[0], (bs,), generator=g)
        loss = (model(X[i]) - Y[i]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (t + 1) % 250 == 0:
            v = mae(model, val)
            if v < best:
                best, best_state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"val": best,
            "id": mae(model, (data["id"][0].to(dev), data["id"][1].to(dev))),
            "ood": mae(model, (data["ood"][0].to(dev), data["ood"][1].to(dev)))}


def run(dev="cpu", steps=3000, seeds=(0, 1, 2, 3, 4)):
    t0 = time.time()
    data = make_split()
    x_dim = data["train"][0].shape[1]
    print("BED  n=%d k=%d  train self_loop=%s -> test self_loop=%.2f  seed root %d" %
          (N, K, TRAIN_LOOPS, OOD_LOOP, SEED_ROOT))
    for name in ("train", "id", "ood"):
        kap = data[name][2]
        print("  %-5s %5d instances  kappa %.2f..%.2f (median %.2f)" %
              (name, kap.numel(), kap.min(), kap.max(), kap.median()))
    print("  the knob is a TIME CHANGE: max|q_ood - q_id| = %.3e  (the target does not move)"
          % float((data["ood"][1] - data["id"][1][:data["ood"][1].shape[0]]).abs().max()))

    op_n = _nparam(OperatorRead(x_dim))
    w, dir_n = match_width(op_n, x_dim)
    print("PARAMETER MATCH  operator=%d  direct=%d (width %d)  ratio=%.3f  input dim=%d"
          % (op_n, dir_n, w, dir_n / op_n, x_dim))

    qm = data["train"][1].mean(0, keepdim=True)
    const = {k: float((qm - data[k][1]).abs().mean()) for k in ("id", "ood")}
    print("CONTROL constant predictor (train-mean q): id=%.4f  ood=%.4f" % (const["id"], const["ood"]))

    arms = {
        "operator(solve)": (lambda: OperatorRead(x_dim, mode="solve"), False),
        "direct(mlp)": (lambda: DirectRead(x_dim, w), False),
        "operator(trunc L=%d)" % DEPTH: (lambda: OperatorRead(x_dim, mode="trunc"), False),
        "operator SHUFFLED": (lambda: OperatorRead(x_dim, mode="solve"), True),
        "direct SHUFFLED": (lambda: DirectRead(x_dim, w), True),
    }
    res = {}
    for name, (mk, shuf) in arms.items():
        m = mk()
        rows = [train(m, data, s, steps=steps, shuffle_labels=shuf, dev=dev) for s in seeds]
        res[name] = rows
        idm = torch.tensor([r["id"] for r in rows])
        om = torch.tensor([r["ood"] for r in rows])
        print("%-22s  id %.4f (sd %.4f)   ood %.4f (sd %.4f)   ood-id %+.4f"
              % (name, idm.mean(), idm.std(), om.mean(), om.std(), om.mean() - idm.mean()))
    print("wall %.1fs on %s" % (time.time() - t0, dev))
    return res, const, {"operator": op_n, "direct": dir_n, "ratio": dir_n / op_n}


if __name__ == "__main__":
    import sys

    dev = "cuda" if "--cuda" in sys.argv else "cpu"
    steps = 3000
    for a in sys.argv[1:]:
        if a.startswith("--steps="):
            steps = int(a.split("=")[1])
    res, const, pc = run(dev=dev, steps=steps)

    print()
    print("(a) PARAMETER MATCH must be within a few percent, or the arms are not comparable")
    assert abs(pc["ratio"] - 1.0) < 0.03, "arms not parameter-matched: ratio %.3f" % pc["ratio"]

    print("(b) MUST-FIRE: the shuffled-label arms must FAIL, or the metric reads the bed")
    for name in ("operator SHUFFLED", "direct SHUFFLED"):
        m = sum(r["ood"] for r in res[name]) / len(res[name])
        print("    %-20s ood=%.4f  vs constant %.4f" % (name, m, const["ood"]))
        assert m >= const["ood"] - 1e-3, "%s beat the constant predictor on shuffled labels" % name

    print("(c) MUST-FIRE: both live arms must BEAT the constant predictor in-distribution,")
    print("    or there is nothing to compare out of distribution")
    for name in ("operator(solve)", "direct(mlp)"):
        m = sum(r["id"] for r in res[name]) / len(res[name])
        print("    %-20s id=%.4f  vs constant %.4f" % (name, m, const["id"]))
        assert m < const["id"], "%s does not beat the constant predictor even in-distribution" % name

    print("(d) THE KILL CONDITION, fixed in advance: direct within +%.3f of operator OOD" % FROZEN_BAR)
    print("    on 5/5 seeds kills the claim.")
    op, di = res["operator(solve)"], res["direct(mlp)"]
    within = [d["ood"] - o["ood"] < FROZEN_BAR for o, d in zip(op, di)]
    print("    per-seed (direct_ood - operator_ood): %s" %
          " ".join("%+.4f" % (d["ood"] - o["ood"]) for o, d in zip(op, di)))
    print("    seeds where direct is within the bar: %d/5" % sum(within))
    print("    VERDICT: the claim is %s" % ("DEAD" if all(within) else "ALIVE"))
