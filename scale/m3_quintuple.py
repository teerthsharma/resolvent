"""M3, the deciding measurement: five cells, batched, bucketed.

Pre-registration: `M3_QUINTUPLE_PREREGISTERED_READING.md`, written before this
file produced a number. Nothing here may be read against a different outcome
table than the one in that document.

THE FIVE CELLS. Cells 3, 4 and 5 share every parameter tensor and every line of
the forward except how the pivot weight vector `alpha` is obtained, so a
contrast between them is a contrast between settling rules, not architectures.

    softmax  the shipped arm, reproduced against its published reading first
    glance   ARM S at t_max = 0; bitwise-bound to softmax by G3
    settled  ARM S, log-domain fixed point, beta = 0.5, implicit gradient N = 21
    twin     alpha = the normalised gate; the setup is paid, the loop is not
    argmax   alpha = one-hot at argmax(gate); the attribution control

WHY CELL 5 EXISTS. Foreman's ARM S report closed on `log_alpha min -182.7498
max -0.0` -- a fixed point that is unique, reached and lopsided. A lopsided
equilibrium may carry nothing a single highest-weight pivot lookup does not, and
no birth gate asks. Cell 5 asks.

MEASURED BEFORE THE RUN, AND IT CHANGES THE READING. At ARM A's geometry alpha
is near ONE-HOT. At M3's geometry it is near UNIFORM -- `make_batch` scales x by
0.1, so the logits are tiny and the query row's softmax over its pivots is flat.
Measured at random init, s=64, d=24, mean over seeds 0-4:

    k    ||settled-twin||/||twin||   ||settled-argmax||/||argmax||   gate max
    8            1.107578e-01                 5.874960e-01          0.129379
    16           1.164525e-01                 6.086490e-01          0.063773
    32           1.277286e-01                 6.331066e-01          0.032274

(1/8 = 0.125, 1/16 = 0.0625, 1/32 = 0.03125 are the uniform values.) The three
cells are therefore genuinely distinct objects here and the gate CAN fire, which
is the precondition this project has failed five times. It also means T1's
premise -- a lopsided equilibrium -- is NOT the regime the money run sits in.
The argmax control is still run, because "distinct at init" is not "distinct
after training".

BATCHING. `scale/arm_s.py` is written per example; M3 trains on [n, s, d] with
n = 8192, so a Python loop inside the training loop is 8192 x 150 settle calls
per unit and is not affordable. The map is batched here and bound BITWISE
against the per-example original by `_bind_batched_against_arm_s`, which runs
before any cell and aborts the run if it fails.

THE GRADIENT. `BatchedSettled` is the implicit form of contract 1.2, in the log
domain: the backward never forms J, and each of the N-1 Neumann terms is one
vector-Jacobian product through a single re-evaluation of the step map at the
fixed point. N = 21 comes from `arm_s.neumann_for(0.5)` and is known before the
run rather than fitted after it. The log domain is used because the shipped
forward is log-domain; the probability-domain `arm_s.Settled` was checked
against it and agrees to 2.998307e-44 / 1.314968e-17 / 7.758876e-13 at seeds
0/1/2 of the ARM A geometry, so the two are the same map -- but only one of them
is the one Foreman ships, and the measured object must be the shipped object.

COST. Reported as analytic FLOPs (`scale/m3_flops.py`) or not at all. K-F is
UNDECIDED and the box is contended; every seconds figure in this file's output
is labelled PROVISIONAL and never enters a verdict sentence.

THREADS ARE PINNED IN THIS FILE. BUCKETED (ADR-001): one unit is one
(cell, k, seed).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                      # pinned HERE, not by the launcher

from ceq import bench                                              # noqa: E402
from scale import arm_s as A                                       # noqa: E402
from scale import m3_capability as M3                              # noqa: E402
from scale import negation_scope as NS                             # noqa: E402
from scale import paired_arm as PA                                 # noqa: E402
from scale.bucket import run_bucket, require_complete              # noqa: E402
from scale.m3_synthetic_settled import contrast                    # noqa: E402
from scale.pivot_probe import select_pivots                        # noqa: E402

NAME = "m3_quintuple_v2"
CELLS = ("softmax", "glance", "settled", "twin", "argmax")
BETA = 0.5

#: The shipped task. Every reading already in results/m3_quintuple_v2.jsonl was
#: taken on it, and `_key` appends nothing for it, so those 25 units keep their
#: byte-identical keys and resume instead of re-running (6685.3 s of journalled
#: work, by their own meta.seconds).
SHIPPED_TASK = "negation_scope"

#: Tasks whose label is `equilibrium_oracle` -- the signed path sum that the ceq
#: resolvent already computes. LOOP_PROMPT.md 1.7d: an arm built on that
#: resolvent reproduces its own forward, so on these tasks `settled - softmax`
#: is credited NOTHING and only `settled - twin` carries credit. Named by
#: ORACLE IDENTITY rather than by a name prefix, so a task registered later
#: against the same oracle is caught without editing this line.
RIGGED_ORACLE_TASKS = frozenset(
    n for n, e in NS.M3_TASKS.items() if e[1] is NS.equilibrium_oracle)
RIG_NOTE = ("VOID -- shares e1_anchor's oracle, arm reproduces its own "
            "forward (LOOP_PROMPT.md 1.7d)")

WEIGHTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / (
    NAME + "_weights")
PUBLISHED_SOFTMAX_8192 = dict(
    s=64, d=24, steps=150, n_train=8192, n_eval=512, seed=0, n_params=4769,
    nrmse0_train=1.003287, nrmse0_eval=1.000340,
    train_nrmse=0.820513, eval_nrmse=0.877168,
    ci_lo=0.830455, ci_hi=0.924226,
)


# ------------------------------------------------------------- the batched map
def batched_pivots(kk: torch.Tensor, k_piv: int) -> torch.Tensor:
    """[n, k] pivot indices, one row per example, matching `arm_s.pivots_of`.

    `arm_s.pivots_of` excludes the query row and then drops pivot 0, whose row
    is entirely zero under a strictly-causal operator and so has empty support.
    Dropping a variable number of entries per example would ragged the batch, so
    row 0 is excluded up front instead of dropped after -- which selects the
    same set, because `select_pivots(exclude=(0, s-1))` cannot return 0 at all.
    `_bind_batched_against_arm_s` checks that claim by value rather than
    trusting this paragraph.
    """
    n, s, _ = kk.shape
    out = [select_pivots(kk[i], min(k_piv, s - 3), exclude=(0, s - 1))
           for i in range(n)]
    return torch.stack(out)


def batched_log_pivot_context(q, kk, v, piv, *, chunk: int = 512,
                              need_gram: bool = True):
    """Batched `arm_s.log_pivot_context`. Returns (log_gram, log_gate, av).

    Chunked over the BATCH so the [n, k, k, s] intermediate never materialises
    whole: at n = 8192, k = 8, s = 64 that tensor is 33 554 432 float64 entries,
    268 435 456 bytes.

    `need_gram=False` skips the Gram entirely and returns None for it. Only the
    settled cell reads it; twin and argmax need the gate alone, and the Gram is
    the dominant term, so building it for them would price three cells at the
    cost of one and make any cost statement about the twin false.
    """
    n, s, d = q.shape
    k = piv.shape[1]
    _, nm = bench._causal_mask_pair(s, 0, str(q.device))
    lg = torch.empty(n, k, k, dtype=torch.float64) if need_gram else None
    lgate = torch.empty(n, k, dtype=torch.float64)
    av = torch.empty(n, k, v.shape[-1], dtype=torch.float64)
    idx_last = torch.tensor([s - 1])
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        pv = piv[lo:hi]                                            # [b, k]
        want = torch.cat([pv, idx_last.expand(hi - lo, 1)], dim=1)  # [b, k+1]
        qs = torch.gather(q[lo:hi], 1,
                          want.unsqueeze(-1).expand(-1, -1, d)).double()
        w = qs @ kk[lo:hi].double().transpose(-2, -1) / math.sqrt(d)
        w = w.masked_fill(nm[want], float("-inf"))
        la = torch.log_softmax(w[:, :-1], dim=-1)                  # [b, k, s]
        lgate[lo:hi] = torch.gather(torch.log_softmax(w[:, -1], dim=-1), 1, pv)
        if need_gram:
            lg[lo:hi] = torch.logsumexp(la.unsqueeze(2) + la.unsqueeze(1),
                                        dim=-1)
        av[lo:hi] = la.exp() @ v[lo:hi].double()
    return lg, lgate, av


def batched_log_alpha_step(log_alpha, log_gate, log_gram, beta):
    """One application of T, in logs, batched. `arm_s.log_alpha_step` verbatim
    with a leading batch dimension."""
    lw = log_gate + beta * torch.logsumexp(log_gram + log_alpha.unsqueeze(1),
                                           dim=-1)
    return lw - torch.logsumexp(lw, dim=-1, keepdim=True)


def batched_settle_log(log_gate, log_gram, beta, max_steps):
    """Fixed number of steps, no early exit: every example in the batch does the
    same work, which keeps this one kernel path rather than n of them."""
    la = log_gate - torch.logsumexp(log_gate, dim=-1, keepdim=True)
    for _ in range(max_steps):
        la = batched_log_alpha_step(la, log_gate, log_gram, beta)
    return la


class BatchedSettled(torch.autograd.Function):
    """The fixed point, differentiated implicitly (contract 1.2), log domain.

    The backward never forms J. Each of the N-1 Neumann terms is one
    vector-Jacobian product through a single re-evaluation of the step map at
    the fixed point, so the truncation error is bounded by kappa**N/(1-kappa)
    with kappa = beta exactly.
    """

    @staticmethod
    def forward(ctx, log_gate, log_gram, beta, max_steps, n_neumann):
        with torch.no_grad():
            la = batched_settle_log(log_gate, log_gram, beta, max_steps)
        ctx.save_for_backward(log_gate, log_gram, la)
        ctx.beta, ctx.n_neumann = beta, n_neumann
        return la

    @staticmethod
    def backward(ctx, grad_out):
        log_gate, log_gram, la = ctx.saved_tensors
        lg = log_gate.detach().requires_grad_(True)
        lG = log_gram.detach().requires_grad_(True)
        a = la.detach().requires_grad_(True)
        with torch.enable_grad():
            out = batched_log_alpha_step(a, lg, lG, ctx.beta)
            y = grad_out.clone()
            acc = grad_out.clone()
            for _ in range(ctx.n_neumann - 1):
                y = torch.autograd.grad(out, a, y, retain_graph=True)[0]
                acc = acc + y
            gg, gG = torch.autograd.grad(out, (lg, lG), acc,
                                         retain_graph=False)
        return gg, gG, None, None, None


# ------------------------------------------------------------------- the arm
class QuintArm(M3.Arm):
    """The shipped softmax arm with row s-1's reading replaced by a pivot
    reading. Every cell shares this class; `cell` selects the alpha rule.

    Only row s-1 can reach the output (`readout(mlp(z))[:, s-1]` and the MLP is
    position-wise), so only that row is replaced. The other rows are left
    exactly as the shipped forward produced them, which is what makes the
    softmax and glance cells reproducible against the published reading.
    """

    def __init__(self, kind: str, s: int, *, cell: str = "softmax",
                 k_piv: int = 8, beta: float = BETA, t_max: int = 21,
                 n_neumann: int = 21):
        super().__init__("softmax", s)
        if cell not in CELLS:
            raise ValueError(cell)
        self.cell, self.k_piv, self.beta = cell, k_piv, beta
        self.t_max, self.n_neumann = t_max, n_neumann

    def _alpha(self, q, kk, x):
        piv = batched_pivots(kk, self.k_piv)
        need_gram = self.cell == "settled"
        log_gram, log_gate, av = batched_log_pivot_context(
            q, kk, x, piv, need_gram=need_gram)
        if self.cell == "settled":
            la = BatchedSettled.apply(log_gate, log_gram, self.beta,
                                      self.t_max, self.n_neumann)
            alpha = la.exp()
        elif self.cell == "twin":
            alpha = (log_gate
                     - torch.logsumexp(log_gate, dim=-1, keepdim=True)).exp()
        elif self.cell == "argmax":
            alpha = torch.zeros_like(log_gate)
            alpha.scatter_(1, log_gate.argmax(dim=-1, keepdim=True), 1.0)
        else:
            raise AssertionError(self.cell)
        return (alpha.unsqueeze(1) @ av).squeeze(1)                # [n, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, s, _ = x.shape
        q, k = self.wq(x), self.wk(x)
        a = bench._softmax_operator(q, k)
        z = x + a @ x
        if self.cell not in ("softmax", "glance"):
            z = z.clone()
            z[:, s - 1] = x[:, s - 1] + self._alpha(q, k, x).to(z.dtype)
        h = self.mlp(z)
        return self.readout(h).squeeze(-1)[:, s - 1]


# ------------------------------------------------- the bind that gates the run
DIAG_SHAPES = ((4, 64, 24, 8, 0), (3, 64, 24, 16, 1), (2, 128, 16, 32, 2))


def _bind_batched_against_arm_s(shapes=DIAG_SHAPES, verbose: bool = True,
                                batch_fn=None) -> bool:
    """BITWISE bind of the batched map against `scale/arm_s.py`, per example.

    Runs before any cell. If it fails, the run does not start: a batched kernel
    that is merely close to the per-example one is a new arm, not an
    optimisation, and standing policy is that a faster path which moves a number
    is a new arm.

    TWO SEPARATE THINGS ARE CHECKED, AND ONLY ONE OF THEM IS AN EQUALITY.

    1. THE MAP. Given the SAME pivot set, `log_gram`, `log_gate`, `av` and 21
       steps of `log_alpha` must be bitwise equal to `arm_s`'s. This is a hard
       gate.

    2. THE PIVOT RULE, WHICH IS A DECLARED DEVIATION AND NOT AN EQUALITY.
       `arm_s.pivots_of` (line 97) selects `k` pivots excluding the query row and
       THEN drops index 0, whose row is all-zero under a strictly causal
       operator; on the examples where index 0 was selected it therefore returns
       `k - 1` pivots. A ragged pivot count cannot be batched, so
       `batched_pivots` excludes index 0 UP FRONT and backfills the freed slot
       with the next-ranked token, always returning exactly `k`. Both rules
       exclude the invalid row; they differ only in whether the slot is
       backfilled. The divergence RATE is measured here and printed, and it is
       carried into the report rather than described as equality.
    """
    #: The bind is arithmetic -- batched kernel against per-example kernel -- so
    #: it holds for any input. It is nonetheless taken on THE RUN'S OWN CORPUS,
    #: because a bind taken on a distribution the run never sees is a bind on a
    #: different tensor. Defaults to the shipped builder, so the negation_scope
    #: path is byte-identical to every run already logged.
    bfn = batch_fn or NS.make_batch
    ok = True
    div_n = div_tot = 0
    for (n, s, d, k_piv, seed) in shapes:
        x, _y, _f, _p = bfn(n, s, d, d_model=M3.D_MODEL, seed=seed)
        torch.manual_seed(seed)
        arm = M3.Arm("softmax", s=s)
        with torch.no_grad():
            q, kk = arm.wq(x), arm.wk(x)
        piv = batched_pivots(kk, k_piv)
        lg, lgate, av = batched_log_pivot_context(q, kk, x, piv, chunk=2)
        la = batched_settle_log(lgate, lg, BETA, 21)

        worst, bits = 0.0, True
        for i in range(n):
            div_tot += 1
            p_ref = A.pivots_of(kk[i], k_piv)
            if not torch.equal(p_ref.sort().values, piv[i].sort().values):
                div_n += 1
            # the MAP bind uses the batched pivot set on both sides, so it
            # isolates the arithmetic from the selection rule.
            g1, gate1, av1 = A.log_pivot_context(q[i], kk[i], x[i], piv[i])
            la1 = gate1 - torch.logsumexp(gate1, dim=-1)
            for _ in range(21):
                la1 = A.log_alpha_step(la1, gate1, g1, BETA)
            for nm_, a_, b_ in (("log_gram", lg[i], g1),
                                ("log_gate", lgate[i], gate1),
                                ("av", av[i], av1),
                                ("log_alpha", la[i], la1)):
                if not torch.equal(a_, b_):
                    bits = False
                    worst = max(worst, float((a_ - b_).abs().max()))
                    if verbose:
                        print(f"    {nm_} differs at i={i}: "
                              f"max abs {float((a_ - b_).abs().max()):.6e}")
        if verbose:
            print(f"  map bind n={n} s={s} d={d} k={k_piv} seed={seed}: "
                  + ("BITWISE" if bits else
                     f"NOT BITWISE, max abs diff = {worst:.6e}"))
        ok = ok and bits
    if verbose:
        print(f"  pivot-rule divergence (DECLARED DEVIATION, not a failure): "
              f"{div_n}/{div_tot} examples where arm_s.pivots_of drops index 0 "
              f"and batched_pivots backfills it")
    return ok


def _g3_glance_is_softmax(verbose: bool = True, batch_fn=None) -> bool:
    """Cell 2 must be bitwise cell 1. This is the G3 bind, taken at the run."""
    x, _y, _f, _p = (batch_fn or NS.make_batch)(8, 64, 24, d_model=M3.D_MODEL,
                                                seed=0)
    torch.manual_seed(0)
    a1 = QuintArm("softmax", 64, cell="softmax")
    torch.manual_seed(0)
    a2 = QuintArm("softmax", 64, cell="glance")
    with torch.no_grad():
        same = torch.equal(a1(x), a2(x))
    if verbose:
        print(f"  G3 glance == softmax, bitwise: {'YES' if same else 'NO'}")
    return same


# ------------------------------------------------------------------ the units
def weights_path(key: str) -> pathlib.Path:
    """Where one unit's trained tensors land. ONE definition, shared by the
    runner, its test and Foreman's consequence-fidelity column -- two copies of
    a path is the same defect as two copies of a pass condition."""
    return WEIGHTS_DIR / (key + ".pt")


def load_unit(path):
    """Rebuild the trained arm from a saved unit. Returns `(model, record)`.

    Lives here rather than in the caller because the constructor arguments that
    make a `QuintArm` the cell it is -- `cell`, `k_piv`, `beta`, `t_max`,
    `n_neumann` -- are this file's, and a consumer that re-derives them derives
    them wrong exactly once and then reports a `twin` number under `settled`.
    """
    rec = torch.load(path, weights_only=True)
    m = QuintArm(rec["kind"], rec["s"], cell=rec["cell"], k_piv=rec["k_piv"],
                 beta=rec["beta"], t_max=rec["t_max"],
                 n_neumann=rec["n_neumann"])
    m.load_state_dict(rec["state_dict"])
    m.eval()
    return m, rec


def _unit(p: dict) -> dict:
    """One cell at one seed. Training is `paired_arm.train_and_predict`'s loop,
    which is `run_arm`'s loop verbatim; only the instantiated class differs.

    `p["task"]` selects the corpus from `negation_scope.M3_TASKS` and MUST reach
    all four batches -- the RED 0-step pair built here and the training pair
    built inside `train_and_predict` -- or the 0-step gate is taken on one task
    while the trained number is taken on another.
    """
    s, d, seed, cell = p["s"], p["d"], p["seed"], p["cell"]
    task = p.get("task", SHIPPED_TASK)
    bfn = NS.M3_TASKS[task][0]

    #: The trained module, captured by identity. `train_and_predict` builds the
    #: model internally and returns predictions, not the model; the object it
    #: trains IS the object this list holds, so no copy and no re-derivation is
    #: involved. `run_arm` builds one first (the 0-step RED reading), so the
    #: TRAINED one is the last, and the count is asserted rather than assumed.
    built = []

    class _A(QuintArm):
        def __init__(self, kind, s_):
            super().__init__(kind, s_, cell=cell, k_piv=p["k"], beta=BETA,
                             t_max=p["t_max"], n_neumann=p["n_neumann"])
            built.append(self)

    xt, yt, _a, _b = bfn(p["n_train"], s, d, d_model=M3.D_MODEL, seed=seed)
    xe, ye, _c, _e = bfn(p["n_eval"], s, d, d_model=M3.D_MODEL,
                         seed=seed + 12345)
    old_m3, old_pa = M3.Arm, PA.Arm
    M3.Arm = PA.Arm = _A
    t0 = time.time()
    try:
        red = M3.run_arm("softmax", xt, yt, xe, ye, s=s, steps=0, seed=seed)
        pred, y_eval, npar = PA.train_and_predict(
            "softmax", s=s, d=d, steps=p["steps"], n_train=p["n_train"],
            n_eval=p["n_eval"], seed=seed, batch_fn=bfn)
    finally:
        M3.Arm, PA.Arm = old_m3, old_pa
    lo, hi = NS.bootstrap_ci(pred, y_eval, seed=seed)
    del t0
    # NO TIMING IN THE JOURNALLED VALUE. `run_bucket`'s resume audit compares
    # the whole value dict for a bitwise match, so a wall-clock field makes
    # every unit non-deterministic BY CONSTRUCTION and fires the audit on a
    # number that is not a measurement. It fired here, on this exact defect:
    # `provisional_seconds` read 110.89 then 168.36 while every NRMSE and CI
    # field was bit-identical (eval_nrmse 0.8771677350487059 both times).
    # `run_bucket` already journals its own {"seconds": ...} into META, which
    # the audit does not compare, so the field was redundant as well as wrong.
    val = dict(eval_nrmse=NS.nrmse(pred, y_eval), n_params=npar,
               nrmse0_train=red["nrmse0_train"], nrmse0_eval=red["nrmse0_eval"],
               marg_lo=lo, marg_hi=hi)

    # THE TRAINED TENSORS, BESIDE THE METRIC. scale/capability_table.py:232
    # records the exact cost of not doing this, in the shipped artifact:
    # "scale/m3_quintuple.py journals metrics only and saves no per-cell
    # weights, so no trained settled/twin/argmax/glance weights exist to
    # intervene on", which is why the consequence-fidelity column (1.7c) reads
    # NOT MEASURED for every arm. The weights do NOT go in the journal value:
    # `run_bucket`'s resume audit compares the whole value dict bitwise, and a
    # tensor in there is the same defect that `provisional_seconds` already was.
    assert len(built) == 2, (
        f"expected the 0-step arm and the trained arm, got {len(built)}")
    mu = float(yt.mean())
    sigma = float(yt.std(unbiased=False)) or 1.0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=built[-1].state_dict(), key=_key(p), task=task,
                    cell=cell, kind="softmax", s=s, d=d, d_model=M3.D_MODEL,
                    k_piv=p["k"], beta=BETA, t_max=p["t_max"],
                    n_neumann=p["n_neumann"], steps=p["steps"],
                    n_train=p["n_train"], n_eval=p["n_eval"], seed=seed,
                    mu=mu, sigma=sigma, eval_nrmse=val["eval_nrmse"],
                    n_params=npar, torch_version=str(torch.__version__)),
               weights_path(_key(p)))
    return val


def _key(p):
    #: THE TASK SUFFIX IS APPENDED ONLY FOR A NON-SHIPPED TASK. Every key in
    #: results/m3_quintuple_v2.jsonl was written before this file had a task at
    #: all; suffixing them all would orphan 25 completed units AND make the
    #: resume audit compare a negation_scope number against an e3 one under the
    #: same key. Two corpora in one bucket is the collision `scale/etask_k5e.py`
    #: opened a whole second runner to avoid.
    task = p.get("task", SHIPPED_TASK)
    return (f"{p['cell']}_k{p['k']}_s{p['s']}_d{p['d']}_st{p['steps']}"
            f"_ntr{p['n_train']}_nev{p['n_eval']}_b{p['t_max']}_sd{p['seed']}"
            + ("" if task == SHIPPED_TASK else f"_task{task}"))


def _units(cells, ks, seeds, **cfg):
    out = []
    for cell in cells:
        for k in ([0] if cell in ("softmax", "glance") else ks):
            for sd in seeds:
                p = dict(cell=cell, k=k, seed=sd, **cfg)
                out.append((_key(p), p))
    return out


CONTRASTS = (("settled", "twin", "HEADLINE"),
             ("settled", "argmax", "HEADLINE CAVEAT"),
             ("settled", "softmax", ""),
             ("twin", "softmax", ""),
             ("argmax", "softmax", ""))


def _argparser() -> argparse.ArgumentParser:
    """The CLI, exposed so its contract can be tested by VALUE rather than by
    driving `main()` with `--help` -- argparse exits 0 on `--help` before it
    reports an unrecognised optional, so that route passes against a file with
    no flag at all."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", type=int, nargs="+", default=[8])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--t-max", type=int, default=21)
    #: WHICH TASK, ported from scale/m3_capability.py:246. Default is the
    #: shipped negation-scope corpus, so every number already in
    #: results/m3_quintuple_v2.jsonl is reproduced by the same command that
    #: produced it and keeps its journal key.
    ap.add_argument("--task", default=SHIPPED_TASK, choices=list(NS.M3_TASKS))
    return ap


def main() -> int:
    a = _argparser().parse_args()
    n_neu = A.neumann_for(BETA)
    bfn = NS.M3_TASKS[a.task][0]
    rigged = a.task in RIGGED_ORACLE_TASKS

    print(f"=== M3 QUINTUPLE  task={a.task} s={a.s} d={a.d} steps={a.steps} "
          f"n_train={a.n_train} n_eval={a.n_eval} ks={a.ks} seeds={a.seeds} "
          f"beta={BETA} t_max={a.t_max} N={n_neu} "
          f"threads={torch.get_num_threads()} torch {torch.__version__} ===")
    print("Pre-registration: M3_QUINTUPLE_PREREGISTERED_READING.md"
          if a.task == SHIPPED_TASK else
          "Pre-registration: E_LADDER_PREREGISTERED_READING.md")
    print("Cost: analytic FLOPs only. Every seconds figure below is "
          "PROVISIONAL, CONTENDED BOX, NOT A MEASUREMENT.")
    if a.task in NS.E_T_STAR:
        print(f"  E-TASK: t* = {NS.E_T_STAR[a.task](a.s)}  (difficulty dial, "
              f"LOOP_PROMPT.md 1.7). Arm hop budget here is 2.")
    if rigged:
        print(f"  RIG CAVEAT -- {a.task} is labelled by equilibrium_oracle, the "
              f"signed path sum the ceq resolvent itself computes.")
        print(f"  {RIG_NOTE}. ONLY `settled vs twin` is creditable on this "
              f"task; every other contrast below is printed VOID.")

    print("\n=== BINDS (the run does not start if either fails) ===")
    print("  REQUIRED -- the shapes this run actually uses:")
    req = tuple((2, a.s, a.d, k, sd) for k in a.ks for sd in (0, 1))
    if not _bind_batched_against_arm_s(req, batch_fn=bfn):
        print("ABORT: batched map is not bitwise scale/arm_s.py at the run's "
              "own shapes.")
        return 1
    print("  DIAGNOSTIC -- other shapes. Reported, NOT gating this run, and "
          "carried into the report as a limit on the follow-on k sweep:")
    _bind_batched_against_arm_s(DIAG_SHAPES, batch_fn=bfn)
    if not _g3_glance_is_softmax(batch_fn=bfn):
        print("ABORT: G3 broken, glance is not bitwise softmax.")
        return 1

    # THE PUBLISHED READING IS A negation_scope READING. Re-taking it on another
    # corpus compares two different tasks and aborts every legitimate run, so it
    # is SKIPPED rather than adapted -- and the skip is printed, because a
    # reproduction gate that quietly stops running is worse than one that never
    # existed.
    if a.task != SHIPPED_TASK:
        print(f"\n=== SOFTMAX FIRST: SKIPPED. PUBLISHED_SOFTMAX_8192 is a "
              f"{SHIPPED_TASK} reading and this run is {a.task}. ===")
        print("  No published reading exists for this task, so this run is "
              "NOT anchored to one. Stated as a limit, not omitted.")
    else:
        print("\n=== SOFTMAX FIRST: the published n_train=8192 reading, "
              "re-taken ===")
        P = PUBLISHED_SOFTMAX_8192
        xt, yt, _f, _p = NS.make_batch(P["n_train"], P["s"], P["d"],
                                       d_model=M3.D_MODEL, seed=P["seed"])
        xe, ye, _g, _h = NS.make_batch(P["n_eval"], P["s"], P["d"],
                                       d_model=M3.D_MODEL,
                                       seed=P["seed"] + 12345)
        got = M3.run_arm("softmax", xt, yt, xe, ye, s=P["s"], steps=P["steps"],
                         seed=P["seed"])
        ok = True
        for f in ("n_params", "nrmse0_train", "nrmse0_eval", "train_nrmse",
                  "eval_nrmse", "ci_lo", "ci_hi"):
            w, g = P[f], got[f]
            same = (w == g) if isinstance(w, int) else (f"{w:.6f}" == f"{g:.6f}")
            ok = ok and same
            ws = str(w) if isinstance(w, int) else format(w, ".6f")
            gs = str(g) if isinstance(w, int) else format(g, ".6f")
            print(f"  {f:>14} published={ws} re-taken={gs} "
                  f"{'MATCH' if same else 'DRIFT'}")
        if not ok:
            print("ABORT: published softmax reading did not reproduce.")
            return 1

    cfg = dict(s=a.s, d=a.d, steps=a.steps, n_train=a.n_train,
               n_eval=a.n_eval, t_max=a.t_max, n_neumann=n_neu, task=a.task)
    us = _units(a.cells, a.ks, a.seeds, **cfg)
    print(f"\n=== {len(us)} units, bucketed, budget {a.budget:.0f}s ===")
    acc = run_bucket(NAME, us, _unit, budget_s=a.budget)
    missing = [k for k, _ in us if k not in __import__(
        "scale.bucket", fromlist=["Journal"]).Journal(NAME).done()]
    if missing:
        print(f"\nPARTIAL: {len(us) - len(missing)}/{len(us)} units done, "
              f"{len(missing)} remaining. Re-run to continue. "
              f"NO VERDICT ON A PARTIAL SET.")
        print(json.dumps(acc))
        return 3

    vals = require_complete(NAME, us)
    by = {}
    for key, p in us:
        by.setdefault((p["cell"], p["k"]), {})[p["seed"]] = vals[key]

    print("\n=== PER-CELL, ALL FIVE SEEDS ===")
    print(f"{'cell':>10} {'k':>4} {'seeds 0..4 eval NRMSE':>56} "
          f"{'mean':>9} {'sd':>9} {'0-step min':>11} {'params':>7}")
    means = {}
    for (cell, k), rows in sorted(by.items()):
        ev = [rows[sd]["eval_nrmse"] for sd in a.seeds]
        m = sum(ev) / len(ev)
        sd_ = (sum((v - m) ** 2 for v in ev) / (len(ev) - 1)) ** 0.5
        means[(cell, k)] = m
        print(f"{cell:>10} {k:>4} {' '.join('%.6f' % v for v in ev):>56} "
              f"{m:>9.6f} {sd_:>9.6f} "
              f"{min(rows[sd]['nrmse0_eval'] for sd in a.seeds):>11.6f} "
              f"{rows[a.seeds[0]]['n_params']:>7}")
        if m >= 1.0:
            print(f"           ^ seed-mean at or above 1.0: this cell did not "
                  f"beat predict-the-mean and is credited with nothing.")

    print("\n=== CONTRASTS, paired bootstrap over seeds, B=10000 (G6) ===")
    print(f"{'arm':>10} {'vs':>10} {'k':>4} {'delta':>10} {'ci_lo':>10} "
          f"{'ci_hi':>10} {'verdict':>15}  note")
    out = {}
    for arm, ref, note in CONTRASTS:
        for k in a.ks:
            ka = (arm, k if arm not in ("softmax", "glance") else 0)
            kr = (ref, k if ref not in ("softmax", "glance") else 0)
            if ka not in by or kr not in by:
                continue
            c = contrast([by[kr][sd]["eval_nrmse"] for sd in a.seeds],
                         [by[ka][sd]["eval_nrmse"] for sd in a.seeds],
                         n_boot=10000, seed=0)
            # LOOP_PROMPT.md 1.7d: on a task labelled by the resolvent's own
            # object, credit flows through `settled - twin` and nothing else.
            # The other rows are printed -- suppressing them is how a reader
            # ends up assuming they were favourable -- but they are printed VOID
            # and their verdict is not journalled as a verdict.
            void = rigged and {arm, ref} != {"settled", "twin"}
            out[f"{arm}_vs_{ref}_k{k}"] = "VOID" if void else c["verdict"]
            print(f"{arm:>10} {ref:>10} {k:>4} {c['delta']:>+10.6f} "
                  f"{c['ci_lo']:>+10.6f} {c['ci_hi']:>+10.6f} "
                  f"{'VOID' if void else c['verdict']:>15}  "
                  f"{RIG_NOTE if void else note}")
            if c["verdict"] == "NO DIFFERENCE":
                print("             ^ pre-registered floor: with five seeds a "
                      "real gap below ~0.05 NRMSE reads NO DIFFERENCE whether "
                      "or not it is real.")
    print("\n" + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
