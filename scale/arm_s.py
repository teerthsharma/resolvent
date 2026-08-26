"""ARM S -- the settling attention arm, built to its birth gates.

WHAT THE ARM IS. Ordinary causal softmax attention produces, for query row i, a
distribution A[i] over the tokens before i, and the output row A[i] @ V. That is
the GLANCE: one pass, no fixed point. ARM S replaces row i's reading with the
fixed point of a settling map over content-selected pivot readings, and leaves
every other row exactly as the glance produced it.

THE MAP, AND WHY IT LIVES IN k DIMENSIONS RATHER THAN s.

Round 6 iteration 0 fixed T on the simplex over the token support S:

    w_p(m) = gate_p * <a_p, m> ** beta,   T(m) = sum_p w_p(m) a_p / sum_p w_p(m)

with a_p the pivot rows, gate_p row i's own softmax weight on pivot p, and beta
the only free parameter. Every iterate of T is a positive combination of the k
fixed vectors a_p, so the whole orbit is determined by its WEIGHT vector alpha in
the k-simplex. Substituting m = sum_q alpha_q a_q gives <a_p, m> = (G alpha)_p
with G = A_P A_P^T the pivot Gram matrix, and the iteration closes in k
dimensions with no reference to s at all:

    alpha  <-  normalise( gate * (G alpha) ** beta )

This is the same map, not an approximation of it: the two orbits agree exactly
once the first step has been taken, and that identity is bound as a control
rather than asserted. Three consequences, all of them the point:

  * COST. Each settling step is one k-by-k matrix-vector product. G and A_P @ V
    are each formed ONCE, at the same order as the glance itself, and the loop
    never touches an s-sized object again. Settling adds O(t* k^2) to an
    O(s^2 d) glance.
  * SCALE-FREE BY CONSTRUCTION. The object the statistics live on has size k,
    which does not grow with s -- constitution ideal 3, obtained from the
    algebra rather than from a tuning schedule.
  * THE CERTIFICATE MOVES WITH IT. The contraction argument factors exactly as
    before, now on the k-simplex: alpha |-> G alpha is linear and positive, hence
    non-expansive in the Hilbert metric (Lemmens-Nussbaum arXiv:1304.7921 Thm 2.9
    with kappa <= 1); x |-> x**beta is order-preserving and homogeneous of degree
    beta, giving d_H(x**b, y**b) <= b d_H(x, y) (ibid. Prop 2.6 with r = beta);
    multiplication by the fixed positive gate is a positive diagonal scaling and
    cancels in the coordinate ratios, so it is a projective isometry. Composing,
    kappa(T) <= beta, exactly, independent of s, of the logit scale, and of the
    pivot readings.

    Birkhoff's tanh(Delta/4) is NOT used and is not needed. At this geometry it
    reads exactly 1.0 in 30 of 30 measured cells (see scale/foreman_hilbert.py),
    which is true and useless. The half of the theory that carries weight here is
    non-expansiveness plus the degree of homogeneity.

POSITIVITY, STRUCTURALLY. G_pq = <a_p, a_q> >= a_p[0] * a_q[0] > 0, because
coordinate 0 lies on the causal support of every non-empty row, so any two pivot
rows overlap there. gate_p > 0 because p < i puts p on row i's support. Hence
alpha stays strictly positive for every iterate and every beta >= 0, with no
appeal to sampling. The one thing this argument does NOT cover is float
underflow -- a_p[0] can reach exactly 0 in float32 -- so gram_audit measures it
instead of assuming it.

WHAT IS NOT SETTLED HERE. Every projection in this file is random-init. No
sentence in it is evidence about a trained model.

THREADS ARE PINNED IN THIS FILE. Other agents share the box, so wall-clock
figures are reported as PROVISIONAL and are paired with analytic FLOP counts,
which contention cannot move.
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
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale import hilbert as hb                                  # noqa: E402
from scale.pivot_probe import clopper_pearson, select_pivots     # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "arm_s.jsonl"


# ------------------------------------------------------------------ the arm

def draw(s: int, d: int, seed: int, dtype: torch.dtype = torch.float32):
    """One ARM A geometry draw: x0 ~ N(0,1), random projections, then q, k, v."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    v = torch.randn(s, d, generator=g)
    return (x0 @ wq).to(dtype), (x0 @ wk).to(dtype), v.to(dtype)


def pivots_of(kk: torch.Tensor, k_piv: int) -> torch.Tensor:
    """Content-selected pivots, with the two structurally invalid ones removed.

    Row 0 of a strictly-causal operator is entirely zero -- softmax over an
    all-masked row, then masked_fill(., 0.0) -- so pivot 0 has empty support, no
    part of the cone, and no normalisation. The query row itself is excluded
    because a pivot reading of the row being settled is not an independent
    reading of it.
    """
    s = kk.shape[0]
    piv = select_pivots(kk, min(k_piv, s - 2), exclude=(s - 1,))
    return piv[piv > 0]


def pivot_context(a: torch.Tensor, v: torch.Tensor, piv: torch.Tensor):
    """Everything the settling loop needs, formed ONCE: (G, gate, AV).

    AV is A_P @ V, a k-by-d matrix. The settled output is alpha @ AV, so the
    s-sized readings are contracted away before the loop starts and are never
    revisited -- that is where the cost argument comes from.
    """
    rows = a[piv].double()
    rows = rows / rows.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    gate = a[a.shape[0] - 1, piv].double()
    return rows @ rows.transpose(0, 1), gate, rows @ v.double()


def log_pivot_context(q: torch.Tensor, kk: torch.Tensor, v: torch.Tensor,
                      piv: torch.Tensor, *, chunk: int = 8):
    """(log_gram, log_gate, AV), built in the LOG domain and never from
    underflowed probabilities.

    WHY THIS EXISTS, measured not assumed. Built in the probability domain the
    arm settles to a VERTEX of the simplex: at the ARM A geometry the query
    row's softmax over its pivots is already one-hot to within float (measured
    gate max 1.000000 / 0.999997 / 0.999709 / 0.999838 at seeds 0-3, s=256,
    k=8), and the true interior fixed point has coordinates around e^-1000,
    which is not representable. float64 rounds it to the nearest vertex, the
    residual reads exactly 0.0 after three steps, and the arm reports a
    converged equilibrium that is really one pivot row copied out.

    The fixed point is not missing -- kappa <= beta < 1 on the open k-simplex,
    which is complete under d_H, so Banach gives existence and uniqueness. It is
    the ARITHMETIC that cannot hold it, the same failure class as reading
    tanh(Delta/4) at Delta = 100. In logs, e^-1000 is -1000.0 and is exact.

    log G_pq = logsumexp_j (log a_p[j] + log a_q[j]) is formed from the
    log-softmax of the logits directly. The k-by-k-by-s intermediate is chunked
    over p rather than materialised.
    """
    s, d = q.shape[0], q.shape[-1]
    _, nm = bench._causal_mask_pair(s, 0, str(q.device))
    # Only k + 1 query rows are ever read, so only those rows of the logit
    # matrix are formed: O(k s d) instead of the O(s^2 d) a full recompute would
    # cost. At s = 1024, k = 32 that is a 31x reduction in the setup term, and
    # it is the difference between the settling arm being free and being a
    # second attention pass.
    want = torch.cat([piv, torch.tensor([s - 1])])
    w = (q[want].double() @ kk.double().transpose(-2, -1)) / math.sqrt(d)
    w = w.masked_fill(nm[want], float("-inf"))
    la_piv = torch.log_softmax(w[:-1], dim=-1)                   # [k, s]
    log_gate = torch.log_softmax(w[-1], dim=-1)[piv]             # [k]
    k = int(piv.numel())
    # ponytail: the Gram matrix costs 2 k^2 s to build and 2 k^2 per step. The
    # alternative never forms it -- (G alpha)_p = <a_p, sum_q alpha_q a_q>, so a
    # step is two O(k s) matvecs and setup drops the k^2 s term entirely. Forming
    # G is cheaper while k + k t*/s < 2 t*, i.e. k < 84 at the measured t* = 44,
    # s = 1024. It is kept because the contract's k sweep is {8, 16, 32}, the
    # uniqueness-safe regime; k = 128 is where this choice costs and it is priced
    # out under K-F rather than optimised for. Upgrade path if k = 128 is ever
    # wanted: the two-matvec route reads 1.47x against the 1.67x measured here.
    log_gram = torch.empty(k, k, dtype=torch.float64)
    for lo in range(0, k, chunk):
        hi = min(lo + chunk, k)
        log_gram[lo:hi] = torch.logsumexp(
            la_piv[lo:hi].unsqueeze(1) + la_piv.unsqueeze(0), dim=-1)
    return log_gram, log_gate, la_piv.exp() @ v.double()


def log_alpha_step(log_alpha: torch.Tensor, log_gate: torch.Tensor,
                   log_gram: torch.Tensor, beta: float) -> torch.Tensor:
    """One application of T, in logs. Exactly the same map as alpha_step."""
    lw = log_gate + beta * torch.logsumexp(log_gram + log_alpha.unsqueeze(0), dim=-1)
    return lw - torch.logsumexp(lw, dim=0)


def settle_log(log_gate: torch.Tensor, log_gram: torch.Tensor, beta: float, *,
               tol: float, max_steps: int):
    """The shipped settle. Residual is d_H, read off the logs by osc(u - v).

    left_cone can still be reported, but in this representation it means a
    genuine -inf or nan rather than an underflow, which is the distinction the
    probability-domain version could not make.
    """
    log_alpha = log_gate - torch.logsumexp(log_gate, dim=0)
    res, left = [], False
    for t in range(1, max_steps + 1):
        nxt = log_alpha_step(log_alpha, log_gate, log_gram, beta)
        r = hb.d_H_logits(nxt, log_alpha)
        if not math.isfinite(r):
            return nxt, res, t, False, True
        res.append(r)
        log_alpha = nxt
        if r < tol:
            return log_alpha, res, t, True, left
    return log_alpha, res, max_steps, False, left


def alpha_step(alpha: torch.Tensor, gate: torch.Tensor, gram: torch.Tensor,
               beta: float) -> torch.Tensor:
    """One application of T, in the k-simplex."""
    w = gate * (gram @ alpha).pow(beta)
    return w / w.sum()


def settle_alpha(gate: torch.Tensor, gram: torch.Tensor, beta: float, *,
                 tol: float, max_steps: int):
    """Iterate to the fixed point, journalling the Hilbert residual per step.

    The residual is d_H(alpha_{t+1}, alpha_t), not an L2 norm: the certificate is
    a statement about d_H, so a run journalled in any other metric could not be
    compared against it. left_cone is reported separately from converged because
    "it stopped" and "it stayed positive" are different facts.
    """
    alpha = gate / gate.sum()                 # start at the glance's own opinion
    res, left = [], False
    for t in range(1, max_steps + 1):
        nxt = alpha_step(alpha, gate, gram, beta)
        r = hb.d_H(nxt, alpha)
        if not math.isfinite(r):
            left = True
            return nxt, res, t, False, left
        res.append(r)
        alpha = nxt
        if r < tol:
            return alpha, res, t, True, left
    return alpha, res, max_steps, False, left


def arm_s(q: torch.Tensor, kk: torch.Tensor, v: torch.Tensor, *, k_piv: int,
          beta: float, t_max: int, tol: float = 1e-12):
    """ARM S forward. t_max = 0 disables settling and returns the glance path.

    G3 BIND: at t_max = 0 the returned tensor is BITWISE identical to
    glance(q, kk, v). It is the same expression, evaluated once, with no
    renormalisation and no reassociation -- see g3_bind for the RED showing that
    a mathematically equivalent rewrite fails exactly this.
    """
    a = bench._softmax_operator(q, kk)
    out = a @ v
    if t_max == 0:
        return out, None
    piv = pivots_of(kk, k_piv)
    log_gram, log_gate, av = log_pivot_context(q, kk, v, piv)
    log_alpha, res, steps, conv, left = settle_log(log_gate, log_gram, beta,
                                                   tol=tol, max_steps=t_max)
    out = out.clone()
    out[-1] = (log_alpha.exp() @ av).to(out.dtype)
    return out, dict(residuals=res, steps=steps, converged=conv, left_cone=left,
                     n_pivots=int(piv.numel()), log_alpha=log_alpha)


def glance(q: torch.Tensor, kk: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The baseline path, written once so the G3 bind compares like with like."""
    return bench._softmax_operator(q, kk) @ v


# --------------------------------------------------- the implicit gradient

def _step_from_logs(alpha, log_gate, log_gram, beta):
    w = log_gate.exp() * (log_gram.exp() @ alpha).pow(beta)
    return w / w.sum()


class Settled(torch.autograd.Function):
    """The fixed point, differentiated implicitly rather than through the loop.

    Contract 1.2: d alpha*/d phi = (I - J)^-1 dT/dphi at the fixed point, with
    (I - J)^-1 truncated to a Neumann sum of N terms and error <= kappa^N/(1-kappa).
    Because kappa = beta by construction, N is known BEFORE the run rather than
    fitted after it, and it is passed in explicitly so that a deliberately short
    N can be SHOWN to break the gradient rather than assumed to.

    The backward never forms J. Each Neumann term is one vector-Jacobian product
    through a single re-evaluation of the step map at alpha*, which is the whole
    reason the implicit form is cheaper than unrolling the solve.
    """

    @staticmethod
    def forward(ctx, log_gate, log_gram, beta, tol, max_steps, n_neumann):
        with torch.no_grad():
            alpha = log_gate.exp()
            alpha = alpha / alpha.sum()
            for _ in range(max_steps):
                nxt = _step_from_logs(alpha, log_gate, log_gram, beta)
                if float((nxt - alpha).abs().max()) < tol:
                    alpha = nxt
                    break
                alpha = nxt
        ctx.save_for_backward(log_gate, log_gram, alpha)
        ctx.beta, ctx.n_neumann = beta, n_neumann
        return alpha

    @staticmethod
    def backward(ctx, grad_out):
        log_gate, log_gram, alpha = ctx.saved_tensors
        lg = log_gate.detach().requires_grad_(True)
        lG = log_gram.detach().requires_grad_(True)
        a = alpha.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _step_from_logs(a, lg, lG, ctx.beta)
            y = grad_out.clone()
            acc = grad_out.clone()
            for _ in range(ctx.n_neumann - 1):
                y = torch.autograd.grad(out, a, y, retain_graph=True)[0]
                acc = acc + y
            gg, gG = torch.autograd.grad(out, (lg, lG), acc, retain_graph=False)
        return gg, gG, None, None, None, None


def neumann_for(beta: float, tol: float = 1e-6) -> int:
    """N for kappa = beta, routed through the published Delta<->kappa inverse so
    it is the same number scale/hilbert.py would give for that kappa:
    kappa = tanh(Delta/4) inverts to Delta = 4 atanh(beta) = 2 log((1+b)/(1-b))."""
    if not 0.0 <= beta < 1.0:
        return -1
    if beta == 0.0:
        return 1
    return hb.neumann_terms(2.0 * math.log((1.0 + beta) / (1.0 - beta)), tol)


# ------------------------------------------------------------- birth gates

def g3_bind(ss, ds, ks, seeds):
    """G3: with settling disabled the arm is BITWISE the glance.

    Paired with the RED it exists to catch. `a` is a softmax operator, so its
    rows already sum to one and dividing by the row sum is a mathematical no-op
    -- but float32 row sums are not exactly 1.0, so the rewrite moves bits. A
    bind that only ever saw the correct path could not tell the two apart, so
    both are drawn over the same instances and both counts are reported.
    """
    ok = bad = dead = 0
    worst = 0.0
    for s in ss:
        for d in ds:
            for seed in seeds:
                q, kk, v = draw(s, d, seed)
                a = bench._softmax_operator(q, kk)
                g = a @ v
                out0, j = arm_s(q, kk, v, k_piv=ks[0], beta=0.5, t_max=0)
                ok += int(torch.equal(g, out0) and j is None)
                # The rewrite is evaluated on LIVE ROWS ONLY. Row 0 of a strictly
                # causal operator sums to exactly 0.0, so dividing by the row sum
                # makes it NaN -- which would make this control "fire" on a
                # division by zero rather than on the float claim it advertises.
                # A control that passes for the wrong reason is worse than one
                # that fails, so the dead row is counted separately.
                rs = a.sum(-1, keepdim=True)
                dead += int((rs == 0).sum())
                renorm = (a[1:] / rs[1:]) @ v
                diff = (g[1:] - renorm).abs().max()
                bad += int(not torch.equal(g[1:], renorm))
                worst = max(worst, float(diff) if torch.isfinite(diff) else 0.0)
    return ok, bad, worst, dead


def gate1_event(ss, ds, ks, seeds, beta, t_max, thresh):
    """Birth gate 1 -- settled != glance, through d_H, and settled != one step.

    TWO contrasts, because only the second one is about settling. d_H(settled,
    glance) can be large merely because a pivot row is not the query row. The
    contrast that decides whether the fixed point earns its name is
    d_H(settled, one step): if a single application already lands on the fixed
    point, the equilibrium clause is decoration and round 5 killed an arm for
    exactly that.
    """
    rows = []
    for s in ss:
        for d in ds:
            for k in ks:
                for seed in seeds:
                    q, kk, v = draw(s, d, seed)
                    piv = pivots_of(kk, k)
                    lG, lg, av = log_pivot_context(q, kk, v, piv)
                    la, res, steps, conv, left = settle_log(
                        lg, lG, beta, tol=1e-12, max_steps=t_max)
                    l0 = lg - torch.logsumexp(lg, dim=0)
                    l1 = log_alpha_step(l0, lg, lG, beta)
                    ratios = [res[i + 1] / res[i] for i in range(len(res) - 1)
                              if res[i] > 0.0]
                    rows.append(dict(
                        s=s, d=d, k=k, seed=seed, beta=beta, steps=steps,
                        converged=conv, left_cone=left, n_pivots=int(piv.numel()),
                        d_glance=hb.d_H_logits(la, l0),
                        d_onestep=hb.d_H_logits(la, l1),
                        res0=res[0] if res else float("nan"),
                        kappa_run=max(ratios) if ratios else float("nan")))
    n = len(rows)
    ng = sum(1 for r in rows if r["d_glance"] > thresh)
    no = sum(1 for r in rows if r["d_onestep"] > thresh)
    return rows, (ng, clopper_pearson(ng, n)), (no, clopper_pearson(no, n)), n


def gate2_gradcheck(k, beta, seed, n_override=None):
    """Birth gate 2 -- the implicit gradient against finite differences.

    `torch.autograd.gradcheck` re-solves the fixed point at every perturbed
    input, so what is being compared is the implicit formula against a finite
    difference of the TRUE fixed point, not against an unrolled solve.
    """
    g = torch.Generator().manual_seed(seed)
    lG = (torch.randn(k, k, generator=g, dtype=torch.float64) - 2.0)
    lG = (lG + lG.transpose(0, 1)) / 2.0
    lg = torch.randn(k, generator=g, dtype=torch.float64)
    lG.requires_grad_(True)
    lg.requires_grad_(True)
    n = neumann_for(beta) if n_override is None else n_override
    fn = lambda a, b: Settled.apply(a, b, beta, 1e-14, 20000, n)   # noqa: E731
    try:
        ok = torch.autograd.gradcheck(fn, (lg, lG), eps=1e-6, atol=1e-8,
                                      rtol=1e-4, nondet_tol=0.0)
        return bool(ok), n, ""
    except Exception as exc:                       # gradcheck raises on failure
        return False, n, str(exc).splitlines()[0][:160]


def gate3_cost(s, d, k, seed, beta, repeats):
    """Birth gate 3 -- cost decomposed, with FLOPs beside the clock.

    The clock is PROVISIONAL: other agents share this box and a contended timing
    is not a measurement. The FLOP counts are arithmetic on shapes and cannot be
    moved by contention, so K-F is argued on those and the clock is shown only
    as corroboration.
    """
    q, kk, v = draw(s, d, seed)
    piv = pivots_of(kk, k)
    kp = int(piv.numel())
    lG, lg, av = log_pivot_context(q, kk, v, piv)
    la, res, steps, conv, left = settle_log(lg, lG, beta, tol=1e-12,
                                            max_steps=2000)
    n = neumann_for(beta)

    def clock(fn):
        fn()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats

    t_glance = clock(lambda: glance(q, kk, v))
    l0 = lg - torch.logsumexp(lg, dim=0)
    t_step = clock(lambda: log_alpha_step(l0, lg, lG, beta))
    t_setup = clock(lambda: log_pivot_context(q, kk, v, piv))
    a0 = la.detach().requires_grad_(True)
    with torch.enable_grad():
        o = _step_from_logs(a0, lg, lG, beta)
        gv = torch.ones_like(a0)
        t_jvp = clock(lambda: torch.autograd.grad(o, a0, gv, retain_graph=True))

    f_glance = 2.0 * s * s * d + 2.0 * s * s * d          # QK^T and A@V
    f_setup = 2.0 * (kp + 1) * s * d + 2.0 * kp * kp * s + 2.0 * kp * s * d
    f_step = 2.0 * kp * kp
    f_jvp = 4.0 * kp * kp
    return dict(
        s=s, d=d, k=k, n_pivots=kp, seed=seed, beta=beta, steps=steps,
        neumann_N=n, repeats=repeats,
        flops_glance=f_glance, flops_setup=f_setup,
        flops_settle=f_step * steps, flops_backward=f_jvp * n,
        flop_ratio=(f_setup + f_step * steps + f_jvp * n) / f_glance,
        prov_t_glance=t_glance, prov_t_setup=t_setup,
        prov_t_settle=t_step * steps, prov_t_backward=t_jvp * n,
        prov_ratio=(t_setup + t_step * steps + t_jvp * n) / t_glance)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--t-max", dest="t_max", type=int, default=2000)
    ap.add_argument("--thresh", type=float, default=1e-6)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    th = torch.get_num_threads()

    print(f"=== G3 BIND -- hops=0 bitwise identity. threads={th} ===")
    ok, bad, worst, dead = g3_bind(a.ss, [a.d], a.ks, a.seeds)
    n_inst = len(a.ss) * len(a.seeds)
    print(f"  t_max=0 BITWISE identical to glance:      {ok}/{n_inst} instances")
    print(f"  RED control, LIVE ROWS ONLY (row-renormalised rewrite, a no-op")
    print(f"    mathematically since softmax rows already sum to one):")
    print(f"    moved bits in:                          {bad}/{n_inst} instances"
          f"   worst |diff| = {worst:.6e}")
    print(f"    dead rows (row sum exactly 0.0, would NaN under the rewrite): "
          f"{dead} across the {n_inst} instances")
    if ok != n_inst or bad == 0:
        print("  G3 FAILED, or its control could not fire. Nothing below counts.")
        return 1

    print(f"\n=== BIRTH GATE 1 -- settled != glance, and settled != one step ===")
    print(f"  beta={a.beta} t_max={a.t_max} tol=1e-12 thresh={a.thresh} nats")
    rows, (ng, cg), (no, co), n = gate1_event(
        a.ss, [a.d], a.ks, a.seeds, a.beta, a.t_max, a.thresh)
    print(f"  d_H(settled, glance)  > {a.thresh}: {ng}/{n} = {ng / n:.6f}"
          f"  95% CI [{cg[0]:.6f}, {cg[1]:.6f}]")
    print(f"  d_H(settled, ONE step) > {a.thresh}: {no}/{n} = {no / n:.6f}"
          f"  95% CI [{co[0]:.6f}, {co[1]:.6f}]   <- the contrast that decides it")
    conv = sum(1 for r in rows if r["converged"])
    left = sum(1 for r in rows if r["left_cone"])
    st = sorted(r["steps"] for r in rows)
    kr = [r["kappa_run"] for r in rows if r["kappa_run"] == r["kappa_run"]]
    print(f"  converged {conv}/{n}   left_cone {left}/{n}   "
          f"steps min {st[0]} median {st[len(st) // 2]} max {st[-1]}")
    print(f"  worst per-step ratio r_(t+1)/r_t over all runs: {max(kr):.6f}"
          f"   (certificate says <= beta = {a.beta})")
    print(f"  d_H(settled, one step) range: "
          f"{min(r['d_onestep'] for r in rows):.6e} .. "
          f"{max(r['d_onestep'] for r in rows):.6f} nats")

    print(f"\n=== BIRTH GATE 2 -- implicit gradient, float64, rtol 1e-4 ===")
    print(f"  {'beta':>6} {'N':>6} {'gradcheck':>11}  note")
    for b in (0.25, 0.5, 0.9):
        good, nn, msg = gate2_gradcheck(6, b, 3)
        print(f"  {b:>6} {nn:>6} {str(good):>11}  {msg}")
    print("  MUST-FIRE (the control that makes the pass mean something):")
    for b, nover in ((0.9, 3), (0.9, 10)):
        good, nn, msg = gate2_gradcheck(6, b, 3, n_override=nover)
        print(f"  {b:>6} {nover:>6} {str(good):>11}  truncated N -- {msg}")

    print(f"\n=== BIRTH GATE 3 -- cost decomposed. FLOPs exact, CLOCK PROVISIONAL ===")
    print("  Other agents share this box. The clock columns are corroboration,")
    print("  not measurement; K-F is argued on the FLOP ratio.")
    print(f"  {'s':>6} {'k':>5} {'steps':>6} {'N':>5} {'flop glance':>13} "
          f"{'flop setup':>12} {'flop settle':>12} {'flop bwd':>10} "
          f"{'FLOP ratio':>11} {'clock ratio':>12}")
    cost = []
    for s in a.ss:
        for k in a.ks:
            r = gate3_cost(s, a.d, k, 0, a.beta, a.repeats)
            cost.append(r)
            print(f"  {r['s']:>6} {r['k']:>5} {r['steps']:>6} {r['neumann_N']:>5} "
                  f"{r['flops_glance']:>13.4g} {r['flops_setup']:>12.4g} "
                  f"{r['flops_settle']:>12.4g} {r['flops_backward']:>10.4g} "
                  f"{1.0 + r['flop_ratio']:>11.6f} {1.0 + r['prov_ratio']:>12.4f}")
    worst_flop = max(1.0 + r["flop_ratio"] for r in cost)
    print(f"\n  K-F gate is 1.5x on the SUM. worst FLOP ratio {worst_flop:.6f} "
          f"-- {'PASS' if worst_flop <= 1.5 else 'FAIL'}")

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        for r in rows + cost:
            fh.write(json.dumps({k: v for k, v in r.items()}) + "\n")
    print(f"  journal: {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
