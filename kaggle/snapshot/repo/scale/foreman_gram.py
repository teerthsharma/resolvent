"""Round 6 iteration 22 -- binding Dr House's leap, or killing it.

THE LEAP, as handed over. Stop certifying T on the state cone; fold one full
iteration into pivot-weight space and certify the LINEAR piece that lives there:

    kappa(T)  <=  beta * tanh( Delta(G) / 4 ),      G_qp = <a_q, a_p>

WHY IT IS NOT MERELY TIGHTER, BUT A DIFFERENT KIND OF CLAIM. In w-space the map
is T_w(w) = normalise( gate * (G w) ** beta ). Normalisation is a constant shift
in every coordinate ratio and the positive diagonal gate cancels the same way,
so both drop out of an oscillation exactly:

    d_H(T_w w, T_w w') = osc_q [ beta * log( (Gw)_q / (Gw')_q ) ]
                       = beta * d_H(G w, G w')

Hence kappa(T_w) = beta * kappa(G), an IDENTITY, not a bound. The only inequality
left is Birkhoff on G -- and G is a k-by-k entrywise-positive LINEAR map, so
Lemmens-Nussbaum arXiv:1304.7921 Thm 2.9 applies verbatim, the linearity
objection that killed the same move on T not applying here.

That theorem is stated as an EQUALITY, kappa(L) = tanh(Delta(L)/4). So the leap
does not predict an inequality that may be satisfied loosely; it predicts a
NUMBER. Sampling can only ever attain a lower bound on a supremum, so the test
has two halves and both are reported:
  * DOES THE BOUND HOLD:  attained <= beta * tanh(Delta(G)/4). A violation
    refutes the factorisation outright.
  * IS IT TIGHT:          attained / beta  vs  tanh(Delta(G)/4).

THE OBJECT TRAP, AND IT CUTS BOTH WAYS. The residual factor 0.961793 that
motivated the leap was read off the beta sweep in scale/foreman_hilbert.py,
whose kappa_emp_max is a supremum over sampled pairs of the STATE-cone map
T_state(m) = R^T( gate * (R m) ** beta ), with R the matrix of pivot readings.
That is NOT the w-space map, and its factorisation carries TWO Birkhoff factors
rather than one:

    kappa(T_state)  <=  beta * tanh(Delta(R)/4) * tanh(Delta(R^T)/4)

which is the shape of Proposition 4 in arXiv:2605.08123. Comparing an attained
state-cone ratio against a w-space tanh(Delta(G)/4) would be the
correct-statement-wrong-object error this project has now made nine times. So
BOTH maps are measured here, at the same geometry, and reported separately.

GEOMETRY IS PINNED TO THE SWEEP THAT PRODUCED 0.961793, and to nothing else:
s=256, d=16, k in {8, 32}, seeds {0, 1, 2}, which is
scale/foreman_hilbert.py --ss 256 --ks 8 32 --seeds 0 1 2 --per-conc 3.
Trained projections are a separate question and are not touched here.

Delta(G) is the EXTREME-RAY diameter -- the largest Hilbert distance between two
COLUMNS of G -- via scale/settle.py::column_diameter. Not a mean over cells and
not a sampled maximum: Birkhoff's hypothesis is a supremum, and a mean of
tanh(Delta_i/4) is a different and smaller quantity.

THREADS PINNED HERE. No wall-clock number is produced or reported by this file.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale import foreman_hilbert as fh                          # noqa: E402
from scale import hilbert as hb                                  # noqa: E402
from scale import settle as st                                   # noqa: E402
from scale import trained_projections as tp                      # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "gram.jsonl"


def cell(s: int, d: int, k: int, seed: int):
    """Rebuild ONE cell of the beta sweep, byte-for-byte as it was measured.

    Every line mirrors foreman_hilbert.delta_audit: same generator order, same
    pivot selection, same restriction to the union support, same renormalisation,
    and it reuses that module's own helper rather than restating it. If this
    drifts, the comparison is against a different object and is worthless.
    """
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    piv = select_pivots(kk, min(k, s - 2), exclude=(i,))
    piv = piv[piv > 0]
    a = bench._softmax_operator(q, kk).double()
    rows, p_max = fh.pivot_readings(a, piv)       # [k, |S|], renormalised
    gate = a[i, piv].clone()
    gate = gate / gate.sum()
    return rows, gate, p_max, piv


def gram_facts(rows: torch.Tensor):
    """Delta(G) by extreme rays, plus the support-overlap facts that decide it.

    The certificate is finite IFF G is entrywise positive, that is, iff every
    pair of pivots overlaps somewhere. effective_support is exp of the Shannon
    entropy of a row -- the number of coordinates the row genuinely occupies,
    which is the quantity that moves when the weights move.
    """
    gram = rows @ rows.transpose(0, 1)
    k = gram.shape[0]
    ent = -(rows.clamp_min(1e-300).log() * rows).sum(-1)
    overlap = ((rows > 0).double() @ (rows > 0).double().transpose(0, 1))
    off = ~torch.eye(k, dtype=torch.bool)
    return dict(
        k=k,
        gram_zeros=int((gram == 0).sum()),
        gram_min=float(gram.min()), gram_max=float(gram.max()),
        delta_G=st.column_diameter(gram),
        eff_support_mean=float(ent.exp().mean()),
        eff_support_min=float(ent.exp().min()),
        overlap_min=float(overlap[off].min()) if k > 1 else float("nan"),
        overlap_mean=float(overlap[off].mean()) if k > 1 else float("nan"),
    )


def attained_w(rows, gate, beta, seed, per_conc=6):
    """Supremum over sampled pairs of d_H(T_w w, T_w w') / d_H(w, w'), in w-space.

    Drawn adversarially: Dirichlet concentrations from 0.05 (nearly a vertex) to
    5.0, plus the k unit vectors themselves, because a supremum estimated from
    interior draws alone is biased low and this whole test is about a supremum.
    """
    gram = rows @ rows.transpose(0, 1)
    k = gram.shape[0]
    gg = torch.Generator().manual_seed(seed + 777)
    ws = [torch.full((k,), 1.0 / k, dtype=torch.float64)]
    for j in range(k):
        e = torch.full((k,), 1e-12, dtype=torch.float64)
        e[j] = 1.0
        ws.append(e / e.sum())
    for conc in (0.05, 0.3, 1.0, 5.0):
        for _ in range(per_conc):
            x = -torch.rand(k, generator=gg).double().clamp_min(1e-300).log()
            x = x.pow(1.0 / conc)
            ws.append(x / x.sum())

    def step(w):
        y = gate * (gram @ w).pow(beta)
        return y / y.sum()

    ts = [step(w) for w in ws]
    best = 0.0
    for x in range(len(ws)):
        for y in range(x + 1, len(ws)):
            din = hb.d_H(ws[x], ws[y])
            dout = hb.d_H(ts[x], ts[y])
            if math.isfinite(din) and din > 1e-9 and math.isfinite(dout):
                best = max(best, dout / din)
    return best


def must_fire():
    """Disjoint-support pivots must give Delta(G) = +inf and no credit past beta.

    Drawn, not hand-built, and counted -- a single hand-picked instance is where
    a control goes vacuous, because the smallest case is usually the one where
    the right and the wrong answer coincide. Eight independent draws each way,
    and the counts are reported.
    """
    out = []
    for seed in range(8):
        g = torch.Generator().manual_seed(1000 + seed)
        k, n = 4, 16
        rows = torch.zeros(k, k * n, dtype=torch.float64)
        for p in range(k):                       # block-disjoint supports
            blk = torch.rand(n, generator=g).double() + 0.1
            rows[p, p * n:(p + 1) * n] = blk / blk.sum()
        f = gram_facts(rows)
        out.append((f["gram_zeros"], f["delta_G"], f["overlap_min"]))
    fired = sum(1 for z, dg, _ in out if z > 0 and dg == float("inf"))
    # The other direction: OVERLAPPING pivots must give a FINITE Delta(G), or
    # the control is only detecting that the probe returns inf for everything.
    ov = []
    for seed in range(8):
        g = torch.Generator().manual_seed(2000 + seed)
        rows = torch.rand(4, 64, generator=g).double() + 0.05
        rows = rows / rows.sum(-1, keepdim=True)
        ov.append(gram_facts(rows)["delta_G"])
    finite = sum(1 for x in ov if math.isfinite(x))
    return fired, len(out), finite, len(ov), out[0][1], max(ov)


# ------------------------------------------- the log-domain Gram, and trained

def log_gram(la: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """log G_pq = logsumexp_j ( log a_p[j] + log a_q[j] ), never via exp.

    WHY THIS IS NOT OPTIONAL. G_pq is an inner product of two softmax rows, so
    it inherits their dynamic range squared. At random-init the smallest Gram
    entry already reads 8.1603e-48 -- four decades from the float64 subnormal
    floor of 4.94e-324 in absolute terms, but the relevant comparison is the
    LOGIT scale, and G_min falls exponentially in it. Trained projections carry
    a logit scale of 0.454379 against the harness init's 0.00266492, so the
    underflow risk at trained is worse, not better. A Gram formed by exp() and
    then multiplied would return exact zeros, Delta(G) would read +inf, and the
    answer would be an arithmetic artifact wearing the clothes of a structural
    fact.

    In logs there is no floor to hit: an entry of e^-2000 is -2000.0 and exact.
    The k-by-k-by-s intermediate is chunked over p rather than materialised.
    """
    k = la.shape[0]
    out = torch.empty(k, k, dtype=torch.float64)
    for lo in range(0, k, chunk):
        hi = min(lo + chunk, k)
        out[lo:hi] = torch.logsumexp(la[lo:hi].unsqueeze(1) + la.unsqueeze(0),
                                     dim=-1)
    return out


def delta_G_log(lg: torch.Tensor) -> float:
    """Extreme-ray projective diameter of G, read off the LOG Gram.

    For a positive matrix Delta is the largest Hilbert distance between two
    COLUMNS, and d_H between two columns is the oscillation of their log
    difference -- which is what d_H_logits computes, exactly. This is the same
    quantity scale/settle.py::column_diameter returns for a representable G, and
    the two are bound against each other in `log_path_bind` below.

    Not a mean over cells and not a sampled maximum. A mean of tanh(Delta_i/4)
    and a sampled sup have each reversed a conclusion in this round already.
    """
    k = lg.shape[0]
    best = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            best = max(best, hb.d_H_logits(lg[:, i], lg[:, j]))
    return best


def log_rows(q: torch.Tensor, kk: torch.Tensor, piv: torch.Tensor, p_max: int):
    """log-softmax of the pivot rows, restricted to the union support [0, p_max).

    Restriction matches foreman_hilbert.pivot_readings so the trained reading is
    the same object as the random-init reading it is compared against. The
    renormalisation that function applies is omitted deliberately: scaling row p
    by d_p multiplies every column of G by d_p in the same coordinate, and a
    common factor per coordinate cancels in the log DIFFERENCE that d_H takes.
    Delta(G) is therefore invariant to it, which is checked rather than assumed
    in `log_path_bind`.
    """
    s, dk = q.shape[0], q.shape[-1]
    _, nm = bench._causal_mask_pair(s, 0, str(q.device))
    w = (q.double() @ kk.double().transpose(-2, -1)) / math.sqrt(dk)
    la = torch.log_softmax(w[piv].masked_fill(nm[piv], float("-inf")), dim=-1)
    return la[:, :p_max]


def log_path_bind(s: int = 256, d: int = 16, k: int = 8, seeds=(0, 1, 2, 3)):
    """The log path must reproduce the probability path where BOTH are valid,
    and must stay finite where only it is.

    Two directions, both drawn and counted. Without the first the log Gram could
    be computing anything; without the second there would be no evidence that it
    was needed.
    """
    agree, worst, worst32 = 0, 0.0, 0.0
    for seed in seeds:
        rows, gate, p_max, piv = cell(s, d, k, seed)
        q, kk = _qk(s, d, seed)
        la = log_rows(q, kk, piv, p_max)
        logd = delta_G_log(log_gram(la))
        # SAME dtype on both sides. The formula is what is under test here, so
        # the probability side is rebuilt from the identical float64 logits
        # rather than reusing cell()'s float32 operator.
        pr = la.exp()
        prob64 = st.column_diameter(pr @ pr.transpose(0, 1))
        if math.isfinite(prob64) and math.isfinite(logd):
            rel = abs(prob64 - logd) / max(prob64, 1e-300)
            worst = max(worst, rel)
            agree += int(rel < 1e-9)
        # Reported separately, and it is a defect in neither path: the
        # published random-init Delta(G) came through bench._softmax_operator,
        # which is float32. This is the size of that difference.
        prob32 = st.column_diameter(rows @ rows.transpose(0, 1))
        if math.isfinite(prob32) and math.isfinite(logd):
            worst32 = max(worst32, abs(prob32 - logd) / max(prob32, 1e-300))
    # The other direction: a logit scale wide enough to underflow the PRODUCT
    # of two softmax entries. float64 exp() dies below about -745.13 nats, so
    # it is the GRAM min, not the row range, that must cross it: scale 200 gives
    # a row range of 1201.8 but a log-Gram min of only -426.67 and does NOT
    # underflow. Scale 400 puts the log-Gram min at -853.32 and it does.
    g = torch.Generator().manual_seed(4242)
    killed = 0
    for _ in range(8):
        u = torch.randn(4, 512, generator=g).double() * 400.0
        la2 = torch.log_softmax(u, dim=-1)
        prob_g = (la2.exp() @ la2.exp().transpose(0, 1))
        if (int((prob_g == 0).sum()) > 0
                and math.isinf(st.column_diameter(prob_g))
                and math.isfinite(delta_G_log(log_gram(la2)))):
            killed += 1
    return agree, len(seeds), worst, worst32, killed, 8


def _qk(s: int, d: int, seed: int):
    """The (q, k) of a random-init cell, in the generator order cell() uses."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    return x0 @ wq, x0 @ wk


def trained_delta_G(seed: int, k_piv: int, n_ex: int):
    """Delta(G) at TRAINED and at INIT projections, one code path, weights differ.

    Uses the cached weights produced by scale/trained_projections.py; nothing is
    retrained here. Both arms run through the identical construction so the only
    difference between the two columns is which projection matrix was loaded --
    the paired design that F-selector taught this project to insist on.
    """
    pair = tp.train_pair(seed=seed)
    out = {"trained": [], "init": []}
    for which, trained in (("init", False), ("trained", True)):
        for idx in range(n_ex):
            q, kk = tp._geometry(pair, trained, idx)
            s = q.shape[0]
            piv = select_pivots(kk, min(k_piv, s - 2), exclude=(s - 1,))
            piv = piv[piv > 0]
            if piv.numel() < 2:
                continue
            p_max = int(piv.max())
            if p_max < 2:
                continue
            la = log_rows(q, kk, piv, p_max)
            lgm = log_gram(la)
            dk = q.shape[-1]
            w = (q.double() @ kk.double().transpose(-2, -1)) / math.sqrt(dk)
            _, nm = bench._causal_mask_pair(s, 0, str(q.device))
            out[which].append(dict(
                delta_G=delta_G_log(lgm),
                log_gram_min=float(lgm.min()),
                logit_scale=float(w[~nm].abs().mean()),
                eff_supp=float(torch.stack(
                    [torch.tensor(tp.effective_support(r.exp())) for r in la]).mean()),
                overlap_min=float(((la > float("-inf")).double()
                                   @ (la > float("-inf")).double().transpose(0, 1))[
                    ~torch.eye(int(piv.numel()), dtype=torch.bool)].min()),
                n_piv=int(piv.numel()), p_max=p_max))
    return pair, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=256)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 0.9])
    a = ap.parse_args()
    q1, q2 = chr(39) + "k" + chr(39), chr(39) + "seed" + chr(39)

    print("=== MUST-FIRE. threads=%d ===" % torch.get_num_threads())
    fired, nf, finite, nv, dg0, dgmax = must_fire()
    print("  disjoint-support pivots -> Delta(G) = +inf : %d/%d draws "
          "(first draw Delta_G=%s)" % (fired, nf, dg0))
    print("  overlapping pivots      -> Delta(G) finite : %d/%d draws "
          "(max Delta_G=%.6f)" % (finite, nv, dgmax))
    if fired != nf or finite != nv:
        print("  CONTROL DID NOT FIRE IN BOTH DIRECTIONS. Nothing below counts.")
        return 1

    print("\n=== Delta(G) AT THE SWEEP GEOMETRY, EXTREME RAYS, EXACT ===")
    print("  s=%d d=%d ks=%s seeds=%s  (the cells that produced 0.961793)"
          % (a.s, a.d, a.ks, a.seeds))
    print("  %4s %5s %5s %8s %11s %10s %11s %9s %11s"
          % ("k", "seed", "|P|", "G zeros", "G min", "Delta(G)", "tanh(D/4)",
             "effsupp", "overlap min"))
    facts = {}
    for k in a.ks:
        for seed in a.seeds:
            rows, gate, p_max, piv = cell(a.s, a.d, k, seed)
            f = gram_facts(rows)
            f.update(s=a.s, d=a.d, k_req=k, seed=seed, p_max=p_max)
            f["kappa_G"] = hb.kappa_cert(f["delta_G"])
            facts[(k, seed)] = (f, rows, gate)
            print("  %4d %5d %5d %8d %11.4e %10.4f %11.8f %9.4f %11.1f"
                  % (k, seed, f["k"], f["gram_zeros"], f["gram_min"],
                     f["delta_G"], f["kappa_G"], f["eff_support_mean"],
                     f["overlap_min"]))

    print("\n=== THE TEST, PER SEED, NOT POOLED ===")
    print("  w-space: kappa(T_w) = beta * kappa(G) is an IDENTITY; the only")
    print("           inequality is Birkhoff on G, stated as an EQUALITY.")
    print("  %4s %5s %6s %11s %11s %11s %11s %6s %8s"
          % ("k", "seed", "beta", "attained", "att/beta", "tanh(D/4)",
             "predicted", "holds", "tight"))
    rowsout = []
    for k in a.ks:
        for seed in a.seeds:
            f, rows, gate = facts[(k, seed)]
            for beta in a.betas:
                att = attained_w(rows, gate, beta, seed)
                pred = beta * f["kappa_G"]
                holds = att <= pred + 1e-9
                ratio = att / beta if beta > 0 else float("nan")
                rowsout.append(dict(k=k, seed=seed, beta=beta, attained=att,
                                    att_over_beta=ratio, kappa_G=f["kappa_G"],
                                    predicted=pred, holds=bool(holds),
                                    delta_G=f["delta_G"]))
                print("  %4d %5d %6.2f %11.8f %11.8f %11.8f %11.8f %6s %8.4f"
                      % (k, seed, beta, att, ratio, f["kappa_G"], pred,
                         str(holds), ratio / f["kappa_G"]))

    bad = [r for r in rowsout if not r["holds"]]
    print("\n  bound holds: %d/%d cells" % (len(rowsout) - len(bad), len(rowsout)))
    sp = [r["att_over_beta"] for r in rowsout]
    print("  attained/beta spread across ALL cells: %.8f .. %.8f"
          % (min(sp), max(sp)))
    tt = [r["att_over_beta"] / r["kappa_G"] for r in rowsout]
    print("  tightness (att/beta divided by tanh(D/4)): %.6f .. %.6f"
          % (min(tt), max(tt)))
    print("  House inverted the POOLED 0.961793 to Delta(G) = 7.8772 nats.")
    print("  measured Delta(G) range at that geometry: %.4f .. %.4f nats"
          % (min(f[0]["delta_G"] for f in facts.values()),
             max(f[0]["delta_G"] for f in facts.values())))

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh_:
        for r in rowsout:
            fh_.write(json.dumps(r) + "\n")
    print("  journal: %s" % JOURNAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
