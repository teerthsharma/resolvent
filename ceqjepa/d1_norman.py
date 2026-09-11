"""REMOVAL-ECHO v1, iteration 2 -- domain D1: Norman, Horlbeck, Replogle et al. 2019,
Science 365:786, GEO GSE133344, read from the scPerturb h5ad (Zenodo record 13350497).

MODALITY. The pre-registration calls D1 "double knockouts". It is not: Norman 2019
is CRISPR ACTIVATION in K562 -- each guide OVEREXPRESSES its target (gain of
function). The Mobius statistic is modality-agnostic, so the additivity test
stands; the "removal" framing does not apply literally, and nothing here relabels
activation as removal. The pre-registration is left as committed.

THE FROZEN PER-PAIR STATISTIC. Written into this file before the data file was
opened and before any interaction was computed on real data:
  x       log1p(1e4 * counts / total_counts), per cell
  genes   the N_GENES highest-mean genes over control + single cells, every
          CRISPRa target removed (a target's own overexpression is the
          perturbation, not the response)
  basis   top K principal axes of x[genes], fit on control + single cells ONLY.
          Doubles never shape the basis, so the surrogate that replaces them runs
          through the same projection and the basis cannot be steered by their
          noise. Cost: an interaction orthogonal to that K-dim subspace is
          invisible -- lost power, never a false positive.
  f(S)    mean PC score of the cells whose targets are exactly S
  I(A,B)  removal_echo.mobius_pair(f, A, B), a K-vector
  T       I' V^-1 I, V = sum over the four groups of Cov_g / n_g, each group with
          its OWN covariance (a perturbation changes the spread, not only the mean)
  p       Hotelling F reference, Krishnamoorthy-Yu Behrens-Fisher df. Analytic,
          because BH at m ~ 130 needs p near 1e-4, which a per-pair bootstrap
          cannot resolve at this cost; its calibration is tested END TO END by the
          cell-level residual bootstrap in `surrogate` instead.
  family  every double whose own group and both singles have >= MIN_CELLS cells
  BH      removal_echo.bh at q = Q

ORDER OF OPERATIONS in `measure`, enforced by the code: the matched-noise
additive surrogate must read near zero, and the detection floor is measured,
BEFORE the first real interaction is computed. A surrogate that reads
non-additive voids the run and no real number is printed.

RUN
  python -m ceqjepa.d1_norman --selfcheck         offline, synthetic, no data
  python -m ceqjepa.d1_norman --data FILE.h5ad    the pre-registered measurement
"""

import argparse
import itertools
import math
import sys

import numpy as np
from scipy import stats

from ceqjepa.removal_echo import bh, mobius_pair, shapley

N_GENES, K, MIN_CELLS, Q = 1000, 10, 30, 0.05
SIZES = (2, 3, 4, 5, 6, 7, 8, 10)          # planted interaction V-length, in sd_I
VOID_FRAC = 0.01                           # additive surrogate mean BH fraction that voids the run
CTRL = frozenset()


def quad(a, b):
    """The four groups of one Mobius contrast, in mobius_pair's sign order + - - +."""
    return frozenset({a, b}), frozenset({a}), frozenset({b}), CTRL


# ------------------------------------------------------------ the frozen test --
def basis(X, fit_rows, k=K):
    """Top-k principal axes of X[fit_rows] -- control and single cells only."""
    Xf = X[fit_rows]
    mu = Xf.mean(0)
    Xf -= mu
    C = (Xf.T @ Xf).astype(np.float64)
    return mu, np.linalg.eigh(C)[1][:, ::-1][:, :k].astype(np.float32)


def hotelling(d, groups):
    """T^2 = d' V^-1 d for d a +-1 sum of independent group means, V = sum Cov_g/n_g.

    F reference with the Krishnamoorthy-Yu (2004) Behrens-Fisher degrees of freedom,
    divisor n_g - 1: exact one-sample Hotelling for one group, Welch-Satterthwaite
    at k = 1. groups: (mean, cov, n) triples. Returns (T, p, nu).
    """
    Vs = [S / n for _, S, n in groups]
    Vi = np.linalg.inv(sum(Vs))
    T = float(d @ Vi @ d)
    k = d.size
    nu = (k + k * k) / sum((np.trace(A @ A) + np.trace(A) ** 2) / (n - 1)
                          for A, (_, _, n) in zip([V @ Vi for V in Vs], groups))
    return T, float(stats.f.sf(T * (nu - k + 1) / (nu * k), k, nu - k + 1)), nu


def project(X, rows, k=K):
    """Basis from control + singles, then per-group (mean, cov, n) of the PC scores.
    Computes no interaction."""
    fit = np.concatenate([r for s, r in rows.items() if len(s) < 2])
    mu, W = basis(X, fit, k)
    Z = X @ W - mu @ W
    return W, {s: (Z[r].mean(0, dtype=np.float64), np.cov(Z[r], rowvar=False), len(r))
               for s, r in rows.items()}


def test_pairs(gs, pairs):
    """The frozen per-pair test. Returns I (m x K), T, p, BH mask."""
    out = []
    for a, b in pairs:
        I = mobius_pair(lambda s: gs[s][0], a, b)
        out.append((I,) + hotelling(I, [gs[s] for s in quad(a, b)])[:2])
    I, T, p = (np.array(c) for c in zip(*out))
    return I, T, p, bh(p, Q)


# ------------------------------------------------ matched-noise surrogate ------
def surrogate(X, rows, rng, out, plant=None):
    """Cell-level residual bootstrap into an ADDITIVE world at the data's own noise.

    Every group is re-drawn with replacement from its OWN cell residuals at its own
    n (so each group keeps its covariance, tails and bimodality), recentred on a
    target mean: control and singles on their observed means, every double on
    f(A) + f(B) - f(ctrl) from those singles. plant adds a known gene-space
    interaction to chosen doubles.
    """
    mean = {s: X[r].mean(0) for s, r in rows.items()}
    for s, r in rows.items():
        tgt = mean[s]
        if len(s) == 2:
            a, b = tuple(s)
            tgt = mean[frozenset({a})] + mean[frozenset({b})] - mean[CTRL]
            if plant and s in plant:
                tgt = tgt + plant[s]
        out[r] = X[rng.choice(r, len(r))] - mean[s] + tgt
    return out


def power_curve(X, rows, pairs, W, gs, rng, sizes, reps):
    """BH's detection floor at this family's m and this data's noise.

    ceil(5% of m) doubles of the additive surrogate get a planted interaction of
    V-length `size` (in sd_I, along a random direction) -- 5% is the counter's own
    line. Returns power (fraction of planted pairs BH finds) and readback (median
    sqrt(max(T - K, 0)) over planted pairs, which must track `size`)."""
    m = len(pairs)
    n_plant = math.ceil(0.05 * m)
    buf = np.empty_like(X)
    power, readback = {}, {}
    for size in sizes:
        hits, rb = [], []
        for _ in range(reps):
            idx = rng.choice(m, n_plant, replace=False)
            plant = {}
            for i in idx:
                w, v = np.linalg.eigh(sum(gs[s][1] / gs[s][2] for s in quad(*pairs[i])))
                e = rng.normal(size=w.size)
                dz = (v * np.sqrt(w)) @ v.T @ (size * e / np.linalg.norm(e))
                plant[frozenset(pairs[i])] = (W @ dz.astype(np.float32))
            _, g2 = project(surrogate(X, rows, rng, buf, plant), rows)
            _, T, _, hit = test_pairs(g2, pairs)
            hits.append(hit[idx].mean())
            rb.extend(np.sqrt(np.maximum(T[idx] - K, 0)))
        power[size], readback[size] = float(np.mean(hits)), float(np.median(rb))
    return power, readback


def null_surrogate(X, rows, pairs, rng, reps):
    """The matched-noise additive must-fire: BH fraction per replicate, pooled p."""
    buf = np.empty_like(X)
    frac, pv = [], []
    for _ in range(reps):
        _, g2 = project(surrogate(X, rows, rng, buf), rows)
        _, _, p, hit = test_pairs(g2, pairs)
        frac.append(hit.mean())
        pv.append(p)
    return np.array(frac), np.concatenate(pv)


# ------------------------------------------------------------- real data -------
# scPerturb NormanWeissman2019_filtered.h5ad, Zenodo 13350497 (v1.4 of concept 7041848),
# 698,680,199 bytes, md5 c870e6967d91c017d9da827bab183cd6 as Zenodo publishes it.
# load() refuses any other file.
DATA_SHA256 = "efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc"


def load(path):
    """Hash-pinned scPerturb h5ad -> (X, rows, pairs, info lines).

    Streams the file's CSC count matrix in gene blocks (361.6M nonzeros do not fit
    this machine's free memory): pass 1 per-cell totals, pass 2 per-gene mean x over
    control + single cells, then only the chosen columns and the target columns
    are read. A cell's condition is its set of CRISPRa targets with
    control / NegCtrl tokens dropped, so 'A_NegCtrl0', 'NegCtrl0_A' and 'A' are one
    single and 'A_B', 'B_A' one double; unlabelled cells are excluded, never
    pooled into control. X holds the tested family's cells as float32 x on the
    frozen gene set.
    """
    import collections
    import hashlib
    import re

    import h5py
    import scipy.sparse as sp

    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            sha.update(block)
    if sha.hexdigest() != DATA_SHA256:
        raise SystemExit("REFUSED: %s has sha256 %s, not the pinned %s"
                         % (path, sha.hexdigest(), DATA_SHA256))

    def strings(a):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)

    def column(f, grp, name):
        o = f[grp][name]
        if isinstance(o, h5py.Group):                   # anndata >= 0.8 categorical
            cats, codes = strings(o["categories"][...]), o["codes"][...]
        elif "categories" in o.attrs:                   # anndata 0.7 categorical
            cats, codes = strings(f[o.attrs["categories"]][...]), o[...]
        else:
            return strings(o[...])
        return np.where(codes >= 0, cats[np.maximum(codes, 0)], "nan")

    ctrl_tok = re.compile(r"(?i)^(control|ctrl|negctrl\d*)$")

    def parse(lab):
        if lab in ("nan", "None", ""):
            return None
        return frozenset(t for t in re.split(r"[_+]", lab) if t and not ctrl_tok.match(t))

    with h5py.File(path, "r") as f:
        idx = f["var"].attrs["_index"]
        genes = column(f, "var", idx.decode() if isinstance(idx, bytes) else idx)
        cond = [parse(lab) for lab in column(f, "obs", "perturbation")]
        count = collections.Counter(cond)
        big = lambda s: count.get(s, 0) >= MIN_CELLS
        singles = sorted((s for s in count if s is not None and len(s) == 1 and big(s)), key=sorted)
        doubles = sorted((s for s in count if s is not None and len(s) == 2), key=sorted)
        pairs = [tuple(sorted(s)) for s in doubles
                 if big(s) and big(CTRL) and all(big(frozenset({g})) for g in s)]
        keep = [CTRL] + singles + [frozenset(p) for p in pairs]
        cid = {s: i for i, s in enumerate(keep)}
        cell = np.array([cid.get(c, -1) for c in cond])
        fit = (cell >= 0) & (cell <= len(singles))          # control + singles
        targets = sorted(set().union(*keep))
        gi = {g: i for i, g in enumerate(genes)}
        tcol = np.array([gi.get(t, -1) for t in targets])

        Xg = f["X"]
        enc = Xg.attrs["encoding-type"]
        enc = enc.decode() if isinstance(enc, bytes) else enc
        if enc != "csc_matrix":
            raise SystemExit("REFUSED: the loader streams a csc X; this file's X is %s" % enc)
        ncell, ngene = (int(v) for v in Xg.attrs["shape"])
        assert (ncell, ngene) == (len(cond), len(genes)), "obs/var do not match X"
        indptr = Xg["indptr"][...].astype(np.int64)

        def cols(g0, g1):
            """Raw counts of genes g0..g1-1: (cell index, count, local indptr)."""
            a, b = indptr[g0], indptr[g1]
            return (Xg["indices"][a:b], Xg["data"][a:b].astype(np.float64),
                    indptr[g0:g1 + 1] - a)

        blocks, g0 = [], 0                          # gene blocks of <= 1e7 nonzeros
        while g0 < ngene:
            g1 = int(np.searchsorted(indptr, indptr[g0] + 10 ** 7, "right")) - 1
            blocks.append((g0, min(ngene, max(g0 + 1, g1))))
            g0 = blocks[-1][1]

        total = np.zeros(ncell)
        for g0, g1 in blocks:
            r, d, _ = cols(g0, g1)
            if g0 == 0 and not np.array_equal(d, np.round(d)):
                raise SystemExit("REFUSED: X does not hold raw counts")
            total += np.bincount(r, weights=d, minlength=ncell)
        fitf, score = fit.astype(np.float64), np.empty(ngene)
        for g0, g1 in blocks:
            r, d, ip = cols(g0, g1)
            M = sp.csc_matrix((np.log1p(1e4 * d / total[r]), r, ip), shape=(ncell, g1 - g0))
            score[g0:g1] = (M.T @ fitf) / fit.sum()
        score[tcol[tcol >= 0]] = -np.inf
        G = np.sort(np.argsort(score)[::-1][:N_GENES])

        tsum = np.zeros((len(keep), len(targets)))  # linear 1e4-normalised sums
        for t, g in enumerate(tcol):
            if g >= 0:
                r, d, _ = cols(g, g + 1)
                k = cell[r] >= 0
                tsum[:, t] = np.bincount(cell[r][k], weights=1e4 * d[k] / total[r][k],
                                         minlength=len(keep))
        pos = np.full(ncell, -1)
        pos[cell >= 0] = np.arange(int((cell >= 0).sum()))
        X = np.zeros((int((cell >= 0).sum()), N_GENES), np.float32)
        for j, g in enumerate(G):
            r, d, _ = cols(g, g + 1)
            k = pos[r] >= 0
            X[pos[r][k], j] = np.log1p(1e4 * d[k] / total[r][k])
    kept = cell[cell >= 0]
    rows = {s: np.nonzero(kept == i)[0] for s, i in cid.items()}

    fold = {}
    for s in singles:
        (g,) = s
        t = targets.index(g)
        if tcol[t] >= 0:
            fold[g] = (tsum[cid[s], t] / count[s] + 0.01) / (tsum[0, t] / count[CTRL] + 0.01)
    r = np.array(list(fold.values()))
    tested = set(pairs)
    info = [
        "DATA: %s" % path,
        "  sha256 %s (pinned)" % DATA_SHA256,
        "  %d cells x %d genes; %d unlabelled (excluded); control %d cells; %d cells carry "
        "3+ targets" % (ncell, ngene, count.get(None, 0), count.get(CTRL, 0),
                        sum(n for s, n in count.items() if s is not None and len(s) > 2)),
        "  %d singles at >= %d cells; %d distinct doubles in the file"
        % (len(singles), MIN_CELLS, len(doubles)),
        "FAMILY: m = %d pairs -- every double whose own group and both singles have >= %d cells"
        % (len(pairs), MIN_CELLS),
        "  excluded doubles: %s" % (", ".join(
            "%s(n=%d; singles n=%s)" % ("+".join(sorted(s)), count[s],
                                       "/".join(str(count.get(frozenset({g}), 0)) for g in sorted(s)))
            for s in doubles if tuple(sorted(s)) not in tested) or "none"),
        "MODALITY, read from the data: each single's OWN target, mean 1e4-normalised "
        "expression in its cells over control (+0.01 both sides)",
        "  %d singles with the target in var: median fold %.2f; %d up > 1.5x; %d down < 1/1.5"
        % (r.size, float(np.median(r)), int((r > 1.5).sum()), int((r < 1 / 1.5).sum())),
        "  ETS2 in ETS2 cells: %s (the paper reports 9.2-fold)"
        % ("%.1f-fold" % fold["ETS2"] if "ETS2" in fold else "not measured"),
        "GENES: top %d by mean x over %d control + single cells; %d measured targets removed"
        % (N_GENES, int(fit.sum()), int((tcol >= 0).sum())),
        "  cells in the tested family: %d" % len(kept),
    ]
    return X, rows, pairs, info


def measure(path):
    rng = np.random.default_rng(0)
    X, rows, pairs, info = load(path)
    print("\n".join(info))
    m = len(pairs)
    W, gs = project(X, rows)                        # no interaction computed yet
    V = [sum(gs[s][1] / gs[s][2] for s in quad(*p)) for p in pairs]
    print("MEASURED NOISE: median within-group sd per PC axis %.4f; median per-pair "
          "sd_I %.4f (sqrt(tr V / K), PC units)"
          % (np.median([np.sqrt(np.mean(np.diag(g[1]))) for g in gs.values()]),
             np.median([np.sqrt(np.trace(v) / K) for v in V])))

    print("\nMUST-FIRE AT MATCHED NOISE: every double replaced by f(A)+f(B)-f(ctrl),")
    print("every group re-drawn from its own real cell residuals, identical pipeline.")
    frac, p0 = null_surrogate(X, rows, pairs, rng, 20)
    print("  20 replicates x m = %d: BH fraction mean %.4f, max %.4f; per-pair p<0.05 "
          "rate %.3f (nominal 0.050)" % (m, frac.mean(), frac.max(), (p0 < 0.05).mean()))
    if frac.mean() >= VOID_FRAC:
        print("  VOID: the pipeline manufactures interactions from an additive world at "
              "this noise. No real number is reported.")
        sys.exit(2)
    print("  PASSED: an additive world at D1's noise reads additive.")

    print("\nDETECTION FLOOR: %d of %d pairs planted per replicate, 10 replicates per size."
          % (math.ceil(0.05 * m), m))
    power, readback = power_curve(X, rows, pairs, W, gs, rng, SIZES, 10)
    for s in SIZES:
        print("  planted %4.1f sd_I -> BH power %.3f  (read back %.2f)"
              % (s, power[s], readback[s]))
    mde = next((s for s in SIZES if power[s] >= 0.8), None)
    print("  minimum detectable effect at 80%% power, m = %d: %s"
          % (m, ("%.0f sd_I" % mde) if mde else "ABOVE %d sd_I" % SIZES[-1]))

    print("\nTHE PRE-REGISTERED MEASUREMENT (first real interaction computed here).")
    I, T, p, hit = test_pairs(gs, pairs)
    D = int(hit.sum())
    ci = stats.binomtest(D, m).proportion_ci(method="wilson")
    print("  %d of %d pairs non-additive at BH q = %.2f: fraction %.4f, "
          "95%% Wilson CI [%.4f, %.4f]" % (D, m, Q, D / m, ci.low, ci.high))
    d = np.sqrt(np.maximum(T - K, 0))
    print("  real interaction V-length sqrt(max(T-K,0)): median %.2f, IQR [%.2f, %.2f]"
          % tuple(np.percentile(d, [50, 25, 75])))
    if mde:
        print("  pairs at or above the %d sd_I floor: %d of %d" % (mde, int((d >= mde).sum()), m))
    lam = 0.5
    pi0 = min(1.0, float((p > lam).mean() / (1 - lam)))
    print("  diagnostic, not pre-registered: Storey 1 - pi0 (lambda 0.5) = %.3f" % (1 - pi0))
    if D:
        share = [np.linalg.norm(I[i]) / sum(np.linalg.norm(v) for v in shapley(
            lambda s: gs[s][0] - gs[CTRL][0], list(pairs[i])).values())
            for i in np.nonzero(hit)[0]]
        print("  survivors: median |I| / (|phi_A| + |phi_B|) against Shapley = %.3f"
              % float(np.median(share)))
    if ci.low > 0.20:
        print("  SCORED: PREDICTION (> 20%) HOLDS -- the whole CI sits above 20%.")
    elif ci.high < 0.05:
        print("  SCORED: COUNTER (< 5%) HOLDS -- the whole CI sits below 5%.")
    else:
        print("  SCORED: point %.4f; the CI does not clear one side on its own." % (D / m))

    print("\nADMISSION EVIDENCE: does the rest of the transcriptome move to a new state?")
    singles = [s for s in rows if len(s) == 1]
    ps, ds = [], []
    for s in singles:
        t, pv, _ = hotelling(gs[s][0] - gs[CTRL][0], [gs[s], gs[CTRL]])
        ps.append(pv)
        ds.append(math.sqrt(max(t - K, 0)))
    print("  %d of %d singles shift the NON-TARGET transcriptome at BH q = %.2f; median "
          "shift %.1f sd (basis fit on these cells: descriptive, not a test)"
          % (int(bh(ps, Q).sum()), len(singles), Q, float(np.median(ds))))


# ------------------------------------------------------------- self-check ------
def _toy(rng, planted=False, n_genes=120, n_single=10, n_pairs=24):
    """Synthetic Perturb-seq in log-expression space. Means are ADDITIVE (control
    base + sum of single effects) unless planted: then pair 0 carries a synergy
    (the double moves twice the additive sum) and pair 1 a planted NEGATIVE
    interaction (suppression, AB = A, so I = -effect(B)). Noise carries what real
    cells have and a Gaussian toy lacks: group-specific covariance, t5 tails, and
    a zero-mean responder / non-responder split along each group's effect axis."""
    genes = ["g%d" % i for i in range(n_single)]
    base = rng.normal(1.0, 0.3, n_genes)
    eff = {g: rng.normal(0, 0.4, n_genes) * (rng.random(n_genes) < 0.3) for g in genes}
    F = rng.normal(0, 0.3, (n_genes, 5))
    combos = list(itertools.combinations(genes, 2))
    pairs = [combos[i] for i in rng.choice(len(combos), n_pairs, replace=False)]
    true_I = {}
    if planted:
        a, b = pairs[0]
        true_I[frozenset(pairs[0])] = eff[a] + eff[b]
        true_I[frozenset(pairs[1])] = -eff[pairs[1][1]]
    X, rows, start = [], {}, 0
    for s in [CTRL] + [frozenset({g}) for g in genes] + [frozenset(p) for p in pairs]:
        n = 3000 if not s else int(rng.integers(MIN_CELLS, 400))
        mu = base + sum(eff[g] for g in s) + true_I.get(s, 0)
        eps = ((rng.standard_t(5, (n, 5)) * rng.uniform(0.5, 2.0, 5)) @ F.T
               + 0.3 * rng.standard_t(5, (n, n_genes)))
        if s:
            pi = rng.uniform(0.2, 0.8)
            eps += np.outer((rng.random(n) < pi) - pi, mu - base)
        X.append((mu + eps).astype(np.float32))
        rows[s] = np.arange(start, start + n)
        start += n
    return np.vstack(X), rows, pairs, true_I


def selfcheck():
    rng = np.random.default_rng(1)
    print("(a) PLANTED NEGATIVE: an ADDITIVE toy with real-cell noise (group-specific")
    print("    covariance, t5 tails, bimodal responders, n from %d) must read additive." % MIN_CELLS)
    frac, pv = [], []
    for _ in range(40):
        X, rows, pairs, _ = _toy(rng)
        _, _, p, hit = test_pairs(project(X, rows)[1], pairs)
        frac.append(hit.mean())
        pv.append(p)
    pv = np.concatenate(pv)
    print("    40 toys x %d pairs: mean BH fraction %.4f; per-pair p<0.05 rate %.3f"
          % (len(pairs), np.mean(frac), (pv < 0.05).mean()))
    assert np.mean(frac) < VOID_FRAC, "an additive game reads non-additive"
    assert 0.02 < (pv < 0.05).mean() < 0.08, "the F reference is miscalibrated at these n"

    print("(b) PLANTED POSITIVE: a synergy and a planted NEGATIVE interaction are found,")
    print("    and each measured I points along its planted truth.")
    X, rows, pairs, true_I = _toy(rng, planted=True)
    W, gs = project(X, rows)
    I, T, p, hit = test_pairs(gs, pairs)
    for i, name in ((0, "synergy AB = 2(A+B)"), (1, "NEGATIVE, AB = A")):
        # Cosine in the test's own metric V^-1. A Euclidean cosine in raw PC units is
        # dominated by the high-variance axes the test down-weights: the first run of
        # this check read -0.227 on a pair the test found at p = 2.7e-5.
        Vi = np.linalg.inv(sum(gs[s][1] / gs[s][2] for s in quad(*pairs[i])))
        truth = W.T @ true_I[frozenset(pairs[i])]
        cos = float(I[i] @ Vi @ truth / math.sqrt((I[i] @ Vi @ I[i]) * (truth @ Vi @ truth)))
        print("    %-20s p = %.2e  BH hit %s  cos_V(I, truth) = %+.3f" % (name, p[i], hit[i], cos))
        # BAR RESTORED TO 0.8. The frozen copy (scratchpad/d1_norman.FROZEN.py:310)
        # asserted cos > 0.8 on a Euclidean cosine. When the readback moved to the
        # V^-1 metric after a failure at -0.227, the bar was also lowered to 0.6 --
        # a post-hoc threshold change. The Inspector flagged it. The metric change
        # stands (it is the right metric); the bar returns to its pre-registered
        # value, the conservative choice. The readbacks (+0.978, +0.910) clear it.
        assert hit[i] and cos > 0.8, "a planted interaction is missed or misread"
    print("    unplanted pairs called: %d of %d" % (int(hit[2:].sum()), len(pairs) - 2))
    assert hit[2:].sum() <= 1, "BH calls unplanted pairs"
    phi = shapley(lambda s: gs[s][0] - gs[CTRL][0], list(pairs[0]))
    eff_ab = gs[frozenset(pairs[0])][0] - gs[CTRL][0]
    assert np.abs(sum(phi.values()) - eff_ab).max() < 1e-12, "Shapley efficiency fails"

    print("(c) THE SURROGATE IS ADDITIVE even when its source is not: built from the")
    print("    planted toy, it must erase both planted interactions.")
    frac, h01 = [], []
    buf = np.empty_like(X)
    for _ in range(20):
        h = test_pairs(project(surrogate(X, rows, rng, buf), rows)[1], pairs)[3]
        frac.append(h.mean())
        h01.append(h[:2].mean())
    print("    20 surrogates: mean BH fraction %.4f; planted pairs still called %.3f"
          % (np.mean(frac), np.mean(h01)))
    assert np.mean(frac) < VOID_FRAC and np.mean(h01) < 0.1, "surrogate keeps interactions"

    print("(d) THE FLOOR'S UNITS: a planted V-length reads back as itself, and power rises.")
    X, rows, pairs, _ = _toy(rng)
    W, gs = project(X, rows)
    power, readback = power_curve(X, rows, pairs, W, gs, rng, (2, 12), 10)
    for s in (2, 12):
        print("    planted %2d sd_I -> power %.2f, read back %.2f" % (s, power[s], readback[s]))
    assert power[12] >= 0.9 > power[2], "power does not rise with the planted size"
    assert abs(readback[12] - 12) < 3, "the planted size is not what the test reads"
    print("SELFCHECK PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--data")
    a = ap.parse_args()
    if a.selfcheck or not a.data:
        selfcheck()
    else:
        measure(a.data)


if __name__ == "__main__":
    main()
