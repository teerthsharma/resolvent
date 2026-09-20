"""DR-1/E: does a TRAINED grade-1 model actually reach below the grade-0 ceiling?

WHY THIS HAD TO BE MEASURED. dr1.py establishes a closed-form ceiling: on its bed
the best possible grade-0 (node-potential) model leaves 0.6859 of ||Y|| as residual,
because a node model's edge predictions live in im(d0) and the target has a curl part
outside it. That is a least-squares optimum for the ENTIRE grade-0 class -- no width,
no depth, no budget improves it. But this project has been burned four times by
headroom that exists in the mathematics and never reaches the loss; most sharply, a
20x jump in kappa moved the raw residual from 22% to 73% of range and moved the
trained separation by 0.0001. So the theorem is not the result. The result is below.

THE ANSWER: YES, AND NOT NARROWLY. A trained grade-1 MLP reaches 0.1041 of ||Y|| on
held-out instances against the closed-form grade-0 ceiling of 0.6824 -- 0.5783 below
it, 74.8 bed-draw sds, positive on 12/12 bed draws. The parameter-matched grade-0 arm
lands at 0.6865, i.e. it SATURATES its own ceiling to within 0.0041 and cannot move.
This is the first separation in this repo that is not inside the error bars, and the
reason is structural: the two arms are different HYPOTHESIS CLASSES, not two
parametrisations of one class, which is exactly the failure mode dr1.py was built to
escape.

RUN (12 bed draws, 512/128/256 train/val/test instances each, 4000 steps, Adam 1e-3,
best-val checkpointing, model seed 0, CPU, wall 327.8 s). Held-out test instances the
model never saw, relative residual ||Y - Yhat|| / ||Y||:

    arm                                    mean    bed sd
    trivial constant predictor           1.0009    0.0012
    (1) GRADE-0 CEILING     closed form  0.6824    0.0026
    (0) RIDGE               closed form  0.3800    0.0071
    (2) GRADE-0 TRAINED                  0.6865    0.0027
    (3) GRADE-1 TRAINED                  0.1041    0.0083

    ceiling - grade1_trained  = +0.5783   95% CI [+0.5734, +0.5832]   12/12 beds
    grade0_trained - grade1_trained = +0.5823   95% CI [+0.5774, +0.5873]

RIDGE IS REPORTED FIRST, and it carries the sharpest single fact here: closed-form
ridge on the raw input already reads 0.3800, which is 0.30 BELOW the grade-0 ceiling.
The cheapest possible grade-1 model -- a linear map, no training loop, no optimiser --
already claims 44% of the headroom. That is what makes the trained result credible
rather than surprising: the headroom is not a fragile quantity that only a lucky
optimiser finds. Ridge is nowhere near the best arm (0.3800 vs 0.1041), so the bed is
not a linear regression either.

BED-DRAW sd VERSUS SEED sd, reported separately because on this project's other beds
the bed draw dominated the seed draw 4.4x (0.0774 against 0.0176) and sizing an effect
against the seed sd overstated it. On THIS bed they are comparable:

    bed-draw sd (12 fresh beds, model seed fixed)   grade-1 0.0083   grade-0 0.0027
    seed sd     (5 model seeds, bed 1000 fixed)     grade-1 0.0072   grade-0 0.0014

so the bed draw leads by only 1.15x, and the effect is ~70x either one. The number
above is sized against the BED-DRAW sd, the larger and more conservative of the two.
No unanimity rule is used anywhere: a 5/5 rule has ~6% power at this project's measured
effect sizes and gets weaker with every added seed, so this reports a mean and a t
interval over bed draws.

CONTROLS, each SEEN TO FIRE:
  (i)   The trained grade-0 arm never dips below the closed-form grade-0 optimum:
        0/12 violations. If it had, the pipeline -- not the theorem -- would be wrong.
  (ii)  SHUFFLED TARGETS: grade-0 1.0352 (sd 0.0126) and grade-1 1.0444 (sd 0.0167),
        both at or above the constant predictor's 1.0009. Neither arm can read the bed
        without the labels, so the metric is reading the model.
  (iii) CURL-FREE BED (curl_weight=0.0), where the ceiling collapses to 5.7e-16 and
        there is no class boundary left to find: the arm gap collapses from +0.5823 to
        +0.0161, a 36x reduction. At 4000 steps that residual +0.0161 is still
        significant; it is OPTIMISER SLACK, not a class effect, and the budget sweep
        below shows it decaying as the budget grows while the curl-bed gap GROWS.

BUDGET SWEEP (4 bed draws, seeds 1000-1003), which is the cleanest statement of the
whole result -- more training moves the grade-1 arm and cannot move the grade-0 arm,
because the grade-0 arm is already at a wall that is a theorem:

    steps   curl bed: ceiling  g0      g1       gap      curl-free: gap  95% CI
     4000            0.6817  0.6856  0.1031   +0.5825            +0.0126 [+0.0005,+0.0247]
    12000            0.6817  0.6828  0.0481   +0.6348            +0.0079 [-0.0014,+0.0171]

At 12000 steps the grade-0 arm sits 0.0011 above its closed-form optimum -- pinned to
the ceiling to four decimals -- while the grade-1 arm has halved again, and the
curl-free control's interval now CONTAINS ZERO. The control fires, the slack decays,
and the separation is the class boundary.

WHAT IS NOT CLAIMED. This measures reachability of a known ceiling on a bed built to
expose it, not a benchmark win. The grade-0 ceiling is an ORACLE quantity (it fits the
node potential to the test target itself), so the trained grade-1 arm is beating an
oracle-fitted member of the rival class, which strengthens the direction of the result
but says nothing about any other bed. The complex is fixed across bed draws (a 3x3
grid patch, 9 nodes / 16 edges / 8 triangles), so a "bed draw" resamples inputs and
targets, not topology.

RUN: python -m ceqjepa.dr1_experiment          (full, ~330 s CPU)
     python -m ceqjepa.dr1_experiment --selfcheck   (fast, planted negative, ~20 s)
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from ceqjepa.dr1 import bed, grade0_best_fit, score_bed

__all__ = ["Grade0", "Grade1", "match_width", "one_bed", "run", "selfcheck"]

SPLIT = (512, 128, 256)                  # train / val / test instances per bed draw
H1 = 64                                  # grade-1 width; grade-0's is bisected to match
STEPS = 4000
BED_SEEDS = tuple(range(1000, 1012))     # 12 bed draws, model seed fixed
MODEL_SEEDS = (0, 1, 2, 3, 4)            # 5 model seeds, bed fixed
FIXED_BED = 1000
RIDGE_LAMBDA = 1e-3                      # score_bed's value, not re-tuned here


class Grade1(nn.Module):
    """X -> edge flow directly. Free to leave im(d0)."""

    def __init__(self, v, e, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(v, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, e))

    def forward(self, x):
        return self.net(x)


class Grade0(nn.Module):
    """X -> node potential p, prediction d0 @ p. Confined to im(d0) by construction.

    d0 is a BUFFER, not a parameter, so it does not enter the parameter match -- and
    it must not, since it is a fixed property of the complex and not capacity.
    """

    def __init__(self, v, d0, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(v, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, v))
        self.register_buffer("d0t", torch.as_tensor(d0.T.copy(), dtype=torch.float32))

    def forward(self, x):
        return self.net(x) @ self.d0t


def _nparam(m):
    return sum(p.numel() for p in m.parameters())


def match_width(target, v, d0, lo=4, hi=4096):
    """Bisect the grade-0 width to the grade-1 parameter count (ood_kappa.py's rule)."""
    best = None
    while lo <= hi:
        w = (lo + hi) // 2
        c = _nparam(Grade0(v, d0, w))
        if best is None or abs(c - target) < abs(best[1] - target):
            best = (w, c)
        if c < target:
            lo = w + 1
        else:
            hi = w - 1
    return best


def instance(seed, curl_weight=1.0, split=SPLIT):
    """One bed draw, split into disjoint train / val / test instances."""
    a, b, c = split
    X, Y, cx = bed(n_instances=a + b + c, k=3, seed=seed, curl_weight=curl_weight)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    return dict(train=(Xt[:a], Yt[:a]), val=(Xt[a:a + b], Yt[a:a + b]),
                test=(Xt[a + b:], Yt[a + b:]),
                Xtr=X[:a], Ytr=Y[:a], Xte=X[a + b:], Yte=Y[a + b:], cx=cx)


@torch.no_grad()
def score(model, split):
    return float(torch.linalg.norm(split[1] - model(split[0])) / torch.linalg.norm(split[1]))


def train(model, d, seed, steps=STEPS, bs=256, lr=1e-3, shuffle=False):
    """Model selection on the validation split only; the test split is touched once."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    X, Y = d["train"]
    if shuffle:
        Y = Y[torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(seed))]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed + 1)
    best, state = float("inf"), None
    for t in range(steps):
        i = torch.randint(0, X.shape[0], (bs,), generator=g)
        loss = (model(X[i]) - Y[i]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (t + 1) % 250 == 0:
            v = score(model, d["val"])
            if v < best:
                best, state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
    model.load_state_dict(state)
    return score(model, d["test"])


def closed_form(d):
    """Arm 0 (ridge, fit on train, scored on test), arm 1 (the ceiling), the constant."""
    Xtr, Ytr, Xte, Yte = d["Xtr"], d["Ytr"], d["Xte"], d["Yte"]
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    W = np.linalg.solve(A.T @ A + RIDGE_LAMBDA * np.eye(A.shape[1]), A.T @ Ytr)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ W
    nY = np.linalg.norm(Yte)
    g0 = np.stack([grade0_best_fit(d["cx"], y) for y in Yte])
    return dict(ridge=float(np.linalg.norm(Yte - pred) / nY),
                ceiling=float(np.linalg.norm(Yte - g0) / nY),
                const=float(np.linalg.norm(Yte - Ytr.mean(0)) / nY))


def one_bed(seed, curl_weight, model_seed, w0, shuffle=False, steps=STEPS, split=SPLIT):
    """All five numbers on one bed draw, closed-form arms first."""
    d = instance(seed, curl_weight, split)
    v, e = d["cx"].n_nodes, len(d["cx"].edges)
    out = closed_form(d)
    out["g0_trained"] = train(Grade0(v, d["cx"].d0, w0), d, model_seed,
                              shuffle=shuffle, steps=steps)
    out["g1_trained"] = train(Grade1(v, e, H1), d, model_seed, shuffle=shuffle, steps=steps)
    return out


def agg(rows, key):
    a = np.array([r[key] for r in rows])
    return a.mean(), a.std(ddof=1)


def ci(diffs, conf=0.95):
    """Mean and t interval. NOT a unanimity rule: 5/5 has ~6% power at this repo's effects."""
    a = np.asarray(diffs)
    h = stats.t.ppf(0.5 + conf / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return a.mean(), a.mean() - h, a.mean() + h, a.std(ddof=1)


def _matched_width():
    _, _, cx = bed(n_instances=2, k=3, seed=0)
    n1 = _nparam(Grade1(cx.n_nodes, len(cx.edges), H1))
    w0, n0 = match_width(n1, cx.n_nodes, cx.d0)
    return cx, w0, n0, n1


def run(bed_seeds=BED_SEEDS, model_seeds=MODEL_SEEDS, steps=STEPS, split=SPLIT):
    t0 = time.time()
    Xr, Yr, cxr = bed(n_instances=200, k=3, seed=0)
    ref = score_bed(Xr, Yr, cxr)
    print("dr1's own default bed, reproduced: ridge %.4f  grade-0 ceiling %.4f  curl frac %.4f"
          % (ref["ridge"], ref["grade0_ceiling"], ref["curl_frac"]))

    cx, w0, n0, n1 = _matched_width()
    print("PARAMETER MATCH  grade-1 %d (width %d)   grade-0 %d (width %d)   ratio %.4f"
          % (n1, H1, n0, w0, n0 / n1))
    assert abs(n0 / n1 - 1.0) < 0.01, "arms not matched to 1%%: ratio %.4f" % (n0 / n1)
    print("%d bed draws, %d/%d/%d train/val/test instances each, %d steps. Held-out test."
          % (len(bed_seeds), *split, steps))

    curl = [one_bed(s, 1.0, 0, w0, steps=steps, split=split) for s in bed_seeds]
    free = [one_bed(s, 0.0, 0, w0, steps=steps, split=split) for s in bed_seeds]
    shuf = [one_bed(s, 1.0, 0, w0, shuffle=True, steps=steps, split=split)
            for s in bed_seeds[:5]]
    seedrun = [one_bed(FIXED_BED, 1.0, ms, w0, steps=steps, split=split) for ms in model_seeds]

    print("\n=== CURL BED, held-out. RIDGE AND THE CEILING ARE CLOSED FORM AND COME FIRST ===")
    print("%-34s %8s %8s" % ("arm", "mean", "bed sd"))
    for key, label in (("const", "trivial constant predictor"),
                       ("ceiling", "(1) GRADE-0 CEILING closed form"),
                       ("ridge", "(0) RIDGE closed form"),
                       ("g0_trained", "(2) GRADE-0 TRAINED"),
                       ("g1_trained", "(3) GRADE-1 TRAINED")):
        print("%-34s %8.4f %8.4f" % (label, *agg(curl, key)))

    print("\nBED-DRAW sd (fresh beds, model seed fixed): grade-1 %.4f   grade-0 %.4f"
          % (agg(curl, "g1_trained")[1], agg(curl, "g0_trained")[1]))
    print("SEED sd     (bed %d fixed, %d seeds)      : grade-1 %.4f   grade-0 %.4f"
          % (FIXED_BED, len(model_seeds), agg(seedrun, "g1_trained")[1],
             agg(seedrun, "g0_trained")[1]))

    d_ceil = [r["ceiling"] - r["g1_trained"] for r in curl]
    m, lo, hi, s = ci(d_ceil)
    print("\nTHE QUESTION  ceiling - grade1_trained, paired per bed draw")
    print("  mean %+.4f   95%% CI [%+.4f, %+.4f]   bed sd %.4f   %.1f bed-draw sds   %d/%d > 0"
          % (m, lo, hi, s, m / s, sum(x > 0 for x in d_ceil), len(d_ceil)))
    print("  per-bed: %s" % " ".join("%+.4f" % x for x in d_ceil))
    m2, lo2, hi2, _ = ci([r["g0_trained"] - r["g1_trained"] for r in curl])
    print("  grade0_trained - grade1_trained: %+.4f  95%% CI [%+.4f, %+.4f]" % (m2, lo2, hi2))

    print("\n=== CONTROLS ===")
    bad = [r for r in curl if r["g0_trained"] < r["ceiling"] - 1e-6]
    print("(i) trained grade-0 never below the closed-form grade-0 optimum: %d/%d violations"
          % (len(bad), len(curl)))
    assert not bad, "a trained node model beat its own class optimum: the pipeline is wrong"

    cm = agg(curl, "const")[0]
    print("(ii) SHUFFLED TARGETS (%d beds) must collapse to the constant %.4f:" % (len(shuf), cm))
    for key in ("g0_trained", "g1_trained"):
        mm, ss = agg(shuf, key)
        print("     %-12s %.4f (sd %.4f)   %s"
              % (key, mm, ss, "FIRED" if mm >= cm - 0.01 else "DID NOT FIRE"))
        assert mm >= cm - 0.01, "%s beat the constant predictor on shuffled labels" % key

    fm, fl, fh, _ = ci([r["g0_trained"] - r["g1_trained"] for r in free])
    print("(iii) CURL-FREE BED, no class boundary to find. ceiling %.3e"
          % agg(free, "ceiling")[0])
    print("      g0 %.4f (sd %.4f)   g1 %.4f (sd %.4f)"
          % (*agg(free, "g0_trained"), *agg(free, "g1_trained")))
    print("      gap %+.4f  95%% CI [%+.4f, %+.4f]  against %+.4f on the curl bed  (%.0fx smaller)"
          % (fm, fl, fh, m2, m2 / fm if fm else float("inf")))
    assert abs(fm) < 0.1 * abs(m2), "the arms separate on a curl-free bed: the effect is not the class"

    print("\nwall %.1fs" % (time.time() - t0))
    return dict(curl=curl, free=free, shuffled=shuf, seeds=seedrun, ceiling_gap=(m, lo, hi, s))


def selfcheck():
    """Fast, with the PLANTED NEGATIVE dr1.py's demo() uses: a curl-free target.

    The planted negative is the point. The same machinery, the same arms, the same
    metric, run on a bed where the THEOREM says the class boundary is absent -- and it
    must report no separation there. A pipeline that separates the arms on a curl-free
    bed is measuring the optimiser, and every number in the docstring would be void.
    """
    split, steps = (256, 64, 128), 800
    cx, w0, n0, n1 = _matched_width()
    print("(a) PARAMETER MATCH  grade-1 %d   grade-0 %d   ratio %.4f" % (n1, n0, n0 / n1))
    assert abs(n0 / n1 - 1.0) < 0.01, "arms not matched to 1%%"

    print("(b) the grade-0 arm's output must lie in im(d0) EXACTLY, or it is not grade 0")
    m = Grade0(cx.n_nodes, cx.d0, w0)
    pred = m(torch.randn(32, cx.n_nodes)).detach().numpy()
    off = max(float(np.linalg.norm(y - grade0_best_fit(cx, y))) for y in pred)
    print("    max residual off im(d0) over 32 random inputs = %.3e" % off)
    assert off < 1e-4, "the grade-0 arm can leave im(d0): it is not the class it claims"

    live = [one_bed(s, 1.0, 0, w0, steps=steps, split=split) for s in (1000, 1001)]
    dead = [one_bed(s, 0.0, 0, w0, steps=steps, split=split) for s in (1000, 1001)]
    gl = np.mean([r["g0_trained"] - r["g1_trained"] for r in live])
    gd = np.mean([r["g0_trained"] - r["g1_trained"] for r in dead])
    print("(c) curl bed      ceiling %.4f  g0 %.4f  g1 %.4f  gap %+.4f"
          % (agg(live, "ceiling")[0], agg(live, "g0_trained")[0], agg(live, "g1_trained")[0], gl))
    print("(d) PLANTED NEGATIVE: curl-free bed, ceiling %.3e  g0 %.4f  g1 %.4f  gap %+.4f"
          % (agg(dead, "ceiling")[0], agg(dead, "g0_trained")[0],
             agg(dead, "g1_trained")[0], gd))
    assert agg(dead, "ceiling")[0] < 1e-10, "the curl-free bed has a nonzero ceiling"
    assert abs(gd) < 0.1 * abs(gl), "the arms separate with no curl present: effect is not the class"
    print("    FIRED: %+.4f with curl, %+.4f without -- %.0fx." % (gl, gd, abs(gl / gd)))

    print("(e) the trained grade-0 arm must not beat its own closed-form optimum")
    worst = min(r["g0_trained"] - r["ceiling"] for r in live)
    print("    min (grade0_trained - ceiling) = %+.4f  (must be >= 0)" % worst)
    assert worst >= -1e-6, "a trained node model beat the grade-0 least-squares optimum"
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        run()
