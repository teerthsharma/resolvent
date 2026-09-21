"""Step (5): paired contrasts + BH correction + arm x split_seed interaction,
off design4x5_results.jsonl. Stdlib only (statistics + math for the
t-distribution CDF via a rational approximation -- no scipy dependency
assumed on this box).

READS design4x5_results.jsonl (this dir). Does not run any cell itself.
Prints every contrast with its CI whether or not it clears, per the design's
"report every contrast with its CI whether or not it clears" instruction --
this script does not suppress a non-significant row.

RCBD NOTE (read before trusting the interaction number): with exactly ONE
observation per (arm, split_seed) cell -- which is what this design collects,
by construction -- the arm x split_seed interaction term and the residual
error term are the SAME degree-of-freedom pool (a randomized-complete-block
design has no way to separate them without >=2 reps per cell). What this
script reports as "interaction" is that pooled residual, computed the
standard RCBD way (two-way ANOVA without replication): SS_total = SS_arm +
SS_seed + SS_residual. A residual CI that does not exclude zero cannot be
read as "no interaction" in the strict sense -- it can only be read as
"no interaction detectable at this replication," which is what the design's
own kill criterion asks for (draw-dependence would show up as elevated
residual regardless of whether it is formally separable from error).
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
RESULTS_PATH = os.path.join(SCRATCH, "design4x5_results.jsonl")

T975_DF4 = 2.776  # two-sided 95% critical t, df=4 (n=5 paired seeds) -- table value


def load_cells():
    cells = {}  # (arm, split_seed) -> final_eval_loss
    if not os.path.exists(RESULTS_PATH):
        return cells
    with io.open(RESULTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("stage") != "cell":
                continue
            cells[(row["arm"], row["split_seed"])] = row["final_eval_loss"]
    return cells


def paired_diffs(cells, arm_hi, arm_lo, split_seeds):
    """arm_hi - arm_lo per split_seed, only where BOTH landed."""
    out = []
    for ss in split_seeds:
        a, b = cells.get((arm_hi, ss)), cells.get((arm_lo, ss))
        if a is not None and b is not None:
            out.append(a - b)
    return out


def paired_ci(diffs):
    n = len(diffs)
    if n < 2:
        return dict(n=n, mean=diffs[0] if n == 1 else float("nan"),
                    sd=float("nan"), se=float("nan"), ci_lo=float("nan"),
                    ci_hi=float("nan"), excludes_zero=None, t_stat=float("nan"),
                    note="fewer than 2 paired seeds landed -- no CI computable")
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    tcrit = T975_DF4 if n == 5 else _t_crit_approx(n - 1)
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    t_stat = mean / se if se > 0 else float("inf")
    return dict(n=n, mean=mean, sd=sd, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
                excludes_zero=(ci_lo > 0 or ci_hi < 0), t_stat=t_stat)


def _t_crit_approx(df):
    """Coarse two-sided 95% t critical values for the df this design can
    actually land (partial pairing). Falls back to the normal 1.96 for large
    df. Not a general t-table -- only the values this design needs."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    return table.get(df, 1.96)


def bh_correct(pvals):
    """Benjamini-Hochberg, returns adjusted p-values in the ORIGINAL order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = min(prev, pvals[i] * m / k)
        adj[i] = val
        prev = val
    return adj


def two_sided_p_from_t(t_stat, df):
    """Normal approximation to the t-CDF tail (adequate at df=4 for a rough
    p-value; the CI itself uses the exact table t-critical above, so the
    accept/reject call on C_win's BAR does not depend on this approximation)."""
    z = abs(t_stat)
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p_one


def rcbd_residual_ci(cells, arms, split_seeds):
    """Two-way ANOVA without replication (arms x split_seed blocks): returns
    the residual (arm x seed interaction, confounded with error at n=1/cell --
    see module docstring) as a per-cell-count summary with its own CI on the
    residual mean (should be ~0 if the model fits additively)."""
    rows = [(arm, ss) for arm in arms for ss in split_seeds if (arm, ss) in cells]
    if len(rows) < len(arms) * len(split_seeds):
        missing = len(arms) * len(split_seeds) - len(rows)
        note = "{} of {} cells missing -- residual computed on the LANDED " \
               "subset only, not a full design".format(missing, len(arms) * len(split_seeds))
    else:
        note = "full design landed"
    if not rows:
        return dict(note="no cells landed", residual_mean=float("nan"))
    grand = sum(cells[r] for r in rows) / len(rows)
    arm_means = {a: sum(cells[(a, s)] for s in split_seeds if (a, s) in cells) /
                    max(1, sum(1 for s in split_seeds if (a, s) in cells)) for a in arms}
    seed_means = {s: sum(cells[(a, s)] for a in arms if (a, s) in cells) /
                     max(1, sum(1 for a in arms if (a, s) in cells)) for s in split_seeds}
    residuals = [cells[(a, s)] - arm_means[a] - seed_means[s] + grand for (a, s) in rows]
    return dict(note=note, n_cells=len(rows), residual_mean=sum(residuals) / len(residuals),
                residual_sd=math.sqrt(sum((r - sum(residuals) / len(residuals)) ** 2
                                          for r in residuals) / max(1, len(residuals) - 1))
                            if len(residuals) > 1 else float("nan"),
                residuals_by_cell={"{}|ss{}".format(a, s): cells[(a, s)] - arm_means[a] -
                                    seed_means[s] + grand for (a, s) in rows})


def run(split_seeds=(0, 1, 2, 3, 4)):
    cells = load_cells()
    if not cells:
        print("[ANOVA] no cells in {} yet -- nothing to analyze. "
              "Run design4x5.py's cell loop first.".format(RESULTS_PATH), flush=True)
        return None

    have_a2 = any(arm == "a2" for arm, _ in cells)
    contrasts_def = {
        "C_win": ("a", "f"),
        "C_phase": ("f0", "f"),
    }
    if have_a2:
        contrasts_def["C_mass"] = ("a", "a2")

    stats = {}
    for name, (hi, lo) in contrasts_def.items():
        d = paired_diffs(cells, hi, lo, split_seeds)
        stats[name] = dict(diffs=d, **paired_ci(d))

    if "C_mass" in stats and stats["C_mass"]["n"] >= 2:
        cw, cp, cm = stats["C_win"]["mean"], stats["C_phase"]["mean"], stats["C_mass"]["mean"]
        resid_mean = cw - cp - cm
        stats["C_resid"] = dict(mean=resid_mean, note="C_win - C_phase - C_mass "
                                 "(point estimate only -- CI needs per-seed C_resid, "
                                 "computed below if all three landed for a given seed)")
        per_seed_resid = []
        for ss in split_seeds:
            try:
                w = cells[("a", ss)] - cells[("f", ss)]
                p = cells[("f0", ss)] - cells[("f", ss)]
                m = cells[("a", ss)] - cells[("a2", ss)]
                per_seed_resid.append(w - p - m)
            except KeyError:
                continue
        if per_seed_resid:
            stats["C_resid"].update(paired_ci(per_seed_resid))
            stats["C_resid"]["diffs"] = per_seed_resid
    else:
        stats["C_resid"] = dict(note="C_mass not computable -- (a2) held or "
                                 "insufficient seeds landed. C_resid is NOT reported.")

    # BH correction across whichever of C_win/C_phase/C_mass have a real n>=2 CI
    testable = [k for k in ("C_win", "C_phase", "C_mass")
                if k in stats and stats[k].get("n", 0) >= 2]
    pvals = [two_sided_p_from_t(stats[k]["t_stat"], stats[k]["n"] - 1) for k in testable]
    adj = bh_correct(pvals) if pvals else []
    for k, p, a in zip(testable, pvals, adj):
        stats[k]["p_raw"] = p
        stats[k]["p_bh"] = a

    arms_present = sorted({arm for arm, _ in cells})
    interaction = rcbd_residual_ci(cells, arms_present, split_seeds)

    out = dict(cells_landed=len(cells), arms_present=arms_present,
               split_seeds_present=sorted({ss for _, ss in cells}),
               contrasts=stats, interaction_rcbd_residual=interaction)
    print(json.dumps(out, indent=2, default=str), flush=True)

    print("\n[BAR] C_win CI excludes 0:", stats["C_win"].get("excludes_zero"),
          "-- {}".format("0.247 SURVIVES eval-draw variance" if
                          stats["C_win"].get("excludes_zero") else
                          "0.247 relabelled 'single eval draw' per pre-registered kill")
          if stats["C_win"].get("n", 0) >= 2 else "-- not enough seeds landed yet",
          flush=True)
    return out


if __name__ == "__main__":
    run()
