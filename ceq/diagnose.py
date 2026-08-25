"""One command. CPU. No GPU, no network, no `trust_remote_code`.

    python -m ceq.diagnose

WHAT IT ANSWERS. If you are building a non-standard attention operator, does it
have CONTENT-CONDITIONAL SIGN -- can a THIRD token decide whether token j helps
or hurts token i? Fixed value and output projections cannot do that, and a
non-negative `a_ij` can only rescale, so this is the one property that separates
a signed operator from softmax at block level rather than at matrix level.

WHY THE DECAY CURVE IS PRINTED WITH THE RATE, ALWAYS. This project measured
0.1641 at the probe's default context of s = 8, called it the module's
distinguishing property, and spent three rounds optimising a val-loss ratio
around it. Swept over context, the rate DECAYS, from
0.17480 at s = 8 to 0.00391 at s = 128 -- because a third token is 1 of ~s
intermediates in the `k >= 2` term of `J = sum_k A^k` that carries the property
at all. The property and the dilution are the same fact. A single rate at a
single context length is precisely the number that misled us, so this tool
refuses to report one without its context dependence.

THE SECOND HALF is the interventional code corpus. `p(y | do(x))` is not
identifiable from observational text, so the operator has to be fitted where
interventions are real and outcomes are measured. Every label is what CPython
printed -- there is no answer key in this repository -- and the held-out cell
requires composing a subtraction with a negation, two sign flips seen separately
in training and never together. The scorer is normalized by the spread of the
target, so 1.0 is exactly predict-the-mean and any arm at or above 1.0 learned
nothing. That line is printed with the arms so a reader does not have to
remember it.

Flags: `--fast` cuts the draw counts for a smoke run, `--json` emits the same
numbers a script can read, `--device cuda` runs the probes on a GPU if you have
one. Nothing here needs one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from . import arms, bench, corpus

OPERATORS = ("softmax", "signed", "sgate", "signmag", "paraformer")
DEPTHS = (1, 2)
DECAY_S = (8, 16, 32, 64, 128)
# Two hop counts, and the difference is not an oversight. TABLE_HOPS = 3 is
# where the seven published calibration numbers live. DECAY_HOPS = 2 is the
# campaign's shipped operator point (`RHO, HOPS = 1.5, 2`), which is what the
# published decay curve was measured at; at 1024 draws this reproduces it
# entry for entry -- 0.17480 / 0.08887 / 0.02637 / 0.01172 / 0.00391 -- rather
# than approximately, and a diagnostic that cannot reproduce the number it
# exists to explain is a second instrument wearing the first one's name.
TABLE_HOPS, DECAY_HOPS = 3, 2

# The published calibration: seven numbers were measured at these settings, so
# the default run reproduces them exactly rather than approximately.
FULL_DRAWS, FAST_DRAWS = 128, 64
# 1024 is the published draw count and reproduces the curve entry for entry.
# The exponent this comment used to justify itself against is WITHDRAWN, and
# the spread that justification quoted is the reason: 128 -> -1.295,
# 256 -> -1.551, 512 -> -1.402 across draw counts alone. A quantity that
# moves 0.26 with the DRAW COUNT is not a published constant. 1024 is kept
# as the count the rate table was measured at; 512 as the fast path.
FULL_DECAY_DRAWS, FAST_DECAY_DRAWS = 1024, 512


def sign_flip_table(n_draws: int, device) -> dict:
    """The probe, over every operator this repository can build, at two depths."""
    return {k: {str(d): bench.sign_flip_rate(k, depth=d, hops=TABLE_HOPS,
                                             n_draws=n_draws, device=device)
                for d in DEPTHS}
            for k in OPERATORS}


def decay_curve(n_draws: int, device, kind: str = "sgate") -> dict:
    """The rate against context length, with relative positions held fixed.

    `i = s-1, j = s/4, c = s/2`. Holding the positions RELATIVE is what makes
    the sweep a statement about context length: pinning `j = 1` while `s` grows
    would confound dilution with the distance between the two tokens.
    """
    rate = {}
    for s in DECAY_S:
        rate[s] = bench.sign_flip_rate(kind, depth=1, hops=DECAY_HOPS, n_draws=n_draws,
                                       s=s, i=s - 1, j=s // 4, c=s // 2,
                                       device=device)
    fit = [(math.log10(s), math.log10(r)) for s, r in rate.items() if r > 0.0]
    slope, r2 = _loglog_fit(fit)
    return dict(rate={str(s): r for s, r in rate.items()}, slope=slope, r2=r2,
                n_fitted=len(fit), n_draws=n_draws,
                published_slope=None, published_r2=None,
                extrapolated_at_2048=(10 ** (_intercept(fit, slope)
                                             + slope * math.log10(2048))
                                      if len(fit) >= 2 else None))


def _intercept(pts, slope: float) -> float:
    if not pts:
        return float("nan")
    return sum(y for _, y in pts) / len(pts) - slope * sum(x for x, _ in pts) / len(pts)


def _loglog_fit(pts) -> tuple[float, float]:
    """OLS on log10 points. Zero rates are dropped, not clamped: log10(0) is not
    a small number, it is undefined, and clamping it would invent a data point."""
    if len(pts) < 2:
        return float("nan"), float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    ss_res = sum((y - my - slope * (x - mx)) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    return slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def interventional(n_train: int, n_test: int, steps: int, device) -> dict:
    """The corpus, its CPython oracle, and the three-arm ladder that scores it."""
    data = corpus.build(n_train, n_test, seed=0)
    scored = arms.run_all(data, device, steps=steps, seed=0)
    return dict(held_out=list(data["held_out"]), n_train=n_train, n_test=n_test,
                steps=steps, predict_the_mean=1.0,
                arms={k: dict(train_nrmse=v["train_nrmse"],
                              ood_nrmse=v["ood_nrmse"],
                              n_params=v["n_params"]) for k, v in scored.items()})


def report(*, fast: bool = False, device=None) -> dict:
    dev = device or torch.device("cpu")
    return dict(
        device=dev.type,
        n_draws=FAST_DRAWS if fast else FULL_DRAWS,
        sign_flip_rate=sign_flip_table(FAST_DRAWS if fast else FULL_DRAWS, dev),
        decay=decay_curve(FAST_DECAY_DRAWS if fast else FULL_DECAY_DRAWS, dev),
        interventional=interventional(256 if fast else 2048,
                                      128 if fast else 512,
                                      120 if fast else 400, dev),
    )


def render(r: dict) -> str:
    out = [f"sign_flip_rate -- can a THIRD token flip the sign of j's influence "
           f"on i?  ({r['n_draws']} draws, device={r['device']})",
           f"  {'operator':<12}" + "".join(f"{'depth=' + d:>10}" for d in
                                           sorted(r["sign_flip_rate"]["softmax"]))]
    for k, row in r["sign_flip_rate"].items():
        out.append(f"  {k:<12}" + "".join(f"{row[d]:>10.4f}"
                                          for d in sorted(row)))
    out += ["",
            "  softmax is the control and reads exactly 0.0: a non-negative "
            "operator composed",
            "  with fixed value projections factors as (weight) x (matrix), so "
            "no third token",
            "  can move a sign at any depth.",
            "",
            f"context decay of that rate -- sgate, depth 1, hops {DECAY_HOPS}, "
            f"{r['decay']['n_draws']} draws, i=s-1 j=s/4 c=s/2"]
    out.append("  " + "".join(f"{'s=' + s:>10}" for s in
                              sorted(r["decay"]["rate"], key=int)))
    out.append("  " + "".join(f"{r['decay']['rate'][s]:>10.5f}" for s in
                              sorted(r["decay"]["rate"], key=int)))
    d = r["decay"]
    #: There is no published exponent to compare against -- it was withdrawn,
    #: and printing "None / None" would read as a missing value rather than a
    #: deliberate one. Say which.
    out += [f"  log-log slope {d['slope']:.3f}  R2 {d['r2']:.4f}   "
            f"(no published exponent: WITHDRAWN as a floor=1e-6 artifact; "
            f"{FULL_DECAY_DRAWS} draws)",
            f"  extrapolated to s=2048: {d['extrapolated_at_2048']:.2e}"
            if d["extrapolated_at_2048"] else "",
            "",
            "  READ THE RATE WITH THIS CURVE OR NOT AT ALL. The property is "
            "carried by the",
            "  k>=2 term of J = sum_k A^k, where a third token is 1 of ~s "
            "intermediates, so",
            "  the property and its dilution are the same fact. A rate quoted "
            "at s=8 says",
            "  nothing about the context your model runs at.",
            "",
            f"interventional code corpus -- CPython is the oracle, held-out "
            f"cell {tuple(r['interventional']['held_out'])}",
            f"  {r['interventional']['n_train']} train / "
            f"{r['interventional']['n_test']} test, "
            f"{r['interventional']['steps']} steps",
            f"  {'arm':<12}{'train_nrmse':>14}{'ood_nrmse':>12}{'params':>10}"]
    for k, v in r["interventional"]["arms"].items():
        out.append(f"  {k:<12}{v['train_nrmse']:>14.4f}{v['ood_nrmse']:>12.4f}"
                   f"{v['n_params']:>10}")
    out += [f"  {'predict-the-mean':<12}{'':>14}"
            f"{r['interventional']['predict_the_mean']:>12.4f}",
            "  nrmse is normalized by the target's own spread, so 1.0 IS "
            "predict-the-mean.",
            "  an arm at or above 1.0 learned nothing, however it ranks against "
            "the others."]
    return "\n".join(x for x in out if x is not None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m ceq.diagnose",
                                description=__doc__.splitlines()[4].strip())
    p.add_argument("--fast", action="store_true",
                   help="fewer draws; a smoke run, not the calibration")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    a = p.parse_args(argv)
    r = report(fast=a.fast, device=torch.device(a.device))
    print(json.dumps(r) if a.json else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
