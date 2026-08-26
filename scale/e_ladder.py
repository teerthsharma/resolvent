"""LADDER E — the settled-twin dose-response across the `t*` rungs. One curve.

    python -m scale.e_ladder

WHY A SECOND READER AND NOT A FLAG ON THE FIRST. `scale/m3_quintuple.py` prints
contrasts WITHIN one run, and one run is one task. The deciding object here is
the contrast READ ACROSS tasks -- how `settled - twin` moves as the label's own
hop count `t*` goes 1, 2, 8, 32 -- and that spans four invocations. Nothing in
this repo printed across the bucket's task axis, so this file does, and it does
nothing else: every number it shows is read out of
`results/m3_quintuple_v2.jsonl`, none is computed by training anything.

THE ONE CREDITABLE CONTRAST, AND WHY THE OTHERS ARE NOT PRINTED AS VERDICTS.
`LOOP_PROMPT.md` 1.7d: every `e3_t*` is labelled by `equilibrium_oracle`, the
signed path sum for which `e1_anchor` is a declared rigged demo, and the settled
arm's own resolvent computes that object. So `settled - softmax` on any rung is
the arm reproducing its own forward and is credited nothing. Credit flows
through `settled - twin` alone -- identical mixture, identical parameter
tensors, differing only in whether the fixed point is iterated.

THE OUTCOME TABLE IS NOT IN THIS FILE'S HEAD. It is in
`E_LADDER_PREREGISTERED_READING.md` section 6, written before any `e3` number
existed, and `OUTCOMES` below is a transcription of it. The constant `0.027260`
is the pre-registered 13-seed resolution computed from the `negation_scope`
pilot SD BEFORE the data landed; it is deliberately NOT recomputed from the
realised spread, because a threshold refitted to the data it judges is not a
threshold. The realised spread is printed beside it instead.

PARTIAL LADDERS. Rows A, C and F quantify over EVERY rung. A ladder missing a
rung cannot license them, and a missing rung is printed NOT RUN, never as a
null -- the whole reason for `bucket.require_complete`.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.bucket import Journal                                   # noqa: E402
from scale.m3_quintuple import NAME, _key                          # noqa: E402
from scale.m3_synthetic_settled import contrast                    # noqa: E402
from scale import negation_scope as NS                             # noqa: E402

RUNGS = ("e3_t1", "e3_t2", "e3_t8", "e3_t32")

#: How many hops each cell's forward can reach. `softmax` forms `x + A@x`, one
#: hop. The pivot cells add a hop-2 term, and `settled` settles the pivot
#: WEIGHTS rather than the token chain, so it too reads a two-hop
#: neighbourhood -- settling does not buy a third hop and must not be scored as
#: though it did.
HOP_BUDGET = {"softmax": 1, "glance": 1, "settled": 2, "twin": 2, "argmax": 2}

#: The cells whose journal key carries `k0` rather than the pivot count.
K0_CELLS = ("softmax", "glance")


def ceiling(t_star: int, hops: int) -> float:
    """`sqrt((t* - hops) / t*)` -- the best NRMSE a `hops`-budget arm can reach.

    The label is a sum of `t*` independent equal-variance path terms
    (`make_equilibrium_batch` draws Rademacher coefficients inside the band, so
    every path weight has modulus 1). An arm that can see `hops` of them leaves
    `t* - hops` unexplained, and NRMSE is the residual standard deviation over
    the label's, hence the square root of the variance ratio. Clamped at 0
    because a budget at or beyond `t*` expresses the label exactly.

    CHECKED, NOT ASSUMED. Against `negation_scope.e_hop_reading` on drawn
    batches of 2048 at the ladder's own eval seed, the closed form is high by at
    most `+0.012088` and typically under `+0.008`:

        e3_t2  k=1  formula 0.707107  measured 0.719195
        e3_t8  k=2  formula 0.866025  measured 0.873949
        e3_t32 k=2  formula 0.968246  measured 0.969735

    `test_the_ceiling_formula_matches_the_drawn_truncation_reading` re-takes
    that comparison. NOTE the stale trap: `results/m3_capability.txt:1212`
    records `e3_t8 k=2 NRMSE=0.815162`, which does NOT agree -- that log
    predates `b[:, s-1] = 0.0` in `make_equilibrium_batch` and describes a
    corpus that no longer exists.
    """
    if t_star <= 0:
        raise ValueError(t_star)
    return math.sqrt(max(0.0, float(t_star - hops)) / float(t_star))

#: E_LADDER_PREREGISTERED_READING.md section 5: `1.96 * 0.050146 / sqrt(13)`,
#: the half-width the section-1.8 e-process floor of 13 seeds would buy at the
#: pilot SD. Fixed before the data landed. Not refitted.
RESOLUTION_13 = 0.027260

#: The absolute bar. `negation_scope.nrmse` is exactly 1.0 for the mean
#: predictor, so a cell at or above it beat nothing and a contrast between two
#: such cells is a contrast between two failures.
PREDICT_THE_MEAN = 1.0


def _rows(journal: pathlib.Path, cells, seeds, cfg) -> dict:
    """{(task, cell): {seed: value}} for whatever is journalled. Missing units
    are simply absent -- this function never invents one."""
    done = Journal(NAME).done() if journal is None else {
        json.loads(l)["key"]: json.loads(l)
        for l in journal.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = {}
    for task in RUNGS:
        for cell in cells:
            for sd in seeds:
                kk = 0 if cell in K0_CELLS else cfg["k"]
                k = _key(dict(cfg, cell=cell, seed=sd, task=task, k=kk))
                if k in done:
                    out.setdefault((task, cell), {})[sd] = done[k]["value"]
    return out


def _means(rows, task, seeds) -> dict:
    """Seed-mean eval NRMSE per cell at one rung, over the seeds ACTUALLY
    present. Partial means are reported with their count rather than withheld,
    because f needs no pairing -- but the count travels with the number so a
    two-seed mean is never read as a five-seed one."""
    out = {}
    for cell in ("settled", "twin", "softmax"):
        got = rows.get((task, cell))
        if not got:
            continue
        ev = [got[s]["eval_nrmse"] for s in seeds if s in got]
        if ev:
            out[cell] = sum(ev) / len(ev)
            out[cell + "_n"] = len(ev)
    return out


def read(journal=None, *, seeds=(0, 1, 2, 3, 4), **cfg) -> dict:
    """The curve. Returns a dict; prints nothing."""
    conf = dict(k=8, s=64, d=24, steps=150, n_train=2048, n_eval=2048,
                t_max=21, n_neumann=21)
    conf.update(cfg)
    rows = _rows(journal, ("settled", "twin", "softmax"), seeds, conf)

    ladder = []
    for task in RUNGS:
        t_star = NS.E_T_STAR[task](conf["s"])
        st, tw = rows.get((task, "settled")), rows.get((task, "twin"))
        if not st or not tw or any(s not in st or s not in tw for s in seeds):
            have = 0 if not st else len(st), 0 if not tw else len(tw)
            # A rung with no COMPLETE pair still has cells worth scoring: f is
            # per cell and needs no pairing, so a partial rung reports the f it
            # can and withholds only the contrast.
            ladder.append(dict(task=task, t_star=t_star, state="NOT RUN",
                               have_settled=have[0], have_twin=have[1],
                               means=_means(rows, task, seeds)))
            continue
        ev_s = [st[s]["eval_nrmse"] for s in seeds]
        ev_t = [tw[s]["eval_nrmse"] for s in seeds]
        c = contrast(ev_t, ev_s, n_boot=10000, seed=0)          # +ve = settled
        dd = c["per_seed_delta"]
        m = sum(dd) / len(dd)
        sd_ = (sum((v - m) ** 2 for v in dd) / (len(dd) - 1)) ** 0.5
        mean_s, mean_t = sum(ev_s) / len(ev_s), sum(ev_t) / len(ev_t)
        ladder.append(dict(
            means=_means(rows, task, seeds),
            task=task, t_star=t_star, state="RUN", delta=c["delta"],
            ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], sd_paired=sd_,
            verdict=c["verdict"], per_seed_delta=dd,
            settled_mean=mean_s, twin_mean=mean_t,
            seeds_favouring_settled=sum(1 for v in dd if v > 0.0),
            # G: a rung where either cell failed to beat predict-the-mean is
            # credited NOTHING in either direction.
            credited=(mean_s < PREDICT_THE_MEAN and mean_t < PREDICT_THE_MEAN),
            excludes_zero=(c["ci_lo"] > 0.0 or c["ci_hi"] < 0.0),
        ))
    # f AND g -- THE AUTHOR'S METHOD: name the part that will not solve as a
    # function, solve everything around it exactly, and let the surroundings
    # report what the function is doing.
    #
    #   f(t*, cell) := achieved(t*, cell) - ceiling(t*, HOP_BUDGET[cell])
    #   g(t*)       := softmax(t*) - settled(t*)
    #
    # `ceiling` is exact and `achieved` is measured, so f is fully determined by
    # the ladder already running, at ZERO extra cost. Its SHAPE is the
    # diagnosis: flat in t* => a fixed readout overhead, the readout binds;
    # growing in t* => the arm degrades as the target recedes, the budget binds;
    # near zero low and jumping at t*=8 => a threshold, and its location is
    # itself the finding.
    #
    # f IS PER CELL AND EACH CELL IS SCORED AGAINST ITS OWN BUDGET. Scoring
    # one-hop `softmax` against the two-hop ceiling would credit it with a
    # shortfall it is not structurally able to close, which is the same error as
    # crediting `settled` with a third hop.
    shortfall, gaps = [], []
    for r in ladder:
        t = r["t_star"]
        for cell in ("settled", "twin", "softmax"):
            m = r.get("means", {}).get(cell)
            if m is None:
                continue
            c = ceiling(t, HOP_BUDGET[cell])
            shortfall.append(dict(task=r["task"], t_star=t, cell=cell,
                                  hops=HOP_BUDGET[cell], achieved=m,
                                  ceiling=c, f=m - c))
        sm = r.get("means", {}).get("softmax")
        st = r.get("means", {}).get("settled")
        tw = r.get("means", {}).get("twin")
        if sm is not None and st is not None:
            gaps.append(dict(task=r["task"], t_star=t, softmax=sm, settled=st,
                             g=sm - st,
                             g_twin=None if tw is None else sm - tw))
    return dict(shortfall=shortfall, gaps=gaps,
                ladder=ladder, seeds=list(seeds), config=conf,
                resolution_13=RESOLUTION_13,
                complete=all(r["state"] == "RUN" for r in ladder),
                all_credited=all(r.get("credited") for r in ladder
                                 if r["state"] == "RUN"))


def verdict(cur: dict) -> tuple[str, str]:
    """Which pre-registered row fired. Returns (row, sentence).

    Ordered so that a two-directional failure is reported as a theory death
    rather than being split into two single-rung readings.
    """
    live = {r["task"]: r for r in cur["ladder"]
            if r["state"] == "RUN" and r["credited"]}
    uncredited = [r["task"] for r in cur["ladder"]
                  if r["state"] == "RUN" and not r["credited"]]
    if uncredited:
        return ("G", "rungs {} have a cell at or above predict-the-mean and "
                     "are credited nothing in either direction: {}"
                .format(len(uncredited), ", ".join(uncredited)))

    def won(t):      # settled beats twin, CI excludes zero
        r = live.get(t)
        return r is not None and r["ci_lo"] > 0.0

    def lost(t):
        r = live.get(t)
        return r is not None and r["ci_hi"] < 0.0

    deep_lost = lost("e3_t8") or lost("e3_t32")
    if won("e3_t1") and deep_lost:
        return ("F", "K-2E FIRES as written at LOOP_PROMPT.md:400 -- settled "
                     "LOSES on a deep-t* rung AND WINS at t*=1. Theory death.")
    if won("e3_t1"):
        return ("E", "G4 SHAPE -- the contrast is nonzero at t*=1, inside every "
                     "arm's hop budget, so it is capacity leakage, not "
                     "settling. The claim degrades to routing-only whatever "
                     "t*=32 says.")
    if won("e3_t32"):
        deltas = [live[t]["delta"] for t in RUNGS if t in live]
        mono = all(b >= a for a, b in zip(deltas, deltas[1:]))
        if not cur["complete"]:
            return ("B", "a single-rung effect at t*=32; the ladder is partial "
                         "so monotonicity is not testable and row A is not "
                         "available.")
        return (("A", "DOSE-RESPONSE. Compatible with zero at t*=1, excludes "
                      "zero and positive at t*=32, and monotone "
                      "non-decreasing across the rungs.")
                if mono else
                ("B", "PARTIAL -- settled wins at t*=32 but the ladder is not "
                      "monotone in t*, so this is a single-rung effect, not a "
                      "dose-response."))
    # H, added to E_LADDER_PREREGISTERED_READING.md at 13:35 before this file
    # read a number: settled only ever LOSES. Rows A/B/E/F all condition on
    # settled winning somewhere and C/D on every CI covering zero, so a
    # strictly-losing ladder fell through the whole table.
    if any(lost(t) for t in RUNGS) and not any(won(t) for t in RUNGS):
        lo = [t for t in RUNGS if lost(t)]
        return ("H", "SETTLING IS A STRICT COST -- the CI excludes zero and is "
                     "NEGATIVE at {} and is positive nowhere. Not a tie: the "
                     "iteration measurably hurts at matched parameters. K-3E's "
                     "conclusion holds a fortiori and the twin ships."
                .format(", ".join(lo)))

    # every credited rung's CI covers zero
    if not cur["complete"]:
        return ("--", "no rung excludes zero, but the ladder is PARTIAL: rows "
                      "A, C and F quantify over every rung and none of them is "
                      "available. No kill may be claimed.")
    big = [t for t in ("e3_t8", "e3_t32")
           if t in live and abs(live[t]["delta"]) >= RESOLUTION_13]
    if big:
        return ("D", "UNDERPOWERED, NOT A KILL -- every CI covers zero, but "
                     "|delta| at {} is at or above the 13-seed resolution "
                     "{:.6f}, so LOOP_PROMPT.md 1.8's 'too small to matter' "
                     "clause does not apply. Route: 5 -> 13 seeds at {} only."
                .format(", ".join(big), RESOLUTION_13, ", ".join(big)))
    return ("C", "K-3E FIRES (LOOP_PROMPT.md:403) -- E3 is solved EQUALLY by "
                 "the twin at matched parameters on its own home terrain, and "
                 "every |delta| is below the 13-seed resolution {:.6f}, so "
                 "1.8's 'too small to matter at this scale' applies. Settling "
                 "retires; the twin ships."
            .format(RESOLUTION_13))


def cur_n(cur: dict, task: str, cell: str) -> int:
    """Seeds behind one cell's mean at one rung."""
    for r in cur["ladder"]:
        if r["task"] == task:
            return r.get("means", {}).get(cell + "_n", 0)
    return 0


def print_f(cur: dict) -> None:
    """f(t*) := achieved - ceiling(t*, hops), per cell, against its OWN budget.

    THE AUTHOR'S METHOD, applied to the hop wall. The part that would not solve
    -- "why do the arms underperform" -- is named as a function; everything
    around it is exact (the ceiling is closed form at every rung, the achieved
    value is measured at every rung already being run); so f is fully determined
    at ZERO extra cost and its SHAPE is the diagnosis.
    """
    print()
    print("=== f(t*) := achieved - ceiling(t*, hops),  "
          "ceiling = sqrt((t*-hops)/t*) ===")
    print("    Each cell scored against ITS OWN budget: softmax 1 hop, pivot "
          "cells 2. n = seeds in the mean.")
    if not cur["shortfall"]:
        print("    (no cell has a reading yet)")
        return
    print(f"{'task':>8} {'t*':>4} {'cell':>9} {'hops':>5} {'n':>3} "
          f"{'achieved':>10} {'ceiling':>10} {'f':>11}")
    for r in cur["shortfall"]:
        print(f"{r['task']:>8} {r['t_star']:>4} {r['cell']:>9} "
              f"{r['hops']:>5} {cur_n(cur, r['task'], r['cell']):>3} "
              f"{r['achieved']:>10.6f} {r['ceiling']:>10.6f} "
              f"{r['f']:>+11.6f}")
    by = {}
    for r in cur["shortfall"]:
        by.setdefault(r["cell"], []).append((r["t_star"], r["f"]))
    for cell, pts in sorted(by.items()):
        pts.sort()
        vals = [v for _, v in pts]
        if len(pts) < 3:
            print(f"    {cell:>9}: {len(pts)} rung(s) -- SHAPE NOT READABLE. "
                  f"Three rungs are needed to separate flat from growing from "
                  f"a threshold. f = "
                  + ", ".join(f"t*={t}:{v:+.6f}" for t, v in pts))
            continue
        spread = max(vals) - min(vals)
        rising = all(b >= a for a, b in zip(vals, vals[1:]))
        shape = ("FLAT in t* -- a fixed overhead; THE READOUT BINDS, not the "
                 "budget" if spread < 0.05 else
                 "GROWING in t* -- the arm degrades as the target recedes; "
                 "THE BUDGET BINDS" if rising else
                 "NON-MONOTONE -- neither a fixed overhead nor a clean "
                 "degradation; report the shape, name no cause")
        print(f"    {cell:>9}: spread {spread:.6f} over "
              f"t*={[t for t, _ in pts]} -> {shape}")


def print_g(cur: dict) -> None:
    """g(t*) := softmax(t*) - settled(t*). NEGATIVE means softmax is better."""
    print()
    print("=== g(t*) := softmax(t*) - settled(t*)   "
          "(NRMSE; NEGATIVE means softmax has the LOWER error) ===")
    if not cur["gaps"]:
        print("    NOT AVAILABLE: no rung yet has both a softmax and a "
              "settled reading in this journal.")
        return
    print(f"{'task':>8} {'t*':>4} {'softmax':>10} {'settled':>10} "
          f"{'g':>11} {'g_twin':>11}")
    for r in cur["gaps"]:
        gt = "--" if r["g_twin"] is None else f"{r['g_twin']:+.6f}"
        print(f"{r['task']:>8} {r['t_star']:>4} {r['softmax']:>10.6f} "
              f"{r['settled']:>10.6f} {r['g']:>+11.6f} {gt:>11}")
    print("    g SHRINKING as t* grows => settling buys something that only "
          "shows at depth.")
    print("    g flat or growing            => it does not.")


def print_scope() -> None:
    """The distinction Foreman corrected, printed BESIDE the result.

    An e3 loss is not a failure of the round's central prediction, and the two
    must not be allowed to blur into one sentence in a later summary.
    """
    print()
    print("SCOPE OF ANY e3 RESULT -- IT IS NOT THE ROUND'S PREDICTION.")
    print("    e3_t* binds equilibrium_oracle, still the signed path sum the "
          "resolvent computes")
    print("    (scale/negation_scope.py::M3_TASKS). The lambda2 / Cheeger "
          "prediction lives on the")
    print("    ABSORBING-CHAIN corpus, which is NOT REGISTERED YET, so it has "
          "never been tested here.")
    print("    A settled loss on this ladder is a statement about e3, NOT "
          "about the lambda2 theory.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    cur = read(seeds=tuple(a.seeds), n_train=a.n_train, n_eval=a.n_eval)

    print("=== LADDER E -- settled minus twin across the t* ladder ===")
    print(f"    journal results/{NAME}.jsonl   seeds {a.seeds}   "
          f"n_train={a.n_train} n_eval={a.n_eval}")
    print("    Pre-registration: E_LADDER_PREREGISTERED_READING.md")
    print("    CONVENTION: delta = mean(NRMSE_twin - NRMSE_settled). "
          "POSITIVE MEANS SETTLED WINS.")
    print("    RIG CAVEAT, every row: e3_t* is labelled by equilibrium_oracle, "
          "the object the")
    print("    ceq resolvent itself computes, so settled-vs-softmax is VOID "
          "here (LOOP_PROMPT.md 1.7d).")
    print("    Only this contrast is creditable on this ladder.\n")
    print(f"{'task':>8} {'t*':>4} {'delta':>10} {'ci_lo':>10} {'ci_hi':>10} "
          f"{'sd_paired':>10} {'settled':>9} {'twin':>9} {'n+':>3}  verdict")
    for r in cur["ladder"]:
        if r["state"] != "RUN":
            print(f"{r['task']:>8} {r['t_star']:>4} {'NOT RUN':>10} "
                  f"{'--':>10} {'--':>10} {'--':>10} {'--':>9} {'--':>9} "
                  f"{'--':>3}  settled {r['have_settled']}/{len(a.seeds)} "
                  f"seeds, twin {r['have_twin']}/{len(a.seeds)}")
            continue
        print(f"{r['task']:>8} {r['t_star']:>4} {r['delta']:>+10.6f} "
              f"{r['ci_lo']:>+10.6f} {r['ci_hi']:>+10.6f} "
              f"{r['sd_paired']:>10.6f} {r['settled_mean']:>9.6f} "
              f"{r['twin_mean']:>9.6f} {r['seeds_favouring_settled']:>3}  "
              f"{r['verdict']}"
              + ("" if r["credited"] else
                 "  [G: a cell is at/above predict-the-mean, CREDITED NOTHING]"))
    print_f(cur)
    print_g(cur)
    print_scope()
    row, why = verdict(cur)
    print(f"\n    ladder complete: {cur['complete']}   "
          f"pre-registered 13-seed resolution: {RESOLUTION_13:.6f}")
    print(f"\n=== PRE-REGISTERED ROW {row} ===\n    {why}")
    if a.json:
        print("\n" + json.dumps(dict(cur, row=row, why=why), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
