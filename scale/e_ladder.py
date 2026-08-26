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
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.bucket import Journal                                   # noqa: E402
from scale.m3_quintuple import NAME, _key                          # noqa: E402
from scale.m3_synthetic_settled import contrast                    # noqa: E402
from scale import negation_scope as NS                             # noqa: E402

RUNGS = ("e3_t1", "e3_t2", "e3_t8", "e3_t32")

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
                k = _key(dict(cfg, cell=cell, seed=sd, task=task))
                if k in done:
                    out.setdefault((task, cell), {})[sd] = done[k]["value"]
    return out


def read(journal=None, *, seeds=(0, 1, 2, 3, 4), **cfg) -> dict:
    """The curve. Returns a dict; prints nothing."""
    conf = dict(k=8, s=64, d=24, steps=150, n_train=2048, n_eval=2048,
                t_max=21, n_neumann=21)
    conf.update(cfg)
    rows = _rows(journal, ("settled", "twin"), seeds, conf)

    ladder = []
    for task in RUNGS:
        t_star = NS.E_T_STAR[task](conf["s"])
        st, tw = rows.get((task, "settled")), rows.get((task, "twin"))
        if not st or not tw or any(s not in st or s not in tw for s in seeds):
            have = 0 if not st else len(st), 0 if not tw else len(tw)
            ladder.append(dict(task=task, t_star=t_star, state="NOT RUN",
                               have_settled=have[0], have_twin=have[1]))
            continue
        ev_s = [st[s]["eval_nrmse"] for s in seeds]
        ev_t = [tw[s]["eval_nrmse"] for s in seeds]
        c = contrast(ev_t, ev_s, n_boot=10000, seed=0)          # +ve = settled
        dd = c["per_seed_delta"]
        m = sum(dd) / len(dd)
        sd_ = (sum((v - m) ** 2 for v in dd) / (len(dd) - 1)) ** 0.5
        mean_s, mean_t = sum(ev_s) / len(ev_s), sum(ev_t) / len(ev_t)
        ladder.append(dict(
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
    return dict(ladder=ladder, seeds=list(seeds), config=conf,
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
    row, why = verdict(cur)
    print(f"\n    ladder complete: {cur['complete']}   "
          f"pre-registered 13-seed resolution: {RESOLUTION_13:.6f}")
    print(f"\n=== PRE-REGISTERED ROW {row} ===\n    {why}")
    if a.json:
        print("\n" + json.dumps(dict(cur, row=row, why=why), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
