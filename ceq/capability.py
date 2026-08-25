"""The capability run: exact-match scores, a matched control, a published number.

    python -m ceq.capability

WHY THIS AND NOT ANOTHER VAL-LOSS DECIMAL. `arXiv:2605.20798` measured what a
val-loss ratio is worth at the scale this campaign is aimed at: two significant
failures land within 2-3% of baseline validation loss and still drop 6-16
CLIMB-points, and 1.2B improver rank predicts 3B improver rank at Spearman
rho = -0.27. The 3.34% parity target sits inside that band. An exact-match
capability score is not a better version of the loss ratio; it is a different
kind of evidence, and the only kind left that means anything here.

WHAT IT WRITES. `results/capability.json`, one entry per split, carrying every
per-seed number and the aggregate, the published reference with its citation,
the in-distribution gate, the truncation ceiling, and a verdict recomputed from
the pre-registered bars. `tests/cameron/test_capability_result.py` recomputes
that verdict from the same numbers, so a report cannot drift from its own data.

THE VERDICT CAN BE A LOSS. `no_win` is a real outcome of this file and is
written without hedging. A negative result on the benchmark chosen precisely
because it could see the property is worth more than another val-loss decimal.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

from . import harness

# Pre-registered in `tests/cameron/test_cogs_harness.py` before any COGS number
# was measured, and restated here so the runner and the check share one value.
# `test_the_runner_uses_the_bars_that_were_pre_registered` fails if they drift.
RESOLUTION_FLOOR = 0.20   # the softmax control must clear this in-distribution
WIN_MARGIN = 0.02         # absolute exact-match margin sgate must take

ARMS = ("softmax", "sgate")
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results" / "capability.json"

# Per-split budget. COGS is a 24,155-item training split with logical forms out
# to 480 tokens; SCAN addprim_jump is 14,670 items of at most 48 actions, so the
# same step count buys far more epochs there.
BUDGET = {
    "cogs": dict(steps=3000, bs=32, lr=3e-4, max_eval=512, max_new=192,
                 gate_eval=256),
    "addprim_jump": dict(steps=4000, bs=32, lr=3e-4, max_eval=512, max_new=64,
                         gate_eval=0),
}


def score_ceiling(split: str, which: str, max_eval: int, max_new: int,
                  seed: int) -> float:
    """Fraction of the SCORED items whose gold target fits inside `max_new`.

    Greedy decoding stops at `max_new`, so an item with a longer gold sequence
    is wrong whatever the model emits. Identical for both arms, so it cannot
    favour either -- but it caps the achievable score, and a number reported
    without its ceiling reads as a comparison against the published 0.35 when
    it is not one. Computed over the same subsample `harness.score` uses, not
    over the whole split, because that is what was actually scored.
    """
    pairs = harness.load_pairs(split, which)
    pick = harness.eval_indices(len(pairs), max_eval, seed)
    return sum(len(pairs[i][1]) <= max_new for i in pick) / len(pick)


def verdict(gate: dict | None, arms: dict) -> str:
    """`gated` when the control cannot learn the task at this budget, otherwise
    the sign of the margin against WIN_MARGIN. Arithmetic, not a sentence."""
    if gate is not None and gate["softmax"] < RESOLUTION_FLOOR:
        return "gated"
    margin = arms["sgate"]["exact_match"] - arms["softmax"]["exact_match"]
    return "sgate_wins" if margin >= WIN_MARGIN else "no_win"


def run_split(split: str, *, device, seeds=(0, 1, 2), **over) -> dict:
    """Both arms, every seed, one identical budget. The operator is the only
    free variable: same steps, lr, batch, seed, data, eval subsample."""
    b = dict(BUDGET[split], **over)
    which = harness.EVAL_WHICH.get(split, "test")
    per_seed, gate_seed = [], []
    t0 = time.time()
    for seed in seeds:
        row, gate = {}, {}
        for kind in ARMS:
            m = harness.train_model(kind, split, steps=b["steps"], device=device,
                                    seed=seed, bs=b["bs"], lr=b["lr"])
            if b["gate_eval"]:
                gate[kind] = harness.score(m, split, "test", device=device,
                                           max_eval=b["gate_eval"], seed=seed,
                                           max_new=b["max_new"])["exact_match"]
            row[kind] = harness.score(m, split, which, device=device,
                                      max_eval=b["max_eval"], seed=seed,
                                      max_new=b["max_new"])
            print(f"  {split} seed={seed} {kind:<8} "
                  f"{which}={row[kind]['exact_match']:.4f} "
                  f"gate={gate.get(kind, float('nan')):.4f} "
                  f"train_loss={row[kind]['train_loss']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        per_seed.append(row)
        if gate:
            gate_seed.append(gate)

    med = lambda k, f: statistics.median(f(r[k]) for r in per_seed)
    arms = {k: dict(per_seed[0][k],
                    exact_match=med(k, lambda a: a["exact_match"]),
                    n_solved=None,
                    per_seed=[r[k]["exact_match"] for r in per_seed])
            for k in ARMS}
    agg_gate = ({k: statistics.median(g[k] for g in gate_seed) for k in ARMS}
                | dict(seed=list(seeds), steps=b["steps"], n_eval=b["gate_eval"],
                       split="test")
                if gate_seed else None)
    return dict(
        split=split, steps=b["steps"], seed=list(seeds), bs=b["bs"], lr=b["lr"],
        device=str(device).split(":")[0], max_new=b["max_new"],
        max_eval=b["max_eval"], eval_split=which, arms=arms,
        in_distribution=agg_gate,
        score_ceiling=score_ceiling(split, which, b["max_eval"], b["max_new"],
                                    seeds[0]),
        reference=harness.REFERENCES[(split, "transformer_from_scratch")],
        altered_attention=harness.REFERENCES.get((split, "altered_attention")),
        verdict=verdict(agg_gate, arms),
        comparison="parameter-matched from-scratch, NOT frontier-model",
        wall_clock_s=round(time.time() - t0, 1),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m ceq.capability")
    p.add_argument("--split", action="append", choices=sorted(BUDGET),
                   help="repeatable; default is every split")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int)
    p.add_argument("--max-eval", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(RESULTS))
    a = p.parse_args(argv)

    over = {k: v for k, v in (("steps", a.steps), ("max_eval", a.max_eval))
            if v is not None}
    seeds = tuple(int(s) for s in a.seeds.split(","))
    dev = torch.device(a.device)
    out = pathlib.Path(a.out)
    # merge rather than overwrite, so `--split cogs` does not delete SCAN
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for split in (a.split or sorted(BUDGET)):
        existing[split] = run_split(split, device=dev, seeds=seeds, **over)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        print(f"{split}: {existing[split]['verdict']}  "
              f"softmax {existing[split]['arms']['softmax']['exact_match']:.4f}  "
              f"sgate {existing[split]['arms']['sgate']['exact_match']:.4f}  "
              f"published {existing[split]['reference']['score']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
