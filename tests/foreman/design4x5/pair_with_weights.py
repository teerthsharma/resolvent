# -*- coding: utf-8 -*-
"""One CRN-matched (a, f) pair at split_seed 0, trained WITH the weights kept.

Why this exists: `r3_eval.train_with_eval` discarded every model it trained, and
`design4x5.py` inherited that, so twenty grid cells land twenty loss numbers and
zero checkpoints. Any analysis that needs the weights -- per-token abstention,
row mass on the arm that actually produced the headline number -- has nothing to
read. `save_model` now exists on the trainer; this runs the one pair that the
abstention test requires, reusing `design4x5.train_arm` unchanged so the arms,
the CRN order, the step budget and the eval subsample are identical to the grid.

It polls for a free card and never kills anything it finds.
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import design4x5 as D  # noqa: E402
import r3_eval as RE  # noqa: E402

SPLIT_SEED = 0
SEED = 0
ARMS = ("a", "f")
OUT = os.path.join(HERE, "pair_with_weights.jsonl")
BOARD = os.path.join(D.REPO if hasattr(D, "REPO") else
                     r"C:\Users\seal\Desktop\New folder (32)", "house-events.jsonl")


def board(**kw):
    kw.setdefault("agent", "Chase")
    try:
        with io.open(BOARD, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw) + "\n")
    except Exception:
        pass


_ORIG_TWE = RE.train_with_eval


def _saving_train_with_eval(**kw):
    """Force the new save_model flag on, and write to a directory of this run's
    own so a grid cell's run_record.json is never overwritten."""
    kw["save_model"] = True
    kw["out_dir"] = kw["out_dir"] + "_pair"
    return _ORIG_TWE(**kw)


def landed():
    if not os.path.exists(OUT):
        return set()
    return {json.loads(l)["arm"] for l in io.open(OUT, encoding="utf-8") if l.strip()}


def main():
    waited = D.poll_until_free(timeout_s=3 * 3600, interval_s=30)
    if not waited:
        print("[PAIR] card never freed within 3h -- not starting", flush=True)
        board(t="finding", item="pair_not_started",
              text="card busy for 3h; matched pair not trained")
        return 1

    crn = D.load_crn()
    n_f, n_f0, n_a, n_a2 = D.measure_params()
    steps = max(1, round(20.0 * n_a / (D.BATCH * D.SEQ)))
    print("[PAIR] steps={} n_a={} n_f={}".format(steps, n_a, n_f), flush=True)

    RE.train_with_eval = _saving_train_with_eval
    try:
        done = landed()
        for arm in ARMS:
            if arm in done:
                print("[PAIR] {} already landed, skipping".format(arm), flush=True)
                continue
            t0 = time.time()
            rec = D.train_arm(arm, seed=SEED, split_seed=SPLIT_SEED,
                              steps=steps, crn=crn)
            row = dict(arm=arm, split_seed=SPLIT_SEED, seed=SEED,
                       final_eval_loss=rec.get("final_eval_loss"),
                       n_params=rec.get("n_params"),
                       model_path=rec.get("model_path"),
                       crn_digest=rec.get("crn_digest"),
                       seconds=round(time.time() - t0, 1))
            with io.open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print("[PAIR] " + json.dumps(row), flush=True)
            board(t="test", state="GREEN", name="pair_{}_saved".format(arm),
                  detail="{} loss={}".format(row["model_path"],
                                             row["final_eval_loss"]))
    finally:
        RE.train_with_eval = _ORIG_TWE

    board(t="finding", item="matched_pair_saved",
          text="(a,f) at split_seed={} now exist on disk with weights".format(SPLIT_SEED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
