"""Chase: C25 (4) -- the ceiling question. When the ceiling is computed by the
BINNED functional (c25_scorer, all three schemes) instead of the unbinned
rollout-variance estimator, how much of the unbinned ceiling survives?

REUSE: rollout generation (generate_selfplay_game, _outcome_onehot,
rollout_result, boards_of) imported straight from Wilson's validated
wil_res_ceiling.py -- only the aggregation step (bin the same phat array
three ways instead of taking its raw variance) is new, and the unbinned
number is recomputed here with the SAME imported functions specifically to
cross-check against the published 0.10116955630126778 before anything
downstream is trusted.

L-REPRO
  seed        : random.Random(0), single stream, consumed in order --
                identical seeding rule to wil_res_ceiling.measure.
  scale       : N_GAMES=60, R=16, max_plies=400 -- reduced from Wilson's
                150 games for wall-clock budget inside this scratchpad-only
                task; flagged as reduced-scale, not a replacement for the
                published number.
  dtype       : float64, CPU only, no torch for the rollouts; c25_scorer's
                float64 torch path for the binned RES.
  command     : python c25_ceiling.py
"""
import io, json, random, sys, datetime
import numpy as np

REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + r"\tests\wilson\resolution")
from wil_res_ceiling import rollout_result, boards_of, SEED, R  # noqa: E402
from ceqjepa.beds.chess import generate_selfplay_game, _outcome_onehot, OUTCOME_NAMES  # noqa: E402

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sys.path.insert(0, SCRATCH)
from c25_scorer import score_forecasts, SCHEMES  # noqa: E402

BOARD_PATH = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
PUBLISHED_UNBINNED_RES = 0.10116955630126778
PUBLISHED_UNBINNED_CI95 = [0.0708, 0.1329]


def board_log(event, **kw):
    rec = dict(agent="Chase", ts=datetime.datetime.utcnow().isoformat() + "Z", event=event, **kw)
    with io.open(BOARD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def collect(n_games, max_plies):
    rng = random.Random(SEED)
    marg, phat = [], []
    for _ in range(n_games):
        g = generate_selfplay_game(rng, max_plies=max_plies)
        marg.append(_outcome_onehot(g.headers.get("Result", "*")))
        bs = boards_of(g)
        b = bs[rng.randrange(len(bs))]
        acc = np.zeros(4)
        for _r in range(R):
            acc += _outcome_onehot(rollout_result(b, rng, max_plies))
        phat.append(acc / R)
    return np.asarray(marg, dtype=np.float64), np.asarray(phat, dtype=np.float64)


def unbinned_res(marg, phat):
    """Exact recomputation of wil_res_ceiling.measure's RES formula, for the
    cross-check gate."""
    var_phat = phat.var(0, ddof=1)
    noise = (phat * (1 - phat)).mean(0) / (R - 1)
    return float((var_phat - noise).sum())


def main(n_games=60, max_plies=400):
    board_log("start", task="c25_ceiling", n_games=n_games, max_plies=max_plies)
    marg, phat = collect(n_games, max_plies)
    unbinned = unbinned_res(marg, phat)
    print(f"[gate] unbinned RES recomputed here: {unbinned:.6f} (n_games={n_games}, "
          f"reduced scale) vs published 0.101170 (n_games=150). "
          f"{'within published CI95 range order' if PUBLISHED_UNBINNED_CI95[0]/2 < unbinned < PUBLISHED_UNBINNED_CI95[1]*2 else 'OUT OF RANGE -- investigate before trusting the binned comparison'}")

    k_star = marg.argmax(1)  # marg is already one-hot per game
    binned = {}
    for s in SCHEMES:
        out = score_forecasts(phat, k_star, 4, s, class_names=OUTCOME_NAMES)
        binned[s] = out["RES"]
        frac = out["RES"] / PUBLISHED_UNBINNED_RES
        print(f"  {s:16s} binned ceiling RES = {out['RES']:.6f}  "
              f"({frac*100:.1f}% of published unbinned {PUBLISHED_UNBINNED_RES:.6f})")

    record = dict(n_games=n_games, max_plies=max_plies, R=R,
                   unbinned_res_recomputed_here=unbinned,
                   published_unbinned_res=PUBLISHED_UNBINNED_RES,
                   binned_ceiling_res=binned,
                   fraction_of_published_unbinned={s: binned[s] / PUBLISHED_UNBINNED_RES for s in SCHEMES})
    with open(SCRATCH + r"\c25_ceiling_results.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    board_log("c25_ceiling_done", unbinned=unbinned, binned=binned,
              fraction_of_published=record["fraction_of_published_unbinned"])
    return record


if __name__ == "__main__":
    main()
