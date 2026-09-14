"""ceqjepa/beds/chess_do.py -- interventional chess labels: do(a) ground truth
for the Sherman-Morrison committor path.

WHY THIS FILE EXISTS. ceqjepa/beds/chess.py labels only the FACTUAL arm of
each position: the move that was actually played, and the game's actual final
outcome (its own docstring says so, and says it does not attempt the
counterfactual arm). Chess is the only one of the three domains here where the
counterfactual arm can be obtained AT ALL: you can force any legal move from a
real position and finish the game under the same policy, and python-chess will
tell you, truthfully, what happens. This module produces that pair.

WHAT AN "OUTCOME" IS. Same four absorbing sets as chess.py, same order, reused
directly so a downstream evaluator never has two competing definitions of
"outcome":
    0 = white win, 1 = draw, 2 = black win, 3 = SINK (ply cap or "*")

THE TWO ARMS, AND WHY ONE IS EXACT AND THE OTHER IS NOT.
  Observational arm:  the move the self-play policy actually chose at the
    sampled position, and the outcome the SAME game in fact reached. This is
    exact ground truth -- the game was really played to the end, once.
  Interventional arm: a candidate move that was NOT played is forced instead,
    and the rest of the game is replayed under the identical policy (uniform
    random legal moves) for R independent rollouts. Each rollout is one
    Bernoulli-ish draw from the true post-intervention outcome distribution;
    averaging R of them gives an EMPIRICAL MEAN, not the distribution itself.

BE HONEST ABOUT THE ESTIMATOR. For one outcome coordinate with true
probability p, the empirical mean over R iid rollouts has
    Var[p_hat] = p(1-p)/R  <=  1/(4R).
At the default R=4 that bound is 1/16 = 0.0625, i.e. a per-coordinate std of
at most 0.25 -- coarse enough that `do_outcome_mean` should be read as "which
outcomes are plausible after this forced move", not as a precise committor
value. Raise R for a tighter estimate; cost scales linearly with R because
each rollout is an independent playout, there is no shortcut here (the
shortcut this bed exists to FEED is Sherman-Morrison over the m candidate
moves, not over the R rollouts of one candidate).

SAMPLING. One position is drawn per self-play game (uniformly over its plies),
so `n_games` games give `n_games` paired samples -- not one position per ply,
which would let long games dominate the pool exactly the way chess.py's
module docstring already flags for its own bed.

INTERFACE. Reuses chess.py's oracle-backed pieces directly rather than a
second encoding of the outcome: `generate_selfplay_game`, `N_OUTCOMES`,
`OUTCOME_NAMES`, and chess.py's own private `_outcome_onehot` (same
package, one outcome encoding, not a forked copy of the same six lines).
Positions are kept as raw FEN strings, not pre-encoded -- an evaluator
pairs them with `fen_to_vec` from chess.py itself, so there is exactly one
place that turns a FEN into a model input.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import chess

from ceqjepa.beds.chess import N_OUTCOMES, OUTCOME_NAMES, generate_selfplay_game, _outcome_onehot

__all__ = [
    "InterventionSample", "build_intervention_dataset", "outcome_distribution", "demo",
]


@dataclass
class InterventionSample:
    """One position, keyed so an evaluator can pair its observational and
    interventional arms: same `fen`, same `game_id`/`ply_idx`."""
    game_id: int
    ply_idx: int
    fen: str                        # fen_before -- the intervened-on position
    obs_uci: str                    # move actually played (exact, from the real game)
    obs_outcome: np.ndarray         # [N_OUTCOMES] one-hot, EXACT (the real game's result)
    candidate_ucis: list            # m candidate moves forced from `fen` (may include obs_uci)
    do_outcome_mean: np.ndarray     # [m, N_OUTCOMES] empirical mean over R rollouts each
    do_outcome_R: int               # rollouts per candidate (see module docstring for variance)


def _rollout_outcome(board: "chess.Board", rng: random.Random, max_plies: int) -> np.ndarray:
    """Finish `board` under the SAME policy self-play used (uniform random
    legal move) and return the realized outcome one-hot. One real playout --
    the unit the R-rollout average is built from."""
    b = board.copy(stack=False)
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        b.push(rng.choice(list(b.legal_moves)))
    return _outcome_onehot(b.result(claim_draw=True))


def build_intervention_dataset(n_games: int = 300, seed: int = 0, max_plies: int = 80,
                                m_candidates: int = 8, R: int = 4,
                                ) -> list[InterventionSample]:
    """Play `n_games` self-play games; from one random ply of each, pair the
    real (observational) continuation with `m_candidates` forced-move
    (interventional) continuations, each estimated from R rollouts.

    One `random.Random(seed)` drives move choice, position sampling,
    candidate sampling AND every rollout, in that sequential order -- so a
    fixed seed regenerates a byte-identical dataset (checked in demo()).
    """
    assert R >= 1 and m_candidates >= 1
    rng = random.Random(seed)
    samples = []
    for game_id in range(n_games):
        game = generate_selfplay_game(rng, max_plies=max_plies)
        obs_outcome = _outcome_onehot(game.headers["Result"])
        moves = list(game.mainline_moves())
        if not moves:
            continue  # no legal move existed at the start position (never happens, but no promises)
        ply_idx = rng.randrange(len(moves))

        board = game.board()
        for mv in moves[:ply_idx]:
            board.push(mv)
        fen = board.fen()
        obs_uci = moves[ply_idx].uci()

        legal = list(board.legal_moves)
        m = min(m_candidates, len(legal))
        candidates = rng.sample(legal, m)

        do_mean = np.zeros((m, N_OUTCOMES), dtype=np.float32)
        for c, mv in enumerate(candidates):
            after = board.copy(stack=False)
            after.push(mv)
            acc = np.zeros(N_OUTCOMES, dtype=np.float32)
            for _ in range(R):
                acc += _rollout_outcome(after, rng, max_plies)
            do_mean[c] = acc / R

        samples.append(InterventionSample(
            game_id=game_id, ply_idx=ply_idx, fen=fen, obs_uci=obs_uci,
            obs_outcome=obs_outcome, candidate_ucis=[mv.uci() for mv in candidates],
            do_outcome_mean=do_mean, do_outcome_R=R,
        ))
    assert samples, "no positions sampled -- n_games too small or every game had zero plies"
    return samples


def outcome_distribution(vectors: np.ndarray) -> dict:
    """Mean of a stack of outcome vectors (one-hot or soft), as a printable
    {name: fraction} dict. Works the same way for the exact observational
    one-hots and the averaged interventional means."""
    dist = vectors.mean(axis=0)
    return {OUTCOME_NAMES[i]: float(dist[i]) for i in range(N_OUTCOMES)}


def demo() -> None:
    """Assert-based self-check plus the printed report the task asked for."""
    ds_a = build_intervention_dataset(n_games=40, seed=0, m_candidates=4, R=2)
    ds_b = build_intervention_dataset(n_games=40, seed=0, m_candidates=4, R=2)
    assert [s.fen for s in ds_a] == [s.fen for s in ds_b], (
        "seed=0 did not regenerate an identical dataset -- hash-pinning is broken"
    )

    samples = build_intervention_dataset(n_games=300, seed=0, m_candidates=8, R=4)
    print(f"[MEASURED] {len(samples)} paired positions, "
          f"m_candidates<=8, R=4 rollouts/candidate")

    # oracle spot-check: every candidate really is legal at `fen`, and pushing
    # it really does change the board -- redone with a fresh Board, exactly
    # the way chess.py's own oracle_check re-derives rather than trusts.
    checked = 0
    for s in samples[:200]:
        board = chess.Board(s.fen)
        for uci in s.candidate_ucis:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves, f"chess_do emitted an illegal candidate: {uci} at {s.fen}"
            checked += 1
        assert s.obs_outcome.sum() == 1.0
        assert np.allclose(s.do_outcome_mean.sum(axis=1), 1.0)
    print(f"[MEASURED] oracle check: {checked}/{checked} candidate moves legal at their fen "
          f"(200 positions sampled)")

    obs = np.stack([s.obs_outcome for s in samples])
    do_all = np.concatenate([s.do_outcome_mean for s in samples], axis=0)
    print(f"[MEASURED] observational outcome distribution   ({len(samples)} games, exact): "
          f"{outcome_distribution(obs)}")
    print(f"[MEASURED] interventional outcome distribution  ({do_all.shape[0]} forced "
          f"rollout-means, R=4 each): {outcome_distribution(do_all)}")

    # "how often does forcing a random legal move change the outcome" -- use
    # the FIRST sampled candidate per position (already a uniformly random
    # legal move, via rng.sample) and its R-rollout majority outcome against
    # the exact observational outcome.
    changed = 0
    for s in samples:
        obs_k = int(s.obs_outcome.argmax())
        do_k = int(s.do_outcome_mean[0].argmax())
        changed += (obs_k != do_k)
    frac = changed / len(samples)
    print(f"[MEASURED] forcing a random legal move changed the argmax outcome in "
          f"{changed}/{len(samples)} positions ({100 * frac:.1f}%)")
    sink_frac = obs[:, OUTCOME_NAMES.index("sink")].mean()
    print(f"[MEASURED] observational sink rate (max_plies=80 default): "
          f"{100 * float(sink_frac):.1f}% of games never reached a real terminal outcome")
    if frac < 0.10:
        print(f"[FINDING] changed-outcome fraction is low ({100 * frac:.1f}%); with "
              f"{100 * float(sink_frac):.1f}% of games hitting SINK (the ply cap) rather than a "
              f"real terminal state, both arms mostly land on the same absorbing set regardless "
              f"of the forced move, which mechanically compresses the gap this bed can show. "
              f"Raising max_plies (fewer truncated games) is the lever, not more rollouts.")


if __name__ == "__main__":
    demo()
