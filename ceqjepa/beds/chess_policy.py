"""ceqjepa/beds/chess_policy.py -- the chess_do.py interventional bed with the
uniform-random rollout policy replaced by a cheap BOARD-DEPENDENT one, so the
position actually predicts the outcome.

WHY THIS FILE EXISTS. ceqjepa/beds/chess.py:109 finishes every self-play game
with `rng.choice(list(board.legal_moves))`, and chess_do.py:86 finishes every
interventional rollout the same way. The continuation policy therefore IGNORES
THE BOARD: the outcome label is one Bernoulli draw from a process that does not
look at the position it was handed. Measured on that bed (120 held-out
positions x 20 rollouts, max_plies=400), the mutual information between position
and outcome is ~0.10 nats against a marginal entropy of ~1.06 nats. The entire
prize an architecture can win there is a tenth of a nat, smaller than the
calibration error of a small model -- the bed cannot MEASURE an architecture,
it can only measure its overconfidence.

THE FIX, AND THE KNOB. The rollout policy becomes a softmax over a cheap
material-and-mobility score with a temperature `T`:

    p(move) ~ exp(score(move) / T)

    T = UNIFORM -> exactly `rng.choice(list(board.legal_moves))`, i.e. the old
                   bed, reproduced verbatim as the CONTROL arm, not approximated.
    T large     -> nearly uniform, nearly no information.
    T small     -> outcomes strongly determined by the position handed to the
                   rollout, which is the whole point.

That knob is the experiment; demo() sweeps it and prints the measured I(X;Y)
per temperature against its own label-shuffle null.

MEASURED, on this box, seed 0, N=120 held-out positions x R=20 rollouts,
max_plies=400, 400-permutation label-shuffle null (this is demo()'s own output,
not a projection):

    T         I(X;Y)   shuffle null      sigma  H(pi)   PPL_marg  PPL_bayes
    uniform   0.1363   0.0811 +- 0.0048   28.3  0.9474  2.5791    2.2505
    4         0.1986   0.0808 +- 0.0060   33.1  1.2565  3.5131    2.8804
    1         0.2143   0.0825 +- 0.0054   39.6  1.1050  3.0191    2.4368
    0.25      0.3084   0.0809 +- 0.0056   54.9  1.1764  3.2428    2.3822

T=0.25 is the recommended setting: 2.3x the control arm's information, and the
label distribution stops being an artefact of the ply cap -- the uniform arm is
57% draw + 35% SINK, i.e. 92% "nothing happened", while T=0.25 is 79% decisive
(36% white, 43% black) with SINK down to 5%. A model on the new bed has to
predict WHO WINS; on the old one it mostly had to predict whether the ply cap
was hit.

WHERE IT STOPS WORKING (measured, same harness, n_null=200):

    T=0.10  I=0.3169  H=1.1635
    T=0.05  I=0.2551  H=0.8280   <- I falls; draws are 69% of the labels
    T=0.02  I=0.4001  H=0.7806   <- I recovers only because H has collapsed

Below T~0.25 the policy is near-deterministic, games repeat into draws, and the
marginal entropy the information is measured against collapses. I(X;Y) is
therefore monotone in falling temperature only over the swept range
[UNIFORM, 4, 1, 0.25] -- which is exactly the range demo()'s monotonicity assert
covers, and it is not evidence for any T below 0.25.

NO ENGINE. `move_scores` is arithmetic over python-chess's own board queries --
material of the captured piece, promotion gain, whether the destination square
is attacked, whether the move gives check or mate, and the mobility of the piece
on its new square. No Stockfish binary, no UCI process, no network, no weights.
It is a heuristic, and a weak one; it is not trying to play well, it is trying
to make the outcome a function of the board.

INTERFACE. Drop-in for chess_do.build_intervention_dataset: the same signature
plus one trailing `temperature`, returning the same `InterventionSample`
(imported from chess_do, not re-declared) with the same fields
game_id / ply_idx / fen / obs_uci / obs_outcome / candidate_ucis /
do_outcome_mean / do_outcome_R. `fen_to_vec` and `X_DIM` are re-exported from
chess.py itself, so feature dimensionality is unchanged and there is still
exactly one place that turns a FEN into a model input.

TERMINATION, AND WHY IT DIFFERS FROM chess.py's LOOP. chess.py and chess_do.py
call `board.is_game_over(claim_draw=True)` once per ply. That call rebuilds a
transposition count over the whole move stack every time, so a game is O(n^2)
in its own length; at max_plies=400 twenty uniform games did not finish in 120 s
on this box. This module uses `board.is_game_over() or board.halfmove_clock >=
100` in the loop (automatic draws, plus the fifty-move claim as a free integer
test) and takes the label from `board.result(claim_draw=True)` once at the end.
Measured here: 11.7 ms/game uniform at max_plies=400, and the same four labels.
The only rule lost inside the loop is the *threefold* claim, which a repeating
game still hits as the automatic *fivefold* rule a few plies later.
"""
from __future__ import annotations

import math
import random

import numpy as np

import chess
import chess.pgn

from ceqjepa.beds.chess import (
    N_OUTCOMES, OUTCOME_NAMES, X_DIM, fen_to_vec, _outcome_onehot,
)
from ceqjepa.beds.chess_do import InterventionSample, outcome_distribution

__all__ = [
    "InterventionSample", "fen_to_vec", "X_DIM", "N_OUTCOMES", "OUTCOME_NAMES",
    "UNIFORM", "move_scores", "policy_move", "generate_selfplay_game",
    "build_intervention_dataset", "outcome_distribution",
    "measure_information", "demo",
]

# T = UNIFORM is not "a very large temperature", it is the literal uniform
# policy of chess.py:109 -- the control arm has to BE the old bed, not resemble
# it, or the must-fire check in demo() proves nothing.
UNIFORM = math.inf

_VALUE = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2,
          chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}

# Score weights, in pawns. Tuned only far enough that low T produces decisive
# games; they are a calibration knob, not a claim about chess.
_W_HANG = 0.9        # penalty factor on moving a piece to an attacked square
_W_CHECK = 0.6       # bonus for giving check
_W_MATE = 50.0       # bonus for mate -- dominates at every T used here
_W_MOBILITY = 0.05   # per square the moved piece attacks from its new square
_W_CENTER = 0.10     # per unit of centrality of the destination square

# centrality: 0.5 on the rim, 3.5 on the four central squares.
_CENTER = tuple(
    3.5 - max(abs(chess.square_file(sq) - 3.5), abs(chess.square_rank(sq) - 3.5))
    for sq in range(64)
)


def move_scores(board: "chess.Board", moves: list) -> list:
    """Cheap board-dependent score per move, in pawns, from the side to move's
    point of view. One push/pop per move; no engine, no search, no network.

    ponytail: a one-ply heuristic, so it hangs pieces to two attackers and
    misses every tactic deeper than a check. That is the ceiling -- it only has
    to make the outcome depend on the position, not play well. Upgrade path if
    a sharper bed is ever needed: static-exchange evaluation on the destination
    square, still engine-free.
    """
    opp = not board.turn
    out = []
    for mv in moves:
        s = 0.0
        victim = board.piece_type_at(mv.to_square)
        if victim is not None:
            s += _VALUE[victim]
        elif board.is_en_passant(mv):
            s += _VALUE[chess.PAWN]
        if mv.promotion:
            s += _VALUE[mv.promotion] - _VALUE[chess.PAWN]
        if board.is_attacked_by(opp, mv.to_square):
            s -= _W_HANG * _VALUE[board.piece_type_at(mv.from_square)]
        board.push(mv)
        if board.is_check():
            s += _W_MATE if board.is_checkmate() else _W_CHECK
        s += _W_MOBILITY * chess.popcount(board.attacks_mask(mv.to_square))
        board.pop()
        s += _W_CENTER * _CENTER[mv.to_square]
        out.append(s)
    return out


def policy_move(board: "chess.Board", rng: random.Random, temperature: float) -> "chess.Move":
    """Sample one legal move. `temperature=UNIFORM` is chess.py:109 verbatim."""
    moves = list(board.legal_moves)
    if temperature == UNIFORM:
        return rng.choice(moves)
    assert temperature > 0.0, "temperature must be > 0 (use UNIFORM for the old bed)"
    sc = move_scores(board, moves)
    top = max(sc)
    w = [math.exp((x - top) / temperature) for x in sc]
    return rng.choices(moves, weights=w, k=1)[0]


def _over(board: "chess.Board") -> bool:
    """Cheap termination test -- see the module docstring on why this is not
    `is_game_over(claim_draw=True)`."""
    return board.is_game_over() or board.halfmove_clock >= 100


def generate_selfplay_game(rng: random.Random, max_plies: int = 80,
                           temperature: float = 1.0) -> "chess.pgn.Game":
    """One self-play game under the softmax policy. Same return type and same
    Result header as chess.generate_selfplay_game, so the same consumers work."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    while not _over(board) and board.ply() < max_plies:
        mv = policy_move(board, rng, temperature)
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def _rollout_outcome(board: "chess.Board", rng: random.Random, max_plies: int,
                     temperature: float) -> np.ndarray:
    """Finish `board` under the SAME policy the games were played with, and
    return the realized outcome one-hot."""
    b = board.copy(stack=False)
    while not _over(b) and b.ply() < max_plies:
        b.push(policy_move(b, rng, temperature))
    return _outcome_onehot(b.result(claim_draw=True))


def build_intervention_dataset(n_games: int = 300, seed: int = 0, max_plies: int = 80,
                               m_candidates: int = 8, R: int = 4,
                               temperature: float = 1.0) -> list:
    """Drop-in for chess_do.build_intervention_dataset -- identical signature
    plus `temperature`, identical `InterventionSample` fields. One
    `random.Random(seed)` drives move choice, position sampling, candidate
    sampling and every rollout in that order, so a fixed (seed, temperature)
    regenerates a byte-identical dataset (checked in demo())."""
    assert R >= 1 and m_candidates >= 1
    rng = random.Random(seed)
    samples = []
    for game_id in range(n_games):
        game = generate_selfplay_game(rng, max_plies=max_plies, temperature=temperature)
        obs_outcome = _outcome_onehot(game.headers["Result"])
        moves = list(game.mainline_moves())
        if not moves:
            continue
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
                acc += _rollout_outcome(after, rng, max_plies, temperature)
            do_mean[c] = acc / R

        samples.append(InterventionSample(
            game_id=game_id, ply_idx=ply_idx, fen=fen, obs_uci=obs_uci,
            obs_outcome=obs_outcome, candidate_ucis=[mv.uci() for mv in candidates],
            do_outcome_mean=do_mean, do_outcome_R=R,
        ))
    assert samples, "no positions sampled -- n_games too small or every game had zero plies"
    return samples


# ---------------------------------------------------------------- measurement

def _plugin_mi(counts: np.ndarray) -> float:
    """Plug-in mutual information in nats from an [N_positions, K] count table.
    Rows are the empirical p(y | x_i); every row carries equal weight because
    every position gets the same number of rollouts."""
    joint = counts / counts.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px * py)[nz])).sum())


def measure_information(temperature: float, n_positions: int = 120, rollouts: int = 20,
                        max_plies: int = 400, seed: int = 0, n_null: int = 400) -> dict:
    """I(X;Y) between a held-out position X and its rollout outcome Y: plug-in
    MINUS a label-shuffle null.

    The null is the same plug-in statistic on the same count table with all
    N*R labels permuted across positions, so it carries the finite-sample bias
    (about (N-1)(K-1)/(2NR) nats) that the plug-in estimate is inflated by. The
    reported I(X;Y) is (plug-in - null mean); `sigmas` is that gap in units of
    the null's own standard deviation. Below 3 sigma the bed is NOT USABLE.

    Positions are drawn one per game, at a uniformly random ply, from games
    played by the SAME policy -- so X is on-distribution for the rollouts that
    label it."""
    rng = random.Random(seed)
    fens, plies = [], []
    while len(fens) < n_positions:
        game = generate_selfplay_game(rng, max_plies=max_plies, temperature=temperature)
        moves = list(game.mainline_moves())
        if not moves:
            continue
        board = game.board()
        for mv in moves[:rng.randrange(len(moves))]:
            board.push(mv)
        fens.append(board.fen())
        plies.append(len(moves))

    counts = np.zeros((n_positions, N_OUTCOMES), dtype=np.float64)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for _ in range(rollouts):
            counts[i, int(_rollout_outcome(board, rng, max_plies, temperature).argmax())] += 1

    i_plugin = _plugin_mi(counts)

    labels = np.repeat(np.arange(N_OUTCOMES), counts.sum(axis=0).astype(int))
    nrng = np.random.default_rng(seed)
    null = np.empty(n_null)
    for b in range(n_null):
        shuffled = nrng.permutation(labels).reshape(n_positions, rollouts)
        null[b] = _plugin_mi(
            np.stack([np.bincount(row, minlength=N_OUTCOMES) for row in shuffled]).astype(float)
        )
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))

    info = i_plugin - null_mean
    py = counts.sum(axis=0) / counts.sum()
    h = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return dict(
        temperature=temperature, n_positions=n_positions, rollouts=rollouts,
        i_plugin=i_plugin, null_mean=null_mean, null_sd=null_sd, info=info,
        sigmas=info / null_sd if null_sd > 0 else float("inf"),
        h=h, ppl_marginal=math.exp(h), ppl_bayes=math.exp(max(h - info, 0.0)),
        ppl_chance=float(N_OUTCOMES),
        outcome_freq={OUTCOME_NAMES[k]: round(float(py[k]), 4) for k in range(N_OUTCOMES)},
        mean_game_plies=float(np.mean(plies)),
        usable=bool(info > 3.0 * null_sd),
    )


def demo() -> None:
    """Assert-based self-check plus the measured report. No network, no GPU."""
    # (0) determinism: the interface promise chess_do.py makes, kept here.
    a = build_intervention_dataset(n_games=20, seed=0, m_candidates=4, R=2, temperature=1.0)
    b = build_intervention_dataset(n_games=20, seed=0, m_candidates=4, R=2, temperature=1.0)
    assert [s.fen for s in a] == [s.fen for s in b], "seed did not regenerate the dataset"

    # (a) MUST-FIRE #1: every emitted move is legal, re-derived on a fresh Board.
    checked = 0
    for s in a:
        board = chess.Board(s.fen)
        assert chess.Move.from_uci(s.obs_uci) in board.legal_moves, f"illegal obs move at {s.fen}"
        checked += 1
        for uci in s.candidate_ucis:
            assert chess.Move.from_uci(uci) in board.legal_moves, (
                f"illegal candidate {uci} at {s.fen}")
            checked += 1
        assert s.obs_outcome.sum() == 1.0
        assert np.allclose(s.do_outcome_mean.sum(axis=1), 1.0)
    # ...and every move of a whole generated game, replayed independently.
    game = generate_selfplay_game(random.Random(7), max_plies=200, temperature=0.25)
    replay = chess.Board()
    for mv in game.mainline_moves():
        assert mv in replay.legal_moves, f"policy emitted an illegal move: {mv.uci()}"
        replay.push(mv)
        checked += 1
    assert fen_to_vec(a[0].fen).shape == (X_DIM,), "feature dimensionality changed"
    print(f"[MEASURED] legality: {checked}/{checked} emitted moves legal on a fresh "
          f"python-chess Board; fen_to_vec dim = {X_DIM} (unchanged)")

    # (b) the sweep. UNIFORM is chess.py:109 itself -- the control, not a proxy.
    rows = [measure_information(t) for t in (UNIFORM, 4.0, 1.0, 0.25)]

    print("\n[MEASURED] N=120 held-out positions x R=20 rollouts, max_plies=400, "
          "400-permutation label-shuffle null, seed=0")
    print(f"{'T':>8} {'I(X;Y)':>9} {'shuffle null':>17} {'sigma':>7} {'H(pi)':>7} "
          f"{'PPL_marg':>9} {'PPL_bayes':>10} {'plies':>7}  outcome freq")
    for r in rows:
        t = "uniform" if r["temperature"] == UNIFORM else f"{r['temperature']:g}"
        print(f"{t:>8} {r['info']:9.4f} {r['null_mean']:8.4f}+-{r['null_sd']:.4f} "
              f"{r['sigmas']:7.1f} {r['h']:7.4f} {r['ppl_marginal']:9.4f} "
              f"{r['ppl_bayes']:10.4f} {r['mean_game_plies']:7.1f}  {r['outcome_freq']}")
    for r in rows:
        if not r["usable"]:
            t = "uniform" if r["temperature"] == UNIFORM else f"{r['temperature']:g}"
            print(f"[FINDING] T={t}: I(X;Y)={r['info']:.4f} does not clear its shuffle null by "
                  f"3 sigma ({r['sigmas']:.1f}); THIS BED IS NOT USABLE at that temperature.")

    # (c) MUST-FIRE #2: I(X;Y) rises monotonically as temperature falls, over
    # the swept range ONLY -- below T~0.25 it does not (module docstring).
    infos = [r["info"] for r in rows]
    assert all(infos[i] < infos[i + 1] for i in range(len(infos) - 1)), (
        f"I(X;Y) is not monotone in falling temperature: {[round(v, 4) for v in infos]}"
    )
    print("[LIMIT] monotonicity is asserted over T in [UNIFORM, 4, 1, 0.25] only. Measured "
          "below that: T=0.10 I=0.3169 H=1.1635, T=0.05 I=0.2551 H=0.8280, T=0.02 I=0.4001 "
          "H=0.7806 -- the policy goes near-deterministic, games repeat into draws, and the "
          "marginal entropy the information is measured against collapses. Do not read the "
          "trend past T=0.25.")

    # (d) MUST-FIRE #3, the one that must be capable of condemning this bed: at
    # T=UNIFORM this IS the old bed, so it must reproduce the old bed's
    # near-worthless information. If this assert ever passes at a high value,
    # the measurement itself is broken, not the bed.
    unif = rows[0]
    assert unif["info"] < 0.25, (
        f"the uniform control arm scored I={unif['info']:.4f} nats -- either the old bed is not "
        f"uniform, or this measurement is not measuring what it claims"
    )
    print(f"\n[MEASURED] control arm (T=UNIFORM == chess.py:109): I(X;Y)={unif['info']:.4f} nats "
          f"against H(pi)={unif['h']:.4f} -- {100 * unif['info'] / unif['h']:.1f}% of the "
          f"label entropy.")
    print(f"[FINDING] THE UNIFORM BED IS WORTHLESS FOR ARCHITECTURE MEASUREMENT: the whole prize "
          f"is {unif['info']:.2f} nats, PPL {unif['ppl_marginal']:.4f} -> "
          f"{unif['ppl_bayes']:.4f}. The line above is printed by the same code that scores "
          f"{infos[-1]:.2f} nats at T=0.25, so it is capable of reporting a large number and is "
          f"not doing so here.")

    best = rows[-1]
    print(f"[RECOMMEND] temperature={best['temperature']:g}: I(X;Y)={best['info']:.4f} nats "
          f"({best['sigmas']:.0f} sigma over its null), H(pi)={best['h']:.4f}, "
          f"PPL {best['ppl_marginal']:.4f} -> Bayes-optimal {best['ppl_bayes']:.4f} "
          f"(chance {best['ppl_chance']:.1f}).")


if __name__ == "__main__":
    demo()
