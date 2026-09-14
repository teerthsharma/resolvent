"""ceqjepa/beds/chess.py -- phase-1 (CHESS) bed for the chess -> English ->
stocks/prediction-markets curriculum. This is the FIRST phase; TinyCEQ
trains on this bed before its checkpoint warm-starts phase 2 (see
ceqjepa/beds/english.py).

REUSES ceq.kdata's PGN machinery rather than re-implementing it:
  - ceq.kdata.label_plies(game) IS THE ORACLE. It replays a chess.pgn.Game
    on a real python-chess Board and recomputes fen_before / uci /
    fen_after / legal straight from the board, never from a stored column
    (ceq/kdata.py:258-281). This module calls it on every game, self-play
    or PGN-sourced alike -- there is no second, hand-rolled legality path.
  - ceq.kdata.iter_games(path) is the documented path to scale: point it at
    an attached PGN (e.g. Kaggle arevel/chess-games) and `ChessBed.from_pgn`
    drives the same label_plies oracle over it.

SOURCE. DEFAULT is self-play generated locally with python-chess -- moves
drawn uniformly from board.legal_moves by a seeded RNG, no file, no
network. "Hash-pinned" here means seed-pinned: the same seed regenerates
byte-identical games (checked in demo()). `ChessBed.from_pgn` is the
documented, tested-but-not-invoked path to Kaggle's arevel/chess-games PGN
dump for scale -- not called by default, per the no-network-in-a-first-cell
constraint.

ABSORBING SETS. The committor targets are the three terminal game outcomes
plus a declared sink, in this fixed order:
    0 = white win, 1 = draw, 2 = black win,
    3 = SINK (the game hit the ply cap, or python-chess reported the
        undecided "*" result -- no real absorbing state was reached inside
        the sampling budget)

q_star IS A REALIZATION, NOT A VALUE. Every ply of one game carries the
SAME one-hot outcome vector: the game's final result via board.result(),
not a per-position value estimate. The effective sample size for the
committor signal is therefore GAMES, not plies -- an 80-ply game
contributes one outcome bit stretched across 80 rows, and batch() samples
plies uniformly, so long games are resampled more than short ones. Stated
here plainly, not left implicit.

x / x_nx encoding is a flat 769-float board vector: 12 piece-type planes
over the 64 squares (768) plus one side-to-move bit, built straight from
the FEN with python-chess's own piece_map() -- the model reads the same
board the oracle validated, not a second encoding of it. x is fen_before,
x_nx is fen_after: the position after the move that was ACTUALLY played.
A PGN (and this generator) labels only that one factual arm per
fen_before -- see the header note on Sherman-Morrison do(a): this bed does
not attempt to supervise counterfactual arms, and does not claim to.

v_idx is the ply index of the sampled position within its own game (0 =
the opening position). Unlike the synthetic Bed's v_idx (an index into one
shared n-vertex chain), it carries no meaning across games and there is no
closed-form q_floor keyed on it -- it is returned only so the batch()
tuple shape matches train.py's Bed, making this bed a drop-in.

INTERFACE PARITY with ceqjepa/train.py's synthetic Bed:
    batch(gen, B) -> (x[B,769] f32, x_nx[B,769] f32, q_star[B,4] f32, v_idx[B] long)
"""
from __future__ import annotations

import random

import numpy as np
import torch

import chess
import chess.pgn

from ceq.kdata import iter_games, label_plies

OUTCOME_NAMES = ("white_win", "draw", "black_win", "sink")
N_OUTCOMES = len(OUTCOME_NAMES)
X_DIM = 12 * 64 + 1  # 768 piece planes + side-to-move

_PIECE_PLANE = {  # (piece_type, color) -> plane index 0..11
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def fen_to_vec(fen: str) -> np.ndarray:
    """769-float board encoding, built with python-chess -- the same
    library the oracle uses, so there is no second, divergent notion of
    what a square holds."""
    board = chess.Board(fen)
    v = np.zeros(X_DIM, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        v[_PIECE_PLANE[(piece.piece_type, piece.color)] * 64 + sq] = 1.0
    v[-1] = 1.0 if board.turn == chess.WHITE else 0.0
    return v


def _outcome_onehot(result: str) -> np.ndarray:
    idx = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}.get(result, 3)  # "*" or anything else -> SINK
    v = np.zeros(N_OUTCOMES, dtype=np.float32)
    v[idx] = 1.0
    return v


def generate_selfplay_game(rng: random.Random, max_plies: int = 80) -> chess.pgn.Game:
    """One random-legal-move self-play game, seeded by `rng`. No network,
    no PGN file: python-chess is both the move source and the oracle."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv = rng.choice(list(board.legal_moves))
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def _rows_from_game(game: "chess.pgn.Game") -> list[dict]:
    onehot = _outcome_onehot(game.headers.get("Result", "*"))
    rows = []
    for ply_idx, ply in enumerate(label_plies(game)):
        assert ply["legal"], f"bed emitted an illegal move: {ply}"
        rows.append(dict(fen_before=ply["fen_before"], fen_after=ply["fen_after"],
                          uci=ply["uci"], outcome=onehot, ply_idx=ply_idx))
    return rows


class ChessBed:
    """Drop-in for ceqjepa.train.Bed. Build once (a fixed pool of labelled
    plies from a fixed set of games), then batch() samples plies from it."""

    n = N_OUTCOMES        # every absorbing set IS an outcome; there is no
    nA = N_OUTCOMES       # separate transient chart the way the synthetic Bed has one
    x_dim = X_DIM

    def __init__(self, rows: list[dict]):
        assert rows, "no plies to sample from"
        self.rows = rows
        # outcome_counts is PER GAME, not per ply -- every ply of a game shares
        # its outcome (module docstring), so counting every row would inflate
        # a long game's vote by its ply count. ply_idx==0 marks a game's first
        # (opening) row, one per game by construction (_rows_from_game).
        self.outcome_counts = {k: 0 for k in OUTCOME_NAMES}
        for r in rows:
            if r["ply_idx"] == 0:
                self.outcome_counts[OUTCOME_NAMES[int(r["outcome"].argmax())]] += 1

    @classmethod
    def build(cls, n_games: int = 200, seed: int = 0, max_plies: int = 80) -> "ChessBed":
        """DEFAULT source: local self-play, no network, no download. Seed-pinned:
        the same `seed` regenerates the identical game set (see demo())."""
        rng = random.Random(seed)
        rows = []
        for _ in range(n_games):
            rows.extend(_rows_from_game(generate_selfplay_game(rng, max_plies=max_plies)))
        return cls(rows)

    @classmethod
    def from_pgn(cls, path) -> "ChessBed":
        """SCALE path: point at an attached PGN (e.g. Kaggle arevel/chess-games)
        and reuse ceq.kdata.iter_games + label_plies exactly as `build` does
        for self-play. Not called by default -- the HARD CONSTRAINT is a
        first cell with no network -- but real, not a promise: exercised by
        tests/gate0/fixtures/games.pgn in this module's own test."""
        rows = []
        for _key, game in iter_games(path):
            rows.extend(_rows_from_game(game))
        return cls(rows)

    def batch(self, gen: torch.Generator, B: int):
        """Drop-in for train.py's Bed.batch: (x, x_nx, q_star, v_idx).
        x / x_nx float32 [B, X_DIM]; q_star float32 [B, N_OUTCOMES] one-hot
        REALIZATION (module docstring); v_idx long [B] ply index."""
        idx = torch.randint(len(self.rows), (B,), generator=gen).tolist()
        x = np.stack([fen_to_vec(self.rows[i]["fen_before"]) for i in idx])
        x_nx = np.stack([fen_to_vec(self.rows[i]["fen_after"]) for i in idx])
        q_star = np.stack([self.rows[i]["outcome"] for i in idx])
        v_idx = torch.tensor([self.rows[i]["ply_idx"] for i in idx], dtype=torch.long)
        return (torch.from_numpy(x), torch.from_numpy(x_nx),
                torch.from_numpy(q_star), v_idx)

    def oracle_check(self, gen: torch.Generator, n_samples: int = 200) -> dict:
        """SHIP THE CHECK: for `n_samples` sampled rows, independently
        re-derive legality and fen_after with a FRESH python-chess Board --
        not by trusting label_plies' stored output, but by redoing the same
        push a second time from fen_before and comparing. Raises on the
        first mismatch; returns a report on success."""
        idx = torch.randint(len(self.rows), (n_samples,), generator=gen).tolist()
        checked = 0
        for i in idx:
            row = self.rows[i]
            board = chess.Board(row["fen_before"])
            mv = chess.Move.from_uci(row["uci"])
            assert mv in board.legal_moves, (
                f"oracle check failed: {row['uci']} illegal at {row['fen_before']}"
            )
            board.push(mv)
            assert board.fen() == row["fen_after"], (
                f"oracle check failed: fen_after mismatch for {row['uci']} at {row['fen_before']}"
            )
            checked += 1
        return dict(checked=checked, pool_size=len(self.rows))


def demo() -> None:
    """Assert-based self-check, plus the printed report the task asked for:
    a real batch's shapes, the outcome distribution over a few hundred
    games, and the oracle check result. No network, no GPU, no download."""
    bed_a = ChessBed.build(n_games=50, seed=0)
    bed_b = ChessBed.build(n_games=50, seed=0)
    assert [r["fen_before"] for r in bed_a.rows] == [r["fen_before"] for r in bed_b.rows], (
        "seed=0 self-play did not regenerate identically -- hash-pinning is broken"
    )

    bed = ChessBed.build(n_games=300, seed=0)
    gen = torch.Generator().manual_seed(0)
    x, x_nx, q_star, v_idx = bed.batch(gen, 8)
    print(f"[MEASURED] batch shapes: x={tuple(x.shape)} x_nx={tuple(x_nx.shape)} "
          f"q_star={tuple(q_star.shape)} v_idx={tuple(v_idx.shape)}")
    assert x.shape == x_nx.shape == (8, X_DIM)
    assert q_star.shape == (8, N_OUTCOMES)
    assert torch.allclose(q_star.sum(-1), torch.ones(8))  # one-hot rows

    total = sum(bed.outcome_counts.values())
    dist = {k: f"{v}/{total} ({100 * v / total:.1f}%)" for k, v in bed.outcome_counts.items()}
    print(f"[MEASURED] outcome distribution over {total} self-play games "
          f"({len(bed.rows)} plies): {dist}")

    report = bed.oracle_check(gen, n_samples=500)
    print(f"[MEASURED] oracle check: {report['checked']}/{report['checked']} sampled plies "
          f"legal, recomputed fen_after matched every time, pool_size={report['pool_size']}")

    # the SCALE path (from_pgn), exercised against the repo's own gate-0
    # fixture -- not a network fetch, just proof the iter_games/label_plies
    # wiring is real code and not merely a docstring promise.
    import pathlib
    fixture = pathlib.Path(__file__).resolve().parents[2] / "tests/gate0/fixtures/games.pgn"
    pgn_bed = ChessBed.from_pgn(fixture)
    pgn_report = pgn_bed.oracle_check(gen, n_samples=min(200, len(pgn_bed.rows)))
    print(f"[MEASURED] from_pgn({fixture.name}): {len(pgn_bed.rows)} plies, "
          f"outcome_counts={pgn_bed.outcome_counts}, oracle check "
          f"{pgn_report['checked']}/{pgn_report['checked']} passed")


if __name__ == "__main__":
    demo()
