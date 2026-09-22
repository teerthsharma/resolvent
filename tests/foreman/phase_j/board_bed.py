"""BOARD-BED: random-legal-game chess board-state prediction bed.

Tokenisation: UCI move -> index in a fixed vocab of size 64*64*5 = 20480
(from_square * 64 * 5 + to_square * 5 + promo), promo in
{none=0, q=1, r=2, b=3, n=4}. Target: 64-square board state after each ply,
13 classes per square (0=empty, 1..6=white P N B R Q K, 7..12=black P N B R Q K).
"""
import chess
import numpy as np

VOCAB_SIZE = 64 * 64 * 5
NUM_CLASSES = 13
PROMO_IDX = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
PIECE_CLASS = {  # (piece_type, color) -> 1..12
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 * 5 + move.to_square * 5 + PROMO_IDX[move.promotion]


def board_to_labels(board: chess.Board) -> np.ndarray:
    labels = np.zeros(64, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        labels[sq] = PIECE_CLASS[(piece.piece_type, piece.color)]
    return labels


def generate_game(rng: np.random.Generator, max_ply: int = 120):
    """One random-legal-move game. Returns (tokens[T], labels[T,64], move_meta[T])
    where move_meta[t] is a dict with is_castling/is_en_passant/is_promotion for
    the move that produced labels[t]."""
    board = chess.Board()
    tokens, labels, meta = [], [], []
    ply = 0
    while not board.is_game_over() and ply < max_ply:
        moves = list(board.legal_moves)
        mv = moves[int(rng.integers(len(moves)))]
        m = {
            "is_castling": board.is_castling(mv),
            "is_en_passant": board.is_en_passant(mv),
            "is_promotion": mv.promotion is not None,
        }
        tokens.append(move_to_index(mv))
        board.push(mv)
        labels.append(board_to_labels(board))
        meta.append(m)
        ply += 1
    return tokens, np.array(labels, dtype=np.int8), meta


def generate_dataset(n_games: int, seed: int = 0, max_ply: int = 120):
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        games.append(generate_game(rng, max_ply))
    return games


# ---------------- floors ----------------

def initial_board_labels() -> np.ndarray:
    return board_to_labels(chess.Board())


def rule_free_tracker_labels(tokens):
    """Replay moves naively: piece at from-square moves to to-square, from-square
    cleared, nothing else (no castling rook move, no e.p. pawn removal, no
    promotion). Returns labels[T,64]."""
    board = np.array(initial_board_labels())
    out = np.zeros((len(tokens), 64), dtype=np.int8)
    for t, tok in enumerate(tokens):
        promo = tok % 5
        rest = tok // 5
        to_sq = rest % 64
        from_sq = rest // 64
        piece = board[from_sq]
        board = board.copy()
        board[from_sq] = 0
        board[to_sq] = piece
        out[t] = board
    return out


def majority_class_labels(train_games) -> np.ndarray:
    """Per-square most frequent class across all plies of all training games."""
    counts = np.zeros((64, NUM_CLASSES), dtype=np.int64)
    for tokens, labels, meta in train_games:
        for row in labels:
            counts[np.arange(64), row] += 1
    return counts.argmax(axis=1).astype(np.int8)


def per_square_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


def exact_match_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).all(axis=-1).mean())


def score_floor_constant(games, const_labels: np.ndarray, min_ply: int):
    """const_labels: fixed (64,) prediction used at every ply >= min_ply."""
    preds, trues = [], []
    for tokens, labels, meta in games:
        for t in range(len(tokens)):
            if t + 1 >= min_ply:  # ply index is 1-based (after move t+1)
                preds.append(const_labels)
                trues.append(labels[t])
    if not preds:
        return None
    preds = np.stack(preds)
    trues = np.stack(trues)
    return {
        "per_square_acc": per_square_accuracy(preds, trues),
        "exact_match_acc": exact_match_accuracy(preds, trues),
        "n_positions": int(len(preds)),
    }


def score_floor_tracker(games, min_ply: int):
    tot_correct_sq = 0
    tot_sq = 0
    exact = 0
    n_pos = 0
    for tokens, labels, meta in games:
        tracker = rule_free_tracker_labels(tokens)
        for t in range(len(tokens)):
            if t + 1 >= min_ply:
                tot_correct_sq += int((tracker[t] == labels[t]).sum())
                tot_sq += 64
                exact += int((tracker[t] == labels[t]).all())
                n_pos += 1
    if n_pos == 0:
        return None
    return {
        "per_square_acc": tot_correct_sq / tot_sq,
        "exact_match_acc": exact / n_pos,
        "n_positions": n_pos,
    }


def special_move_stats(games):
    """Fraction of plies (across all games, all plies) that are castling /
    en passant / promotion -- the three rules floor (iii) ignores."""
    n_castle = n_ep = n_promo = n_ply = 0
    n_games_with_any = 0
    for tokens, labels, meta in games:
        game_has = False
        for m in meta:
            n_ply += 1
            if m["is_castling"]:
                n_castle += 1
                game_has = True
            if m["is_en_passant"]:
                n_ep += 1
                game_has = True
            if m["is_promotion"]:
                n_promo += 1
                game_has = True
        if game_has:
            n_games_with_any += 1
    return {
        "n_ply": n_ply,
        "n_games": len(games),
        "castling_frac": n_castle / n_ply if n_ply else 0.0,
        "en_passant_frac": n_ep / n_ply if n_ply else 0.0,
        "promotion_frac": n_promo / n_ply if n_ply else 0.0,
        "games_with_any_special_move_frac": n_games_with_any / len(games) if games else 0.0,
    }


def run_bed(n_games=4000, n_holdout=500, seed=0, max_ply=120):
    games = generate_dataset(n_games, seed=seed, max_ply=max_ply)
    train_games = games[:-n_holdout]
    holdout_games = games[-n_holdout:]

    init_labels = initial_board_labels()
    maj_labels = majority_class_labels(train_games)

    result = {
        "vocab_size": VOCAB_SIZE,
        "num_classes": NUM_CLASSES,
        "n_games": n_games,
        "n_holdout": n_holdout,
        "max_ply": max_ply,
        "seed": seed,
        "floors": {},
        "special_move_stats_holdout": special_move_stats(holdout_games),
    }
    for min_ply in (40, 80):
        result["floors"][f"ply>={min_ply}"] = {
            "initial_board": score_floor_constant(holdout_games, init_labels, min_ply),
            "empty_majority": score_floor_constant(holdout_games, maj_labels, min_ply),
            "rule_free_tracker": score_floor_tracker(holdout_games, min_ply),
        }
    return result


if __name__ == "__main__":
    import json
    r = run_bed()
    print(json.dumps(r, indent=2))
