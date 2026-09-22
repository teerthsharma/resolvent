"""Assert-based test for board_bed.py (BOARD-BED row). Run directly: python test_board_bed.py"""
import numpy as np
import chess
import board_bed as bb


def test_vocab_size():
    assert bb.VOCAB_SIZE == 64 * 64 * 5 == 20480


def test_move_to_index_roundtrip():
    board = chess.Board()
    mv = chess.Move.from_uci("e2e4")
    idx = bb.move_to_index(mv)
    promo = idx % 5
    rest = idx // 5
    to_sq = rest % 64
    from_sq = rest // 64
    assert from_sq == chess.E2 and to_sq == chess.E4 and promo == 0


def test_initial_board_labels():
    labels = bb.initial_board_labels()
    assert labels.shape == (64,)
    assert labels[chess.E1] == 6  # white king
    assert labels[chess.E8] == 12  # black king
    assert labels[chess.E4] == 0  # empty


def test_generate_game_deterministic():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    t1, l1, m1 = bb.generate_game(rng1, max_ply=20)
    t2, l2, m2 = bb.generate_game(rng2, max_ply=20)
    assert t1 == t2
    assert np.array_equal(l1, l2)
    assert len(t1) <= 20


def test_rule_free_tracker_matches_truth_when_no_special_moves():
    # A short forced sequence with zero castling/ep/promotion: tracker must
    # exactly match ground truth every ply.
    board = chess.Board()
    tokens = []
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        mv = chess.Move.from_uci(uci)
        tokens.append(bb.move_to_index(mv))
        board.push(mv)
    tracker = bb.rule_free_tracker_labels(tokens)
    truth = bb.board_to_labels(board)
    assert np.array_equal(tracker[-1], truth)


def test_rule_free_tracker_wrong_on_castling():
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]:
        board.push(chess.Move.from_uci(uci))
    tokens = [bb.move_to_index(chess.Move.from_uci(u))
              for u in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"]]
    tracker = bb.rule_free_tracker_labels(tokens)
    board.push(chess.Move.from_uci("e1g1"))  # castling
    truth = bb.board_to_labels(board)
    assert not np.array_equal(tracker[-1], truth)  # rook didn't move in tracker


def test_run_bed_small_smoke():
    r = bb.run_bed(n_games=30, n_holdout=10, seed=0, max_ply=20)
    assert r["vocab_size"] == 20480
    for min_ply in ("ply>=40", "ply>=80"):
        for floor in ("initial_board", "empty_majority", "rule_free_tracker"):
            v = r["floors"][min_ply][floor]
            assert v is None or 0.0 <= v["per_square_acc"] <= 1.0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed}/{len(tests)} tests failed")
    print(f"ALL {len(tests)} PASSED")
