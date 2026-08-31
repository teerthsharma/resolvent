"""G0.5 (K-DATA) -- the loader's hash gate and the five hygiene rules, each with
a PLANTED violation that the guard has to catch.

WHAT RESTS ON THIS. Nothing runs on Kaggle until Gate-0 is green locally, and
G0.5 is the item that decides whether a number produced on Kaggle is traceable
to bytes. Two failure modes are specifically in scope, because both have
happened in this repo already:

  * A CORPUS THAT MOVED UNDER A GREEN SUITE. `data/README.md` states a
    "20,000-line head" for a file this box measures at 211,765 lines
    (`tests/loop/test_corpus_is_recoverable_and_verifiable.py:31-40`), so the
    only tracked recipe describes a different file and `data/CHECKSUMS.sha256`
    is the sole thing separating a correct re-fetch from a wrong one. Every
    source G0.5 loads therefore gets a pin, and an ABSENT pin is an error and
    not a skip -- a loader that quietly proceeds when it cannot check is worse
    than one that has no check at all, because it reports success.

  * A GUARD WHOSE PASS HALF IS VACUOUS. This repo has struck 14 vacuous
    controls. So every rule below is tested twice: once that the guard FIRES on
    a planted violation, and once that the clean input it passes was capable of
    tripping it. A FEN-dedupe test on a corpus whose openings never repeat
    proves nothing; `test_fen_leak_is_real_before_it_is_removed` measures the
    leak first and fails if it is zero.

THE FIVE RULES, and where each is planted:

  1. split by GAME, never by row      -- test_row_level_split_is_rejected
  2. FEN dedupe across splits         -- test_fen_leak_is_{real,removed}
  3. n-gram overlap census, a number  -- test_ngram_census_{catches,clean}
  4. tokenizer frozen and hashed      -- test_tokenizer_fingerprint_moves_*
  5. oracle label RECOMPUTED at load  -- test_corrupt_stored_columns_change_nothing

FIXTURES, not the real multi-GB sets. `tests/gate0/fixtures/` holds 36 PGN
games (deliberately sharing three opening trunks so FENs repeat across games),
a 512-row stored-label sidecar whose columns exist only to be corrupted, and a
24-article enwik8-shaped XML. Correctness here does not depend on any file
being present that a clean clone lacks.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ceq import kdata

FX = pathlib.Path(__file__).resolve().parent / "fixtures"
PGN = FX / "games.pgn"
EVALS = FX / "evals_shard.jsonl"
WIKI = FX / "wiki.xml"
ENWIK8_SHORT = FX / "enwik8_short.bin"


# --------------------------------------------------------------------------
# 1. hashing and the pin gate
# --------------------------------------------------------------------------

def test_sha256_file_agrees_with_hashlib_and_separates_two_files():
    """Non-degenerate: a hasher that returned a constant would pass half of this."""
    assert kdata.sha256_file(PGN) == hashlib.sha256(PGN.read_bytes()).hexdigest()
    assert kdata.sha256_file(PGN) != kdata.sha256_file(WIKI)


def test_the_digest_is_printed_at_load(capsys):
    """The author's rule: every dataset's SHA-256 is PRINTED at load."""
    man = {"sources": {"fx": {"sha256": kdata.sha256_file(PGN)}}}
    kdata.verify_file("fx", PGN, manifest=man)
    out = capsys.readouterr().out
    assert kdata.sha256_file(PGN) in out, "verify_file did not print the digest it checked"
    assert "fx" in out


def test_an_absent_pin_is_an_error_and_not_a_skip():
    """THE RULE THAT MATTERS MOST. Silence on an uncheckable load is the defect."""
    with pytest.raises(kdata.MissingPin):
        kdata.verify_file("fx", PGN, manifest={"sources": {"fx": {"sha256": None}}})
    with pytest.raises(kdata.MissingPin):
        kdata.verify_file("not_in_manifest", PGN, manifest={"sources": {}})


def test_a_planted_one_byte_edit_is_caught(tmp_path):
    """Planted violation: the pin is right, the bytes are not."""
    good = PGN.read_bytes()
    pin = hashlib.sha256(good).hexdigest()
    bad = tmp_path / "games.pgn"
    bad.write_bytes(good.replace(b"e2e4", b"d2d4", 1) if b"e2e4" in good
                    else good[:-1] + bytes([good[-1] ^ 1]))
    assert bad.read_bytes() != good, "the plant did not change the file"
    man = {"sources": {"fx": {"sha256": pin}}}
    kdata.verify_file("fx", PGN, manifest=man)          # non-degenerate: clean passes
    with pytest.raises(kdata.HashMismatch):
        kdata.verify_file("fx", bad, manifest=man)


def test_the_shard_is_hashed_while_streaming_and_never_slurped():
    """The 34.4 GB evaluations set is pinned on ONE streamed shard.

    Non-degeneracy is the second assertion: a reader that called `fh.read()`
    with no size would produce the same digest and defeat the whole design, so
    the spy fails the test rather than the digest.
    """
    calls: list[object] = []

    class Spy:
        def __init__(self, fh):
            self._fh = fh

        def read(self, size=-1):
            calls.append(size)
            return self._fh.read(size)

    with EVALS.open("rb") as fh:
        reader = kdata.HashedLineReader(Spy(fh))
        rows = [json.loads(line) for line in reader]
        digest = reader.hexdigest()

    assert digest == hashlib.sha256(EVALS.read_bytes()).hexdigest()
    assert len(rows) == 512
    assert calls and all(s not in (-1, None) for s in calls), (
        f"HashedLineReader slurped the file: read() sizes {set(calls)}"
    )


def test_a_streamed_shard_whose_digest_moved_is_rejected():
    with EVALS.open("rb") as fh:
        reader = kdata.HashedLineReader(fh)
        for _ in reader:
            pass
        good = reader.hexdigest()
    man = {"sources": {"evals": {"sha256": good}}}
    kdata.verify_digest("evals", good, manifest=man)     # non-degenerate
    with pytest.raises(kdata.HashMismatch):
        kdata.verify_digest("evals", "0" * 64, manifest=man)


# --------------------------------------------------------------------------
# 2. hygiene rule 1 -- split by GAME, never by row
# --------------------------------------------------------------------------

def test_the_split_is_by_game_and_every_split_is_populated():
    keys = [k for k, _ in kdata.iter_games(PGN)]
    assert len(keys) == 36 and len(set(keys)) == 36
    assign = kdata.split_by_game(keys)
    counts = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
    assert all(counts[s] > 0 for s in counts), f"a split is empty: {counts}"
    assert sum(counts.values()) == 36


def test_row_level_split_is_rejected():
    """PLANTED: one game's plies dealt out row-by-row across train and val."""
    keys = [k for k, _ in kdata.iter_games(PGN)]
    assign = kdata.split_by_game(keys)
    clean = [(k, assign[k]) for k in keys for _ in range(4)]
    kdata.check_game_split(clean)                        # non-degenerate: clean passes

    leaked = list(clean)
    victim = keys[0]
    leaked.append((victim, "val" if assign[victim] != "val" else "train"))
    with pytest.raises(kdata.HygieneViolation) as e:
        kdata.check_game_split(leaked)
    assert victim in str(e.value)


def test_the_game_split_is_deterministic_and_the_salt_actually_moves_it():
    keys = [k for k, _ in kdata.iter_games(PGN)]
    assert kdata.split_by_game(keys) == kdata.split_by_game(keys)
    assert kdata.split_by_game(keys, salt="v17") != kdata.split_by_game(keys), (
        "the salt is ignored, so the split is not re-drawable"
    )


# --------------------------------------------------------------------------
# 3. hygiene rule 2 -- FEN dedupe across splits
# --------------------------------------------------------------------------

def _fens_by_split():
    assign = kdata.split_by_game([k for k, _ in kdata.iter_games(PGN)])
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for key, game in kdata.iter_games(PGN):
        for ply in kdata.label_plies(game):
            out[assign[key]].add(ply["fen_before"])
    return out


def test_fen_leak_is_real_before_it_is_removed():
    """NON-DEGENERACY FOR THE WHOLE RULE. If a correct by-GAME split leaked no
    FEN, the dedupe below would be a no-op and its PASS would mean nothing."""
    fens = _fens_by_split()
    leak = len((fens["val"] | fens["test"]) & fens["train"])
    assert leak > 0, (
        "the by-game split leaked no FEN, so the dedupe test is vacuous; "
        "the fixture must share opening trunks across games"
    )


def test_fen_dedupe_removes_the_leak_without_emptying_the_holdout():
    fens = _fens_by_split()
    kept, report = kdata.fen_dedupe(fens)
    assert report["dropped"] > 0
    assert not (kept["val"] | kept["test"]) & kept["train"]
    assert kept["val"] and kept["test"], "dedupe emptied a holdout split"
    assert kept["train"] == fens["train"], "dedupe must drop from the holdout, not train"


# --------------------------------------------------------------------------
# 4. hygiene rule 3 -- split by ARTICLE, and the n-gram census
# --------------------------------------------------------------------------

def test_the_nominal_90_5_5_offsets_fall_mid_article():
    """NON-DEGENERACY. Article snapping is only a guard if the raw byte offsets
    the author specified actually land inside an article on this fixture."""
    data = WIKI.read_bytes()
    starts = set(kdata.article_starts(data))
    nominal = [int(len(data) * f) for f in (0.90, 0.95)]
    assert not any(o in starts for o in nominal), (
        f"nominal offsets {nominal} already sit on article boundaries; nothing to snap"
    )


def test_split_by_article_snaps_to_boundaries_and_covers_the_file():
    data = WIKI.read_bytes()
    bounds = kdata.split_by_article(data)
    starts = kdata.article_starts(data)
    assert bounds["train"][0] == 0
    assert bounds["test"][1] == len(data)
    for name in ("val", "test"):
        assert bounds[name][0] in starts, f"{name} does not begin at an article start"
    assert bounds["train"][1] == bounds["val"][0] and bounds["val"][1] == bounds["test"][0]
    assert all(hi > lo for lo, hi in bounds.values()), f"an empty split: {bounds}"


def test_an_unsnapped_article_boundary_is_rejected():
    """PLANTED: a boundary moved one byte off an article start."""
    data = WIKI.read_bytes()
    bounds = kdata.split_by_article(data)
    kdata.check_article_bounds(data, bounds)             # non-degenerate: clean passes
    lo, hi = bounds["val"]
    bad = dict(bounds, train=(0, lo + 1), val=(lo + 1, hi))
    with pytest.raises(kdata.HygieneViolation):
        kdata.check_article_bounds(data, bad)


def test_ngram_census_catches_planted_contamination():
    """PLANTED: the holdout is a verbatim slice of train."""
    data = WIKI.read_bytes()
    train, held = data[: len(data) // 2], data[100:1100]
    c = kdata.ngram_census(train, held, n=32, modulus=1)
    assert c["sampled"] > 0
    assert c["rate"] == pytest.approx(1.0), c


def test_ngram_census_is_near_zero_on_disjoint_text():
    """The other half of the same guard: a census that always returns 1.0 is
    not a census."""
    train = b"".join(bytes([65 + (i * 7) % 26]) for i in range(20000))
    held = b"".join(bytes([97 + (i * 11) % 26]) for i in range(20000))
    c = kdata.ngram_census(train, held, n=32, modulus=1)
    assert c["sampled"] > 0
    assert c["rate"] == 0.0, c


def test_modulus_subsampling_is_exact_on_the_residue_class_it_keeps():
    """The 90 MB enwik8 run subsamples by hash residue, and what that buys is
    NOT that the subsampled rate equals the full rate.

    It does not, and the first draft of this test asserted that it did. The
    residue filter is a cluster sample over n-gram VALUES: a value occurring at
    many positions contributes all of them or none. Measured on this 11,752-byte
    fixture at n=32: full census rate 0.35982 over 7,805 positions, modulus-8
    census rate 0.42385 over 998 -- a 0.064 gap, about 4 binomial sigma, which
    is the cluster variance and not an error.

    What IS exact, and is what the design rests on, is the VERDICT: a held-out
    n-gram the filter keeps gets the same hit/miss answer from the subsampled
    train set as from the complete one, because the filter reads the n-gram's
    value and so applies identically on both sides. That is asserted here with
    no tolerance at all.
    """
    data = WIKI.read_bytes()
    train, held = data[: len(data) // 2], data[len(data) // 3 :]

    def hashes(buf, modulus):
        out: list[int] = []
        for arr in kdata._kept_hashes(buf, 32, modulus):
            out += arr.tolist()
        return out

    full_train, sub_train = set(hashes(train, 1)), set(hashes(train, 8))
    queried = hashes(held, 8)
    assert 0 < len(sub_train) < len(full_train)
    assert queried and sub_train <= full_train
    assert all((v in sub_train) == (v in full_train) for v in queried)
    # non-degenerate: the residue class contains both hits and misses, so the
    # agreement above is not the agreement of two constants
    verdicts = {v in full_train for v in queried}
    assert verdicts == {True, False}, verdicts


# --------------------------------------------------------------------------
# 4b. enwik8 IS the first 100,000,000 bytes of enwik9 -- the slice guard
#
# enwik8 is now read as a SLICE of the attached `jamesmcguigan/hutter-prize`
# dataset's `enwik9` member, not a standalone uploaded file. The slice is a
# load-bearing step in its own right: a truncated mount or a partial stream
# yields fewer than the ruled byte count, and hashing whatever arrived would
# silently pin a smaller, different object under the enwik8 name.
# --------------------------------------------------------------------------

def test_a_short_read_of_the_enwik8_slice_is_rejected_not_hashed():
    """PLANTED: the stream ends before the slice rule's byte count arrives."""
    with ENWIK8_SHORT.open("rb") as fh:
        with pytest.raises(kdata.ShortRead):
            kdata.read_enwik8_slice(fh, n=1_000_000)


def test_a_full_read_of_the_enwik8_slice_is_not_short():
    """Non-degenerate: the guard does not fire when the stream actually holds
    at least `n` bytes, and it returns exactly those bytes -- not more, not
    fewer."""
    data = ENWIK8_SHORT.read_bytes()
    with ENWIK8_SHORT.open("rb") as fh:
        got = kdata.read_enwik8_slice(fh, n=len(data))
    assert got == data

    with ENWIK8_SHORT.open("rb") as fh:
        got_prefix = kdata.read_enwik8_slice(fh, n=len(data) - 10)
    assert got_prefix == data[:-10], "a stream with bytes to spare must still be sliced at n"


# --------------------------------------------------------------------------
# 5. hygiene rule 4 -- the tokenizer is raw bytes, frozen, and hashed
# --------------------------------------------------------------------------

def test_the_tokenizer_is_raw_bytes_and_lossless():
    data = WIKI.read_bytes()[:4096]
    ids = kdata.encode(data)
    assert kdata.VOCAB_SIZE == 256
    assert ids == list(data)
    assert bytes(ids) == data


def test_the_fingerprint_is_stable_across_calls():
    assert kdata.tokenizer_fingerprint() == kdata.tokenizer_fingerprint()
    assert len(kdata.tokenizer_fingerprint()) == 64


def test_the_fingerprint_moves_if_the_tokenizer_moves(monkeypatch):
    """PLANTED: the tokenizer changed. A fingerprint that does not move is a
    fingerprint of nothing."""
    before = kdata.tokenizer_fingerprint()
    monkeypatch.setattr(kdata, "VOCAB_SIZE", 257)
    assert kdata.tokenizer_fingerprint() != before

    monkeypatch.setattr(kdata, "VOCAB_SIZE", 256)
    monkeypatch.setattr(kdata, "encode", lambda b: [x ^ 1 for x in b])
    assert kdata.tokenizer_fingerprint() != before


def test_the_manifest_records_the_live_tokenizer_fingerprint():
    """A stale manifest is the same defect as a stale checksum."""
    man = kdata.load_manifest()
    assert man["tokenizer"]["fingerprint"] == kdata.tokenizer_fingerprint(), (
        "results/k_data_manifest.json was written against a different tokenizer; "
        "re-run `python -m ceq.kdata --write`"
    )
    assert man["tokenizer"]["kind"] == "raw_bytes"


# --------------------------------------------------------------------------
# 6. hygiene rule 5 -- every oracle label RECOMPUTED at load
# --------------------------------------------------------------------------

def test_labels_are_recomputed_and_agree_with_python_chess():
    """Non-degeneracy for the corruption test below: the loader must actually
    produce the fields the stored columns claim to give, and produce them
    right, or 'nothing changed' would be trivially true of nothing."""
    import chess

    keys = [k for k, _ in kdata.iter_games(PGN)]
    assert keys
    _, game = next(iter(kdata.iter_games(PGN)))
    plies = kdata.label_plies(game)
    assert len(plies) >= 10
    board = chess.Board()
    for ply in plies:
        assert ply["fen_before"] == board.fen()
        mv = chess.Move.from_uci(ply["uci"])
        assert ply["legal"] is (mv in board.legal_moves)
        board.push(mv)
        assert ply["fen_after"] == board.fen()


def test_corrupt_stored_columns_change_nothing(tmp_path):
    """PLANTED: every stored label column in the shard is wrecked.

    The output must be bitwise identical, because nothing reads them. The first
    assertion proves the plant landed, so a loader that read a column it should
    not would have something to be caught by.
    """
    before = [kdata.label_plies(g) for _, g in kdata.iter_games(PGN)]

    original = EVALS.read_bytes()
    wrecked = tmp_path / "evals_shard.jsonl"
    with wrecked.open("w", encoding="utf-8", newline="\n") as fh:
        for line in original.decode().splitlines():
            row = json.loads(line)
            row["stored_fen_after"] = "8/8/8/8/8/8/8/8 w - - 0 1"
            row["stored_cp"] = 999999
            row["stored_legal"] = False
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    assert wrecked.read_bytes() != original, "the plant did not corrupt anything"

    after = [kdata.label_plies(g) for _, g in kdata.iter_games(PGN)]
    assert after == before
    assert not any("999999" in json.dumps(p) for plies in after for p in plies)
    assert not any(p.get("fen_after") == "8/8/8/8/8/8/8/8 w - - 0 1"
                   for plies in after for p in plies)


def test_no_label_field_is_sourced_from_a_stored_column():
    """Structural half of the same rule: the emitted keys are the recomputed
    ones, and no `stored_*` column survives into a label."""
    _, game = next(iter(kdata.iter_games(PGN)))
    ply = kdata.label_plies(game)[0]
    assert set(ply) == {"fen_before", "uci", "san", "fen_after", "legal"}
    assert not any(k.startswith("stored") for k in ply)


# --------------------------------------------------------------------------
# 7. the manifest itself -- sources, licences, and the rejected list
# --------------------------------------------------------------------------

REQUIRED_SOURCES = {
    "lichess_chess_games", "lichess_chess_evaluations", "enwik8",
    "tinystories_cdla", "bed_m", "bed_k", "bed_1",
}
REJECTED = {
    "robikscube/this-week-in-chess-archive",
    "dimitrioskourtikakis/gm-games-chesscom",
    "thedevastator/tinystories-narrative-classification",
    "nightfury1103/enwik8",
    # considered when enwik8 was repointed to jamesmcguigan/hutter-prize:
    "lanceni/enwik8",
    "nguyenatu/enwik8",
    "yorkyong/text8-zip",
}


def test_the_manifest_covers_every_attached_source_with_a_licence():
    man = kdata.load_manifest()
    assert REQUIRED_SOURCES <= set(man["sources"]), REQUIRED_SOURCES - set(man["sources"])
    for name, src in man["sources"].items():
        assert src.get("licence"), f"{name} has no licence recorded"
        assert src.get("status") in ("PINNED", "UNPINNED_AWAITING_KAGGLE"), src


def test_every_struck_source_is_recorded_with_its_reason():
    """The author's DO-NOT-USE list is binding, so it is data and not prose."""
    man = kdata.load_manifest()
    assert REJECTED <= set(man["rejected"]), REJECTED - set(man["rejected"])
    for slug, reason in man["rejected"].items():
        assert len(reason) > 20, f"{slug} rejected without a stated reason"


def test_pinned_sources_carry_what_the_hash_was_computed_over():
    """A hash with no stated extent cannot be re-checked. The 34.4 GB set in
    particular is pinned on ONE shard, and that has to be legible."""
    man = kdata.load_manifest()
    for name, src in man["sources"].items():
        if src["status"] == "PINNED":
            assert src.get("sha256") and len(src["sha256"]) == 64, name
            assert src.get("hashed_over"), f"{name} pins a hash over an unstated extent"
        else:
            assert src.get("sha256") is None, f"{name} claims UNPINNED but carries a hash"


# --------------------------------------------------------------------------
# 8. the generators -- BED-M / BED-K / BED-1 regenerate deterministically
# --------------------------------------------------------------------------

def test_bed_generators_are_deterministic_and_distinct():
    """Twice, same seed, same hash -- and three different beds, three different
    hashes, so the signature is not hashing a constant."""
    sigs = {name: (kdata.bed_signature(name), kdata.bed_signature(name))
            for name in ("bed_m", "bed_k", "bed_1")}
    for name, (a, b) in sigs.items():
        assert a == b, f"{name} did not regenerate identically"
    firsts = {name: a for name, (a, _) in sigs.items()}
    assert len(set(firsts.values())) == 3, firsts


def test_the_manifest_records_the_generator_signatures_it_claims():
    man = kdata.load_manifest()
    for name in ("bed_m", "bed_k", "bed_1"):
        src = man["sources"][name]
        assert src["kind"] == "generator", src
        assert src["generator"] and src["seed"] is not None, src
        assert src["sha256"] == kdata.bed_signature(name), (
            f"{name}: manifest signature is stale against a re-run on this box"
        )
