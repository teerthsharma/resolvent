"""K-DATA -- the v17-K loader, its hash gate, and the five hygiene rules.

Gate-0 item G0.5. The Kaggle notebook is supposed to call this and nothing
else: attach a dataset, hand its path (or its file object, for the 34.4 GB set)
to `verify_file` / `HashedLineReader`, and get either a printed SHA-256 that
matched a pin or an exception. There is no third outcome. A loader that
proceeds when it cannot check is the failure this module exists to prevent,
so an ABSENT pin raises `MissingPin` exactly like a wrong one raises
`HashMismatch`.

WHAT THIS MODULE IS NOT. It builds no corpus and defines no task. BED-M is
`ceq/corpus.py`, BED-K and BED-1 are `ceq/beds/`, and all three are reached
here only through `bed_signature`, which regenerates them at a recorded seed
and hashes the result. Nothing here is trained and nothing here reads a GPU.

THE FIVE HYGIENE RULES, and the function that enforces each:

  1. splits by GAME (chess) and by ARTICLE (enwik8), never by row
         `split_by_game` / `check_game_split`
         `split_by_article` / `check_article_bounds`
  2. FEN dedupe across splits, because opening positions repeat
         `fen_dedupe`
  3. n-gram overlap census on enwik8, printed as a number
         `ngram_census`
  4. tokenizer = raw bytes, frozen, hashed into the manifest
         `encode` / `tokenizer_fingerprint`
  5. every oracle label RECOMPUTED at load with python-chess
         `label_plies` -- and note it takes a GAME, never a row of a table,
         so there is no stored column in scope for it to read

Each has a planted-violation test in `tests/gate0/test_g05_data.py`.

STREAMING, NOT COPYING. `lichess/chess-evaluations` is 34.4 GB and the pin is
on ONE shard. `HashedLineReader` reads in fixed-size blocks and accumulates the
digest as it goes, so the peak resident cost is one block regardless of shard
size, and the digest is available the moment the stream ends. A reader that
called `fh.read()` with no size would produce the same digest and lose the
property; the test spies on the call to make sure it does not.

    python -m ceq.kdata                        # self-check, ~2 s
    python -m ceq.kdata --write                # (re)write results/k_data_manifest.json
    python -m ceq.kdata --write --enwik8 PATH  # ... and pin + census the real file
"""
from __future__ import annotations

import bisect
import hashlib
import json
import pathlib
from typing import Iterator

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results" / "k_data_manifest.json"

#: byte-level n-gram census defaults. `n` is in BYTES because the tokenizer is
#: raw bytes; `modulus` subsamples by n-gram VALUE (see `ngram_census`).
CENSUS_N = 64
CENSUS_MODULUS = 64

SPLITS = ("train", "val", "test")
DEFAULT_FRACS = (0.90, 0.05, 0.05)


class MissingPin(Exception):
    """No expected hash exists for this source. Loud by design: a load that
    cannot be checked must not be reported as a load that checked out."""


class HashMismatch(Exception):
    """The bytes are not the bytes the pin was taken over."""


class HygieneViolation(Exception):
    """A split, a dedupe or a boundary rule was broken."""


class ShortRead(Exception):
    """Fewer bytes arrived than a slice rule requires. A truncated mount or a
    partial stream is a different object than the slice a pin was taken
    over, and must not be hashed as if it were the whole thing."""


# --------------------------------------------------------------------------
# hashing and the pin gate
# --------------------------------------------------------------------------

def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class HashedLineReader:
    """Iterate a binary stream line by line while hashing every byte read.

    For the 34.4 GB evaluations set: the shard is never materialised, and
    `hexdigest()` after the iteration is the digest of exactly the bytes that
    were consumed. Blocks are read with an explicit size, so the resident cost
    is `chunk` and not the shard.
    """

    def __init__(self, fh, chunk: int = 1 << 20):
        self._fh = fh
        self._chunk = chunk
        self._h = hashlib.sha256()

    def __iter__(self) -> Iterator[bytes]:
        buf = b""
        while True:
            block = self._fh.read(self._chunk)
            if not block:
                break
            self._h.update(block)
            *lines, buf = (buf + block).split(b"\n")
            yield from lines
        if buf:
            yield buf

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def _pin(name: str, manifest: dict | None):
    man = manifest if manifest is not None else load_manifest()
    src = man.get("sources", {}).get(name)
    if src is None:
        raise MissingPin(
            f"{name!r} is not in the manifest. Add it to results/k_data_manifest.json "
            "with its licence and its pin before loading it."
        )
    if not src.get("sha256"):
        raise MissingPin(
            f"{name!r} carries no pinned sha256 (status {src.get('status')!r}). "
            "Pin it from the attached copy before any cell counts; do not proceed "
            "on an unchecked load."
        )
    return src["sha256"]


def verify_digest(name: str, digest: str, manifest: dict | None = None) -> str:
    """Print the digest, then compare it against the pin. Raises or returns."""
    print(f"[MEASURED] {name} sha256={digest}")
    expected = _pin(name, manifest)
    if digest != expected:
        raise HashMismatch(
            f"{name}: expected {expected}, measured {digest}. Either the upstream "
            "revision moved or the local copy was edited; do not re-record the pin "
            "without saying which."
        )
    return digest


def verify_file(name: str, path, manifest: dict | None = None) -> str:
    return verify_digest(name, sha256_file(path), manifest)


def load_manifest(path=None) -> dict:
    p = pathlib.Path(path or MANIFEST_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; run `python -m ceq.kdata --write` to regenerate it."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# hygiene rule 4 -- the tokenizer is raw bytes, frozen, and hashed
# --------------------------------------------------------------------------

VOCAB_SIZE = 256


def encode(data: bytes) -> list[int]:
    """Raw bytes. The identity on 0..255, and there is nothing to fit."""
    return list(data)


def tokenizer_fingerprint() -> str:
    """SHA-256 over the tokenizer's COMPLETE extensional definition.

    A byte tokenizer is a total function on a finite 256-element domain, so
    hashing its output on the whole domain plus its vocabulary size is not a
    proxy for the definition -- it IS the definition. Any change to `encode`
    or to `VOCAB_SIZE` moves this hash, which is the property the manifest
    depends on; a fingerprint of the source text would instead move on a
    comment edit and miss a monkeypatched replacement.
    """
    fn = globals()["encode"]
    probe = bytes(range(256))
    payload = json.dumps(
        {"kind": "raw_bytes", "vocab_size": int(globals()["VOCAB_SIZE"]),
         "table": list(fn(probe))},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# hygiene rule 1a -- chess splits by GAME, never by row
# --------------------------------------------------------------------------

def iter_games(path):
    """Yield `(game_key, chess.pgn.Game)`. The key is the whole unit of split.

    Lichess PGN carries the game URL in `Site`, which is the natural game id.
    Where it is absent the ordinal stands in; two games sharing a key land in
    the same split, which errs toward containment rather than leakage.
    """
    import chess.pgn

    with pathlib.Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        i = 0
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            yield (game.headers.get("Site") or f"game:{i}"), game
            i += 1


def split_by_game(keys, fracs=DEFAULT_FRACS, salt: str = "") -> dict:
    """Deterministic per-GAME assignment. No row is ever split from its game.

    Assignment is by the hash of the key, not by position, so appending games
    upstream does not reshuffle the ones already assigned. `salt` re-draws the
    whole split and exists so a re-draw is an explicit, recorded act.
    """
    lo, mid = fracs[0], fracs[0] + fracs[1]
    out = {}
    for key in keys:
        digest = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2 ** 64
        out[key] = SPLITS[0] if u < lo else (SPLITS[1] if u < mid else SPLITS[2])
    return out


def check_game_split(rows) -> None:
    """`rows` is any iterable of `(game_key, split)`. Raises if one game's rows
    landed in more than one split -- i.e. if the split was taken by row."""
    seen: dict[str, str] = {}
    for key, split in rows:
        if key in seen and seen[key] != split:
            raise HygieneViolation(
                f"game {key!r} appears in both {seen[key]!r} and {split!r}: the "
                "split was taken by ROW, not by GAME"
            )
        seen[key] = split


# --------------------------------------------------------------------------
# hygiene rule 5 -- every oracle label RECOMPUTED at load
# --------------------------------------------------------------------------

def label_plies(game) -> list[dict]:
    """Replay the game on a real board and emit what the board says.

    Legality and the next FEN are computed by python-chess from the position,
    never read from a column. The signature is the point: this takes a GAME
    object, so no stored label is even reachable from here. A shard's
    `stored_*` columns can be set to anything at all without moving one byte
    of this output.
    """
    board = game.board()          # a python-chess Board; it is the oracle
    out = []
    for mv in game.mainline_moves():
        legal = mv in board.legal_moves
        row = {
            "fen_before": board.fen(),
            "uci": mv.uci(),
            "san": board.san(mv) if legal else "",
            "legal": legal,
        }
        board.push(mv)
        row["fen_after"] = board.fen()
        out.append({k: row[k] for k in ("fen_before", "uci", "san", "fen_after", "legal")})
    assert all(isinstance(r["legal"], bool) for r in out)
    return out


# --------------------------------------------------------------------------
# hygiene rule 2 -- FEN dedupe across splits
# --------------------------------------------------------------------------

def fen_dedupe(fens_by_split: dict) -> tuple[dict, dict]:
    """Drop from the holdouts every FEN train has already seen.

    A by-GAME split is necessary and not sufficient: two different games share
    the opening trunk move for move, so the first dozen positions of a val game
    are verbatim in train. Train is never trimmed -- the holdout is what has to
    be clean -- and `test` is additionally trimmed against `val` so the two
    holdouts do not score the same position twice.
    """
    train = set(fens_by_split.get("train", ()))
    val = set(fens_by_split.get("val", ())) - train
    test = set(fens_by_split.get("test", ())) - train - val
    kept = {"train": train, "val": val, "test": test}
    before = sum(len(fens_by_split.get(s, ())) for s in SPLITS)
    report = {
        "dropped": before - sum(len(kept[s]) for s in SPLITS),
        "train": len(train), "val": len(val), "test": len(test),
        "leaked_val": len(set(fens_by_split.get("val", ())) & train),
        "leaked_test": len(set(fens_by_split.get("test", ())) & train),
    }
    return kept, report


# --------------------------------------------------------------------------
# enwik8 IS the first 100,000,000 bytes of enwik9 (the Hutter Prize's own
# definition). The bytes are carried by the attached Kaggle dataset
# `jamesmcguigan/hutter-prize` (CC-BY-SA-3.0, member `enwik9`); this module
# never reads past the slice.
# --------------------------------------------------------------------------

ENWIK8_SLICE_BYTES = 100_000_000


def read_enwik8_slice(fh, n: int = ENWIK8_SLICE_BYTES) -> bytes:
    """Read exactly the first `n` bytes of a binary stream over enwik9.

    The slice is a load-bearing step, not a formality: a truncated mount or a
    partial stream ends before `n` bytes arrive, and hashing whatever showed
    up would pin a different, smaller object under the enwik8 name without
    saying so. `ShortRead` fires instead of a silent short hash.
    """
    data = fh.read(n)
    if len(data) < n:
        raise ShortRead(
            f"expected the first {n:,} bytes of enwik9 (the enwik8 slice); "
            f"got only {len(data):,} before the stream ended"
        )
    return data


# --------------------------------------------------------------------------
# hygiene rule 1b -- enwik8 splits by ARTICLE at the fixed 90/5/5 offsets
# --------------------------------------------------------------------------

def article_starts(data: bytes, tag: bytes = b"<page>") -> list[int]:
    """Byte offsets of the line on which each article opens."""
    out, i = [], data.find(tag)
    while i != -1:
        out.append(data.rfind(b"\n", 0, i) + 1)
        i = data.find(tag, i + len(tag))
    return out


def split_by_article(data: bytes, fracs=DEFAULT_FRACS) -> dict:
    """The author's fixed 90/5/5 by byte offset, SNAPPED to article starts.

    The two rules read as if they conflict -- "fixed 90/5/5 split by byte
    offset" and "splits by ARTICLE, never by row" -- and they do not: the
    offsets fix the split, and snapping each to the next article start makes
    the split reproducible from the offsets alone while leaving no article
    straddling a boundary. Both numbers are returned so the snap is visible
    rather than silent.
    """
    starts = article_starts(data)
    if len(starts) < 3:
        raise HygieneViolation(
            f"only {len(starts)} article boundaries found; cannot split by article"
        )
    n = len(data)
    nominal = [int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))]
    cuts = []
    for want in nominal:
        j = bisect.bisect_left(starts, want)
        while j < len(starts) and (cuts and starts[j] <= cuts[-1]):
            j += 1
        if j >= len(starts):
            raise HygieneViolation(
                f"no article start at or after byte {want} left to snap to"
            )
        cuts.append(starts[j])
    bounds = {"train": (0, cuts[0]), "val": (cuts[0], cuts[1]), "test": (cuts[1], n)}
    check_article_bounds(data, bounds)
    bounds["_nominal"] = tuple(nominal)
    return bounds


def check_article_bounds(data: bytes, bounds: dict) -> None:
    """Raises unless the splits tile the file and no article straddles a cut."""
    starts = set(article_starts(data))
    spans = [bounds[s] for s in SPLITS]
    if spans[0][0] != 0 or spans[-1][1] != len(data):
        raise HygieneViolation(f"splits do not cover the file: {spans}")
    for (a_lo, a_hi), (b_lo, _) in zip(spans, spans[1:]):
        if a_hi != b_lo:
            raise HygieneViolation(f"splits are not contiguous at {a_hi} / {b_lo}")
        if a_hi not in starts:
            raise HygieneViolation(
                f"boundary at byte {a_hi} is not an article start: an article "
                "straddles the split, which is a split by ROW"
            )
        if a_hi <= a_lo:
            raise HygieneViolation(f"empty split {(a_lo, a_hi)}")


# --------------------------------------------------------------------------
# hygiene rule 3 -- the n-gram overlap census
# --------------------------------------------------------------------------

_CENSUS_BASE = np.uint64(1099511628211)


def _kept_hashes(data: bytes, n: int, modulus: int, chunk: int = 1 << 23):
    """Rolling 64-bit hashes of every n-byte window, filtered to one residue.

    Filtering is on the hash VALUE, not the position, so an n-gram is kept in
    the train set exactly when the same n-gram would be queried from the
    holdout. Inside the kept residue class the census is therefore EXACT, and
    the class is an unbiased sample of n-gram values -- which is what makes a
    90 MB census fit in memory at all (90M windows would not).
    """
    a = np.frombuffer(data, dtype=np.uint8)
    if a.size < n:
        return
    pos, step = 0, max(chunk, n)
    while pos + n <= a.size:
        end = min(pos + step, a.size)
        m = end - pos - n + 1
        if m <= 0:
            break
        seg = a[pos:end]
        h = np.zeros(m, dtype=np.uint64)
        for k in range(n):
            h *= _CENSUS_BASE
            h += seg[k:k + m].astype(np.uint64)
        yield h if modulus <= 1 else h[h % np.uint64(modulus) == 0]
        pos = end - n + 1


def ngram_census(train: bytes, held: bytes, n: int = CENSUS_N,
                 modulus: int = CENSUS_MODULUS) -> dict:
    """How much of the holdout's n-gram mass already occurs in train. A NUMBER.

    `rate` is over holdout POSITIONS in the sampled residue class, so a
    verbatim-copied holdout reads 1.0 and disjoint text reads 0.0. `modulus=1`
    is the full census and is what the fixtures use; the enwik8 run subsamples.
    """
    with np.errstate(over="ignore"):
        seen: set[int] = set()
        for arr in _kept_hashes(train, n, modulus):
            seen.update(arr.tolist())
        sampled = hits = 0
        for arr in _kept_hashes(held, n, modulus):
            vals = arr.tolist()
            sampled += len(vals)
            hits += sum(1 for v in vals if v in seen)
    return {"n": n, "modulus": modulus, "train_ngrams_kept": len(seen),
            "sampled": sampled, "hits": hits,
            "rate": (hits / sampled) if sampled else 0.0}


# --------------------------------------------------------------------------
# the generators -- BED-M, BED-K, BED-1 regenerate from seed
# --------------------------------------------------------------------------

#: generator + fixed kwargs whose output `bed_signature` hashes. Changing any
#: number here changes the pin, which is the intent: the pin is on the OUTPUT
#: of a named call, not on a file nobody can re-derive.
#:
#: NO NEW CONSTRUCTIONS. Every parameter below is one the campaign already
#: runs, cited rather than chosen here, so the pin is over the bed the repo's
#: published numbers used and not over a fourth bed invented by this module:
#:   bed_m  ceq/diagnose.py:123 + tests/w4/test_w4_intervention.py:45-47
#:   bed_k  tests/beds/test_bed_k.py:297  (its own determinism case)
#:   bed_1  tests/beds/test_bed_1.py:49,96
BED_SPECS = {
    "bed_m": {"generator": "ceq.corpus.build",
              "kwargs": {"n_train": 384, "n_test": 128, "seed": 0}},
    "bed_k": {"generator": "ceq.beds.bed_k.build_delay",
              "kwargs": {"n": 500, "d": 4, "seed": 7}},
    # jitter > 0 deliberately: at jitter=0 bed_1.build ignores `seed` by design
    # (ceq/beds/bed_1.py:126-128), and a seed-inert signature would not prove
    # the regeneration is seeded at all.
    "bed_1": {"generator": "ceq.beds.bed_1.build",
              "kwargs": {"T": 0.25, "seed": 11, "jitter": 0.05}},
}


def _feed(obj, h) -> None:
    """Canonical byte feed for a generator's manifest dict."""
    if isinstance(obj, np.ndarray):
        h.update(b"nd" + str(obj.dtype).encode() + str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            h.update(f"|{k}|".encode())
            _feed(obj[k], h)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            _feed(v, h)
        h.update(b"]")
    elif isinstance(obj, (np.generic,)):
        _feed(obj.item(), h)
    else:
        h.update(repr(obj).encode())


def bed_signature(name: str) -> str:
    """Regenerate the bed at its recorded seed and hash the whole manifest."""
    import importlib

    spec = BED_SPECS[name]
    mod, _, fn = spec["generator"].rpartition(".")
    built = getattr(importlib.import_module(mod), fn)(**spec["kwargs"])
    h = hashlib.sha256()
    _feed(built, h)
    return h.hexdigest()


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

#: The attach list asks for BED-M "as the intact 211,765-line file (not
#: regenerated; hash-pinned)". No such file exists, and this records the
#: measurement rather than manufacturing a file to match the description.
BED_M_FINDING = (
    "FINDING, not a pin over the file the attach list describes. The list asks "
    "for BED-M as an 'intact 211,765-line file (not regenerated; hash-pinned)'. "
    "Every file in this working tree over 1 MB was line-counted on 2026-08-31: "
    "exactly one has 211,765 lines, and it is data/tinystories_20k.txt "
    "(18,167,706 bytes, sha256 276781813f9ae1690789e727ef4b3e5f877dc6233fbd0b9c"
    "dd6acba930e685e5, already pinned in data/CHECKSUMS.sha256) -- TinyStories, "
    "not a chain corpus. BED-M has no on-disk artifact at all: it is "
    "ceq/corpus.py's build(), whose labels come from CPython at call time "
    "(ceq/beds/__init__.py:3 names ceq/corpus.py as BED-M). So 211,765 is "
    "TinyStories' line count attached to the wrong corpus, and BED-M is pinned "
    "here the only way it can be -- on the regenerated output at the campaign's "
    "own parameters. Nothing was regenerated to make a count match."
)

REJECTED = {
    "robikscube/this-week-in-chess-archive":
        "licence reads '(c) Original Authors' -- not an open licence, so "
        "redistribution and derived-model terms are undetermined. Struck by the "
        "author's attach list.",
    "dimitrioskourtikakis/gm-games-chesscom":
        "chess.com source. The consequence labels are joined to lichess "
        "evaluations BY FEN, and a chess.com corpus breaks that join. Struck by "
        "the author's attach list.",
    "thedevastator/tinystories-narrative-classification":
        "mislabelled licence, and it is the classification cut rather than the "
        "LM cut. Struck by the author's attach list in favour of the "
        "CDLA-Sharing-1.0 cut.",
    "nightfury1103/enwik8":
        "unlicensed mirror of the Hutter Prize file, zero votes. Struck on the "
        "same licence-unknown grounds as lanceni/enwik8 below; the licensed "
        "jamesmcguigan/hutter-prize superset is used instead so the bpb parity "
        "anchors cite the same bytes the published numbers used.",
    "lanceni/enwik8":
        "exactly 100,000,000 bytes -- the right size -- but its licence reads "
        "unknown, the same status that got nightfury1103/enwik8 struck. Size "
        "matching the target is not a licence.",
    "nguyenatu/enwik8":
        "apache-2.0, but the dataset contains BPE tokenizer JSONs, not the "
        "corpus itself -- an open licence over the wrong artifact.",
    "yorkyong/text8-zip":
        "unknown licence, and text8 is a different cut of the Hutter Prize "
        "corpus (lowercased, punctuation-stripped) from enwik8, not a "
        "substitute for it.",
}


def _sources(enwik8_pin: dict | None) -> dict:
    return {
        "lichess_chess_games": {
            "kind": "kaggle_attach", "url": "https://www.kaggle.com/datasets/arevel/chess-games",
            "licence": "CC0-1.0", "size": "1.56 GB", "sha256": None,
            "status": "UNPINNED_AWAITING_KAGGLE",
            "role": "move SEQUENCES for next-state prediction; python-chess "
                    "recomputes legality and next-FEN at load, no label stored",
            "loader": "ceq.kdata.iter_games + ceq.kdata.label_plies",
            "split_unit": "GAME (Site header)",
            "note": "pin with ceq.kdata.verify_file over the whole attached PGN "
                    "the first time it is attached, then record the digest here",
        },
        "lichess_chess_evaluations": {
            "kind": "kaggle_attach_streamed",
            "url": "https://www.kaggle.com/datasets/lichess/chess-evaluations",
            "licence": "CC0-1.0", "size": "34.4 GB", "sha256": None,
            "status": "UNPINNED_AWAITING_KAGGLE",
            "role": "consequence labels (eval delta), joined to the games BY FEN",
            "loader": "ceq.kdata.HashedLineReader (streamed; never copied locally)",
            "split_unit": "n/a -- joined to the game split by FEN",
            "note": "the pin is on ONE STREAMED SHARD, not the 34.4 GB set. Record "
                    "the shard's member name in hashed_over alongside its digest.",
        },
        "enwik8": {
            "kind": "kaggle_attach_sliced",
            "url": "https://www.kaggle.com/datasets/jamesmcguigan/hutter-prize",
            "licence": "CC-BY-SA-3.0",
            "dataset": "jamesmcguigan/hutter-prize",
            "member": "enwik9",
            "slice_rule": f"first {ENWIK8_SLICE_BYTES:,} bytes of enwik9 -- the "
                          "Hutter Prize's own definition of enwik8",
            "size": f"{ENWIK8_SLICE_BYTES:,} bytes (slice of a "
                    "1,000,000,000-byte member)",
            "role": "bpb parity bar against published anchors",
            "loader": "ceq.kdata.read_enwik8_slice + ceq.kdata.split_by_article "
                      "(fixed 90/5/5 by byte offset, snapped to <page> starts)",
            "split_unit": "ARTICLE (<page>)",
            **(enwik8_pin or {"sha256": None, "status": "UNPINNED_AWAITING_KAGGLE"}),
        },
        "tinystories_cdla": {
            "kind": "kaggle_attach",
            "url": "https://www.kaggle.com/datasets/alexkarev/tinystories-train-ready",
            "licence": "CDLA-Sharing-1.0", "size": "595 MB", "sha256": None,
            "status": "UNPINNED_AWAITING_KAGGLE",
            "role": "the New York demo cut, confirmed in scope by the author",
            "loader": "ceq.kdata.verify_file",
            "split_unit": "STORY",
            "note": "NOT data/tinystories_20k.txt. That local file is a different "
                    "cut (211,765 lines, sha256 2767818... , pinned in "
                    "data/CHECKSUMS.sha256) whose upstream slice is unrecoverable "
                    "from the artifact -- see tests/loop/"
                    "test_corpus_is_recoverable_and_verifiable.py.",
        },
        **{name: {
            "kind": "generator",
            "generator": spec["generator"], "seed": spec["kwargs"].get("seed"),
            "kwargs": spec["kwargs"],
            "licence": "this repository",
            "sha256": bed_signature(name), "status": "PINNED",
            "hashed_over": f"the full manifest dict returned by "
                           f"{spec['generator']}(**{spec['kwargs']}), canonicalised "
                           f"by ceq.kdata._feed",
            "role": {"bed_m": "the chain corpus (ceq/corpus.py)",
                     "bed_k": "delayed-cause bed",
                     "bed_1": "splitting-probability bed"}[name],
            **({"note": BED_M_FINDING} if name == "bed_m" else {}),
        } for name, spec in BED_SPECS.items()},
    }


def build_manifest(enwik8_path=None) -> dict:
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    enwik8_pin, census = None, None
    if enwik8_path:
        p = pathlib.Path(enwik8_path)
        with p.open("rb") as fh:
            data = read_enwik8_slice(fh)
        bounds = split_by_article(data)
        enwik8_pin = {
            "sha256": hashlib.sha256(data).hexdigest(), "status": "PINNED",
            "hashed_over": f"the first {len(data):,} bytes of enwik9 (member of "
                           "the jamesmcguigan/hutter-prize Kaggle dataset, "
                           "CC-BY-SA-3.0) -- the Hutter Prize's own definition "
                           "of enwik8",
            "bytes": len(data),
            "articles": len(article_starts(data)),
            "split_offsets": {s: list(bounds[s]) for s in SPLITS},
            "nominal_offsets": list(bounds["_nominal"]),
        }
        tr = data[bounds["train"][0]:bounds["train"][1]]
        census = {s: ngram_census(tr, data[bounds[s][0]:bounds[s][1]])
                  for s in ("val", "test")}
    return {
        "schema": "ceq.kdata/1",
        "gate_item": "G0.5 (K-DATA)",
        "git_head": head,
        "tokenizer": {"kind": "raw_bytes", "vocab_size": VOCAB_SIZE,
                      "frozen": True, "fingerprint": tokenizer_fingerprint(),
                      "fingerprint_over": "sha256 of {kind, vocab_size, encode over "
                                          "the complete 0..255 domain}"},
        "hygiene": {
            "chess_split_unit": "GAME", "text_split_unit": "ARTICLE",
            "fen_dedupe": "holdout minus train, then test minus val "
                          "(ceq.kdata.fen_dedupe)",
            "oracle_labels": "RECOMPUTED at load by python-chess from the board; "
                             "ceq.kdata.label_plies takes a Game and cannot reach "
                             "a stored column",
            "ngram_census": census or "NOT MEASURED -- no enwik8 path given",
            "census_params": {"n": CENSUS_N, "modulus": CENSUS_MODULUS,
                              "unit": "bytes"},
        },
        "sources": _sources(enwik8_pin),
        "rejected": REJECTED,
    }


def demo() -> None:
    """Assert-based self-check on the committed fixtures. No network, no GPU."""
    fx = ROOT / "tests" / "gate0" / "fixtures"
    keys = [k for k, _ in iter_games(fx / "games.pgn")]
    assign = split_by_game(keys)
    fens = {s: set() for s in SPLITS}
    for key, game in iter_games(fx / "games.pgn"):
        for ply in label_plies(game):
            fens[assign[key]].add(ply["fen_before"])
    leak = len((fens["val"] | fens["test"]) & fens["train"])
    kept, report = fen_dedupe(fens)
    # `dropped` also removes val/test duplicates, so it is >= the train leak
    assert leak > 0 and report["dropped"] >= leak, (leak, report)
    assert not (kept["val"] | kept["test"]) & kept["train"]
    assert kept["val"] and kept["test"], report

    data = (fx / "wiki.xml").read_bytes()
    bounds = split_by_article(data)
    c = ngram_census(data[: len(data) // 2], data[100:1100], n=32, modulus=1)
    assert c["rate"] == 1.0, c
    print(f"[MEASURED] fixtures: {len(keys)} games, FEN leak {leak} -> 0 after dedupe, "
          f"article bounds {[bounds[s] for s in SPLITS]} (nominal {bounds['_nominal']}), "
          f"planted-contamination census rate {c['rate']}")
    print(f"[MEASURED] tokenizer fingerprint {tokenizer_fingerprint()}")
    for name in BED_SPECS:
        a, b = bed_signature(name), bed_signature(name)
        assert a == b, name
        print(f"[MEASURED] {name} regenerates to {a} twice")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--enwik8", default=None)
    args = ap.parse_args()
    demo()
    if args.write:
        man = build_manifest(args.enwik8)
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        print(f"[MEASURED] wrote {MANIFEST_PATH}")
        print(json.dumps(man["hygiene"]["ngram_census"], indent=2))
