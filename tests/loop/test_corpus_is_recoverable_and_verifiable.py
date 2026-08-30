"""The corpora are not committed, so their recovery instructions have to be right.

WHAT RESTS ON THIS. `data/` holds eleven third-party files, all gitignored by
licence -- correctly, and `data/README.md` names a source URL and a paper citation
for every one. Round 10 iteration 1 measured what their absence costs: a clean
clone of the same commit reds 56 tests (17 corpus-dependent files: 56 failed,
177 passed, 31 skipped, 4 xfailed), because `git ls-files data/` returns one file
and no tracked runner fetches the rest.

The response to that finding is NOT to commit the corpora. It is to make a
re-fetch CHECKABLE, because a re-fetch that silently differs is worse than a
missing file: the tests go green and every number moves.

TWO GAPS, MEASURED THIS ITERATION.

1. NO CHECKSUM EXISTED. `git grep -in "sha256|md5|checksum"` over the docs and
   the `scale/`, `ceq/`, `tests/` trees returned nothing for any file under
   `data/`, and `scale/merkle.py` -- the repo's own hashing instrument -- does not
   cover `data/`. So the corpus that produced the shipped parity ratio 1.0334 at
   3.3M parameters could be replaced by a different upstream revision and nothing
   would notice. `data/CHECKSUMS.sha256` now records all eleven.

   That file deliberately does NOT use a `.txt` extension: `.gitignore:30` ignores
   `data/*.txt`, and a checksum file that is itself untracked verifies nothing.
   The first draft of it was written as `CHECKSUMS.txt`, was silently ignored, and
   hashed itself -- both caught before it shipped.

2. THE STATED DERIVATION DOES NOT MATCH THE FILE. `data/README.md` says
   `tinystories_20k.txt` is "a 20,000-line head of the TinyStories train split".
   Measured: 211,765 lines, 105,109 of them blank, zero `<|endoftext|>` markers.
   The file is 10.6x the stated size.

   The likely reading is 20,000 STORIES rather than lines -- the head of the file
   is one multi-paragraph story, blank-line separated -- but this test does not
   assert which side is wrong, because nothing in the repo settles it. What it
   asserts is that the two disagree, which is the defect: `data/README.md` is the
   ONLY tracked instruction for recreating this file, and following it literally
   produces a 20,000-line corpus that is not the one every published number used.

WHY THIS IS RED AND NOT A DOC EDIT. Correcting the README to say "20,000 stories"
would be a guess. The honest repair is for whoever fetched it to state the exact
upstream operation -- split, revision, slice -- and for the checksum to confirm it.
Until then the instruction is known-wrong and the test says so.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SUMS = DATA / "CHECKSUMS.sha256"
CORPUS = DATA / "tinystories_20k.txt"
README = DATA / "README.md"


def recorded_sums() -> dict[str, str]:
    """{relative path: sha256} from the tracked checksum file."""
    out: dict[str, str] = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, name = line.partition(" ")
        out[name.strip().lstrip("*")] = digest.strip()
    return out


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_a_checksum_file_exists_and_is_tracked():
    """The gap this closes. Without it, a re-fetch cannot be told from the original."""
    assert SUMS.exists(), f"{SUMS} missing; a re-fetch of data/ is unverifiable"
    assert SUMS.suffix != ".txt", (
        ".gitignore:30 ignores data/*.txt, so a checksum file named .txt is itself "
        "untracked and verifies nothing"
    )
    assert len(recorded_sums()) >= 11, "checksum file does not cover all eleven corpora"


@pytest.mark.parametrize("name", sorted(recorded_sums()) if SUMS.exists() else [])
def test_present_corpus_matches_its_recorded_hash(name: str):
    """Guards re-fetch drift. Skips rather than fails where the corpus is absent.

    A clean clone has none of these; that is by design and is not this test's
    business. What IS its business is a file that is present and different.
    """
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not fetched here; see data/README.md for its source")
    assert sha256(path) == recorded_sums()[name], (
        f"{name} differs from the copy that produced the published numbers. "
        "Either the upstream revision moved or the local file was edited; "
        "do not re-record the hash without saying which."
    )


def test_the_readme_derivation_matches_the_corpus_it_describes():
    """THE DEFECT. The only tracked recipe for this file describes a different file."""
    if not CORPUS.exists():
        pytest.skip("tinystories_20k.txt not fetched here")
    stated = re.search(r"([\d,]+)-line head", README.read_text(encoding="utf-8"))
    assert stated, "data/README.md no longer states a line count for tinystories_20k.txt"
    claimed = int(stated.group(1).replace(",", ""))
    actual = sum(1 for _ in CORPUS.open("rb"))
    assert claimed == actual, (
        f"data/README.md says a {claimed:,}-line head; the corpus has {actual:,} lines "
        f"({actual / claimed:.1f}x). It is the only tracked instruction for recreating "
        "this file, and following it literally yields a corpus that is not the one every "
        "published number used. Repair by stating the exact upstream operation "
        "(split, revision, slice), not by editing the number to match."
    )
