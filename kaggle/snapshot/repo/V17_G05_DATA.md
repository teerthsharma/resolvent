# V17-K Gate-0 · G0.5 (K-DATA) — the loader, its hash gate, and the five hygiene rules

**LEAD CAVEAT.** Nothing here was trained, no bed cell was priced, no arm was run,
and the GPU was not touched — this item is I/O and hashing only. Of the five
sources on the attach list, **exactly one could be pinned against its real bytes
on this box** (enwik8, downloaded under the single authorised download). The three
Kaggle-native sources are recorded `UNPINNED_AWAITING_KAGGLE` with `sha256: null`,
and the loader **raises `MissingPin` on every one of them** — that is the designed
behaviour, not a gap papered over, and it is what makes G0.5 **GREEN as a gate and
not as a data pin.** The Kaggle-side half is the author's; it is a two-line edit to
`results/k_data_manifest.json` per source and the gate then passes those sources
too. One item on the attach list **does not exist as described** and is reported
below rather than manufactured (§6, BED-M).

Box: Windows 11, Python 3.11.9, RTX 4060 Laptop **idle throughout**.
`git rev-parse HEAD` at start and end: `ab5b48547884e04258276e6e808d5a71ea65f917`
(both; **no writing git command was run**). `git status --porcelain` at start:
clean. At end:

```
 M ceq/hf/train.py            <- not mine (parallel agent)
?? ceq/kdata.py               <- mine
?? kaggle/                    <- not mine
?? requirements-kaggle.txt    <- not mine
?? results/k_data_manifest.json   <- mine
?? scripts/k_cert.py          <- not mine
?? tests/gate0/               <- test_g05_data.py + fixtures/ mine; g08/g09 not mine
```

Files created, and no others: `ceq/kdata.py`, `tests/gate0/test_g05_data.py`,
`tests/gate0/fixtures/{games.pgn,evals_shard.jsonl,wiki.xml}`,
`results/k_data_manifest.json`, this file.

---

## 1. Verdict table

### 1a. One row per source

| # | source | licence | pinned? | sha256 | measured over |
|---|---|---|---|---|---|
| 1 | `arevel/chess-games` (Lichess PGN) | CC0-1.0 | **NO** — `UNPINNED_AWAITING_KAGGLE` | `null` | not attachable locally; loader **raises `MissingPin`** |
| 2 | `lichess/chess-evaluations` | CC0-1.0 | **NO** — `UNPINNED_AWAITING_KAGGLE` | `null` | streaming path built and tested; pin goes on **one shard**, not 34.4 GB |
| 3 | **enwik8** (canonical Hutter file) | GFDL / CC-BY-SA | **YES** | `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8` | the extracted `enwik8`, **100,000,000 bytes** `[MEASURED]` |
| 3z | enwik8.zip as downloaded | same | **YES** | `547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc` | **36,445,475 bytes** from `http://mattmahoney.net/dc/enwik8.zip` `[MEASURED]` |
| 4a | **BED-M** | this repo | **YES**, as a generator | `2f282a5d0e9ac412b9644e19969590c0f855e5ccb3db73436364bcb925f7d24d` | `ceq.corpus.build(n_train=384, n_test=128, seed=0)` output `[MODULE]` |
| 4b | **BED-K** | this repo | **YES**, as a generator | `15de94b47ba07e8b2d118deac26306689a7c55d0b243699e7d04e0881a5c56e4` | `ceq.beds.bed_k.build_delay(n=500, d=4, seed=7)` output `[MODULE]` |
| 4c | **BED-1** | this repo | **YES**, as a generator | `f73ca0e60712dec446165131c07576d03173653b14cfdeec9ce1f581bb6881d2` | `ceq.beds.bed_1.build(T=0.25, seed=11, jitter=0.05)` output `[MODULE]` |
| 5 | `alexkarev/tinystories-train-ready` | CDLA-Sharing-1.0 | **NO** — `UNPINNED_AWAITING_KAGGLE` | `null` | not attachable locally; loader **raises `MissingPin`** |
| — | tokenizer (raw bytes, frozen) | n/a | **YES** | `f8e843a8f9e2fb311ec4d57886de2e11598080e018223e0bf01da8ab91f81e14` | `{kind, vocab_size=256, encode over the complete 0..255 domain}` `[MODULE]` |

The three bed parameter sets are **cited, not chosen here** — `ceq/diagnose.py:123`
with `tests/w4/test_w4_intervention.py:45-47` for BED-M, `tests/beds/test_bed_k.py:297`
for BED-K, `tests/beds/test_bed_1.py:49,96` for BED-1 `[INHERITED]`. NO NEW
CONSTRUCTIONS: pinning them at sizes invented by this module would have pinned a
fourth bed nobody runs.

### 1b. One row per hygiene rule

| rule | code | planted violation | guard fires | non-degeneracy of the PASS half |
|---|---|---|---|---|
| split by **GAME**, never by row | `split_by_game` / `check_game_split` | one game's plies dealt into a second split | **YES** — `HygieneViolation` naming the game | clean 144-row split passes; all three splits non-empty over 36 games `[MEASURED]` |
| split by **ARTICLE**, never by row | `split_by_article` / `check_article_bounds` | a boundary moved **1 byte** off an article start | **YES** — `HygieneViolation` | separate test proves the **nominal** 90/5/5 offsets land *mid-article* on the fixture, so the snap is doing work |
| **FEN dedupe across splits** | `fen_dedupe` | none needed — the leak is real | **YES** | **the leak is measured first: 14 FENs shared train↔holdout on a *correct* by-game split** `[MEASURED]`; after dedupe 0, and both holdouts still non-empty |
| **n-gram overlap census** | `ngram_census` | holdout = verbatim slice of train | **YES** — rate `1.0` exactly | disjoint text reads **exactly `0.0`** on the same call, so the census is not a constant |
| **tokenizer frozen + hashed** | `tokenizer_fingerprint` | `VOCAB_SIZE` 256→257; then `encode` swapped for `b ^ 1` | **YES** — fingerprint moves in **both** plants | fingerprint stable across calls; manifest's recorded value asserted equal to the live one, so a stale manifest reds |
| **oracle label RECOMPUTED at load** | `label_plies` | all 512 rows' `stored_fen_after`, `stored_cp`, `stored_legal` wrecked | **n/a — nothing to fire.** Output **bitwise identical** | labels first checked ply-by-ply against a `chess.Board` replay, so "unchanged" is not "unchanged and empty"; and the emitted key set is asserted to be exactly `{fen_before, uci, san, fen_after, legal}` with no `stored_*` survivor |

`label_plies` takes a **`Game` object**, not a table row. That is the structural
guarantee behind the rule: there is no stored column reachable from inside it, so
the corruption test is a demonstration of a property the signature already forces.

---

## 2. RED evidence

TDD, RED first. Two RED stages were captured.

**RED-0 — the module did not exist.**

```
ImportError while importing test module 'tests\gate0\test_g05_data.py'.
tests\gate0\test_g05_data.py:47: in <module>
    from ceq import kdata
E   ImportError: cannot import name 'kdata' from 'ceq'
=========================== short test summary info ===========================
ERROR tests/gate0/test_g05_data.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.31s
```

**RED-1 — a 9-line stub exposing every name, none implemented.** Collection error
replaced by 29 individual failures, so each guard is shown to be reached and unmet
rather than skipped by a collection abort:

```
    def _todo(*a, **k):
>       raise NotImplementedError(f"ceq.kdata.{name} not implemented")
E       NotImplementedError: ceq.kdata.load_manifest not implemented
=========================== short test summary info ===========================
FAILED tests/gate0/test_g05_data.py::test_sha256_file_agrees_with_hashlib_and_separates_two_files
FAILED tests/gate0/test_g05_data.py::test_the_digest_is_printed_at_load
FAILED tests/gate0/test_g05_data.py::test_an_absent_pin_is_an_error_and_not_a_skip
FAILED tests/gate0/test_g05_data.py::test_a_planted_one_byte_edit_is_caught
FAILED tests/gate0/test_g05_data.py::test_the_shard_is_hashed_while_streaming_and_never_slurped
FAILED tests/gate0/test_g05_data.py::test_a_streamed_shard_whose_digest_moved_is_rejected
FAILED tests/gate0/test_g05_data.py::test_the_split_is_by_game_and_every_split_is_populated
FAILED tests/gate0/test_g05_data.py::test_row_level_split_is_rejected
FAILED tests/gate0/test_g05_data.py::test_the_game_split_is_deterministic_and_the_salt_actually_moves_it
FAILED tests/gate0/test_g05_data.py::test_fen_leak_is_real_before_it_is_removed
FAILED tests/gate0/test_g05_data.py::test_fen_dedupe_removes_the_leak_without_emptying_the_holdout
FAILED tests/gate0/test_g05_data.py::test_the_nominal_90_5_5_offsets_fall_mid_article
FAILED tests/gate0/test_g05_data.py::test_split_by_article_snaps_to_boundaries_and_covers_the_file
FAILED tests/gate0/test_g05_data.py::test_an_unsnapped_article_boundary_is_rejected
FAILED tests/gate0/test_g05_data.py::test_ngram_census_catches_planted_contamination
FAILED tests/gate0/test_g05_data.py::test_ngram_census_is_near_zero_on_disjoint_text
FAILED tests/gate0/test_g05_data.py::test_ngram_census_modulus_subsampling_agrees_with_the_full_count
FAILED tests/gate0/test_g05_data.py::test_the_tokenizer_is_raw_bytes_and_lossless
FAILED tests/gate0/test_g05_data.py::test_the_fingerprint_is_stable_across_calls
FAILED tests/gate0/test_g05_data.py::test_the_fingerprint_moves_if_the_tokenizer_moves
FAILED tests/gate0/test_g05_data.py::test_the_manifest_records_the_live_tokenizer_fingerprint
FAILED tests/gate0/test_g05_data.py::test_labels_are_recomputed_and_agree_with_python_chess
FAILED tests/gate0/test_g05_data.py::test_corrupt_stored_columns_change_nothing
FAILED tests/gate0/test_g05_data.py::test_no_label_field_is_sourced_from_a_stored_column
FAILED tests/gate0/test_g05_data.py::test_the_manifest_covers_every_attached_source_with_a_licence
FAILED tests/gate0/test_g05_data.py::test_every_struck_source_is_recorded_with_its_reason
FAILED tests/gate0/test_g05_data.py::test_pinned_sources_carry_what_the_hash_was_computed_over
FAILED tests/gate0/test_g05_data.py::test_bed_generators_are_deterministic_and_distinct
FAILED tests/gate0/test_g05_data.py::test_the_manifest_records_the_generator_signatures_it_claims
29 failed in 0.54s
```

**RED-2 — one test was wrong and is recorded rather than deleted.** After the
implementation landed, 28/29 passed and
`test_ngram_census_modulus_subsampling_agrees_with_the_full_count` failed:

```
E       AssertionError: ({'hits': 2808, 'modulus': 1, 'n': 32, 'rate': 0.35981547924141466, ...},
                         {'hits': 423,  'modulus': 8, 'n': 32, 'rate': 0.42384769539078154, ...})
E       assert 0.06403221614936688 < 0.05
```

The **test** was wrong, not the module. Residue subsampling is a *cluster* sample
over n-gram VALUES — a value occurring at many positions contributes all of its
positions or none — so the subsampled **rate** carries cluster variance and has no
right to sit inside a binomial tolerance (the 0.064 gap is ≈4σ binomial, which is
the cluster effect). The claim the design actually rests on is exact and is now what
is asserted, with no tolerance: **a held-out n-gram the filter keeps gets the same
hit/miss verdict from the subsampled train set as from the complete one**, because
the filter reads the n-gram's *value* and applies identically on both sides. The
replacement test also asserts the queried set contains **both** hits and misses, so
the agreement is not the agreement of two constants. The measured numbers that
falsified the first version are kept in the test's own docstring.

**GREEN, final:**

```
tests/gate0/test_g05_data.py .............................    [100%]
29 passed in 0.83s
```

(`pytest tests/gate0/` as a directory currently reds at collection on
`test_g08_autopilot.py` and `test_g09_mars.py`, which belong to a parallel agent and
red on `results/k_cert_local.json missing -- run scripts/k_cert.py once`. Not this
item's files and not this item's failure.)

---

## 3. The hashes, and what each was measured over

`[MEASURED]` this box this run unless marked otherwise.

| what | sha256 | extent |
|---|---|---|
| `enwik8.zip` | `547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc` | 36,445,475 bytes as delivered by `http://mattmahoney.net/dc/enwik8.zip` |
| `enwik8` (extracted) | `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8` | 100,000,000 bytes, the single zip member |
| tokenizer | `f8e843a8f9e2fb311ec4d57886de2e11598080e018223e0bf01da8ab91f81e14` | `sha256(json({kind:"raw_bytes", vocab_size:256, table:encode(bytes(range(256)))}))` — the **complete extensional definition**, not a source-text hash, so a comment edit does not move it and a monkeypatched replacement cannot hide from it |
| BED-M | `2f282a5d0e9ac412b9644e19969590c0f855e5ccb3db73436364bcb925f7d24d` | canonicalised manifest dict of `ceq.corpus.build(384, 128, seed=0)` |
| BED-K | `15de94b47ba07e8b2d118deac26306689a7c55d0b243699e7d04e0881a5c56e4` | canonicalised manifest dict of `ceq.beds.bed_k.build_delay(500, 4, seed=7)` |
| BED-1 | `f73ca0e60712dec446165131c07576d03173653b14cfdeec9ce1f581bb6881d2` | canonicalised manifest dict of `ceq.beds.bed_1.build(0.25, seed=11, jitter=0.05)` |
| `data/tinystories_20k.txt` | `276781813f9ae1690789e727ef4b3e5f877dc6233fbd0b9cdd6acba930e685e5` | 18,167,706 bytes; re-measured here and **matches** `data/CHECKSUMS.sha256` `[INHERITED]` |

**Generator determinism, proved by running twice** `[MEASURED]`: each of BED-M,
BED-K, BED-1 was regenerated twice in the same process and produced an identical
digest, and the three digests are mutually distinct (so `bed_signature` is not
hashing a constant). `bed_1` is pinned at `jitter=0.05` deliberately: at
`jitter=0` its `build` **ignores the seed by design** (`ceq/beds/bed_1.py:126-128`),
and a seed-inert signature would not have proved the regeneration is seeded at all.

### enwik8 splits and the census

`split_by_article` reconciles the two rules the attach list states — "fixed 90/5/5
split by byte offset" and "splits by ARTICLE, never by row" — by taking the offsets
and **snapping each to the next `<page>` start**, reporting both numbers so the snap
is visible. All `[MEASURED]`:

| | nominal offset | snapped offset | drift |
|---|---|---|---|
| train/val | 90,000,000 | **90,042,869** | +42,869 B |
| val/test | 95,000,000 | **95,000,818** | +818 B |

12,347 articles; train `[0, 90042869)`, val `[90042869, 95000818)`, test
`[95000818, 100000000)`.

**N-GRAM OVERLAP CENSUS ON THE REAL enwik8 — THE NUMBER** `[MEASURED]`, 64-byte
n-grams, residue modulus 64, train side complete over 1,376,466 kept n-grams:

| holdout | sampled positions | hits | **rate** |
|---|---|---|---|
| val | 77,686 | 1,436 | **0.018485 (1.85 %)** |
| test | 77,614 | 1,256 | **0.016183 (1.62 %)** |

**Control, because a single subsampled estimate is not a measurement.** Re-run at
modulus 16 — 4× the sample, 5,510,207 kept train n-grams: val rate **0.017472**
(5,417 / 310,031) `[MEASURED]`. The two residue classes are disjoint samples of
n-gram values and agree to 0.001, so the ~1.8 % figure is the census and not an
artefact of one class.

Reading: **about 1.8 % of held-out 64-byte windows already occur verbatim in the
90 % train split.** That is intrinsic to Wikipedia — templates, infoboxes, licence
boilerplate and stock phrasing recur across articles — and it is exactly the number
a bpb parity claim has to be quoted against. It is not removable by a better split;
it is a property of the corpus, now measured instead of assumed.

---

## 4. Licence note per source

**Attached (public, Kaggle-native, internet OFF).**
1. `arevel/chess-games` — **CC0-1.0**, 1.56 GB. Move *sequences* only; python-chess
   recomputes legality and next-FEN at load, so no label is stored and none is read.
2. `lichess/chess-evaluations` — **CC0-1.0**, 34.4 GB. Attached and **streamed**;
   the pin is on **one shard**, joined to (1) by FEN for the eval-delta consequence
   labels. `HashedLineReader` reads fixed 1 MiB blocks and accumulates the digest,
   so peak resident cost is one block regardless of shard size. A test spies on the
   file object and **fails if `read()` is ever called without a size** — a slurping
   reader would produce the identical digest and silently destroy the property, so
   the digest alone could not have caught it.

**Uploaded by the author (private; hash printed).**
3. **enwik8** — canonical `http://mattmahoney.net/dc/enwik8.zip`, Hutter Prize
   distribution of GFDL / CC-BY-SA Wikipedia text. Fixed 90/5/5 by byte offset,
   article-snapped. **This was the one authorised download and it succeeded.**
4. The synthetic beds — this repository's own work. BED-K and BED-1 as
   generator + seed + expected hash, regeneration proved deterministic. BED-M: see §6.

**In scope.**
5. `alexkarev/tinystories-train-ready` — **CDLA-Sharing-1.0**, 595 MB, the correctly
   licensed cut. Recorded distinct from the local `data/tinystories_20k.txt`, which
   is a *different* cut whose upstream slice is unrecoverable from the artifact
   (`tests/loop/test_corpus_is_recoverable_and_verifiable.py`) `[INHERITED]`.

**DO NOT USE — binding, and recorded as data in `results/k_data_manifest.json["rejected"]`
so it is enforceable rather than prose. A test asserts all four are present with a
stated reason.**

| struck slug | why |
|---|---|
| `robikscube/this-week-in-chess-archive` | licence reads "© Original Authors" — not an open licence; redistribution and derived-model terms undetermined |
| `dimitrioskourtikakis/gm-games-chesscom` | chess.com source. Consequence labels join to **Lichess** evaluations **by FEN**; a chess.com corpus breaks that join |
| `thedevastator/tinystories-narrative-classification` | mislabelled licence, and it is the **classification** cut, not the LM cut |
| `nightfury1103/enwik8` | unlicensed mirror, zero votes. The canonical mattmahoney file is used so the bpb parity anchors cite the same bytes the published numbers used |

---

## 5. The pin gate, and why an absent pin is an error

`verify_file(name, path)` **prints the measured digest first**, then compares. Three
outcomes and no fourth:

* match → returns the digest;
* mismatch → **`HashMismatch`**, proved by a planted one-byte edit;
* **no pin, or the source is not in the manifest at all → `MissingPin`.**

The third is the one that matters. This repo has already been bitten by a corpus
whose only tracked recipe describes a different file — `data/README.md` claims a
"20,000-line head" for a file measured at 211,765 lines — with the checksum as the
sole thing separating a correct re-fetch from a wrong one `[INHERITED]`
(`tests/loop/test_corpus_is_recoverable_and_verifiable.py:31-40`). A loader that
skips quietly when it cannot check is worse than one with no check, because it
reports success. So the three Kaggle sources, being unpinned, **cannot be loaded
through this module today** — by design, and asserted by
`test_an_absent_pin_is_an_error_and_not_a_skip`.

---

## 6. BED-M — the finding

The attach list asks for BED-M "as the intact **211,765-line** file (not
regenerated; hash-pinned)". **That file does not exist, and it was not created.**

What was measured `[MEASURED]`, this box, this run:

* `data/`, `results/`, `ceq/corpus.py` and `scale/r10_corpus_spec.py` were searched.
  **Every file in the working tree over 1 MB was line-counted.** Exactly one has
  **211,765** lines: **`data/tinystories_20k.txt`** — 18,167,706 bytes, sha256
  `276781813f9ae1690789e727ef4b3e5f877dc6233fbd0b9cdd6acba930e685e5`, already pinned
  in `data/CHECKSUMS.sha256`. It is **TinyStories, not a chain corpus.**
* **BED-M has no on-disk artifact at all.** It is `ceq/corpus.py`'s `build()`, whose
  labels are produced by CPython at call time — `ceq/beds/__init__.py:3` names
  `ceq/corpus.py` as BED-M in as many words `[INHERITED]`. There is no line-oriented
  file for it to be "intact" as.

**Reading: 211,765 is TinyStories' line count attached to the wrong corpus.** The
two nearest sources on the attach list — item 4 (the beds) and item 5 (TinyStories)
— appear to have crossed. Consequently BED-M is treated exactly like BED-K and
BED-1: **generator + seed + expected output hash, regeneration proved deterministic
by running it twice.** Nothing was regenerated to make a count match, and no file
was written to make the description true.

**This changes the author's Kaggle upload list**: the "intact BED-M file" is not a
thing to upload. Uploading `data/tinystories_20k.txt` in its place would be the
wrong corpus *and* would duplicate item 5 with an unlicensed-provenance cut.

---

## 7. What COSTS must record

`results/k_data_manifest.json` is the ingest surface. Exact JSON paths:

| COSTS field | manifest path | value today |
|---|---|---|
| tokenizer fingerprint | `tokenizer.fingerprint` | `f8e843a8f9e2fb311ec4d57886de2e11598080e018223e0bf01da8ab91f81e14` |
| tokenizer kind / vocab | `tokenizer.kind`, `tokenizer.vocab_size` | `raw_bytes`, `256` |
| **every source's digest** | `sources.*.sha256` | 4 pinned, 3 `null` |
| **what each digest covers** | `sources.*.hashed_over` | required non-empty wherever `status == PINNED`, asserted by test |
| pin status | `sources.*.status` | `PINNED` \| `UNPINNED_AWAITING_KAGGLE` |
| licence per source | `sources.*.licence` | required non-empty, asserted by test |
| **enwik8 census rate** | `hygiene.ngram_census.{val,test}.rate` | `0.018485` / `0.016183` |
| census parameters | `hygiene.census_params` | `{n: 64, modulus: 64, unit: bytes}` — **quote the rate only with these** |
| enwik8 split offsets | `sources.enwik8.split_offsets` + `.nominal_offsets` | snapped and nominal, both |
| bed generator + seed | `sources.bed_*.{generator,seed,kwargs}` | cited campaign parameters |
| split units | `hygiene.chess_split_unit`, `hygiene.text_split_unit` | `GAME`, `ARTICLE` |
| the struck list | `rejected.*` | 4 slugs, each with a reason |
| **the BED-M finding** | `sources.bed_m.note` | the full §6 finding, verbatim, as a string |
| provenance | `git_head`, `schema` | `ab5b485…`, `ceq.kdata/1` |

Regenerate with `python -m ceq.kdata --write --enwik8 PATH`. **The manifest is not
free-standing prose**: `test_the_manifest_records_the_live_tokenizer_fingerprint`
and `test_the_manifest_records_the_generator_signatures_it_claims` red if it drifts
from what this box measures, so a stale manifest cannot pass silently.

---

## 8. Call on G0.5

**GREEN — for the local half, which is the half assigned.**

Green means: the loader exists; the hash gate has three outcomes and no silent
fourth; all five hygiene rules have code *and* a planted-violation test that catches
them; every PASS half carries its own non-degeneracy check; enwik8 is downloaded,
pinned and censused with a control; the three bed generators regenerate
deterministically; the manifest is machine-checked against this box.

Green does **not** mean the Kaggle sources are pinned. They are not, they cannot be
from here, and the loader **refuses them**. G0.5 does not become a *data* green
until the author attaches each of the three and records its digest — which is when
`test_pinned_sources_carry_what_the_hash_was_computed_over` starts checking them too.

**One item is RED as specified and reported rather than fixed**: the intact
211,765-line BED-M file (§6). It is not a blocker for G0.5 — BED-M is pinned the
only way a generator can be — but it is a **correction to the upload list** the
author must make before he uploads anything.

---

## Limits

The n-gram census subsamples by hash residue (modulus 64), so the reported rate is
an estimate of the full-position rate carrying cluster variance, not an exhaustive
count; the modulus-16 control agrees to 0.001 but does not remove the variance, and
a complete modulus-1 census of 90 M windows was not attempted on an 8 GB box. The
rolling hash is a 64-bit polynomial and admits adversarially-constructed collisions;
nothing in enwik8 is adversarial, but the number is a census and not a proof. The
PGN, eval-shard and wiki fixtures are generated by this item and are the only chess
and article data the guards have ever run against — the split, dedupe and article
logic is correct on 36 games and 24 articles and has **never seen a real Lichess PGN
or the real `<page>` stream**, so a malformed real game (variant, `FEN` header,
truncated mainline) is untested. `HashedLineReader` is proved to hash correctly and
not to slurp; it has been run against a 111 KB shard and never against a real one.
The three Kaggle sources have not been attached, so their licences are recorded from
the author's attach list, not verified against the dataset pages `[INHERITED]`. The
enwik8 digest is of the file this box downloaded on 2026-08-31 and was **not**
compared against an independent published reference — the 100,000,000-byte length is
the only external cross-check applied. `python-chess` is distribution version
**1.999**, which is a shim providing `chess` **1.11.2**; the API surface used here
(`chess.pgn.read_game`, `Board.fen`, `Board.legal_moves`, `Board.san`, `Move.uci`)
is stable, but the pin that matters is `chess==1.11.2`, not the shim's version
string. Nothing was trained, no cell was priced, no verdict was issued, and no
writing git command was run.
