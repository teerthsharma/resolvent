# Q2-second -- the pre-registered second-corpus transfer check

LAW L-REFLECTOR: fixed structure printed before either arm is scored --
see the `q2_second_fixed_structure` board event and the JSON block
`q2_second.py` prints at the top of its own stdout, BEFORE any training.

## Why this row exists

`q2_certificate.py` (Q2's first row) found hard-concrete beating a matched
softmax twin by 0.2326-0.2622 nats at 5/5 seeds on TinyStories, repetition
<1x for the first time in this project -- but its own pre-registration says
that result is **not written as a win** until a second corpus, in a
different domain, confirms it. This is that row.

## The one variable changed: the corpus

- **First corpus (Q2)**: `data/tinystories_20k.txt` -- synthetic
  children's stories, short dependencies, 18,167,706 bytes.
- **Second corpus (this row)**: WikiText-103-raw -- encyclopedic English,
  long-form articles, CC BY-SA 4.0.

**Reachability.** Reachable from this box without credentials via
`datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)`.

**Pin.** `scratchpad/build_wikitext_corpus.py` streams the train split,
groups lines into one WikiText ARTICLE per document (paragraph breaks
*within* an article are collapsed to single newlines so `DocByteBatches`'s
`"\n\n"` split lands on article boundaries, the same granularity
TinyStories gets from one-story-per-blank-line-run -- not paragraph-level,
which a first attempt at this script produced and which was corrected
before any training ran), stopping once >=100 MiB of article text is
gathered. Output: `scratchpad/wikitext103_corpus.txt`.

| field | value |
|---|---|
| file on disk (`wc -c` equivalent) | **105,326,029 bytes** (`file_bytes_total`, measured via `os.path.getsize`) |
| full-file sha256 | `3523ac1d05d016ba9524b4473689882104b8b947e01757108bdbf3fed64b21c9` |
| bytes actually loaded by `train_with_eval` (post `max_bytes=64MiB` truncation -- RE's own default, unchanged) | **67,239,422** -- this is the number that sets `CORPUS_TOKENS`/`repetition_factor`, measured via `RE._corpus_text(path, 64*1024*1024).encode("utf-8")`, not assumed |
| sha256 of the loaded slice | `3b81cdca44f089e6da108c424323477bb547091a9b8a1bc8c83f53cd23b980d7` |
| documents in the loaded slice | 3,866 (3,479 train / 387 val at `val_frac=0.1`) |
| licence | CC BY-SA 4.0 (Salesforce/wikitext, wikitext-103-raw-v1, Wikipedia Good/Featured articles) |
| source | `huggingface.co/datasets/Salesforce/wikitext`, train split, streamed |

The `max_bytes` cap was **not raised** to chase a lower repetition number --
it is `train_with_eval`'s existing default, identical to what Q2's first
row used (TinyStories, at 18,167,706 bytes, never needed to be truncated by
the same cap; WikiText, at 105,326,029 bytes on disk, does). That is the
corpus being larger than the unchanged cap, not a tuned parameter.

## Same protocol, everything else byte-identical to q2_certificate.py

Shape: `hidden=128 n_layers=3 n_heads=8 d_head=16 seq=512 vocab=256`
(byte-level, unchanged -- verified rather than assumed: WikiText-103-raw is
narrow UTF-8 text and needed no vocabulary adjustment). `batch=8 lr=3e-4`.
`split_seed=0` FIXED across all 5 seeds -- only model init/training draws
vary with seed. `steps = round(20 * n_params_arm_a / (batch*seq)) = 3538`,
recomputed here (not copy-pasted) from measured `numel()`, and it landed on
the identical 3,538 Q2's first row used, because `n_a=724,608` did not
move. Both arms (softmax twin over `sgate`'s skeleton via the monkeypatched
`CEQAttention.forward`; hard-concrete over `R.build_repaired(operator="smprime")`
with `arm_smprime.magnitude` monkeypatched to `R.FORMS["hard_concrete"]`)
are the same functions imported read-only from `r3_eval.py` and
`r1_gate.py`, unedited.

## BOTH repetition factors (printed before any arm was scored)

| | this corpus (WikiText-103) | TinyStories (Q2's first row, for reference) |
|---|---:|---:|
| repetition_factor(a) softmax | **0.2155x** | 0.7977x |
| repetition_factor(f) hard-concrete | **0.2158x** | 0.7985x |

Both <1x on both corpora -- generalization-admissible throughout; WikiText's
larger corpus gives more headroom below the ceiling, not less.

## L-REPRO

Command: `python q2_second.py 5` (argv[1] = n_seeds, default 5).
**Seeds actually run: 5 of 5 requested, both arms -- 10/10 cells
completed, run serially (one training process on the GPU at a time, per
the card's 8,188 MiB budget), each cell's row written to
`scratchpad/q2_second_results.jsonl` as it landed.** Device: CUDA (RTX
4060 Laptop GPU, 8,188 MiB), torch 2.14.0+cu126. Peak GPU memory observed
mid-run: 2,875 MiB (well inside budget, no host spill). Wall clock:
2,966.1s for all 10 cells (softmax ~74-77s/cell, hard_concrete
~512-535s/cell -- same asymmetry Q2's first row measured, same cause: the
gated arm's complex-valued path-product readout, not the shape or corpus).

n_docs=3,866 (n_train_docs=3,479, n_val_docs=387) at split_seed=0, matching
`q2_second_fixed_structure`'s own printed pin exactly.

## Results (5 seeds each; steps=3538)

| arm | n_params | final_eval_loss (5 seeds) | mean |
|---|---:|---|---:|
| (a) softmax twin | 724,608 | 1.8359, 1.8144, 1.8609, 1.8477, 1.8450 | 1.8408 |
| (f) hard-concrete gates | 725,391 | 1.5048, 1.5167, 1.5101, 1.5131, 1.5164 | 1.5122 |

Per-seed diff (f minus a): -0.3311, -0.2977, -0.3508, -0.3345, -0.3286
(mean -0.3285).

## Verdict

**TRANSFERS.** (f) beats (a) by more than 0.01 nats at **5 of 5** seeds on
WikiText-103 -- the smallest per-seed gap is -0.2977, **30x** the 0.01
tie band. Per the pre-registration's TRANSFERS branch: this result has now
survived a domain change (synthetic short-dependency stories ->
encyclopedic long-form text) with the shape, optimizer, split convention,
seed set, and arms all held fixed, and **may now be written as a win**,
citing both corpora and both repetition factors:

- TinyStories (18,167,706 bytes, repetition <1x): mean diff -0.2473, 5/5.
- WikiText-103 (67,239,422 bytes loaded, repetition <1x): mean diff
  -0.3285, 5/5.

The effect did not shrink going to the harder, longer-dependency domain --
it grew (-0.2473 -> -0.3285 nats). That is evidence against the
corpus-specific-artifact explanation the pre-registration itself set out
as the alternative to TRANSFERS.

## Limits

`max_bytes=64MiB` truncates the 105 MB WikiText file gathered; the run does
not use the full ~100M-token WikiText-103 corpus, only the RE default's
cap (kept identical to Q2's first row on principle -- see "same protocol"
above), so this is a statement about the *domain* transferring, not about
scaling the corpus size itself, which remains untested. The by-document
split's granularity differs in absolute document *size* across corpora
(TinyStories: 105,095 short one-paragraph stories; WikiText: 3,866 much
longer multi-paragraph articles) even though both use the same `"\n\n"`
delimiter convention -- a real property of the domains being compared, not
a protocol change, but worth naming: WikiText's val split has far fewer,
much larger held-out units (387 docs vs. 10,510), which is unavoidable
given article-level granularity and was not tuned.
