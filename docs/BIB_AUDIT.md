# Bibliography audit — 2026-09-03, coordinator, RUN

`references.bib` is the merge of the eight sweep bibliographies (508 entries parsed,
51 same-key duplicates dropped, 49 different-key/same-identifier entries folded into
one canonical key each — the alias table is `bib_aliases.md`). **408 canonical
entries.**

The Scholar Sidekick audit service returned "not subscribed", so the same
identifier-level check (real identifier paired with the wrong title — the Topaz et al.
fabrication pattern) was run directly against the two registries:

| keyed by | entries | resolved | title agreement below threshold | unresolved |
|---|---|---|---|---|
| arXiv id (`eprint`, or a `10.48550/arXiv.` DOI) | 278 | **278** (export.arxiv.org API, 5 batches) | 1 — `bottou-2013-counterfactual` (cited with its subtitle; the API title is the short form; same paper) | 0 |
| DOI (Crossref `works` API) | 92 | **91** | 6 — all subtitle truncations of books or a capitalised registry title: `bollt-2001-misplaced`, `durrett-2019-probability`, `edelsbrunner-2009-computational-topology`, `pearl-2009-causality`, `robbins-1971-almost-supermartingales`, `villani-2009-optimal` (same works) | 1 — `singh-2007-mapper`, DOI `10.2312/SPBG/SPBG07/091-100` is not in Crossref (Eurographics digital library); carried `[U]`, as the topology sweep already marked it |
| neither (books, 19th–20th-century papers, proceedings without DOI) | 38 | — | carried with the sweep's catalogue / ISBN / zbMATH / DBLP marks and its `[V]`/`[U]` tag | — |

Title agreement is `difflib` ratio on lower-cased alphanumerics; thresholds 0.8 (arXiv)
and 0.75 (Crossref). Every entry below threshold was read by eye and is the cited
work under a shorter registry title.

What this audit does **not** establish: that any source supports the sentence it is
cited for. That is the sweeps' `[V]` (abs page fetched, title matched) and `[V-eq]`
(equation transcribed, hypotheses stated, instance run) marks, which are per-source in
`sweep_*.md`; the paper carries the mark beside each citation that is load-bearing.
The record's rule (`99777ab:PRIOR_ART.md` §1.2) stands: theorem numbers and equation numbers
read through a rendered page should be re-checked against the compiled PDF before
publication.
