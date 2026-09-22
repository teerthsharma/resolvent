# WILSON — Phase K, it.K0 (oncologist lane): DATA, harness, MFU

WD = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/wilson`

## Verdict

The DATA row is met: **3,004,523,761 GPT-2 tokens** (2,994,211,739 train + 10,312,022 val) are sharded, hashed and listed in the manifest. The harness is built, and resume is bitwise only in `--deterministic` mode. **The measured MFU is below the assumed 0.35 at R0 (0.163–0.171) and R1 (0.289–0.319). It is above 0.35 at R2 (0.364–0.438) and R3 (0.423–0.545).** Measured hours per full run are therefore about 2x Y4 at R0, 1.1–1.2x Y4 at R1, and 0.64–0.96x Y4 at R2–R3.

## 1. DATA (contract row DATA, §4)

- **License, verbatim.** Source: dataset card README.md at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, line 632, section "Licensing Information". The card's sha256 is `a0cc8998…716061e`. Front matter, line 2: `license: odc-by`.
  > The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0** [license](https://opendatacommons.org/licenses/by/1-0/). The use of this dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use).
- **Subset.** The card's line 533 reads: "`sample-10BT`: a subset randomly sampled from the whole dataset of around 10B gpt2 tokens". Files 000–003 of 000–013 were fetched.
- **Source files.** Each was fetched with curl from `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/<file>`. Each file's sha256 was recomputed on disk and matches the HF LFS sha256 (4/4 match):

| file | bytes | sha256 |
|---|---|---|
| 000_00000.parquet | 2,152,819,114 | b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871 |
| 001_00000.parquet | 2,152,222,432 | 3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693 |
| 002_00000.parquet | 2,151,796,315 | 547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7 |
| 003_00000.parquet | 2,152,437,524 | 22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb |

- **Tokeniser.** tiktoken 0.12.0, `gpt2` encoding, `encode_ordinary`. EOT 50256 is prepended to every document, which puts exactly one EOT between consecutive documents.
- **Shards.** uint16 little-endian with no header, in `C:/Users/seal/datasets/fineweb_edu/` (outside the repository).
  - 30 train shards, `train_000000.bin` … `train_000029.bin`. Shards 0–28 hold 100,000,000 tokens each (200,000,000 bytes). The last holds 94,211,739 tokens.
  - 1 val shard, `val_000000.bin`: 10,312,022 tokens, 20,624,044 bytes.
  - Total shard bytes: 6,009,047,522.
  - Every shard's sha256 was re-hashed from disk and matches its write-time hash (31/31).
- **Validation hold-out, by document.** The val shard holds the first 10,000 documents, in row order, of `000_00000.parquet`. Train holds the other 2,906,000 documents.
  - EOT count check: val has 10,000 EOTs and train has 2,906,000, equal to the document counts. The maximum id is 50256.
  - Synthetic round-trip test (600 documents, including a literal `<|endoftext|>` in the text): decode(shards) == source documents exactly.
- **Speed.** Tokenisation took 132.8 s on 24 processes.
- **Manifest.** `C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/DATA_MANIFEST.json` (10,823 bytes). It carries the per-shard file, bytes, tokens and sha256; the source files with their sha256, LFS sha256 and URL; the tokeniser and its version; the split rule; the total tokens; the license, verbatim; and the sha256 of each script. The raw parquet (8.6 GB) is kept in `C:/Users/seal/datasets/fineweb_edu/raw/`.

## 2. Harness

- **Path.** `WD/train_ladder.py` (470 lines, sha256 `d524b13005d86bd4…`). Written by a sonnet nurse, then patched by Wilson.
- **Rungs.** R0–R3 as in the contract table, vocab 50,304, `--ctx` 1024 or 4096.
  - d_head 64, pre-LN, GELU MLP 4x, bias-free linears, tied embeddings, no position table.
  - Measured parameters vs the contract's Np: 7,227,648 / 23,478,400 / 50,938,880 / 123,606,528 against Np 7,225,344 / 23,470,080 / 50,921,472 / 123,568,128. The difference is the LayerNorm parameters.
- **Arms.**
  - `--attn alibi`: the ALiBi twin (a_L). Fixed slopes use Press et al.'s recipe, including non-power-of-2 head counts. The additive causal bias goes through `F.scaled_dot_product_attention`. Nothing is learned in the position path.
  - Other lanes swap attention with `--attn <file.py>:<Class>`, a class with constructor `(d_model, n_heads, layer_idx, n_layers, ctx)` and forward `x -> x`. The harness logs that file's sha256.
  - `flite.py` is **not wired**: its bias L is shared across heads, so per-head ALiBi would need a B·H reshape.
- **Precision.** bf16 autocast on CUDA; fp32 master weights and AdamW state; cross-entropy on fp32 logits.
  - The ALiBi bias is cast to bf16, because SDPA requires the mask dtype to equal q's. The worst-case rounding is 0.0078 for |bias| < 4 and 0.0625 for |bias| < 32. Its effect on loss was **not measured**.
- **Selftest.** `python train_ladder.py --selftest` (CPU) passes 8/8:
  - param counts;
  - slopes (8 heads = 2^-1…2^-8);
  - ALiBi vs an explicit fp32 reference, max |diff| 1.79e-07;
  - causality (0.0 before t, 1.20 at t);
  - CPU resume bitwise, 8/8 losses.
- **Other features.**
  - Seeding: `torch.manual_seed`; data comes from `np.random.default_rng([seed, k])` per micro-batch, so the loader has no state; no `hash()`.
  - JSONL logging of config, step, eval and end records.
  - Atomic checkpoint; `--resume`; `--stop_at` splits a run without changing its LR schedule.
  - Polls `design4x5.poll_until_free(600)` before any CUDA call.
  - Aborts when allocated memory exceeds device memory. WDDM spills to system RAM instead of raising OOM: R0 b16 ran at 6,945 vs 93,973 tok/s.
- **GPU resume check, on real shards** (R0, ctx 1024, b8, 400 steps straight vs 200 + resume 200 in a fresh process):
  - **Default mode: RED.** 262/400 step losses differ, the first at **step 139, before the resume point**. Max |dloss| is 2.22e-04, and val loss at step 400 is 6.74183 in both runs. The source is run-to-run GPU nondeterminism, not the resume path.
  - **`--deterministic`: GREEN.** 400/400 step losses and 4/4 evals are bitwise equal, with val loss 6.742448 at step 400. This mode costs 14.4% of throughput (82,501 vs 96,384 tok/s).
- **Real-data runs.**
  - R0 b8, 400 steps: val loss 7.820 → 6.742 (3.28M tokens), 96,384 tok/s.
  - R1 b4, 100 steps: val loss 7.184, 57,348 tok/s, MFU 0.327.
  - An earlier R1 b8 real run fell from about 55k to about 21k tok/s after step ~40. The cause is **unverified**: the poll only gates the start of a run, so a GPU job from another lane is possible but was not observed. That run is excluded from the table.

## 3. MFU

**Setup.** ALiBi twin, bf16, SDPA, no `torch.compile`, synthetic tokens (throughput does not depend on the data). MFU = (6N + 6·L·ctx·d)·tok/s / 26.77e12, with N measured. Test `K0_mfu_measured` was red on a stub, then green on 14/14 fitting runs. Logs are in `WD/runs/mfu_*`, and `WD/analyse_mfu.py` rebuilds the table below.

| rung | ctx | batch | steps | tok/s | MFU vs 26.77 | peak mem GiB |
|---|---|---|---|---|---|---|
| R0 | 1024 | 8 | 300 | 93,973 | 0.163 | 5.63 |
| R0 | 1024 | 4 | 300 | 90,158 | 0.157 | 2.88 |
| R0 | 4096 | 2 | 300 | 81,892 | 0.171 | 5.69 |
| R0 | 4096 | 1 | 300 | 78,324 | 0.164 | 2.93 |
| R1 | 1024 | 8 | 100 | 55,522 | 0.317 | 6.25 |
| R1 | 1024 | 4 | 100 | 55,913 | 0.319 | 3.29 |
| R1 | 4096 | 2 | 100 | 35,666 | 0.251 | 6.40 |
| R1 | 4096 | 1 | 100 | 41,066 | 0.289 | 3.44 |
| R2 | 1024 | 8 | 40 | 19,073 | 0.236 | 7.23 |
| R2 | 1024 | 4 | 40 | 35,433 | 0.438 | 3.97 |
| R2 | 4096 | 2 | 40 | 14,115 | 0.214 | 7.46 |
| R2 | 4096 | 1 | 40 | 24,005 | 0.364 | 4.20 |
| R3 | 1024 | 4 | 30 | 18,291 | 0.545 | 5.64 |
| R3 | 4096 | 1 | 30 | 11,697 | 0.423 | 5.99 |

The following runs were red because allocated memory exceeded 8 GiB (a WDDM spill): R0 c1024 b16 (11.15 GiB), R3 c1024 b8 (8.68 GiB), R3 c4096 b2 (9.03 GiB). The R2 b8 rows ran at 7.2–7.5 GiB of 8 and are memory-pressured: 0.236 vs 0.438 at b4.

**The 26.77 TF peak.**
- **Re-measured today** with the C24 recipe (bf16 matmul, 10 warmup + 50 timed, 5 repetitions; `WD/peak.jsonl`):
  - 4096³: 26.17 / 32.45 / 31.56 / 30.59 / 31.12 TF.
  - 8192³: 30.38 / 29.87 / 29.46 / 29.08 / 28.69 TF, falling as the SM clock dropped to 2340 MHz (max 3105 MHz).
  - The attained range is 26.17–32.45 TF. Against the 4096 median of repetitions 2–5 (31.34 TF), every MFU above scales by 0.854.
- **Where 26.77 and C24 are recorded.**
  - `house-events.jsonl:3670` (Cameron, finding `attained_peak`: "4096x4096 bf16 matmul, 50 iters (10 warmup) … 26.77 TFLOPS attained in 0.257s wall") and `:3685` ("C24 done … attained_peak=26.77 TFLOPS"). This file is **gitignored** (`.gitignore:54`).
  - `tests/foreman/phase_k/instances/yukawa.py:76` cites it (the directory is untracked), as does `PHASE_K_CONTRACT.md:91`.
  - The script and results that measured it are **outside the repository**, in the session scratchpad: `cal_C24_peak.py`, `cal_C24_peak.md:30`, and `cal_C24_peak_results.json:17,28` (26.769).
  - `SESSION_2026_09_21.md:220` in the scratchpad gives the range as "26.77–32.58".
  - No tracked file records the measurement, so **as a repository record it is unverified**.
  - No vendor bf16 tensor spec for the RTX 4060 Laptop GPU was fetched, so the figure is verifiable only as a measurement on this card. It lies inside today's range.

## 4. Hours per full run (20 tokens per parameter)

Hours = 20·N / measured tok/s, which needs no peak figure. Each row uses the fastest fitting batch per rung and context.

| rung | ctx | tok/s | hours @20N (measured N) | hours @20·Np | Y4 @MFU 0.35 (this ctx) | contract table (ctx 4096) |
|---|---|---|---|---|---|---|
| R0 | 1024 | 93,973 | 0.43 | 0.43 | 0.20 | — |
| R0 | 4096 | 81,892 | 0.49 | 0.49 | 0.24 | 0.2 |
| R1 | 1024 | 55,913 | 2.33 | 2.33 | 2.12 | — |
| R1 | 4096 | 41,066 | 3.18 | 3.18 | 2.62 | 2.6 |
| R2 | 1024 | 35,433 | 7.99 | 7.98 | 9.98 | — |
| R2 | 4096 | 24,005 | 11.79 | 11.79 | 12.26 | 12.3 |
| R3 | 1024 | 18,291 | 37.54 | 37.53 | 58.47 | — |
| R3 | 4096 | 11,697 | 58.71 | 58.69 | 70.92 | 70.9 |

Y4 recomputed from its own formula at ctx 4096 reproduces the contract's 0.2 / 2.6 / 12.3 / 70.9. At ctx 1024 the same formula gives 0.20 / 2.12 / 9.98 / 58.47.

## 5. Fetch orders (§7 "Still [U]")

Three haiku nurses fetched these records. Wilson re-checked the arXiv items against the arXiv API (`WD/arxiv_records.xml`) and the physics items against the Crossref and INSPIRE APIs (`WD/cr_*.json`, `WD/insp_*.json`).

| item | title | authors | year | identifier | one verbatim sentence | label |
|---|---|---|---|---|---|---|
| FineWeb-Edu | HuggingFaceFW/fineweb-edu dataset card | HuggingFaceFW | card rev 87f0914 | README.md l.632 | see §1 | verified (Wilson, curl of the card; the nurse's README fetch failed) |
| Kaplan 2020 | Scaling Laws for Neural Language Models | Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei | 2020 | arXiv:2001.08361 | "We study empirical scaling laws for language model performance on the cross-entropy loss." | verified |
| Chinchilla 2022 | Training Compute-Optimal Large Language Models | Hoffmann, Borgeaud, Mensch, … Vinyals, Sifre (22 authors) | 2022 | arXiv:2203.15556 | "We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget." | verified |
| BABILong | BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack | Kuratov, Bulatov, Anokhin, Rodkin, Sorokin, Sorokin, Burtsev | 2024 | arXiv:2406.10149 | "BABILong includes a diverse set of 20 reasoning tasks, including fact chaining, simple induction, deduction, counting, and handling lists/sets." | verified; the abstract has no "multi-hop" |
| Looped transformers (1) | Looped Transformers as Programmable Computers | Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos | 2023 | arXiv:2301.13196 | "We present a framework for using transformer networks as universal computers by programming them with specific weights and placing them in a loop." | verified |
| Looped transformers (2) | Looped Transformers are Better at Learning Learning Algorithms | Yang, Lee, Nowak, Papailiopoulos | 2023 | arXiv:2311.12424 | "Transformers have demonstrated effectiveness in in-context solving data-fitting problems from various (latent) models, as reported by Garg et al." | verified |
| DEQ | Deep Equilibrium Models | Bai, Kolter, Koltun | 2019 | arXiv:1909.01377 | "We present a new approach to modeling sequential data: the deep equilibrium model (DEQ)." | verified |
| APPNP | Predict then Propagate: Graph Neural Networks meet Personalized PageRank | Gasteiger (Klicpera), Bojchevski, Günnemann | 2018 (ICLR 2019) | arXiv:1810.05997 | "In this paper, we use the relationship between graph convolutional networks (GCN) and PageRank to derive an improved propagation scheme based on personalized PageRank." | verified; the nurse's copy dropped "In this paper," and Wilson corrected it |
| Dyson 1949a | The Radiation Theories of Tomonaga, Schwinger, and Feynman | F. J. Dyson | 1949 | Phys. Rev. 75, 486–502; doi:10.1103/PhysRev.75.486 | "A unified development of the subject of quantum electrodynamics is outlined, embodying the main features both of the Tomonaga-Schwinger and of the Feynman radiation theory." | verified (APS page via the nurse, plus INSPIRE and Crossref) |
| Dyson 1949b | The S Matrix in Quantum Electrodynamics | F. J. Dyson | 1949 | Phys. Rev. 75, 1736–1755; doi:10.1103/PhysRev.75.1736 | "Scattering processes, including the creation and annihilation of particles, are completely described by the S matrix of Heisenberg." | verified against INSPIRE's copy of the APS abstract (record 9087) and Crossref metadata; the APS page itself was blocked for the nurse |
| Heisenberg 1943 | Die „beobachtbaren Größen" in der Theorie der Elementarteilchen | W. Heisenberg | 1943 | Z. Phys. 120, 513–538; doi:10.1007/BF01329800 | "Die bekannten Divergenzschwierigkeiten in der Theorie der Elementarteilchen zeigen, daß die zukünftige Theorie in ihren Grundlagen eine universelle Konstante von der Dimension einer Länge enthalten wird, die in die bisherige Form der Theorie offenbar nicht widerspruchsfrei eingebaut werden kann." | verified against INSPIRE's Springer-sourced abstract (record 2930845) and Crossref; the Springer page is login-walled |
| Higgs 1964 (PRL) | Broken Symmetries and the Masses of Gauge Bosons | Peter W. Higgs | 1964 | Phys. Rev. Lett. 13, 508–509; doi:10.1103/PhysRevLett.13.508 | none: neither Crossref nor INSPIRE carries an abstract; only the title is verbatim | metadata verified; **sentence unverified** |
| Higgs 1964 (PL) | Broken symmetries, massless particles and gauge fields | P.W. Higgs | 1964 | Phys. Lett. 12, 132–133; doi:10.1016/0031-9163(64)91136-9 | none: no abstract in either record; ScienceDirect returned 403 | metadata verified; **sentence unverified** |

## Limits

- **Probe length.** The R2 and R3 MFU values come from 40- and 30-step probes on synthetic tokens. Multi-hour thermal behaviour is unmeasured: the 8192³ peak fell 5.6% across five repetitions. The R2/R3 hours are therefore **unverified for full-length runs**.
- **Excluded levers.** No `torch.compile` and no fused or chunked cross-entropy were tried. At R0, the fp32 logits dominate memory (5.63 GiB at b8), which caps the batch.
- **bf16 bias rounding.** Its effect on the ALiBi bias was not measured.
- **Default-mode reproducibility.** Default GPU mode is not bitwise reproducible run to run. Only `--deterministic` is.
- **Nurses.** 4 were dispatched: 3 haiku (fetch) and 1 sonnet (harness).
- **Processes.** None of this lane's processes are left running.
- **Disk.** `WD/runs` holds 7.4 GB of logs and checkpoints (the R3 checkpoints included).
