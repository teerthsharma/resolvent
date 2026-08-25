# Benchmark corpora — not redistributed

The measurement code expects these files here, but they are **not committed**:
they are third-party datasets with their own licences, and every one is
fetchable upstream. Excluding them keeps this repository to its own work.

| file | source | citation |
|---|---|---|
| `cogs_{train,dev,test,gen}.tsv` | [github.com/najoungkim/COGS](https://github.com/najoungkim/COGS) | Kim & Linzen, *COGS: A Compositional Generalization Challenge Based on Semantic Interpretation*, EMNLP 2020, arXiv:2010.05465 |
| `scan_{simple,length,addprim_jump}_{train,test}.txt` | [github.com/brendenlake/SCAN](https://github.com/brendenlake/SCAN) | Lake & Baroni, *Generalization without Systematicity*, ICML 2018, arXiv:1711.00350 |
| `tinystories_20k.txt` | [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759 |

`tinystories_20k.txt` is a 20,000-line head of the TinyStories train split, used
as a byte-level corpus for the small from-scratch parity runs.

**Every published number in `DONE.md` that depends on these files names the file
it used.** Re-fetching upstream reproduces them; nothing here has been modified
from source.
