# 05 — REPAIRS: every break as a plan with a price and a verification number

MERCURY · LESTRADE — pricing, scheduling, execution. This book does not interpret; it prices,
orders and verifies. Written 2026-09-05 under `docs/canon/CHARTER.md` at HEAD `99777ab`.

**Preface — what this book decides, and the verse that decides it first.** The book decides
the order in which the broken things are repaired, the price of each repair in GPU-seconds
and evenings, and the single number that shows each repair worked. The first decision is
verse 05.2: the box that decides every research number no longer runs the stack it was
certified on — `torch 2.14.0+cpu` and `numpy 2.4.6` were installed over the certified
`torch 2.5.1+cu121` / `numpy 1.26.4`, CUDA is not visible to the installed torch, and
`pytest -q` at the root collects 2,580 tests and runs 0 behind 22 collection errors (RUN
2026-09-06). Until 05.3 lands, no number in any book can be RUN on the certified device, so
05.3 is evening 1 and nothing is scheduled before it. The book then prices, in order: the
device re-certification split into a ≤20 GPU-s half and a 523.9 s half (05.5); the three
harness debts of `scripts/v15_r1.py` with the test that proves each (05.6–05.8); the
prose-coupled test policy and the registry the deletion severed (05.9); the Gate-0
circularity as a written rule (05.10); the five open rulings with the default each takes
(05.11); the package, the residual and the unpinned sources (05.12); the documentation
discipline with its census command (05.13); the first five evenings and the critical path
(05.14); and a scheduling row for every census item another book closes (05.15).

**Price basis, stated once.** GPU-seconds are quoted from the record's own unit prices on
the certified RTX 4060 Laptop at `ab5b485`: one 150-step control cell $1.524$ s `[FITTED]`,
one shape cell on the softmax corner $1.681$ s `[FITTED + RUN]`, the capped cell $1.884$ s,
fixed cost per invocation $4.0$ s (`READ docs/PLAN.md:971-978 @ 99777ab`); one ragged LM step
at the Q3 shape $0.299025$ s `[MEASURED]` (`READ COSTS.md:212,239 @ 99777ab`); the full
device certificate $523.9$ s `[MEASURED]` (`READ COSTS.md:62 @ 99777ab`). Every one of these
is a per-op floor under a $2.0\times$–$6.6\times$ dispatch gap (`READ scale/m3_flops.py:101-121 @
99777ab`) and a laptop clock that is not stationary at $\pm12$ per cent
(`READ docs/PLAN.md:967 @ 99777ab`); prices are therefore "at least" until 05.5 re-measures
them on the re-certified stack. An evening is one sitting; the first five evenings are each
capped at $20$ GPU-s by the budget **declared in 05.1** — `[ASSUMED]`, with its reason
printed there. The CHARTER carries no such cap: RUN `grep -c "20 GPU" docs/canon/CHARTER.md`
reads $0$, and the book-05 scope row is `docs/canon/CHARTER.md:269` (live), which names
"the price in evenings and GPU-s" and no number.

---

### 05.1 — The ordering rule: cheapest-decisive-first, with the evening cap

**Statement.** Repairs are ordered by the pair (price of the kill, decisiveness of the
kill), cheapest first, and ties are broken toward the repair whose kill withdraws more of the
canon if it fires. A repair whose kill cannot fire on the tree as it stands is not scheduled
(reachability rule). The order is: environment (05.2–05.4) → certificate (05.5) → harness
(05.6–05.8) → tests (05.9) → gate rule (05.10) → rulings (05.11) → package and sources
(05.12) → documentation census (05.13). Every evening among the first five costs at most
$20$ GPU-s; the first evening above the cap is the full certificate rerun at $523.9$ s, and it
is evening 6.

**The cap, declared here because nothing else declares it.** $20.0$ GPU-s per evening among
the first five is an `[ASSUMED]` budget of this book, not an inherited threshold. Its
reason: the largest single repair price among the first five is 05.7's interleaved 8 cells
at $17.63$ s DERIVED, and $20.0$ leaves one invocation's fixed cost ($4.0$ s, READ
`docs/PLAN.md:978 @ 99777ab`) of headroom above it without admitting a second 8-cell run —
so the cap is the smallest round number that admits every priced repair singly and refuses
every pair of them. It is **not** read from the CHARTER: RUN `git ls-tree -r 99777ab
--name-only | grep -ci canon` reads $0$ (the canon is untracked at the pin, `git status`
reads `?? docs/canon/`), and RUN `grep -c "20 GPU" docs/canon/CHARTER.md` on the live file
reads $0$. Every `docs/canon/` citation in this book is therefore to the **live** file with
no `@ 99777ab` pin, and the canon's own pin is its birth commit, which does not exist yet.

**Hypotheses.** The certified device is the RTX 4060 Laptop, $7.996$ GiB, `sm_8.9`
(`READ COSTS.md:56-58 @ 99777ab`); the driver on the box reads `595.79` with
`CUDA Version: 13.2` (RUN `nvidia-smi`, 2026-09-05), which serves a `cu121` wheel by the
driver's backward compatibility (`[ASSUMED]` — the toolkit compatibility rule is vendor
documentation, not measured here; 05.3's verification command is what measures it). No
Kaggle number enters any price (V-22).

**Evidence.** Unit prices: `READ docs/PLAN.md:971-978 @ 99777ab`; `READ COSTS.md:62,212 @
99777ab`. The dispatch gap: `READ scale/m3_flops.py:101-121 @ 99777ab`. The cap of $20$ GPU-s:
`[ASSUMED]`, declared above with its reason; the book-05 scope row it is **not** taken from
is `READ docs/canon/CHARTER.md:269` (live, no pin — RUN `git ls-tree -r 99777ab --name-only
| grep -ci canon` reads $0$).

**Mechanism.** M-8 (pricing every arm at one arm's rate — each price below names its arm);
P-8 (an upper bound stated as a price — every price carries "at least"); D-6 (a repair
written and never started — every repair here has an evening number and a verification
command).

**Kill.** Any verse of this book whose evening lies among the first five and whose priced
GPU-seconds exceed $20$ at the quoted unit prices. Frozen threshold: $20.0$ GPU-s. Instrument:
the arithmetic printed in each verse's Kill line. Price: $0$. Planted negative: the full
certificate at $523.9$ s, which the rule must refuse from the first five — and does (it is
evening 6).

**The rule fired on this book's own first draft, and the split is applied here.** 05.6's
kill needs an identical-seed pair at `--device cuda --seeds 0 --arms softmax` and a third run
at `--s 32 --arms arm_pl` (a gated arm, because the bind row the `--s` clause reads is
emitted only for `GATED_ARMS = ("arm_pl", "arm_smprime")`, RUN
`grep -n "GATED_ARMS *=" scripts/v15_r1.py` reads `:146`), priced
$2\times(1.524+4.0)+(1.884+4.0)=16.93$ s — $11.05$ `[FITTED]` plus a $5.88$ s `[ASSUMED]`
ceiling, because both unit prices are defined at $s=64$ and the third run is at $s=32$
(05.6's price paragraph, and its kill clause (v) audits the ceiling on the same run); 05.7's
needs 8 interleaved cells over arms
$\{$`arm_pl`, `softmax`$\}$ and seeds $0$–$3$, priced $4\times1.884+4\times1.524+4.0=17.63$ s.
The two need different `--arms` and different `--seeds`, so they cannot share an invocation,
and a single evening 3 would cost $16.93+17.63=34.56$ GPU-s — above the frozen $20.0$. Under
this verse's own If-killed the evening is split, most-decisive first: **evening 3** is 05.7's
8 interleaved cells ($17.63$ s, the timer is the instrument every later price depends on),
**evening 3b** is 05.6's pair plus the `--s 32` run ($16.93$ s), and **evening 3c** is 05.8's
one `arm_smprime` cell ($1.884+4.0=5.88$ s) — 3b and 3c cannot be merged
($16.93+5.88=22.81$ s, above the cap). 05.14 carries the three sittings and their sum. **The
phrase "05.6's pair is inside the same invocation set" is deleted from 05.14**: it named a
shared invocation that the differing `--arms` and `--seeds` make impossible.

**If killed.** The offending verse's evening is split: the ≤20 GPU-s sub-run first (the
one that decides the most), the remainder pushed to evening 6 or later. Hypotheses: the
verse's kill is decidable on the sub-run alone. Evidence: the verse's own price arithmetic.
Kill of the replacement: the sub-run does not decide the verse's kill (its kill number is
not produced by the sub-run) — then the verse leaves the first five entirely and the
critical path in 05.14 lengthens by one evening.

**Terminal.** The order stands as a list even if every price is wrong: the canon licenses
"repair in this order" and withdraws every GPU-second figure until 05.5 re-measures them.

---

### 05.2 — B1: the environment break, with its mechanism

**Statement.** The installed stack on the deciding box is not the certified stack, and the
drift explains exactly 13 of the 22 root collection errors; the other **9** are broken
imports of test modules deleted at `c71527a` (05.9), and **none** of the 9 opens a prose
document. Installed (RUN `pip list`, 2026-09-05): `torch 2.14.0` (`torch.__version__`
reads `2.14.0+cpu`, `torch.version.cuda` reads `None`, `torch.cuda.is_available()` reads
`False`), `torchvision 0.20.1+cu121`, `torchaudio 2.5.1+cu121`, `numpy 2.4.6`,
`pandas 2.1.2`, `transformers 5.3.0`, `ripser 0.6.14`, `persim 0.3.8`, `scipy 1.17.1`,
`pytest 9.0.3`, `triton-windows 3.7.1.post27`, Python `3.11.9`. Certified
(`READ COSTS.md:56-60 @ 99777ab`; `READ requirements.txt:33,36 @ 99777ab`):
`torch 2.5.1+cu121`, `numpy 1.26.4`, Python `3.11.9`. Two causal chains produce the 13
environment errors:

1. `torchvision 0.20.1+cu121` was built against `torch 2.5.1`; under `torch 2.14.0` its
   import raises `RuntimeError: operator torchvision::nms does not exist` and, on re-entry,
   `AttributeError: partially initialized module 'torchvision' has no attribute 'extension'
   (most likely due to a circular import)` (RUN). `transformers 5.3.0` imports
   `torchvision.transforms.InterpolationMode` from `image_utils.py:53` on the path to
   `modeling_utils`, so `from transformers import PreTrainedModel` raises
   `ModuleNotFoundError: Could not import module 'PreTrainedModel'` (RUN), and
   `ceq.hf.modeling_ceq` fails with it (RUN). **8 files** error on this chain:
   `tests/chase/test_resume_checkpoint.py`, `tests/chase/test_scale_hazards.py`,
   `tests/gate0/test_g02_resume.py`, `test_g03_persist.py`, `test_g12_train_operator.py`,
   `test_g13_beta_learnable.py`, `test_g15_noise_floor.py`, `test_g16_lrt_pinned.py`.
2. `pandas 2.1.2` (site-packages dated 2026-06-10) is compiled against the numpy-1 ABI;
   under `numpy 2.4.6` its `interval.pyx` raises `ValueError: numpy.dtype size changed …
   Expected 96 from C header, got 88 from PyObject`; `ripser` and `persim` import pandas and
   fail with it (RUN). **5 files** error on this chain: `tests/foreman/test_topology_washout.py`,
   `tests/jupiter/test_coherence_and_rip.py`, `test_page_lattice.py`, `test_page_trend.py`,
   `test_v20_r15_it6_q2_outside_bound.py`.

Sum: $8+5=13$ environment errors; $22-13=9$ residual errors, and the residual is **not**
prose-coupled. RUN `python -m pytest -q --collect-only -p no:cacheprovider`, 2026-09-06,
classified by raised exception: **zero** `FileNotFoundError`, **zero** `AssertionError`,
**zero** `.md` name anywhere in the 22 tracebacks. The 9 are a **fourth class** — a test
importing a *test module* deleted at `c71527a`:

| file | raises |
|---|---|
| `tests/jupiter/test_v20_r15_it26_live_claim.py` | `ModuleNotFoundError: No module named 'tests.jupiter.test_v20_r15_it20_citation_freeze'` (`:101`) |
| `tests/jupiter/test_v20_r15_it29_overturns_can_fail.py` | `… 'tests.jupiter.test_v20_r15_it27_star_lands_and_overturns'` (`:48`) |
| `tests/jupiter/test_v20_r15_it30_phrase_grammar.py` | `… 'tests.jupiter.test_v20_r15_it27_star_lands_and_overturns'` (`:47`) |
| `tests/mars_v20/test_it18_census_zero_bad_was_never_a_citation_claim.py` | `… 'tests.jupiter.test_v20_r15_it14_theory_table'` (`:27`) |
| `tests/mars_v20/test_it18_wing_identity_is_self_certified.py` | `ImportError: cannot import name 'test_v20_r15_freeze_manifest' from 'tests.saturn'` (`:36`) |
| `tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py` | `ImportError: … 'test_v20_r15_freeze_manifest' …` (`:34`) |
| `tests/mars_v20/test_it25_the_repair_record_reads_as_the_defect.py` | `… 'tests.jupiter.test_v20_r15_it20_citation_freeze'` (`:31`) |
| `tests/saturn/test_v20_r15_it19_wing_identity.py` | `ImportError: … 'test_v20_r15_freeze_manifest' …` (`:30`) |
| `tests/saturn/test_v20_r15_it29_planted_negative_rederived.py` | `ModuleNotFoundError: … 'tests.saturn.test_v20_r15_it27_wing_arm_citation'` (`:27`) |

Exception census over the same run (RUN, `grep -E "^E   " | sort | uniq -c`): $7$
`AttributeError: partially initialized module 'torchvision'`, $1$ `RuntimeError: operator
torchvision::nms does not exist`, $8$ `ModuleNotFoundError: Could not import module
'PreTrainedModel'`, $5$ `ValueError: numpy.dtype size changed`, $3+2+2+1+1=9$
test-module import breaks. Arithmetic: $7+1+8+5=21$ error lines over the 13 environment
files, because each of the 8 chain-1 files raises **two** lines (the `torchvision` failure —
`RuntimeError` on the first import, `AttributeError` on the 7 re-entries — and the
`PreTrainedModel` failure it causes), $8\times2+5\times1=21$; and $9$ lines over the 9
fourth-class files, one each. $13+9=22$.

**The drift is dated, and the dating evidence is `[ASSUMED]`, because an mtime is a filesystem
write and not a producer record.** The `torch`, `functorch`, `numpy` and `numpy-2.4.6.dist-info`
directories in site-packages carry mtime 2026-09-03 23:30–23:31 (RUN
`ls --time-style=long-iso`), three days after the certificate of 2026-08-31 22:26. That reading
is tagged **`[ASSUMED] — mtime is a filesystem write, not a producer record`**, and this tree
proves why in its own `results/`: RUN over `results/`, $331$ files carry $176$ distinct mtimes,
and the $18$ files 05.2's Terminal names share the single minute 2026-09-04 00:41 across only
**four** sub-millisecond stamps ($1788462678.5325522$, $\ldots5385008$ on eleven files,
$\ldots5439267$ on two, $\ldots5445810$ on four) — eighteen files, four write instants, one
bulk handling event. A stamp that records eighteen files as four writes cannot be read as
eighteen measurements, and the same objection applies to a package directory.

The **dated** half of the drift is carried on the install record instead, which pip writes once
at install and never touches again: `numpy-2.4.6.dist-info/` holds `INSTALLER`, `RECORD`,
`WHEEL`, `METADATA`, `DELVEWHEEL` and `entry_points.txt`, all at 2026-09-03 23:30 (RUN
`ls --time-style=long-iso -l`). Its **undated** half is stronger still and needs no clock at
all: the directory's *name* is `numpy-2.4.6.dist-info` and there is no `numpy-1.26.4.dist-info`
beside it, so the interpreter that answers `import numpy` is carrying a distribution the
certificate never named — a fact about which files exist, not about when. The canon reads the
drift off the name and quotes the date `[ASSUMED]`.

**Hypotheses.** The interpreter is `<home>\AppData\Local\Programs\Python\Python311\python.exe`
— a machine-wide CPython 3.11 install, not a repo-local venv — (RUN); the root suite is
collected with `pytest.ini`'s `norecursedirs` excluding `attic` and
`kaggle` (`READ pytest.ini:19 @ 99777ab`); the counts are of files, not of tests.

**Evidence.** RUN `python -m pytest -q --collect-only -p no:cacheprovider` at the root,
**2026-09-06**: `2580 tests collected, 22 errors in 15.35s`, `Interrupted: 22 errors during
collection` (a second execution the same day read `14.08s`; the wall-clock is quoted as a
range $14.08$–$15.35$ s and never to the second). RUN import probes as quoted; RUN exception
census as tabled. The counts $2971$ / $26$ / $17.69$ s that this verse carried before repair
are **withdrawn**: they do not reproduce on the tree and no journal holds them, so under the
number rule (P-1, P-2) their only home was a prior draft and they are struck from the book.
The charter's own row reads **13** collection errors (`READ docs/canon/CHARTER.md:227`,
live, no pin — RUN `git ls-tree -r 99777ab --name-only | grep -ci canon` reads $0$, so no
`docs/canon/` line is citable at `99777ab`) — that is the environment half, and it is
confirmed exactly by the RUN here; the RUN reads 22 because the 9 fourth-class files, present
on disk and importing test modules deleted at `c71527a`, also error at collection. Both
counts are RUN and both are right about what they count.

**Mechanism.** V-22 (a constant carried in from another system — every determinism and cost
number in `COSTS.md` §1 was measured on the 2.5.1 stack); P-3 (a stale claim never
retracted — `requirements.txt` still says "installed build is 2.5.1+cu121"); V-16 (an
instrument that cannot measure reporting a pass — a suite that cannot collect reports no
failures).

**Kill.** After 05.3's install, the same collection command reads any number of errors other
than exactly the **9** fourth-class files tabled above. Frozen threshold: environment errors
$=0$, total errors $=\mathbf{9}$ until 05.9 lands, then $0$. Instrument: `python -m pytest -q
--collect-only -p no:cacheprovider`. Price: $0$ GPU-s, $14$–$16$ s CPU (RUN $14.08$ and
$15.35$ s). Planted negative: the present tree, which reads 22 (RUN) — the kill fires on it
today; and, for the *lower* side, the pinned tree with any one of the 9 files removed, on
which the count reads $8$ and the kill fires again, so the threshold is two-sided and is not
satisfied by a repair that merely deletes evidence.

**If killed** (errors remain after the pin). The residual errors are classified by a
**four**-class `grep` over the tracebacks, the fourth class added because the three-class
classifier this verse shipped before repair could not name a single one of the 9 residuals
and would have sent every one of them to `CORRECTIONS.md`:

| class | selector on the traceback | disposition |
|---|---|---|
| ABI | `numpy.dtype size changed` | 05.3 step 3 |
| torch chain | `PreTrainedModel` or `torchvision` | 05.3 step 2 |
| prose | a `.md` name in the traceback | 05.9's policy |
| **deleted test module** | `ModuleNotFoundError: No module named 'tests.` or `ImportError: cannot import name 'test_` — confirmed by `grep -nE "^\s*(from\|import) tests\." <file>` naming a path absent from disk | 05.9's restore-or-retire pass |

Hypotheses: every residual error names one of the four classes. Evidence: RUN classification
above — the four-class rule assigns all 22 of today's errors ($13+9$) and leaves none over.
Kill of the replacement: an error outside the four classes — then the environment is not the
only break and the residual file is filed in `CORRECTIONS.md` as a new verse with its own
traceback line. That is the last link.

**Terminal.** The canon licenses "the deciding box ran a stack it was not certified on from
2026-09-03 23:30" as a RUN fact, and withdraws every RUN number dated after 2026-09-03 23:30
that does not print its own stack string — the flat `torch` key 05.4 freezes as the predicate,
`box.torch` where a producer writes the richer record. That withdrawal is **not** vacuous, and the
parenthetical this verse carried before repair — "none exists in `results/`" — is false: RUN
`find results -type f -newermt "2026-09-04 00:40" ! -newermt "2026-09-04 00:42"` reads
**18** paths, not 1, and the newest is not the Kaggle log —
`results/kaggle_v17k_output/ceq-v17-k.log`, `results/k_cert_local.json`,
`results/k_data_manifest.json`, `results/r4_price_probe.json`, `results/v15_r1.jsonl`,
`results/v17k_r4_floor.jsonl`, `results/v17k_r4_retake.jsonl`, `results/v17k_watch.log`,
`results/v20_m14_cheeger.txt`, `results/v20_r15_it10_mercury_rescore.jsonl`,
`results/v20_r15_it11_jupiter_census.jsonl`, `results/v20_r15_it11_mercury_draw4.jsonl`,
`results/v20_r15_it11_mercury_smprime.jsonl`, `results/v20_r15_it30_mercury_landings.txt`,
`results/v20_r15_it30_mercury_screen.txt`, `results/v20_r15_it6_seeds8_15.jsonl`,
`results/v20_r15_it8_armpl_b.jsonl`, `results/v20_r15_it8_armpl_seeds8_15.jsonl`. Those
files **do** print a stack: `results/v20_r15_it8_armpl_b.jsonl`'s header carries
`"torch": "2.5.1+cu121"`, `"deterministic_warn_only": true`, `"cublas_workspace_config":
":4096:8"` (RUN). What they print is a **flat** `torch` key and **no `box` record** (RUN: the
union of all row keys of `results/v15_r1.jsonl` over its 26 lines contains `torch`,
`torch_version` and no `box`) — which is the defect 05.4 must catch, and 05.4's kill is
re-frozen on the flat key there. Two further RUN facts bound what the mtimes mean: all 18
share the minute 2026-09-04 00:41, which is a bulk filesystem touch and **not** 18 write
times, so the mtime is evidence that the files were *handled* after the drift, never that
they were *measured* after it; and the drift-window census is therefore a trigger for 05.4's
stack census, not a withdrawal of the 18 by itself.

---

### 05.3 — B1: the pin set as a specification, with the order of installation and the number it must print

**Statement.** The certified stack is restored in the system interpreter in this order, each
step verified before the next:

| step | command (executed by the author; no writer runs an installer) | verification | number it must print |
|---|---|---|---|
| 1 | `pip uninstall -y torch torchvision torchaudio` | `python -c "import torch"` | `ModuleNotFoundError` |
| 2 | `pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` | `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `2.5.1+cu121 12.1 True NVIDIA GeForce RTX 4060 Laptop GPU` |
| 3 | `pip install numpy==1.26.4` | `python -c "import numpy, ripser, persim; print(numpy.__version__)"` | `1.26.4` and no exception |
| 4 | none (`transformers 5.3.0`, `scipy 1.17.1`, `pytest 9.0.3` are already at the pinned versions — RUN) | `python -c "from transformers import PreTrainedModel; import ceq.hf.modeling_ceq; print('ok')"` | `ok` |
| 5 | none | `python -m pytest -q --collect-only -p no:cacheprovider` | `9 errors` (the deleted-test-module set tabled in 05.2 and handled in 05.9), then `0 errors` after 05.9 |
| 6 | none | `requirements.txt:6`'s own re-derivation line, quoted verbatim: `python -c "import torch,transformers,numpy,scipy; print(torch.__version__, transformers.__version__, numpy.__version__, scipy.__version__)"` | `2.5.1+cu121 5.3.0 1.26.4 1.17.1` |

**Step 6's field order is the cited line's, not this book's.** The string this verse froze before
repair — `2.5.1+cu121 1.26.4 1.17.1 5.3.0` — is a **different field order** from the command it
named as its authority: `requirements.txt:6` prints `torch, transformers, numpy, scipy` (RUN
`sed -n '6p' requirements.txt`), so its output is `2.5.1+cu121 5.3.0 1.26.4 1.17.1`. A frozen
string in one order attributed to a command that prints another is P-6, line-reference drift
turned into a false quotation, and a verifier executing `:6` and comparing against the old
string would have read a failure on a correctly pinned box. The command and the string above are
now the cited line's, character for character; this book adds no command of its own at step 6.

`torchvision 0.20.1+cu121` and `torchaudio 2.5.1+cu121` are the builds already on the box
(RUN); pinning them beside torch keeps step 2 from resolving a torchvision that would
re-create chain 1. `pandas 2.1.2` is left as installed: it is compiled for numpy 1 and is
the reason step 3 is `1.26.4` and not a numpy-2 pandas rebuild (the certified stack had
numpy `1.26.4`; a pandas upgrade would move a package the certificate never named).

**Hypotheses.** Network access to `download.pytorch.org` and PyPI at install time; Python
`3.11.9` unchanged; the wheel index still serves the version. Both index facts are RUN,
2026-09-05: `pip index versions torch --index-url https://download.pytorch.org/whl/cu121`
lists `2.5.1+cu121` as its newest; `pip index versions numpy` lists `1.26.4`. No local wheel
cache holds either (RUN `pip cache list` — empty for torch and numpy).

**Evidence.** Certified versions `READ COSTS.md:56-60 @ 99777ab`; pins
`READ requirements.txt:33-39 @ 99777ab`; the re-derivation command `READ requirements.txt:6 @
99777ab`; the Kaggle stack that must **not** be pinned here — `python 3.12.13`,
`torch 2.10.0+cu128` `[RUN]` off the Kaggle log — `READ requirements-kaggle.txt:12-13 @
99777ab`, `READ ceq/compat.py:5-6 @ 99777ab`; index availability RUN as above.

**Mechanism.** V-22 (no Kaggle version enters the local pin); P-2 (the number's home is the
verification line, printed, not a chat); M-1 (train and eval on different preprocessing —
here, different torch builds — refused by step 6's single line).

**Kill.** Step 2's verification prints anything other than `2.5.1+cu121 12.1 True NVIDIA
GeForce RTX 4060 Laptop GPU`, or step 3's raises. Frozen: exact string equality on the
version and CUDA fields, `True` on availability. Instrument: the `python -c` lines above.
Price: $0$ GPU-s; install wall-clock `NOT MEASURED — needs the install itself`. Planted
negative: the present box, on which step 2's verification prints `2.14.0+cpu None False
None` (RUN) — the kill fires on it today.

**If killed** — *replacement 1, a fresh interpreter*: `py -3.11 -m venv .venv` at the
repository root (already ignored: `READ .gitignore:12-13 @ 99777ab`, where `:12` reads
`.venv/` and `:13` reads `venv/` — the pre-repair pin `:13` alone names the *other* of the two
patterns, P-6, and a `.venv` created at the root is matched by `:12`) and steps 2–6
inside it, so that no package installed by another project (the `pip freeze` on this box
lists 300+ distributions from other projects, `READ requirements-kaggle.txt:52-56 @ 99777ab`)
can pull a build the certificate never named. Hypotheses: `py -3.11` resolves to `3.11.9`.
Evidence: RUN `python --version` reads `3.11.9`. Kill: the venv's step-2 line differs from
the frozen string — then the wheel, not the interpreter, is the break. *Replacement 2, the
alternative CUDA build*: `torch==2.5.1+cu118` from `whl/cu118` with the same torchvision and
torchaudio versions at `+cu118`. Hypotheses: the `cu118` index serves `2.5.1`. Evidence: **RUN**
`pip index versions torch --index-url https://download.pytorch.org/whl/cu118`, 2026-09-06, prints
`torch (2.7.1+cu118)` with available versions `2.7.1+cu118, 2.7.0+cu118, 2.6.0+cu118,
2.5.1+cu118, 2.5.0+cu118, 2.4.1+cu118, 2.4.0+cu118, 2.3.1+cu118, 2.3.0+cu118, 2.2.2+cu118,
2.2.1+cu118, 2.2.0+cu118, 2.1.2+cu118, 2.1.1+cu118, 2.1.0+cu118, 2.0.1+cu118, 2.0.0+cu118` —
`2.5.1+cu118` is served. The `[ASSUMED]` tag this replacement carried before repair, with
"none beyond the assumption" in its Evidence field, is **struck**: the probe costs $0$ GPU-s and
one command, the same command the verse had already run for `cu121` in the same session, so a
named-not-derived replacement (P-4) stood where a RUN was one line away. Kill: step 2's
verification **under the `cu118` index prints anything other than** `2.5.1+cu118 11.8 True NVIDIA
GeForce RTX 4060 Laptop GPU` — exact string equality on all four fields, the same frozen form as
the verse's own kill with `12.1` replaced by `11.8`. The kill this replacement carried before
repair fired on that success string, so the second link had no refutation at all (V-3, V-10) and
a `cu118` install that worked perfectly was scored as a kill. Planted negative: the present box,
on which the same line prints `2.14.0+cpu None False None` (RUN) — the kill fires on it today,
under either index. The **consequence** of the replacement succeeding is not a kill and is moved
where it belongs, into this verse's Terminal: the box then runs a **different certified stack**,
every number in `COSTS.md` §1 is stale for it by V-22, 05.5's full certificate becomes mandatory
before any price is quoted, and the canon records `cu118` in every journal stack string from that
moment. This link is more decisive than replacement 1 because it changes the object the
certificate names rather than where it is installed, and its kill is one printed line.

**Terminal.** Two outcomes, and the canon licenses a different sentence on each. *If the `cu118`
route installs and its frozen line prints*: the certified device exists again but is **not the
same object** — the canon licenses every future number on `2.5.1+cu118` and withdraws every
`COSTS.md` §1 constant as a comparand for it (V-22, no price crosses the build boundary), makes
05.5's full $523.9$ s certificate mandatory before any price is quoted, and requires `cu118` in
the flat `torch` string of every journal row 05.4's predicate reads. *If no `torch 2.5.1` wheel
installs on this box at either index*, the certified device no longer
exists as a reproducible object: the canon licenses every number in `COSTS.md` §1 and
`results/k_cert_local.json` as **history at `ab5b485`**, withdraws every `[FITTED + RUN]`
price in `docs/PLAN.md` §5.7 as a price for future work, and makes 05.5's full certificate
on whatever stack installs the new certificate — with its own stack string in every journal
(the flat `torch` key of 05.4's frozen predicate, and `box.torch` in the certificate itself,
which is the one producer that writes the record) and no number pooled with the old one.

---

### 05.4 — B1: the drift detector — the stack asserted in every journal row, and the census that catches a silent reinstall

**Statement — one predicate, frozen on the key producers actually emit.** The rule this verse
shipped before repair decided two different predicates: its Kill named `box.torch` while its
instrument grepped the flat string `"torch": "2.5.1+cu121"`, and RUN over
`results/v20_r15_it8_armpl_b.jsonl` shows the header carries a flat top-level `"torch":
"2.5.1+cu121"` and **no `box` object at all**, so the grep found the string and the stated
kill fired on the missing field — every journal row in `results/` passed the instrument and
failed the kill. RUN over the union of all row keys of `results/v15_r1.jsonl` (26 lines):
`torch` and `torch_version` present, `box` absent. RUN over all of `results/`:
$46$ occurrences of `"torch": "2.5.1+cu121"` and $130$ of `"torch_version":
"2.5.1+cu121"` in `*.jsonl`, $2$ of `"torch": "2.5.1+cu121"` in `*.json`, and **$38$
`*.jsonl` files with no `torch` key of any kind**. The predicate is therefore frozen on the
**flat `torch` key**, which is what `scripts/v15_r1.py` and its siblings emit:

> A `results/**/*.jsonl` file written after evening 1 is RUN evidence iff its header line
> carries a top-level `"torch"` key whose value is exactly `2.5.1+cu121`. Absence of the key
> and any other value are both failures, and they are separate clauses with separate plants.

The `box` record remains the **richer** form and is required of any producer that already
writes it: after 05.3, every journal row written by a producer that imports
`scripts/k_cert.py::box_record` carries the keys that function emits, and the count is
device-dependent, so it is printed as a RUN listing rather than asserted as a fixed set. RUN
`json.load(open('results/k_cert_local.json'))['box'].keys()` returns **13** keys, in this
order: `device`, `torch`, `python`, `platform`, `threads`, `cuda_available`,
`allow_tf32_matmul`, `allow_tf32_cudnn`, `cublas_workspace_config`, `device_name`,
`capability`, `total_vram_bytes`, `n_gpus`, with `torch: 2.5.1+cu121`. `READ
scripts/k_cert.py:669-680 @ 99777ab` shows why $13$ is not the record's length in general:
`box_record` builds **ten** keys unconditionally (`device` … `device_name="cpu"`) and adds
`capability`, `total_vram_bytes` and `n_gpus` only inside `if torch.cuda.is_available():`, so a
CPU box writes a **10**-key record and the certified card a 13-key one. The sentence this verse
carried before repair — that the record "holds exactly these" seven, plus six named after —
was wrong in three ways at once: it opened on a seven-key set that is not the record, it wrote
the ninth key as `CUBLAS_WORKSPACE_CONFIG` where the emitted key is lower-case
`cublas_workspace_config` (RUN `:674`, `cublas_workspace_config=os.environ.get(...)`), and it
asserted a fixed length for a record whose length is a branch on `cuda_available`. The seven
are re-stated as what they always were, a **required subset**: `torch`, `python`, `platform`,
`cuda_available`, `allow_tf32_matmul`, `allow_tf32_cudnn`, `cublas_workspace_config` are
present in both branches and are what a `box` record must carry to be read as one. But `box` is
**not** the kill's predicate, because no `.jsonl`
in `results/` has ever carried it and a kill no row can satisfy is not a kill (V-10, V-16).
The canon's drift census is the one-line comparison
`python -c "import torch,numpy;print(torch.__version__,numpy.__version__)"` against the
frozen string `2.5.1+cu121 1.26.4`, run at the start of every evening and printed at the top
of that evening's report. A row whose flat `torch` key is absent or differs from the frozen
string is not RUN evidence for any book.

**Hypotheses.** Every producer of new rows emits a header line with a flat `torch` key (RUN:
`scripts/v15_r1.py:607`'s `emit(dict(t="header", …))` does). One exception is RUN and named
rather than smoothed over: of the 38 files with no `"torch"` key,
`results/v20_r15_it11_jupiter_census.jsonl` carries `torch_version` instead, so the rule's
selector is the key **`"torch"`** with its closing quote — `"torch_version"` does not match it,
and that one file is caught by clause (a) as a producer that emitted the wrong key name, which
is the correct verdict and not a false positive. The comparison is string equality, so
`2.5.1+cu118` (05.3 replacement 2) is a **new** frozen string, declared once in
`CORRECTIONS.md`, never a tolerated variant. Rows written **before** evening 1 are out of
scope: they are judged by 05.2's Terminal, not by this census.

**Evidence.** RUN key censuses as printed above (`grep -rho '"torch[_a-z]*": "[^"]*"'
results --include=*.jsonl | sort | uniq -c` reads exactly two lines, $46$ and $130$, both
`2.5.1+cu121`; the no-`torch`-key file list is RUN by iterating `find results -name '*.jsonl'`
and testing for the key). `READ scripts/k_cert.py:668-681 @ 99777ab` for the richer `box`
form; RUN `results/k_cert_local.json["box"]["torch"]` reads `2.5.1+cu121`; the drift this
rule would have caught: site-packages mtimes 2026-09-03 23:30 (RUN, 05.2) against a
certificate of 2026-08-31 22:26 (`READ COSTS.md:62 @ 99777ab`).

**Mechanism.** P-1 (a number with no live producer — a row without its flat `torch` key has
no producer on a named stack); V-16 and M-2 (the defect this repair removes: a kill and its
instrument deciding two different predicates, so the instrument reported a pass on every row
the kill condemned); V-15 (clause (b), which had no planted negative and is recorded OPEN);
V-22; M-16 (the thread lane broken by the person who filed the rule — the lane
`torch.set_num_threads(8)` is in the same record).

**Kill.** Two clauses, one predicate, both in scope `results/**/*.jsonl`, both decidable on
the tree today:

- **(a) key absent.** A `results/**/*.jsonl` file written after evening 1 whose header line
  carries no top-level `"torch"` key. Instrument: `grep -rL '"torch"' results --include=*.jsonl`.
  Frozen: presence of the key.
  **Planted negative, in scope and firing today (RUN):** the **38** `results/*.jsonl` files
  that carry no `torch` key of any kind — `aggregator_mech`, `agg_matched`, `arm_a`,
  `arm_a_k1`, `arm_s`, `b1_collapse`, `b1_decomposition`, `cameron_aggregators`,
  `chase_m3_synth`, `equilibrium`, `foreman_consequence`, `foreman_looped`,
  `foreman_signfloor`, `foreman_theta_tv`, `gram`, `hilbert`, `journal_census`,
  `k3_concavity`, `m2`, `m3_quintuple`, `m3_quintuple_v2`, `m3_quintuple_v2_cuda`, `max_row`,
  `r10_it10_frontier`, `r10_it12_e4prime_spread`, `r10_it13_saturn_binding`,
  `r10_it14_corpus_spec`, `r10_it15_dual_oracle`, `r10_it16_admissibility`,
  `r10_it17_battery`, `r10_it18_scope_census`, `r10_it19_priced_1c`, `r10_it20_coherence`,
  `r2`, `s2`, `tau_trajectory`, `v20_r15_it11_jupiter_census`, `wilson_arms` (all
  `.jsonl` under `results/`). The clause fires on every one of them, which is the whole point:
  those files are pre-evening-1 history and are precisely the rows the rule must refuse the
  moment any of them is quoted as post-pin evidence.
- **(b) value wrong.** A `results/**/*.jsonl` file written after evening 1 whose `torch`
  value is not exactly `2.5.1+cu121`. Instrument:
  `grep -rho '"torch[_a-z]*": "[^"]*"' results --include=*.jsonl | sort | uniq -c` — one line
  per distinct stack string, and the census reads the count of distinct values, frozen at
  **1**. Frozen: string equality against `2.5.1+cu121`.
  **Planted negative — NOT MEASURED in scope, and stated as such rather than borrowed.** RUN
  today, `results/**/*.jsonl` holds exactly one distinct value ($46$ occurrences of
  `2.5.1+cu121`), so **no in-scope row makes clause (b) fire**, and the verse does not pretend
  otherwise. `results/kaggle_v17k_output/ceq-v17-k.log:11` reads `torch   2.10.0+cu128` (RUN)
  and is the only non-frozen stack string anywhere under `results/` — but it is a `.log`, not
  a `.jsonl` row, so it lies **outside clause (b)'s scope and cannot make it fire**; the
  refusal of that file by scope is not detection by the rule, and calling it a plant was the
  V-15 defect this repair removes. The owed in-scope plant is
  `NOT MEASURED — needs one post-evening-1 `.jsonl` written on a deliberately non-frozen
  stack`; it arrives for free if 05.3's replacement 2 (`cu118`) is taken, since the first
  post-pin journal then reads `2.5.1+cu118` and clause (b) must fire on it before
  `CORRECTIONS.md` re-freezes the string. **Until that plant exists, clause (b) is a
  condemning rule with no planted negative and is recorded OPEN**; clause (a) carries the
  verse.

Price: $0$ GPU-s for both clauses.

**If killed** (a file lands without the key, clause (a)). Its rows are quoted as
`NOT MEASURED — stack unrecorded` and its producer is re-run once with the key. Hypotheses:
the producer is deterministic on its seed (Ruling 1's forward-only bitwise clause). Evidence:
`READ COSTS.md:141-151 @ 99777ab`. Kill of the replacement: the re-run's value differs from
the row's by more than $0$ bitwise on a forward-only cell — then the original row was not a
measurement of the certified stack and is superseded with an L-G2 marker, never deleted.

**If killed** (clause (b) is unreachable, which it is today). The replacement is a
**different object measured by a different instrument**: instead of asking a journal what
stack wrote it, the census asks the **interpreter** what stack is loaded, at the moment the
journal is opened. The one-line comparison
`python -c "import torch,numpy;print(torch.__version__,numpy.__version__)"` against
`2.5.1+cu121 1.26.4` is RUN at the head of every evening and its output is pasted into the
evening report beside the journal filenames written that evening; a journal filename that
appears in an evening report whose head line is not the frozen string is withdrawn.
Hypotheses: one evening is one process tree on one stack. Evidence: RUN today the command
prints `2.14.0+cpu 2.4.6`, which is **not** the frozen string — so this replacement's kill
fires on the present box, which clause (b) could not do. Kill of the replacement: an evening
report with no head line — then the evening produced no attributable evidence and every
journal it wrote is `NOT MEASURED — stack unrecorded`. That is the last link.

**Terminal.** The canon licenses only rows that print their stack; every other row is a
number without provenance and is withdrawn from every book's Evidence field. Clause (b)
stands **OPEN** until an in-scope plant exists: while it is open, the canon licenses the
absence census (clause (a)) and the interpreter census (the replacement) as the drift
detector, and withdraws the sentence "a wrong stack string in a journal is caught by this
rule" — nothing in `results/` has ever carried one, so the rule has never been shown to
catch it.

---

### 05.5 — B2: device re-certification — the ≤20 GPU-s half on evening 2, the 523.9 s half on evening 6, the acceptance numbers

**Statement.** `results/k_cert_local.json` is stale by construction: it records `torch
2.5.1+cu121` at `git_head ab5b485` (`READ COSTS.md:58,63 @ 99777ab`; RUN
`["box"]["torch"]`), and no certificate exists for any stack installed since. Re-certification
is `scripts/k_cert.py` unchanged, in two runs:

- **Evening 2 (≤20 GPU-s):** `python scripts/k_cert.py --only determinism,zerostep --out
  results/k_cert_local_e2.json`. Determinism at reduction length 64: 8 repeats, $n=512$,
  both flag regimes (`READ scripts/k_cert.py:579-640 @ 99777ab`), priced $\approx1$ GPU-s
  (`READ docs/PLAN.md:534 @ 99777ab`, N-01's class). Zero-step gate: 2 arms × 5 $n$ × 3 seeds
  $=30$ forward-only cells over `cell_ns = [2048, 4096, 8192, 16384, 32768]` (RUN, the
  certificate's `cell_ns` key); bounded above by the full-step laws summed over the ladder,
  $3\times\big[(0.093066+0.186465+0.373597+0.748531+1.499743)+(0.010162+0.020271+0.040436+0.080663+0.160907)\big]=9.64$
  s **`[FITTED, extrapolated at $n=16384$ and $n=32768$]`** from
  `READ COSTS.md:229-238 @ 99777ab` (a forward is less than a
  forward-plus-backward-plus-Adam step). Total $\le 10.64$ GPU-s, same class, under the cap.

  **The pin is the ten rows, and the range this verse carried before repair held eight of them
  and two paragraphs of prose.** `COSTS.md:222-235 @ 99777ab` opens at `:222`, which reads
  "`alloc` is the measured allocated peak; `proj resv` applies the worst" — the chunk table's
  prose preamble, running to the budget line at `:225` — and closes at `:235`, which is the
  `softmax` $n=4096$ row, so the cited range **omitted** the two rows the sum's last two terms
  come from: `softmax` $n=16384$ ($0.080663$ s) at `:237` and $n=32768$ ($0.160907$ s) at
  `:238`. A citation that excludes two of the ten numbers it sums is P-6, line-reference drift
  carried into a load-bearing arithmetic. The ten `s/step` values live at
  **`COSTS.md:229-238`** exactly — five `arm_smprime` rows at `:229-233`, five `softmax` rows
  at `:234-238` (RUN `git show 99777ab:COSTS.md | sed -n '229,238p'`, and RUN
  `git show 99777ab:COSTS.md | grep -n "0.093066\|0.010162\|1.499743\|0.160907"` reads
  `229`, `234`, `233`, `238`).

  **Why the class is not DERIVED, stated because the arithmetic alone would hide it.** Two of the
  ten inputs are not measurements. `READ COSTS.md:232-233 @ 99777ab` tags the arm's $n=16384$
  ($0.748531$ s) and $n=32768$ ($1.499743$ s) rows `[FITTED]`, extrapolated, and
  `READ COSTS.md:76-79 @ 99777ab` records why the record refused them as arm measurements: on the
  $7.996$ GiB card the arm's caching allocator reserves $10.578$ GiB at $n=16384$ and $13.969$ GiB
  at $n=32768$, so the driver pages over PCIe and the loop times the bus, which is why the arm's
  throughput law is fitted over three points and not five. The sum $3\times(2.901402+0.312439)
  =9.641523$ is arithmetically right and its class is the weakest of its inputs: a DERIVED tag over
  an extrapolated input is P-8 dressed as a derivation. **What makes the ladder runnable anyway,
  despite the training-cell ceiling:** the zero-step gate's cells are **forward-only**, and at
  `ab5b485` all thirty were reachable — `READ COSTS.md:137-139 @ 99777ab`, "2 arms × 5 values of
  `n_train` × 3 seeds = **30 cell shapes, 30 reachable, 30 pass**", worst margin
  $1.957\times10^{-3}$ — so the two shapes the *training* law could not measure are shapes the
  *zero-step* stage has already run. The bound is therefore an over-estimate of a stage that is
  known reachable, not an estimate of a stage that is known unreachable, and evening 2's price is
  re-measured against the certificate it writes rather than quoted forward.
- **Evening 6 (523.9 s `[MEASURED]`):** `python scripts/k_cert.py --out
  results/k_cert_local_e6.json`, all five stages, the record's own run-of-record price
  (`READ COSTS.md:62 @ 99777ab`).

**The split filenames are mandatory, not stylistic, because the script does not append — it
overwrites, and the docstring that says otherwise is a stale claim about code.** Both runs
above wrote to `results/k_cert_local.json` before repair on the strength of
`READ scripts/k_cert.py:3-5 @ 99777ab`, "writes a record keyed by `box` so a later Kaggle run
APPENDS a second certificate beside this one rather than overwriting it". RUN
`git show 99777ab:scripts/k_cert.py | sed -n '925,928p'` reads
`out = pathlib.Path(a.out)` / `out.parent.mkdir(parents=True, exist_ok=True)` /
`out.write_text(json.dumps(cert, indent=2), encoding="utf-8")` — an unconditional
`write_text` on whatever path `--out` names. RUN `git show 99777ab:scripts/k_cert.py | grep -c
"json.load("` reads $\mathbf{1}$, and that one call is `json.loads(a.probe)` at `:700`, a
command-line string: **the script never reads an existing certificate**, holds no
`boxes` collection, and has no merge path. "APPENDS" describes the *convention* of one box
writing to one filename, not code, and is P-3, a stale claim never retracted. The consequence
runs against this book and is recorded at its worse reading: writing evening 2 or evening 6 to
`results/k_cert_local.json` would **destroy** the `ab5b485` certificate, which L-G2 forbids
outright, so the split filenames `_e2` and `_e6` are the only L-G2-safe route and the file
`results/k_cert_local.json` is never written again by any evening of this book. The three
certificates are read **side by side by path** — the `ab5b485` history, the evening-2 half and
the evening-6 whole — and are never merged into one file, because merging them needs a reader
that does not exist and writing one is code the standing rule forbids
(`NOT MEASURED — needs a certificate merger, which this canon will not build`). In the
acceptance table below the **measured** side of every row is the evening's own file —
`results/k_cert_local_e2.json` for the two evening-2 rows, `results/k_cert_local_e6.json` for
the five rows marked "(evening 6)" — and the **comparand** column is the `ab5b485` reading as
`COSTS.md` prints it.

Acceptance numbers, frozen from the record's own thresholds, each read against the same
stage of the `ab5b485` certificate:

| stage | acceptance | the `ab5b485` reading it is compared with |
|---|---|---|
| determinism, hop and full forward | bitwise, $\max|\Delta|=0.0$, flag OFF **and** ON | `0.0 / 0.0` both regimes (`READ COSTS.md:141-149`) |
| determinism, gradient | bitwise flag OFF; **raises** `cumsum_cuda_kernel` flag ON | `0.0` / `NOT EXECUTABLE` (`READ COSTS.md:149-151`) |
| zero-step gate | `30/30` reachable and passing at `GATE_TOL = 1e-3` | `30/30`, worst margin $1.957\times10^{-3}$ (`READ COSTS.md:135-139`) |
| bar (evening 6) | worst $\delta/\mathrm{tol}<50\%$ | $8.58\%$, headroom $5.83\times$ (`READ COSTS.md:130-133`) |
| throughput (evening 6) | both laws refitted at $R^2\ge0.99$ (the script's `min_r2`, RUN) | softmax $0.999998$, arm $1.000000$ (`READ COSTS.md:73-74`) |
| throughput, within-stack band (evening 6) | one 150-step cell at $n=2048$ inside $2.2\times$ the **newly refitted** law of the *same* stack — the units the $2.2\times$ bar was calibrated in | S-20's kill line, `READ docs/PLAN.md:463 @ 99777ab`: "the cell above $2.2\times$ **the law**", one measured cell against its own fitted law on one stack, invocation fixed cost included |
| throughput, across-stack drift | each law's $s/\mathrm{step}$ at $n=2048$ within $\mathbf{2.0\times}$ of the `ab5b485` value of **the same arm** — `[ASSUMED]`, declared here, reason below | `softmax` $0.010162$ s (`READ COSTS.md:234 @ 99777ab`), `arm_smprime` $0.093066$ s (`READ COSTS.md:229 @ 99777ab`) |
| memory (evening 6) | `C_RESIDUAL`, `C_OPERATOR` re-solved within $\pm5\%$ of $17.874$ / $3.823$ | `READ COSTS.md:89-90 @ 99777ab` — the two **fp32** rows; the pre-repair pin `:91-94` named the two `bf16_autocast` rows ($3.341$ B/elem operator, $2.383$ B/elem residual, the second of which the record marks "**WRONG, +8.3 %, optimistic**") and a block quote, which are different constants under a different dtype |

**The across-stack drift bar, declared here because nothing else calibrates one.** The
$2.2\times$ figure this verse imported before repair is S-20's own kill bar and lives at
`READ docs/PLAN.md:463 @ 99777ab` (**not** `:464`, which is S-20's price line, "$\approx 17.4$
s for 8 seeds"). At `:463` it reads "the cell above $2.2\times$ **the law**" — **one measured
150-step cell against its own fitted law, on one stack, with the $4.0$ s invocation fixed cost
inside it**. That is a within-stack cell-versus-law band. Using it as a refitted-law-versus-
`ab5b485`-law bar across a stack boundary is V-17, a threshold read outside the units it was
calibrated in, and M-3, a scale carried where it was never measured. The two acceptances are
therefore split into the two rows above, and the across-stack row gets its own number:
$\mathbf{2.0\times}$, `[ASSUMED]`, on $s/\mathrm{step}$ at $n=2048$ between the refitted law
and the `ab5b485` law **of the same arm**. Reason: it must clear the record's own measured
same-stack instability — one cell re-run on one box read $14.390$ and $17.839$ s, a
$1.265\times$ spread over three observations spanning $14.101\ldots17.839$ s (`READ
V17_R4_RETAKE_PRICE.md:143-150 @ 99777ab`) — and the $\pm12$ per cent clock non-stationarity
the plan quotes prices under (`READ docs/PLAN.md:967 @ 99777ab`); $2.0$ is the smallest round
multiple above both with headroom, and it is declared, not inherited. Its **planted negative**
is in scope and decidable at $0$ GPU-s from `COSTS.md` alone: feed the comparison the *wrong
arm's* law — `arm_smprime` at $n=2048$ ($0.093066$ s) against `softmax` at the same $n$
($0.010162$ s) is a ratio of $9.16$, and the bar must fire, because M-8 (pricing every arm at
one arm's rate) is exactly the mis-keying an across-arm drift census would commit silently.
Its **true negative**, equally required so the bar is not vacuous: the record's own
$14.390$-versus-$17.839$ s pair, ratio $1.24$, on which the bar must **not** fire.

**Hypotheses.** 05.3 landed and 05.4's census printed `2.5.1+cu121 1.26.4` that evening;
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported **before** process start — set from inside Python
after CUDA is initialised it does not take, and strict mode then fails the forward as well,
`READ V17_R4_RETAKE_PRICE.md:178-181 @ 99777ab` verbatim, quoted live at `READ
docs/sources/sections/sec_cost.md:122 @ c71527a` (the pin `@ 99777ab` is unavailable for
`sec_cost.md`: RUN `git show 99777ab:docs/sources/sections/sec_cost.md` reads `fatal: path …
exists on disk, but not in '99777ab'`; RUN `git log --oneline -- docs/sources/sections/sec_cost.md`
reads one commit, `c71527a`). The pin this verse carried before repair, `docs/PLAN.md:460`,
does **not** carry the claim as its own: RUN `grep -c CUBLAS docs/PLAN.md` reads $0$, and
`:460` is S-20's build line (`torch.equal(O(0), PV)` against `softmaxAttn`, ChaCAL-diag at
$\gamma=0.9$) which cites `V17_R4_RETAKE_PRICE.md:178-181` rather than stating it. Threads
pinned at 8 after imports (`READ scripts/k_cert.py:686-687 @ 99777ab`).

**Evidence.** As pinned in the table; the two-run split is DERIVED from the stage prices;
the `--only` interface RUN (`scripts/k_cert.py --help`, choices `throughput,memory,bar,
zerostep,determinism`).

**Mechanism.** V-22 (no threshold carried across the stack boundary — every acceptance is
compared, never pooled); M-2 (thresholds frozen here, before the run); V-16 (a stage that
raises is "not executable", never "pass"); P-1 (the certificate has a live producer).

**Kill.** Any table row failing on evening 2 or 6. Frozen as printed. Instrument:
`scripts/k_cert.py` and its `tests/gate0/test_g06_kcert.py` must-fire negatives
(`READ scripts/k_cert.py:9 @ 99777ab` — "each half carries its must-fire negative in
`tests/gate0/test_g06_kcert.py`" — and `READ scripts/k_cert.py:154 @ 99777ab` — "can plant a
fabricated reading in each and watch the corresponding half of the certificate fire"; the
pre-repair pin `:6-7` reads "this one rather than overwriting it. The local box DECIDES;
Kaggle REPRODUCES." and names no test at all, P-6). Price: $\le10.64$ GPU-s (evening 2), $523.9$ s
(evening 6). Planted negatives: the gradient row's `cumsum` raise is itself the plant the
determinism stage must reproduce (`READ scripts/k_cert.py:644 @ 99777ab`, `cumsum_under_flag`'s
`def` line; the pre-repair `:642` is two lines above it, inside the preceding comment block);
for the throughput drift bar, the `ab5b485` law evaluated at $n=16384$ for the arm, which
the record itself refused as a bus measurement ($10.578$ GiB reserved, `READ COSTS.md:78`)
and which the refit must again exclude — a shape entering the fit with reserved bytes above
the card is the kill firing on a known-bad plant.

**If killed** — *hop or forward not bitwise under the flag*: Ruling 1's regime narrows to
`warn_only=True` for every cell, the deciding-forward clause (`READ V17K_RULINGS.md:39-45 @
99777ab`) is withdrawn for this stack, and every deciding cell is journalled with
`deterministic_regime = warn_only` and its 8-repeat $\max|\Delta|$ printed (N-01's fallback,
`READ docs/PLAN.md:533 @ 99777ab`). Hypotheses: the drift is below the label sd, printed.
Evidence: `NOT MEASURED — needs the evening-2 run`. Kill: $\max|\Delta|$ above the realised
paired sd of any bed (book 04's S-62) — then no equality claim on that bed is licensed.
*The within-stack cell outside $2.2\times$ its own refitted law*: every `[FITTED + RUN]` price
in §5.7 is re-tagged the P-8 floor and re-measured on evening 6 before any GPU-minute figure
is quoted; hypotheses: the refit reaches $R^2\ge0.99$; kill: it does not — the throughput law
is withdrawn and prices are quoted only as measured points at the shapes measured.
*The across-stack ratio outside the `[ASSUMED]` $2.0\times$*: the installed stack is a
different cost object from `ab5b485` and **no** price crosses the boundary (V-22); every
`COSTS.md` §1 price is re-tagged history and the first five evenings' GPU-s figures in 05.14
are re-derived from the refitted law before evening 7; hypotheses: the refitted law exists
(the row above passed). Kill of that replacement: the refitted law does not exist either —
then no law is quoted and every price in this book falls back to measured points at the
shapes measured, which is the same last link as the row above. *Zero-step gate below
30/30*: the failing `(arm, n)` shapes are struck from the ladder (`cell_ns` shrinks, never
widens — Ruling 6f's direction), and 05.15's rows that need them are repriced.

**Terminal.** With no certificate on the installed stack the canon licenses no price and no
determinism sentence for future work; it licenses the `ab5b485` certificate as history and
the protocol above as the instrument, and withdraws every "faster to train" T1 number (book
03) until the certificate exists.

---

### 05.6 — B16(a): `S, D = 64, 24` as module constants, and the absent `seq_len` flag

**Statement.** `scripts/v15_r1.py:137` reads `S, D = 64, 24` and `:138` `T_STAR = 2`; the
parser at `:547-558` has nine `add_argument` calls and none for `s`, `d` or `t_star` (RUN
`grep -n add_argument`). The repair specification: three flags `--s` (default 64), `--d`
(default 24), `--t-star` (default 2), each written into the journal header
(`emit(dict(t="header", … s=S, d=D, …))` at `:607` already prints them) and into the cell id
at `:829`; the module constants are deleted and **every** site that reads `S`, `D`, `T_STAR`
reads the parsed values.

**The complete site list, re-run and printed, because the list this verse shipped before
repair was short by ten lines and the kill could not detect the omission.** RUN
`grep -nE '\bS\b' scripts/v15_r1.py`, `grep -nE '(^|[^_A-Za-z])D([^_A-Za-z]|$)'`,
`grep -n 'T_STAR'`, then discarding the docstring and comment lines that read "ARM S-M'"
rather than the constant (`:8, :10, :62, :70, :192, :299, :319, :336, :400, :436, :446,
:469, :487, :491, :499, :509, :564, :780, :800`):

| constant | definition | code sites that read it |
|---|---|---|
| `S` | `:137` | `:506, :508, :588, :607, :656, :659, :673, :684, :686, :697, :698, :699, :708, :745, :802, :827, :837` |
| `D` | `:137` | `:588, :607, :684, :699, :708, :827, :837` |
| `T_STAR` | `:138` | `:165, :586, :587, :607, :686, :697, :810, :826, :829, :907` |

The list this verse carried before repair (`:165, :586-587, :686, :697, :699, :708, :810,
:826-829, :907`) omitted `:506, :508, :588, :656, :659, :673, :684, :745, :802, :837` — ten
sites, of which two are load-bearing for the kill itself: `:506`
`rows.append(identity_bind(kind, S, device=device))` and `:508` `for s in (8, S)` build the
**label bind**, and `:656` selects the bind row by `r["s"] == S`. A repair executed to the
short list, run at `--s 32`, would bind the identity at $s=64$ while training at $s=32$,
write `s=32` in the header at `:607`, and **pass the kill as it was frozen** — the book
reproducing, on itself, the exact P-6 line-reference drift and M-1 train/eval mismatch it
files against the record. That is why the kill below gains a clause the header cannot fake.

The verification is a **same-stack identical-seed pair**: two runs at the defaults,
one seed, on the re-certified device, must read the **same `eval_nrmse` float**, and the header
must read `s=64, d=24, t_star=2`; a third run at `--s 32 --arms arm_pl` must write `s=32` in
the header, a different cell id, **and two `t="bind"` rows reading `s=8` and `s=32`**.

**The comparison is float `==` on two JSON scalars, not `torch.equal`, because the quantity
the clause decides is not a tensor.** RUN over `results/v15_r1.jsonl`: `eval_nrmse` is a JSON
number in every cell row (`grep -o '"eval_nrmse": [^,]*'` reads `0.6456144346724637`,
`0.6444541428485409`, `1.1524300284608222`, …), sitting in the same key set as `train_nrmse`
and `dist_to_floor`; `torch.equal` takes two tensors and cannot be applied to it at all, so
the clause as this verse froze it named an instrument that could not be run on its own object
(V-16). The comparison in force: **the two rows' `eval_nrmse` floats are not `==`**, with
**both values printed at full `repr` precision on the evening report's line**, so a reader can
see the two numbers rather than a boolean. Full-precision float equality is the same bar the
struck instrument intended — the journal already writes `eval_nrmse` at 16 significant digits,
RUN above — and it is decidable with no library call at all.

**Hypotheses.** Ruling 5: `scripts/v15_r1.py` is the named Q1/Q2 instrument and is hashed
into the identity manifest (`READ V17K_RULINGS.md:66-70 @ 99777ab`), so the repair moves the
hash — the manifest is re-issued with the flags, never patched around them. `BED-M` at
`e3_t2` is the task (`:165`). The identical-seed pair is compared on the **same** stack; the
retake journal's rows (`results/v15_r1.jsonl`, 26 lines, RUN) were written on `2.5.1+cu121`
at `ab5b485` and are a comparison, not a bar (V-22).

**Evidence.** RUN `sed -n 137,138p scripts/v15_r1.py`; RUN grep of the flag list; RUN the
three constant greps whose output is the site table above; RUN `grep -n "GATED_ARMS *="
scripts/v15_r1.py` reads `:146`; RUN over `results/v15_r1.jsonl` for the bind rows' `s`
field and for the header's `arms: ['arm_pl','softmax']`;
`READ docs/CEQ_SHAPE.md:1277-1280 @ 99777ab` (the debt as filed); `READ V17K_RULINGS.md:338-339
@ 99777ab` (Ruling 8's note that `T_STAR` is a module constant with no flag).

**The `T_STAR` pin, re-read line by line, with the two halves of the finding against it
separated.** The pin this verse carried before repair, `:336-337`, is the wrong pair of lines:
RUN `git show 99777ab:V17K_RULINGS.md | sed -n '336,337p'` reads "it: *local decides, and
local cannot run it.* There was no re-take to price / because there was nothing to re-take."
The note is at **`:338-339`** — "(Secondary, and not the reason: `T_STAR = 2` is a module
constant at / `scripts/v15_r1.py:124` with no `--t-star` flag.)" — and it is re-pinned there.
Two further facts are recorded rather than smoothed. First, **the pin's own line number for
the constant has drifted**: `V17K_RULINGS.md:339 @ 99777ab` names `scripts/v15_r1.py:124`,
and RUN `sed -n '137,138p' scripts/v15_r1.py` on the tree today reads `S, D = 64, 24` at
`:137` and `T_STAR = 2` at `:138` — the constant moved thirteen and fourteen lines, which is
P-6 inside the source this verse cites, and the verse's own `:137`/`:138` are the live
readings, not the pin's. Second, **the "Ruling 8" attribution stands and the finding against
it is refuted by RUN**: the attack read "`grep -n '^\*\*RULING' V17K_RULINGS.md` lists
Rulings 1-7 at :39,:47,:56,:61,:66,:71,:83 and Ruling 10' at :389 — there is no Ruling 8",
which is true of that one regex and false of the file. Rulings 8, 9 and 10′ are written with a
markdown heading, not a bold run: RUN
`git show 99777ab:V17K_RULINGS.md | grep -n '^# RULING'` reads
`275:# RULING 2a`, **`328:# RULING 8 — Q2 DROPPED`**, `354:# RULING 9`, `389:# RULING 10′`.
`:338-339` sits at $328<338<354$, inside Ruling 8's block, so the attribution is correct as
written and only the line pair was wrong. The canon records the sharper diagnosis: a
`^\*\*RULING` grep is not a census of the rulings, and a heading-blind census is P-6 in the
instrument rather than P-7 in the verse.

**Mechanism.** D-3 (a dial that does not vary describes nothing — $s$ and $d$ could not vary
at all); P-6 (line-reference drift — every site is listed by line at `99777ab` so the repair
can be checked against the pin); M-16.

**Kill.** Five clauses, the fifth added with the price it audits. (i) The identical-seed pair at the defaults has
**`eval_nrmse` floats that are not `==`**, both values printed at full `repr` precision beside the
comparison. (ii) The header omits any of `s`, `d`, `t_star`. (iii) The `--s 32` run writes
`s=64` in the header. **(iv) The `--s 32` run's `t="bind"` rows do not read `s=8` and `s=32`**
— the clause that catches the ten omitted sites, decidable because the journal already
carries `s` on every bind row (RUN over `results/v15_r1.jsonl`: two `t="bind"` rows, `s` $=8$
and $64$, residuals $6.661338147750939\times10^{-16}$ and $7.549516567451064\times10^{-15}$;
the `for s in (8, S)` loop at `:508` is what emitted them). **(v) The `--s 32` run's journalled
`secs` exceeds $1.884$ s**, the ceiling the price below assumes. Frozen: float `==` on the two
`eval_nrmse` scalars, both printed; exact
header strings; `{8, 32}` as the bind-row `s` set; $1.884$ s as the $s=32$ ceiling. Instrument: two invocations of the repaired
script at `--device cuda --seeds 0 --arms softmax`, plus one at `--device cuda --seeds 0
--arms arm_pl --s 32` — a **gated** arm, because `GATED_ARMS = ("arm_pl", "arm_smprime")` at
`:146` (RUN) and `--arms softmax` emits no bind row at all, so clause (iv) run on `softmax`
would be vacuous (V-11). Price: $2\times(1.524+4.0)+(1.884+4.0)=16.93$ s, evening 3b — **and
the third term is a ceiling in another shape's units, tagged as one rather than quoted as a
price.** `READ docs/PLAN.md:971 @ 99777ab` defines $1.524$ s as "one 150-step softmax control
cell, $n=2048$, **$s=64$**" and `:978` defines $1.884$ s as "the capped cell", READ, at the
same shipped $s=64$; the third run is at **$s=32$**, a shape neither unit was measured at.
`READ docs/PLAN.md:967 @ 99777ab` states the rule the record itself obeys: "the increment was
measured at the shipped $s=64$ and **never scaled from a smaller $s$** (M-3)". So the third
run's own price is `NOT MEASURED — needs the $s=32$ cell's own timing`, and $16.93$ s is
carried as $2\times(1.524+4.0)=11.05$ s `[FITTED]` **plus** $1.884+4.0=5.88$ s
`[ASSUMED — the $s=64$ capped cell used as a ceiling on the $s=32$ cell]`, the direction
printed: less solve depth per position at $s=32$ than at $s=64$ is asserted, not measured, and
the plan forbids deriving it. The assumption is falsifiable on the same run at $0$ extra cost
and is added to this verse's kill as clause (v): **the `--s 32` run's journalled `secs`
exceeds $1.884$ s** — then the ceiling is wrong, evening 3b is repriced from the measured cell,
and 05.1's cap arithmetic is re-run against it before evening 3c is scheduled. Planted
negatives, two: (1) the unrepaired script, which raises on
`--s 32` (`argparse` rejects the unknown flag — RUN by inspection of the parser at `:547-558`,
exit code 2) — clauses (ii)–(iv) fire on it today; (2) a repair executed to the **short**
list, on which the header reads `s=32` and clauses (i)–(iii) all pass while the bind rows
read `s=8` and `s=64` — clause (iv) fires and nothing else does, which is the negative that
makes clause (iv) load-bearing rather than decorative.

**If killed** — *clause (iv) fires alone (the bind rows read $s=64$ while the header reads
$s=32$)*: the repair is incomplete at the ten omitted sites, and the fix is mechanical and
priced at $0$ GPU-s — the three greps above are re-run after the edit and their output
compared line-for-line against the site table, before any cell is spent. Hypotheses: the
greps are exhaustive for the three identifiers (they are, up to the docstring lines listed).
Evidence: the site table, RUN. Kill of that replacement: a grep output line that the table
does not carry — then a fourth identifier reads the shape and the manifest hash is withheld
until it is named. *The two `eval_nrmse` floats are not `==` (clause (i))*. **The replacement this verse carried before repair was
the same test on another device, and it died to the same kill; it is struck and re-derived on a
different object.** It read: verify the flag repair on CPU at `--device cpu --threads 8`,
"where the record read flag-OFF identical-seed repeats bitwise on `eval_nrmse`
(`READ V17K_RULINGS.md:373-376 @ 99777ab`)". Two defects. First, the pin is wrong:
`V17K_RULINGS.md:373-376 @ 99777ab` is Ruling 2a's **floor rule** —
`δ_nrmse = |eval_nrmse⁽ᵃ⁾ − eval_nrmse⁽ᵇ⁾|` over an identical-seed repeat "on the certified
device under the ruled regime (`warn_only=True`)" — a **non-zero floor to be measured**, not a
statement of bitwise equality (RUN `git show 99777ab:V17K_RULINGS.md | sed -n '374,379p'`). The
bitwise sentence is at **`:381-382`** — "a flag-OFF identical-seed repeat during pricing came
back **bitwise identical** on `eval_nrmse`" — and it **names no device**; "during pricing" was
on the certified card, and no CPU reading of it exists anywhere. The premise "on CPU, where the
record read … bitwise" is therefore unsupported at any pin, and the CPU pair is
`NOT MEASURED — needs the evening-3 CPU run`. Second, and the reason it is struck rather than
re-pinned: the replacement is an identical-seed-pair equality test on `eval_nrmse`, which
is **clause (i)'s object and clause (i)'s number**, on the same instrument, moved only to
another device — and it was written with the same wrong operator, `torch.equal` on a JSON
scalar, so it named an instrument that cannot be applied to the quantity it decides on either
device (V-16, in the replacement as well as in the clause). If the repair broke the script's seeding, it breaks it on CPU too and the replacement
dies with the verse — V-9.

*The replacement, on a different object, by a different instrument, at a different number.*
The flag repair is verified on the **identity-bind residuals**, not on the eval cell. Every
invocation clauses (ii)–(iv) already pay for emits `t="bind"` rows through
`identity_bind(kind, s, …)` at `:506` under `for s in (8, S)` at `:508` — a code path the
training loop never enters — and each row carries `residual`, `row_sum_drift`,
`normalizer_drift` and a `mutations` dict of four deliberately broken binds. **Hypotheses.** The
bind rows are emitted by any gated arm (`GATED_ARMS` at `:146`); the comparison is an absolute
bar in the residual's own units and never a bitwise match against `ab5b485`, because that
journal was written on another stack (V-22). **Evidence.** RUN over `results/v15_r1.jsonl`, the
two `t="bind"` rows read `residual` $6.661338147750939\times10^{-16}$ at $s=8$ and
$7.549516567451064\times10^{-15}$ at $s=64$, with `normalizer_drift`
$4.440892098500626\times10^{-16}$ and $3.774758283725532\times10^{-15}$. **Kill of the
replacement.** Either bind row of the repaired script reads `residual` $\ge10^{-14}$ — one
order of magnitude above the larger of the two the un-repaired script produced, frozen here —
or its `mutations` dict is absent. **Planted negative, RUN-populated and firing today:** the
same two rows' eight mutation residuals, `drop_key_bias` $0.9749459015140511$,
`drop_value_rescale` $0.9165274652308163$, `drop_bos_sink` $1.0$, `half_key_bias`
$0.48449311856985267$ at $s=8$ and $1.5715211921402767$, $1.9148407189594014$, $1.0$,
$0.7215251981895894$ at $s=64$ — eight deliberately broken binds spanning
$0.4845\ldots1.9148$, every one of them fourteen orders of magnitude above the $10^{-14}$ bar,
so the bar fires on all eight. **True negative:** the two unmutated residuals, on which it must
not fire, and does not. Price: $0$ additional GPU-s — the rows arrive inside evening 3b's
invocations. This link is strictly cheaper than the one it replaces (it buys no run at all) and
strictly more decisive (it separates a broken bind from a broken clock, which the eval-cell pair
cannot). *Kill of that replacement*: the mutation residuals themselves fall below $10^{-14}$ —
then `identity_bind`'s four mutations no longer perturb the bind, the instrument is dead
(V-16), and the manifest hash is withheld; the last link is the un-repaired script kept as the
instrument with `s=64, d=24` stated as a fixed geometry in every verse that cites it.

**Terminal.** The canon licenses `scripts/v15_r1.py` only at $s=64$, $d=24$, $t^\star=2$,
and withdraws every sentence that scales its numbers in $s$ (book 03's cost law must use its
own producer, N-02).

---

### 05.7 — B16(b) and B14: synchronised timers, interleaved arms, run order journalled — with the planted negative

**Statement.** `scripts/v15_r1.py` times with `time.time()` at **four** sites and contains no
`torch.cuda.synchronize` (RUN `grep -n "time\.time()" scripts/v15_r1.py` reads `:249`, `:267`,
`:585`, `:992`; RUN `grep -c synchronize scripts/v15_r1.py` reads $\mathbf{0}$; `READ
V20_R15_JOURNAL.md:53,1690-1694 @ 99777ab`, C17), so every CUDA `secs` is un-synchronised
host wall-clock, and the strongest correlate of `secs` in the record is run order,
$\rho=+0.7029$, $p=0.0024$ (`READ V20_R15_JOURNAL.md:1692-1693 @ 99777ab`). The repair
specification, binding on this script and on every timer book 03 specifies:

1. `torch.cuda.synchronize()` immediately before `t0` (`:249`) and immediately before `secs`
   is read (`:267`), on every CUDA cell; the old column is **kept** as `secs_unsync` beside the
   new `secs_sync` so the plant travels with the instrument. **The two whole-run timers are
   named here and disposed of explicitly, because a repair specification that names two of
   four `time.time()` sites and is silent on the other two is P-6 turned on the repair
   itself.** RUN `sed -n '585p;992p' scripts/v15_r1.py`: `:585` is `t_run = time.time()` and
   `:992` is `wall = time.time() - t_run` — one bracket around the **whole invocation**,
   spanning argument parsing, data generation, every cell of every arm and the journal write.
   That quantity is **never quoted as a cost anywhere in this canon**: every price in this
   book and every ratio book 03 is licensed to quote is a per-cell `secs` or a sum of per-cell
   `secs` plus the $4.0$ s invocation fixed cost (`READ docs/PLAN.md:978 @ 99777ab`), never
   `wall`. `wall` is therefore **exempt from clause 1 and simultaneously banned as a cost**:
   it is journalled as an invocation-level bookkeeping field, it is not synchronised, and no
   verse in any book may cite it, derive from it, or compare it across arms — a `wall` reading
   is `NOT MEASURED — needs a synchronised per-cell timer` for every purpose the canon has.
   The exemption is enforced, not asserted: it is clause (iv) of the Kill.
2. Arms interleaved `A B A B …` within one invocation and one thread lane; the journal
   carries `run_order` (an integer) and `invocation_id`.
3. The report prints Spearman $\rho(\texttt{run\_order}, \texttt{secs\_sync})$ with an
   **exact permutation** $p$ over the $N!$ orderings.

Verification on evening 3: 8 cells (`arm_pl`, `softmax`, seeds 0–3), interleaved, `--device
cuda`.

**Hypotheses.** $N=8$; the exact permutation test's finest achievable $p$ is
$1/8!=2.48\times10^{-5}$, below the $0.05$ it is read against (M-9 satisfied); cells at
$n_{\rm train}=2048$, $s=64$; the two arms differ in cost ($1.884$ s against $1.524$ s at the
unit prices), which is what makes the blocked-order plant correlate.

**Evidence.** RUN `grep -c synchronize scripts/v15_r1.py` $=\mathbf{0}$ and RUN
`grep -n "time\.time()" scripts/v15_r1.py` $=$ `:249, :267, :585, :992`, all four printed
rather than the two the training loop uses; `READ V20_R15_JOURNAL.md:53 @
99777ab`; prices `READ docs/PLAN.md:971,978 @ 99777ab`.

**Mechanism.** C17 (the record's own finding, `READ V20_R15_JOURNAL.md:53`); M-15 (a
descriptive statistic without its null — the permutation $p$ is the null); V-15 (a
condemning rule with a planted negative — the blocked order); V-16.

**Kill.** (i) On any of the 8 CUDA cells `secs_sync < secs_unsync` (the synchronised clock
can only add waiting; a shorter reading means the bracketing is misplaced); **(ii) the mean
paired gap $\bar g = \mathrm{mean}_{i\le8}(\texttt{secs\_sync}_i-\texttt{secs\_unsync}_i)$
satisfies $\bar g < s_{\rm unsync}/\sqrt{8}$, where $s_{\rm unsync}$ is the sample standard
deviation of `secs_unsync` over the same 8 cells** (the harness never dispatched
asynchronously by an amount this instrument can resolve, so the instrument cannot measure
what it claims — V-16); (iii) the interleaved run's permutation $p\le0.05$ on
$\rho(\texttt{run\_order}, \texttt{secs\_sync})$; **(iv) any sentence in any book of this
canon quotes `wall` (the `:585`/`:992` whole-run bracket) as a cost, a price, a ratio or an
input to one** — the exempted timer read as the thing it was exempted from being. Frozen at
$\mathbf{0}$ such sentences. Instrument, $0$ GPU-s, decidable on the tree today:
`grep -rn '\bwall\b' docs/canon/*.md`, every hit read for whether it is a cost claim. RUN
2026-09-06 across `docs/canon/` the count of hits quoting `wall` as a cost is $\mathbf{0}$,
and clause (iv) is the standing rule that keeps it there once the field is journalled.
**Planted negative for (iv), in scope and firing:** the record's own withdrawn ratios, which
were whole-run wall-clock readings quoted as costs — `0.3455` GPU-h, `10.17×`, `41.9×`,
`2.85`/`118.8` (`READ V20_R15_JOURNAL.md:53 @ 99777ab`, C17) — every one of which clause (iv)
refuses, and which are the reason this verse's Terminal withdraws them. **True negative:**
the per-cell `secs_sync` figures of evening 3, which are costs and which clause (iv) must not
touch, and does not.

**Clause (ii)'s tolerance, frozen in the timer's own units, because "$=0.000$" was neither.**
The gap is the wall-clock of the trailing `torch.cuda.synchronize()` call. It is a float
seconds difference and is never exactly $0.0$; at three-decimal rounding a sub-millisecond
sync prints `0.000`, so the clause as it stood either never fired (read as exact float
equality) or fired on any nonzero gap (read as "the printed string is not `0.000`"), and the
verse named neither the comparison nor the rounding — M-2, a threshold not frozen, and V-11.
The bar is now the instrument's own resolution at $N=8$: the standard error of the
un-synchronised clock over the same 8 cells. A mean gap below one standard error is a gap the
8-cell design cannot distinguish from zero, and that is exactly the condition under which the
synchronised column buys nothing. **Both columns are printed per cell**, all 8 rows, with
$\bar g$, $s_{\rm unsync}$ and $s_{\rm unsync}/\sqrt{8}$ on the summary line; no rounding is
applied before the comparison. Hypothesis making the clause non-vacuous: $s_{\rm unsync}>0$
over the 8 cells — the record's own three observations of one cell span
$14.101\ldots17.839$ s (`READ V17_R4_RETAKE_PRICE.md:143-150 @ 99777ab`), so a zero sample sd
at $N=8$ would itself be the reportable event, and the clause is then read as `NOT MEASURED —
needs a clock with nonzero spread`.

Frozen as printed. Instrument: the repaired script, 8 cells. Price:
$4\times1.884+4\times1.524+4.0=17.63$ s DERIVED, **evening 3**; **plus** the planted-negative
run, which is a second invocation of the same 8 cells and costs the same
$4\times1.884+4\times1.524+4.0=17.63$ s DERIVED, **evening 7**. Planted negative: the same 8
cells in **blocked** order `A A A A B B B B`; the kill's clause (iii) must fire on it
($p\le0.05$) because order then encodes arm identity and the arms' costs differ — if it does
not fire on the plant, the order test is condemned as powerless at $N=8$ and $N$ is doubled to
16 ($8\times1.884+8\times1.524+4.0=31.26$ s DERIVED, **evening 8**, conditional on the plant
failing to fire).

**The plant has an evening and a price because a plant scheduled nowhere is a plant never
run.** Before repair this verse charged only the interleaved run and deferred the plant to "a
later evening" with no number, and 05.14 carried neither the $17.63$ s nor the conditional
$31.26$ s in its table or its total — D-6, a repair written and never started, and M-8, the
kill priced at one of its two runs' rate. The plant cannot share evening 3's invocation: the
interleaved and blocked orders are two different `--arms`/`--seeds` orderings of the same eight
cells and $17.63+17.63=35.26$ s is above 05.1's frozen $20.0$ cap, so it is its own sitting.
It sits at evening 7 rather than inside the first five because clause (iii) is read on the
**interleaved** run first: if $p>0.05$ there, the instrument's verification number for evening
3 already exists, and the plant is what licenses the *sentence* "the order test can detect a
run-order effect", which no book quotes before evening 7. 05.14's table and its total carry
both rows.

**If killed** — *(i) or (ii)*: the timer is moved outside the training loop to bracket the
whole cell including its first backward, and the cell's `secs_sync` is re-read; hypotheses:
the first step's allocator warm-up is inside the bracket on both columns; kill: $\bar g$ is
still below $s_{\rm unsync}/\sqrt{8}$ — the arms run synchronously on this stack and
`secs_unsync` is already an honest clock, which is recorded with $\bar g$, $s_{\rm unsync}$
and the eight paired rows printed, and the `secs_sync` column is retired as the redundant
one. *(iii) fires on the
interleaved run*: the run-order effect is real on this box (thermal or clock), and every
cost ratio is reported with `run_order` as a covariate and quoted only from same-invocation
pairs, the record's own rule (`READ scale/m3_flops.py:119-121 @ 99777ab`); hypotheses:
same-invocation pairs exist for every ratio; kill: a ratio quoted across invocations — it is
struck. That is the last link.

**Terminal.** The canon licenses no GPU-second ratio between arms from `scripts/v15_r1.py`
until the repair lands, and withdraws every it.3-onward ratio C17 already voided
(`0.3455` GPU-h, `10.17×`, `41.9×`, `2.85`/`118.8`, `READ V20_R15_JOURNAL.md:53`).

---

### 05.8 — B16(c): `lambda_hat` carries one bit

**Statement.** At `scripts/v15_r1.py:383`, `lambda_hat = float(lg.mean())` averages
$\log m_k$ over every position, so one $m_k=0$ sends it to $-\infty$; the finite-mean
sibling `lambda_hat_live` spans **`:384-385`** (the `if bool(fin.any()) else float("-inf")`
tail is the second line) and `frac_gate_annihilated` is at **`:386`**, both already existing
(RUN `sed -n '383,387p' scripts/v15_r1.py`). The `:384`/`:385` pair this verse's Statement
carried before repair was off by one column and by one line: `:385` is `lambda_hat_live`'s own
continuation, not `frac_gate_annihilated`, so a reader executing the repair at the printed
pins would have re-labelled the wrong expression — P-6 inside the one paragraph that tells an
editor which line to touch. The rest of this verse already pinned `:386` correctly (the
identity paragraph and clause (C) both read `sed -n '386p'`), so the Statement was the single
drifted site and is now consistent with them. The repair is documentary and one-line: every ledger, card and verse
reads `lambda_hat_live` **with** `frac_gate_annihilated` printed beside it, and
`lambda_hat` is renamed `lambda_hat_incl_annihilated` so that no reader mistakes it for a
mean.

**The identity, written on the denominator the code actually divides by.** The formula this
verse carried before repair read $\hat f_{\rm ann}=1-|F|/(n s)$, and $ns$ is not the
denominator. RUN `sed -n '386p' scripts/v15_r1.py` reads
`frac_gate_annihilated=float((~fin).double().mean())`, a mean over every element of `~fin`;
`fin = torch.isfinite(lg)` at `:380`; and `lg` is already restricted to the live band —
`m, theta = m[:, live], theta[:, live]` at `:364` before `lg = torch.log(m)` at `:371`. So the
denominator is $n\cdot|{\rm live}|$, where $n$ is the eval batch (`a.n_eval`) and
${\rm live}=\{{\rm head}+1,\dots,S-1\}$ with ${\rm head}=S-1-T^\star$ (RUN
`sed -n '697,698p' scripts/v15_r1.py`), so
$$|{\rm live}| = S-1-{\rm head} = T^\star,$$
which at the module defaults $S=64$, $T^\star=2$ is $|{\rm live}|=\mathbf{2}$ — **not** $64$.
Reading the frozen kill number as $1/(ns)$ would have made the single-annihilated-gate
quantum $1/(64n)$ where the code produces $1/(2n)$, a factor of $32$ at the shipped geometry,
which is P-7 (a name divided by the wrong referent) and V-8. The identity in force, with
$F=\{k\in{\rm live}\ \text{at any of the}\ n\ \text{rows}:\ \log m_k\ \text{finite}\}$:
$$\hat\lambda_{\rm live}=\frac{1}{|F|}\sum_{k\in F}\log m_k,\qquad
\hat f_{\rm ann}=1-\frac{|F|}{n\,|{\rm live}|},\qquad |{\rm live}|=T^\star,$$
and $|{\rm live}|$ is **printed in every row that quotes $\hat f_{\rm ann}$**, because
$T^\star$ becomes a flag under 05.6 and the denominator therefore varies from run to run.

**$F$ is a finiteness set, not a positivity set, and the difference is itself a kill.** The
code's complement is `~torch.isfinite(lg)`, which counts `nan` and `+inf` as well as $-\infty$.
The identity above is exact for the intended path — $m_k\ge0$, so $\log m_k\in[-\infty,\infty)$
and the only non-finite value reachable is $-\infty$ at $m_k=0$ — and any `nan` or `+inf`
entry in `lg` is **itself a kill**, not an annihilation: `nan` means the blend at `:363`
produced a non-number and `+inf` means $m_k$ overflowed, and either makes $\hat f_{\rm ann}$
count a failure as a zero gate. Clause (C) below decides it. The pair
$(\hat\lambda_{\rm live},\hat f_{\rm ann})$ is the reading; $\hat\lambda$ alone is
the indicator $\mathbb{1}[\hat f_{\rm ann}>0]$ in disguise.

**The $-\infty$ path exists on exactly one arm, and that arm is now on the schedule.** RUN
`sed -n 355,392p scripts/v15_r1.py`: `lg = torch.log(m)` at `:371` — the only line that can
produce $\log 0$ — is inside `elif kind == "arm_smprime"` (`:362`). The `arm_pl` branch at
`:361` reads `lg = model.heads(x)[0][:, live]`, a **log-gate head**: there is no $m$ and no
logarithm, so no annihilation can arise. The `softmax` branch at `:376-379` returns
`lambda_hat=0.0, lambda_hat_live=0.0, frac_gate_annihilated=0.0` as constants and exits
before `lg` is read at all. 05.7's evening-3 cells are `arm_pl` and `softmax` (RUN over
`results/v15_r1.jsonl`: header `arms: ['arm_pl','softmax']`), so **on the schedule as it
stood, this verse's kill could not fire on either arm registered** — V-11, a precondition
never met at any real draw, and V-3. The repair registers the missing cell: **one
`arm_smprime` cell is scheduled at evening 3c**, `--device cuda --seeds 0 --arms arm_smprime`,
priced $1.884+4.0=5.88$ s at the capped-cell unit price, and it is the instrument the kill
below is decided on. 05.1's split and 05.14's table both carry it.

**Hypotheses.** $m_k\ge0$ (the gate is a magnitude); $n\,|{\rm live}|$ positions per cell with
$|{\rm live}|=T^\star=2$ at the module defaults and printed per run (RUN
`sed -n '697,698p' scripts/v15_r1.py`), **not** $ns$; the cell is an
`arm_smprime` cell — `arm_pl` has a log-gate head and `softmax` returns constants at
`:376-379` (RUN), so neither is in this verse's scope.

**Evidence.** RUN of the lines quoted; `READ docs/CEQ_SHAPE.md:1280 @ 99777ab` (the debt as
filed); the record's own attachment of the Koopman field to `:384` and not `:383` —
`READ docs/CEQ_SHAPE.md:**1222**` `@ 99777ab`, whose row reads "transfer-operator / Koopman
spectral theory, attached to `lambda_hat_live` at `:384` and not `lambda_hat` at `:383` —
one bit". The pin this verse carried before repair, `:1219`, is a markdown table separator:
RUN `sed -n '1219p' docs/CEQ_SHAPE.md` reads `|---|---|---|---|---|`. The substantive debt
pin at `:1280` was and remains correct (RUN `git show 99777ab:docs/CEQ_SHAPE.md | sed -n
'1280p'`).

**Mechanism.** V-8 (a label that is constant — $-\infty$ on every cell with one annihilated
gate); V-16; P-7 (a name with no referent — `lambda_hat` reads as a mean and is not one).

**Kill — two clauses, on two different objects, at two different prices.**

**(A) The journal census, $0$ GPU-s, decidable on the tree today.** The verse this replaces
priced a "$0$ GPU-s journal replay" against `results/chase_k1_replay_it0.txt`'s class and
against `results/v15_r1.jsonl`, where the instrument does not exist: RUN over
`results/v15_r1.jsonl` (26 lines), the union of all row keys contains no `lambda_hat`, no
`lambda_hat_live`, no `frac_gate_annihilated` and no per-position `log m` array, because that
journal's arms are `arm_pl` and `softmax`. The instrument does exist elsewhere, and it is
named here: `results/v17k_r4_retake.jsonl` carries **408** rows with a `lambda_hat` field
across all three arms (RUN). The census, one pass over that file, no new code:

| clause | frozen | RUN today, 2026-09-06 |
|---|---|---|
| every row with `frac_gate_annihilated` $>0$ reads `lambda_hat` $=$ `-Infinity` | $0$ violations | **131** such rows, all `arm_smprime`, **$0$** violations (`grep -o '"lambda_hat": [^,]*' \| grep -v numeric` reads `131 "lambda_hat": -Infinity`) |
| every row with `frac_gate_annihilated` $=0$ reads `lambda_hat` bitwise equal to `lambda_hat_live` | $0$ violations | **277** such rows (`arm_pl` $136$, `softmax` $136$, `arm_smprime` $5$), **$0$** violations |
| no row with `frac_gate_annihilated` $>0$ reads a finite `lambda_hat` | $0$ | **$0$** |

**Planted negative, in scope and populated:** the 277 `frac = 0` rows are the negative
control — a broken finite mask (`fin = torch.isfinite(lg)` at `:380`) makes `lambda_hat` and
`lambda_hat_live` diverge **there**, on rows nothing was planted into, so the census condemns
the mask rather than the arithmetic; and the 131 `-Infinity` rows are the positive control
that shows the census can see the annihilation at all. A census with only one of the two
populations is powerless, and both are RUN-nonempty.

**(B) The $10^{-12}$ arithmetic clause, $5.88$ GPU-s, evening 3c.** `lambda_hat_live` must
equal the mean of $\log m$ over the finite positions to $10^{-12}$, and
`frac_gate_annihilated` must equal the non-finite count divided by $n\,|{\rm live}|$ exactly,
with $|{\rm live}|$ printed on the same row. **No journal
in `results/` carries the per-position $\log m$ vector** (RUN: no such array in any row key
union), so clause (B) is **not** decidable at $0$ GPU-s and its instrument is moved out of the
If-killed and priced here: **one `arm_smprime` cell re-run with `--trace-every 1`** (`:556`),
$1.884+4.0=5.88$ GPU-s, evening 3c. Until that cell runs, clause (B) reads
`NOT MEASURED — needs the per-position trace`.

**(C) The non-finite class clause, $0$ GPU-s on the traced cell of (B), and free before it.**
`~torch.isfinite(lg)` at `:386` does not distinguish $-\infty$ from `nan` and `+inf`, so
$\hat f_{\rm ann}$ silently counts an arithmetic failure as an annihilated gate. The clause:
**any traced position with `lg` equal to `nan` or `+inf` is a kill of the cell, not an
annihilation** — the blend at `:363` produced a non-number, or $m_k$ overflowed, and the cell's
$\hat f_{\rm ann}$ is withdrawn rather than read. Frozen: the count of `nan` and `+inf` entries
in the traced `lg`, at $\mathbf{0}$. Instrument: the same `--trace-every 1` cell as (B), no
extra invocation. **Planted negative, in scope and firing on the journal today:** the $131$
`-Infinity` rows of `results/v17k_r4_retake.jsonl` are the *admitted* non-finite class, and
clause (C)'s selector must **not** fire on them while firing on any other non-finite token.
RUN over that file, tokenising `"<field>": <NaN|Infinity|-Infinity>` field by field, the whole
non-finite census is: `lambda_hat` $\to$ `-Infinity` $131$; `lambda_hat_0step` $\to$
`-Infinity` $8$; `dyn_range_bound` $\to$ `Infinity` $30$; `dyn_range_bound_0step` $\to$
`Infinity` $12$; `exp_scan` $\to$ `NaN` $2$. Both $\hat\lambda$ fields carry **zero** `NaN` and
**zero** bare `Infinity`, so clause (C) reads $\mathbf{0}$ today and the two populations are
separated by the selector rather than by assumption — and the selector is shown to be a
selector and not a tautology by the $44$ `Infinity` and $2$ `NaN` tokens it must ignore in the
three neighbouring columns, which a field-blind `grep` for the tokens would have swept in.
Until (B)'s cell runs, (C) is decidable only on the aggregate `lambda_hat` field and reads
`NOT MEASURED at the position level — needs the per-position trace`; its aggregate half is RUN
and reads $0$.

**If killed** — *(A) fires*: the defect is in the finite mask or the annihilation counter,
both at `:380-386`, and it is localised without a GPU by re-running the census restricted to
the offending `kind` and `seed`; hypotheses: the offending rows share an arm. Evidence: the
census output. Kill of that replacement: the violations do not share an arm — then the mask
is not arm-specific and every `lambda_hat`-derived sentence in every book is withdrawn
pending clause (B)'s cell. *(B)'s trace path emits no per-position gates*: hypotheses: the
`--trace-every` path at `:556` reaches the gate columns (RUN `:736-743` shows the `t="trace"`
row emitting `lambda_hat`, `lambda_hat_live`, `frac_gate_annihilated` — the three scalars, and
**not** the vector). Kill of that replacement: it emits the three scalars only, which RUN
already shows it does — so clause (B) has **no instrument in the tree**, the alternative would
be a new column function, which is code the standing rule forbids, and the loss is carried by
the Terminal rather than by a fourth link.

**Terminal.** The canon licenses `frac_gate_annihilated` as the only gate statistic from the
un-repaired script and withdraws every sentence about $\hat\lambda$'s sign or magnitude that
cites `:383`. With clause (B)'s cell unrun, it licenses in addition the journal census of
clause (A) — `$0$ violations over 408 rows of `results/v17k_r4_retake.jsonl`, RUN 2026-09-06`
— as evidence that the **mask** is consistent, and withdraws any claim that the **value** of
$\hat\lambda_{\rm live}$ has been checked against the positions it averages: that check has no
instrument in the tree and is `NOT MEASURED — needs the per-position trace`.

---

### 05.9 — B3: the prose-coupled test policy (jointly with book 04), the 9 erroring files, and the registry the deletion severed

**Statement.** The policy, stated jointly with book 04's instruments section: **a test
asserts code or a journal (`results/**/*.jsonl`, `results/*.json`), never a prose document**;
a test that opens a `.md` file is a documentation check and lives under `attic/tests/`,
which `pytest.ini` does not collect (`READ pytest.ini:19 @ 99777ab`).

**The census, re-run 2026-09-06 with its rule printed beside it, because the three constants
this verse shipped before repair are produced by no quoting rule.** The rule, stated so it can
be executed: over `tests/**/*.py` excluding `__pycache__`, a *name* is a quoted string
matching `["'][A-Za-z0-9_.-]+\.md["']` — a single token, no path separator, no whitespace, so
a docstring sentence that happens to contain `.md` is not a name; a name is *absent* if the
path `<repo root>/<name>` does not exist:

| quantity | frozen | RUN 2026-09-06 | what this verse claimed before repair |
|---|---|---|---|
| test `.py` files on disk | — | **280** | 280 (correct) |
| files naming some `.md` | — | **15** | 68 |
| files naming an absent `.md` | **0** after (b) | **5** | 59 |
| distinct absent documents named | — | **8** | 68 |
| files under `tests/` that **open** an absent `.md` | **0** | **0** | (not measured) |
| files mentioning `.md` anywhere, comments included | — | 141 | — |

Under the loosest quoted rule (any quoted string ending `.md`, glob patterns `*.md` and
`DONE*.md` admitted) the three counts read $16$ / $7$ / $10$; the two globs are patterns, not
documents, and excluding them returns the frozen $15$ / $5$ / $8$. The **complete** list of
the five, with the names each carries and how it carries them (RUN):

| file | absent names | how |
|---|---|---|
| `tests/beds/test_bed_k.py` | `CEQ_V15_CONTRACT.md` | docstring citation at `:1`, `:139`, `:157` — never opened |
| `tests/chase/test_schedule_rebuild.py` | `THEORY.md` | docstring at `:1` and an assertion **message** at `:61` — never opened |
| `tests/loop/test_merkle_journal.py` | `c.md`, `c1.md`, `c2.md` | `tmp_path / "c1.md"` fixtures written by the test itself at `:83-84`, `:101`, `:113`, `:137` — not root documents at all |
| `tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py` | `ralph-loop.local.md` | `REPO / ".claude" / "ralph-loop.local.md"` at `:44` — a harness config path, not a round report |
| `tests/mercury/test_r9_table_cut.py` | `capability_table_v0.md`, `capability_table_v1.md` | `assert md.name == …` at `:150`, `:153` — asserts a **produced filename**, reads nothing |

**The decisive number is the last row of the first table: $\mathbf{0}$.** RUN, over all 280
files, the count of lines that both name an absent `.md` and carry `read_text`, `open(`,
`.exists(` or `read_bytes` is **zero**. No test under `tests/` opens an absent prose
document. The prose-coupling this book was written to price **does not exist on the tree
today**, and move plan (b) below applies to at most 5 files — and on inspection to **0**,
since none of the five is coupled to a deleted round report in any sense the policy
condemns.

The **13 files listed as erroring at collection** are likewise struck: four of the thirteen do
not exist. RUN `[ -e ]`: `tests/loop/test_attic_never_removes_the_last_must_fire.py` ABSENT,
`tests/saturn/test_v20_r15_freeze_manifest.py` ABSENT,
`tests/loop/test_no_struck_constant_ships.py` ABSENT (all deleted at `c71527a`); RUN
`ls tests/mercury` lists no `test_v20_r15_it26_*` file at all — the it26 file is
`tests/jupiter/test_v20_r15_it26_live_claim.py`; `…_it27_` and `…_it34_` survive only as
`__pycache__` artefacts. The real population is the **9** files of 05.2's fourth-class table,
each with its raised exception printed there; and **none of the 9 raises
`FileNotFoundError` or `AssertionError`, and none names a `.md`** (RUN exception census,
05.2). Every one of them fails on an `import` of a *test module* deleted at `c71527a`, which
is a different defect from prose coupling and gets a different repair below.

Deletions from the pin to HEAD, **counted by the instrument that is stable at the pin**: RUN
`git diff --name-only --diff-filter=D 99777ab HEAD` reads **350** paths. The two instruments
this verse used before repair are both re-run and both recorded. RUN
`git diff --cached --name-only --diff-filter=D | wc -l` reads **0** — the deletions were
committed as `c71527a`, so the staged-diff form measures an empty index and produces the count
in no session after the commit; it is struck as this verse's instrument, P-1 with the producer
alive but pointed at the wrong tree. RUN
`git show --name-only --diff-filter=D --pretty=format: c71527a | grep -c .` reads **350** and
is correct, but it is a single-commit form: it counts what `c71527a` deleted, not what is
absent from HEAD, and the two agree here only because no other commit between `99777ab` and
HEAD deletes a path. The `99777ab HEAD` range form is the one frozen, because it is what the
kill's threshold is about. The breakdown, RUN and **summed in print**:
$$244\ \text{root `.md`} + 53\ \texttt{tests/**/*.py} + 2\ \text{other } \texttt{tests/}
+ 14\ \texttt{.superpowers/} + 34\ \texttt{attic/} + 3\ \text{loose} = \mathbf{350},$$
the two other `tests/` files being `tests/deimos/DEIMOS_REPORT.md` and
`tests/mars/MARS_REPORT_IT2.md`, and the three loose paths `house-events.jsonl`,
`scale/chase_struck_coverage.py`, `scale/doc_readings.py`. The **34** is a repair: this verse
wrote $35$ before, and $244+53+2+14+35+3=351$ — a breakdown that did not sum to its own total,
which no reader could have caught because the sum was never printed. RUN
`git diff --name-only --diff-filter=D 99777ab HEAD | grep -c '^attic/'` reads $\mathbf{34}$.
Every class count above is RUN by the same command with its own `grep -c`, and the six add to
the total exactly. RUN
`git ls-tree -r 99777ab --name-only | grep -c '^tests/.*\.py$'` reads **333** test files at
`99777ab`, 280 on disk. One deletion severs a live producer:
`tests/loop/test_no_struck_constant_ships.py` is the **struck-constant registry** —
`STRUCK.md:7-13` calls it the registry and itself a pointer, `inspector.py:455-462` runs it and
asserts it non-empty, and `scripts/render_struck.py:17` reads it; RUN `python
scripts/render_struck.py` raises `FileNotFoundError` on that path. The plan: (a) restore that
one file from the pin (`git show 99777ab:tests/loop/test_no_struck_constant_ships.py`) — it
asserts against shipped code and against the lead documents, which is a code assertion
about prose *absence*, the one form the policy admits (it must read `0` hits, not a line
number); (b) move to `attic/tests/` every file the census finds **opening** an absent root
`.md` — RUN 2026-09-06 that set is **empty**, so (b) is a **no-op on the tree as it stands**
and is retained only as the standing rule for files written later (a git move by the
coordinator; no writer moves a file); **(b′) the repair the 9 collection errors actually
need**, which is not a move at all: each of the 9 imports a *test module* deleted at
`c71527a`, so each is either restored from the pin beside its importer
(`git show 99777ab:tests/jupiter/test_v20_r15_it20_citation_freeze.py` and the three
siblings — `…it27_star_lands_and_overturns`, `…it14_theory_table`,
`tests/saturn/test_v20_r15_freeze_manifest.py`, `tests/saturn/test_v20_r15_it27_wing_arm_citation.py`)
or the importer is retired to `attic/tests/` with it; the choice is the coordinator's and is
recorded per file in the evening report, because restoring a module that itself asserts a
deleted document would re-import the defect; (c) the census must then read `0` files under
`tests/` **opening** an absent root `.md` (it already does); **(d) the ordering, frozen here
because without it the kill fires on a correctly executed repair**: steps (a)–(c) move and
restore files that *name* `.md` documents, and **not one of the 9 collection errors is a prose
read** — all nine raise `ImportError`/`ModuleNotFoundError` on
`tests.saturn.test_v20_r15_freeze_manifest`, `tests.jupiter.test_v20_r15_it14_theory_table`,
`tests.jupiter.test_v20_r15_it20_citation_freeze`,
`tests.jupiter.test_v20_r15_it27_star_lands_and_overturns` and
`tests.saturn.test_v20_r15_it27_wing_arm_citation`, and none opens a `.md` (RUN exception census,
05.2). The collection number is therefore frozen at **`9` until (b′) lands and `0` only after
(b′)**; a verifier running (a), (b) and (c) alone and then reading the collection clause at `0`
would read a failure on a repair executed exactly as written (V-16, an instrument that cannot
measure what it is pointed at; M-17, the correction record classified as the defect). The final
verification is `pytest -q --collect-only` reading `0 errors`, and it is read **after (b′)**,
never after (c).

**Hypotheses.** The scan's rule is the one printed in the census table above — a quoted
single token matching `["'][A-Za-z0-9_.-]+\.md["']`, absent if the repo-root path does not
exist — and it is executed, not described; a file
naming one of the 11 surviving root documents is not counted as coupled (`MISTAKES.md`,
`STRUCK.md`, `V17K_RULINGS.md`, `CONTRACT.md`, `COSTS.md`, `CEQ_V20_R15_CONTRACT.md`,
`MODEL_CARD.md`, `README.md`, `MATHEMATICS.md`, `V16_CALIBRATION.md`,
`workdonenewseal.md`) — those 9 lead-document assertions in the registry test are
absence checks and stay.

**Evidence.** RUN counts as printed, each beside the rule that produced it; RUN
`git cat-file -e 99777ab:<path>` returns clean for all five restore targets named in (b′), so
the restore command is executable and is not P-4 scaffolding; `READ STRUCK.md:7-13,21 @
99777ab`; `READ inspector.py:455-462 @ 99777ab`; `READ scripts/render_struck.py:17 @
99777ab`; `READ docs/sources/README.md:17-20 @ 99777ab` (the deletion and the pin). The
charter's row counts **56** test files (`READ docs/canon/CHARTER.md:229`, live, no pin — RUN
`git ls-tree -r 99777ab --name-only | grep -ci canon` reads $0$); the RUN here counts 53
`tests/**/*.py` deletions **at `c71527a`** and **5** surviving files that name an absent
`.md`, of which **0** open one — three different scans of one event, each stated with its
rule, and the third of them says the coupling the charter row asserts no longer has a
population.

**Mechanism.** P-5 (doc rot pointing at nothing — `STRUCK.md`, `inspector.py` and
`render_struck.py` now point at a deleted file); P-6 (line-reference drift — this verse's own
13-file list named four files that do not exist, which is the mechanism turned on the book);
P-1 (a number with no producer — the $68$ / $59$ / $68$ triple, which no quoting rule
generates and which is struck from the canon here); V-16 (a dead suite reports no failures —
`pytest.ini:12-18`'s own words); M-17 (a census that classified the correction record as the
defect — the registry test is a correction record and was deleted with the prose it guards
against).

**Kill.** After (a), (b′), (c) and the ordering (d): `python scripts/render_struck.py` does not print a table
with `12 entries` (`READ STRUCK.md:18 @ 99777ab` — RUN `grep -n "12 entries" STRUCK.md` reads
`18:Rendered from \`tests/loop/test_no_struck_constant_ships.py\` at \`aa82df7\`. 12 entries.`,
so this pin is **correct as written** and the attack's re-pin to `:17` is refused by its own
instrument: `:17` is blank), or the **opens-an-absent-`.md`** census
reads $>0$, or collection reads $>0$ errors. Frozen: `12`, `0`, `0`. Instrument: the three
commands. Price: $0$ GPU-s; collection $14$–$16$ s CPU (RUN). Planted negatives, both RUN
today: (1) `render_struck.py` raises `FileNotFoundError: … tests\loop\test_no_struck_constant_ships.py`
on the present tree — clause 1 fires; (2) collection reads $22$ — clause 3 fires. Clause 2
(the census) reads **$0$ on the present tree and therefore does not fire**, and the verse says
so rather than claiming a plant it does not have: the naming census reads $5$, the opening
census reads $0$, and a rule frozen at $0$ that already reads $0$ is satisfied by
construction (V-10). Clause 2's own planted negative is therefore stated as owed and
**decidable at $0$ GPU-s the moment it is needed**: a file the coordinator moves *into*
`tests/` from `attic/tests/` that opens a deleted round report must make the census read
$\ge1$; until such a file exists the clause is a standing rule with no population, marked
OPEN, and clauses 1 and 3 carry the verse.

**The plant for the *policy* itself, re-derived so that it writes no file.** The plant this verse
carried before repair was "a temp test that opens `README.md` and asserts a line number must be
**caught** by the census … and the plant must then read `1`" — a plant that requires a new `.py`
file under `tests/`, which the canon forbids outright: `CHARTER.md` §3, "**No code file is
written** — no `.py`, `.sh`, `.lean`, `.ipynb`, no scratch helper in the tree". A condemning rule
whose only planted negative is an act the constitution forbids has no planted negative (V-15,
D-6). The plant is replaced by one that runs entirely inside `git`, on the **pinned tree**, where
the prose-coupled population still exists by the hundred and no file is created:

```
git grep -lE "[\"'][A-Za-z0-9_.-]+\.md[\"']" 99777ab -- 'tests/*.py' | wc -l
git grep -lE "[\"'][A-Za-z0-9_.-]+\.md[\"'].*(read_text|open\(|\.exists\(|read_bytes)|(read_text|open\(|\.exists\(|read_bytes).*[\"'][A-Za-z0-9_.-]+\.md[\"']" 99777ab -- 'tests/*.py' | wc -l
```

RUN 2026-09-06 at `99777ab`: the first reads **68**, the second **20**, over **30** matching
lines. Both are positive, so the census instrument is shown to *detect* prose coupling on a
population that exists, at $0$ GPU-s, with no file written and no working-tree change — which is
what a planted negative is for. Two facts are recorded with it rather than smoothed over. First,
the pinned count **68** is exactly the first element of the struck triple $68$ / $59$ / $68$
(Finding 15, batch 1): the number was never fabricated from nothing — it is the **pin's** naming
count read as though it were the tree's, which is P-6 (a count carried across a commit boundary)
rather than P-1 (a number with no producer), and the canon records the sharper diagnosis. Second,
the pin's own *opening* count is $20$, not $68$ and not $59$, so even at `99777ab` the population
the charter row describes was a third of the size the row's own number implies. The clause the
plant serves is book 04's stricter rule — "opens a `.md`", not "opens an absent `.md`" — and on
the pinned tree that rule reads $20$ where the absent-document rule reads less.

**If killed** — *the registry does not render 12*: the restored file's registry dict has
drifted from `STRUCK.md`'s 12 rows; the file at the pin is the registry of record and
`STRUCK.md` is regenerated from it (`READ STRUCK.md:12` — RUN `grep -n "Regenerate this file"
STRUCK.md` reads `12:Regenerate this file from the registry; never edit the table by hand:`;
the `:14-16` this verse pinned before repair is the **code fence** below it, ` ``` ` /
`python scripts/render_struck.py > STRUCK.md` / ` ``` `, which is the command and not the
rule, so the sentence was cited to the three lines under it — P-6), never the reverse; hypotheses:
the pin's file renders; kill: it does not render at the pin either — then the registry has
no producer at any commit and every struck constant is re-listed as a row in
`CORRECTIONS.md` with its `STRUCK.md` line as evidence, which is the last link. *Collection
still reads $>0$ after (b′)*: a restored module itself imports a third deleted module, or
itself asserts a deleted document; the fix is a **different object**, not a deeper restore —
the importer is retired to `attic/tests/` and the *assertion it carried* is re-filed as a row
in `CORRECTIONS.md` naming the constant or claim it guarded, so the guarantee survives the
file. Hypotheses: every one of the 9 importers guards a statable claim. Evidence: RUN, each
of the 9 imports a named symbol (`:27`–`:48`, tabled in 05.2), so the claim is recoverable
from the import site. Kill of that replacement: an importer whose imported symbol is not a
claim but a fixture — then nothing is re-filed and the file is retired with a one-line note,
which is the last link. *The census reads $>0$ after the move*: the remaining files are listed
by name in the evening report and moved in a second pass; kill: a file that both opens a
journal and a `.md` — it is split by the coordinator into two files, the journal half kept.

**Terminal.** The canon licenses no test result from the root suite while collection reads
$>0$ errors, licenses the 12 struck constants as struck on `STRUCK.md`'s authority alone
until the registry renders, and withdraws every sentence of the form "the suite is green".
It withdraws in addition, and permanently, the sentence this book carried before repair that
**59 test files are coupled to deleted prose**: RUN 2026-09-06 the number of files under
`tests/` that open an absent `.md` is $\mathbf{0}$, the number that so much as name one is
$\mathbf{5}$, and none of the five is a round-report coupling. B3 is therefore closed not by a
move but by a **re-measurement**: the population the row describes is empty, and what remains
of the row is the 9 deleted-test-module imports, which are a different defect with a different
repair. Clause 2 of the Kill stands **OPEN** for want of a population to plant into.

---

### 05.10 — B17: the Gate-0 circularity resolved as a written rule

**Statement.** Gate 0 closes at open-rulings $=0$, and rows 1 and 2 are blocked on numbers
only the Kaggle run produces (`READ V17K_RULINGS.md:10-17 @ 99777ab`). The rule this book
writes, replacing "open-rulings = 0" as the launch condition:

> **Gate 0 is GREEN when every row's pre-launch half is landed in a file and every row's
> post-run half is an explicit empty slot whose producer is named **by `file:line`** in the
> frozen producer table below, **where the named line, read at the pin, emits the slot's own
> value**, and whose acceptance number is frozen before launch.**
> A row with a post-run half and no producer named by `file:line` is RED; a row whose named
> `file:line` is a real line that emits something other than the slot's value is **also** RED,
> and it is the second clause that does the work, because a filled field is the easy half. A
> number that arrives from the run into a slot with no frozen acceptance is not a closure; it
> is a reading.

**The producer table, written here because the list the rule names was never filed.** The rule
this verse shipped before repair referred its central term to "the frozen deciding-cell list
under Ruling 6f", and that list does not exist at any commit: `READ V17K_RULINGS.md:150 @
99777ab` reads "cell-id list frozen at launch under Ruling 6f. Owed by the **envelope node**" —
*owed*, not filed; `:80` states 6f's rule and `:145` states that "B2's membership is a list, not
a description"; RUN `git grep -l "cell-id list" 99777ab` finds no file holding one. A kill whose
predicate is membership of a list nobody wrote is not decidable on any row (V-10, P-4), so the
rule is re-written on a term this book can freeze and a reader can execute: the producer is the
**`file:line` that emits the slot's value**, and the table is filed here as a specification, which
`CHARTER.md` §3 permits inside markdown. It may shrink at launch, never grow (Ruling 6f's
direction).

| slot | producer, by `file:line` | pin |
|---|---|---|
| `FLOOR_TRAIN_ABS_DLOSS` (row 1) | `scripts/k_noise_floor.py:379`, `abs_delta_final_loss=abs(final_a - final_b)` | RUN `sed -n '379p' scripts/k_noise_floor.py` |
| $\Lambda$ per $\beta$ (row 2) | `kaggle/ceq_v17k.ipynb`, **cell 18 only**, appended after `T.train(...)` | `READ V17_R10P_LRT.md:501 @ 99777ab` |
| row 3 | none — the row is entirely pre-launch | — |

**Row 2's producer had no producer.** The string "cell 18" appears nowhere in
`ceq/compat.py`: `READ ceq/compat.py:48-54` is about `ensure_generation_mixin`'s alias being
process-local and why K-COMPAT runs before `python -m ceq.hf.smoke` in a subprocess, and RUN
`grep -rn "cell 18" --include=*.md .` hits only `kaggle/snapshot/repo/V17_R10P_LRT.md` at `:29`,
`:80`, `:501` and `:508`. That is P-2, a number sourced to a line that does not carry it, and P-5,
a pointer at nothing. The pin is corrected to the document that does name the cell —
`V17_R10P_LRT.md:501 @ 99777ab`, "`kaggle/ceq_v17k.ipynb`, **cell 18 only**, appended after
`T.train(...)` and its summary prints" — and one discrepancy in the required repair itself is
printed rather than reconciled: the path the finding names, `kaggle/snapshot/repo/V17_R10P_LRT.md`,
is **not at the pin** (RUN `git cat-file -e 99777ab:kaggle/snapshot/repo/V17_R10P_LRT.md` reads
`fatal: path … exists on disk, but not in '99777ab'`; RUN `git log --oneline --` on it reads
`9ce3048` and `208cf69`). At `99777ab` the file is at the **root**, `V17_R10P_LRT.md`, and `:501`
there carries the line verbatim (RUN `git show 99777ab:V17_R10P_LRT.md | sed -n '501p'`). The
snapshot copy is cited live as `READ kaggle/snapshot/repo/V17_R10P_LRT.md:501 @ 9ce3048` beside
the primary pin.

Applied to the three rows:

| row | pre-launch half (landed locally) | post-run half (slot, producer cell, frozen acceptance) |
|---|---|---|
| 1 determinism | L-TOL amendment filed (`READ V17K_RULINGS.md:112-119`); hop/forward bitwise both regimes and gradient `NOT EXECUTABLE` under the flag on the certified stack (05.5 evening 2); the **local** floor at reduced scale (below) | `FLOOR_TRAIN_ABS_DLOSS` on the Kaggle card, producer: the first two identical-seed chunks of Q3, `abs_delta_final_loss` from `scripts/k_noise_floor.py:379`; acceptance: the value is **printed**, never compared to the local floor (V-22), and every training comparison on that card is read against it |
| 2 the corner | Ruling 10′'s criterion text with its constants $3.841$, $\ln n$ ($8.29$ at $n=4000$ `[RUN]` at the pin), the minimum detectable departure formula (`READ V17K_RULINGS.md:389-436`); the two-forward-pass code path smoked on the local reduced-scale checkpoint (below) | $\Lambda$ per $\beta$ on the trained Q3 model, producer: `kaggle/ceq_v17k.ipynb` cell 18 (`READ V17_R10P_LRT.md:501 @ 99777ab`; the pre-repair pin `ceq/compat.py:50-52` carries no cell number and is struck); acceptance: the three-branch card with `≥95 %` / `≤5 %` pinned (`READ V17K_RULINGS.md:299-303`) and the gradient census word |
| 3 matched params | entirely pre-launch: exact counts $25{,}736{,}232$ against $25{,}728{,}000$, $+8{,}232=8\cdot(2\cdot(512+1)+3)=+0.03200\%$ (`READ V17K_RULINGS.md:237-243`), matched by Ruling 3; the one COSTS line is drafted at the pin (`git show 99777ab:V17_R1_R3_R7_EDITS.md`) and lands by the coordinator's paste — verification `grep -c '25,736,232' COSTS.md` reads `1` | none |

**What a local smoke run at reduced scale settles**, evening 4: `python
scripts/k_noise_floor.py --steps 30 --device cuda` at the script's defaults ($d=512$, $L=8$,
$H=8$, seq $512$, batch $8$ — `READ scripts/k_noise_floor.py:399-405 @ 99777ab`, the Q3 shape)
runs two identical-seed chunks; price $2\times30\times0.299025=17.94$ s DERIVED from the
`[MEASURED]` step, under the cap.

**The $17.94$ s is an "at least", and the direction is printed rather than left to the reader.**
`READ COSTS.md:239 @ 99777ab` prices the Q3 step at $0.299025$ s `[MEASURED]`, and
`READ COSTS.md:67-69 @ 99777ab` states what produced it: "Median over 3 independent child
processes of the median of $\ge12$ timed steps and $\ge3.0$ s of timed work, **2 warm-up steps
discarded**". That is a **steady-state** per-step unit with allocator and kernel warm-up
excluded by construction. This verse runs $30$ steps per chunk (RUN
`sed -n '399p' scripts/k_noise_floor.py` reads `ap.add_argument("--steps", type=int,
default=200)`, so $30$ is well below the script's own default and the warm-up is a larger
fraction of it), and the first two steps of **each** of the two chunks are warm-up steps the
unit price does not cover. The arithmetic $2\times30\times0.299025=17.94$ is exact; what it is
**not** is an estimate of the run. Four of the sixty steps are above the steady-state floor by
an unmeasured amount, so the realised wall-clock is **at least** $17.94$ s and never below it,
which is the same "at least" every other price in this book carries (P-8, a floor read as a
point) and is the sense in which the cap comparison is honest: a floor under the cap does not
prove the run is under the cap. The overrun is bounded by nothing this book has measured —
`NOT MEASURED — needs the $30$-step cell's own timing`, which evening 4 produces at $0$ extra
cost as the run's own wall-clock, and against which $17.94$ s is checked as a floor rather
than as a prediction.

It settles three things and not a fourth: (i) the harness
executes end-to-end on the certified stack under `warn_only=True` and writes
`results/k_noise_floor.json` with `abs_delta_final_loss` and `measurement_validity` (`:379,
:428`); (ii) the **local** floor at 30 steps, a number for local training claims only;
(iii) that a checkpoint exists on which Ruling 10′'s two forward passes can be smoked
(row 2's pre-launch half). It does **not** settle the Kaggle floor — a floor carried across
the device boundary is V-22 and is refused by the rule above.

**Hypotheses.** The Q3 step price is a floor (P-8), so $17.94$ s is "at least"; the local
floor at 30 steps is not the floor at 200 or at a chunk (M-3, pilot spread is not realised
spread) and is labelled by its `FLOOR_CHUNK_SPEC` (`READ V17K_RULINGS.md:169-173`); **this verse's producer
table** — not Ruling 6f's owed-and-never-filed cell-id list — is the frozen list, and it may
shrink at launch, never grow (Ruling 6f's direction).

**Evidence.** As pinned; the reduced-scale price DERIVED; the Kaggle run-1 log, which
halted in cell 2 on a code-source gate before any pip list (`READ requirements-kaggle.txt:12-19
@ 99777ab`; RUN `results/kaggle_v17k_output/ceq-v17-k.log:22` reads `[HALT] a code source is
configured -- set CODE_DATASET_DIR or REPO_URL above`), so no post-run half has ever been
produced.

**Mechanism.** V-10 (a gate satisfied by construction — the rule forbids a slot without an
acceptance); V-22; M-7 (a pre-registration with a hole — the empty slot is the hole, named);
D-4 (order: nothing behind the yes is started).

**Kill.** A row of Gate 0 whose post-run slot has no producer named by `file:line` in **the
producer table of this verse**, or whose acceptance is written after the run's timestamp; or the
local smoke exits non-zero or writes `measurement_validity` other than the script's valid token.
Frozen: the three-row producer table above, timestamp order, exit `0`. Instrument: the producer
table read against the rulings table, and the smoke run. Price: $17.94$ s GPU, evening 4.

**Planted negative — a slot whose producer field is FILLED, which the rule must still refuse.**
The plant this verse carried before repair was row 1 as `V17K_RULINGS.md` states it at the pin,
"a slot (`FLOOR_TRAIN_ABS_DLOSS`) with an owner (\"the floor node\") and **no producer of any
kind**". That plant is a **construction**: a rule keyed on a field being empty fires on
emptiness by definition, so it demonstrates nothing about the rule's power and is V-10 and
V-15 — the condemning rule still has no negative it could have failed to catch. It is struck as
the plant and kept only as the reading of that row. The plant in force is row 2's **pre-repair
pin**, `ceq/compat.py:50-52`, which is a filled producer field naming a file that exists and
lines that exist, and which the rule must refuse on the second clause. RUN
`git show 99777ab:ceq/compat.py | sed -n '48,53p'`: the three lines read "Step \"3b\" runs /
`python -m ceq.hf.smoke` in a SUBPROCESS, which gets a fresh interpreter and no / alias --
which is why K-COMPAT runs BEFORE it". They emit no $\Lambda$, no $\beta$, no likelihood ratio
and no tensor of any kind; RUN
`git show 99777ab:ceq/compat.py | sed -n '50,52p' | grep -cin "lambda\|lrt\|beta"` reads
$\mathbf{0}$. The whole file reads $\mathbf{1}$ on the same pattern — `:44`,
"`tests/gate0/test_g16_lrt_pinned.py::test_the_notebook_computes_lambda_at_the_`" — and that
single hit is what made the pin look plausible: the module *names the test that checks the
notebook computes $\Lambda$*, seven lines above the cited range, and names it in a docstring.
Naming the test is not emitting the value, which is exactly the distinction clause two draws.
The refusal, shown: the field is filled, so clause one passes; the named
line does not emit the slot's value, so clause two fails and the row reads **RED** — and row 2
carried that pin, in this book, until batch 2 replaced it. **True negative, equally required:**
row 1's producer as this verse now files it, `scripts/k_noise_floor.py:379`, RUN
`git show 99777ab:scripts/k_noise_floor.py | sed -n '379p'` reads
`abs_delta_final_loss=abs(final_a - final_b)` — the slot's own value, emitted on the named
line — on which the rule must **not** fire, and does not. The two together show clause two is a
test and not a formality: it separates two filled fields by what their lines emit.

**If killed** — *the smoke fails under `warn_only`*: the harness is run on CPU at the same
30 steps (`--device cpu`), which the script's docstring says it was proved on
(`READ scripts/k_noise_floor.py:4-6`); hypotheses: the CPU pair is bitwise; kill: it is not —
the harness, not the device, is broken, and row 1's post-run half has no producer until it
is repaired (the row is RED and the gate is RED; nothing launches, which is the standing
rule anyway). *The author rules the split illegitimate* (05.11 does not list this as one of
the five; it is the register's own open question, `READ V17K_RULINGS.md:15-17`): the gate's
text reverts to "open-rulings = 0", rows 1 and 2 cannot close before the run, and the run
cannot start — the canon then licenses only the local programme (books 01–04) and the
Kaggle rows stay `NOT MEASURED`, which is `docs/PLAN.md` §5.10's last table row
(`READ docs/PLAN.md:1075` — RUN `grep -n "N-24 (the yes withheld)" docs/PLAN.md` reads
`1075:| N-24 (the yes withheld) | nothing local | the whole local programme; the Kaggle rows
stay \`NOT MEASURED\` |`. The `:1077` this verse cited before repair is the section's
horizontal rule; the attack's re-pin to `:1076` is a blank line. Both are P-6 and the RUN
line number is the one frozen).

**Terminal.** The canon licenses "nothing launches without the author's explicit yes" as
the invariant that survives every reading of the gate, and withdraws every Kaggle-side
number from every book until the yes and the run exist.

---

### 05.11 — B18: the five open author rulings, the default each takes, and what each default costs if wrong

**Statement.** Each ruling is open at the pin (`READ docs/CEQ_SHAPE.md:1268-1270 @ 99777ab`).
The default is chosen by Ruling 2a's precedence rule — *the criterion that licenses the
weaker sentence wins ties* (`READ V17K_RULINGS.md:319-320 @ 99777ab`) — never by which reading
scores.

**The Ruling 2a attribution is verified by heading census and stands; the attack against it is
refuted by RUN and recorded here so the pin is never re-litigated.** The attack read that "the
cited lines sit outside that ruling's block", on the ground that Ruling 2a's heading is at
`:275` and Ruling 10′ supersedes its criterion at `:389`. Both of those facts are true and
neither puts `:319-322` outside Ruling 2a. RUN
`git show 99777ab:V17K_RULINGS.md | grep -n '^# RULING'` reads `275:# RULING 2a`,
`328:# RULING 8`, `354:# RULING 9`, `389:# RULING 10′`, so Ruling 2a's block is
`:275`–`:327` and $275<319<322<328$: the **GRADIENT CENSUS** paragraphs at `:306-317` and the
**PRECEDENCE** paragraph at `:318-322` are subsections *of* Ruling 2a, not a neighbouring
section. RUN `git show 99777ab:V17K_RULINGS.md | sed -n '318,322p'` reads "**PRECEDENCE.**
This is the DEFAULT criterion. If the β node's proposal arrives / with a defensible basis that
differs, it is adjudicated against this one by / **which claims less** — the criterion that
licenses the weaker sentence wins / ties. (L-SIGN's direction, applied to definitions.)" —
verbatim, under Ruling 2a's heading. Ruling 10′ supersedes Ruling 2a's **PINNED criterion**
(`|β_i,final − 1| ≤ 5·δ_β,i`, `:297-299`) and says nothing about the precedence clause, which
is a tie-break rule over criteria and survives its own criterion's supersession — the canon
uses it for exactly that, to pick between two readings of an open ruling. The citation is
therefore kept **with** its ruling number, and the two line numbers that would have been lost
had the attribution been dropped — the block boundary `:328` and the supersession `:389` — are
printed here so a later reader can re-run the census rather than re-guess it.

| # | open ruling | default the plan takes | what the default costs if the author rules the other way |
|---|---|---|---|
| 1 | clause-1 tail: `CP-lower > 0.5` two-sided ($0.4762$, W3 fails) or one-sided ($0.5156$, W3 clears) (`READ V20_R15_JOURNAL.md:1530-1556 @ 99777ab`) | **two-sided 95 % Clopper–Pearson** — the reading that claims less; no entrant clears clause (1) and the R15 arena returns no winner | zero GPU-s: the same 16 cells are re-read one-sided ($0$ s); the cost is one scoreboard row (the R15 $+12$) under-claimed for the interval between default and ruling, and no number falsified |
| 2 | the F0–F4 rubric, cited and never written (`READ docs/PLAN.md:1405`; `READ V20_R15_JOURNAL.md:2656`) | **the canon grades nothing on F0–F4**; every claim carries the charter's evidence class and, for Lean rows, the `[M]/[S]/[D]` grade; the R15 theory table's twelve cells are not re-scored | if the author supplies a rubric: the twelve cells are re-graded by reading, $0$ GPU-s, one evening; no verse in books 01–06 depends on an F-grade |
| 3 | the Kaggle attach behind the chess witness: `lichess_chess_games` and `lichess_chess_evaluations` are UNPINNED (`READ COSTS.md:36-37`), no chess bed is **registered** in `kdata.BED_SPECS` and no arm consumes a chess corpus (`READ V20_R15_JOURNAL.md:2425-2428 @ 99777ab`) | **no attach is requested, and none is needed**: the chess bed is book 09's **BED-C-G** (09.2) — python-chess self-play under the uniform-random-legal-move policy, generator $+$ seed $+$ hash under Ruling 7 — at $\mathbf{0}$ Kaggle attaches, with `python-chess 1.11.2` installed (`[RUN]`, `READ V20_R15_JOURNAL.md:2425-2427 @ 99777ab`) and `ceq/kdata.py:258` `label_plies` replaying the mainline through a `chess.Board` today (**BUILT, RUNS TODAY**, same rows). The first chess cell is 09.13 at $0$ GPU-s; the first GPU cell is 09.14 | if the author reverses it, the reversal is **additive, not enabling**: the two Kaggle attaches become a *second, optional* source — book 09's **BED-C-D**, one lichess monthly dump pinned by its own SHA-256 into `results/k_data_manifest.json` — and `lichess/chess-evaluations` is replaced locally by 09.6's engine recompute, never read as a label. Price: $2$ evenings after the yes for the digests and one natural-ground bed; nothing local is lost and nothing local was waiting (09.37: "Nothing in this book launches, uploads or attaches on Kaggle") |
| 4 | the gate's blindness to F4 cells (three cells the it.35 gate cannot read; `READ V20_R15_JOURNAL.md:2803-2808`, J-14) | **J-14 as written**: F4 is NOT-PUT with an admission condition; in the canon the admission condition is S-12's domain census at $0$ GPU-s (book 04), and a cell with no instrument is `NOT MEASURED — needs <instrument>` | $0$ GPU-s: three cells re-labelled; if the author rules F4 leap-material, book 07's attacks on those cells are re-opened by reading |
| 5 | TERMINAL against NOT-PUT — two tokens for one predicate and none for "a bound an experiment removes" (`READ V20_R15_JOURNAL.md:8072-8080`) | **four states, mapped onto the verse**: TERMINAL = the verse's Terminal sentence; NOT-PUT = `NOT MEASURED — needs <instrument>`; LEAPABLE = a DERIVED block owed; OPEN = a Kill not yet run; the R15 tokens are not used in the canon | renaming only, $0$ GPU-s; the mapping is a table in `CORRECTIONS.md` if the author names the fourth state differently |

**Row 3's reversal column contradicted the constitution and book 09, and both are named.** The
cell this verse carried before repair read "two Kaggle attaches behind the yes … Ruling 7 cannot
apply to an external PGN", and its default read that "the chess witness is not a bed" and that the
canon's categorical-state witness is BED-S's argmin head. Both are false against
`CHARTER.md` §6's own scope for book 09 — "the local data route (lichess database dumps or
**python-chess-generated games, hashed and pinned per Ruling 7**) **before any Kaggle attach**" —
and against book 09 as written (09.2 registers BED-C-G as generator $+$ seed $+$ hash; 09.37
states that nothing in that book attaches). Ruling 7's own form is generator $+$ seed $+$ expected
hash, regenerated and asserted in-notebook (`READ V17K_RULINGS.md:83-86 @ 99777ab`), which a
generated corpus satisfies exactly and which the pre-repair cell asserted it could not. That is
V-22 (a constraint carried in from a system where the corpus was an external attach) and P-5 (a
pointer at a route the canon does not take). The corrected reading is the row above: the chess bed
exists locally at $0$ attaches, and the attach is an optional second source.

**Hypotheses.** No default changes a number; each is a reading rule. The two-sided default
in row 1 is the one the record's own office refused to pick because picking after the data
is M-2 (`READ V20_R15_JOURNAL.md:1545-1549`) — the canon picks the weaker before any new cell
exists, which is the M-2-safe moment.

**Evidence.** As pinned; the interval values are `[RUN]` at the pin and READ here.

**Mechanism.** M-2 (defaults frozen before new data); D-7 (the counter of each default is
the other reading, printed); P-7 (row 5's tokens had no referent); V-10 (row 4: a gate blind
to cells it cannot read is a gate satisfied by construction on those cells).

**Kill — a decidable property of the defaults, not an event.** The kill this verse carried before
repair was "the author rules against a default", frozen at "the count of reversed defaults, $0$ to
$5$", instrument "the author's ruling". That names no draw, no seed rule, no $N$ and no
instrument in the repository: no number anywhere can make it fire and none can make it fail to
fire, which is exactly what `CHARTER.md` §2's reachability rule refuses (V-11, M-3). An author's
ruling is an **event** the canon records, never a kill it can price. The kill is re-written on the
defaults themselves, and it is decidable today at $0$ GPU-s by inspection:

> **Any default in the table above whose reversal changes a number rather than a reading.**

Frozen: every cell of the table's right-hand column reads $\mathbf{0}$ GPU-s **and** falsifies no
number, for all five rows. Instrument: reading the five cells — row 1 re-reads the same 16 cells
one-sided ($0$ s, one scoreboard row under-claimed, no number falsified); row 2 re-grades twelve
cells by reading ($0$ s, no verse in books 01–06 depends on an F-grade); row 3 adds an optional
second source and enables nothing that was blocked ($0$ s locally, $2$ evenings *after* a yes that
gates nothing local); row 4 re-labels three cells ($0$ s); row 5 renames four states ($0$ s). $N$:
the five rows, exhaustively, at $N=5$. Price: $0$ GPU-s, and it is decidable **before** any
ruling exists, which is the point — a default that could only be tested by asking the author is
not a default, it is a question. **Planted negative, in scope and firing:** row 3's own
pre-repair cell, which read "two Kaggle attaches behind the yes … Ruling 7 cannot apply to an
external PGN" — a reversal that *did* change a number (it made the chess bed's Kaggle-attach count
$2$ instead of $0$ and denied that a generated corpus is pinnable), and the kill fires on it, as
it did in the paragraph above.

**A second plant, because the first one is a repaired cell and the rule that picks the defaults
needs a negative of its own.** The sentence this verse carried before repair — that every
default is filed "with the ruling slot left open beside it, so the plant fires (the slot is
visibly open) rather than being closed by construction" — is not a plant at all: a slot's being
open is a property of how this book laid out its own table, not a measurement, and a rule that
fires on visible openness fires on layout by definition (V-10, V-15). The plant in force is a
**sixth row written into the same table with its ruling slot as open as the other five**, which
the precedence rule must nonetheless refuse:

| # | open ruling | the plant's "default" | why the precedence rule refuses it |
|---|---|---|---|
| **6 (plant)** | clause-1 tail, the same open ruling as row 1 | **one-sided 95 % Clopper–Pearson**, `CP-lower` $=0.5156$ | it licenses the **stronger** sentence |

The refusal is arithmetic and RUN at the pin, not a judgement.
`READ V20_R15_JOURNAL.md:1532-1534 @ 99777ab` prints, for `arm_pl`, $12/16$ crossings at rate
$0.7500$, "**CP-lower two-sided 95 %** $=0.4762$ — **FAILS by** $0.0238$" and
"**CP-lower one-sided 95 %** $=0.5156$ — **CLEARS**". The clause-1 bar is `CP-lower > 0.5`:
the one-sided reading clears it and returns a winner with the R15 $+12$ scoreboard row; the
two-sided reading does not and the arena returns none. Ruling 2a's precedence rule reads, at
`READ V17K_RULINGS.md:319-320 @ 99777ab`, "**which claims less** — the criterion that licenses
the weaker sentence wins ties", so the plant's row is refused **on its content**, with both its
slot and row 1's slot equally open. The plant is a filled default, indistinguishable from the
five by layout, and the rule separates it from them by which sentence it licenses — which is
the only thing a default-picking rule can be tested on. **True negative, equally required:** row
5, a pure renaming of four states onto the verse's own fields, which licenses no sentence at all
and on which neither the kill nor the precedence rule fires — and neither does. The author's
ruling, when it arrives, is filed verbatim in `CORRECTIONS.md` as an **event** with its date, and
row 1's history is what that filing protects: the office that refused to pick and filed
⟨CLAUSE_1_TAIL⟩ is why every default here is filed as a *default*, never as a closure.

**If killed** (a default whose reversal changes a number). That default is **withdrawn as a
default** and re-filed as an OPEN question with no default at all, because a default that costs a
number is a decision the canon is not licensed to take without the author (`CHARTER.md` §0's
instruction source rule). Hypotheses: the reversal's cost is readable from the table's right-hand
column at $0$ GPU-s. Evidence: the five cells as printed. Kill of the replacement: the question
cannot be stated without assuming one of the two readings — then it is not a question either, and
the affected verses in books 01–06 print `NOT MEASURED — needs the author's ruling` in place of
every number the reading touches. That is the last link. The chain is two links deep because
every reversal is a reading, not a run.

**Terminal.** The canon licenses the five defaults as *defaults*, never as rulings, and
withdraws any sentence that treats a default as the author's word.

---

### 05.12 — B19: the HuggingFace package, the parameter-count residual, and the three UNPINNED Kaggle sources with a local alternative for each

**Statement.** `ceq/hf/` holds four modules (`configuration_ceq.py`, `modeling_ceq.py`,
`smoke.py`, `train.py`, RUN `ls`); **no trained checkpoint exists** and none may be named
(`READ docs/CEQ_SHAPE.md:1288-1289 @ 99777ab`; `READ scripts/export_hf_weights.py:3-9` — the
only exported weights are 4,769-parameter probe arms and may not be called
`model.safetensors`); `MODEL_CARD.md:19` opens with "The operator does not work". The
package plan is N-23 (`READ docs/PLAN.md:856-860`): the shape replaces the dead operator,
parity at $\gamma=0$ against the package's own softmax attention, counts recounted per arm,
card rewritten limits-first, no checkpoint. Verification on evening 5, CPU, $0$ GPU-s:
`python -m ceq.hf.smoke` exits `0` and prints parity `True` — today it cannot run at all
(RUN: `import ceq.hf.modeling_ceq` raises on `PreTrainedModel`, 05.2). The parameter
residual: $25{,}736{,}232-25{,}728{,}000=8{,}232=n_{\rm layers}\cdot\big(2(d+1)+3\big)$ at
$n_{\rm layers}=8$, $d=512$ (`READ V17K_RULINGS.md:237 @ 99777ab`), the arm carrying more;
Ruling 3 matches it and forbids re-architecting to close it (`READ V17K_RULINGS.md:56-59`).

**The census, printed once tensor by tensor, because the formula and its cross-check were being
read as two incompatible inventories.** `READ V17K_RULINGS.md:239-241 @ 99777ab` names the
per-layer extras as "the arm's **two per-position heads** (`m_head`, `theta_head`, each a
`[1,d]` weight plus a `[1]` bias) and **three scalar switches** (`beta`, `qk`, `g`)", i.e. two
width-$(d{+}1)$ tensors and three scalars; `READ V17K_RULINGS.md:243-244 @ 99777ab` cross-checks
with `4,806 − 4,769 = 37 = 2·16 + 5` at $d_{\rm model}=16$, which reads as two width-$d$
tensors and **five** scalars. Both total $37$ at $d=16$, so kill (ii) — "the recount differs
from the formula, an unnamed tensor exists" — could not distinguish them, and a recount landing
on either would have passed. They are the **same** census written two ways, and the census is
this:

| tensor | shape | count |
|---|---|---|
| `m_head` weight | $[1,d]$ | $d$ |
| `m_head` bias | $[1]$ | $1$ |
| `theta_head` weight | $[1,d]$ | $d$ |
| `theta_head` bias | $[1]$ | $1$ |
| `beta` | scalar | $1$ |
| `qk` | scalar | $1$ |
| `g` | scalar | $1$ |
| **per layer** | — | $\mathbf{2d+5}$ |

and $2(d+1)+3 = 2d+2+3 = 2d+5$ identically, so the "three scalar switches" decomposition and
the "five scalars" decomposition differ only in whether the two biases are counted with their
weights or with the switches. Both are the seven tensors above. Instances: at $d=16$,
$2\cdot16+5=37$, the record's cross-check; at $d=512$, $2\cdot512+5=1{,}029$ per layer and
$8\cdot1{,}029=8{,}232$, the residual. Kill (ii) is decidable against **the seven-row table**,
not against either shorthand: a recount that names six tensors or eight is the unnamed-tensor
event, and a recount that names these seven and totals anything but $2d+5$ is an arithmetic
error in the recount. The three
UNPINNED sources (`READ COSTS.md:36-38`):

| source | why unpinned | local alternative |
|---|---|---|
| `tinystories_cdla` | a Kaggle attach; unhashable until a session attaches it | `data/tinystories_20k.txt`, $18{,}167{,}706$ bytes, $211{,}765$ lines (RUN `ls -l`; `READ V17K_RULINGS.md:262-266`), SHA-256 `276781813f9ae1690789e727ef4b3e5f877dc6233fbd0b9cdd6acba930e685e5` (`READ data/CHECKSUMS.sha256:? @ 99777ab` — the line naming that file), already `ceq/hf/train.py:49`'s `_LOCAL_CORPUS` |
| `lichess_chess_games` | a Kaggle attach; no registered bed consumes it | none needed under 05.11 row 3's default; if the author reverses it, the attach's SHA-256 is the pin |
| `lichess_chess_evaluations` | streamed (`streaming=True` needs a network, `READ requirements-kaggle.txt:44-46`) | none exists and none can: a streamed source has no bytes to hash; it is refused as a bed source by Ruling 7 until an attach with bytes replaces it |

`enwik8` is PINNED at `2b49720e…c024a8`, byte-identical between the local copy and the
first $10^8$ bytes of `jamesmcguigan/hutter-prize`'s `enwik9` `[MEASURED]`
(`READ kaggle/README.md:31-38`).

**Hypotheses.** The residual formula names every extra tensor (`m_head`, `theta_head`, three
scalar switches) and the control has no parameter the arm lacks (`READ V17K_RULINGS.md:240-243`);
N-23's shape module adds a $\gamma$ scalar and boundary-row identities, which are
parameter-free rows, so the recount is the formula plus $1$ per layer or per model — printed,
not assumed.

**Evidence.** As pinned; RUN listings; RUN import failure.

**Mechanism.** P-1 (no card sentence with a bracket filled by anything but a FOUND cell);
P-3 (the card's header is the retraction and stays); Ruling 3 / M-8; Ruling 7 / P-1 for the
sources.

**Kill.** (i) `python -m ceq.hf.smoke` exits non-zero after 05.3, or parity reads `False`;
(ii) the recount after N-23 differs from $n_{\rm layers}(2(d+1)+3)+c_\gamma$ with $c_\gamma$
the printed count of $\gamma$ scalars — an unnamed tensor exists; (iii) any card bracket
filled without a FOUND `kind` in `results/`. Frozen: exit `0`, `True`, exact count equality,
zero unfounded brackets. Instrument: the smoke, `sum(p.numel())` per arm, the FOUND census
by `kind`, **against the seven-row tensor census above**. Price: $0$ GPU-s (CPU forward at
$s=64$). Planted negative for (ii): a recount of the seven-row census with `theta_head`'s bias
dropped, which totals $2d+4$ and must fire — at $d=16$ it reads $36$ against the record's
measured $4{,}806-4{,}769=\mathbf{37}$ (`READ V17K_RULINGS.md:243-244 @ 99777ab`), a
one-parameter miss the census resolves and neither shorthand could, since $2\cdot16+5$ and
$2\cdot(16{+}1)+3$ both read $37$ whichever of the two biases is missing from the reader's
inventory. **True negative:** the census as printed, which reproduces $37$ at $d=16$ and
$8{,}232$ at $d=512$, $n_{\rm layers}=8$, and must not fire. For (iii): `MODEL_CARD.md`'s header numbers, which are FOUND
(`results/m3_quintuple_v2*.jsonl`, RUN listing) and must pass.

**If killed** — *(i)*: the package ships no shape and the card's capability paragraph is
§5.10 tree B's sentence (`READ docs/PLAN.md:1054`); hypotheses: the softmax module of the
package still runs; kill: it does not — the package is withdrawn from the tree's deliverables
and 05.15's B19 row moves to "after the arm exists". *(ii)*: the tensor is named in the
card's count table and the count is re-derived tensor by tensor; kill: the arm's count then
exceeds the control's by more than $0.032\%$ — Ruling 3's match no longer holds by
magnitude and the canon reports the residual as an unmatched difference in every table
header until the author rules. *(iii)*: the bracket is emptied to `NOT MEASURED`.

**Terminal.** The canon licenses the package as a **parity harness with no checkpoint**
and withdraws every sentence about a shipped model until a FOUND training cell exists.

---

### 05.13 — B24: the documentation discipline that keeps the tree at its 2026-09-05 size, with the census that detects a violation

**Statement.** The tree on 2026-09-05 holds **11** `.md` files at the root and **21** regular
files (RUN `find . -maxdepth 1 -type f`); at `99777ab` the root held **255** `.md` (RUN
`git ls-tree`). The rules, each with the count it enforces:

1. Root `.md` count is **11** and the set is fixed: `CEQ_V20_R15_CONTRACT.md`, `CONTRACT.md`,
   `COSTS.md`, `MATHEMATICS.md`, `MISTAKES.md`, `MODEL_CARD.md`, `README.md`, `STRUCK.md`,
   `V16_CALIBRATION.md`, `V17K_RULINGS.md`, `workdonenewseal.md` (RUN `ls *.md`).
2. Four classes never enter the tracked tree: per-iteration reports (`_IT[0-9]+_`), loop
   prompts (`^LOOP_PROMPT`, `_ARCHIVE\.md$`), contracts and deltas of finished rounds
   (`^CEQ_V1[0-9]_.*DELTA\.md$`; a finished round's contract is read at the pin), SDD ledgers
   (`^\.superpowers/`), and the working boards (`^(DONE|BOARD|CHECKLIST|BACKLOG|NOTES|REPORT)\.md$`).
3. `docs/canon/CORRECTIONS.md` is the only door: a flaw in any book is a row there
   (`READ docs/canon/CHARTER.md:5-9`), never an edit and never a new report file.
4. A report file per session is **regenerated, never accumulated**: one file,
   `workdonenewseal.md`, overwritten each session (it is the root file the author reads,
   `READ docs/canon/CHARTER.md` §5 row B12's citation of it); the previous session's text
   survives only in git history.
5. Journals (`results/**/*.jsonl`, `results/*.json`) are appended, never moved or deleted
   (L-G2); they are not documents and rule 1 does not count them.

The census, run at the end of every evening and printed in the report:

```
find . -maxdepth 1 -name '*.md' | wc -l                                   # must read 11
git ls-files | grep -cE '_IT[0-9]+_|^LOOP_PROMPT|_ARCHIVE\.md$|^CEQ_V1[0-9]_.*DELTA\.md$|^\.superpowers/|^(DONE|BOARD|CHECKLIST|BACKLOG|NOTES|REPORT)\.md$'   # must read 0
git ls-files 'docs/canon/*.md' | wc -l                                    # must read 12 after birth (00–09, CHARTER, CORRECTIONS)
```

**The third number is 12, and the 10 this verse froze before repair would have fired on a correct
birth commit.** `CHARTER.md` §6 lists **ten** books — `00_NORTH_STAR.md`, `01_THEORY_ACCURACY.md`,
`02_THEORY_TRAINING.md`, `03_KERNEL.md`, `04_BEDS_AND_INSTRUMENTS.md`, `05_REPAIRS.md`,
`06_PREDICTIONS.md`, `07_ATTACKS.md`, `08_ARCHITECTURE.md`, `09_CHESS_AND_MARKETS.md` — plus
`CHARTER.md` itself and `CORRECTIONS.md`: $10+1+1=\mathbf{12}$. The parenthetical "(00–07,
CHARTER, CORRECTIONS)" omitted books 08 and 09, the two the author's own §0 clarification adds,
and froze the census at their absence; a coordinator committing the canon as the CHARTER
specifies it would have read `12` against a frozen `10` and the kill would have fired on the
correct tree (M-2, a threshold frozen against the wrong object; P-1, a count with no producer —
no command in the tree produces $10$). RUN today: `git ls-files 'docs/canon/*.md' | wc -l` reads
$\mathbf{0}$ — the canon is untracked (`git status` reads `?? docs/canon/`) — and RUN
`ls docs/canon/` lists ten files on disk (01–06, 08, 09, CHARTER, CORRECTIONS), books 00 and 07
being the coordinator's, written last. The census line is therefore read **only after the birth
commit**, and before it the correct reading is $0$, not a violation.

**Hypotheses.** The census's third line runs after the birth commit that tracks `docs/canon/`;
before it that line reads $0$ (RUN) and is not a violation. The staging hypothesis this verse
carried before repair — "the census runs after the coordinator's commit of the 350 staged
deletions (until then `git ls-files` still lists the deleted paths, and the second line reads the
pin's values)" — is **deleted as spent**: the deletions were committed at `c71527a`, not staged,
and RUN today `git diff --cached --name-only --diff-filter=D | wc -l` reads $\mathbf{0}$ while
the second census line on the tracked tree reads $\mathbf{0}$ already, not $133$ and not the pin's
values. `docs/sources/` (35 files: `design/` 9, `sweep/` 8, `judge/`, `plan/`,
`sections/`, the brief and notes — RUN listing) is admitted once as the paper's sources
(`READ docs/sources/README.md:1-5`) and is itself frozen: no file is added to it after birth.

**Evidence.** RUN counts on disk against the pin for each regex class: `_IT[0-9]+_` disk $0$ /
pin $98$; `^LOOP_PROMPT` $0/7$; `_ARCHIVE\.md$` $0/6$; boards $0/5$; `^\.superpowers/` $0/14$;
deltas $0/3$. The record's own count of the process becoming the product: round 15 produced
102 files and a 40-row corrections index (`READ docs/canon/CHARTER.md:250`, live, no pin — that is
census row **B24**, this verse's own row; the pre-repair pin `:231` is row B5, "no CEQ arm has been
trained", and is struck).

**Mechanism.** M-17 (the correction record classified as the defect — the census counts
report files, never journals or the registry); P-3 (a stale claim never retracted — one
report file, overwritten, cannot accumulate stale claims that are no longer read); P-5.

**Kill.** Any census line reading other than its frozen number after the birth commit.
Frozen: `11`, `0`, `12`. Instrument: the three lines. Price: $0$. **Planted negative, re-stated on
the pinned tree because the working-tree plant is spent:** line 2's regex run against `99777ab`
through `git ls-tree` rather than `git ls-files` —

```
git ls-tree -r 99777ab --name-only | grep -cE '_IT[0-9]+_|^LOOP_PROMPT|_ARCHIVE\.md$|^CEQ_V1[0-9]_.*DELTA\.md$|^\.superpowers/|^(DONE|BOARD|CHECKLIST|BACKLOG|NOTES|REPORT)\.md$'
```

— RUN 2026-09-06 reads $\mathbf{127}$, and the census fires on it. Two corrections travel with
that plant, both against this book. First, the working-tree plant this verse carried before
repair is **spent**: RUN `git ls-files | grep -cE '…'` on the tree today reads $\mathbf{0}$, not
the pin's values, because `c71527a` committed the deletions, so the plant fired on a state that no
longer exists and the clause was left with no negative at all (V-15). Second, the number is
$\mathbf{127}$, **not the $133$** this verse printed as "(RUN, summed per class)": the per-class
counts are $98$, $7$, $6$, $3$, $14$, $5$ and their sum is $133$, but the census line counts
**files**, and RUN `git ls-tree -r 99777ab --name-only | grep -E '^LOOP_PROMPT' | grep -cE
'_ARCHIVE\.md$'` reads $\mathbf{6}$ — all six `_ARCHIVE.md` files are `LOOP_PROMPT_*_ARCHIVE.md`
and are counted twice by the sum. $133-6=127$. A per-class sum quoted as the output of a
file-counting command is P-1, a number whose producer is a different command from the one named;
the frozen plant is the command's own reading, $127$.

**If killed** — *a forbidden path is tracked*: the path is deleted by the coordinator in the
next commit and its content, if load-bearing, becomes a `CORRECTIONS.md` row citing it at
the commit that held it; hypotheses: the content is citable by `git show <sha>:<path>`;
kill: the path was never committed (a working-tree-only file) — then it has no pin, its
numbers are `NOT MEASURED` (P-2), and the file is deleted without a row. *The root count
moves*: a twelfth root document is either a canon book misplaced (moved under
`docs/canon/`) or a report (deleted); the last link is the count itself, re-frozen in
`CORRECTIONS.md` only by an author instruction.

**Terminal.** The canon licenses the tree's 2026-09-05 shape as the reference and withdraws
citation rights from any file that is not in the tree at a named commit.

---

### 05.14 — The first five evenings, in order, with prices, and the critical path in evenings

**Statement.**

| evening | what | verses | GPU-s (at least) | verification number |
|---|---|---|---|---|
| 1 | environment: the pin set, steps 1–6; the drift census printed | 05.2, 05.3, 05.4 | $0$ | `2.5.1+cu121 12.1 True NVIDIA GeForce RTX 4060 Laptop GPU`; collection `9 errors` (the deleted-test-module set of 05.2 only) |
| 2 | re-certification, the ≤20 GPU-s half: `--only determinism,zerostep` | 05.5 | $\le10.64$ `[FITTED, extrapolated at $n=16384,32768$]` | hop/forward bitwise both regimes; gradient raises under the flag; `30/30` |
| 3 | the timer: synchronised, interleaved, order-journalled, 8 cells (`arm_pl`, `softmax`, seeds 0–3) | 05.7 | $17.63$ DERIVED | `secs_sync ≥ secs_unsync` on 8/8; $\bar g \ge s_{\rm unsync}/\sqrt{8}$ with both columns and all 8 rows printed; permutation $p>0.05$ interleaved |
| 3b | the harness flags: the identical-seed pair at the defaults, plus the `--s 32 --arms arm_pl` run | 05.6 | $11.05$ `[FITTED]` $+$ $5.88$ `[ASSUMED — the $s=64$ capped cell as a ceiling on the $s=32$ cell]` $=16.93$ | header `s=64,d=24,t_star=2`; the `--s 32` header reads `s=32`; its two `t="bind"` rows read `s=8` and `s=32`, each with `residual` $<10^{-14}$; the `--s 32` run's `secs` $\le1.884$ s, or evening 3b is repriced from the measured cell |
| 3c | the one-bit reading: the journal census at $0$ GPU-s, then one `arm_smprime` cell with `--trace-every 1` | 05.8 | $5.88$ DERIVED | census `0 violations` over 408 rows of `results/v17k_r4_retake.jsonl`; the traced cell's `lambda_hat_live` to $10^{-12}$ and `frac_gate_annihilated` exact against $n\,\lvert{\rm live}\rvert$ with $\lvert{\rm live}\rvert=T^\star$ printed; zero `nan` and zero `+inf` in the traced `lg` |
| 4 | Gate-0 rule written; local reduced-scale floor smoke, two chunks of 30 steps at the Q3 shape | 05.10 | $17.94$ DERIVED | exit `0`; `abs_delta_final_loss` printed with its `FLOOR_CHUNK_SPEC`; rows 1–3 each with a landed pre-launch half and a named post-run producer |
| 5 | the five defaults filed; the package smoke on CPU; the registry restore and the (b′) import repair; the documentation census | 05.9, 05.11, 05.12, 05.13 | $0$ | smoke exit `0`, parity `True`; `render_struck.py` prints `12 entries`; opens-an-absent-`.md` census `0`; documentation census `11 / 0 / 12`; collection `0 errors` |

Evening 3 is **three sittings**, not one: RUN-priced, 05.7's 8 cells ($17.63$ s), 05.6's three
runs ($16.93$ s) and 05.8's one cell ($5.88$ s) need different `--arms` and different
`--seeds`, so no two of them share an invocation, and $17.63+16.93=34.56$ s and
$16.93+5.88=22.81$ s are both above the frozen $20.0$ cap. The phrase "05.6's pair is inside
the same invocation set" that this table carried before repair named an invocation that
cannot exist and is deleted.

**The evenings beyond the five, priced, because two of this book's kills buy runs that sat in no
table.** 05.7's kill depends on a **blocked-order** plant that the verse deferred to "a later
evening" with no number, and on a conditional $N\to16$ re-run; neither appeared here before
repair, so the book carried a kill whose planted negative was scheduled nowhere (D-6) and a
total that omitted it (M-8). Both now have an evening and a price:

| evening | what | verses | GPU-s (at least) | verification number |
|---|---|---|---|---|
| 6 | the full certificate, all five stages, `--out results/k_cert_local_e6.json` | 05.5 | $523.9$ `[MEASURED]` | bar $\delta/\mathrm{tol}<50\%$; both laws $R^2\ge0.99$; the within-stack cell inside $2.2\times$ its own refitted law; the across-stack ratio inside the `[ASSUMED]` $2.0\times$; `C_RESIDUAL`/`C_OPERATOR` within $\pm5\%$ |
| 7 | 05.7's **planted negative**: the same 8 cells in blocked order `A A A A B B B B` | 05.7 | $4\times1.884+4\times1.524+4.0=17.63$ DERIVED | permutation $p\le0.05$ on $\rho(\texttt{run\_order},\texttt{secs\_sync})$ — the plant must fire |
| 8 | conditional, only if evening 7's plant does **not** fire: the same design at $N=16$ | 05.7 | $8\times1.884+8\times1.524+4.0=31.26$ DERIVED | the plant fires at $N=16$, or the order test is condemned as powerless and every ratio is quoted from same-invocation pairs only |

Evening 7 cannot be merged into evening 3: the interleaved and blocked orders are two orderings
of the same eight cells and $17.63+17.63=35.26$ s is above the frozen $20.0$ cap. **The book's
full priced programme** is therefore $69.02$ (evenings 1–5, seven sittings)
$+\,523.9$ (evening 6) $+\,17.63$ (evening 7) $=\mathbf{610.55}$ GPU-s, and
$610.55+31.26=\mathbf{641.81}$ GPU-s if evening 8 is triggered — both at the class of their
weakest input, `[FITTED, extrapolated at $n=16384$ and $n=32768$]`. The $\le69.02$ figure below
is the **first five evenings only** and is never quoted as the book's total.

Evening 6 is the full certificate ($523.9$ s `[MEASURED]`), the first evening above the cap;
it runs beside `docs/PLAN.md`'s four-evening Lean gate (J-L0 → J-L3 → J-L10 → J-L18) and
adds nothing to the path; evenings 7 and 8 sit behind it and add nothing to the path either,
because no book quotes the order test's *power* before a verdict is filed. **Critical path:** $1 \to 2 \to$ the plan's 14 evenings to a
verdict (`READ docs/PLAN.md:876-889 @ 99777ab`), because S-20 — the first card that runs a
cell — requires the certified stack (evening 1) and the determinism regime it runs under
(evening 2); evenings 3, 4 and 5 run beside the Lean gate. `READ docs/PLAN.md:885 @ 99777ab` reads, verbatim:
"**13.5, called 14 evenings to a verdict and a filed dossier.** Add M-5.3 (the ladder on a
survivor, 1) for **15**; add the cost-law branch S-66(3) → M-5.2(1) … for **19**." With this
book's $+2$ (evenings 1 and 2, which S-20 cannot start without) the three numbers are
$14+2=\mathbf{16}$, $15+2=\mathbf{17}$ and $19+2=\mathbf{21}$: **sixteen evenings to a verdict**
from the present tree; **seventeen** with M-5.3's ladder; **twenty-one** with the cost-law branch
S-66 → M-5.2. The "eighteen" and "twenty-two" this table printed before repair were arithmetic
errors against the very line they READ — $+3$ and $+3$ on a plan line that reads $15$ and $19$ —
and no extra evening was named anywhere as their source, which is P-1 with the producer sitting in
the citation itself. Total GPU-s over the seven sittings that make up the
**first five evenings only** (evenings 6, 7 and 8 are priced in the table above and are not in
this sum): $0+10.64+17.63+16.93+5.88+17.94+0 = \mathbf{\le 69.02}$, the weakest input's
class — `[FITTED, extrapolated at $n=16384$ and $n=32768$]`, from evening 2's two extrapolated
throughput rows (05.5) — and not DERIVED, each
sitting under $20$. The $\le46.21$ this table carried before repair summed a merged evening 3
that the cap forbids and is withdrawn.

**Hypotheses.** The plan's 14 evenings were counted at HEAD `207e7b9` on the certified stack
and are inherited unchanged (D-3: the count comes from the DAG); evenings 3, 3b and 3c are
three sittings of one calendar evening or three calendar evenings, at the author's choice —
the cap binds the **invocation set**, not the calendar, and none of the three shares an
invocation with another; the author executes the installer on evening 1 (no writer runs one).

**Evidence.** Prices DERIVED in the verses named; the plan's path READ as pinned.

**Mechanism.** D-3 (iteration count from the DAG's critical path); M-8; D-6 (every evening
has a command and a number, so none is a repair written and never started).

**Kill.** Any evening's verification number missing from that evening's report, or any
evening's summed GPU-s above $20$ at the measured (not the floor) prices. Frozen: $20.0$;
presence of every number. Instrument: the evening report and the journals it cites. Price:
$0$. Planted negative: evening 6, which the rule must place outside the five — it does.

**If killed** — *an evening overruns*: it is split by 05.1's rule and the path lengthens by
one; hypotheses: the split is decidable on the sub-run; kill: it is not — the evening is
moved after the Lean gate and the path lengthens by one. *A verification number is
missing*: the evening is not counted as done, the next evening does not start (D-4), and
the report says so; there is no replacement for a missing number.

**Terminal.** The canon licenses "sixteen evenings to a verdict from 2026-09-05, of which
the first five are seven sittings costing under $70$ GPU-s in total and none above $20$, and
of which this book's own priced programme through evening 7 is $610.55$ GPU-s, $641.81$ if
evening 8 is triggered" as a
schedule of class `[FITTED, extrapolated at $n=16384$ and $n=32768$]` — evening 2's two
extrapolated rows are the weakest input and set the class of the sum — and withdraws it the moment 05.5's within-stack row reads a cell outside
$2.2\times$ its own refitted law or its across-stack row reads a ratio outside the
`[ASSUMED]` $2.0\times$.

---

### 05.15 — Scheduling: every other census row, the book that closes it, and where it sits on the path

**Statement.** Each row below is closed by a verse in the named book; this book fixes only
its place in the schedule, its prerequisite from this book, and its price class.

| row | what is broken (one line) | closing book | prerequisite here | earliest evening | price class |
|---|---|---|---|---|---|
| B4 | the label class: one scalar at one position (D-1); Q6 F4 with no state axis | 04, 01 | none (a specification) | any | $0$ GPU-s |
| B5 | no CEQ arm trained; BED-S has no cell, sd or $t^\star$ | 04, 06 | 05.3, 05.5 (S-20 needs both) | $2+4$ (after the Lean gate) | $\approx34$ s per pair **DERIVED** |
| B6 | the calibration column: 9 of 9 adverse, $p=0.0352$ | 06 | none | any | $0$ |
| B7 | D-APPROX: "cannot represent" needs a bound before "cannot fit" | 01 | none | any | $0$ |
| B8 | obstruction theorems vacuous at the geometry ($384<544$; $1.30$ at $L=1$) | 01 | none | any | $0$ |
| B9 | the Cheeger sentence on a non-reversible $P$ | 01 | none | any | $0$ |
| B10 | Bellman optimality in multichain constrained MDPs | 01 | none | any | $0$ |
| B11 | hop$_k\to$committor reduction unwritten; two skylines without a cell | 01, 04 | 05.3, 05.5 for the cells | $2+4$ | S-35's $\approx64.8$ s `[ASSUMED]` |
| B12 | the corner descent: $\beta$ pins, $\hat\gamma$ PINNED predicted; 3 of 8 seeds diverging, $\hat a_{\max}$ to $285.07$ | 02 | 05.3, 05.6 (S-52 runs the capped seeds through the flagged script) | 3b | S-52's $\approx9.65$ s `[FITTED]` |
| B13 | determinism: `solve_triangular` bitwise but undocumented; `cumsum` raises; training inherits the backward hole | 03 | 05.5 evening 2 (the fourth quantity, N-01, is added to the same run) | 2 | $\approx1$ GPU-s |
| B14 | timers unsynchronised, arms not interleaved, $\rho=+0.7029$ | 03 (the cost law), 05.7 (the instrument) | 05.7 | 3 | $17.63$ s here; S-66's $206$–$537$ s band in book 03 |
| B15 | the kernels do not exist; dense not runnable at $n=2048$, $s=4096$ ($\approx137$ GB) | 03 | 05.5 evening 6 (the memory law re-solved) | 6 | `NOT MEASURED — needs the chunked kernel` |
| B20 | W2 `arm_phase` NAMED not FOUND; route back one `make_arm` branch, $41$–$130$ GPU-s | 04 | 05.3 | $2+4$ | $41$–$130$ GPU-s DERIVED |
| B21 | the read is occupied (ChaCAL); K-E1 predicts a column sink reproduces the row condition | 01, 04 | 05.3, 05.5 | $2+4$ | S-31/S-64 $\approx52$ s, escalation $\approx44$ s |
| B22 | the bibliography verified at the identifier, not the equation; ChaCAL's diagonal convention on one fetch | 01, 04 | none | any | $0$ |
| B23 | the magnitude interval closed vs half-open; D-R3 two inverse registrations | 04 | none | any | $0$ |
| B25 | "faster to train" never defined or measured; only per-op floors and withdrawn ratios | 02, 03, 04 | 05.5, 05.7 (T1 needs synchronised timers on a certified stack) | 6 | S-66's band; T2 inside the arena pair |
| **B26** | no architecture document: interfaces, shapes, forward pass, solve, boundary rows, heads, training loop and budgets scattered over `ceq/arm_smprime.py`, `ceq/lm.py`, `ceq/hf/modeling_ceq.py`, `ceq/hf/train.py`, `MATHEMATICS.md` | **08** | **none** (a specification; the book is written, not run) | **any** | $\mathbf{0}$ GPU-s |
| **B27** | the two uses the author names — prediction-market trades and chess — have no bed, pipeline, label, floor or matched baseline; the chess witness is behind three UNPINNED Kaggle sources and was never run | **09**, 04 | **none for the $0$ GPU-s cells** (09.10's census, 09.13's first chess cell and 09.32's market census run on CPU under the python-chess generator route of 09.2, at $0$ Kaggle attaches — 05.11 row 3); **05.3 and 05.5 for any GPU cell** (09.14) | **any** for the $0$ GPU-s cells; $2+4$ for 09.14 | $\mathbf{0}$ GPU-s for 09.10, 09.13, 09.32; 09.14's price as book 09 derives it |

**B26 and B27 were missing from this book entirely, and they are the author's own two rows.** RUN
2026-09-06 `grep -c "B26\|B27" docs/canon/05_REPAIRS.md` on the pre-repair file read
$\mathbf{0}$: neither string appeared anywhere in the book, so the census rows carrying
`CHARTER.md` §0's two deciding uses — prediction-market trades and chess — had no prerequisite,
no earliest evening and no price class in the one book that prices and schedules, and the "Census
rows closed" table stopped at B25. That is D-6 (a repair scheduled nowhere is a repair never
started) and L-VERSE (a census row with no verse). Both rows are added above and to the closed
table below. The scheduling consequence is that **B27's first three cells are unblocked today**:
they need no certified stack, because they run no GPU cell.

**B5's class is the source's own, and it is DERIVED, not `[FITTED + RUN]`.** The $34$ s carried
pair is `READ docs/PLAN.md:974 @ 99777ab`, "one bed-cell pair, 8 seeds each | $29.6$ s in one
invocation; **$34$ s** carried (two invocations, the conservative reading the record used) |
**DERIVED**". The `[FITTED + RUN]` tag this row carried before repair belongs to a **different
unit** at a **different line**: `READ docs/PLAN.md:972 @ 99777ab`, "one 150-step shape cell on
the softmax corner | $1.681$ s | `[FITTED + RUN]`". Attaching the stronger of two classes to
the weaker of two units is P-2 (a number's home stated as somewhere it does not live) and P-1
(a class with no producer for the number it is attached to): $34$ s is an arithmetic doubling
of a fitted cell price plus two invocation fixed costs, and nothing in it was RUN.

**Hypotheses.** "Earliest evening" counts from this book's evening 1; "$2+4$" means after
evening 2 and the plan's four-evening Lean gate (L-LEAN: no training before the identity
theorems are green). Every price is the plan's own, READ at `99777ab`, and is a floor.

**Evidence.** `READ docs/canon/CHARTER.md:225-253` (the census B1–B27, **live, no pin**: RUN
`git ls-tree -r 99777ab --name-only | grep -ci canon` reads $0$, so no `docs/canon/` line is
citable at `99777ab`; the pre-repair citation `:206-232 @ 99777ab` named the laws table plus
B1–B6 at a commit that holds no canon at all, and is struck);
`READ docs/PLAN.md:965-999 @ 99777ab` (prices); `READ docs/PLAN.md:876-889 @ 99777ab` (the path).

**Mechanism.** D-1 dependency (each row names its prerequisite in this book, so parallel
dispatch is only on rows with none); D-4 (rows behind the Lean gate are staged, not
started); L-LEAN.

**Kill.** A row started before its prerequisite verse's verification number exists in an
evening report. Frozen: the prerequisite column. Instrument: the evening reports. Price:
$0$. Planted negative: B5 on the present tree — S-20 cannot run today (no certified stack),
and the rule refuses it, which is correct.

**If killed** (a row is started early). Its numbers are quoted as `NOT MEASURED — box
unrecorded` by 05.4 and the row is re-run after its prerequisite; hypotheses: the re-run's
price is the row's; kill: the early run's numbers differ from the re-run's beyond the
bed's realised sd — the early run is superseded with an L-G2 marker.

**Terminal.** The canon licenses the schedule as a dependency table and withdraws any
result whose row ran ahead of its prerequisite.

---

## Census rows closed

| row | verse |
|---|---|
| B1 | 05.2, 05.3, 05.4 |
| B2 | 05.5 |
| B3 | 05.9 (policy jointly with 04) |
| B16 | 05.6, 05.7, 05.8 |
| B17 | 05.10 |
| B18 | 05.11 |
| B19 | 05.12 |
| B24 | 05.13 |
| B14 | 05.7 (the instrument; the cost law is book 03) |
| B4–B15, B20–B23, B25 | 05.15 (scheduling; closed in the book named per row) |
| B26 | 05.15 (scheduling; closed in book 08) |
| B27 | 05.15 (scheduling; closed in book 09, with 05.11 row 3 for the attach question) |

## Kills, cheapest first

| verse | kill number (frozen) | price | replacement verse or link |
|---|---|---|---|
| 05.13 | census lines ≠ `11 / 0 / 12` after the birth commit | $0$ | delete or move; row in `CORRECTIONS.md`; count re-frozen only by the author |
| 05.11 | a default whose reversal changes a number rather than a reading (all five right-hand cells must read $0$ GPU-s and falsify nothing) | $0$ | the default is withdrawn and re-filed as an OPEN question with no default |
| 05.4 (a) | a post-evening-1 `results/**/*.jsonl` with no flat `"torch"` key | $0$ | re-run with the key; supersede with L-G2 marker |
| 05.4 (b) | a post-evening-1 `.jsonl` whose flat `"torch"` ≠ `2.5.1+cu121` | $0$ | **OPEN** — no in-scope plant exists (RUN: one distinct value in `results/**/*.jsonl`); replacement is the per-evening interpreter census, which fires on the present box |
| 05.8 (A) | the journal census over `results/v17k_r4_retake.jsonl` reads $>0$ violations | $0$ | localise by `kind`/`seed`; else withdraw every $\hat\lambda$ sentence |
| 05.8 (B) | traced cell's `lambda_hat_live` off the finite mean by $>10^{-12}$, or `frac_gate_annihilated` inexact against $n\,\lvert{\rm live}\rvert$ ($\lvert{\rm live}\rvert=T^\star=2$ at the defaults, **not** $ns$) | $5.88$ s | `NOT MEASURED — needs the per-position trace`; no further link |
| 05.8 (C) | any `nan` or `+inf` in the traced `lg` (field-scoped: RUN today, both $\hat\lambda$ fields of `results/v17k_r4_retake.jsonl` carry $0$ of either, against $44$ `Infinity` and $2$ `NaN` in three neighbouring columns) | $0$ inside (B)'s cell | the cell's $\hat f_{\rm ann}$ is withdrawn, not read |
| 05.9 | `render_struck.py` ≠ `12 entries`; opens-an-absent-`.md` census $>0$ (**OPEN**, no population); collection $>0$ | $0$ ($14$–$16$ s CPU) | restore from the pin; (b′) restore-or-retire the 9 importers; re-file the guarded claim |
| 05.12 | smoke exit ≠ `0`; parity ≠ `True`; the recount names other than the **seven tensors** of 05.12's census or totals other than $2d+5$ per layer; unfounded bracket | $0$ (CPU) | tree-B card; tensor-by-tensor recount; bracket emptied |
| 05.2 | collection errors ≠ 9 after the pin, ≠ 0 after 05.9 | $0$ ($14$–$16$ s CPU) | four-class residual classifier; new `CORRECTIONS.md` verse |
| 05.3 | step-2 line ≠ `2.5.1+cu121 12.1 True NVIDIA GeForce RTX 4060 Laptop GPU` | $0$ | venv → `cu118` with a fresh certificate → Terminal (history at `ab5b485`) |
| 05.15 | a row started before its prerequisite's number exists | $0$ | re-run; supersede |
| 05.1 | a first-five evening above $20$ GPU-s | $0$ | split; push to evening 6+ |
| 05.14 | a missing verification number; an evening above $20$ measured | $0$ | split; the next evening waits (D-4) |
| 05.5 (evening 2) | hop/forward not bitwise; gradient not raising under the flag; zero-step $<30/30$ — measured into `results/k_cert_local_e2.json`, never over `results/k_cert_local.json`, which the script would **overwrite** (`write_text` at `:927`, no `json.load` of the out path) | $\le10.64$ s | `warn_only` with printed $\max|\Delta|$; ladder shrinks |
| 05.6 | identical-seed pair not `torch.equal`; header fields missing; `--s 32` writes `64`; **the `--s 32` run's bind rows ≠ $\{8,32\}$**; **the `--s 32` run's `secs` $>1.884$ s (the assumed ceiling)** | $11.05$ `[FITTED]` $+5.88$ `[ASSUMED]` | re-grep the site table; **the identity-bind residual route at $0$ extra GPU-s** ($10^{-14}$ bar, plant = the eight mutation residuals $0.4845\ldots1.9148$ in `results/v15_r1.jsonl`); then fixed geometry stated everywhere |
| 05.7 | `secs_sync < secs_unsync` on any cell; $\bar g < s_{\rm unsync}/\sqrt{8}$ with both columns printed; interleaved permutation $p\le0.05$ | $17.63$ s (evening 3) $+\,17.63$ s for the blocked-order plant (evening 7) $+\,31.26$ s if $N\to16$ (evening 8) | re-bracket; retire `secs_sync`; same-invocation ratios only |
| 05.10 | a Gate-0 row with a post-run slot and no `file:line` producer in 05.10's own producer table, **or one whose named line emits something other than the slot's value** (plant: `ceq/compat.py:50-52`, filled and refused); smoke exit ≠ `0` | $17.94$ s | CPU smoke; gate reverts to open-rulings $=0$ and stays RED |
| 05.5 (evening 6) | bar $\delta/\mathrm{tol}\ge50\%$; $R^2<0.99$; one 150-step cell outside $2.2\times$ its own refitted law (within-stack, `PLAN:463`'s units); across-stack ratio outside the `[ASSUMED]` $2.0\times$; memory constants off by $>5\%$ | $523.9$ s | re-tag every `[FITTED + RUN]` price as a floor; quote measured points only; no price crosses the stack boundary (V-22) |

---

## Attacks answered

MARS's findings on this book, repair round 1, batch 1 of 5. Every finding appears **verbatim**
— verse, flaw, mechanism, number, `replacement_survives`, required repair, severity — beside
the repair applied or the sentence now in force. Nothing here is softened; where the repair
made the finding read *worse* than filed (05.9's census, whose true population turned out to
be $0$ rather than the $5$ the finding measured, and 05.4's plant, which is not merely
out-of-scope but has no in-scope substitute anywhere in `results/`), the worse reading is the
one recorded.

---

**Finding 1 · verse 05.1 · severity `strike` · `replacement_survives: false`**

> **flaw** — The 20 GPU-s evening cap — this verse's only frozen kill threshold — is sourced
> to a line that does not exist at the pin and says nothing of the kind in the live file.
> **mechanism** — P-1, P-2, V-17.
> **number** — `git ls-tree -r 99777ab --name-only | grep -i canon` returns 0 lines
> (docs/canon/ is untracked, git status `?? docs/canon/`); CHARTER.md:248 live is the B22
> census row; `grep -n "20 GPU" docs/canon/CHARTER.md` returns 0 hits; the book-05 scope row
> is CHARTER.md:269 and contains no cap.
> **required repair** — Either freeze the 20 GPU-s cap as an [ASSUMED] budget declared in
> this verse with its reason, or delete the cap and re-order 05.14 on decisiveness alone. The
> If-killed ("split into a <=20 GPU-s sub-run") is defined by the same constant and dies with
> it.

**Repair applied — 05.1, and the Price basis of the preface.** The first branch is taken: the
cap is **declared here** as an `[ASSUMED]` budget of this book with its reason printed — $20.0$
is the smallest round number that admits every priced repair singly ($17.63$ s, the largest)
with one invocation's fixed cost ($4.0$ s, `READ docs/PLAN.md:978 @ 99777ab`) of headroom, and
refuses every pair of them. Both of the finding's RUN facts are reproduced inside the verse:
`git ls-tree -r 99777ab --name-only | grep -ci canon` reads $0$, `grep -c "20 GPU"
docs/canon/CHARTER.md` reads $0$. Every `docs/canon/` citation in this book is re-stated as a
**live** citation with no `@ 99777ab` pin, and the book-05 scope row is cited at
`docs/canon/CHARTER.md:269` (live), which names "the price in evenings and GPU-s" and no
number. The If-killed keeps its split rule, now defined by a constant this verse owns.

---

**Finding 2 · verse 05.1 · severity `strike` · `replacement_survives: true`**

> **flaw** — The ordering rule's kill fires on the schedule the book itself ships: evening 3
> exceeds the frozen 20 GPU-s.
> **mechanism** — M-8, D-6.
> **number** — 05.6's kill requires `--device cuda --seeds 0 --arms softmax` priced
> 2x(1.524+4.0)+(1.524+4.0)=16.57 s; 05.7's requires 8 interleaved cells over arms {arm_pl,
> softmax} and seeds 0-3 priced 4x1.884+4x1.524+4.0=17.63 s. Different `--arms` and `--seeds`,
> so no shared invocation: evening 3 = 34.20 GPU-s > 20.0.
> **required repair** — Apply 05.1's own split rule to evening 3 in 05.14: 05.7's 8 cells at
> evening 3 (17.63 s), 05.6's pair at evening 3b or evening 6 (16.57 s); delete the phrase
> "05.6's pair is inside the same invocation set".

**Repair applied — 05.1 and 05.14.** The split is executed in 05.14's table, which now carries
**evening 3** (05.7, $17.63$ s), **evening 3b** (05.6, $16.93$ s) and **evening 3c** (05.8,
$5.88$ s) as three sittings with three verification rows. The phrase "05.6's pair is inside the
same invocation set" is **deleted** from 05.14 and its impossibility stated in its place. Two
numbers moved against the finding and are recorded as worse, not better: 05.6's price rises
from $16.57$ to $\mathbf{16.93}$ s, because Finding 10's repair forces the `--s 32` run onto a
**gated** arm (`arm_pl`, $1.884$ s) so the new bind-row clause is reachable at all — so the
merged evening 3 would have been $\mathbf{34.56}$ GPU-s, not $34.20$; and 05.8's cell cannot
join 3b either ($16.93+5.88=22.81>20.0$), which the finding did not reach. The first five
evenings' total rises from $\le46.21$ to $\le\mathbf{69.02}$ GPU-s over seven sittings, and
05.14's Terminal is re-worded to license that.

---

**Finding 3 · verse 05.2 · severity `strike` · `replacement_survives: false`**

> **flaw** — The RUN counts that carry the Statement and the Kill's planted negative do not
> reproduce on the tree today.
> **mechanism** — P-1, P-2.
> **number** — RUN `python -m pytest -q --collect-only -p no:cacheprovider` at the root,
> 2026-09-06: `2580 tests collected, 22 errors in 14.08s`, `Interrupted: 22 errors during
> collection`. The verse states 2971 tests, 26 errors, 17.69 s.
> **required repair** — Re-run and re-print the collection numbers with their date, and
> re-derive the environment/residual split from that run. The If-killed classifier is keyed to
> the 26/13/13 arithmetic and must be re-derived with it.

**Repair applied — 05.2 Statement, Evidence, Kill, If-killed.** The command was re-run
2026-09-06 and reads `2580 tests collected, 22 errors in 15.35s` — a second execution the same
day, confirming the finding's `22` and `2580` exactly and its wall-clock to a range; the
Evidence now quotes **$14.08$–$15.35$ s** as a range and never to the second. The triple
$2971$ / $26$ / $17.69$ s is **struck from the book** by name, under the number rule (P-1,
P-2): no journal holds it and it does not reproduce. The split is re-derived from the new run:
$13$ environment $+\ 9$ residual $=22$, with a per-exception census printed
($7$ `AttributeError` torchvision, $1$ `RuntimeError` nms, $8$ `ModuleNotFoundError`
`PreTrainedModel`, $5$ `ValueError` numpy dtype, $9$ test-module import breaks) and the
line-versus-file arithmetic stated ($8\times2+5\times1=21$ error lines over 13 environment
files, $9$ over 9). 05.3's step-5 verification and 05.14's evening-1 verification are
re-frozen from `13 errors` to `9 errors` with them.

---

**Finding 4 · verse 05.2 · severity `strike` · `replacement_survives: false`**

> **flaw** — The residual after the environment pin is 9, not 13, and none of the 9 is
> prose-coupled — a fourth error class exists that the verse's three-class residual classifier
> cannot name.
> **mechanism** — P-1, V-16.
> **number** — RUN classification of the 22: 13 environment (the 8 torchvision/transformers
> files and the 5 pandas/numpy files listed in the verse verify exactly). The other 9 raise
> `ImportError: cannot import name 'test_v20_r15_freeze_manifest' from 'tests.saturn'` (3),
> `ModuleNotFoundError: No module named 'tests.jupiter.test_v20_r15_it20_citation_freeze'` (2),
> `...it27_star_lands_and_overturns` (2), `...it14_theory_table` (1),
> `tests.saturn.test_v20_r15_it27_wing_arm_citation` (1). Zero FileNotFoundError, zero
> AssertionError, zero `.md` names.
> **required repair** — Re-freeze the Kill at `total errors = 9 until 05.9 lands, then 0`. Add
> a fourth residual class to the If-killed classifier: a test importing a test module deleted
> at c71527a (grep `from tests\.` / `import tests\.`). As frozen, the kill fires on a correctly
> executed pin.

**Repair applied — 05.2 Statement, Kill, If-killed; 05.9 throughout.** The finding's
classification reproduces exactly, including the $3/2/2/1/1$ split of the 9. The Kill is
re-frozen: **environment errors $=0$, total errors $=9$ until 05.9 lands, then $0$**, and it is
now **two-sided** — a count of $8$ fires it too, so a repair that deletes a file instead of
fixing it does not pass. The **fourth class is added** to the If-killed classifier as a fourth
table row, selector
`ModuleNotFoundError: No module named 'tests.` or `ImportError: cannot import name 'test_`,
confirmed by `grep -nE "^\s*(from|import) tests\." <file>` naming a path absent from disk, and
the verse states that the four-class rule assigns all 22 of today's errors and leaves none
over. Each of the 9 is tabled by file with its raised exception and the importing line number
(`:101`, `:48`, `:47`, `:27`, `:36`, `:34`, `:31`, `:30`, `:27`). 05.9's plan gains step
**(b′)**, the repair these 9 actually need — restore the five deleted test modules from the pin
(`git cat-file -e 99777ab:<path>` returns clean for all five, RUN) or retire the importer with
its guarded claim re-filed in `CORRECTIONS.md` — because a move to `attic/tests/` does not fix
an import.

---

**Finding 5 · verse 05.2 · severity `strike` · `replacement_survives: false`**

> **flaw** — The Terminal's parenthetical is falsified: a post-drift results/ file exists that
> does print its stack, and the newest file is not the Kaggle log.
> **mechanism** — P-1, V-22.
> **number** — RUN `find results -type f -newermt "2026-09-04 00:40" ! -newermt "2026-09-04
> 00:42" | wc -l` reads **18**, not 1. `results/v20_r15_it8_armpl_b.jsonl` (mtime 2026-09-04
> 00:41) has header `"torch": "2.5.1+cu121"`, `"deterministic_warn_only": true`,
> `"cublas_workspace_config": ":4096:8"`.
> **required repair** — Replace "(none exists in results/ ...)" with the 18-file list and the
> observation that those rows print a flat `torch` key, not a `box` record — which is the
> defect 05.4 must actually catch.

**Repair applied — 05.2 Terminal.** The parenthetical is deleted and named false in the verse's
own words. All **18** paths are listed by name; the header fields of
`results/v20_r15_it8_armpl_b.jsonl` are quoted verbatim as the finding gives them; and the
observation is stated that what those files print is a **flat `torch` key with no `box`
record** (RUN over `results/v15_r1.jsonl`'s 26-line key union: `torch`, `torch_version`
present, `box` absent), which is handed to 05.4 as the predicate to freeze. One fact is added
beyond the finding, in the direction that weakens the book's own withdrawal: all 18 share the
minute 2026-09-04 00:41, which is a **bulk filesystem touch and not 18 write times**, so the
mtime window is evidence the files were handled after the drift and never that they were
measured after it — the census is a trigger for 05.4's stack census, not a withdrawal of the
18 by itself.

---

**Finding 6 · verse 05.4 · severity `strike` · `replacement_survives: false`**

> **flaw** — The Kill and its instrument decide different predicates; every existing journal
> row passes the instrument and fails the kill.
> **mechanism** — V-16, M-2.
> **number** — Kill: "a row with no `box.torch` field, or with `box.torch` != `2.5.1+cu121`".
> Instrument: `grep -L '"torch": "2.5.1+cu121"'`. RUN over `results/v20_r15_it8_armpl_b.jsonl`:
> the header carries a flat top-level `"torch": "2.5.1+cu121"` and no `box` object at all — the
> grep finds the string, the stated kill fires on the missing field. RUN over
> `results/v15_r1.jsonl` key set: `torch`, `torch_version` present, `box` absent.
> **required repair** — Freeze one predicate. Either the instrument becomes `python -c` reading
> `row["box"]["torch"]` and raising on absence, or the kill is re-written on the flat key that
> producers actually emit. As written the rule cannot be applied to any row in results/.

**Repair applied — 05.4 Statement, Hypotheses, Evidence, Kill.** The second branch is taken and
**one** predicate is frozen, on the **flat `torch` key**, because that is what producers emit
and a `box`-keyed rule is satisfiable by no row that has ever existed (V-10). The Statement now
opens by naming the two-predicate defect in full and prints the census that decides it: RUN over
`results/`, $46$ occurrences of `"torch": "2.5.1+cu121"` and $130$ of `"torch_version":
"2.5.1+cu121"` in `*.jsonl`, $2$ of `"torch"` in `*.json`, and **38 `*.jsonl` files carrying no
`torch` key of any kind**. The frozen rule is quoted as a block: a `results/**/*.jsonl` written
after evening 1 is RUN evidence iff its header line carries a top-level `"torch"` whose value is
exactly `2.5.1+cu121`. The `box` record is retained as the richer form required of producers
that already write it — `results/k_cert_local.json["box"]` holds 13 fields, RUN — and is
explicitly **not** the kill's predicate.

---

**Finding 7 · verse 05.4 · severity `strike` · `replacement_survives: false`**

> **flaw** — The planted negative is refused by the rule's own scope rather than detected by
> it — the verse says so in its own words.
> **mechanism** — V-15, V-10.
> **number** — Kill scope: `results/**/*.jsonl`. Plant:
> `results/kaggle_v17k_output/ceq-v17-k.log` (RUN :11 reads `torch 2.10.0+cu128`), which the
> verse concedes "is not a `.jsonl` row and carries no `box`". A plant outside the kill's scope
> cannot make it fire.
> **required repair** — Supply an in-scope plant: one `results/**/*.jsonl` row (or a copy)
> whose `box.torch` reads a non-frozen string, and show the grep/reader selecting it. Without
> it the rule is a condemning rule with no planted negative.

**Repair applied — 05.4 Kill, and one clause left OPEN.** The kill is split into two clauses
with two separate plants, and the outcome is **worse** than the finding's required repair
allows, so it is recorded as such rather than papered over.
**Clause (a), key absent** — plant supplied, in scope, firing today: the **38**
`results/*.jsonl` files that carry no `torch` key of any kind, every one of them named in the
verse. **Clause (b), value wrong** — the required in-scope plant **does not exist and cannot be
manufactured without writing a file**, which the standing rule forbids: RUN,
`results/**/*.jsonl` holds exactly **one** distinct stack string ($46$ occurrences of
`2.5.1+cu121`), and the only non-frozen string anywhere under `results/` is the Kaggle log's
`torch   2.10.0+cu128` at `:11`, which is a `.log` and stays outside clause (b)'s scope. The
verse now says this in the finding's own terms — refusal by scope is not detection by the rule,
and calling the log a plant was the V-15 defect — tags the owed plant
`NOT MEASURED — needs one post-evening-1 .jsonl written on a deliberately non-frozen stack`,
and records **clause (b) OPEN**.
**Terminal now in force for clause (b):** *"while it is open, the canon licenses the absence
census (clause (a)) and the interpreter census (the replacement) as the drift detector, and
withdraws the sentence 'a wrong stack string in a journal is caught by this rule' — nothing in
`results/` has ever carried one, so the rule has never been shown to catch it."*
A **genuinely different replacement** is derived rather than a rename: it changes the object
(the **interpreter**, not the journal), the instrument (`python -c "import torch,numpy;
print(torch.__version__, numpy.__version__)"` at the head of every evening, pasted into the
evening report beside that evening's journal filenames) and the kill number (the frozen string
`2.5.1+cu121 1.26.4`). Its plant fires on the present box today: RUN, the command prints
`2.14.0+cpu 2.4.6`.

---

**Finding 8 · verse 05.5 · severity `strike` · `replacement_survives: false`**

> **flaw** — The 2.2x throughput-drift bar — a frozen acceptance number — is cited to the wrong
> line and, at the right line, is a different comparand imported across a boundary it was never
> calibrated for.
> **mechanism** — V-17, M-3.
> **number** — `grep -n "2\.2" docs/PLAN.md`: the bar is at :463, not :464 (:464 is S-20's
> price line, `~17.4 s for 8 seeds`). PLAN:463 reads "the cell above $2.2\times$ **the law**" —
> one measured 150-step cell against its own fitted law on one stack, including the $4.0$ s
> invocation fixed cost. 05.5 uses it as a refitted-law-versus-ab5b485-law drift bar across a
> stack boundary.
> **required repair** — Cite PLAN:463 and either (a) restate the acceptance as "one 150-step
> cell at n=2048 within 2.2x of the newly refitted law", the units the bar was calibrated in,
> or (b) declare a new across-stack drift bar with its own [ASSUMED] tag and its own planted
> negative.

**Repair applied — 05.5 acceptance table and Hypotheses; both branches taken.** The single
throughput-drift row is **split into two rows**. Row (a), *within-stack band*: one 150-step cell
at $n=2048$ inside $2.2\times$ the newly refitted law of the same stack, cited
`READ docs/PLAN.md:463 @ 99777ab` with the quoted phrase "the cell above $2.2\times$ **the
law**"; the mis-citation `:464` is named in the verse as S-20's price line
("$\approx17.4$ s for 8 seeds") and corrected. Row (b), *across-stack drift*: a **new**
`[ASSUMED]` bar of $\mathbf{2.0\times}$ on $s/\mathrm{step}$ at $n=2048$ between the refitted
law and the `ab5b485` law **of the same arm**, declared in the verse with its reason — it must
clear the record's own measured same-stack instability ($14.390$ vs $17.839$ s, spread
$1.265\times$ over $14.101\ldots17.839$ s, `READ V17_R4_RETAKE_PRICE.md:143-150 @ 99777ab`) and
the $\pm12$ per cent clock non-stationarity (`READ docs/PLAN.md:967 @ 99777ab`), and $2.0$ is
the smallest round multiple above both. Its **planted negative is in scope and costs $0$
GPU-s**: feed the comparison the wrong arm's law — `arm_smprime` $0.093066$ s against `softmax`
$0.010162$ s at $n=2048$ (`READ COSTS.md:224,230 @ 99777ab`), ratio $9.16$ — and the bar must
fire, which is exactly the M-8 mis-keying an across-arm drift census commits silently. Its
**true negative** is stated too, so the bar is not vacuous: the $14.390$/$17.839$ pair, ratio
$1.24$, on which it must not fire. The If-killed now carries a separate branch per row, and the
across-stack branch withdraws every `COSTS.md` §1 price as history under V-22 rather than
re-tagging it.

---

**Finding 9 · verse 05.5 · severity `strike` · `replacement_survives: true`**

> **flaw** — The CUBLAS export hypothesis cites a line that contains no such statement; the
> string does not occur in the cited file at all.
> **mechanism** — P-2, P-5.
> **number** — `grep -n CUBLAS docs/PLAN.md` returns **0 hits**. `docs/PLAN.md:460` is S-20's
> build line (`torch.equal(O(0), PV)` against softmaxAttn, ChaCAL-diag at gamma=0.9). The
> claim's actual home is `V17_R4_RETAKE_PRICE.md:178-181`, quoted at
> `docs/sources/sections/sec_cost.md:122 @ 99777ab`.
> **required repair** — Re-pin to `READ docs/sources/sections/sec_cost.md:122 @ 99777ab`, which
> carries the READ of V17_R4_RETAKE_PRICE.md:178-181 verbatim.

**Repair applied — 05.5 Hypotheses; the required repair is applied with one pin corrected.**
The claim is re-pinned to its **actual home**, `READ V17_R4_RETAKE_PRICE.md:178-181 @ 99777ab`,
which RUN `git show 99777ab:V17_R4_RETAKE_PRICE.md | sed -n '178,181p'` returns verbatim: "set
from inside Python after CUDA is initialised it does not take, and strict mode then fails the
**forward** as well". The finding's own target is **not usable at the pin it names** and the
verse says so: RUN `git show 99777ab:docs/sources/sections/sec_cost.md` returns
`fatal: path 'docs/sources/sections/sec_cost.md' exists on disk, but not in '99777ab'`, and RUN
`git log --oneline -- docs/sources/sections/sec_cost.md` reads a single commit, `c71527a`. The
live quotation is therefore cited as `READ docs/sources/sections/sec_cost.md:122 @ c71527a`
beside the primary pin, with the discrepancy printed rather than silently reconciled. The
mis-citation `docs/PLAN.md:460` is named in the verse: `grep -c CUBLAS docs/PLAN.md` reads $0$,
and `:460` cites `V17_R4_RETAKE_PRICE.md:178-181` rather than stating the claim.

---

**Finding 10 · verse 05.6 · severity `strike` · `replacement_survives: false`**

> **flaw** — The list of "every site that reads S, D, T_STAR" is incomplete, and the frozen kill
> cannot detect the omission — a repair executed as specified reproduces the exact mechanism the
> book invokes against itself.
> **mechanism** — M-1, P-6, D-3.
> **number** — RUN grep over scripts/v15_r1.py finds S also at :506
> (`rows.append(identity_bind(kind, S, device=device))`), :508 (`for s in (8, S)`), :588, :656,
> :659, :673, :684 — none in the verse's list (:165, :586-587, :686, :697, :699, :708, :810,
> :826-829, :907). With `--s 32`, the identity bind at :506/:508 still binds at S=64 while the
> run trains at 32; the header at :607 still writes s=32, so the kill passes.
> **required repair** — Re-run the grep and print the complete site list; add a kill clause that
> the bind row's `s` field equals the parsed `--s` (the journal already carries `s` per row —
> `results/v15_r1.jsonl` key set, RUN).

**Repair applied — 05.6 Statement, Evidence, Kill, If-killed.** The three greps are re-run and
the **complete** site table is printed in the verse, per constant, with the docstring lines that
read "ARM S-M'" rather than the constant listed and discarded (`:8, :10, :62, :70, :192, :299,
:319, :336, :400, :436, :446, :469, :487, :491, :499, :509, :564, :780, :800`). The verse names
its own omission: **ten** sites, `:506, :508, :588, :656, :659, :673, :684, :745, :802, :837` —
three more than the finding enumerates (`:745`, `:802`, `:837` are added by the re-run) — and
states in its own words that a repair executed to the short list reproduces P-6 and M-1 on the
book itself. **Kill clause (iv) is added**: the `--s 32` run's `t="bind"` rows must read
$\{8, 32\}$. It is reachable because the journal already carries `s` per bind row (RUN over
`results/v15_r1.jsonl`: two `t="bind"` rows, $s=8$ and $s=64$, residuals
$6.661338147750939\times10^{-16}$ and $7.549516567451064\times10^{-15}$, emitted by the
`for s in (8, S)` loop at `:508`). One consequence the finding did not reach is recorded and
priced: bind rows are emitted only for `GATED_ARMS = ("arm_pl", "arm_smprime")` (RUN, `:146`),
so clause (iv) run on `--arms softmax` would be **vacuous (V-11)** — the `--s 32` invocation is
therefore moved to `--arms arm_pl`, the price rises from $16.57$ to $\mathbf{16.93}$ s, and
Finding 2's evening arithmetic is re-derived with it. A **second planted negative** is added for
clause (iv) specifically: a repair executed to the short list, on which clauses (i)–(iii) all
pass while the bind rows read $\{8, 64\}$ — the negative that makes clause (iv) load-bearing.

---

**Finding 11 · verse 05.7 · severity `strike` · `replacement_survives: true`**

> **flaw** — Kill clause (ii) has no frozen precision, so it either never fires or fires on any
> nonzero gap.
> **mechanism** — M-2, V-11.
> **number** — Frozen as `mean(secs_sync - secs_unsync) = 0.000` over 8 cells. The gap is the
> wall-clock of the trailing `torch.cuda.synchronize()` call, which is never exactly 0.0 in
> float seconds; at 3-decimal rounding a sub-millisecond sync reads 0.000. The verse names
> neither the comparison nor the rounding.
> **required repair** — Freeze a tolerance in the timer's own units, e.g. "mean gap below the
> 8-cell paired sd of secs_unsync, printed", and print both columns per cell.

**Repair applied — 05.7 Kill clause (ii) and If-killed.** The tolerance is frozen in the
instrument's own units: clause (ii) fires iff
$\bar g = \mathrm{mean}_{i\le8}(\texttt{secs\_sync}_i-\texttt{secs\_unsync}_i) <
s_{\rm unsync}/\sqrt{8}$, the **standard error** of the un-synchronised clock over the same 8
cells — one standard error, not one standard deviation, because $\bar g$ is a mean of 8 and the
question is whether the design can resolve it from zero. **Both columns are printed per cell**,
all 8 rows, with $\bar g$, $s_{\rm unsync}$ and $s_{\rm unsync}/\sqrt{8}$ on the summary line
and **no rounding applied before the comparison** — the verse says so explicitly, since the
`0.000` rounding is the mechanism the finding names. The verse states the failure mode it
replaces in the finding's own terms (exact float equality never fires; printed-string equality
fires on any nonzero gap) and adds the hypothesis that makes the new clause non-vacuous:
$s_{\rm unsync}>0$ over the 8 cells, with a zero sample sd at $N=8$ tagged
`NOT MEASURED — needs a clock with nonzero spread`. The If-killed branch is re-keyed to
$\bar g$ against $s_{\rm unsync}/\sqrt{8}$, and it now retires the **`secs_sync`** column as the
redundant one (the verse previously retired `secs_unsync`, which is the wrong column to drop
when the finding's conclusion is that the un-synchronised clock is already honest).

---

**Finding 12 · verse 05.8 · severity `strike` · `replacement_survives: false`**

> **flaw** — The Kill cannot fire on either arm the schedule registers: the -inf path exists
> only in the arm_smprime branch, which evening 3 does not run.
> **mechanism** — V-11, V-3.
> **number** — `scripts/v15_r1.py:371` `lg = torch.log(m)` is inside `elif kind ==
> "arm_smprime"`. `:361` for `arm_pl` reads `lg = model.heads(x)[0][:, live]` — a log-gate head,
> no m. `:376-379` for softmax returns constants and exits. 05.7's evening-3 cells are `arm_pl`
> and `softmax`; RUN over `results/v15_r1.jsonl` header: `arms: ['arm_pl','softmax']`.
> **required repair** — Register an arm_smprime cell for the plant with its own price and
> evening, or restate the verse as a documentary rename with a NOT MEASURED kill and the
> arm_smprime cell named as the instrument owed.

**Repair applied — 05.8 Statement, Hypotheses; 05.1 and 05.14 schedule.** The first branch is
taken. The verse now opens by reproducing the finding's line reading in full — `:371`
`lg = torch.log(m)` inside `elif kind == "arm_smprime"` at `:362`; `:361` `arm_pl`'s log-gate
head with no $m$; `:376-379` `softmax` returning
`lambda_hat=0.0, lambda_hat_live=0.0, frac_gate_annihilated=0.0` and exiting before `lg` is
read — and states in its own words that on the schedule as it stood **the kill could not fire
on either registered arm (V-11, V-3)**. The missing cell is **registered**: one `arm_smprime`
cell, `--device cuda --seeds 0 --arms arm_smprime`, priced $1.884+4.0=\mathbf{5.88}$ GPU-s at
**evening 3c**, carried in 05.1's split paragraph, in 05.14's evening table and in the
cheapest-first kill table. The Hypotheses field is narrowed to say that `arm_pl` and `softmax`
are **out of this verse's scope**, so the verse no longer asserts anything about arms whose
code path cannot produce the reading.

---

**Finding 13 · verse 05.8 · severity `strike` · `replacement_survives: true`**

> **flaw** — The Kill's instrument does not exist: no journal in results/ carries the columns or
> the per-position vector it claims to re-read, and re-reading a row "through the column
> function" is new code.
> **mechanism** — P-1, V-16.
> **number** — RUN over `results/v15_r1.jsonl` (26 lines): the union of all row keys contains no
> `lambda_hat`, no `lambda_hat_live`, no `frac_gate_annihilated` and no per-position `log m`
> array. The verse's own If-killed concedes the vector may not be journalled ("if journalled,
> else the cell is re-run with --trace-every 1").
> **required repair** — Move the `--trace-every 1` re-run from the If-killed to the Kill, price
> it at one cell (1.884+4.0 = 5.88 s), and tag the $0$ GPU-s replay `NOT MEASURED — needs the
> per-position trace`.

**Repair applied — 05.8 Kill, split into (A) and (B); the required repair is applied and one
instrument is found that the finding did not reach.** The finding's RUN reproduces exactly for
`results/v15_r1.jsonl`, and the verse quotes it. The `--trace-every 1` re-run **is moved into
the Kill** as **clause (B)**, priced at one capped cell, $1.884+4.0=\mathbf{5.88}$ GPU-s,
evening 3c; the $10^{-12}$ arithmetic clause is the only thing it decides, and until that cell
runs clause (B) reads `NOT MEASURED — needs the per-position trace`. The $0$ GPU-s replay
against `results/chase_k1_replay_it0.txt`'s class is **struck** as an instrument that does not
exist. Beyond the finding: a **different journal does carry the three columns**, and it is named
as **clause (A)**, decidable at $0$ GPU-s today — `results/v17k_r4_retake.jsonl` holds **408**
rows with a `lambda_hat` field across all three arms (RUN), of which **131** (`arm_smprime`,
`frac_gate_annihilated`$>0$) read `"lambda_hat": -Infinity` with **$0$ violations**, and **277**
(`arm_pl` $136$, `softmax` $136$, `arm_smprime` $5$) read `frac_gate_annihilated`$=0$ with
`lambda_hat` bitwise equal to `lambda_hat_live`, **$0$ violations**. Both populations are
RUN-nonempty, which gives the clause a positive control **and** a true negative: the $277$
zero-annihilation rows condemn a broken finite mask (`fin = torch.isfinite(lg)` at `:380`) on
rows nothing was planted into. The If-killed's second branch now carries the RUN fact that
`:736-743`'s `t="trace"` row emits the three **scalars** and not the vector, so clause (B)'s
hypothesis is stated with its own refutation already in evidence, and the chain ends at the
Terminal rather than inventing a fourth link that would require code.

---

**Finding 14 · verse 05.8 · severity `strike` · `replacement_survives: true`**

> **flaw** — The Koopman-field attribution cites a markdown table separator.
> **mechanism** — P-2.
> **number** — `sed -n '1219p' docs/CEQ_SHAPE.md` reads `|---|---|---|---|---|`.
> **required repair** — Locate the Koopman-field sentence by grep and re-pin, or delete the
> clause; the substantive debt is already pinned at docs/CEQ_SHAPE.md:1280.

**Repair applied — 05.8 Evidence.** The sentence is located by `grep -n Koopman
docs/CEQ_SHAPE.md` and re-pinned to `READ docs/CEQ_SHAPE.md:1222 @ 99777ab`, whose row reads
"transfer-operator / Koopman spectral theory, attached to `lambda_hat_live` at `:384` and not
`lambda_hat` at `:383` — one bit"; the pin holds at the commit, RUN
`git show 99777ab:docs/CEQ_SHAPE.md | sed -n '1222p'`. The finding's own number is reproduced in
the verse — `sed -n '1219p'` reads `|---|---|---|---|---|` — so the mis-citation is recorded, not
quietly corrected. `:1280` is re-verified at the pin and stands.

---

**Finding 15 · verse 05.9 · severity `strike` · `replacement_survives: false`**

> **flaw** — The three census constants are fabricated: no quoting rule produces them.
> **mechanism** — P-1, P-2.
> **number** — RUN under the verse's own rule (a quoted string ending in `.md` with no `/`) over
> tests/**/*.py: 280 files (verse: 280, correct), **15** naming some `.md` (verse: 68), **5**
> naming an absent root `.md` (verse: 59), **8** distinct absent documents (verse: 68). Under
> the loosest quoted rule (any quoted string ending `.md`): 17 / 8 / 12. Files merely mentioning
> `.md` anywhere (comments included): 141.
> **required repair** — Re-run the census with the rule printed beside it and re-freeze the
> kill's third number. The move plan (b) applies to 5-8 files, not 59.

**Repair applied — 05.9 Statement, Hypotheses, Evidence, Mechanism, Kill, Terminal; and the
finding is made worse, not better.** The census is re-run 2026-09-06 **with its rule printed as
an executable pattern** — a quoted single token matching `["'][A-Za-z0-9_.-]+\.md["']`, no path
separator, no whitespace, absent if the repo-root path does not exist — and reads $280$ / $15$ /
$5$ / $8$, reproducing the finding exactly; the loosest rule reads $16$ / $7$ / $10$, and the
verse states that the two extra names are the glob **patterns** `*.md` and `DONE*.md`, which are
not documents, so excluding them returns $15$ / $5$ / $8$. `141` files mention `.md` anywhere.
The triple $68$ / $59$ / $68$ is **struck by name** under P-1. Each of the five files is tabled
with the names it carries and **how** it carries them, and that table produces a number the
finding did not reach: **not one of the five opens the document it names** — `test_bed_k.py` and
`test_schedule_rebuild.py` cite in a docstring and an assertion message, `test_merkle_journal.py`
writes `tmp_path / "c1.md"` fixtures it creates itself, `test_it18_timer_liveness…` points at a
harness config `.claude/ralph-loop.local.md`, `test_r9_table_cut.py` asserts a produced
filename. RUN over all 280 files, the count of lines that both name an absent `.md` and carry
`read_text`, `open(`, `.exists(` or `read_bytes` is **$\mathbf{0}$**. **Move plan (b) therefore
applies to $0$ files, not $5$–$8$ and not $59$**, and it is retained only as a standing rule for
files written later. The Kill's third number is re-frozen on the **opens**-an-absent-`.md`
census at $0$, and — because a rule frozen at $0$ that already reads $0$ is satisfied by
construction (V-10) — **that clause is recorded OPEN for want of a population to plant into**,
with clauses 1 (`render_struck.py` raises today, RUN) and 3 (collection reads $22$ today, RUN)
carrying the verse.
**Terminal now in force for clause 2 of 05.9's Kill:** *"B3 is closed not by a move but by a
re-measurement: the population the row describes is empty, and what remains of the row is the 9
deleted-test-module imports, which are a different defect with a different repair. Clause 2 of
the Kill stands OPEN for want of a population to plant into."*

---

**Finding 16 · verse 05.9 · severity `strike` · `replacement_survives: false`**

> **flaw** — Four of the thirteen files enumerated as erroring at collection do not exist on
> disk.
> **mechanism** — P-5, P-6.
> **number** — RUN `[ -e ]`: `tests/loop/test_attic_never_removes_the_last_must_fire.py` ABSENT;
> `tests/saturn/test_v20_r15_freeze_manifest.py` ABSENT;
> `tests/loop/test_no_struck_constant_ships.py` ABSENT (all deleted in c71527a).
> `ls tests/mercury` shows no `test_*.py` at all, so `tests/mercury/test_v20_r15_it26_...`
> cannot error — the it26 file is `tests/jupiter/test_v20_r15_it26_live_claim.py`.
> `test_v20_r15_it27_...` and `..._it34_` exist only as `__pycache__` artifacts.
> **required repair** — Replace the list with the 9 files the collection run actually names
> (F5's classification), each with its raised exception.

**Repair applied — 05.9 Statement; 05.2 Statement.** The 13-file list is **deleted** and its
four non-existent entries named in the verse: RUN `[ -e ]` reads ABSENT for
`tests/loop/test_attic_never_removes_the_last_must_fire.py`,
`tests/saturn/test_v20_r15_freeze_manifest.py` and `tests/loop/test_no_struck_constant_ships.py`
(all deleted at `c71527a`); RUN `ls tests/mercury` lists no `test_v20_r15_it26_*` file, the it26
file being `tests/jupiter/test_v20_r15_it26_live_claim.py`; `…_it27_` and `…_it34_` survive only
as `__pycache__` artefacts. One correction to the finding's own wording is recorded rather than
copied: `ls tests/mercury` **does** list `test_*.py` files — twelve of them, `test_r9_*` and
`test_v20_r15_it*` — none of which is an it26 file, so the conclusion holds and the premise is
restated accurately. The replacement is the **9-file table in 05.2**, each row carrying the
file, its raised exception verbatim and the importing line number; 05.9 points at it and adds
that **none of the 9 raises `FileNotFoundError` or `AssertionError` and none names a `.md`**, so
they are not prose-coupling at all. The verse's heading is changed from "the 13 erroring files"
to "the 9 erroring files", the Mechanism field gains **P-6 turned on the book itself**, and the
`git diff --cached --name-only --diff-filter=D` command is corrected to
`git show --name-only --diff-filter=D --pretty=format: c71527a` because the staged form reads
**0** today — the deletions were committed, not staged. The $350$ / $244$ / $53$ / $333$ / $280$
counts reproduce under the corrected command (RUN) and stand.

---

**Batch 1 summary.** 16 findings filed, all severity `strike`; 16 applied. Two clauses stand
**OPEN** with their Terminal sentences now in force — 05.4's clause (b), for want of an in-scope
planted negative that cannot be manufactured without writing a file; and 05.9's Kill clause 2,
for want of a population to plant into. Both are replaced by a genuinely different instrument on
a different object (the interpreter census; the deleted-test-module restore-or-retire pass),
each with its own kill number and each firing on the present tree. **Eleven** verses were
rewritten — 05.1, 05.2, 05.3 (step 5), 05.4, 05.5, 05.6, 05.7, 05.8, 05.9, 05.14, 05.15 (B12's
evening) — plus the cheapest-first kill table and 05.2's and 05.3's Terminals, which were
re-keyed off `box.torch` onto the flat `torch` key 05.4 now freezes. Three constants are struck from the canon by name and appear
nowhere else in this book: $2971$ / $26$ / $17.69$ s (05.2's collection triple) and
$68$ / $59$ / $68$ (05.9's census triple).

---

MARS's findings on this book, repair round 1, **batch 2 of 5**. Same rule: every finding verbatim
— verse, flaw, mechanism, number, `replacement_survives`, required repair, severity — beside the
repair applied or the sentence now in force. Where the repair made a finding read *worse* than
filed (05.13's plant, whose pinned count is $127$ and not the $133$ the finding and the verse both
quote; 05.9's plant, whose pinned opening population is $20$ and not the $68$ the struck triple
implied), the worse reading is the one recorded.

---

**Finding 17 · verse 05.9 · severity `strike` · `replacement_survives: false`**

> **flaw** — The prescribed repair cannot clear the errors its own kill measures: the residual
> errors are broken imports of deleted test modules, not prose reads.
> **mechanism** — V-16, M-17.
> **number** — All 9 non-environment errors raise ImportError/ModuleNotFoundError on
> `tests.saturn.test_v20_r15_freeze_manifest`, `tests.jupiter.test_v20_r15_it14_theory_table`,
> `tests.jupiter.test_v20_r15_it20_citation_freeze`,
> `tests.jupiter.test_v20_r15_it27_star_lands_and_overturns`,
> `tests.saturn.test_v20_r15_it27_wing_arm_citation`. None opens a `.md`. Steps (a)-(c) move
> `.md`-naming files; the frozen kill then still reads `>0 errors`.
> **required repair** — Add step (d): the 9 files importing deleted sibling test modules are
> moved to attic/tests/ or their imports are restored from the pin, and re-freeze the kill's
> collection number at 0 only after (d).

**Repair applied — 05.9 plan, Kill.** The restore-or-retire pass the finding requires exists in the
plan as **(b′)**, added in batch 1 under Finding 4, and this batch adds what (b′) alone did not
carry: the **freezing order, filed as step (d)**. The plan now states in the verse's own words that
steps (a)–(c) move and restore files that *name* `.md` documents, that **not one of the 9
collection errors is a prose read**, and that the collection number is frozen at `9` **until (b′)
lands and `0` only after (b′)** — so a verifier executing (a), (b) and (c) and then reading the
collection clause at `0` would have read a failure on a repair performed exactly as written
(V-16, M-17). The Kill's opening line is re-worded to "After (a), (b′), (c) and the ordering (d)",
and the final verification `pytest -q --collect-only` reading `0 errors` is read after (b′), never
after (c). The five deleted modules the 9 importers name are listed by module path inside step (d),
matching the finding's list exactly.

---

**Finding 18 · verse 05.9 · severity `strike` · `replacement_survives: true`**

> **flaw** — The second planted negative requires writing a .py file into tests/, which the
> canon's own §3 forbids.
> **mechanism** — D-6.
> **number** — "A second plant for the policy itself: a temp test that opens `README.md` and
> asserts a line number must be **caught** by the census ... and the plant must then read `1`."
> CHARTER §3: "No code file is written — no `.py`, `.sh`, `.lean`, `.ipynb`, no scratch helper in
> the tree".
> **required repair** — Replace the plant with one that requires no new file: run the census
> against `git show 99777ab:tests/` (where prose-coupled files exist by the hundred) and show it
> reading a positive count, or against one surviving file that opens a present root `.md`.

**Repair applied — 05.9 Kill, the policy plant; and the finding is sharpened against this book.**
The forbidden plant is deleted and quoted in the verse as what it was: a condemning rule whose only
planted negative was an act the constitution forbids, which is no planted negative at all (V-15,
D-6). The replacement runs entirely inside `git` on the pinned tree and writes nothing — two
`git grep -l … 99777ab -- 'tests/*.py' | wc -l` lines, printed in the verse. RUN 2026-09-06 at
`99777ab`: **68** files name a quoted `.md` token, **20** carry it on a line that also calls
`read_text`, `open(`, `.exists(` or `read_bytes`, over **30** such lines. Both positive, $0$ GPU-s,
no file created. Two facts are recorded beyond the finding, one of which reads worse for this book
than the finding does. First, the pinned naming count **68** is exactly the first element of the
triple $68$ / $59$ / $68$ struck in batch 1: the number was not fabricated but **carried across a
commit boundary**, so the diagnosis is sharpened from P-1 (no producer) to P-6 (a count read at the
wrong commit), and the canon records the sharper one. Second, the pin's own *opening* count is
**20**, not $68$ and not $59$ — so even at `99777ab` the population `CHARTER.md:229`'s row
describes was under a third of the size that row's own number implies.

---

**Finding 19 · verse 05.10 · severity `strike` · `replacement_survives: false`**

> **flaw** — The rule's central term names a list that was never written, so the Kill cannot be
> evaluated on any row.
> **mechanism** — V-10, P-4.
> **number** — `V17K_RULINGS.md:150 @ 99777ab`: "cell-id list frozen at launch under Ruling 6f.
> Owed by the **envelope node**" — the list is owed, not filed. `V17K_RULINGS.md:80` states 6f's
> rule; `:145` states "B2's membership is a list, not a description". No frozen list exists in the
> tree. The Kill reads "a row whose post-run slot has no producer cell id in the frozen list".
> **required repair** — Either write the frozen deciding-cell list in this verse (it is a
> specification, permitted), or re-write the rule so the producer is named by file:line of the
> script that emits the slot's value — which for row 1 is already available
> (`scripts/k_noise_floor.py:379`).

**Repair applied — 05.10 Statement, Hypotheses, Kill; both branches taken.** The rule's central term
is re-written on `file:line` **and** the resulting list is filed here as a three-row producer table,
which `CHARTER.md` §3 permits inside markdown. Row 1's producer is
`scripts/k_noise_floor.py:379`, `abs_delta_final_loss=abs(final_a - final_b)` (RUN
`sed -n '379p'`); row 2's is `kaggle/ceq_v17k.ipynb` cell 18, pinned at
`READ V17_R10P_LRT.md:501 @ 99777ab`; row 3 has none, being entirely pre-launch. The verse
reproduces the finding's own reading of `V17K_RULINGS.md:150` — *owed*, not filed — and states that
a kill whose predicate is membership of a list nobody wrote is not decidable on any row (V-10,
P-4). The Kill is re-frozen on this verse's table, its instrument is the table read against the
rulings table, and its planted negative is row 1 as the rulings state it at the pin: a slot with an
owner and **no producer of any kind**, read RED. The plant is strengthened past the finding: the
same row read against this verse's table reads GREEN on the producer clause and RED on the
acceptance clause, so the two clauses are shown to be independently reachable rather than firing
together on every row.

---

**Finding 20 · verse 05.10 · severity `strike` · `replacement_survives: false`**

> **flaw** — Row 2's producer cell id is cited to a line that contains no cell number; the string
> is sourced from an unrelated file.
> **mechanism** — P-2, P-5.
> **number** — `ceq/compat.py:50-52` reads about running `python -m ceq.hf.smoke` in a subprocess
> and why K-COMPAT runs before it. `grep -rn "cell 18"` across the tree hits only
> `kaggle/snapshot/repo/V17_R10P_LRT.md:29,80,501,508`.
> **required repair** — Re-pin to `READ kaggle/snapshot/repo/V17_R10P_LRT.md:501 @ 99777ab`
> ("`kaggle/ceq_v17k.ipynb`, **cell 18 only**"), or name the notebook cell by its own hash.

**Repair applied — 05.10 row 2 and the producer table; with one correction to the required repair
itself, printed rather than reconciled.** The finding's readings reproduce exactly: RUN
`sed -n '48,54p' ceq/compat.py` is `ensure_generation_mixin`'s process-local alias and why K-COMPAT
precedes the subprocess smoke, and RUN `grep -rn "cell 18" --include=*.md .` hits only
`kaggle/snapshot/repo/V17_R10P_LRT.md` at `:29`, `:80`, `:501`, `:508`. The `ceq/compat.py:50-52`
pin is **struck** in the row and named as struck. The path the finding names for the re-pin,
however, **does not exist at the pin**: RUN
`git cat-file -e 99777ab:kaggle/snapshot/repo/V17_R10P_LRT.md` reads
`fatal: path … exists on disk, but not in '99777ab'`, and RUN `git log --oneline --` on it reads
`9ce3048` and `208cf69`. At `99777ab` the file is at the **root** — RUN
`git show 99777ab:V17_R10P_LRT.md | sed -n '501p'` returns "`kaggle/ceq_v17k.ipynb`, **cell 18
only**, appended after `T.train(...)` and its" verbatim — so the primary pin is
`READ V17_R10P_LRT.md:501 @ 99777ab` and the snapshot copy is cited live beside it as
`READ kaggle/snapshot/repo/V17_R10P_LRT.md:501 @ 9ce3048`. The notebook-cell hash alternative is
not taken: computing it would require running code that does not exist
(`NOT MEASURED — needs a notebook-cell digest instrument`), and the document pin is decidable at
$0$ GPU-s today.

---

**Finding 21 · verse 05.11 · severity `strike` · `replacement_survives: false`**

> **flaw** — The Kill is not decidable on any registered bed: it names no draw, no seed rule, no N
> and no instrument that exists in the repository.
> **mechanism** — V-11, M-3.
> **number** — "Kill. The author rules against a default. Frozen: the count of reversed defaults,
> $0$ to $5$ ... Instrument: the author's ruling, filed verbatim in `CORRECTIONS.md`." CHARTER §2
> reachability rule: "Every kill names the draw, seed rule and $N$ at which it is decidable". No
> number in the repository can make this fire, and none can make it fail to fire.
> **required repair** — Re-write the kill as a decidable property of the defaults themselves —
> e.g. "any default whose reversal changes a number rather than a reading" (checkable at $0$ by
> inspection of the table's right-hand column, whose every cell must read $0$ GPU-s and no number
> falsified) — and file the author's ruling as an event, not a kill.

**Repair applied — 05.11 Kill and If-killed.** The kill is re-written exactly as the finding
prescribes: **any default in the table whose reversal changes a number rather than a reading**,
frozen on all five right-hand cells reading $0$ GPU-s and falsifying no number, instrument the five
cells themselves, $N=5$ exhaustively, price $0$, decidable **before** any ruling exists — which is
the property the pre-repair kill lacked, and the verse says so in the finding's terms (an author's
ruling is an event the canon records, never a kill it can price). The author's ruling is re-filed
as an **event** with its date in `CORRECTIONS.md`. **The new kill's planted negative is this
book's own row 3 as it stood before this batch**, which did change a number (it made the chess
bed's Kaggle-attach count $2$ instead of $0$) and on which the kill fires — Finding 22 is that
firing. Its **true negative** is row 5, a pure renaming, on which the kill must not and does not
fire. The If-killed no longer prices a reversal: a default whose reversal costs a number is
**withdrawn as a default** and re-filed as an OPEN question with no default, because the canon is
not licensed to take that decision without the author; the last link prints
`NOT MEASURED — needs the author's ruling` in every verse the reading touches.

---

**Finding 22 · verse 05.11 · severity `strike` · `replacement_survives: false`**

> **flaw** — Row 3's reversal cost contradicts the CHARTER and book 09: the chess bed's local
> route already exists and requires no Kaggle attach.
> **mechanism** — V-22, P-5.
> **number** — CHARTER §6, book 09 scope: "the local data route (lichess database dumps or
> **python-chess-generated games, hashed and pinned per Ruling 7**) **before any Kaggle attach**".
> `V20_R15_JOURNAL.md:2425-2427 @ 99777ab` (RUN at the pin): `python-chess` **installed**,
> `[RUN] 1.11.2`; `ceq/kdata.py:258 label_plies` replays the mainline through a `chess.Board` —
> **BUILT, RUNS TODAY**. Ruling 7's form is generator + seed + expected hash
> (`V17K_RULINGS.md:83-86`), which a generated corpus satisfies. The verse writes "two Kaggle
> attaches behind the yes ... Ruling 7 cannot apply to an external PGN".
> **required repair** — Re-write the reversal column: the chess bed is a python-chess generator +
> seed + hash under Ruling 7 at $0$ Kaggle attaches; the Kaggle attach is a second, optional source
> pinned by its SHA-256. Reconcile with book 09's verse ids.

**Repair applied — 05.11 row 3, both cells, and a new paragraph naming the contradiction.** Every
one of the finding's readings is verified at the pin and reproduced: RUN
`git show 99777ab:V20_R15_JOURNAL.md | sed -n '2423,2430p'` reads `python-chess` **installed**,
`[RUN] 1.11.2`, and `ceq/kdata.py:258 label_plies` **BUILT, RUNS TODAY**; RUN
`sed -n '255,262p' ceq/kdata.py` shows `label_plies` replaying on a real board; RUN
`git show 99777ab:V17K_RULINGS.md | sed -n '83,86p'` reads Ruling 7 as "generator + seed + expected
hash, regenerated and asserted in-notebook". The **default** cell is re-written: the chess bed is
book 09's **BED-C-G** (09.2), python-chess self-play under $\pi_u$, generator $+$ seed $+$ hash
under Ruling 7, at $\mathbf{0}$ Kaggle attaches, first cell 09.13 at $0$ GPU-s and first GPU cell
09.14. The **reversal** cell is re-written as additive, not enabling: the attaches become book 09's
**BED-C-D**, one lichess monthly dump pinned by its own SHA-256 into `results/k_data_manifest.json`,
with `lichess/chess-evaluations` replaced locally by 09.6's engine recompute; $2$ evenings after
the yes, and nothing local was waiting (09.37). The struck sentence "Ruling 7 cannot apply to an
external PGN" is named in the verse as V-22 and P-5, and the pre-repair default's claim that "the
chess witness is not a bed" is named as contradicting `CHARTER.md` §6 and book 09 as written. Two
knock-on repairs travel with it: 05.15 gains a **B27** row scheduled on the local route, and this
finding is the planted negative Finding 21's new kill needed.

---

**Finding 23 · verse 05.13 · severity `strike` · `replacement_survives: false`**

> **flaw** — The third census line's frozen number contradicts the CHARTER's own book table: ten
> books plus CHARTER plus CORRECTIONS is twelve, not ten.
> **mechanism** — P-1, M-2.
> **number** — CHARTER §6 lists books 00, 01, 02, 03, 04, 05, 06, 07, **08 `08_ARCHITECTURE.md`**,
> **09 `09_CHESS_AND_MARKETS.md`** plus `CORRECTIONS.md`; `ls docs/canon/` RUN shows 01, 02, 03,
> 04, 05, 06, 08, 09, CHARTER, CORRECTIONS already on disk. The verse freezes
> `git ls-files 'docs/canon/*.md' | wc -l` at **10** "(00–07, CHARTER, CORRECTIONS)". RUN today:
> **0** (untracked).
> **required repair** — Re-freeze the third census number at 12 and correct the parenthetical to
> (00–09, CHARTER, CORRECTIONS). As written the kill fires on a correct birth commit.

**Repair applied — 05.13 census block, Kill; and 05.14's evening-5 row and the cheapest-first kill
table with them.** The third line is re-frozen at **12** with the parenthetical **(00–09, CHARTER,
CORRECTIONS)**, and the arithmetic is printed: ten books $+$ `CHARTER.md` $+$ `CORRECTIONS.md`
$=12$. The verse states in its own words that the frozen `10` omitted books **08** and **09** — the
two the author's §0 clarification adds — and that a coordinator committing the canon as the CHARTER
specifies it would have read `12` against a frozen `10`, firing the kill on a correct birth commit
(M-2, a threshold frozen against the wrong object; P-1, no command in the tree produces $10$). RUN
today is recorded: the line reads $\mathbf{0}$ because `docs/canon/` is untracked, and
`ls docs/canon/` lists ten files, books 00 and 07 being the coordinator's and written last; the
line is read **only after the birth commit**, and $0$ before it is not a violation. The frozen
triple is `11 / 0 / 12` everywhere it appears: 05.13's Kill, 05.14's evening-5 verification and the
cheapest-first kill table.

---

**Finding 24 · verse 05.13 · severity `strike` · `replacement_survives: true`**

> **flaw** — The Hypotheses and the planted negative are already spent: the deletion commit landed,
> and the census reads its frozen values today rather than the pin's.
> **mechanism** — P-3, V-15.
> **number** — RUN `git diff --cached --name-only --diff-filter=D | wc -l` = **0** (deletions
> committed at c71527a). RUN
> `git ls-files | grep -cE '_IT[0-9]+_|^LOOP_PROMPT|_ARCHIVE\.md$|^CEQ_V1[0-9]_.*DELTA\.md$|^\.superpowers/|^(DONE|BOARD|CHECKLIST|BACKLOG|NOTES|REPORT)\.md$'`
> = **0**, not 133. The verse's hypothesis reads "until then `git ls-files` still lists the deleted
> paths, and the second line reads the pin's values".
> **required repair** — Delete the staging hypothesis; re-state the plant as `git ls-files` at
> 99777ab via `git ls-tree -r 99777ab --name-only | grep -cE ...`, which still reads 133, and
> record that line 2 already reads 0 on the tree.

**Repair applied — 05.13 Hypotheses and Kill; and the finding's own number is corrected downward,
against this book.** Both RUN facts reproduce: `git diff --cached --name-only --diff-filter=D |
wc -l` reads $\mathbf{0}$ and `git ls-files | grep -cE '…'` reads $\mathbf{0}$. The staging
hypothesis is **deleted as spent** and named as spent in the verse, with P-3 (a stale claim never
retracted) and V-15 (a condemning rule whose plant fires on a state that no longer exists) recorded
against it. The plant is re-stated on the pinned tree through `git ls-tree`, printed as a runnable
block. **But the pinned reading is $\mathbf{127}$, not the $133$ the finding and the verse both
quote**, and the canon records the worse number: the census line counts **files**, while $133$ is
the sum of six per-class counts ($98+7+6+3+14+5$), and RUN
`git ls-tree -r 99777ab --name-only | grep -E '^LOOP_PROMPT' | grep -cE '_ARCHIVE\.md$'` reads
$\mathbf{6}$ — every `_ARCHIVE.md` file is a `LOOP_PROMPT_*_ARCHIVE.md` and is counted twice by the
sum. $133-6=127$. A per-class sum quoted as the output of a file-counting command is P-1, a number
whose producer is a different command from the one named, and the frozen plant is now the command's
own reading, $127$.

---

**Finding 25 · verse 05.14 · severity `strike` · `replacement_survives: true`**

> **flaw** — Evening 3 is under-priced and the first-five total is wrong; the verse hedges the
> arithmetic instead of stating it.
> **mechanism** — M-8, D-6.
> **number** — 05.6 (16.57 s) and 05.7 (17.63 s) require incompatible invocations (F2), so
> evening 3 = 34.20 GPU-s and the five-evening total is 0 + 10.64 + 34.20 + 17.94 + 0 = **62.78**,
> not "$\le46.21$ DERIVED, each evening under $20$". The table's own cell reads "or $16.57$ s on
> its own if run separately, whichever the evening holds under $20$" — a hedge where CHARTER §2
> requires a number.
> **required repair** — Split evening 3 per 05.1's rule, print the two sub-runs with their own
> prices, and re-derive the total. Delete the "whichever" clause.

**Repair applied in batch 1 under Finding 2, and this batch records that the applied numbers are
worse than the ones this finding derives.** 05.14's table carries **three** sittings, not two —
evening 3 (05.7, $17.63$ s), evening 3b (05.6, $16.93$ s) and evening 3c (05.8, $5.88$ s) — each
with its own verification row. The "whichever the evening holds under $20$" clause is **deleted**;
RUN `grep -n "whichever" docs/canon/05_REPAIRS.md` reads $0$ hits. Two of the finding's numbers move
against the book and both are recorded as the worse reading: 05.6's price is $\mathbf{16.93}$ s, not
$16.57$, because batch-1 Finding 10 forces the `--s 32` run onto the **gated** arm `arm_pl`
($1.884$ s) for the bind-row clause to be reachable at all — so a merged evening 3 would have cost
$\mathbf{34.56}$ GPU-s, not $34.20$; and 05.8's $5.88$ s cell, which this finding does not count,
cannot join 3b either ($16.93+5.88=22.81>20.0$). The five-evening total is therefore
$0+10.64+17.63+16.93+5.88+17.94+0=\mathbf{\le 69.02}$ over seven sittings, not $62.78$ and not
$\le46.21$. Batch 2 adds one further correction the finding did not reach: that total's class is
**not DERIVED** but `[FITTED, extrapolated at $n=16384$ and $n=32768$]`, the class of its weakest
input (Finding 32), and 05.14's Terminal is re-tagged with it.

---

**Finding 26 · verse 05.14 · severity `strike` · `replacement_survives: true`**

> **flaw** — Two of the three schedule totals are arithmetic errors against the plan line they
> READ.
> **mechanism** — P-1.
> **number** — `docs/PLAN.md:885 @ 99777ab`: "**13.5, called 14 evenings to a verdict and a filed
> dossier.** Add M-5.3 (the ladder on a survivor, 1) for **15**; add the cost-law branch S-66(3)
> -> M-5.2(1) ... for **19**." With the verse's own +2 (evenings 1 and 2) the three numbers are 16,
> **17**, **21**. The verse prints "Sixteen evenings ... eighteen with M-5.3's ladder; twenty-two
> with the cost-law branch".
> **required repair** — Correct eighteen to seventeen and twenty-two to twenty-one, or state the
> extra evening's source explicitly.

**Repair applied — 05.14 critical-path paragraph.** The plan line is verified at the pin — RUN
`git show 99777ab:docs/PLAN.md | sed -n '885p'` returns "**13.5, called 14 evenings to a verdict
and a filed dossier.** Add M-5.3 (the ladder on a survivor, 1) for **15**; add the cost-law branch
$\text{S-66}(3)\to\text{M-5.2}(1)$ if a cost law is demanded before the ladder, for **19**." — and
is now quoted in the verse. The three totals are re-derived in print: $14+2=\mathbf{16}$,
$15+2=\mathbf{17}$, $19+2=\mathbf{21}$. **Eighteen** becomes **seventeen** and **twenty-two**
becomes **twenty-one**. The first branch of the required repair is taken rather than the second,
and the verse says why: no extra evening was named anywhere as the source of the $+3$, so there was
nothing to state explicitly — a number with its producer sitting in its own citation and
contradicting it is P-1.

---

**Finding 27 · verse 05.15 · severity `strike` · `replacement_survives: false`**

> **flaw** — The scheduling table omits B26 and B27 — the two census rows that carry the author's
> own §0 use cases — and neither string appears anywhere in the book.
> **mechanism** — D-6, L-VERSE.
> **number** — RUN `grep -n "B26\|B27" docs/canon/05_REPAIRS.md` returns **0 hits**. CHARTER §5's
> census runs B1–B27; B26 (no architecture document, owner 08) and B27 (prediction-market trades
> and chess have no bed, pipeline, label, floor or matched baseline, owner 09) have no
> prerequisite, no earliest evening and no price class. The "Census rows closed" table likewise
> stops at B25.
> **required repair** — Add two rows: B26 — closing book 08, prerequisite none (a specification),
> earliest evening any, $0$ GPU-s; B27 — closing book 09, prerequisite 05.3 and 05.5 for any GPU
> cell, first cell at $0$ GPU-s under the python-chess generator route (F47), earliest evening any
> for the $0$ cell.

**Repair applied — 05.15 scheduling table, a new paragraph, and the "Census rows closed" table.**
Both rows are added exactly as prescribed. **B26**: closing book **08**, prerequisite **none** (a
specification — the book is written, not run), earliest evening **any**, $\mathbf{0}$ GPU-s.
**B27**: closing book **09** with 04, prerequisite **none for the $0$ GPU-s cells** (09.10's
census, 09.13's first chess cell and 09.32's market census run on CPU under the python-chess
generator route of 09.2, at $0$ Kaggle attaches, which is Finding 22's corrected reading of 05.11
row 3) and **05.3 and 05.5 for any GPU cell** (09.14); earliest evening **any** for the $0$ GPU-s
cells and $2+4$ for 09.14. The finding's "(F47)" is a book-09 finding id and is not citable from
this book, so the route is named by book 09's own verse ids instead, as `CHARTER.md` §6 requires
books to cross-reference. The verse records the finding's own count against itself — RUN on the
pre-repair file, `grep -c "B26\|B27"` read $\mathbf{0}$ — and names the mechanism: D-6, a repair
scheduled nowhere is a repair never started, and L-VERSE, a census row with no verse. The "Census
rows closed" table gains a B26 row and a B27 row. The scheduling consequence is stated: **B27's
first three cells are unblocked today**, needing no certified stack because they run no GPU cell.

---

**Finding 28 · verse book-wide (05.1, 05.2, 05.9, 05.13, 05.15) · severity `strike` ·
`replacement_survives: false`**

> **flaw** — Every citation of the constitution is inadmissible: docs/canon/ is not in the commit
> the book pins to, and none of the five line numbers resolves against the live file either.
> **mechanism** — P-2, P-6, L-EQ.
> **number** — RUN `git ls-tree -r 99777ab --name-only | grep -i canon` returns **0 lines**; git
> status reads `?? docs/canon/`. Against the live CHARTER.md: :208 is the Ruling 7 law row (B1 is
> :227); :210 is D-CALIB (B3 is :229); :231 is B5 (B24 is :250); :206-232 is the laws table plus
> B1–B6 (the census is :225-253); :248 is B22 (the book-05 scope row is :269). Only `:5-9`
> (CORRECTIONS.md as the only door) resolves correctly.
> **required repair** — Drop the `@ 99777ab` from every docs/canon/ citation and re-pin to the live
> line numbers (:227, :229, :250, :225-253, :269), with a note that the canon's own pin is the
> birth commit and does not exist until then.

**Repair applied — 05.1, 05.2, 05.9, 05.13, 05.15; three of the five were repaired in batch 1 under
Finding 1 and two remained.** The audit is RUN and printed here: `grep -n "docs/canon/CHARTER.md"`
over the repaired book returns nine citations, and RUN
`grep -n "docs/canon/[^ ]*@ 99777ab"` returns **$0$ hits** — no `docs/canon/` line carries a pin
anywhere in the book. Batch 1 corrected the book-05 scope row to `:269` (05.1, twice), B1 to `:227`
(05.2) and B3 to `:229` (05.9), each tagged "live, no pin" with the reason RUN beside it. **Batch 2
corrects the two that survived**: 05.13's round-15 citation moves from `:231` — which is row **B5**,
"no CEQ arm has been trained", and is named as struck in the verse — to
`READ docs/canon/CHARTER.md:250`, row **B24**, this verse's own row; and 05.15's Evidence moves from
`READ docs/canon/CHARTER.md:206-232 @ 99777ab` — the laws table plus B1–B6, at a commit that holds
no canon at all — to `READ docs/canon/CHARTER.md:225-253` (live), the census B1–B27, with the pin
struck by name. `:5-9` stands, as the finding says. The note the repair requires is carried in
05.1: the canon's own pin is its **birth commit**, which does not exist yet, so every
`docs/canon/` citation in this book is to the live file.

---

**Finding 29 · verse 05.3 · severity `repair` · `replacement_survives: true`**

> **flaw** — Step 6's frozen output string is in a different field order than the re-derivation
> line it cites as its authority.
> **mechanism** — P-6.
> **number** — `requirements.txt:6` prints `torch.__version__, transformers.__version__,
> numpy.__version__, scipy.__version__` = `2.5.1+cu121 5.3.0 1.26.4 1.17.1`. Step 6 freezes
> `2.5.1+cu121 1.26.4 1.17.1 5.3.0`.
> **required repair** — Either quote requirements.txt:6's command verbatim with its own order, or
> drop the claim that :6 "names" this re-derivation and present step 6 as the canon's own command.

**Repair applied — 05.3 step 6; the first branch is taken.** RUN `sed -n '6p' requirements.txt`
returns `#   python -c "import torch,transformers,numpy,scipy; print(torch.__version__,
transformers.__version__, numpy.__version__, scipy.__version__)"`, confirming the finding's field
order exactly. Step 6's command is replaced by that line quoted verbatim and its frozen string by
that command's own output, `2.5.1+cu121 5.3.0 1.26.4 1.17.1`. A paragraph beside the table names
the defect rather than silently swapping the string: a frozen string in one field order attributed
to a command that prints another is P-6 turned into a false quotation, and a verifier executing
`:6` on a correctly pinned box and comparing against the old string would have read a **failure**.
This book adds no command of its own at step 6.

---

**Finding 30 · verse 05.3 · severity `repair` · `replacement_survives: false`**

> **flaw** — Replacement 2 (the cu118 build) is named, not derived: its Evidence field reads "none
> beyond the assumption", and the probe that would settle it costs $0$ and was run for cu121 in
> the same session.
> **mechanism** — P-4.
> **number** — The verse RUNs `pip index versions torch --index-url
> https://download.pytorch.org/whl/cu121` and `pip index versions numpy` but records the cu118
> index as `[ASSUMED]`, "not probed this session". Price of the probe: $0$ GPU-s, one command.
> **required repair** — RUN `pip index versions torch --index-url
> https://download.pytorch.org/whl/cu118` and replace the [ASSUMED] with the printed list, or
> strike replacement 2 and let replacement 1 fall straight to the Terminal.

**Repair applied — 05.3 replacement 2, Evidence; the first branch is taken and the probe is RUN.**
RUN 2026-09-06 `pip index versions torch --index-url https://download.pytorch.org/whl/cu118` prints
`torch (2.7.1+cu118)` with available versions `2.7.1+cu118, 2.7.0+cu118, 2.6.0+cu118,
2.5.1+cu118, 2.5.0+cu118, 2.4.1+cu118, 2.4.0+cu118, 2.3.1+cu118, 2.3.0+cu118, 2.2.2+cu118,
2.2.1+cu118, 2.2.0+cu118, 2.1.2+cu118, 2.1.1+cu118, 2.1.0+cu118, 2.0.1+cu118, 2.0.0+cu118` — the
full list is printed in the verse and **`2.5.1+cu118` is served**, so the route exists. The
`[ASSUMED]` tag and the Evidence field reading "none beyond the assumption" are **struck by name**,
with the verse recording that the probe was one command at $0$ GPU-s — the same command the verse
had already run for `cu121` in the same session — so a named-not-derived replacement (P-4) stood
where a RUN was one line away.

---

**Finding 31 · verse 05.3 · severity `repair` · `replacement_survives: false`**

> **flaw** — Replacement 2's Kill fires on the replacement succeeding; no condition is stated under
> which the cu118 route fails, so the second link has no refutation.
> **mechanism** — V-3, V-10.
> **number** — As written: "Kill: step 2's line prints `2.5.1+cu118 11.8 True ...`" — that is the
> success string of the replacement's own verification.
> **required repair** — Re-write the kill as "step 2 under the cu118 index prints anything other
> than `2.5.1+cu118 11.8 True NVIDIA GeForce RTX 4060 Laptop GPU`", and move the consequence (a
> different certified stack, V-22, mandatory 05.5) into the Terminal where it belongs.

**Repair applied — 05.3 replacement 2's Kill and the verse's Terminal.** The kill is re-written
verbatim as the finding prescribes: step 2 **under the `cu118` index prints anything other than**
`2.5.1+cu118 11.8 True NVIDIA GeForce RTX 4060 Laptop GPU`, exact string equality on all four
fields — the verse's own frozen form with `12.1` replaced by `11.8`. The verse states in its own
words that the pre-repair kill fired on the success string, so the second link had **no refutation
at all** (V-3, V-10) and a `cu118` install that worked perfectly was scored as a kill. A planted
negative is added and fires today: the present box prints `2.14.0+cpu None False None` (RUN) under
either index. The consequence is moved into the **Terminal**, which now carries two outcomes and a
different licensed sentence on each — if `cu118` installs, the certified device exists again but is
**not the same object**, every `COSTS.md` §1 constant is withdrawn as a comparand under V-22,
05.5's full $523.9$ s certificate is mandatory before any price is quoted, and `cu118` is required
in the flat `torch` string of every journal row 05.4's predicate reads; if no `torch 2.5.1` wheel
installs at either index, the pre-existing history-at-`ab5b485` sentence stands.

---

**Finding 32 · verse 05.5 · severity `repair` · `replacement_survives: true`**

> **flaw** — The <=10.64 GPU-s evening-2 price is tagged DERIVED but two of its ten inputs are
> extrapolated [FITTED] values at shapes the record itself refused as arm measurements.
> **mechanism** — P-8, M-3, V-22.
> **number** — `COSTS.md:232-233` tag the arm at n=16384 (0.748531) and n=32768 (1.499743)
> `[FITTED]`, extrapolated; `COSTS.md:76-79` records why — the allocator reserves 10.578 GiB and
> 13.969 GiB on a 7.996 GiB card, so the loop timed the bus. The sum
> 3*(2.901402+0.312439)=9.641523 is arithmetically right; its class is not DERIVED.
> **required repair** — Re-tag the price `[FITTED, extrapolated at n=16384 and n=32768]` and add
> the sentence that a forward-only zero-step cell at those n was reachable at ab5b485
> (COSTS.md:137-139, 30/30 reachable), which is what makes the ladder runnable despite the
> training-cell ceiling.

**Repair applied — 05.5 evening-2 bullet, and the tag propagated to every place the price is
quoted.** All three readings are verified at the pin: RUN
`git show 99777ab:COSTS.md | sed -n '228,236p'` shows the arm's $n=16384$ and $n=32768$ rows
carrying "`[FITTED]`, extrapolated"; RUN `sed -n '74,80p'` shows the allocator reserving $10.578$
GiB and $13.969$ GiB on the $7.996$ GiB card, "so the driver pages over PCIe and the loop times the
bus", which is why the arm's law is fitted over three points and not five; RUN `sed -n '135,140p'`
shows "**30 cell shapes, 30 reachable, 30 pass**", worst margin $1.957\times10^{-3}$. The price is
re-tagged **`[FITTED, extrapolated at $n=16384$ and $n=32768$]`** in 05.5's bullet, in 05.14's
evening-2 row, in 05.14's seven-sitting total ($\le69.02$, whose class is now its weakest input's
rather than DERIVED) and in 05.14's Terminal. The sentence the repair requires is added in the
finding's own terms: the zero-step gate's cells are **forward-only** and all thirty were reachable
at `ab5b485`, so the two shapes the *training* law could not measure are shapes the *zero-step*
stage has already run — the bound over-estimates a stage known reachable rather than estimating one
known unreachable, and evening 2's price is re-measured against the certificate it writes rather
than quoted forward.

---

**Batch 2 summary.** 16 findings filed — 12 `strike`, 4 `repair`; **16 applied, 0 left OPEN**.
Nine verses were rewritten: 05.3 (step 6, replacement 2's Evidence and Kill, the Terminal), 05.5
(the evening-2 price class), 05.9 (step (d), the Kill's opening line, the policy plant), 05.10 (the
producer table, row 2's pin, Hypotheses, Kill), 05.11 (row 3 both cells, the Kill, the If-killed),
05.13 (the census block, Hypotheses, Kill), 05.14 (the critical-path totals, the price class),
05.15 (the B26 and B27 rows, Evidence) and the closing tables (Census rows closed, Kills cheapest
first). Two replacements that died with their verse were **re-derived on a different object rather
than renamed**: 05.10's producer term, which moves from a never-filed Ruling-6f cell-id list to a
`file:line` table filed here as a specification, decidable at $0$ GPU-s on a tree that exists; and
05.11's Kill, which moves from an unreachable event ("the author rules against a default") to a
property of the five defaults themselves, decidable before any ruling exists, with 05.11 row 3's
own pre-repair cell as its planted negative and row 5 as its true negative. Two numbers moved
**against this book** and are recorded at their worse reading: 05.13's pinned plant reads
$\mathbf{127}$, not the $133$ that both the finding and the verse quote, because six
`LOOP_PROMPT_*_ARCHIVE.md` files are counted twice by the per-class sum; and 05.9's pinned policy
plant reads $\mathbf{20}$ files *opening* a `.md` at `99777ab` against $68$ *naming* one, so the
population `CHARTER.md:229`'s row describes was under a third of that row's own number even at the
pin — and the struck constant $68$ is identified as the pin's naming count read as the tree's, P-6
rather than P-1. One number in a finding's own required repair is corrected in print rather than
reconciled: the path `kaggle/snapshot/repo/V17_R10P_LRT.md` does not exist at `99777ab`, where the
file is at the root, so 05.10's row-2 pin is `READ V17_R10P_LRT.md:501 @ 99777ab` with the snapshot
copy cited live at `@ 9ce3048`. Two constants are struck from this book by name and appear nowhere
else in it: `2.5.1+cu121 1.26.4 1.17.1 5.3.0` (05.3's step-6 string, in a field order its cited
command does not print) and $10$ (05.13's third census number, which would have fired the kill on a
correct birth commit).

---

MARS's findings on this book, repair round 1, **batch 3 of 5**. Same rule: every finding
verbatim — verse, flaw, mechanism, number, `replacement_survives`, required repair, severity —
beside the repair applied or the sentence now in force. Three findings read **worse** after
repair than when filed, and the worse reading is the one recorded: 05.5's split certificate
filename is not merely deliberate but **mandatory**, because `scripts/k_cert.py` overwrites and
the finding's first branch would destroy the `ab5b485` certificate; 05.4's `box` record holds
**13** keys and not the 14 the finding counts, and its length is a branch on `cuda_available`
rather than a fixed set; and 05.2's `results/` mtimes span **176** distinct values, not the 5
the finding reads, with the 18 drift-window files carrying four sub-millisecond stamps and not
one.

---

**Finding 33 · verse 05.5 · severity `repair` · `replacement_survives: true`**

> **flaw** — Three load-bearing COSTS.md pins name the wrong lines; the range as written omits
> two of the ten numbers it sums.
> **mechanism** — P-6.
> **number** — The ten s/step values live at COSTS.md:229-238, not :222-235 (:222-225 is the
> proj-resv prose; the cited range excludes softmax n=16384 at :237 and n=32768 at :238).
> 0.010162 is at :234 and 0.093066 at :229, not ":230,224". C_RESIDUAL 17.874 and C_OPERATOR
> 3.823 are at :89-90, not :91-94 (:91-92 are the bf16 rows).
> **required repair** — Re-pin to COSTS.md:229-238, :234, :229 and :89-90.

**Repair applied — 05.5's evening-2 bullet, its across-stack drift row and its memory row.**
All three re-pins are RUN-verified at the pin: `git show 99777ab:COSTS.md | sed -n '229,238p'`
prints the ten `s/step` rows, five `arm_smprime` at `:229-233` and five `softmax` at
`:234-238`; `grep -n` on the four extreme values reads `229` ($0.093066$), `233` ($1.499743$),
`234` ($0.010162$), `238` ($0.160907$); `sed -n '89,90p'` prints the two **fp32** rows,
`C_RESIDUAL` $17.874$ and `C_OPERATOR` $3.823$. A paragraph is added to the evening-2 bullet
naming what the old range held instead — `:222` is the chunk table's prose preamble and `:225`
its budget line, and the range closed at `:235`, the `softmax` $n=4096$ row, **excluding the
two rows the sum's last two terms come from**. The memory row records what `:91-94` actually
names: the two `bf16_autocast` rows, one of which the record marks "**WRONG, +8.3 %,
optimistic**" — different constants under a different dtype, so the pre-repair pin cited a
constant the record itself refuses in place of one it confirms.

---

**Finding 34 · verse 05.6 · severity `repair` · `replacement_survives: true`**

> **flaw** — The third run is priced at a unit defined at a different shape.
> **mechanism** — M-3, V-17, M-8.
> **number** — `docs/PLAN.md:971 @ 99777ab` defines $1.524$ s as "one 150-step softmax control
> cell, $n=2048$, $s=64$". The `--s 32` run is charged the same $1.524$ s. The arithmetic
> 2x(1.524+4.0)+(1.524+4.0)=16.572 is right; the third term's shape is not.
> **required repair** — Price the `--s 32` run as `NOT MEASURED — needs the shape's own
> timing`, or bound it above by the s=64 cell with the direction printed.

**Repair applied — 05.6's Kill price line, its clause list, 05.1's split arithmetic and 05.14's
evening-3b row.** The finding's arithmetic quotes a pre-batch-1 draft (an earlier batch had
already moved the third term from $1.524$ to the capped cell's $1.884$ s), and its substance is
untouched by that: **both** units are defined at $s=64$ and the third run is at $s=32$.
`READ docs/PLAN.md:971 @ 99777ab` reads "one 150-step softmax control cell, $n=2048$, $s=64$";
`:978` reads "the capped cell | $1.884$ s | READ", at the same shipped $s$; and
`READ docs/PLAN.md:967 @ 99777ab` carries the rule the record obeys and this book had broken —
"the increment was measured at the shipped $s=64$ and **never scaled from a smaller $s$**
(M-3)". Both branches of the required repair are taken, because either alone leaves a hole: the
third run's **own price** is now `NOT MEASURED — needs the $s=32$ cell's own timing`, and the
$16.93$ s figure is carried as $2\times(1.524+4.0)=11.05$ s `[FITTED]` **plus**
$1.884+4.0=5.88$ s `[ASSUMED — the $s=64$ capped cell used as a ceiling on the $s=32$ cell]`,
with the direction printed and named as asserted rather than derived. The assumption is not
left standing on its own word: 05.6's Kill gains **clause (v)** — the `--s 32` run's journalled
`secs` exceeds $1.884$ s — which audits the ceiling on the same run at $0$ extra cost and, if
it fires, reprices evening 3b from the measured cell before evening 3c is scheduled. The class
propagates to 05.1's split arithmetic, 05.14's evening-3b row and the cheapest-first kill
table.

---

**Finding 35 · verse 05.6 · severity `repair` · `replacement_survives: false`**

> **flaw** — The If-killed replacement's Evidence is mis-attributed; the cited lines state a
> non-zero floor, not bitwise equality, and say nothing about CPU.
> **mechanism** — P-2, V-17.
> **number** — `V17K_RULINGS.md:373-376` is Ruling 2a's floor rule: `delta_nrmse =
> |eval_nrmse^(a) - eval_nrmse^(b)|` over an identical-seed repeat "on the certified device
> under the ruled regime (warn_only=True)". The bitwise sentence is at :381-382 — "a flag-OFF
> identical-seed repeat during pricing came back bitwise identical on eval_nrmse" — and names
> no device. The replacement's premise ("on CPU, where the record read flag-OFF identical-seed
> repeats bitwise") is unsupported at the pin.
> **required repair** — Re-pin to V17K_RULINGS.md:381-382, delete "on CPU" unless a CPU reading
> is produced, and tag the CPU pair `NOT MEASURED — needs the evening-3 CPU run`.

**Repair applied — 05.6's If-killed clause (i), and the replacement re-derived on a different
object rather than re-pinned.** The required repair is executed in full: RUN
`git show 99777ab:V17K_RULINGS.md | sed -n '374,379p'` confirms `:373-376` is Ruling 2a's floor
rule, a **non-zero quantity to be measured** on the certified device under `warn_only=True`;
RUN `sed -n '381,382p'` confirms the bitwise sentence sits there and **names no device**; the
premise "on CPU" is deleted and the CPU pair is tagged
`NOT MEASURED — needs the evening-3 CPU run`. But re-pinning alone leaves the V-9 defect the
finding's `replacement_survives: false` records, and it is repaired rather than papered over:
the replacement was `torch.equal` on `eval_nrmse` over an identical-seed pair — **clause (i)
verbatim**, same instrument, same number, moved to another device — so a repair that broke the
script's seeding would kill the replacement with the verse. The replacement in force is on a
**different object, by a different instrument, at a different number**: the identity-bind
residuals emitted through `identity_bind(kind, s, …)` at `:506` under `for s in (8, S)` at
`:508`, a code path the training loop never enters. Kill: either bind row reads `residual`
$\ge10^{-14}$ (one order above the larger of the two the un-repaired script produced), or the
`mutations` dict is absent. **Planted negative, RUN-populated and firing:** the eight mutation
residuals in `results/v15_r1.jsonl` — $0.9749459015140511$, $0.9165274652308163$, $1.0$,
$0.48449311856985267$ at $s=8$ and $1.5715211921402767$, $1.9148407189594014$, $1.0$,
$0.7215251981895894$ at $s=64$ — every one fourteen orders above the bar. **True negative:** the
two unmutated residuals, $6.661338147750939\times10^{-16}$ and $7.549516567451064\times10^{-15}$.
Price: $0$ additional GPU-s, so the link is strictly cheaper than the one it replaces and
strictly more decisive, separating a broken bind from a broken clock, which the eval-cell pair
cannot do.

---

**Finding 36 · verse 05.7 · severity `repair` · `replacement_survives: true`**

> **flaw** — The planted negative the kill depends on has no evening and no price in the
> schedule.
> **mechanism** — D-6, M-8, V-15.
> **number** — The verse charges the interleaved run 4x1.884+4x1.524+4.0=17.632 s and adds
> "plus the planted-negative run below on a later evening" — a second 17.63 s, plus
> 8x1.884+8x1.524+4.0=31.26 s if N doubles to 16. Neither appears in 05.14's table nor in its
> <=46.21 total.
> **required repair** — Schedule the blocked-order plant with an evening number and add 17.63 s
> (and the conditional 31.26 s) to 05.14's arithmetic.

**Repair applied — 05.7's Kill price line and a new table of evenings 6–8 in 05.14.** The
blocked-order plant is **evening 7** at $4\times1.884+4\times1.524+4.0=17.63$ s DERIVED; the
conditional $N\to16$ re-run is **evening 8** at $8\times1.884+8\times1.524+4.0=31.26$ s DERIVED
(the verse had written "evening 7" for the doubling, which would have put the plant and its own
escalation in one sitting). Neither can join evening 3: $17.63+17.63=35.26$ s is above the
frozen $20.0$ cap. 05.14 now carries evenings 6, 7 and 8 in a table beside the first five, with
each one's verification number, and its arithmetic is stated twice so neither figure can stand
in for the other: $\le69.02$ GPU-s is **the first five evenings only**, and the book's **full
priced programme** is $69.02+523.9+17.63=\mathbf{610.55}$ GPU-s, $641.81$ if evening 8 is
triggered. The $\le46.21$ the finding quotes was already withdrawn in batch 1 and replaced by
$\le69.02$; the defect the finding names survived that replacement unchanged, because the new
total omitted the plant exactly as the old one had. 05.14's Terminal and the cheapest-first
kill table carry all three figures.

---

**Finding 37 · verse 05.8 · severity `repair` · `replacement_survives: true`**

> **flaw** — The stated identity for frac_gate_annihilated does not match the code, so the
> frozen kill number 1/(ns) is false as written.
> **mechanism** — P-7, V-8.
> **number** — `scripts/v15_r1.py:386` computes
> `frac_gate_annihilated=float((~fin).double().mean())` with `fin = torch.isfinite(lg)` at :380,
> over `lg` already restricted to `[:, live]` (:364, :371). The denominator is n*|live|, not
> n*s, and `~fin` counts nan and +inf as well as -inf. The verse asserts f_ann = 1 - |F|/(ns)
> with F = {k : m_k > 0}.
> **required repair** — State the denominator as n*|live| with |live| printed, and either
> restrict the identity to lg finite-vs--inf or add a clause that any nan/+inf entry is itself
> a kill.

**Repair applied — 05.8's Statement, Hypotheses, clause (B) and a new clause (C).** The
denominator is corrected to $n\,|{\rm live}|$ and $|{\rm live}|$ is printed, with its value
derived at the pin rather than asserted: RUN `sed -n '697,698p' scripts/v15_r1.py` reads
`head = S - 1 - T_STAR` and `live = list(range(head + 1, S))`, so
$|{\rm live}| = S-1-{\rm head} = T^\star$, which is $\mathbf{2}$ at the module defaults $S=64$,
$T^\star=2$ — **not** $64$. The magnitude of the error is printed with it: reading the quantum
as $1/(ns)$ would put a single annihilated gate at $1/(64n)$ where the code produces $1/(2n)$, a
factor of $\mathbf{32}$ at the shipped geometry. Because $T^\star$ becomes a flag under 05.6,
$|{\rm live}|$ is required on **every row that quotes $\hat f_{\rm ann}$**, not stated once.
Both branches of the required repair are taken: the identity is restricted to the finite-versus-
$-\infty$ reading, **and** clause **(C)** is added — any traced position with `lg` equal to
`nan` or `+inf` is a kill of the cell, not an annihilation, and the cell's $\hat f_{\rm ann}$ is
withdrawn rather than read. Clause (C)'s planted negative is field-scoped and RUN: tokenising
`"<field>": <NaN|Infinity|-Infinity>` over `results/v17k_r4_retake.jsonl` gives `lambda_hat`
$\to$ `-Infinity` $131$, `lambda_hat_0step` $\to$ `-Infinity` $8$, `dyn_range_bound` $\to$
`Infinity` $30$, `dyn_range_bound_0step` $\to$ `Infinity` $12$, `exp_scan` $\to$ `NaN` $2$ — so
both $\hat\lambda$ fields read $0$ on the clause today, and the selector is shown to be a
selector rather than a tautology by the $44$ `Infinity` and $2$ `NaN` tokens it must ignore in
three neighbouring columns, which a field-blind `grep` would have swept in.

---

**Finding 38 · verse 05.9 · severity `repair` · `replacement_survives: true`**

> **flaw** — The deletion breakdown sums to 351 and the attic class is off by one.
> **mechanism** — P-1.
> **number** — RUN `git diff --name-only --diff-filter=D 99777ab HEAD`: 350 paths total; root
> `.md` 244, `tests/**/*.py` 53, other `tests/` 2 (tests/deimos/DEIMOS_REPORT.md,
> tests/mars/MARS_REPORT_IT2.md), `.superpowers/` 14, `attic/` **34**, plus house-events.jsonl,
> scale/chase_struck_coverage.py, scale/doc_readings.py. The verse writes 35 attic/, giving
> 244+53+2+14+35+3 = 351.
> **required repair** — Correct 35 to 34 and print the sum 244+53+2+14+34+3 = 350.

**Repair applied — 05.9's deletion paragraph.** RUN
`git diff --name-only --diff-filter=D 99777ab HEAD | grep -c '^attic/'` reads $\mathbf{34}$, and
every one of the six class counts is re-run by the same command with its own `grep -c`. The sum
is now **printed** as a displayed equation, $244+53+2+14+34+3=\mathbf{350}$, which is the repair
that matters: the pre-repair breakdown did not add to its own total and no reader could have
caught it, because the addition was never written down. The two other `tests/` files and the
three loose paths are named in the same sentence so the reader can re-run the partition rather
than take it.

---

**Finding 39 · verse 05.9 · severity `repair` · `replacement_survives: true`**

> **flaw** — The instrument named for the 350-path count no longer produces it.
> **mechanism** — P-1.
> **number** — RUN `git diff --cached --name-only --diff-filter=D | wc -l` reads **0** — the
> deletions were committed as `c71527a`. RUN `git show --name-only --diff-filter=D
> --pretty=format: c71527a | grep -c .` reads 350.
> **required repair** — Re-cite the count as RUN `git diff --name-only --diff-filter=D 99777ab
> HEAD`, which is stable at the pin.

**Repair applied — 05.9's deletion paragraph, all three instruments re-run and recorded.** The
count is re-cited to `git diff --name-only --diff-filter=D 99777ab HEAD`, RUN **350**, and it is
the frozen instrument because it is what the kill's threshold is about — what is absent from
HEAD relative to the pin. The `--cached` form is RUN and reads $\mathbf{0}$ and is **struck** as
this verse's instrument: it measures an empty index and produces the count in no session after
the commit, P-1 with the producer alive but pointed at the wrong tree. The single-commit form
`git show --name-only --diff-filter=D --pretty=format: c71527a | grep -c .` is RUN and reads
**350** and is correct, but it is recorded as *a different question* — what `c71527a` deleted,
not what is absent from HEAD — and the two agree here only because no other commit between
`99777ab` and HEAD deletes a path. Naming that coincidence is the point: the two numbers would
diverge silently on the next deleting commit, and the range form is the one that stays right.

---

**Finding 40 · verse 05.10 · severity `repair` · `replacement_survives: true`**

> **flaw** — The planted negative is a construction: the rule reads row 1 RED because the row's
> producer field is empty, which is a property of writing the field, not a measurement.
> **mechanism** — V-10, V-15.
> **number** — "Planted negative: row 1 as it stands at the pin — a slot
> (`FLOOR_TRAIN_ABS_DLOSS`) with an owner (\"the floor node\") but **no cell id** and no frozen
> acceptance (`READ V17K_RULINGS.md:163-168`) — the rule reads it RED today." The pin verifies
> (:163-168 is the slot with the floor node as owner), but a rule keyed on field-emptiness
> fires on emptiness by definition.
> **required repair** — Supply a plant with a filled producer field that the rule must still
> refuse — e.g. a slot naming a cell id absent from the (once written) frozen list — and show
> the refusal.

**Repair applied — 05.10's Gate-0 rule gains a second clause, and its plant is replaced.** The
rule now reads GREEN only when the named `file:line` **emits the slot's own value**, and a row
whose named line is real but emits something else is RED on the second clause — "it is the
second clause that does the work, because a filled field is the easy half". The old plant is
struck as a plant (kept only as that row's reading) and named for what it was: a rule keyed on
emptiness fires on emptiness by definition, V-10 and V-15, and demonstrates nothing about the
rule's power. The plant in force is **row 2's own pre-repair pin**, `ceq/compat.py:50-52` — a
filled producer field naming a file and lines that exist, which this book carried until batch 2
replaced it. The refusal is shown, not asserted: RUN
`git show 99777ab:ceq/compat.py | sed -n '48,53p'` reads "Step \"3b\" runs / `python -m
ceq.hf.smoke` in a SUBPROCESS, which gets a fresh interpreter and no / alias -- which is why
K-COMPAT runs BEFORE it", and RUN
`git show 99777ab:ceq/compat.py | sed -n '50,52p' | grep -cin "lambda\|lrt\|beta"` reads
$\mathbf{0}$. One reading runs against this book and is printed rather than dropped: the same
pattern over the **whole** file reads $\mathbf{1}$, at `:44`,
"`tests/gate0/test_g16_lrt_pinned.py::test_the_notebook_computes_lambda_at_the_`" — the module
*names the test that checks the notebook computes $\Lambda$*, seven lines above the cited range,
which is why the pin looked plausible; naming the test is not emitting the value, and that is
exactly the distinction clause two draws. **True negative:** row 1's producer as filed here,
`scripts/k_noise_floor.py:379`, RUN reading `abs_delta_final_loss=abs(final_a - final_b)` — the
slot's own value on the named line — on which the rule must not fire, and does not.

---

**Finding 41 · verse 05.11 · severity `repair` · `replacement_survives: true`**

> **flaw** — The planted negative fires because a slot is visibly open — a property of the
> verse's own layout, not a measurement.
> **mechanism** — V-10, V-15.
> **number** — "the canon's default is filed as a **default**, with the ruling slot left open
> beside it, so the plant fires (the slot is visibly open) rather than being closed by
> construction."
> **required repair** — Supply a plant that a default-picking rule must refuse: a sixth
> 'default' written in the same table that silently resolves an open ruling in the direction
> that claims more, and show the precedence rule rejecting it.

**Repair applied — 05.11's Kill gains a second plant, written as a sixth row of the same
table.** The layout plant is named for what it is and struck: a slot's being open is a property
of how this book laid out its own table, and a rule that fires on visible openness fires on
layout by definition, V-10 and V-15. The plant in force is **row 6**, a sixth "default" on the
**same open ruling as row 1** — the clause-1 tail — with its ruling slot as open as the other
five, choosing the **one-sided** 95 % Clopper–Pearson at `CP-lower` $=0.5156$. The refusal is
arithmetic and RUN at the pin, not a judgement: `READ V20_R15_JOURNAL.md:1532-1534 @ 99777ab`
prints, for `arm_pl`, $12/16$ crossings at rate $0.7500$, "CP-lower two-sided 95 % $=0.4762$ —
**FAILS by** $0.0238$" and "CP-lower one-sided 95 % $=0.5156$ — **CLEARS**"; the bar is
`CP-lower > 0.5`, so the one-sided reading returns a winner with the R15 $+12$ scoreboard row
and the two-sided reading returns none; and Ruling 2a's precedence rule at
`READ V17K_RULINGS.md:319-320 @ 99777ab` reads "**which claims less** — the criterion that
licenses the weaker sentence wins ties", so row 6 is refused **on its content**, with both slots
equally open. That is the property the finding demanded: the plant is indistinguishable from the
five by layout, and the rule separates it by which sentence it licenses. **True negative:** row
5, a pure renaming of four states onto the verse's own fields, which licenses no sentence at
all, and on which neither the kill nor the precedence rule fires.

---

**Finding 42 · verse 05.12 · severity `repair` · `replacement_survives: true`**

> **flaw** — The residual formula and its planted negative give the same total by two
> incompatible tensor censuses, so kill (ii) cannot distinguish them.
> **mechanism** — M-8, P-7.
> **number** — Formula: $n_{\rm layers}(2(d+1)+3)$ — two width-$(d{+}1)$ tensors (weight+bias)
> and three scalars; at $d=512$, $8\cdot(2\cdot513+3)=8\cdot1029=8{,}232$. Plant,
> `V17K_RULINGS.md:243-244 @ 99777ab`: $4{,}806-4{,}769=37=2\cdot16+5$ at $d_{\rm model}=16$ —
> two width-$d$ tensors and **five** scalars. Both total 37 at $d=16$; the censuses differ by
> two named tensors. Kill (ii) is "the recount differs from the formula — an unnamed tensor
> exists".
> **required repair** — Print the tensor-by-tensor census once (m_head weight [1,d] + bias [1],
> theta_head weight [1,d] + bias [1], beta, qk, g) and show it giving 2(d+1)+3 = 2d+5,
> reconciling both decompositions; then kill (ii) is decidable.

**Repair applied — 05.12's Statement gains the seven-row census, and kill (ii) and its plant are
re-keyed onto it.** The census is printed once: `m_head` weight $[1,d]$, `m_head` bias $[1]$,
`theta_head` weight $[1,d]$, `theta_head` bias $[1]$, `beta`, `qk`, `g`, totalling
$\mathbf{2d+5}$ per layer, with $2(d+1)+3 = 2d+2+3 = 2d+5$ shown identically, so the two
decompositions differ only in whether the biases are counted with their weights or with the
switches — they are the same seven tensors. Both source readings are re-pinned to the lines that
carry them: `READ V17K_RULINGS.md:239-241 @ 99777ab` for the "two per-position heads … three
scalar switches" wording, `:243-244` for the $4{,}806-4{,}769=37=2\cdot16+5$ cross-check, and
`:237` for the $8{,}232$ row. Instances are printed at both geometries: $2\cdot16+5=37$ and
$8\cdot(2\cdot512+5)=8\cdot1{,}029=8{,}232$. Kill (ii) is now decidable **against the seven-row
table** — a recount naming six tensors or eight is the unnamed-tensor event — and its planted
negative is replaced, because the old plant was the very cross-check that could not distinguish
the censuses. The plant in force is a recount of the seven rows with `theta_head`'s bias
dropped: it totals $2d+4$, reading $36$ at $d=16$ against the record's measured $\mathbf{37}$,
and it must fire. That is the one-parameter miss neither shorthand could catch, since
$2\cdot16+5$ and $2\cdot(16{+}1)+3$ both read $37$ whichever bias is missing from the reader's
inventory. **True negative:** the census as printed, reproducing $37$ and $8{,}232$.

---

**Finding 43 · verse 05.15 · severity `repair` · `replacement_survives: true`**

> **flaw** — B5's price carries the wrong evidence class against the source it READs.
> **mechanism** — P-1, P-2.
> **number** — The verse writes "$\approx34$ s per pair `[FITTED + RUN]`". `docs/PLAN.md:974 @
> 99777ab` tags the 34 s carried pair **DERIVED**; `[FITTED + RUN]` at :972 belongs to the
> $1.681$ s shape cell, a different unit.
> **required repair** — Re-tag B5's price DERIVED.

**Repair applied — 05.15's B5 row and a paragraph naming where the tag came from.** RUN
`git show 99777ab:docs/PLAN.md | sed -n '974p'` reads "one bed-cell pair, 8 seeds each | $29.6$
s in one invocation; **$34$ s** carried (two invocations, the conservative reading the record
used) | **DERIVED**", and `:972` reads "one 150-step shape cell on the softmax corner | $1.681$
s | `[FITTED + RUN]`". B5's price is re-tagged **DERIVED**, and the mechanism is stated rather
than silently corrected: attaching the stronger of two classes to the weaker of two units is
P-2, a number's home stated as somewhere it does not live, and P-1, a class with no producer for
the number it is attached to — $34$ s is an arithmetic doubling of a fitted cell price plus two
invocation fixed costs, and nothing in it was RUN.

---

**Finding 44 · verse 05.2 · severity `note` · `replacement_survives: true`**

> **flaw** — File mtime is used as production provenance for a load-bearing dating claim, but
> the tree's mtimes record a checkout, not a run.
> **mechanism** — P-1, P-2.
> **number** — RUN: `results/` holds 331 files over **5** distinct mtimes; 18 share the single
> second 2026-09-04 00:41 with identical nanosecond stamps (1788462678.5445809000).
> **required repair** — Tag the site-packages mtime evidence `[ASSUMED] — mtime is a filesystem
> write, not a producer record`, and carry the dated drift on the `numpy-2.4.6.dist-info`
> install record instead.

**Repair applied — 05.2's dating sentence, with the finding's own numbers corrected against it.**
The site-packages mtime reading is tagged **`[ASSUMED] — mtime is a filesystem write, not a
producer record`**, and the dated half of the drift moves to the install record:
`numpy-2.4.6.dist-info/` holds `INSTALLER`, `RECORD`, `WHEEL`, `METADATA`, `DELVEWHEEL` and
`entry_points.txt`, all at 2026-09-03 23:30 (RUN `ls --time-style=long-iso -l`), written once by
pip and never touched again. A stronger, **undated** half is added that the finding did not ask
for and that needs no clock at all: the directory's *name* is `numpy-2.4.6.dist-info` and RUN
`ls -d numpy-1.26.4.dist-info` reads `No such file or directory`, so the interpreter answering
`import numpy` carries a distribution the certificate never named — a fact about which files
exist, not about when. The finding's own numbers are corrected in print rather than adopted: RUN
over `results/`, $331$ files carry $\mathbf{176}$ distinct mtimes, not $5$; and the $18$
drift-window files carry **four** sub-millisecond stamps ($1788462678.5325522$ on one,
$\ldots5385008$ on eleven, $\ldots5439267$ on two, $\ldots5445810$ on four), not one identical
stamp. Eighteen files at four write instants is still a bulk handling event and still cannot be
read as eighteen measurements, so the finding's conclusion stands on a reading its own numbers
do not.

---

**Finding 45 · verse 05.3 · severity `note` · `replacement_survives: true`**

> **flaw** — Line drift on the .gitignore pin used to justify the venv route.
> **mechanism** — P-6.
> **number** — `.gitignore:12` is `.venv/`; `:13` is `venv/`.
> **required repair** — Cite `.gitignore:12-13 @ 99777ab`.

**Repair applied — 05.3's replacement 1.** RUN `git show 99777ab:.gitignore | sed -n '12,13p'`
reads `.venv/` then `venv/`. The pin is `READ .gitignore:12-13 @ 99777ab` and both patterns are
printed, with the consequence named: replacement 1 creates `.venv` at the repository root, which
is matched by `:12`, so the pre-repair pin `:13` alone named the *other* of the two patterns and
the citation did not cover the route it was there to justify.

---

**Finding 46 · verse 05.4 · severity `note` · `replacement_survives: true`**

> **flaw** — The claim that the certificate's box record holds "exactly these" seven fields is
> false.
> **mechanism** — P-1.
> **number** — RUN `json.load(open('results/k_cert_local.json'))['box']` returns 14 keys:
> device, torch, python, platform, threads, cuda_available, allow_tf32_matmul, allow_tf32_cudnn,
> cublas_workspace_config, device_name, capability, total_vram_bytes, n_gpus. The verse names 7.
> **required repair** — Replace "holds exactly these" with the 14-key RUN listing, or state the
> seven as the required subset.

**Repair applied — 05.4's `box` paragraph, and the count corrected against the finding.** Both
branches of the required repair are taken. The RUN listing is printed in the certificate's own
key order — and it is **13** keys, not the 14 the finding claims: the finding's own enumeration
names thirteen (`device`, `torch`, `python`, `platform`, `threads`, `cuda_available`,
`allow_tf32_matmul`, `allow_tf32_cudnn`, `cublas_workspace_config`, `device_name`, `capability`,
`total_vram_bytes`, `n_gpus`) and totals them at fourteen, so the finding reads one over its own
list and the worse reading is the one recorded. The seven are re-stated explicitly as the
**required subset**, which is what they always were. Two further defects the finding did not
reach are repaired with it. First, the verse wrote the ninth key as `CUBLAS_WORKSPACE_CONFIG`
where the emitted key is lower-case `cublas_workspace_config` (RUN
`git show 99777ab:scripts/k_cert.py | sed -n '674p'`), so a census grepping the verse's spelling
would have found nothing. Second, and the reason "exactly these" was unfixable as a fixed set at
any length: RUN `sed -n '669,680p'` shows `box_record` building **ten** keys unconditionally and
adding `capability`, `total_vram_bytes` and `n_gpus` only inside `if torch.cuda.is_available():`
— a CPU box writes a **10**-key record and the certified card a 13-key one, so the record's
length is a branch on `cuda_available` and no fixed count is assertable.

---

**Finding 47 · verse 05.5 · severity `note` · `replacement_survives: true`**

> **flaw** — Two instrument pins name lines that do not carry the referenced object.
> **mechanism** — P-6.
> **number** — `tests/gate0/test_g06_kcert.py` is named at `scripts/k_cert.py:9` and `:154`, not
> `:6-7` (:6-7 read "this one rather than overwriting it. The local box DECIDES; Kaggle
> REPRODUCES."). `cumsum_under_flag` is defined at `k_cert.py:644`, not `:642`.
> **required repair** — Re-pin to k_cert.py:9,154 and k_cert.py:644.

**Repair applied — 05.5's Kill line.** Both re-pins are RUN at the pin. `:9` reads "each half
carries its must-fire negative in `tests/gate0/test_g06_kcert.py`" and `:154` reads "can plant a
fabricated reading in each and watch the corresponding half of the certificate fire" — the two
lines that name the instrument and say what it does; `:6-7` names no test at all, and the verse
records what it does say. `:644` is `cumsum_under_flag`'s `def` line; the pre-repair `:642` is
two lines above it, inside the preceding comment block, so the pin named a comment as the
definition.

---

**Finding 48 · verse 05.5 · severity `note` · `replacement_survives: true`**

> **flaw** — Evening 2 writes to a separate filename, which bypasses the append-beside property
> the verse cites as the reason the old certificate survives.
> **mechanism** — L-G2.
> **number** — The command is `--out results/k_cert_local_e2.json`; the cited justification is
> `scripts/k_cert.py:3-5` ("writes a record keyed by `box` so a later ... run APPENDS a second
> certificate beside this one rather than overwriting it"). A different path is not an append.
> **required repair** — Either write evening 2 to `results/k_cert_local.json` and rely on the
> box keying, or state that the split file is deliberate and that the two certificates are
> merged by hand at evening 6.

**Repair applied — 05.5's two run bullets and a paragraph between them; and the finding reads
worse after repair than when filed.** The finding's first branch is **unusable and would have
destroyed evidence**, and the second is not merely available but mandatory. RUN
`git show 99777ab:scripts/k_cert.py | sed -n '925,928p'` reads `out = pathlib.Path(a.out)` /
`out.parent.mkdir(parents=True, exist_ok=True)` /
`out.write_text(json.dumps(cert, indent=2), encoding="utf-8")` — an unconditional `write_text`
on whatever `--out` names. RUN `grep -c "json.load("` on the same file reads $\mathbf{1}$, and
that one call is `json.loads(a.probe)` at `:700`, a command-line string: **the script never
reads an existing certificate**, holds no `boxes` collection and has no merge path anywhere. So
"APPENDS" at `:3-5` describes the *convention* of one box writing to one filename, not code, and
is P-3, a stale claim never retracted; writing evening 2 **or evening 6** to
`results/k_cert_local.json` would overwrite the `ab5b485` certificate, which L-G2 forbids
outright. Evening 6's `--out` is therefore moved to `results/k_cert_local_e6.json` as well —
before repair it wrote to `results/k_cert_local.json` on exactly the property the script does
not have — and `results/k_cert_local.json` is never written again by any evening of this book.
The three certificates are read **side by side by path** and are never merged into one file: a
merger needs a reader that does not exist and writing one is code the standing rule forbids,
so the merge the finding's second branch proposes is
`NOT MEASURED — needs a certificate merger, which this canon will not build`. The acceptance
table now names which file is the measured side of each row.

---

**Batch 3 summary.** 16 findings filed — 11 `repair`, 5 `note`; **16 applied, 0 left OPEN**.
Fourteen verses and tables were rewritten: 05.1 (the split arithmetic's class), 05.2 (the dating
sentence), 05.3 (replacement 1's pin), 05.4 (the `box` paragraph), 05.5 (the evening-2 and
evening-6 bullets, the append paragraph, the across-stack and memory acceptance rows, the Kill's
instrument and plant pins), 05.6 (the Kill's clause list and price, the If-killed replacement for
clause (i)), 05.7 (the Kill's price and the plant's schedule), 05.8 (the Statement's identity,
the Hypotheses, clause (B), a new clause (C)), 05.9 (the deletion paragraph), 05.10 (the Gate-0
rule's second clause and its plant), 05.11 (the Kill's second plant), 05.12 (the seven-row
tensor census and kill (ii)'s plant), 05.14 (the evenings 6–8 table, the 3b and 3c rows, the
totals, the Terminal) and 05.15 (B5's class), plus the cheapest-first kill table. One replacement
that died with its verse was **re-derived on a different object rather than renamed**: 05.6's
clause-(i) route, which moves from `torch.equal` on `eval_nrmse` at another device — clause (i)
verbatim — to the identity-bind residuals at a $10^{-14}$ bar, on a code path the training loop
never enters, with eight RUN-populated mutation residuals spanning $0.4845\ldots1.9148$ as its
planted negative and two unmutated residuals as its true negative, at $0$ additional GPU-s. Two
planted negatives that fired by construction were replaced by plants with **filled** fields that
a rule must still refuse: 05.10's, which moves from an empty producer field to
`ceq/compat.py:50-52`, a real file and real lines that emit no $\Lambda$; and 05.11's, which
moves from a visibly open slot to a sixth default on the same open ruling, refused by Ruling 2a
on its content. Three numbers moved **against this book** and are recorded at their worse
reading: `scripts/k_cert.py` **overwrites** and never appends, so the split certificate filenames
are mandatory and evening 6's `--out` had to move too; the `box` record holds **13** keys on a
CUDA box and **10** on a CPU box, so no fixed count is assertable and the finding's 14 is one
over its own list; and `results/` carries **176** distinct mtimes, not 5, with the 18
drift-window files at four stamps rather than one. One constant is struck from this book by name
and appears nowhere else in it: $35$ (05.9's `attic/` deletion count, which made its own
breakdown sum to $351$ against a total of $350$).
