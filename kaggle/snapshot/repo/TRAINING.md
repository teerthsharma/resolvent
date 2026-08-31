# TRAINING.md — training this model on free compute

Binding constraint: **no budget.** Nothing below needs a paid GPU, a paid API, or
a subscription.

Every number carries a tag. **[RUN]** was executed in producing this file, on
this box (Windows 11, Python 3.11.9, torch 2.5.1+cu121, CPU-only,
`torch.set_num_threads(2)`, `CUDA_VISIBLE_DEVICES=` empty). **[READ]** is quoted
from a file in this repository, with the path. **[ESTIMATED]** is arithmetic on
an assumption, and the assumption is named in the same line.

---

## 0. Three numbers this guide had to correct before it could be written

| number as briefed | what it actually is | evidence |
|---|---|---|
| "25,707,520 parameters" | that is `sizing.params()`, the ANALYTIC count. The model actually built is **25,728,000**. `ceq/sizing.py:132` counts the MLP as `d*f + f*d` with no bias, but `ceq/hf/modeling_ceq.py:419` uses `nn.Linear` at PyTorch's default `bias=True`. The gap is `8 * (2048 + 512) = 20,480`, exactly. | **[RUN]** built `T.build(**DEFAULTS)`, summed `p.numel()`: analytic 25,707,520, real 25,728,000, difference 20,480 |
| "2.45 GiB against 14.5 GiB" | **stale.** Current preflight returns **2.72 GiB against a 13.50 GiB budget**, headroom +10.78 GiB. `README.md:521` already says the old pair is stale. | **[RUN]** `preflight(gpu="T4-16GB", **DEFAULTS)` |
| "pivot_unsigned 2.166 s/step and 1720 MB, pivot_signed 3.350 s/step and 1855 MB, composed 7.473 s/step and 3150 MB" | **`2.166`, `3.350`, `1720 MB` and `1855 MB` do not exist anywhere in this repository.** Only `7.473 s/step` and `3149.98 MB` exist (`DONE.md:726-727`), and they are for a 4769-parameter probe arm at `n_train=8192` on CPU — not for the training path. "composed" is also not an arm name; the four arms are `("softmax", "pivot_signed", "pivot_unsigned", "windowed_signed")`. | `scale/m3_capability.py:65`; exhaustive repo search |

The measured arm wall-clocks that DO exist, `results/m3_capability.txt:902,926,950`,
all at `s=64 d=24 steps=150 n_train=8192 n_eval=512 seed=0`, 4769 params per arm:

| arm | wall clock | wall/150 |
|---|---|---|
| `softmax` | 95.123 s | 0.634 s/step |
| `pivot_unsigned` | 217.575 s (rerun 249.385 s) | 1.450 s/step (rerun 1.663) |
| `pivot_signed` | 300.856 s | 2.006 s/step |

Those are **capability-probe** arms, not the training path. `wall/150` includes
setup and eval, so read it as an upper bound on the step. **Do not plan a
training run off them.** Section 2 has the training-path numbers.

---

## 1. The parameter count, exactly

Shape knobs and their defaults, `ceq/hf/train.py:45-46` **[READ]**:

```python
DEFAULTS = dict(hidden_size=512, n_layers=8, n_heads=8, seq=512, batch=8,
                vocab_size=256)
```

`vocab_size=256` is a **byte-level** vocabulary — there is no tokenizer to train
or download, which is one reason this fits a free tier at all.

The formula, `ceq/sizing.py:128-140` **[READ]**:

```python
d, f = cfg.d_model, cfg.d_ff_mult * cfg.d_model      # d_ff_mult = 4
per_layer = (3*d*d + d*d) + (d*f + f*d) + 4*d
n  = cfg.n_layers * per_layer
n += cfg.vocab * d      # token embedding
n += cfg.seq * d        # learned positions
n += 2 * d              # final LayerNorm
n += cfg.vocab * d      # untied lm_head (tie_word_embeddings=False)
```

plus the `20,480` of MLP bias the formula omits, per section 0.

**[RUN]** exact counts, `tied_embeddings=False`, `vocab=256`, signed arm, T4-16GB:

| shape | hidden | L | H | seq | batch | params (analytic) | state GiB | act GiB | total GiB | fits T4 |
|---|---|---|---|---|---|---|---|---|---|---|
| resume-test shape | 16 | 1 | 2 | 8 | 2 | 11,488 | 0.0002 | 0.0000 | 0.0002 | yes |
| **smallest that trains at all** | 128 | 4 | 4 | 256 | 8 | **887,040** | 0.0149 | 0.1462 | 0.1610 | yes |
| **notebook DEFAULTS** | 512 | 8 | 8 | 512 | 8 | **25,707,520** analytic / **25,728,000** real | 0.4310 | 2.2841 | **2.7150** | yes |
| d512 L12 | 512 | 12 | 8 | 512 | 8 | 38,298,624 | 0.6420 | 3.4222 | 4.0642 | yes |
| d768 L12 | 768 | 12 | 12 | 512 | 8 | 85,759,488 | 1.4377 | 5.1294 | 6.5670 | yes |
| d512 L8 seq 1024 | 512 | 8 | 8 | 1024 | 8 | 25,969,664 | 0.4354 | 7.8831 | 8.3185 | yes |
| `CFG_300M` shape, batch **1** | 1024 | 24 | 16 | 2048 | 1 | 304,711,680 | 5.1081 | 21.7502 | 26.8583 | **no** |

The 11,488-parameter row is the shape `tests/chase/test_resume_checkpoint.py:21`
actually trains, so "trains at all" is not a guess. The 887,040 row is the
smallest shape worth a real run.

**Note the seq-1024 row.** Doubling `seq` at a fixed parameter count moved
activations 2.28 → 7.88 GiB, a 3.45× jump for a 2× sequence, because this
operator materialises `[B,H,S,S]` and autograd retains a measured 3.9 of them per
layer (`ceq/sizing.py:47-48`). **Sequence length, not parameter count, is what
runs you out of a free GPU.**

---

## 2. What actually fits, measured

**[RUN]** `ceq.hf.train.preflight(gpu="T4-16GB", **DEFAULTS)` → `ok = True`:

```
signed d=512 L=8 H=8 seq=512 batch=8: params 25,707,520 -> state 0.43 GiB +
activations 2.28 GiB = 2.72 GiB against a 13.50 GiB budget on T4-16GB.
Headroom +10.78 GiB.
  softmax control at the same shape: 1.06 GiB.
```

The 13.50 GiB budget is `15.0 GiB * 0.90` (`ceq/sizing.py:79`) — the 0.90 is the
fraction the allocator can hand out after the CUDA context and fragmentation.
The **operator arm costs 2.56× the softmax control** at this shape (2.72 / 1.06),
and that ratio is the thing to watch, not the absolute figure.

**[RUN]** training-path step cost on THIS CPU, local corpus, `steps=6`, wall/6
including model build and data load. **A CPU number. It is not a T4 number and
must not be quoted as one:**

| shape | real params | 6 steps, wall | s/step | loss[0] → loss[5] |
|---|---|---|---|---|
| d128 L4 H4 seq 256 b8 | **889,600** | 3.24 s | **0.540** | 5.6172 → 4.7543 |
| notebook DEFAULTS | **25,728,000** | 79.69 s | **13.281** | 5.6375 → 3.3271 |

Corpus `data/scan_addprim_jump_train.txt`, 2,579,619 bytes, so no download is in
the timing. `peak_bytes` reads 0 because CUDA peak stats do not exist on CPU.
Both real counts exceed `sizing.params()` by exactly the MLP bias: 889,600 −
887,040 = 2,560 = `4 * (512 + 128)`; 25,728,000 − 25,707,520 = 20,480 =
`8 * (2048 + 512)`. **Declared shortcut: 6 steps only, and wall/6 includes model
build, corpus load and the checkpoint write, so it overstates the steady-state
step.** For contrast, `house-events.jsonl:2087` records the same shape at **min
4454.2 ms/step, median 6799.9 ms/step at 20 torch threads** on this box, with the
median flagged by its own author as contaminated by concurrent load. Two threads
against twenty explains the direction of the gap.

**Which arm.** The training path's operator is selected by `CEQConfig.operator`,
default `sgate` at `rho=1.5, lam=0.10, hops=2` (`ceq/hf/configuration_ceq.py:39`),
with `signed` as the alternative. The four arms in section 0 are the
`scale/m3_capability.py` capability probe and are a different code path
entirely. The operator/control step ratio that IS measured for the training path
is `MEASURED_STEP_RATIO = {1024: 3.13, 2048: 5.49}`,
`tests/chase/test_scale_sizing.py:494` **[READ]** — measured on an RTX 4060
Laptop, **not** on a T4 or an A100.

---

## 3. The free options, with their real limits

**[READ]** `README.md:537-544`, fetched 2026-08-25 by an earlier pass. These are
repo-recorded, not re-fetched by this file — **re-check the two that decide the
plan (session cap, weekly quota) before committing a multi-week schedule.**

| tier | GPU | session cap | quota | persistent disk |
|---|---|---|---|---|
| **Kaggle Notebooks** | P100-16GB or 2×T4-16GB | **12 h** | **30 GPU-h/week, published** | **20 GB auto-saved `/kaggle/working`** |
| Colab free | not published | 12 h | **not published** | none native (Drive mount only) |
| Paperspace/Gradient free | Quadro M4000 | 6 h | not published | 5 GB |
| Lightning AI free | T4 / A10 | ~4 h restart cycle | 80 GPU-h/month | 50–100 GB (sources disagree) |
| SageMaker Studio Lab | T4 | 4 h / 24 h | 4 GPU-h/day | 15 GB — **closed to new signups** |
| HF Spaces ZeroGPU | RTX Pro 6000 48/96GB | **60 s per call** | 5 min/day free | ephemeral |

**Use Kaggle.** It is the only free tier with all three of: a published quota, a
12-hour session, and a working directory that survives a kill without mounting a
second product. Colab's unpublished quota cannot be planned against. ZeroGPU at
60 s per call cannot train anything regardless of the card.

One more free-tier fact that matters here: the Triton multi-zoom kernel needs
compute capability ≥ 8.0 and a T4 is sm_75, but **the training path is pure
torch and never touches Triton** (`tests/chase/test_ceq_hub_package.py::test_training_the_shipped_model_needs_no_triton_and_no_ampere`).
A free T4 runs this. **[READ]**

---

## 4. Resume is the whole plan, and it already works

Every free tier caps the session. Without resume, the largest trainable model is
whatever finishes in **one** uninterrupted session — a capability ceiling, not a
convenience gap. With resume it is whatever the weekly quota buys.

`ceq/hf/train.py::train(..., resume_from=None)`. `trainer_state.pt` carries
**exactly four keys** (`ceq/hf/train.py:269-271`) **[READ]**:

| key | contents |
|---|---|
| `optimizer` | `opt.state_dict()` — AdamW per-parameter moments |
| `step` | `start_step + steps` |
| `torch_rng_state` | `torch.get_rng_state()` |
| `data_gen_state` | `gen.get_state()` — batches are drawn by index from `gen`, so **the generator state IS the dataloader position** |

**`steps` counts steps ADDED, not a new total** (`ceq/hf/train.py:198`), and
`seed` is ignored on resume.

**[RUN]** `python -m pytest tests/chase/test_resume_checkpoint.py -q` →
**`3 passed, 2 warnings in 38.31s`**. The three are:

1. stitched N+N losses match an uninterrupted 2N run to `1e-5` relative;
2. **stitched weights are BITWISE identical** — `torch.equal` over every named
   parameter, zero tolerance;
3. the **must-fire control**: `AdamW.load_state_dict` is monkeypatched to a
   no-op so the moments are silently dropped, and the test REQUIRES the weights
   to diverge. Without it, a green (2) could mean the comparison is blind
   rather than the resume correct.

### Chunk size

**Do not compute this from a table. Measure it in the first session.** The two
T4 extrapolations already in this repo **disagree by 4–9×** (`house-events.jsonl:2087`:
6ND at 65 TFLOP/s and MFU 0.25–0.40 gives 24–39 ms/step; measured-CPU divided by
an unverified 20–40× speedup gives 111–223 ms/step). Neither is a T4
measurement. So:

```
run 50 steps with log_every=10, read seconds/step off the log  -> r
chunk_steps = floor(0.80 * session_cap_seconds / r)
```

The `0.80` leaves room for install, data load, checkpoint write and the ragged
end of a session. For Kaggle's 12 h cap that is `0.80 * 43200 = 34,560 s` of
training per chunk. **[ESTIMATED]** at the optimistic 24–39 ms/step that is
886k–1.44M steps per session; at the pessimistic 111–223 ms/step it is 155k–311k.
The range is why you measure `r` instead of trusting either end.

One calibration that IS a real T4 arithmetic, `house-events.jsonl:2091` **[READ]**:
training the 25.7M shape on one TinyStories epoch is `6ND = 7.2432e16` FLOPs,
which on a T4 at its quoted 65 TFLOP/s fp16 is **3.10 / 1.55 / 1.03 hours at MFU
10 / 20 / 30%** — one 12-hour session fits three epochs even at the pessimistic
MFU.

---

## 5. The honest ceiling

**Nothing in this project has ever trained above 3,652,096 parameters**
(`RESEARCH.md:25`; the other trained size is 3,319,296). `MODEL_CARD.md:59`:
"Nothing above 3.65M has ever run." **[READ]**

The gate the user set is 300M. `CFG_300M` is d 1024, L 24, H 16, seq 2048,
vocab 32000 untied = **304,711,680 parameters** **[RUN]**.

| | value |
|---|---|
| notebook default shape, real count | **25,728,000** **[RUN]** |
| as a fraction of the 300M gate | **8.44 %** (25,728,000 / 304,711,680) **[RUN]** |
| Chinchilla budget for that shape, 20 tok/non-embedding-param | **503,644,160 tokens** **[RUN]** |
| what the notebook ships: 2000 × 8 × 512 | **8,192,000 tokens = 1.627 %** of it **[RUN]** |

### What free compute can reach

**[READ]** `README.md:555-560`, MFU labelled and assumed, operator cost = control
× the measured 3.13× step ratio:

| | MFU 0.30 | MFU 0.15 |
|---|---|---|
| largest **operator** run finishing in ONE 12-h session | **37.8M** (d 512, L 12, 11.9 h) | **25.2M** (d 512, L 8, 10.6 h) |
| with resume, 4 weeks of Kaggle quota | 100.7M | 71.4M |
| 12 weeks | 177.0M | 127.4M |
| 26 weeks | 265.9M | 192.7M |
| memory ceiling only, free T4, no checkpointing | 308.3M | — |
| memory ceiling only, `grad_checkpoint=True` | 737.4M | — |

### What it cannot reach

**The 300M gate is not reachable free.** A matched pair at seq 1024 is 41.4
A100-h (control) + 129.7 A100-h (operator) = **171.1 A100-hours = 1,095 T4-hours
at MFU 0.30 = 37 weeks of Kaggle's entire free weekly quota spent on nothing
else**. At seq 2048 it is 304.0 A100-h, 65 weeks. **[READ]** `README.md:565-568`.

**Memory was never the binding constraint — a 300M model FITS a free T4 (ceiling
308.3M, 737.4M with checkpointing). Time is what stops it.**

So, stated plainly: a free-tier plan reaches **~37.8M in one session** and
**~265.9M in 26 weeks of uninterrupted quota at an optimistic MFU 0.30**. It
does **not** reach a 300M matched pair, and 26 weeks of a free weekly quota is
not a plan anyone should build on. The realistic free deliverable is the
**25.7M shape at a full Chinchilla budget**, which is 7.0× the highest this
project has ever trained and finishes inside one free Kaggle session — and it is
**8.44 % of the gate, so it does not settle the question the gate asks.** Any
document quoting a 25.7M result without the 300M gate beside it is overclaiming.

---

## 6. The exact commands

Fresh Kaggle or Colab notebook, GPU runtime selected. `python -u` throughout:
**an unbuffered redirect is why one run in this project died leaving two 0-byte
files** (`results/r3_it11_pivot_8192.log`, still 0 bytes on disk).

### 6.1 Install

```bash
pip -q install -U 'transformers==5.3.0' datasets huggingface_hub accelerate
python -u -c "import torch, platform; print('torch', torch.__version__, platform.system(), platform.machine()); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Pin `transformers==5.3.0` and not `>=5.0`. `requirements.txt:8-17` **[READ]**:
`ceq/hf/modeling_ceq.py` sets six private `PreTrainedModel` attributes that are
undocumented HF internals; any release including a patch can rename one, and the
notebook's `>=5.0` will happily install it.

### 6.2 Code

```bash
git clone -q https://github.com/<you>/<repo>.git ceq-project
cd ceq-project && python -u -c "from ceq.hf import train; print('ok', train.__file__)"
```

### 6.3 Preflight — refuses a shape that cannot fit, before anything downloads

```bash
python -u - <<'PY'
from ceq.hf import train as T
CONFIG = dict(T.DEFAULTS)                 # hidden 512, L8, H8, seq 512, batch 8
ok, msg = T.preflight(gpu="T4-16GB", grad_checkpoint=False, **CONFIG)
print(msg)
assert ok, "shape does not fit: lower seq or batch, or grad_checkpoint=True"
PY
```

On Kaggle's P100 pass `gpu="T4-16GB"` anyway — it is the more conservative
budget of the two and `sizing.GPUS` has no P100 entry.

### 6.4 Data — streamed, not downloaded

```bash
python -u - <<'PY'
import os
from ceq.hf import train as T
DATA = "/kaggle/working/corpus.txt"       # Colab: "corpus.txt"
if not os.path.exists(DATA):
    open(DATA, "w", encoding="utf-8").write(
        T.load_open_text("roneneldan/TinyStories", max_bytes=32*1024*1024))
print(os.path.getsize(DATA) / 1024**2, "MiB")
PY
```

### 6.5 Train — chunk 1

`OUT` must live on the **persistent** disk (`/kaggle/working`, auto-saved 20 GB)
or the session cap eats the checkpoint.

```bash
python -u - <<'PY' 2>&1 | tee -a /kaggle/working/train.log
from ceq.hf import train as T
r = T.train(out_dir="/kaggle/working/ceq-run-000", steps=20000, device="cuda",
            data_path="/kaggle/working/corpus.txt", gpu="T4-16GB",
            grad_checkpoint=False, log_every=50, **dict(T.DEFAULTS))
print("params", f"{r['n_params']:,}", "peak GiB", r["peak_bytes"]/1024**3)
print("start_step", r["start_step"], "steps added", r["steps"])
print("final loss", r["losses"][-1], "worst row L1", min(r["row_l1_min"]))
PY
```

Read `s/step` off the first 50-step log line and set `steps` for the next chunk
from section 4's rule. `log_every=50` is the default and it is what makes that
possible.

### 6.6 Resume — every chunk after the first

`steps` is **steps ADDED**. `out_dir` must be a NEW directory each chunk;
`resume_from` points at the previous one.

```bash
python -u - <<'PY' 2>&1 | tee -a /kaggle/working/train.log
from ceq.hf import train as T
PREV, NEXT = "/kaggle/working/ceq-run-000", "/kaggle/working/ceq-run-001"
r = T.train(out_dir=NEXT, steps=20000, device="cuda",
            data_path="/kaggle/working/corpus.txt", gpu="T4-16GB",
            resume_from=PREV, log_every=50, **dict(T.DEFAULTS))
print("resumed from step", r["start_step"], "-> now", r["start_step"] + r["steps"])
PY
```

Sanity check before trusting a chunk boundary — `start_step` must equal the
previous chunk's total, and `trainer_state.pt` must exist:

```bash
python -u -c "import torch,os; p='/kaggle/working/ceq-run-000/trainer_state.pt'; print(os.path.exists(p)); print(sorted(torch.load(p, map_location='cpu', weights_only=True).keys()))"
```

Expect `True` and `['data_gen_state', 'optimizer', 'step', 'torch_rng_state']`.

### 6.7 Verify it loads the way a downloader loads it

```bash
python -u - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM
OUT = "/kaggle/working/ceq-run-001"
assert {"configuration_ceq.py", "modeling_ceq.py"} <= set(os.listdir(OUT))
m = AutoModelForCausalLM.from_pretrained(OUT, trust_remote_code=True).to("cpu").eval()
assert not m.lm_head.weight.is_meta, "lm_head is on the meta device"
print(m.generate(torch.randint(0, 256, (1, 8)), max_new_tokens=16, do_sample=False))
PY
```

### 6.8 Save / push

Do this **at the end of every session**, not only at the end of the run.

```bash
python -u - <<'PY'
from huggingface_hub import login
from ceq.hf import train as T
login()                       # reads HF_TOKEN from the env or prompts
print(T.push("/kaggle/working/ceq-run-001", "<user>/<repo>", private=True))
PY
```

The token is read from the login prompt or `HF_TOKEN`. **Never write it into a
notebook cell** — a committed Kaggle notebook is a published artifact.

---

## 7. What this guide does not establish

- **No number here was measured on a T4 or a P100.** Every GPU figure is either
  `ceq/sizing.py`'s calibrated model (calibrated on an RTX 4060 Laptop, not on a
  T4) or an MFU-assumed estimate. The first session's measured `s/step` supersedes
  all of them.
- **The free-tier limits in section 3 are repo-recorded, dated 2026-08-25**, not
  re-fetched here. Session caps and quotas change.
- **`sizing.params()` is 20,480 low at this shape** and the same bias omission
  scales with `n_layers`; at `CFG_300M` it is `24 * (4096 + 1024) = 122,880`.
  Nothing in the memory conclusions moves (0.08 %), but every parameter count in
  this repo inherits it.
- **Parity with softmax is not established at any size.** At 3,319,296 matched
  parameters, 800 steps on TinyStories, signed val loss 2.0064 against softmax
  1.4282 = 1.405× (`colab/train_ceq.ipynb`, markdown cell 0). **Run the softmax
  control in the same session before believing any number this produces.**
- **`python -m pytest tests/ -q` has never completed** in this project
  (`MODEL_CARD.md:246`), so there is no total pass/fail count to quote. The
  resume test in section 4 was run individually and is green.
