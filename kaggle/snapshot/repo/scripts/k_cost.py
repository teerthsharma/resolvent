"""G0.7 K-COST -- the chunk table, the weekly quota, and the failure ledger.

READS `results/k_cert_local.json` AND FITS NOTHING OF ITS OWN. Every step cost
below comes from a law `scripts/k_cert.py` refitted on this box in this run,
with its R^2 carried through; the two numbers this file measures directly --
the LM chunk shape's step cost and the checkpoint write -- are measured HERE
because `k_cert`'s ladder is the research-cell ladder and the Q3 chunk is a
different shape. Nothing is inherited: `V16_PRICING.md`'s and `TRAINING.md`'s
figures are quoted only as the things these are compared against (L-TIME).

THE THREE OUTPUTS, and what each is actually a claim about:

  * THE CHUNK TABLE. How many steps fit in a chunk of <= 11 h, at every
    candidate shape, FROM THE REFITTED LAW. `--chunk-hours` is 11 of the 12 h
    session cap; the 1 h margin is DERIVED in section 1 from measured
    quantities plus two named assumptions, not chosen.
  * THE WEEKLY QUOTA. 30 GPU-h against 11 h chunks is not 2.7 chunks: the
    remainder is a chunk too, and a chunk that runs out of quota mid-way costs
    the same as a chunk that is killed. Priced both ways.
  * THE FAILURE LEDGER. Four kill points, each with its recovery path and its
    cost in LOST GPU-MINUTES. Priced TWICE: once against the periodic-
    checkpoint interface as it stands in the WORKING TREE (`save_every`,
    `latest_checkpoint`, two alternating slots), and once against the version
    at `git HEAD`, which has neither. The difference between those two columns
    is the whole value of that node's work and is stated as a number.

NOTHING TRAINS AS A RESEARCH READING (L-LEAN). The LM timing runs on
`torch.randint` token ids used for their SHAPE only; no loss is reported, no
corpus is scored, no verdict is formed. The gradient is reported ONLY as
finite/not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq import sizing                                              # noqa: E402
from scripts.k_cert import (CELL_NS, GIB, MIN_TIMED_SECONDS, TIMED,  # noqa: E402
                            WARMUP, predict_secs, steps_in_hours)

#: The Kaggle session cap and the weekly quota, both from `TRAINING.md:142`
#: which cites `README.md:537-544`. `[INHERITED]` -- they are the platform's
#: published limits, not measurements, and re-measuring them is not possible
#: from this box.
SESSION_CAP_H = 12.0
WEEKLY_QUOTA_H = 30.0
WORKING_DIR_BYTES = 20_000_000_000          # 20 GB, decimal, TRAINING.md:142

#: The chunk cap the contract names. Section 1 derives the margin it leaves.
CHUNK_H = 11.0

#: The Q3 chunk shape: `ceq.hf.train.DEFAULTS`, which is what
#: `kaggle/ceq_v17k.ipynb` cell 17 trains and what `V17_G02_G03_CHECKPOINT.md`
#: sized the checkpoint against.
LM_SHAPE = dict(hidden_size=512, n_layers=8, n_heads=8, seq=512, vocab_size=256)
LM_BATCH = 8

#: The devices the chunk table is read on. The local box DECIDES; the two
#: Kaggle cards are what it must be reproduced on, and their budgets come from
#: `ceq/sizing.py::GPUS` -- a module constant, cited, not re-derived here.
KAGGLE_GPUS = ("T4-16GB",)


def gib(b):
    return b / GIB


# ------------------------------------------------------------- measurements

def _median_steps(step_fn, label):
    """Median seconds per step over a window bounded in BOTH steps and seconds,
    for `scripts/k_cert.py`'s measured reason: a window shorter than a few
    seconds reads a laptop GPU that is still ramping its clock."""
    per = []
    while True:
        t0 = time.perf_counter()
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per.append(time.perf_counter() - t0)
        if len(per) >= WARMUP + TIMED and sum(per[WARMUP:]) >= MIN_TIMED_SECONDS:
            break
    body = per[WARMUP:]
    return dict(label=label, median=float(statistics.median(body)), steps=len(per),
                spread=float(max(body) - min(body)))


def measure_lm_step(device="cuda") -> dict:
    """Seconds per (forward + backward + `AdamW.step`) at the Q3 chunk shape.

    Built through `ceq.hf.train.build`, so it is the shipped model and not a
    second copy of it. Random token ids: step cost is a function of shape, and
    no loss value is read out of this loop.
    """
    from ceq.hf import train as T

    torch.manual_seed(0)
    model = T.build(**LM_SHAPE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, LM_SHAPE["vocab_size"],
                      (LM_BATCH, LM_SHAPE["seq"]), device=device)

    def step():
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    r = _median_steps(step, "lm_chunk_shape")
    #: L-LEAN: the ONLY thing read off a gradient here.
    r["grad_finite"] = bool(all(torch.isfinite(p.grad).all()
                                for p in model.parameters() if p.grad is not None))
    r["n_params"] = sum(p.numel() for p in model.parameters())
    r["peak_bytes"] = float(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0
    del model, opt, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return r


def measure_checkpoint_write(device="cuda") -> dict:
    """Seconds and bytes for ONE complete checkpoint at the Q3 chunk shape.

    Calls `ceq.hf.train`'s OWN saver when the working tree offers one, so the
    number prices the shipped path including its atomic rename; falls back to
    the `save_pretrained` + `torch.save` pair `git HEAD` performs, and says
    which it measured. The checkpoint BYTES are re-counted off the filesystem
    rather than inherited.
    """
    from ceq.hf import train as T

    torch.manual_seed(0)
    model = T.build(**LM_SHAPE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, LM_SHAPE["vocab_size"],
                      (LM_BATCH, LM_SHAPE["seq"]), device=device)
    model(input_ids=x, labels=x).loss.backward()
    opt.step()                                  # so the moments are real tensors
    gen = torch.Generator().manual_seed(1)

    tmp = tempfile.mkdtemp(prefix="k_cost_ckpt_")
    out = os.path.join(tmp, "chunk")
    atomic = hasattr(T, "_save_checkpoint")
    try:
        t0 = time.perf_counter()
        if atomic:
            T._save_checkpoint(model, opt, gen, 1, out)
        else:
            os.makedirs(out, exist_ok=True)
            model.save_pretrained(out)
            torch.save(dict(optimizer=opt.state_dict(), step=1,
                            torch_rng_state=torch.get_rng_state(),
                            data_gen_state=gen.get_state()),
                       os.path.join(out, "trainer_state.pt"))
        secs = time.perf_counter() - t0
        files = {f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    finally:
        del model, opt, x
        if device == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(seconds=float(secs), bytes=sum(files.values()), files=files,
                path="ceq.hf.train._save_checkpoint (atomic, working tree)" if atomic
                     else "save_pretrained + torch.save (git HEAD, NOT atomic)",
                atomic=atomic)


def measure_startup() -> dict:
    """The part of a session's fixed cost this box can actually measure.

    `import torch` is already paid by the time this runs, so it is measured in
    a CHILD, which is also the only honest way to time an import.
    """
    import subprocess
    code = ("import time;t=time.perf_counter();import torch;"
            "import transformers;print(time.perf_counter()-t)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        imp = float(r.stdout.strip().splitlines()[-1])
    except Exception:
        imp = float("nan")
    t0 = time.perf_counter()
    from ceq.hf import train as T
    T.preflight(**LM_SHAPE, batch=LM_BATCH, gpu="T4-16GB")
    return dict(import_seconds=imp, preflight_seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------ the model

def chunk_row(name, secs_per_step, hours, law=None):
    n_steps = int(hours * 3600.0 / secs_per_step)
    return dict(shape=name, secs_per_step=secs_per_step, chunk_hours=hours,
                steps_per_chunk=n_steps,
                law=None if law is None else "exp(%.4f)*n^%.4f R^2 %.6f"
                                             % (law["log_a"], law["b"], law["r2"]))


def quota_arithmetic(chunk_h, weekly_h, per_chunk_overhead_h):
    """Chunks per week, and the two ways of counting them.

    A chunk costs `chunk_h` of GPU time PLUS its own fixed overhead -- the
    session's install, import, data build and final save all burn quota,
    because Kaggle bills the session and not the optimiser. Counting 30/11 and
    calling it 2.7 charges nothing for four startups.
    """
    billed = chunk_h + per_chunk_overhead_h
    whole = int(weekly_h // billed)
    left = weekly_h - whole * billed
    return dict(chunk_hours=chunk_h, overhead_hours=per_chunk_overhead_h,
                billed_hours_per_chunk=billed, weekly_quota_hours=weekly_h,
                whole_chunks=whole, remainder_hours=left,
                naive_chunks=weekly_h / chunk_h,
                remainder_is_a_chunk=left > per_chunk_overhead_h,
                remainder_training_hours=max(0.0, left - per_chunk_overhead_h))


def failure_ledger(*, secs_per_step, chunk_steps, save_every, ckpt_seconds,
                   startup_seconds, has_periodic):
    """Four kill points, each priced in LOST GPU-MINUTES, twice.

    `lost` is GPU time that produced nothing recoverable. It is NOT the same as
    time-to-recover: a resume also pays a startup, and that is a separate
    column because it is paid whether or not anything was lost.
    """
    r = secs_per_step
    chunk_s = chunk_steps * r

    def m(s):
        return round(s / 60.0, 2)

    period_s = save_every * r if save_every else None
    rows = []

    rows.append(dict(
        kill="during warm-up (install / import / data build, before step 0)",
        recovery="restart the chunk from `latest_checkpoint(prev_out_dir)`; the "
                 "previous chunk's directory was never opened for writing, so it "
                 "is untouched by construction",
        lost_min=m(startup_seconds), restart_min=m(startup_seconds),
        worst_lost_min=m(startup_seconds),
        same_without_periodic=True,
        note="No gradient step has run, so nothing trainable is lost either way. "
             "This is the one row the periodic-checkpoint work does not change."))

    if has_periodic:
        rows.append(dict(
            kill="mid-chunk (session cap, kernel death, quota exhaustion)",
            recovery="`latest_checkpoint(out_dir)` returns the newer of the two "
                     "slots; resume into a NEW out_dir",
            lost_min=m(period_s / 2.0), worst_lost_min=m(period_s),
            restart_min=m(startup_seconds), same_without_periodic=False,
            note="Uniform kill time over the interval gives an EXPECTED loss of "
                 "half a period; the worst case is one full period."))
    else:
        rows.append(dict(
            kill="mid-chunk (session cap, kernel death, quota exhaustion)",
            recovery="resume from the PREVIOUS chunk's out_dir -- there is no "
                     "mid-chunk state to return to",
            lost_min=m(chunk_s / 2.0), worst_lost_min=m(chunk_s),
            restart_min=m(startup_seconds), same_without_periodic=True,
            note="The whole elapsed chunk is lost. This is the row the periodic "
                 "interface exists to change."))

    rows.append(dict(
        kill="during a checkpoint write (`save_pretrained` shards, then the "
             "atomic `trainer_state.pt`)",
        recovery=("the slot being written is the one that dies; the OTHER slot is "
                  "complete and untouched, and `latest_checkpoint` skips the "
                  "half-written one because its `trainer_state.pt` does not load"
                  if has_periodic else
                  "there is only one directory and it is the one being written; a "
                  "kill inside `save_pretrained` leaves partial shards beside a "
                  "STALE trainer_state.pt, which loads -- so the directory reads "
                  "resumable and is not"),
        lost_min=m(period_s + ckpt_seconds) if has_periodic else m(chunk_s),
        worst_lost_min=m(period_s + ckpt_seconds) if has_periodic else m(chunk_s),
        restart_min=m(startup_seconds), same_without_periodic=not has_periodic,
        note=("Alternating slots are what make this recoverable: a single slot "
              "overwritten in place would be destroyed at exactly the moment it "
              "is being replaced."
              if has_periodic else
              "STRICTLY WORSE THAN ABSENT. `save_pretrained` is not atomic and "
              "runs BEFORE the state file, so the failure mode is a directory "
              "that looks resumable and is not.")))

    rows.append(dict(
        kill="during the FINAL save (end of chunk)",
        recovery=("the two periodic slots are deleted only AFTER `out_dir` is "
                  "verified complete, so the newest slot is still on disk and "
                  "`latest_checkpoint` returns it"
                  if has_periodic else
                  "nothing to fall back to inside this chunk; resume from the "
                  "previous chunk"),
        lost_min=m(period_s + ckpt_seconds) if has_periodic else m(chunk_s),
        worst_lost_min=m(period_s + ckpt_seconds) if has_periodic else m(chunk_s),
        restart_min=m(startup_seconds), same_without_periodic=not has_periodic,
        note=("The tail since the last slot, plus the write. The full chunk is "
              "NOT at risk." if has_periodic else
              "The entire chunk is lost at the last possible moment, which is "
              "also the moment the most has been invested in it.")))

    overhead = (chunk_steps // save_every) * ckpt_seconds if save_every else 0.0
    return dict(rows=rows, save_every=save_every, has_periodic=has_periodic,
                period_seconds=period_s, chunk_seconds=chunk_s,
                checkpoint_seconds=ckpt_seconds,
                periodic_overhead_seconds=overhead,
                periodic_overhead_frac=overhead / chunk_s if chunk_s else None)


def save_every_for_overhead(chunk_steps, secs_per_step, ckpt_seconds, frac):
    """`save_every` that keeps periodic-checkpoint overhead at or under `frac`.

    Solved rather than chosen: `k` checkpoints cost `k * ckpt_seconds` against a
    chunk of `chunk_steps * secs_per_step`, so `k <= frac * chunk_s /
    ckpt_seconds` and `save_every = ceil(chunk_steps / k)`.
    """
    chunk_s = chunk_steps * secs_per_step
    k = int(frac * chunk_s / ckpt_seconds)
    if k < 1:
        return None, 0
    return int(math.ceil(chunk_steps / k)), k


# ===================================================================== report

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cert", default=str(ROOT / "results" / "k_cert_local.json"))
    ap.add_argument("--chunk-hours", type=float, default=CHUNK_H)
    ap.add_argument("--overhead-frac", type=float, default=0.01,
                    help="periodic-checkpoint wall-clock overhead budget, used "
                         "to SOLVE save_every rather than pick it")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    cert = json.loads(pathlib.Path(a.cert).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = dict(cert_when=cert["when"], cert_box=cert["box"]["device_name"],
               cert_git=cert["git"]["head"], chunk_hours=a.chunk_hours)
    print("=== K-COST  from %s  (%s, %s) ==="
          % (a.cert, cert["box"]["device_name"], cert["when"]), flush=True)

    # --------------------------------------------------- 1. the margin, derived
    print("\n--- 1. THE MARGIN: why 11 h and not 12 ---", flush=True)
    lm = measure_lm_step(dev)
    ck = measure_checkpoint_write(dev)
    su = measure_startup()
    out.update(lm_step=lm, checkpoint=ck, startup=su)
    print("  [MEASURED] LM chunk shape %s batch %d: %.4f s/step over %d steps "
          "(spread %.3f s), %d params, peak %.3f GiB, grad finite %s"
          % (LM_SHAPE, LM_BATCH, lm["median"], lm["steps"], lm["spread"],
             lm["n_params"], gib(lm["peak_bytes"]), lm["grad_finite"]), flush=True)
    print("  [MEASURED] checkpoint write: %.3f s, %d B = %.4f GiB, via %s"
          % (ck["seconds"], ck["bytes"], gib(ck["bytes"]), ck["path"]), flush=True)
    print("  [MEASURED] import torch+transformers %.2f s; preflight %.3f s"
          % (su["import_seconds"], su["preflight_seconds"]), flush=True)

    #: The margin's floor is measured; the two parts this box cannot see are
    #: named as assumptions with the reason, not folded in silently.
    install_s = 180.0          # [ASSUMED]
    data_s = 300.0             # [ASSUMED]
    measured_floor = su["import_seconds"] + su["preflight_seconds"] + ck["seconds"] \
        + lm["median"]
    assumed = install_s + data_s
    margin_s = (SESSION_CAP_H - a.chunk_hours) * 3600.0
    out["margin"] = dict(margin_seconds=margin_s, measured_floor=measured_floor,
                         assumed_install=install_s, assumed_data=data_s,
                         covered_multiple=margin_s / (measured_floor + assumed))
    print("  margin = (%.1f - %.1f) h = %.0f s. It must cover:" % (
        SESSION_CAP_H, a.chunk_hours, margin_s), flush=True)
    print("    [MEASURED] import + preflight + final checkpoint + one ragged step "
          "= %.1f s" % measured_floor, flush=True)
    print("    [ASSUMED]  pip install of the wheel set = %.0f s -- not measurable "
          "from this box; a Kaggle image already carries torch, so this is the "
          "cost of the repo's own requirements only" % install_s, flush=True)
    print("    [ASSUMED]  dataset attach + corpus build = %.0f s -- G0.5 owns the "
          "real number; 300 s is the order the notebook's data cells imply" % data_s,
          flush=True)
    print("    => measured+assumed %.0f s, covered %.1fx by the margin"
          % (measured_floor + assumed, out["margin"]["covered_multiple"]), flush=True)

    # ------------------------------------------------------- 2. the chunk table
    print("\n--- 2. THE CHUNK TABLE: steps in <= %.0f h, from the REFITTED law ---"
          % a.chunk_hours, flush=True)
    laws = cert["throughput"]["laws"]
    mem = {(c["kind"], c["n"]): c for c in cert.get("memory", {}).get("cells", [])}
    #: THE RESIDENCY TEST IS ON RESERVED BYTES, NOT ALLOCATED. `k_cert`
    #: measured the workhorse arm at `n = 16384` ALLOCATING 6.296 GiB -- under
    #: an 8 GiB card -- while RESERVING 10.578 GiB from it, and paging. A table
    #: that asks "does the allocated peak fit the budget" would have called that
    #: shape resident and put a PCIe step cost in a chunk row. What a card must
    #: hold is what the caching allocator asks it for.
    #:
    #: Reserved on the LOCAL card is not reserved on a Kaggle card, so the T4
    #: column is a PROJECTION and is labelled one: the reserved/allocated ratio
    #: measured here in the RESIDENT regime only, applied to the allocated peak,
    #: against `ceq/sizing.py::GPUS["T4-16GB"]`'s budget.
    resident = [c for c in mem.values()
                if c.get("fits") and c.get("resident") and c.get("reserved")]
    infl = (max(c["reserved"] / c["peak"] for c in resident) if resident else 1.0)
    tot_t4, usable_t4 = sizing.GPUS["T4-16GB"]
    budget_t4 = tot_t4 * usable_t4
    out["reserved_inflation"] = infl
    print("  [MEASURED] worst reserved/allocated over the RESIDENT shapes on this "
          "box: %.3f -- the T4 column applies it to the allocated peak" % infl,
          flush=True)
    rows = []
    print("  %-12s %-7s %12s %14s %10s %10s %s"
          % ("arm", "n", "s/step", "steps/%.0fh" % a.chunk_hours,
             "alloc GiB", "proj resv", "T4-16GB (%.2f GiB budget)" % gib(budget_t4)),
          flush=True)
    for kind, law in sorted(laws.items()):
        for n in CELL_NS:
            r = predict_secs(law, n)
            c = mem.get((kind, n))
            peak = c["peak"] if c and c.get("fits") else None
            fitted = n in law.get("fitted_over", [])
            proj = None if peak is None else peak * infl
            rows.append(dict(arm=kind, n=n, secs_per_step=r,
                             steps_per_chunk=steps_in_hours(law, n, a.chunk_hours),
                             extrapolated=not fitted, peak=peak,
                             local_reserved=c.get("reserved") if c else None,
                             local_resident=c.get("resident") if c else None,
                             projected_reserved_t4=proj,
                             frac_of_t4=None if proj is None else proj / budget_t4,
                             fits_t4=None if proj is None else proj <= budget_t4))
            v = rows[-1]
            print("  %-12s %-7d %12.6f %14d %10s %10s %s%s"
                  % (kind, n, r, v["steps_per_chunk"],
                     "%.3f" % gib(peak) if peak else "n/a",
                     "%.3f" % gib(proj) if proj else "n/a",
                     "n/a" if proj is None else
                     ("yes (%.0f%% of budget)" % (100 * v["frac_of_t4"])
                      if v["fits_t4"] else "NO (%.0f%% of budget)" % (100 * v["frac_of_t4"])),
                     "" if fitted else "  [EXTRAPOLATED: law fitted over %s]"
                     % law.get("fitted_over")), flush=True)
    lm_steps = int(a.chunk_hours * 3600.0 / lm["median"])
    print("  %-12s %-7s %12.6f %14d %10.3f %10s %s"
          % ("LM (Q3)", "b=%d" % LM_BATCH, lm["median"], lm_steps,
             gib(lm["peak_bytes"]), "%.3f" % gib(lm["peak_bytes"] * infl),
             "measured directly, not fitted"), flush=True)
    out["chunk_table"] = dict(cells=rows, lm=dict(
        shape=LM_SHAPE, batch=LM_BATCH, secs_per_step=lm["median"],
        steps_per_chunk=lm_steps, tokens_per_chunk=lm_steps * LM_BATCH * LM_SHAPE["seq"]))
    print("  LM chunk tokens: %d steps x %d x %d = %d tokens"
          % (lm_steps, LM_BATCH, LM_SHAPE["seq"],
             lm_steps * LM_BATCH * LM_SHAPE["seq"]), flush=True)

    # ------------------------------------------------------- 3. weekly quota
    print("\n--- 3. THE WEEKLY QUOTA: %0.0f GPU-h ---" % WEEKLY_QUOTA_H, flush=True)
    oh_h = (measured_floor + assumed) / 3600.0
    q = quota_arithmetic(a.chunk_hours, WEEKLY_QUOTA_H, oh_h)
    out["quota"] = q
    print("  naive %.1f / %.1f = %.2f chunks -- WRONG, it charges nothing for "
          "startup" % (WEEKLY_QUOTA_H, a.chunk_hours, q["naive_chunks"]), flush=True)
    print("  billed per chunk = %.1f h training + %.3f h startup = %.3f h"
          % (a.chunk_hours, oh_h, q["billed_hours_per_chunk"]), flush=True)
    print("  => %d whole chunks + %.3f h remainder (%.3f h of it trainable)"
          % (q["whole_chunks"], q["remainder_hours"], q["remainder_training_hours"]),
          flush=True)
    weekly_lm = int((q["whole_chunks"] * a.chunk_hours
                     + q["remainder_training_hours"]) * 3600 / lm["median"])
    out["quota"]["lm_steps_per_week"] = weekly_lm
    out["quota"]["lm_tokens_per_week"] = weekly_lm * LM_BATCH * LM_SHAPE["seq"]
    print("  at the Q3 shape that is %d steps = %d tokens per week"
          % (weekly_lm, weekly_lm * LM_BATCH * LM_SHAPE["seq"]), flush=True)

    # ------------------------------------------------------ 4. failure ledger
    print("\n--- 4. THE FAILURE LEDGER ---", flush=True)
    from ceq.hf import train as T
    import inspect
    has_periodic = "save_every" in inspect.signature(T.train).parameters
    se, k = save_every_for_overhead(lm_steps, lm["median"], ck["seconds"],
                                    a.overhead_frac)
    print("  periodic-checkpoint interface in the WORKING TREE: %s "
          "(`save_every` in train(), `latest_checkpoint` %s)"
          % ("PRESENT" if has_periodic else "ABSENT",
             "present" if hasattr(T, "latest_checkpoint") else "absent"), flush=True)
    print("  save_every solved for <=%.0f%% overhead: %s steps (%d writes per chunk)"
          % (100 * a.overhead_frac, se, k), flush=True)
    #: The trade, printed rather than argued: the author picks the row, and the
    #: two columns that move are wall-clock overhead and worst-case loss.
    print("  %-10s %-12s %-14s %-16s %s"
          % ("overhead", "save_every", "writes/chunk", "worst lost GPU-min",
             "bytes written over the chunk"), flush=True)
    trade = []
    for f in (0.001, 0.005, 0.01, 0.05):
        s_e, kk = save_every_for_overhead(lm_steps, lm["median"], ck["seconds"], f)
        if s_e is None:
            continue
        trade.append(dict(overhead_frac=f, save_every=s_e, writes=kk,
                          worst_lost_min=s_e * lm["median"] / 60.0,
                          bytes_written=kk * ck["bytes"]))
        print("  %-10.1f%% %-12d %-14d %-16.2f %.1f GB"
              % (100 * f, s_e, kk, trade[-1]["worst_lost_min"],
                 trade[-1]["bytes_written"] / 1e9), flush=True)
    out["save_every_trade"] = trade
    out["save_every"] = dict(value=se, writes_per_chunk=k,
                             overhead_frac=a.overhead_frac,
                             interface_present=has_periodic)
    for label, hp in (("WITH periodic checkpoints (working tree)", True),
                      ("WITHOUT them (git HEAD)", False)):
        led = failure_ledger(secs_per_step=lm["median"], chunk_steps=lm_steps,
                             save_every=se if hp else 0, ckpt_seconds=ck["seconds"],
                             startup_seconds=measured_floor + assumed, has_periodic=hp)
        out["ledger_with" if hp else "ledger_without"] = led
        print("\n  %s:" % label, flush=True)
        for row in led["rows"]:
            print("    kill %-58s worst lost %8.2f GPU-min   restart %6.2f min"
                  % (row["kill"][:58], row["worst_lost_min"], row["restart_min"]),
                  flush=True)
        if hp:
            print("    periodic overhead: %d writes x %.3f s = %.1f s = %.3f%% of "
                  "the chunk" % (k, ck["seconds"], led["periodic_overhead_seconds"],
                                 100 * led["periodic_overhead_frac"]), flush=True)
    w = out["ledger_with"]["rows"][1]["worst_lost_min"]
    wo = out["ledger_without"]["rows"][1]["worst_lost_min"]
    out["periodic_value_gpu_min"] = wo - w
    print("\n  WHAT THE PERIODIC INTERFACE IS WORTH: a mid-chunk kill costs %.2f "
          "GPU-min with it and %.2f without -- %.0fx, %.1f GPU-min saved on one "
          "kill, %.1f%% of the weekly quota." % (
              w, wo, wo / w if w else float("inf"), wo - w,
              100 * (wo - w) / 60.0 / WEEKLY_QUOTA_H), flush=True)

    print("\n  disk: one checkpoint %.4f GiB; /kaggle/working %d B holds %d of them; "
          "the 4-directory steady state is %.2f%%"
          % (gib(ck["bytes"]), WORKING_DIR_BYTES,
             WORKING_DIR_BYTES // ck["bytes"],
             100 * 4 * ck["bytes"] / WORKING_DIR_BYTES), flush=True)

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("\nwrote %s" % a.json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
