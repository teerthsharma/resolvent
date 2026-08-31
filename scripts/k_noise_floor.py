"""RULING 1 -- the training noise floor HARNESS. The measurement runs on
Kaggle, not here (`V17_GPU_QUEUE.md`: "it bounds *training* claims, and
training runs on Kaggle. Measuring it locally would bound the wrong device.
Not a deciding number."). This file builds the harness, proves it on CPU at a
tiny shape, and hands it to the notebook -- exactly `V17_GPU_QUEUE.md`'s
standing rule for this item.

WHAT THE MEASUREMENT IS (`V17K_RULINGS.md` RULING 1, verbatim). Two training
chunks from an IDENTICAL SEED and identical configuration, run under the
regime training will actually use: CUDA,
`torch.use_deterministic_algorithms(True, warn_only=True)` -- strict mode is
not executable in the backward, because autograd differentiates `cumprod`
with `cumsum`, which has no deterministic CUDA kernel (`COSTS.md` section 1.6
measures this directly: the arm's gradient RAISES under strict mode with
`cumsum_cuda_kernel does not have a deterministic implementation`). The
`|Delta|` between the two chunks' final losses is the noise floor -- the
amount by which two runs that SHOULD be identical actually differ. Every
training claim in v17-K is then held to that floor, never to bitwise equality
(L-TOL, as amended: bitwise for replay + forward; the measured floor for
training).

RULING 2a GENERALISES THIS. "A quantity's noise floor is measured in that
quantity's units from the identical-seed pair" -- `|Delta|` final loss is one
instance, not the whole rule. The SAME pair, read differently, also yields a
per-parameter `|Delta|` keyed by name (`per_parameter_deltas`, wired into the
one call as `delta_param_names=`) -- the round's immediate need is `δ_β,i`
per layer, `smprime`'s `beta`, which is RULING 2a's own pinning criterion
`|β_i,final − 1| ≤ 5·δ_β,i` in β's own units. Nothing about this is a beta
special case: any parameter name gets its floor from the same two chunks,
with no new run and no new harness.

THE ONE CALL. `measure_noise_floor(out_dir_a=..., out_dir_b=..., seed=...,
steps=..., batch=..., seq=..., hidden_size=..., n_layers=..., n_heads=...,
device="cuda", delta_param_names=None, **overrides)` runs both chunks through
`ceq.hf.train.train()`, sets and restores the determinism regime around them,
and returns the one JSON record `COSTS.md` section 4 ingests as
`FLOOR_TRAIN_ABS_DLOSS` / `FLOOR_CHUNK_SPEC` / `FLOOR_BETA_DELTA`. Nothing
else for the notebook to assemble.

THREE REFUSALS, ALL LOAD-BEARING:

  1. `assert_identical_pair` -- a `|Delta|` from two chunks that differ in
     seed, shape, step count, data or regime is a COMPARISON, not a floor
     (this repo has struck 15 vacuous controls for exactly that class of
     error). It compares each chunk's OWN recorded provenance -- what
     `train()` actually did, read back off its `run_record.json` -- not the
     caller's arguments, so a silent divergence (operator resolving
     differently, device falling back, a short run) is still caught even
     though the public call only accepts one seed/shape for both chunks.
  2. `assert_not_self_comparison` -- tells a genuine `|Delta| = 0` apart from
     comparing one chunk to itself. Two structural checks, both on DISK (the
     same reason `tests/gate0/helpers.py::load_state` reads a checkpoint back
     off disk rather than trusting the live objects a run left in memory):
     `out_dir_a` and `out_dir_b` must be different directories, and both
     `run_record.json` files must postdate the call that is asking about
     them (a stale, leftover directory is a self-comparison in disguise).
     The two chunks' final losses are then read FROM THOSE SAME ON-DISK
     records, keyed only by `out_dir`, rather than from the in-memory
     objects `train()` happened to return -- so a `final_b` that quietly
     read `record_a` again cannot compile.
  3. `assert_matching_parameters` -- RULING 2a's own requirement: the
     per-parameter comparison is only over the two checkpoints' SHARED name
     set at MATCHED shapes, and a mismatch raises naming every offending key
     rather than silently comparing over `set(a) & set(b)`. Reused by
     `per_parameter_deltas`, which also repeats refusal 2's directory-equality
     half on its own (so it stays safe to call standalone on any two
     checkpoints, not only from inside this one call) -- a genuine
     per-parameter `|Delta| = 0` and a self-compared one are told apart the
     same way the loss-space one is: the self-comparison path raises before
     a number is ever produced, so a returned 0 can only be the former.

NOT A DECIDING NUMBER, NOT A RESEARCH READING (L-LEAN). This module trains no
bed, scores no cell, forms no verdict. `measurement_validity` in the return is
`"FLOOR"` only when `device == "cuda"` -- the certified device RULING 1 and
`V17_GPU_QUEUE.md` both name. Everything else, in practice this file's own
CPU proof, comes back tagged `"HARNESS_TEST_ONLY"` so it cannot be cited as
the floor by mistake.
"""
from __future__ import annotations

import os

#: Same reasoning and the same setting as `scripts/k_cert.py`: BEFORE any
#: torch import and before any cuBLAS handle exists, so the run's regime and
#: the bar's certification regime are the same flag at the same setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import pathlib                                                     # noqa: E402
import sys                                                         # noqa: E402
import time                                                        # noqa: E402

import torch                                                       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq.hf.train import train                                     # noqa: E402
from scripts.k_cert import box_record, git                         # noqa: E402

SCHEMA = "k_noise_floor/1"

#: Slack on the "freshness" comparison in `assert_not_self_comparison`,
#: between `time.time()` (Python's wall clock) and `os.path.getmtime()` (the
#: filesystem's, which on Windows/NTFS is not guaranteed to agree with the
#: former down to the microsecond). Two files written milliseconds after
#: `t_call_start` is captured have, in practice, occasionally come back with
#: an mtime a few milliseconds BELOW it -- clock/filesystem quantization, not
#: staleness. A genuinely stale, reused directory is older by seconds to
#: weeks, not milliseconds, so this slack costs the check nothing it exists
#: to catch.
_FRESHNESS_SLACK_S = 0.5

#: What two chunks must share for a `|Delta|` between them to be a FLOOR
#: (RULING 1: "two identical-seed chunks") rather than a comparison. Every
#: field the ruling itself names -- "seed, shape, step count, data or
#: regime" -- and no more.
_MUST_MATCH = ("seed", "hidden_size", "n_layers", "n_heads", "seq", "batch",
              "vocab_size", "steps", "data_path", "operator", "device",
              "deterministic_algorithms", "warn_only")


def assert_identical_pair(spec_a: dict, spec_b: dict) -> None:
    """Raise unless every field in `_MUST_MATCH` agrees between two chunk
    specs. Pure and standalone so it can be proven wrong before it is trusted
    (`tests/gate0/test_g15_noise_floor.py`)."""
    diffs = {k: (spec_a[k], spec_b[k]) for k in _MUST_MATCH if spec_a[k] != spec_b[k]}
    if diffs:
        raise ValueError(
            "not an identical pair -- {} field(s) differ: {}. A |Delta| "
            "measured from two different configurations is a comparison, "
            "not a floor (V17K_RULINGS.md RULING 1).".format(len(diffs), diffs))


def _read_run_record(out_dir) -> tuple[dict, float]:
    """A chunk's `run_record.json`, read back off disk, plus that file's
    mtime. Both the self-comparison check and the final loss used for
    `|Delta|` go through this -- never through the in-memory dict `train()`
    happened to return -- so the number is bound to the directory on disk,
    not to which Python variable a caller wrote it into."""
    p = os.path.join(out_dir, "run_record.json")
    with open(p, encoding="utf-8") as fh:
        record = json.load(fh)
    return record, os.path.getmtime(p)


def assert_not_self_comparison(out_dir_a, out_dir_b, t_call_start: float) -> dict:
    """Tell a real `|Delta| = 0` apart from comparing one chunk to itself.

    Two failure modes, both checked on disk:

      1. SAME DIRECTORY -- the second `train()` call overwrote the first
         chunk's own record, so there is only ever one run on disk and any
         `|Delta|` computed from it is 0 by construction, not measurement.
      2. STALE REUSE -- one of the two directories already held a
         `run_record.json` from an earlier, unrelated invocation, and this
         call never actually wrote a fresh one into it (a caller pointing the
         harness at a leftover directory, or a bug that silently skips the
         second `train()` call). `t_call_start`, captured before either chunk
         ran, is the line: a genuine run of THIS call leaves both mtimes at
         or after it.

    Returns the evidence (paths + mtimes) so it travels in the JSON record
    rather than living only in a passed assertion.
    """
    abs_a, abs_b = os.path.abspath(out_dir_a), os.path.abspath(out_dir_b)
    if abs_a == abs_b:
        raise ValueError(
            "out_dir_a and out_dir_b are the same directory ({}) -- this "
            "would compare one chunk to itself, not two".format(abs_a))
    _, mtime_a = _read_run_record(out_dir_a)
    _, mtime_b = _read_run_record(out_dir_b)
    cutoff = t_call_start - _FRESHNESS_SLACK_S
    stale = out_dir_a if mtime_a < cutoff else (
        out_dir_b if mtime_b < cutoff else None)
    if stale is not None:
        raise ValueError(
            "run_record.json in {} predates this call (mtime {} < call start "
            "{}) -- it was not freshly trained by this invocation, and a "
            "floor measured against a stale directory is a self-comparison "
            "in disguise".format(stale, min(mtime_a, mtime_b), t_call_start))
    return dict(out_dir_a=abs_a, out_dir_b=abs_b,
               run_record_mtime_a=mtime_a, run_record_mtime_b=mtime_b,
               t_call_start=t_call_start)


def _set_regime() -> dict:
    """`use_deterministic_algorithms(True, warn_only=True)`, RULING 1's
    regime -- read back rather than assumed, so the JSON record carries what
    is ACTUALLY in force, not what was requested."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
               cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def assert_matching_parameters(params_a: dict, params_b: dict) -> None:
    """RULING 2a's guard on the general primitive below: a per-parameter
    `|Delta|` keyed by name is only meaningful if both checkpoints name the
    SAME parameters at the SAME shapes. Raises naming every mismatch rather
    than silently comparing over `set(a) & set(b)` -- a silent intersection
    would drop exactly the parameter a caller most needed to see was missing.
    """
    names_a, names_b = set(params_a), set(params_b)
    if names_a != names_b:
        raise ValueError(
            "parameter name sets differ -- only in A: {}; only in B: {}. A "
            "per-parameter floor over a silently intersected key set is not "
            "a floor.".format(sorted(names_a - names_b), sorted(names_b - names_a)))
    bad_shapes = {n: (tuple(params_a[n].shape), tuple(params_b[n].shape))
                 for n in names_a if params_a[n].shape != params_b[n].shape}
    if bad_shapes:
        raise ValueError("parameter shape mismatch: {}".format(bad_shapes))


def per_parameter_deltas(out_dir_a, out_dir_b, names=None) -> dict:
    """RULING 2a, THE GENERAL PRIMITIVE: `|Delta|`, elementwise, keyed by
    parameter name, read from the SAME identical-seed pair `measure_noise_
    floor` already trained -- no new run, any quantity, not a beta special
    case. "A quantity's noise floor is measured in that quantity's units
    from the identical-seed pair" (`V17K_RULINGS.md` RULING 2a) -- this is
    that rule applied to whatever `names` asks for.

    `names`:
      * a STRING -- a substring filter over every parameter name in the
        checkpoint (e.g. `"self_attn.beta"` picks up every layer's beta
        without the caller enumerating layer indices).
      * an iterable of exact names -- each one required to exist; an unknown
        name raises rather than being silently dropped.
      * `None` (default) -- every parameter.

    SELF-COMPARISON GUARD, same shape as `assert_not_self_comparison`'s first
    check, one level down: `out_dir_a == out_dir_b` would read one
    checkpoint's parameters against themselves and report 0 for every name,
    for a reason that has nothing to do with reproducibility. Checked BEFORE
    either checkpoint is loaded. When called from `measure_noise_floor`, the
    pair has already passed the fuller staleness check too
    (`assert_not_self_comparison`); this function repeats only the cheap half
    so it stays safe to call standalone, on any two checkpoint directories,
    without a new harness.
    """
    if os.path.abspath(out_dir_a) == os.path.abspath(out_dir_b):
        raise ValueError(
            "out_dir_a and out_dir_b are the same directory -- this would "
            "compare a checkpoint's parameters to themselves, not two "
            "independent runs'")

    from ceq.hf.modeling_ceq import CEQForCausalLM
    params_a = dict(CEQForCausalLM.from_pretrained(out_dir_a).named_parameters())
    params_b = dict(CEQForCausalLM.from_pretrained(out_dir_b).named_parameters())
    assert_matching_parameters(params_a, params_b)

    if names is None:
        wanted = sorted(params_a)
    elif isinstance(names, str):
        wanted = sorted(n for n in params_a if names in n)
    else:
        wanted = list(names)
        missing = [n for n in wanted if n not in params_a]
        if missing:
            raise ValueError(
                "requested parameter name(s) not in either checkpoint: {}"
                .format(missing))

    out = {}
    for n in wanted:
        d = (params_a[n].detach().double() - params_b[n].detach().double()).abs()
        out[n] = dict(shape=list(params_a[n].shape),
                     delta=d.tolist(),
                     max_abs_delta=float(d.max()) if d.numel() else 0.0,
                     mean_abs_delta=float(d.mean()) if d.numel() else 0.0)
    return out


def _floor_label(device: str) -> str:
    """`"FLOOR"` only on the certified device RULING 1 and `V17_GPU_QUEUE.md`
    both name. Anything else comes back tagged so it cannot be cited as the
    floor by mistake -- factored out so the label is unit-tested against both
    branches, not just asserted in the negative on a CPU-only run."""
    return "FLOOR" if device == "cuda" else (
        "HARNESS_TEST_ONLY -- NOT THE FLOOR (device={!r} != 'cuda', "
        "V17_GPU_QUEUE.md: the measurement runs on Kaggle)".format(device))


def measure_noise_floor(*, out_dir_a, out_dir_b, seed, steps, batch, seq,
                        hidden_size, n_layers, n_heads, device="cuda",
                        vocab_size=256, data_path=None, delta_param_names=None,
                        **overrides) -> dict:
    """THE ONE CALL. Run two identical-seed training chunks through
    `ceq.hf.train.train()`, hold RULING 1's determinism regime around them,
    refuse a non-identical or self-compared pair, and return the JSON record
    `COSTS.md` section 4 ingests.

    `out_dir_a` and `out_dir_b` must be two distinct, unused directories.
    `resume_from` is never passed -- both chunks train FROM SCRATCH under the
    same seed; `train()`'s own resume-bitwise guarantee (RULING 1's class B1)
    is a different claim and is not what this measures.

    `delta_param_names` (RULING 2a, zero extra runtime cost: the two
    checkpoints already exist on disk). `None` (default) skips the
    checkpoint reload entirely and `result["per_parameter_delta"]` is `None`.
    Given a name filter -- e.g. `"self_attn.beta"` for the round's immediate
    need, `δ_β,i` per layer -- `per_parameter_deltas(out_dir_a, out_dir_b,
    names=delta_param_names)` runs against this SAME pair and its result is
    attached keyed by parameter name. See `per_parameter_deltas` for the
    general primitive and its own guards.
    """
    if steps < 1:
        raise ValueError("measure_noise_floor needs steps >= 1; a floor "
                         "needs at least one completed training step")
    if os.path.abspath(out_dir_a) == os.path.abspath(out_dir_b):
        raise ValueError("out_dir_a and out_dir_b must be different "
                         "directories -- the same directory would compare "
                         "one chunk to itself")

    prior = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    t_call_start = time.time()
    regime = _set_regime()
    try:
        record_a = train(out_dir=out_dir_a, steps=steps, batch=batch, seq=seq,
                         hidden_size=hidden_size, n_layers=n_layers,
                         n_heads=n_heads, device=device, vocab_size=vocab_size,
                         data_path=data_path, seed=seed, resume_from=None,
                         save_every=0, **overrides)
        record_b = train(out_dir=out_dir_b, steps=steps, batch=batch, seq=seq,
                         hidden_size=hidden_size, n_layers=n_layers,
                         n_heads=n_heads, device=device, vocab_size=vocab_size,
                         data_path=data_path, seed=seed, resume_from=None,
                         save_every=0, **overrides)
    finally:
        torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])

    data_fp = data_path or "<default corpus>"

    def spec(record):
        return dict(seed=seed, hidden_size=hidden_size, n_layers=n_layers,
                   n_heads=n_heads, seq=seq, batch=batch,
                   vocab_size=vocab_size, steps=record["steps"],
                   data_path=data_fp, operator=record["operator"],
                   device=record["device"],
                   deterministic_algorithms=regime["deterministic_algorithms"],
                   warn_only=regime["warn_only"])

    assert_identical_pair(spec(record_a), spec(record_b))
    self_check = assert_not_self_comparison(out_dir_a, out_dir_b, t_call_start)

    # The number ITSELF comes off disk, keyed only by out_dir -- see
    # `_read_run_record`'s docstring for why this matters more than it looks.
    disk_a, _ = _read_run_record(out_dir_a)
    disk_b, _ = _read_run_record(out_dir_b)
    final_a, final_b = disk_a["losses"][-1], disk_b["losses"][-1]
    same_len = len(disk_a["losses"]) == len(disk_b["losses"])

    per_parameter_delta = (
        per_parameter_deltas(out_dir_a, out_dir_b, names=delta_param_names)
        if delta_param_names is not None else None)

    return dict(
        schema=SCHEMA,
        measurement="R1_TRAINING_NOISE_FLOOR",
        measurement_validity=_floor_label(device),
        seed=seed,
        config=dict(hidden_size=hidden_size, n_layers=n_layers,
                   n_heads=n_heads, seq=seq, batch=batch,
                   vocab_size=vocab_size, steps=steps, data_path=data_fp,
                   operator=disk_a["operator"], overrides=dict(overrides)),
        regime=regime,
        box=box_record(device),
        step_count=record_a["steps"],
        chunk_a=dict(out_dir=self_check["out_dir_a"], final_loss=final_a,
                    n_losses=len(disk_a["losses"]),
                    run_record_mtime=self_check["run_record_mtime_a"]),
        chunk_b=dict(out_dir=self_check["out_dir_b"], final_loss=final_b,
                    n_losses=len(disk_b["losses"]),
                    run_record_mtime=self_check["run_record_mtime_b"]),
        abs_delta_final_loss=abs(final_a - final_b),
        trajectory_bitwise_identical=same_len and disk_a["losses"] == disk_b["losses"],
        per_parameter_delta=per_parameter_delta,
        identical_pair_check="PASSED",
        self_comparison_check=self_check,
        git=dict(head=git("rev-parse", "HEAD"), porcelain=git("status", "--porcelain")),
        elapsed_s=round(time.time() - t_call_start, 3),
    )


# ============================================================== CLI, for parity
# with scripts/k_cert.py and scripts/k_cost.py; the notebook calls
# measure_noise_floor() directly and does not need this.

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "k_noise_floor.json"))
    ap.add_argument("--out-dir-a", default=str(ROOT / "results" / "_nf_chunk_a"))
    ap.add_argument("--out-dir-b", default=str(ROOT / "results" / "_nf_chunk_b"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--operator", default=None,
                    help="e.g. smprime; forwarded to train() as an override")
    ap.add_argument("--delta-param-filter", default=None,
                    help="RULING 2a: substring match over checkpoint "
                         "parameter names, e.g. 'self_attn.beta' -- attaches "
                         "per_parameter_delta to the record. Omit to skip it")
    a = ap.parse_args(argv)

    overrides = {"operator": a.operator} if a.operator else {}
    rec = measure_noise_floor(
        out_dir_a=a.out_dir_a, out_dir_b=a.out_dir_b, seed=a.seed,
        steps=a.steps, batch=a.batch, seq=a.seq, hidden_size=a.hidden_size,
        n_layers=a.n_layers, n_heads=a.n_heads, vocab_size=a.vocab_size,
        data_path=a.data_path, device=a.device,
        delta_param_names=a.delta_param_filter, **overrides)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("{}  |Delta| final loss = {:.6e}  ({})".format(
        rec["measurement_validity"], rec["abs_delta_final_loss"], out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
