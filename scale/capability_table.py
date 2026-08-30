"""Capability table v0 -- the negatives included, from artifacts that already exist.

    python -m scale.capability_table            # writes results/capability_table_v0.*
    python -m scale.capability_table --artifact ceq/hf_artifact

S3a. The table and the HuggingFace package are assembled from what is already
on disk: the completed M3 quintuple journal, the e-process, and the task
registry. Nothing here trains anything and nothing here uploads anything.

THE RULE THAT MAKES IT HONEST. Every cell prints its verdict, and where the
verdict is NO DIFFERENCE the table says NO DIFFERENCE. The headline contrast
`settled - twin` covers zero and is printed as such, with its interval, beside
the wins. A table that shows only wins is marketing.

THREE THINGS THIS FILE REFUSES TO DO, each because the repo has done it before:

1. IT DOES NOT NAME AN ARM THAT IS NOT IN THE ROW. `m3_synthetic_settled.
   verdict_of` returns the literal strings "SETTLED WINS" / "TWIN WINS", which
   is correct for the one contrast it was written for and wrong for every
   other. The shipped run log carries the consequence at
   `results/m3_quintuple.txt:56`:

       argmax    softmax    8  -0.118456  -0.134115  -0.102204       TWIN WINS

   -- a verdict naming `twin`, in a row that contains neither `twin` nor
   `settled`. That is the G3 shape (an arm reporting another arm's number or
   name). `_verdict` below takes the interval AND the two arm names.

2. IT DOES NOT MIX TWO ESTIMATORS UNDER ONE LABEL. Two families of interval
   for the same contrast exist in this repository today:

       results/m3_quintuple.txt:51   settled twin  -0.048587  +0.031557
       CHECKLIST.md:1167             settled twin  -0.042903  +0.031557

   The first is the Monte-Carlo percentile bootstrap at `B = 10000, seed = 0`
   that the run actually executed. The second is the EXACT percentile over all
   5**5 = 3125 paired resamples -- a legitimate and in fact better estimator at
   this seed count, since the resampling distribution is finite and needs no
   Monte Carlo at all. Both are defensible; only one of them was labelled, and
   both were labelled `B=10000`. This file computes the interval it prints,
   from the journal, and stamps the estimator name into the same row. See the
   `limits` paragraph for the one endpoint that belongs to neither.

3. IT DOES NOT LET `undecided` READ AS WEAK EVIDENCE. `eprocess.max_attainable(5)
   = 3.80169140625` against `THRESHOLD = 40.0`, and `MIN_T_MIXTURE = 13`. Five
   seeds cannot cross in either direction whatever the data say, so the word
   travels with the ceiling that makes it structural.

MUST-FIRE. `tests/chase/test_capability_table.py` plants a 0.5-NRMSE gap into a
copy of the journal, in each direction, and requires the same builder to print
`settled WINS` and then `twin WINS`. Without that, `NO DIFFERENCE` in the
shipped table would be indistinguishable from a verdict function that cannot
return anything else.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import itertools
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import eprocess as EP                                   # noqa: E402
from scale.m3_synthetic_settled import contrast                    # noqa: E402
from scale.m3_quintuple import task_of                             # noqa: E402
from scale.negation_scope import M3_TASKS                          # noqa: E402

JOURNAL = ROOT / "results" / "m3_quintuple_v2.jsonl"

#: Trained probe-arm tensors shipped alongside the card, written by
#: `scripts/export_hf_weights.py`. `render` reads this so the card cannot deny
#: weights the artifact directory actually contains. The denial was true when v0
#: was cut and became false at commit 0162bdd; the README on disk was corrected
#: by hand and the generator was not, so the next `write_artifact` call would
#: have reverted the correction. Read here rather than asserted, for the reason
#: `_weights_present` gives below.
WEIGHTS_MANIFEST = ROOT / "ceq" / "hf_artifact" / "weights" / "MANIFEST.json"


def shipped_weights(path=WEIGHTS_MANIFEST) -> list[dict]:
    """The VERIFIED entries of the weight manifest, or `[]` if there is none.

    Unverified entries are dropped rather than reported: the export policy
    deletes a checkpoint that fails its reload check, so an unverified entry
    means the manifest itself is mid-write or damaged, and a card must not
    advertise a tensor nothing has re-evaluated.
    """
    try:
        m = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [w for w in m.get("shipped", []) if w.get("verified")]


def _weights_present(seeds) -> int:
    """How many of THIS table's cells have trained tensors on disk.

    Counted rather than asserted, because the sentence "no per-cell weights
    exist" was true when it was written and stops being true the moment a unit
    is run by the patched runner -- and this table is published to HuggingFace,
    where a stale absence claim is a false statement about an artifact rather
    than a stale comment.
    """
    from scale.m3_quintuple import CELLS, weights_path, _key      # noqa: PLC0415
    n = 0
    for cell in CELLS:
        for sd in seeds:
            p = dict(cell=cell, k=0 if cell in ("softmax", "glance") else 8,
                     s=64, d=24, steps=150, n_train=8192, n_eval=512,
                     t_max=21, seed=sd, task="negation_scope")
            n += weights_path(_key(p)).exists()
    return n
OUT_MD = ROOT / "results" / "capability_table_v0.md"
OUT_JSON = ROOT / "results" / "capability_table_v0.json"

#: The build a plain cut writes. NOT v0: BOARD.md records v0 as frozen evidence
#: stamped `journal_commit 9629616`, cited as historical, and v1 as the current
#: build -- and it records the workaround, "OUT_MD/OUT_JSON hardcode the v0
#: paths and the plain command would clobber committed evidence", which left the
#: rule in a document and the landmine in the tool. Selecting the version is now
#: the tool's job, so cutting the table no longer means editing two constants by
#: hand first. v0 stays writable; it takes saying `--version v0`.
DEFAULT_VERSION = "v1"


def out_paths(version: str) -> tuple:
    """(markdown, json) for a table version. One place that spells the names."""
    return (ROOT / "results" / f"capability_table_{version}.md",
            ROOT / "results" / f"capability_table_{version}.json")

SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 10000
BOOT_SEED = 0

#: The section-5 credit bar. An arm at or above this did not beat the mean
#: predictor and is credited with nothing, whatever its contrast says.
PREDICT_THE_MEAN = 1.0

#: (arm, reference, note). Headline first. `glance` is included precisely
#: because its row is zero BY CONSTRUCTION: G3 binds it bitwise to `softmax` at
#: `t_max = 0`, so a table that could not print an exact structural zero and
#: label it as one would be repeating this project's own recorded defect.
CONTRASTS = (
    ("settled", "twin", "HEADLINE"),
    ("settled", "argmax", "HEADLINE CAVEAT -- argmax is the attribution control"),
    ("settled", "softmax", ""),
    ("twin", "softmax", ""),
    ("argmax", "softmax", ""),
    ("glance", "softmax",
     "G3 binds glance bitwise to softmax at t_max = 0: this row is zero BY "
     "CONSTRUCTION, not a measured tie"),
)

#: WHICH TASK THIS TABLE IS ABOUT, and how a reader checks it. This was once an
#: inference -- `scale/m3_quintuple.py` had no `--task` flag, so the task could
#: only be read off the producer. That stopped being true when `--task` was
#: registered on `m3_quintuple._argparser` (`choices=list(NS.M3_TASKS)`,
#: defaulting to `SHIPPED_TASK`), and the task became a property of each journal
#: ROW: `m3_quintuple._key` appends `_task<name>` for every non-shipped task and
#: `m3_quintuple.task_of` parses it back. `read_journal` below filters on that
#: parser, so an e3 row sharing the bucket cannot enter a negation-scope table.
#: The name is still pinned here because the table is about one corpus.
TASK = "negation_scope"
TASK_SOURCE = ("scale/m3_quintuple.py registers --task (m3_quintuple._argparser, "
               "choices=list(negation_scope.M3_TASKS), default SHIPPED_TASK) and "
               "stamps the task into the journal key; m3_quintuple.task_of reads "
               "it back, and a bare key means the shipped task")


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "UNTRACKED"


def read_journal(path=JOURNAL) -> dict:
    """`{cell: {seed: value_dict}}` from a bucket journal.

    The journal nests every measurement under `value`. Reading the top level
    only is the twelfth vacuous control this project shipped; the key is read
    explicitly here so a schema change raises instead of returning nothing.
    """
    rows: dict[str, dict[int, dict]] = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # ONE JOURNAL, MANY TASKS since `--task` landed. This table is a
        # `negation_scope` table (TASK above); an e3 row in the same bucket is a
        # different corpus and must not enter it. Filtered by the format's own
        # parser rather than by a second copy of the key grammar here.
        if task_of(r["key"]) != TASK:
            continue
        cell, _, tail = r["key"].partition("_")
        # The key travels with the value. A contrast row cites the journal
        # records it was computed from, and reconstructing that key from the
        # arm and the geometry would be a second copy of the grammar this
        # function is the sole parser of -- the P-6 shape.
        v = dict(r["value"])
        v["_journal_key"] = r["key"]
        rows.setdefault(cell, {})[int(tail.rpartition("_sd")[2])] = v
    return rows


def _geometry(path=JOURNAL) -> str:
    """`s64_d24_st150_ntr8192_nev512_b21`, from the first key that has it."""
    first = pathlib.Path(path).read_text(encoding="utf-8").splitlines()[0]
    key = json.loads(first)["key"]
    return "_".join(key.split("_")[2:-1])


def _verdict(ci_lo: float, ci_hi: float, arm: str, ref: str) -> str:
    """The verdict, as a pure function of the interval AND the two arm names.

    Strict at zero: an interval touching zero is not an interval excluding it
    (G6). `delta = NRMSE_ref - NRMSE_arm`, so a strictly positive interval means
    the ARM has the lower error.
    """
    if ci_lo > 0.0:
        return "{} WINS".format(arm)
    if ci_hi < 0.0:
        return "{} WINS".format(ref)
    return "NO DIFFERENCE"


def exact_percentile_ci(per_seed_delta) -> tuple:
    """The EXACT 95% percentile pair over all `n**n` paired resamples.

    At five seeds the paired bootstrap distribution is finite -- `5**5 = 3125`
    resamples, 126 distinct values -- so the percentile the Monte-Carlo draw
    estimates is computable outright, `3.2x` cheaper than `B = 10000` and with
    zero Monte-Carlo error. This is the family the prose deliverables adopted
    (`CHECKLIST.md:1239`, `DONE.md:1839-1862`); the table prints the
    Monte-Carlo family that the run executed. Both are stamped into every row
    so the two homes of a headline can be told apart by name rather than by
    guessing which one is a typo -- the P-1 that M-9 records.

    Percentile indices are `contrast()`'s, so the only difference between the
    two families is the resample set, never the rule applied to it.
    """
    n = len(per_seed_delta)
    reps = sorted(sum(c) / n for c in itertools.product(per_seed_delta, repeat=n))
    return (reps[int(0.025 * len(reps))],
            reps[min(len(reps) - 1, int(0.975 * len(reps)))])


def build(journal=JOURNAL, *, seeds=SEEDS, n_boot=N_BOOT,
          boot_seed=BOOT_SEED) -> dict:
    """The whole table as data. `render` turns it into prose; nothing else does."""
    journal = pathlib.Path(journal)
    rows = read_journal(journal)
    seeds = tuple(seeds)
    estimator = "paired percentile bootstrap B={} seed={}".format(n_boot, boot_seed)

    params = {rows[c][s]["n_params"] for c in rows for s in seeds if s in rows[c]}
    if len(params) != 1:
        raise ValueError("arms are not parameter-matched: {}".format(sorted(params)))

    arms = []
    for cell in sorted(rows):
        if not all(s in rows[cell] for s in seeds):
            continue
        ev = [rows[cell][s]["eval_nrmse"] for s in seeds]
        mean = statistics.fmean(ev)
        arms.append(dict(
            arm=cell, task=TASK, n_params=rows[cell][seeds[0]]["n_params"],
            per_seed=ev, nrmse_mean=mean, nrmse_sd=statistics.stdev(ev),
            marginal_lo=rows[cell][seeds[0]]["marg_lo"],
            marginal_hi=rows[cell][seeds[0]]["marg_hi"],
            nrmse0_eval_min=min(rows[cell][s]["nrmse0_eval"] for s in seeds),
            beats_predict_the_mean=mean < PREDICT_THE_MEAN,
            consequence_fidelity="NOT MEASURED",
        ))

    contrasts = []
    for arm, ref, note in CONTRASTS:
        if arm not in rows or ref not in rows:
            continue
        if not all(s in rows[arm] and s in rows[ref] for s in seeds):
            continue
        c = contrast([rows[ref][s]["eval_nrmse"] for s in seeds],
                     [rows[arm][s]["eval_nrmse"] for s in seeds],
                     n_boot=n_boot, seed=boot_seed)
        ex_lo, ex_hi = exact_percentile_ci(c["per_seed_delta"])
        contrasts.append(dict(
            arm=arm, ref=ref, note=note, delta=c["delta"],
            ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], n_boot=n_boot,
            boot_seed=boot_seed, n_seeds=len(seeds), estimator=estimator,
            per_seed_delta=c["per_seed_delta"],
            ci_exact_lo=ex_lo, ci_exact_hi=ex_hi,
            estimator_exact="exact percentile over all {}**{} = {} paired "
                            "resamples".format(len(seeds), len(seeds),
                                               len(seeds) ** len(seeds)),
            journal_keys=[rows[a][s_]["_journal_key"]
                          for a in (arm, ref) for s_ in seeds],
            seeds_favouring_arm=sum(1 for d in c["per_seed_delta"] if d > 0.0),
            verdict=_verdict(c["ci_lo"], c["ci_hi"], arm, ref),
        ))

    live = EP.live(journal, ref="twin", arm="settled")
    ceiling = EP.max_attainable(live["t"])
    ep = dict(live)
    ep.update(threshold=EP.THRESHOLD, min_t_mixture=EP.MIN_T_MIXTURE,
              min_t_single_arm=EP.MIN_T_SINGLE_ARM, clip_c=EP.CLIP_C, b=EP.B,
              alpha=EP.ALPHA, alpha_family=EP.ALPHA_FAMILY,
              n_directions=EP.N_DIRECTIONS, ceiling_here=ceiling)
    if live["decision"] is None and live["void"] is None:
        ep["text"] = (
            "{} -- STRUCTURAL, not a data outcome: max_attainable({}) = {!r} "
            "< THRESHOLD = {!r}, and MIN_T_MIXTURE = {}, so five seeds cannot "
            "cross in either direction whatever the data say."
            .format(live["text"], live["t"], ceiling, EP.THRESHOLD,
                    EP.MIN_T_MIXTURE))

    # Name the repository, not the machine. The absolute form recorded one
    # user's home directory, and cutting from a git worktree made it name a
    # temporary directory that is deleted with the worktree -- provenance no
    # other reader can resolve. A journal outside the tree has no repo-relative
    # name, so it is reported as given.
    rel_journal = (journal.relative_to(ROOT).as_posix()
                   if journal.is_relative_to(ROOT) else str(journal))
    return dict(
        provenance=dict(
            journal=rel_journal,
            journal_commit=_git("log", "-1", "--format=%h", "--", rel_journal),
            head_commit=_git("log", "-1", "--format=%h"),
            n_seeds=len(seeds), seeds=list(seeds), n_boot=n_boot,
            boot_seed=boot_seed, task=TASK, task_source=TASK_SOURCE,
            geometry=_geometry(journal), n_params=params.pop(),
            built_by="scale/capability_table.py",
            registry=sorted(M3_TASKS),
            producer="scale/m3_quintuple.py --cells softmax glance settled "
                     "twin argmax --seeds 0 1 2 3 4 --ks 8",
            run_log="results/m3_quintuple.txt",
            pre_registration="M3_QUINTUPLE_PREREGISTERED_READING.md",
        ),
        arms=arms, contrasts=contrasts, eprocess=ep,
        consequence_fidelity=dict(
            measured=False, owner="Foreman (LOOP_PROMPT.md 1.7c)",
            weights_for_this_table=_weights_present(seeds),
            reason=(
                "scale/m3_quintuple.py NOW SAVES per-cell weights "
                "(results/m3_quintuple_v2_weights/, one .pt per unit carrying "
                "the state_dict plus every constructor argument and the "
                "mu/sigma standardisation), so the blocker this field used to "
                "name is gone for units run from that change onward. IT IS NOT "
                "GONE FOR THIS TABLE. The {} negation_scope units this table "
                "is built from were journalled BEFORE the change and carry no "
                "tensors; {} of the {} weight files this table would need are "
                "on disk. Re-running them to emit weights costs 6685.3 s by "
                "their own meta.seconds and has not been paid. The only other "
                "trained weights on disk are results/phaseD_weights_*.pt, "
                "whose metrics.kind is pivot_unsigned -- a different arm "
                "family, from scale/trained_projections.py."
            ).format(len(seeds) * len(rows), _weights_present(seeds),
                     len(seeds) * len(rows)),
        ),
        limits=LIMITS.format(*_task_census(journal)),
    )


LIMITS = (
    "What this table does not cover, collected here rather than scattered "
    "through the rows. "
    "(a) EVERY TASK IN THE CORPUS IS STATIC. Both registered oracles are "
    "closed-form functions of the input with no fixed point: "
    "`negation_scope.oracle` is `x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` "
    "(scale/negation_scope.py:68) and `counter_squared_oracle` is "
    "`x[:, :, CH_FLIP].sum(dim=1) ** 2` (scale/negation_scope.py:127). A "
    "settling arm has nothing to settle toward on either, so the headline "
    "contrast was near-zero by construction and the NO DIFFERENCE verdict is "
    "evidence about these tasks, not about settling in general. "
    "(b) `counter_squared` HAS ZERO QUINTUPLE ROWS, and not for want of a flag. "
    "It is registered in M3_TASKS and runs under `scale/m3_capability.py --task "
    "counter_squared`; `scale/m3_quintuple.py` now registers `--task` too, and "
    "{} of the {} rows in `results/m3_quintuple_v2.jsonl` carry an e3 task "
    "suffix written through it ({}), against {} bare `negation_scope` keys. "
    "`counter_squared` was simply never "
    "run under the quintuple. Every number in THIS table is `negation_scope` at "
    "one geometry because `read_journal` filters on `m3_quintuple.task_of`, not "
    "because the corpus is all the producer can emit. "
    "(c) FIVE SEEDS CANNOT DECIDE ANYTHING ANYTIME-VALIDLY. "
    "`eprocess.MIN_T_MIXTURE = 13`; `max_attainable(5) = 3.80169140625` against "
    "`THRESHOLD = 40.0`. The intervals here are fixed-sample, read once, and "
    "carry no anytime validity. A real gap below roughly 0.05 NRMSE reads NO "
    "DIFFERENCE at five seeds whether or not it is real "
    "(M3_QUINTUPLE_PREREGISTERED_READING.md:87). "
    "(d) CONSEQUENCE FIDELITY (1.7c) IS AN EMPTY COLUMN. It needs trained "
    "weights per arm and the quintuple journals metrics only. "
    "(e) THE INTERVALS ARE MONTE-CARLO, AND THE PROSE DELIVERABLES CARRY A "
    "SECOND FAMILY. At five seeds the paired bootstrap distribution is finite "
    "(5**5 = 3125 resamples, 126 distinct values), so the exact percentile is "
    "computable and differs from the B=10000 draw: exact "
    "`[-0.042903167939657406, +0.03155692900564091]` against Monte-Carlo "
    "`[-0.04858658448547344, +0.03155692900564091]` for `settled - twin`. "
    "CHECKLIST.md:1167 and STATE.md:19 print the exact pair under the label "
    "`B=10000`. One further endpoint, `+0.146551` for `settled - softmax` at "
    "CHECKLIST.md:1168 and DONE.md:338, is in NEITHER family: it is absent from "
    "all 3125 exact values and from the 97.5th percentile at every Monte-Carlo "
    "seed 0..399. It is carried here as an open defect, not adopted. "
    "(f) COST IS NOT PRICED. `meta.seconds` in the journal is a contended box "
    "and is labelled PROVISIONAL by its producer; no wall-clock number enters "
    "this table."
)


def _task_census(journal) -> tuple:
    """(n_e3, n_rows, breakdown, n_bare) counted from the journal, not stated.

    THESE NUMBERS ARE COUNTED BECAUSE THE LAST THREE WERE STORED. `LIMITS`
    clause (b) shipped "60 of the 100 rows ... 15 each": 60 and 15 were true of
    an 85-row journal and went stale the moment one more unit landed, and 100
    was never right at all. A count asserted about a file that grows is a
    stale claim with a delay fuse, which is the same defect this clause was
    rewritten to repair.
    """
    counts: dict = {}
    n_rows = 0
    with open(journal, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_rows += 1
            k = json.loads(line)["key"]
            t = task_of(k)
            counts[t] = counts.get(t, 0) + 1
    e3 = {k: v for k, v in counts.items() if k.startswith("e3_")}
    breakdown = ", ".join(f"{k} {e3[k]}" for k in sorted(e3))
    return sum(e3.values()), n_rows, breakdown, counts.get(TASK, 0)


def render(t: dict, manifest=WEIGHTS_MANIFEST) -> str:
    """The card. `manifest` is the weight manifest THIS card speaks for.

    It is a parameter and not a constant because the card describes the folder
    it sits in: `write_artifact` passes the manifest of the directory it is
    writing, so a package built anywhere else advertises its own `weights/` and
    not this repository's. Defaulting to the canonical artifact keeps
    `results/capability_table_v0.md` a statement about the shipped package.
    """
    p = t["provenance"]
    L = []
    L.append("# Capability table v0 -- CEQ signed pivot-routed attention")
    L.append("")
    L.append("Task `{}` at `{}`, {} seeds, `n_params = {}` on every arm. "
             "Journal `{}` at commit `{}` (HEAD `{}`). Pre-registered reading: "
             "`{}`. Produced by `{}`."
             .format(p["task"], p["geometry"], p["n_seeds"], p["n_params"],
                     pathlib.Path(p["journal"]).name, p["journal_commit"],
                     p["head_commit"], p["pre_registration"], p["producer"]))
    L.append("")
    L.append("**Softmax is the baseline and is measured first.** A verdict of "
             "NO DIFFERENCE is printed wherever that is what the interval says.")
    L.append("")
    shipped = shipped_weights(manifest)
    if not shipped:
        L.append("**This package carries NO trained weights.** v0 is the card "
                 "and the modelling code; the numbers below come from "
                 "`results/m3_quintuple_v2.jsonl`, which journals metrics and "
                 "not tensors. A trained checkpoint is a separate deliverable "
                 "and does not exist yet.")
    else:
        L.append("**This package ships no language-model weights, and its table "
                 "cells carry no weights either.** The numbers below come from "
                 "`results/m3_quintuple_v2.jsonl`, which journals metrics and "
                 "not tensors. What IS shipped is a separate, verified set of "
                 "trained probe-arm tensors under `weights/` -- {} {}-parameter "
                 "`{}` checkpoints at task `{}`, one per seed. See \"Weights "
                 "shipped with this package\" below; none of them loads into "
                 "`CEQForCausalLM` and none is named `model.safetensors`."
                 .format(len(shipped),
                         "{:,}".format(shipped[0]["n_params"]),
                         shipped[0]["cell"], shipped[0]["task"]))
    L.append("")
    L.append("## Arms")
    L.append("")
    L.append("| task | arm | eval NRMSE (mean of {} seeds) | sd | marginal 95% CI, seed {} | params | beats predict-the-mean | consequence fidelity (1.7c) |"
             .format(p["n_seeds"], p["seeds"][0]))
    L.append("|---|---|---|---|---|---|---|---|")
    for r in t["arms"]:
        L.append("| `{}` | `{}` | {:.6f} | {:.6f} | [{:.6f}, {:.6f}] | {} | {} | {} |"
                 .format(r["task"], r["arm"], r["nrmse_mean"], r["nrmse_sd"],
                         r["marginal_lo"], r["marginal_hi"], r["n_params"],
                         "yes" if r["beats_predict_the_mean"] else
                         "**NO -- credited with nothing**",
                         r["consequence_fidelity"]))
    L.append("")
    L.append("Per-seed readings, in seed order:")
    L.append("")
    for r in t["arms"]:
        L.append("* `{}`: {}".format(
            r["arm"], "  ".join("{:.6f}".format(v) for v in r["per_seed"])))
    L.append("")
    L.append("## Contrasts")
    L.append("")
    c0 = t["contrasts"][0] if t["contrasts"] else None
    L.append("`delta = NRMSE(reference) - NRMSE(arm)`, so positive means the arm "
             "has the lower error. Estimator: {}. Strict at zero -- an interval "
             "touching zero does not exclude it.".format(
                 c0["estimator"] if c0 else "n/a"))
    L.append("")
    L.append("**Every row carries BOTH interval families, each under its own "
             "name, and the two differ.** At five seeds the paired resample "
             "space is finite, so the percentile the Monte-Carlo draw "
             "estimates is also computable outright: the `{}` column is the "
             "{}. It is not a second measurement and not a correction -- it is "
             "the same per-seed deltas under a different resampling rule, and "
             "a headline quoted from one family will not match the other. "
             "Prose deliverables in this repository adopted the exact family "
             "(`CHECKLIST.md:1239`); this card prints the Monte-Carlo family "
             "the run executed and names both, so a reader who finds two "
             "endpoints for one headline can tell which instrument produced "
             "each instead of assuming one is a typo."
             .format("exact 95% CI", c0["estimator_exact"] if c0 else "n/a"))
    L.append("")
    L.append("Both families are computed from the same {} journal records in "
             "`{}`, seeds {} -- for the row below, keys `{}` and `{}` with the "
             "`_sd<seed>` suffix over those seeds. No number in this section "
             "is transcribed from another document."
             .format(len(c0["journal_keys"]) if c0 else 0,
                     pathlib.Path(p["journal"]).name,
                     ", ".join(str(x) for x in p["seeds"]),
                     "<arm>_k<k>_" + p["geometry"],
                     "<reference>_k<k>_" + p["geometry"]))
    L.append("")
    L.append("| arm | reference | delta | 95% CI (Monte-Carlo) | exact 95% CI | seeds favouring arm | verdict | note |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in t["contrasts"]:
        L.append("| `{}` | `{}` | {:+.6f} | [{:+.6f}, {:+.6f}] | [{:+.6f}, {:+.6f}] | {}/{} | **{}** | {} |"
                 .format(r["arm"], r["ref"], r["delta"], r["ci_lo"], r["ci_hi"],
                         r["ci_exact_lo"], r["ci_exact_hi"],
                         r["seeds_favouring_arm"], r["n_seeds"], r["verdict"],
                         r["note"] or ""))
    L.append("")
    ep = t["eprocess"]
    L.append("## The headline cell is undecided, and that is structural")
    L.append("")
    L.append("```")
    L.append(ep["text"])
    L.append("```")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append("| `E_t` settled direction | `{!r}` |".format(ep["e_settled"]))
    L.append("| `E_t` twin direction | `{!r}` |".format(ep["e_twin"]))
    L.append("| paired seeds `t` | {} |".format(ep["t"]))
    L.append("| ceiling `max_attainable({})` | `{!r}` |".format(
        ep["t"], ep["ceiling_here"]))
    L.append("| `THRESHOLD` = `1/alpha`, `alpha = {}/{}` | `{!r}` |".format(
        ep["alpha_family"], ep["n_directions"], ep["threshold"]))
    L.append("| `MIN_T_MIXTURE = {}` | seeds needed before ANY crossing is "
             "possible |".format(ep["min_t_mixture"]))
    L.append("| decidable at this `t` | {} |".format(
        "yes" if ep["can_decide"] else "**no, by arithmetic fixed before the run**"))
    L.append("")
    L.append("The word `undecided` here is a property of the schedule, not of "
             "the data: at five seeds the mixture cannot reach the threshold "
             "even if every seed lands at the clip bound `B = {!r}`."
             .format(ep["b"]))
    L.append("")
    cf = t["consequence_fidelity"]
    L.append("## Consequence fidelity (1.7c): NOT MEASURED")
    L.append("")
    L.append("Owner: {}. {}".format(cf["owner"], cf["reason"]))
    L.append("")
    if shipped:
        L.append("## Weights shipped with this package")
        L.append("")
        L.append("`weights/` carries {} safetensors files, one per seed, all "
                 "`{}` at task `{}`, geometry `{}`:"
                 .format(len(shipped), shipped[0]["cell"], shipped[0]["task"],
                         shipped[0]["geometry"].partition("_")[2]
                         .replace("_sd{}".format(shipped[0]["seed"]), "")
                         .replace("_task{}".format(shipped[0]["task"]), "")))
        L.append("")
        L.append("| seed | file (`weights/`) | eval NRMSE (journal) |")
        L.append("|---|---|---|")
        for w in sorted(shipped, key=lambda w: w["seed"]):
            L.append("| {} | `{}` | {:.10f} |".format(
                w["seed"], pathlib.Path(w["file"]).name,
                w["eval_nrmse_journal"]))
        L.append("")
        L.append("Seed mean **{:.6f}**. These tensors carry NO capability "
                 "claim: they are the ship candidate's arm at the one rung "
                 "where the cells cleared predict-the-mean, exported for "
                 "inspection and intervention work. Read the rung's own "
                 "interval before quoting any of them."
                 .format(statistics.fmean(
                     w["eval_nrmse_journal"] for w in shipped)))
        L.append("")
        L.append("Provenance, per file (`scripts/export_hf_weights.py`, manifest "
                 "in `weights/MANIFEST.json`): metrics matched by journal key "
                 "against `results/m3_quintuple_v2.jsonl`, then each tensor set "
                 "reloaded from disk into a fresh `scale.m3_quintuple.QuintArm` "
                 "and re-evaluated on the task's own eval batch. Worst "
                 "|delta NRMSE vs journal| over the {} shipped files is "
                 "{:.3e}, against an acceptance bar of {:.0e}. A checkpoint that "
                 "fails is deleted and left out of the manifest rather than "
                 "shipped with a caveat, so an unverified entry never reaches "
                 "this table."
                 .format(len(shipped),
                         max(w["abs_delta_vs_journal"] for w in shipped),
                         max(w["tolerance"] for w in shipped)))
        L.append("")
    L.append("## Limits")
    L.append("")
    L.append(t["limits"])
    L.append("")
    return "\n".join(L)


def write_artifact(t: dict, out_dir, *, with_model: bool = False) -> str:
    """`README.md` + `capability_table_v0.json` in `out_dir`. No network.

    `with_model` additionally writes the three files `ceq/hf/train.py:297`
    requires before it will upload anything -- `config.json`,
    `configuration_ceq.py`, `modeling_ceq.py`. Still no network: the upload is
    `upload_command`'s text and nothing calls it.

    NO LANGUAGE-MODEL WEIGHTS ARE WRITTEN, and that is deliberate. Calling
    `CEQForCausalLM(CEQConfig()).save_pretrained(...)` here produces a
    1,901,686,656-byte `model.safetensors` at random initialisation -- measured,
    not estimated. A random-init tensor file in a folder whose card reports
    trained NRMSE is the shape of claim this table exists to refuse, and it is
    also two gigabytes of nothing.

    THIS FUNCTION OVERWRITES `README.md` IN `out_dir`. That is the whole hazard
    it used to carry: `render` asserted "This package carries NO trained
    weights", which was true when v0 was cut and stopped being true at commit
    `0162bdd`, when `scripts/export_hf_weights.py` shipped five verified `twin`
    probe checkpoints and `weights/MANIFEST.json` into `ceq/hf_artifact/`. The
    README there was corrected by hand and this generator was not, so the next
    call would have silently reverted the correction and republished a false
    absence claim. `render` now reads `shipped_weights()` and reports what the
    manifest actually holds, so the overwrite is safe in both directions: no
    manifest, and the card says no weights; a manifest, and the card names every
    file in it. `tests/neptune/test_capability_table_truth.py` fails if that
    stops being so. Probe checkpoints are still NOT a `CEQForCausalLM`
    checkpoint, which remains S3b's deliverable and does not exist.
    """
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if with_model:
        from ceq.hf import configuration_ceq as _cfg_mod
        from ceq.hf import modeling_ceq as _mdl_mod
        from ceq.hf.configuration_ceq import CEQConfig
        CEQConfig().save_pretrained(str(d))
        for mod in (_cfg_mod, _mdl_mod):
            src = pathlib.Path(mod.__file__)
            (d / src.name).write_text(src.read_text(encoding="utf-8"),
                                      encoding="utf-8")
    (d / "README.md").write_text(
        render(t, manifest=d / "weights" / "MANIFEST.json"), encoding="utf-8")
    (d / "capability_table_v0.json").write_text(
        json.dumps(t, indent=1), encoding="utf-8")
    return str(d)


def upload_command(out_dir, repo_id: str = "<OWNER>/<REPO>") -> str:
    """The command that WOULD publish this. Never executed from this file.

    An outward-facing publish needs the author's say-so; this returns the exact
    call so the decision is theirs and the artifact is ready when it is made.
    """
    return (
        "python - <<'PY'\n"
        "from huggingface_hub import HfApi, login\n"
        "login()                       # reads HF_TOKEN from the env or prompts\n"
        "api = HfApi()\n"
        "api.create_repo({repo!r}, private=True, exist_ok=True)\n"
        "print(api.upload_folder(repo_id={repo!r}, folder_path={path!r},\n"
        "                        commit_message='capability table v0'))\n"
        "PY".format(repo=repo_id, path=str(out_dir))
    )


def _argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scale.capability_table")
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help="which capability_table_<version>.{md,json} to write; "
                         "v0 is frozen evidence and must be named explicitly")
    ap.add_argument("--journal", default=str(JOURNAL))
    ap.add_argument("--artifact", default=None,
                    help="directory to write the HF-shaped package into")
    ap.add_argument("--repo-id", default="<OWNER>/<REPO>")
    ap.add_argument("--card-only", action="store_true",
                    help="skip the three model files ceq/hf/train.py::push "
                         "requires; the folder is then a card, not a package")
    return ap


def main(argv=None) -> int:
    a = _argparser().parse_args(argv)
    out_md, out_json = out_paths(a.version)

    t = build(a.journal)
    md = render(t)
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(md)
    print("\nwrote {}".format(out_md))
    print("wrote {}".format(out_json))
    if a.artifact:
        d = write_artifact(t, a.artifact, with_model=not a.card_only)
        print("wrote artifact {}".format(d))
        print("\n=== THE UPLOAD IS NOT PERFORMED. This is the command:\n")
        print(upload_command(d, a.repo_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
