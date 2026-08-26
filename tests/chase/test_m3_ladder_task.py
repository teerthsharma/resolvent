"""RED-first bind for the two things `scale/m3_quintuple.py` was missing.

Both are named in the round's dispatch and both are checkable by value:

  1. NO `--task` FLAG. `scale/m3_capability.py:247` has one; the quintuple did
     not, so the deciding cell could not be pointed at the E-task corpus that
     `scale/negation_scope.py::M3_TASKS` now registers. Porting it is only half
     the job: the flag has to reach BOTH batches -- the RED 0-step reading and
     the training loop -- and it has to leave the 25 already-journalled
     `negation_scope` unit keys byte-identical, or the bucket re-runs 6685 s of
     completed work and the resume audit compares two different tasks.

  2. NO PER-CELL WEIGHTS. `scale/capability_table.py:232` states the consequence
     of that in the shipped artifact, verbatim: the consequence-fidelity column
     reads NOT MEASURED because "scale/m3_quintuple.py journals metrics only and
     saves no per-cell weights, so no trained settled/twin/argmax/glance weights
     exist to intervene on."

EVERY ABSENCE CLAIM HERE HAS A PLANTED CASE THAT CHANGES IT. A test that only
asserts "the shipped key is unchanged" passes on a `_key` that ignores its
argument entirely; a test that only asserts "the batch function was called"
passes on a runner that calls it once and then trains on something else. Each
check below is paired with a control that must FAIL the same assertion.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import m3_quintuple as Q                                # noqa: E402
from scale import m3_capability as M3                              # noqa: E402
from scale import negation_scope as NS                             # noqa: E402

#: Small enough to train in seconds, large enough to be a legal instance of
#: every task: `make_batch` needs `1 <= d < s - 1`, `make_equilibrium_batch`
#: needs `1 <= t_star <= s - 1`, and `select_pivots(exclude=(0, s-1))` needs
#: `k <= s - 3`. NOT hand-built -- every tensor below is DRAWN by the registered
#: builder for the task under test.
#: `s = 48` and not less: `e3_t32` needs `t_star=32 <= s-1`, and the deepest
#: rung of the ladder has to be constructible or the test binds only the rungs
#: that were already easy.
TINY = dict(s=48, d=8, steps=2, n_train=16, n_eval=16, k=4, t_max=3,
            n_neumann=3)


@pytest.fixture(autouse=True)
def _weights_go_to_tmp(tmp_path, monkeypatch):
    """`results/` is the evidence directory. A test run must not leave 23 kB
    checkpoints of a 16-example toy in it, where a later reader finds them next
    to the real ones and cannot tell which is which from the filename alone."""
    monkeypatch.setattr(Q, "WEIGHTS_DIR", tmp_path / "weights")


def _params(cell="twin", seed=0, task="negation_scope", **over):
    p = dict(cell=cell, seed=seed, task=task, **TINY)
    p.update(over)
    return p


# ---------------------------------------------------------------- the flag ---
def test_task_flag_offers_the_whole_registry():
    """The flag exists and its choices are the REGISTRY, not a hand-copied
    subset that silently drops the rung the ladder needs.

    THE FIRST VERSION OF THIS TEST WAS VACUOUS AND IS RECORDED HERE RATHER THAN
    DELETED. It drove `main()` with `["--task", "e3_t32", "--help"]` and asserted
    `SystemExit(0)`. argparse handles `--help` during the parse and exits 0
    before it reports an unrecognised optional, so that assertion passed against
    the unpatched file, which had no `--task` at all -- a scan structurally
    incapable of a hit. The parser is asked directly instead.
    """
    ap = Q._argparser()
    act = {a.dest: a for a in ap._actions}["task"]
    assert act.default == "negation_scope", "the shipped task is not the default"
    assert set(act.choices) == set(NS.M3_TASKS), (
        f"choices {sorted(act.choices)} are not the registry "
        f"{sorted(NS.M3_TASKS)}")
    for rung in ("e3_t1", "e3_t2", "e3_t8", "e3_t32"):
        assert rung in act.choices

    # the control: the parser must REJECT a task that is not registered
    with pytest.raises(SystemExit):
        ap.parse_args(["--task", "e3_t7"])


# ------------------------------------------------- the journal keys survive ---
def test_shipped_journal_keys_are_byte_identical():
    """The 25 `negation_scope` units already on disk must still be found.

    PLANTED CASE: the same parameters under a different task must produce a
    DIFFERENT key, or the two corpora collide in one bucket and the resume audit
    compares a negation_scope number against an e3 number.
    """
    jl = ROOT / "results" / "m3_quintuple_v2.jsonl"
    done = {json.loads(l)["key"]
            for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert done, "no journalled units to bind against"

    shipped = _params(cell="settled", seed=0, task="negation_scope",
                      s=64, d=24, steps=150, n_train=8192, n_eval=512,
                      k=8, t_max=21, n_neumann=21)
    key = Q._key(shipped)
    assert key == "settled_k8_s64_d24_st150_ntr8192_nev512_b21_sd0"
    assert key in done, f"{key!r} would re-run 25 completed units"

    # the control
    moved = Q._key(dict(shipped, task="e3_t8"))
    assert moved != key, "e3 and negation_scope units share a journal key"
    assert moved not in done


# -------------------------------------------- the flag reaches BOTH batches ---
def test_unit_draws_every_tensor_from_the_selected_task():
    """`--task` must route the RED batch, the eval batch AND the training batch.

    The counter is on the registered builder itself, so this cannot pass on a
    runner that builds one batch from the task and then trains on the shipped
    corpus: `paired_arm.train_and_predict` builds two more, and all four must be
    attributed to the same builder.

    PLANTED CASE: the shipped builder must record ZERO calls while an e3 task is
    selected. Round 7 struck twelve vacuous controls; a call-counter that only
    ever counts up is the thirteenth.
    """
    calls = {"e3_t2": 0, "negation_scope": 0}
    real_e3 = NS.M3_TASKS["e3_t2"][0]
    real_ns = NS.M3_TASKS["negation_scope"][0]

    def wrap(name, fn):
        def inner(*a, **kw):
            calls[name] += 1
            return fn(*a, **kw)
        return inner

    old = dict(NS.M3_TASKS)
    NS.M3_TASKS["e3_t2"] = (wrap("e3_t2", real_e3),) + old["e3_t2"][1:]
    NS.M3_TASKS["negation_scope"] = ((wrap("negation_scope", real_ns),)
                                     + old["negation_scope"][1:])
    try:
        Q._unit(_params(cell="twin", seed=0, task="e3_t2"))
    finally:
        NS.M3_TASKS.update(old)

    assert calls["e3_t2"] >= 4, (
        f"only {calls['e3_t2']} of the four batches came from e3_t2 "
        f"(RED train, RED eval, training train, training eval)")
    assert calls["negation_scope"] == 0, (
        f"the shipped corpus was built {calls['negation_scope']} time(s) "
        f"during an e3_t2 unit")


def test_the_task_actually_changes_the_number():
    """Two rungs of the ladder at one seed must not read the same NRMSE.

    This is the check that a `--task` flag which is accepted, stored and then
    ignored cannot pass. `e3_t1` and `e3_t32` differ only in the nilpotency
    index of the drawn operator, so equal NRMSE to the last bit means the label
    never changed.
    """
    a = Q._unit(_params(cell="twin", seed=0, task="e3_t1"))
    b = Q._unit(_params(cell="twin", seed=0, task="e3_t32"))
    assert a["eval_nrmse"] != b["eval_nrmse"]


# ------------------------------------------------------------- the weights ---
def test_saved_weights_reproduce_the_journalled_number():
    """The saved file must rebuild the arm and REPRODUCE `eval_nrmse` bitwise.

    A `state_dict` on its own does not do this. It does not name the cell -- the
    whole difference between `settled`, `twin` and `argmax` -- and it does not
    carry `mu`/`sigma`, the label standardisation `paired_arm.train_and_predict`
    applies on the way out. Without those, `model(x)` is not the prediction that
    produced the number, and `foreman_consequence.consequence_fidelity` takes
    `sigma` as a named argument.

    PLANTED CASE: perturbing ONE entry of ONE saved tensor must move the NRMSE.
    Without it this asserts only that two copies of the same object agree.
    """
    p = _params(cell="settled", seed=1, task="e3_t8")
    val = Q._unit(p)
    path = Q.weights_path(Q._key(p))
    assert path.exists(), f"no weights at {path}"

    model, rec = Q.load_unit(path)
    assert rec["cell"] == "settled" and rec["task"] == "e3_t8"
    bfn = NS.M3_TASKS[rec["task"]][0]
    xe, ye, _f, _pp = bfn(rec["n_eval"], rec["s"], rec["d"],
                          d_model=rec["d_model"], seed=rec["seed"] + 12345)
    with torch.no_grad():
        pred = model(xe) * rec["sigma"] + rec["mu"]
    got = NS.nrmse(pred, ye)
    assert got == val["eval_nrmse"], (
        f"reloaded weights read {got!r}, journal says {val['eval_nrmse']!r}")

    # the control: break one number and watch the reading move
    with torch.no_grad():
        model.readout.weight[0, 0] += 1.0
        broken = NS.nrmse(model(xe) * rec["sigma"] + rec["mu"], ye)
    assert broken != got, "the NRMSE does not depend on the loaded weights"


def test_saved_weights_carry_the_full_rebuild_recipe():
    """Foreman rebuilds `QuintArm` from this file and nothing else, so every
    constructor argument has to be in it. Named one by one rather than as a
    length check, because a length check passes on the wrong five keys."""
    p = _params(cell="twin", seed=2, task="e3_t1")
    Q._unit(p)
    rec = torch.load(Q.weights_path(Q._key(p)), weights_only=True)
    for f in ("state_dict", "kind", "cell", "s", "d", "k_piv", "beta", "t_max",
              "n_neumann", "d_model", "mu", "sigma", "seed", "task", "steps",
              "n_train", "n_eval", "eval_nrmse", "n_params", "key"):
        assert f in rec, f"saved unit does not carry {f!r}"
    assert rec["n_params"] == sum(v.numel() for v in rec["state_dict"].values())


# ------------------------------------------------------- the ladder reader ---
#: `verdict` is a ROUTING TABLE, so the test enumerates its branches rather than
#: drawing instances: the object under test is the mapping from (which CIs
#: exclude zero, in which direction, at which rung) to a pre-registered row, and
#: that mapping has no distribution to draw from. The NUMBERS that reach it are
#: drawn -- they come out of the bucket journal via `read` -- and
#: `test_the_reader_refuses_a_verdict_on_an_empty_ladder` binds that path.
from scale import e_ladder as EL                                   # noqa: E402


def _rung(task, t_star, delta, half=0.01, credited=True):
    """One ladder row with a CI of the requested width around `delta`."""
    return dict(task=task, t_star=t_star, state="RUN", delta=delta,
                ci_lo=delta - half, ci_hi=delta + half, sd_paired=0.02,
                verdict="x", per_seed_delta=[delta] * 5, settled_mean=0.8,
                twin_mean=0.8, seeds_favouring_settled=3, credited=credited,
                excludes_zero=not (delta - half <= 0.0 <= delta + half))


def _cur(rows, complete=True):
    return dict(ladder=rows, seeds=[0, 1, 2, 3, 4], config={},
                resolution_13=EL.RESOLUTION_13, complete=complete,
                all_credited=all(r.get("credited") for r in rows))


def test_verdict_routes_every_preregistered_row():
    T = ("e3_t1", "e3_t2", "e3_t8", "e3_t32")
    ts = (1, 2, 8, 32)
    tiny = EL.RESOLUTION_13 / 2          # below the 13-seed resolution
    big = EL.RESOLUTION_13 * 2           # above it

    def L(ds, halves=None, credited=(True,) * 4):
        halves = halves or (0.01,) * 4
        return [_rung(t, s, d, h, c)
                for t, s, d, h, c in zip(T, ts, ds, halves, credited)]

    # A: covers zero at t*=1, excludes zero positive at t*=32, monotone
    row, _ = EL.verdict(_cur(L((0.000, 0.02, 0.05, 0.10),
                               halves=(0.01, 0.03, 0.01, 0.01))))
    assert row == "A", row

    # B: wins at t*=32 but not monotone
    row, _ = EL.verdict(_cur(L((0.000, 0.09, 0.02, 0.10),
                               halves=(0.01, 0.005, 0.03, 0.01))))
    assert row == "B", row

    # C: every CI covers zero and every |delta| below the 13-seed resolution
    row, _ = EL.verdict(_cur(L((tiny, -tiny, tiny, -tiny),
                               halves=(0.05,) * 4)))
    assert row == "C", row

    # D: every CI covers zero but |delta| at a deep rung is above it
    row, _ = EL.verdict(_cur(L((tiny, tiny, big, tiny), halves=(0.2,) * 4)))
    assert row == "D", row

    # E: wins at t*=1 -- capacity leakage, whatever the deep rungs say
    row, _ = EL.verdict(_cur(L((0.08, 0.0, 0.0, 0.09),
                               halves=(0.01, 0.05, 0.05, 0.01))))
    assert row == "E", row

    # F: wins at t*=1 AND loses at t*=8 -- both directions
    row, _ = EL.verdict(_cur(L((0.08, 0.0, -0.09, 0.0),
                               halves=(0.01, 0.05, 0.01, 0.05))))
    assert row == "F", row

    # G: a cell at or above predict-the-mean at one rung
    row, _ = EL.verdict(_cur(L((0.0,) * 4,
                               credited=(True, True, False, True))))
    assert row == "G", row

    # H: loses somewhere, wins nowhere
    row, _ = EL.verdict(_cur(L((0.0, 0.0, -0.09, -0.12),
                               halves=(0.05, 0.05, 0.01, 0.01))))
    assert row == "H", row


def test_no_kill_is_claimed_on_a_partial_ladder():
    """Rows A, C and F quantify over EVERY rung. The pre-registration forbids
    claiming them when a rung is missing, and this is the check that the code
    obeys the document rather than the document describing the code."""
    T, ts = ("e3_t1", "e3_t2", "e3_t8", "e3_t32"), (1, 2, 8, 32)
    tiny = EL.RESOLUTION_13 / 2
    rows = [_rung(t, s, tiny, 0.05) for t, s in zip(T, ts)]
    assert EL.verdict(_cur(rows, complete=True))[0] == "C"
    # the control: the SAME numbers with one rung absent must NOT read C
    partial = rows[:3] + [dict(task="e3_t32", t_star=32, state="NOT RUN",
                               have_settled=0, have_twin=0)]
    row, why = EL.verdict(_cur(partial, complete=False))
    assert row == "--", row
    assert "PARTIAL" in why


def test_the_reader_refuses_a_verdict_on_an_empty_ladder(tmp_path):
    """Read against a journal with no e3 units at all: every rung NOT RUN, and
    the counts are reported rather than a zero being invented."""
    j = tmp_path / "empty.jsonl"
    j.write_text("", encoding="utf-8")
    cur = EL.read(j)
    assert cur["complete"] is False
    assert [r["state"] for r in cur["ladder"]] == ["NOT RUN"] * 4
    assert all(r["have_settled"] == 0 and r["have_twin"] == 0
               for r in cur["ladder"])
    assert EL.verdict(cur)[0] == "--"


# ------------------------------------------- one bucket, more than one task ---
def test_an_e3_row_in_the_bucket_does_not_reach_the_negation_scope_table():
    """A second task in the same journal must be INVISIBLE to the shipped table.

    THIS IS A REGRESSION TEST FOR A DEFECT THE TASK SUFFIX CAUSED. Four
    independent copies of the key grammar existed -- `capability_table.
    read_journal`, `eprocess._parse_key`, `tests/chase/test_capability_table.
    py::_rows` and `m3_quintuple._key` itself -- and three of them read
    everything after `_sd` as the seed. The first journalled
    `..._sd0_taske3_t1` unit raised `ValueError: invalid literal for int() with
    base 10: '0_taske3_t1'` and took 14 of the 16 capability-table tests down
    with it.

    The control is the second half: the SAME builder over the SAME journal with
    the e3 line removed must produce byte-identical output. Without it this
    asserts only that the builder does not crash.
    """
    from scale import capability_table as CT

    src = (ROOT / "results" / "m3_quintuple_v2.jsonl").read_text(
        encoding="utf-8").splitlines()
    ns = [l for l in src if l.strip() and Q.task_of(json.loads(l)["key"])
          == Q.SHIPPED_TASK]
    assert ns, "no negation_scope rows to build a table from"

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        clean = pathlib.Path(td) / "clean.jsonl"
        mixed = pathlib.Path(td) / "mixed.jsonl"
        clean.write_text("\n".join(ns) + "\n", encoding="utf-8")
        # a PLANTED e3 row, drawn from a real unit rather than invented: take a
        # journalled negation_scope record and relabel its key to the e3 rung.
        rec = json.loads(ns[0])
        rec["key"] = rec["key"] + "_taske3_t8"
        mixed.write_text("\n".join(ns) + "\n" + json.dumps(rec, sort_keys=True)
                         + "\n", encoding="utf-8")

        assert Q.task_of(rec["key"]) == "e3_t8"
        assert Q.task_of(json.loads(ns[0])["key"]) == Q.SHIPPED_TASK
        a = CT.build(journal=clean)
        b = CT.build(journal=mixed)

    for t in (a, b):
        t["provenance"].pop("journal", None)
        t["provenance"].pop("journal_commit", None)
    assert json.dumps(a["arms"], sort_keys=True) == \
        json.dumps(b["arms"], sort_keys=True)
    assert json.dumps(a["contrasts"], sort_keys=True) == \
        json.dumps(b["contrasts"], sort_keys=True)
