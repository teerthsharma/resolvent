"""V20 R15 it.6 -- the fresh-seed run's OWN integrity, not its verdict.

This node binds three things and deliberately binds nothing else:

  1. REGIME MATCH. The new journal's header must name the same regime as
     `results/v17k_r4_retake.jsonl:1` on every field that decides
     comparability -- device, both determinism flags, floor_1, steps, the
     cell's shape, and the instrument hash. A cell taken under a different
     regime is not poolable with the retake and the round has already refused
     one cross-device pool.
  2. CONTROL. Seeds 0 and 1 re-run under THIS invocation must reproduce the
     retake's `eval_nrmse` to the bit. The pipeline is known reproducible per
     seed (`results/v17k_r4_floor.jsonl` is an independent run that already
     does this), so a mismatch would convict the invocation, not the pipeline.
  3. COVERAGE. Ten `arm_smprime` cells, seeds {0,1,8..15}, no others.

It asserts NOTHING about whether VENUS's or MARS's prediction held. That is
scored at it.29 and it is not this office's row.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl"
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"

REGIME_FIELDS = ("task", "t_star", "s", "d", "n_train", "n_eval", "steps",
                 "lr", "d_model", "device", "deterministic_algorithms",
                 "deterministic_warn_only", "cublas_workspace_config",
                 "threads", "torch", "floor_1", "instrument_hash")


def _rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _header(p):
    return [r for r in _rows(p) if r.get("t") == "header"][0]


def _cells(p, kind="arm_smprime"):
    return {r["cell"].rsplit("seed", 1)[1]: r
            for r in _rows(p) if r.get("t") == "cell" and r["kind"] == kind}


def test_regime_matches_the_retake_field_for_field():
    new, old = _header(NEW), _header(RETAKE)
    diff = {k: (old.get(k), new.get(k)) for k in REGIME_FIELDS
            if old.get(k) != new.get(k)}
    assert diff == {}, f"regime drift vs results/v17k_r4_retake.jsonl:1 -> {diff}"


def test_seeds_0_and_1_reproduce_the_retake_bitwise():
    new, old = _cells(NEW), _cells(RETAKE)
    got = {s: (old[s]["eval_nrmse"], new.get(s, {}).get("eval_nrmse"))
           for s in ("0", "1")}
    bad = {s: v for s, v in got.items() if v[0] != v[1]}
    assert bad == {}, f"control cells did not reproduce (retake, new): {bad}"


def test_ten_cells_seeds_0_1_and_8_through_15():
    seeds = sorted(int(s) for s in _cells(NEW))
    assert seeds == [0, 1, 8, 9, 10, 11, 12, 13, 14, 15], seeds


def test_every_cell_carries_the_columns_the_report_publishes():
    cols = ("eval_nrmse", "frac_gate_annihilated", "frac_gate_annihilated_0step",
            "a_hat_min", "a_hat_max", "lambda_hat", "unit_root", "secs")
    #: the four trained switches and the zero-gate count are NOT cell-record
    #: fields -- `scripts/v15_r1.py` puts them on the cell's own identity
    #: manifest under `smp_values`, filled from the TRAINED module. Reading
    #: them off the `t="bind"` row instead would report the constructor's
    #: values as the cell's.
    smp = ("beta", "qk", "g", "route", "n_zero_gates")
    missing = {}
    for k, r in _cells(NEW).items():
        m = [c for c in cols if c not in r]
        m += [c for c in smp if c not in r.get("manifest", {}).get("smp_values", {})]
        if m:
            missing[k] = m
    assert missing == {}, missing
