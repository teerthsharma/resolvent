"""MARS it.9 -- does seed 9's unselected `arm_pl` blow-up fit the M2 story?

`CEQ_V20_R15_CONTRACT.md:181-185` states the M2 GATE-LANDSCAPE THEOREM: an
open-range gate whose `a_hat_max` has no attained supremum as `|w| -> inf`.
`V20_R15_IT7_JUPITER.md:180-198` measures that theorem on exactly three
`arm_pl` cells -- seeds 2, 3, 7, all from `results/v17k_r4_retake.jsonl` --
and calls them "that theorem's conclusion measured". Those three cells are
the ONLY prior evidence for the M2 divergence story; every other `arm_pl`
cell in the round reads `lambda_hat ~= -1.45`.

`results/v20_r15_it8_armpl_b.jsonl` seed 9 arrived UNSELECTED (it was not one
of the seeds anyone picked to test the story) with `lambda_hat > 0`. If it is
a genuine fourth instance of M2, its fields should sit inside the envelope
the three known instances already staked out -- not just share the sign of
`lambda_hat`. This test builds that envelope from the real seeds 2/3/7 rows
and checks seed 9 against it, field by field, in one failure message so nothing
is hidden behind an early `assert`.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"
ARMPL_B = ROOT / "results" / "v20_r15_it8_armpl_b.jsonl"

DIVERGENT_SEEDS = (2, 3, 7)  # V20_R15_IT7_JUPITER.md:180-182
PROBE_SEED = 9


def _cells(path):
    rows = [json.loads(l) for l in path.open()]
    return [r for r in rows if r.get("t") == "cell" and r.get("kind") == "arm_pl"]


def _by_seed(cells):
    return {c["seed"]: c for c in cells}


def test_seed9_falls_inside_the_seeds_2_3_7_m2_envelope_field_by_field():
    retake = _by_seed(_cells(RETAKE))
    armpl_b = _by_seed(_cells(ARMPL_B))

    divergent = [retake[s] for s in DIVERGENT_SEEDS]
    assert all(c["lambda_hat"] > 0 for c in divergent), (
        "fixture invariant broken: seeds 2/3/7 in v17k_r4_retake.jsonl are "
        "no longer the round's divergent trio"
    )

    seed9 = armpl_b[PROBE_SEED]

    fields = ["lambda_hat", "a_hat_max", "gate_r2", "eval_nrmse"]
    envelope = {f: (min(c[f] for c in divergent), max(c[f] for c in divergent))
                for f in fields}

    failures = []
    for f in fields:
        lo, hi = envelope[f]
        v = seed9[f]
        inside = lo <= v <= hi
        status = "inside" if inside else "OUTSIDE"
        failures.append(
            f"  {f:14s} seed9={v!r:24s} envelope=[{lo!r}, {hi!r}]  -> {status}"
        )
        if not inside:
            failures[-1] += "  <-- FAILS envelope containment"

    outside = [f for f in fields if not (envelope[f][0] <= seed9[f] <= envelope[f][1])]

    report = (
        "seed 9 (results/v20_r15_it8_armpl_b.jsonl) vs the seeds-2/3/7 envelope "
        "(results/v17k_r4_retake.jsonl), field by field:\n"
        + "\n".join(failures)
        + f"\n\nfields outside envelope: {outside or 'none'}"
    )

    assert not outside, report
