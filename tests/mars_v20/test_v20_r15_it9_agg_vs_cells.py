"""MARS it.9 -- attack on the it.8 `arm_pl` headline, from the file's own rows.

Two properties the round's reading of `results/v20_r15_it8_armpl_b.jsonl` needs:

  A. the registered aggregate verdict and the per-cell crossing count are the
     same statement about the same eight cells;
  B. every published CI on a BOUNDED quantity lies inside its bound. `gate_r2`
     and `sign_acc` are both <= 1 by construction, so a CI whose upper end
     exceeds 1 is an interval the estimand cannot occupy.

Both are asserted against the real journal, never a reconstruction.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "results" / "v20_r15_it8_armpl_b.jsonl"


def rows():
    return [json.loads(l) for l in P.open()]


def test_aggregate_verdict_agrees_with_per_cell_crossing_count():
    rs = rows()
    agg = [r for r in rs if r.get("t") == "agg"][0]
    cells = [r for r in rs if r.get("t") == "cell"]
    floor_1 = agg["floor_1"]
    crossed = [c for c in cells if c["eval_nrmse"] < floor_1]
    assert (len(crossed) > len(cells) / 2) == bool(agg["crosses"]), (
        f"per-cell count says {len(crossed)}/{len(cells)} cells cross "
        f"floor_1={floor_1!r}, aggregate row says crosses={agg['crosses']!r} "
        f"(mean={agg['mean']!r}, dist_to_floor={agg['dist_to_floor']!r})"
    )


def test_published_cis_on_bounded_quantities_stay_inside_the_bound():
    rs = rows()
    probe = [r for r in rs if r.get("t") == "probe"][0]
    bad = {k: v for k, v in probe.items()
           if k.endswith("_ci") and k.startswith(("gate_r2", "sign_acc"))
           and isinstance(v, list) and v[1] > 1.0}
    assert not bad, f"CI upper end above the quantity's own bound of 1.0: {bad}"


def test_control_the_aggregate_row_is_reproducible_from_the_cells():
    """GREEN control. If this fails, the two RED tests above are reading a
    journal I cannot reconstruct, and neither is evidence of anything."""
    import math, statistics
    from scipy import stats
    rs = rows()
    agg = [r for r in rs if r.get("t") == "agg"][0]
    v = [c["eval_nrmse"] for c in rs if c.get("t") == "cell"]
    m, sd = statistics.fmean(v), statistics.stdev(v)
    half = stats.t.ppf(0.975, df=len(v) - 1) * sd / math.sqrt(len(v))
    assert (m, sd, m + half) == (agg["mean"], agg["sd"], agg["ci_hi"])
    assert bool(m + half < agg["floor_1"]) is bool(agg["crosses"])


def test_the_fresh_seeds_have_a_softmax_control_on_the_same_seeds():
    """The it.8 headline reads `arm_pl` 7-of-8 FRESH against `softmax` 0/8.
    A head-to-head needs both arms on the same seeds."""
    import glob
    fresh = set(range(8, 16))
    have = {c["seed"] for p in glob.glob(str(ROOT / "results" / "*.jsonl"))
            for c in (json.loads(l) for l in open(p))
            if c.get("t") == "cell" and c.get("kind") == "softmax"}
    assert fresh <= have, (
        f"softmax cells exist only for seeds {sorted(have)}; the fresh seeds "
        f"{sorted(fresh - have)} have no softmax control anywhere in results/"
    )
