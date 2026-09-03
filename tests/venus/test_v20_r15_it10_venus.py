"""VENUS it.10 — the three it.9 objections, made test-bound.

Every claim in V20_R15_IT10_VENUS.md that touches the record is asserted here.
Filed 2026-09-02.
"""
import glob, json, math, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR1 = 0.7071067811865476


def _cells():
    out = []
    for f in glob.glob(str(ROOT / "results" / "*.jsonl")):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (isinstance(d, dict) and {"kind", "seed", "eval_nrmse"} <= d.keys()
                    and d.get("t") != "rescore"):   # banked cells only
                d["_file"] = Path(f).name
                out.append(d)
    return out


# ---------------------------------------------------------------- CLAIM 1
def test_eval_draw_is_one_pinned_batch_so_n_eff_is_one():
    """n_eff = 1: the eval batch is drawn ONCE, at a literal seed, above the loop."""
    src = (ROOT / "scripts" / "v15_r1.py").read_text(encoding="utf-8").splitlines()
    pinned = [i for i, l in enumerate(src) if "seed=12345" in l]
    draw = [i for i in pinned if "print(" not in src[i]]
    assert len(pinned) == 2 and len(draw) == 1, (pinned, draw)  # one draw, one banner echo
    loop = [i for i, l in enumerate(src) if re.search(r"for seed in a\.seeds", l)]
    assert len(draw) == 1, f"expected one pinned eval draw, got lines {draw}"
    assert loop, "arm/seed loop not found"
    assert draw[0] < loop[0], "eval draw must sit above the cell loop"
    # and nothing on the command line moves it
    assert "12345" not in "".join(l for l in src if "add_argument" in l)
    print(f"eval draw pinned at v15_r1.py:{draw[0]+1}; cell loop opens at :{loop[0]+1}")


def test_planted_negative_a_per_cell_eval_seed_would_break_claim_1():
    """Control: the claim is falsifiable — it fails on a source that re-seeds per cell."""
    fake = ["for seed in a.seeds:", "    batch_fn(n, seed=12345)"]
    draw = [i for i, l in enumerate(fake) if "seed=12345" in l]
    loop = [i for i, l in enumerate(fake) if re.search(r"for seed in a\.seeds", l)]
    assert not (draw[0] < loop[0])


# ---------------------------------------------------------------- CLAIM 2
def test_softmax_was_never_run_past_seed_7_in_the_banked_record():
    """MARS it.9 strike 3. True of the banked cells; repaired by the it.10 rescore."""
    seeds = {c["kind"]: set() for c in _cells()}
    for c in _cells():
        seeds[c["kind"]].add(c["seed"])
    assert seeds["softmax"] == set(range(8)), sorted(seeds["softmax"])
    assert seeds["arm_pl"] == set(range(16))
    assert seeds["arm_smprime"] == set(range(16))


def test_the_contrast_survives_pairing_on_seeds_0_to_7():
    """MARS's weakest strike, answered on the paired subset instead of argued."""
    best = {}
    for c in _cells():
        if c["seed"] < 8:
            best.setdefault((c["kind"], c["seed"]), c["eval_nrmse"])
    pl = sum(1 for (k, s), v in best.items() if k == "arm_pl" and v < FLOOR1)
    sm = sum(1 for (k, s), v in best.items() if k == "softmax" and v < FLOOR1)
    print(f"paired seeds 0-7: arm_pl {pl}/8 below floor_1, softmax {sm}/8")
    assert (pl, sm) == (5, 0)


# ---------------------------------------------------------------- CLAIM 3
def test_the_headline_verdict_flips_on_one_cell():
    f = ROOT / "results" / "v20_r15_it8_armpl_b.jsonl"
    rows = [json.loads(l) for l in open(f)]
    agg = [d for d in rows if d.get("t") == "agg"][0]
    assert agg["crosses"] is False and agg["n"] == 9
    cells = sorted((d["seed"], d["eval_nrmse"]) for d in rows
                   if {"kind", "seed", "eval_nrmse"} <= d.keys())
    v9 = [x for _, x in cells]
    v8 = [x for s, x in cells if s != 9]

    def ci_hi(v):
        from scipy.stats import t
        m, sd = st.fmean(v), st.stdev(v)
        return m + t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))

    assert math.isclose(ci_hi(v9), agg["ci_hi"], rel_tol=1e-12)   # GREEN control
    assert ci_hi(v9) > FLOOR1 and ci_hi(v8) < FLOOR1
    print(f"n=9 ci_hi={ci_hi(v9)!r} crosses=False ; n=8 ci_hi={ci_hi(v8)!r} crosses=True")


# ---------------------------------------------------------------- CLAIM 4
def test_lambda_sign_separates_the_floor_on_all_16_arm_pl_cells():
    best = {}
    for c in _cells():
        if c["kind"] == "arm_pl":
            best[c["seed"]] = c
    assert len(best) == 16
    for s, c in best.items():
        assert (c["lambda_hat"] > 0) == (c["eval_nrmse"] >= FLOOR1), (s, c["lambda_hat"])
    above = sorted(s for s, c in best.items() if c["eval_nrmse"] >= FLOOR1)
    assert above == [2, 3, 7, 9]


# ---------------------------------------------------------------- CLAIM 5
def test_the_pinned_draws_scale_error_cannot_flip_a_single_crossing():
    """n_eff=1 splits into a SCALE channel and a SAMPLE channel. This bounds the
    scale channel: rescaling every cell by the pinned draw's own denominator
    error (MARS it.9 §5: std(y_ev)=1.429381 vs sqrt(T*)=1.414214) moves no cell
    across floor_1, because the smallest relative margin is 3.5x the shift."""
    k = 1.429381 / 1.414214
    best = {}
    for c in _cells():
        best[(c["kind"], c["seed"])] = c["eval_nrmse"]
    for arm, n in (("arm_pl", 16), ("arm_smprime", 16), ("softmax", 8)):
        v = [x for (a, _), x in best.items() if a == arm]
        assert len(v) == n
        assert sum(1 for x in v if x < FLOOR1) == sum(1 for x in v if x * k < FLOOR1)
    pl = [x for (a, _), x in best.items() if a == "arm_pl" and x < FLOOR1]
    margin = min((FLOOR1 - x) / FLOOR1 for x in pl)
    print(f"scale shift={k-1:.6f}  min relative margin={margin:.6f}  ratio={margin/(k-1):.2f}")
    assert margin > 3 * (k - 1)


# ------------------------------------------- CLAIM 6 — the it.10 MERCURY rescore
RESCORE = ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl"


def _rescore():
    tab = {}
    for line in open(RESCORE):
        d = json.loads(line)
        if d.get("t") == "rescore":
            tab.setdefault((d["kind"], d["seed"]), {})[d["eval_seed"]] = d["eval_nrmse"]
    return tab


def test_crossing_indicator_is_invariant_across_three_eval_draws_on_all_cells():
    """The n_eff=1 objection, answered on the indicator the round scores."""
    tab = _rescore()
    assert len(tab) == 18 and all(len(v) == 3 for v in tab.values())
    for key, v in tab.items():
        assert len({x < FLOOR1 for x in v.values()}) == 1, (key, v)
    for e in (12345, 12346, 20260902):
        # it.13 repair (MARS it.12 F2): two counts are not a paired contrast.
        assert {s for (k, s) in tab if k == "arm_pl"} == {s for (k, s) in tab if k == "softmax"},             "unpaired: the arms did not run on the same seeds"
        pl = sum(1 for (k, _), v in tab.items() if k == "arm_pl" and v[e] < FLOOR1)
        sm = sum(1 for (k, _), v in tab.items() if k == "softmax" and v[e] < FLOOR1)
        assert (pl, sm) == (8, 0), (e, pl, sm)
    # control: the pinned draw reproduces the banked it.8 cell bitwise
    assert tab[("arm_pl", 9)][12345] == 1.281779592990027


def test_seed_15_is_the_cell_that_flips_first_and_its_margin_is_one_spread():
    """Pre-registration for it.11: draw noise on the weakest crossing cell is the
    same size as its distance to the floor."""
    tab = _rescore()
    rank = sorted(
        ((max(v.values()) - min(v.values())) / min(abs(x - FLOOR1) for x in v.values()), s)
        for (k, s), v in tab.items() if k == "arm_pl")
    assert rank[-1][1] == 15 and rank[-2][1] == 12
    assert 0.95 < rank[-1][0] < 1.05, rank[-1]
    print(f"fragility ranking (spread/margin): {[(s, round(r,3)) for r, s in reversed(rank)]}")
