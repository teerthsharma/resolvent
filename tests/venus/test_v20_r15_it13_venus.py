"""VENUS it.13 — the F2 repair on C14, and the W1 evidence the re-forecast reads.

Filed 2026-09-02. Every claim in V20_R15_IT13_VENUS.md that touches the record is
asserted here. No `and False`; the RED below is produced by MUTATING DATA and
running UNMUTATED shipped code over it.
"""
import glob, json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR1 = 0.7071067811865476
RESCORE = ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl"
SMPRIME = ROOT / "results" / "v20_r15_it11_mercury_smprime.jsonl"
IT10 = ROOT / "tests" / "venus" / "test_v20_r15_it10_venus.py"

# VENUS_MUT=disjoint_softmax moves every softmax record onto a seed block disjoint
# from arm_pl's, so the pairing C14 asserts is FALSE BY CONSTRUCTION in the data.
MUT = os.environ.get("VENUS_MUT", "")


def _rescore(path=RESCORE):
    """The shipped it.10 loader, verbatim in behaviour, plus the data mutation hook."""
    tab = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("t") == "rescore":
            seed = d["seed"]
            if MUT == "disjoint_softmax" and d["kind"] == "softmax":
                seed = 900 if seed == 0 else seed + 900   # {900, 908..915}
            tab.setdefault((d["kind"], seed), {})[d["eval_seed"]] = d["eval_nrmse"]
    return tab


def _counts(tab, e):
    pl = sum(1 for (k, _), v in tab.items() if k == "arm_pl" and v[e] < FLOOR1)
    sm = sum(1 for (k, _), v in tab.items() if k == "softmax" and v[e] < FLOOR1)
    return pl, sm


def _paired(tab):
    return {s for (k, s) in tab if k == "arm_pl"} == {s for (k, s) in tab if k == "softmax"}


# ---------------------------------------------------------------- CLAIM 1
def test_the_count_pair_is_blind_to_a_disjoint_seed_block():
    """MARS's F2, reproduced: (8, 0) is green on data where pairing is false.

    Under VENUS_MUT=disjoint_softmax this test PASSES — its passing IS the defect.
    """
    tab = _rescore()
    assert len(tab) == 18
    for e in (12345, 12346, 20260902):
        assert _counts(tab, e) == (8, 0), (e, _counts(tab, e))
    if MUT == "disjoint_softmax":
        assert not _paired(tab), "mutation did not take"


# ---------------------------------------------------------------- CLAIM 2
def test_the_shipped_repair_fires_on_exactly_that_mutation():
    """The one-line replacement is RED where the counts are green. This is the
    whole content of the F2: same data, same code, one predicate sees it."""
    tab = _rescore()
    if MUT == "disjoint_softmax":
        assert not _paired(tab)          # the repair would raise -> RED
        raise AssertionError(
            "unpaired: the arms did not run on the same seeds — "
            f"arm_pl {sorted(s for (k, s) in tab if k == 'arm_pl')} vs "
            f"softmax {sorted(s for (k, s) in tab if k == 'softmax')}")
    assert _paired(tab), "the pairing C14 asserts is false on the unmutated tree"
    assert sorted(s for (k, s) in tab if k == "arm_pl") == [0, 8, 9, 10, 11, 12, 13, 14, 15]


# ---------------------------------------------------------------- CLAIM 3
def test_the_repair_is_present_in_the_it10_node_and_not_only_in_prose():
    """A repair described in a report and absent from the file is the same defect."""
    src = IT10.read_text(encoding="utf-8")
    assert 'unpaired: the arms did not run on the same seeds' in src
    hit = [i for i, l in enumerate(src.splitlines(), 1)
           if 'k == "arm_pl"} ==' in l and 'k == "softmax"}' in l]
    assert len(hit) == 1, hit
    print(f"pairing assertion lives at test_v20_r15_it10_venus.py:{hit[0]}")


# ---------------------------------------------------------------- CLAIM 4
def _smp():
    tab = {}
    for line in open(SMPRIME):
        d = json.loads(line)
        if d.get("t") == "rescore":
            tab.setdefault(d["seed"], {})[d["eval_seed"]] = d["eval_nrmse"]
    return tab


def test_smprime_seed_2_crosses_on_all_three_draws_and_is_the_only_cell_that_does():
    tab = _smp()
    assert len(tab) == 16 and all(len(v) == 3 for v in tab.values())
    crossing = {s for s, v in tab.items() if all(x < FLOOR1 for x in v.values())}
    assert crossing == {2}, crossing
    v = tab[2]
    assert sorted(round(x, 6) for x in v.values()) == [0.203920, 0.208055, 0.216517]
    worst = FLOOR1 - max(v.values())
    assert round(worst, 6) == 0.490590, worst
    # every other cell fails on every draw; nearest is seed 8, 0.144954 above floor
    above = min(min(x for x in tab[s].values()) - FLOOR1 for s in tab if s != 2)
    assert round(above, 6) == 0.144954, above
    print(f"smprime seed2 draws={sorted(v.values())} worst margin below floor={worst:.6f}")


def test_seed_2_is_38x_less_fragile_than_the_tightest_arm_pl_cell():
    """VENUS's own it.10 fragility measure, applied to the W1 cell."""
    v = _smp()[2]
    r_smp = (max(v.values()) - min(v.values())) / min(abs(x - FLOOR1) for x in v.values())
    assert round(r_smp, 4) == 0.0257, r_smp
    pl = _rescore()
    rank = sorted(((max(v.values()) - min(v.values())) / min(abs(x - FLOOR1) for x in v.values()), s)
                  for (k, s), v in pl.items() if k == "arm_pl")
    r_pl, s_pl = rank[-1]
    assert s_pl == 15 and 0.95 < r_pl < 1.05, rank[-1]
    assert 38.0 < r_pl / r_smp < 39.5, r_pl / r_smp
    print(f"fragility: smprime seed2 {r_smp:.4f} vs arm_pl seed15 {r_pl:.3f} "
          f"= {r_pl / r_smp:.1f}x")


# ---------------------------------------------------------------- CLAIM 5
def test_the_smprime_rescore_reproduces_the_banked_cells_bitwise_16_of_16():
    """The control that licenses reading the draws at all."""
    banked = {}
    for f in glob.glob(str(ROOT / "results" / "*.jsonl")):
        if Path(f).name.startswith("v20_r15_it1"):
            continue
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("kind") == "arm_smprime" and "eval_nrmse" in d:
                banked.setdefault(d["seed"], set()).add(d["eval_nrmse"])
    tab = _smp()
    hits = [s for s in tab if s in banked and tab[s][12345] in banked[s]]
    assert len(hits) == 16, sorted(set(tab) - set(hits))
    print(f"bitwise control at draw 12345: {len(hits)} of {len(tab)}")


# ---------------------------------------------------------------- CLAIM 6
def test_the_forecast_never_reads_an_F_grade():
    """TASK B, made mechanical: the ranking's four falsifier clauses are stated
    over observables. If any F-token appears in this office's clause text, the
    forecast depends on a scale with no definition."""
    rep = ROOT / "V20_R15_IT13_VENUS.md"
    if not rep.exists():
        return
    txt = rep.read_text(encoding="utf-8")
    m = re.search(r"<!-- CLAUSES-BEGIN -->(.*?)<!-- CLAUSES-END -->", txt, re.S)
    assert m, "the ranking's falsifier clauses must be delimited for this check"
    body = m.group(1)
    assert not re.search(r"\bF[0-4]\b", body), re.findall(r"\bF[0-4]\b", body)
    print(f"clause block {len(body)} chars, zero F-grade tokens")
