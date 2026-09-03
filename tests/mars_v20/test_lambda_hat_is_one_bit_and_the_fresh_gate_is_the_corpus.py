"""MARS, CEQ v20 R15 it.7 -- two STRIKES against the round's most load-bearing
measurement, `results/v20_r15_it6_seeds8_15.jsonl` (MERCURY it.6, ten cells).

STRIKE 9  `lambda_hat` is a one-bit column that the round reads as a rate.
STRIKE 10 `frac_gate_annihilated` is STILL the corpus sign census on the fresh
          seeds, and the two cells that escape it escape UPWARD.

MISTAKES.md M-18 is the law STRIKE 10 runs under:

    "Before a diagnostic is registered, run it on the CORPUS ALONE with no arm
     and report its value and its null. A diagnostic whose no-arm value equals
     its expected with-arm value is measuring the corpus."

Nothing here trains, writes, or touches the network. Every number is read from
the committed journal or counted off the corpus builder. No GPU is required.
"""
from __future__ import annotations

import json
import math
import pathlib

from scale import negation_scope as ns

REPO = pathlib.Path(__file__).resolve().parents[2]
FRESH = REPO / "results" / "v20_r15_it6_seeds8_15.jsonl"
RETAKE = REPO / "results" / "v17k_r4_retake.jsonl"
D_MODEL = 16
NEG_INF = float("-inf")


def _rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _header(rows):
    return next(r for r in rows if r.get("t") == "header")


def _cells(rows, kind="arm_smprime"):
    """The trained end-of-run cells: the rows carrying `eval_nrmse` that are not
    one of the per-step early-warning traces (`step` is set on those)."""
    return sorted((r for r in rows if r.get("kind") == kind
                   and "eval_nrmse" in r and r.get("step") is None),
                  key=lambda r: r["seed"])


def _corpus_sign_census(h):
    """The eval draw's own sign counts on the live band -- NO ARM.

    Built exactly as `scripts/v15_r1.py:698-701` builds it: `head = S-1-T_STAR`,
    `live = range(head+1, S)`, eval batch at the HARDCODED `seed=12345`, which
    is why this quantity cannot depend on the cell seed.
    """
    t_star, n_eval, s = h["t_star"], h["n_eval"], h["s"]
    head = s - 1 - t_star
    live = list(range(head + 1, s))
    x = ns.make_equilibrium_batch(n_eval, s, h["d"], d_model=D_MODEL,
                                  t_star=t_star, seed=12345, device=None)[0]
    a = x[:, live, ns.CH_DRIVE].reshape(-1)
    n = a.numel()
    neg, pos = int((a < 0).sum()), int((a > 0).sum())
    return {"n": n, "neg": neg, "pos": pos,
            "frac_neg": neg / n, "frac_pos": pos / n}


# ------------------------------------------------------------------ CONTROLS
# The instrument validates itself before it is allowed to condemn anything.

def test_control_the_fresh_journal_is_readable_and_has_ten_cells():
    cells = _cells(_rows(FRESH))
    assert [c["seed"] for c in cells] == [0, 1, 8, 9, 10, 11, 12, 13, 14, 15], \
        [c["seed"] for c in cells]
    assert all("lambda_hat" in c and "lambda_hat_live" in c
               and "frac_gate_annihilated" in c for c in cells)


def test_control_seeds_0_and_1_reproduce_the_retake_bitwise():
    """MERCURY's in-run control, verified rather than accepted. If this fails
    the ten cells are not comparable to the retake's eight and every pooled
    reading below is void."""
    fresh = {c["seed"]: c for c in _cells(_rows(FRESH))}
    old = {c["seed"]: c for c in _cells(_rows(RETAKE))}
    for s in (0, 1):
        a, b = fresh[s], old[s]
        shared = [k for k in set(a) & set(b) if k not in ("t", "tag", "secs")]
        diff = [(k, a[k], b[k]) for k in shared
                if a[k] != b[k] and not (isinstance(a[k], float)
                                         and math.isnan(a[k]))]
        assert not diff, (s, diff)


def test_control_lambda_hat_live_is_finite_and_does_vary():
    """The control that makes STRIKE 9 a strike and not a tautology: the SIBLING
    column, computed off the same vector two lines away, is finite and
    discriminating. A dead instrument would flatten both."""
    cells = _cells(_rows(FRESH))
    live = [c["lambda_hat_live"] for c in cells]
    assert all(math.isfinite(v) for v in live), live
    assert len(set(live)) == len(live), live


# -------------------------------------------------------------- STRIKE 9

def test_lambda_hat_carries_more_than_the_bit_frac_gate_annihilated_gt_zero():
    """`scripts/v15_r1.py:383` is `lambda_hat=float(lg.mean())` -- the mean over
    ALL live positions, `-inf` ones included. `:386` is
    `frac_gate_annihilated=float((~fin).double().mean())`. A single non-finite
    entry drags an unweighted mean to `-inf`, so the two columns are the SAME
    PREDICATE and `lambda_hat` has no magnitude left to read.

    `V20_R15_IT5_VENUS.md:128` reads it as a magnitude -- "the arm has no fitted
    decay because the hop has been switched off" -- and
    `V20_R15_IT6_JUPITER.md:280-284` inherits that reading for Q3.
    """
    cells = _cells(_rows(FRESH)) + _cells(_rows(RETAKE))
    same = [c["seed"] for c in cells
            if (c["lambda_hat"] == NEG_INF) == (c["frac_gate_annihilated"] > 0.0)]
    assert len(same) < len(cells), (
        f"on {len(same)}/{len(cells)} cells `lambda_hat == -inf` is EXACTLY the "
        f"predicate `frac_gate_annihilated > 0` -- one zero gate in 8192 forces "
        f"it. The column restates a column published beside it, and the round "
        f"reads it as a rate. The finite sibling `lambda_hat_live` "
        f"(v15_r1.py:384) does discriminate: "
        f"{[(c['seed'], round(c['lambda_hat_live'], 4)) for c in _cells(_rows(FRESH))]}")


def test_unit_root_and_lambda_hat_are_independent_so_neither_contradicts_the_other():
    """The it.5 reading held `unit_root == False` beside `lambda_hat == -inf` to
    be impossible, and seeds 11 and 13 read exactly that. `:387` is
    `unit_root=bool(a_max >= 1.0)` where `a_max = float(lg.max().exp())`. A MAX
    ignores `-inf` unless every entry is `-inf`; a MEAN does not ignore one. The
    pair is not a contradiction, it is the generic case, and an anomaly budget
    spent on it is spent on nothing."""
    cells = _cells(_rows(FRESH))
    both = [(c["seed"], c["a_hat_max"], c["unit_root"]) for c in cells
            if c["lambda_hat"] == NEG_INF and not c["unit_root"]]
    assert not both, (
        f"cells reading `lambda_hat == -inf` AND `unit_root == False`: {both}. "
        f"These two columns share no arithmetic -- one is a mean over every "
        f"position, the other a max over the same vector.")


# -------------------------------------------------------------- STRIKE 10

def test_fresh_seeds_frac_gate_annihilated_is_not_the_corpus_sign_census():
    """M-18's registered check, re-run on cells the corpus argument had never
    seen. The eval draw sits at the hardcoded `seed=12345`
    (`scripts/v15_r1.py:700`) and does not move with the cell seed, so a column
    equal to its sign census is blind to the arm by construction."""
    rows = _rows(FRESH)
    c = _corpus_sign_census(_header(rows))
    corpus = {c["frac_neg"], c["frac_pos"]}
    cells = _cells(rows)
    hits = [(r["seed"], r["frac_gate_annihilated"]) for r in cells
            if r["frac_gate_annihilated"] in corpus]
    assert not hits, (
        f"{len(hits)}/{len(cells)} independently seeded cells publish a gate "
        f"diagnostic equal to a CORPUS-ONLY statistic. corpus: neg={c['neg']} "
        f"({c['frac_neg']!r}) pos={c['pos']} ({c['frac_pos']!r}) of n={c['n']}. "
        f"cells: {hits}")


def test_the_two_escaping_cells_are_not_a_second_corpus_value():
    """Seeds 11 and 13 read `0.99462890625` and `0.976806640625`, values the
    retake's eight never produced. If those were ALSO corpus counts the column
    would be corpus-only outright; they are not, which is what makes them the
    file's only two arm readings -- and both are MORE annihilated, not less.
    `V20_R15_IT5_VENUS.md:240` names the falsifier "fails with a LIVE gate",
    which these two cannot fire."""
    rows = _rows(FRESH)
    c = _corpus_sign_census(_header(rows))
    corpus = {c["frac_neg"], c["frac_pos"]}
    odd = [(r["seed"], r["frac_gate_annihilated"], round(r["eval_nrmse"], 6))
           for r in _cells(rows) if r["frac_gate_annihilated"] not in corpus]
    assert not odd, (
        f"cells whose gate column is an ARM reading rather than the census: "
        f"{odd}. Both fail ABOVE 1.0 while the eight census-valued cells fail "
        f"at 0.85-0.93; the column separates the population it does not "
        f"describe.")
