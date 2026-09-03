"""MARS it.12 — TASK B. The four constants the round asserts with nothing.

Inspector it.11 (`V20_R15_IT11_INSPECTOR.md:59,552,589`): `1.084523` exists in no
`.py`; `+0.7176` and `-0.0324` are "computed nowhere"; `grep -rn spearman` over
`tests/`, `scripts/`, `ceq/` returns nothing for these cells.

The two Spearman figures are the coordinator's, produced in an unbound shell at
it.8 (`V20_R15_JOURNAL.md:1408-1412`) to adjudicate JUPITER against VENUS
(`V20_R15_JOURNAL.md:47`, corrections-index row C11). They settled that dispute.

This file (a) records the absence as a node that is RED on the unmutated repo,
and (b) supplies the replacement route: the figures recomputed from the banked
JSONL, asserted at six decimals, with a planted negative proving the node bites.
"""
import json
import os
import pathlib

import numpy as np
import pytest
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = (
    "results/v17k_r4_floor.jsonl",
    "results/v17k_r4_retake.jsonl",
    "results/v20_r15_it6_seeds8_15.jsonl",
)


def _cells():
    """The 16 arm_smprime cells, seeds 0/1 deduplicated exactly as it.8 states."""
    seen = {}
    for name in SOURCES:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            smp = (rec.get("manifest") or {}).get("smp_values")
            if smp is None:
                continue
            row = (rec["eval_nrmse"], smp["qk"], smp["beta"])
            prior = seen.get(rec["seed"])
            # the dedup is only legitimate because the duplicates are bitwise equal
            assert prior is None or prior == row, f"seed {rec['seed']} differs across files"
            seen[rec["seed"]] = row
    # MARS_MUT plants a negative in the DATA, never in the assertion. `and False`
    # is the V-16 class this round is being audited for; it is not used here.
    mut = os.environ.get("MARS_MUT", "")
    if mut == "swap_qk":                       # exchange two cells' qk ranks
        a, b = seen[2], seen[13]
        seen[2], seen[13] = (a[0], b[1], a[2]), (b[0], a[1], b[2])
    elif mut == "beta_sign":                   # push beta to order the outcome
        seen = {k: (v[0], v[1], v[0]) for k, v in seen.items()}
    return seen


def _rhos(seen, drop=()):
    keys = [k for k in sorted(seen) if k not in drop]
    y = np.array([seen[k][0] for k in keys])
    qk = np.array([seen[k][1] for k in keys])
    beta = np.array([seen[k][2] for k in keys])
    return len(keys), spearmanr(qk, y), spearmanr(beta, y)


def test_the_four_constants_stay_bound_by_some_python_other_than_this_file():
    """ATTACK THAT DID NOT FIRE, converted to a standing guard.

    MARS opened it.12 intending this node as the strike: the Inspector recorded
    (`V20_R15_IT11_INSPECTOR.md:59,552,589`) that `1.084523`, `+0.7176` and
    `-0.0324` were asserted by no Python anywhere. On the unmutated tree at it.12
    the node is GREEN, because JUPITER landed
    `tests/jupiter/test_v20_r15_it12_constants.py` in the same iteration and paid
    the debt. The debt was real; it is no longer open. The node is kept so that
    deleting his file turns it RED again.

    `1.084523`  -> V20_R15_IT6_JUPITER.md:427,494  (WITHDRAWN by his :138-181)
    `0.717647`  -> V20_R15_JOURNAL.md:1409          (adjudicated C11 against VENUS)
    `0.032353`  -> V20_R15_JOURNAL.md:1408,47       (the number that struck the M1 transfer)
    """
    hay = ""
    for sub in ("tests", "ceq", "scripts", "scale"):
        for path in (ROOT / sub).rglob("*.py"):
            if path.name == pathlib.Path(__file__).name:
                continue  # this file is the complaint, not the binding
            hay += path.read_text(encoding="utf-8", errors="ignore")
    missing = [c for c in ("1.084523", "0.717647", "0.032353") if c not in hay]
    assert not missing, (
        f"{len(missing)} of 3 load-bearing constants appear in no .py outside this "
        f"file: {missing}. Each decided a ruling the round still rests on."
    )


def test_it8_adjudication_spearman_recomputed_exactly():
    """REPLACEMENT ROUTE. Binds V20_R15_JOURNAL.md:1408-1412 at six decimals."""
    seen = _cells()
    n, qk, beta = _rhos(seen)
    assert n == 16
    assert round(qk.correlation, 6) == 0.717647
    assert round(qk.pvalue, 6) == 0.001748
    assert round(beta.correlation, 6) == -0.032353
    assert round(beta.pvalue, 6) == 0.905320
    y = {k: v[0] for k, v in seen.items()}
    best = min(y, key=y.get)
    worst = max(y, key=y.get)
    assert (best, round(y[best], 6), round(seen[best][2], 6)) == (2, 0.20392, 1.343933)
    assert (worst, round(y[worst], 6), round(seen[worst][2], 6)) == (13, 1.203324, 1.989505)


def test_the_recompute_node_bites_planted_negative():
    """CONTROL. Without this, the node above is green-only and proves nothing."""
    seen = _cells()
    seen[13] = (0.10, seen[13][1], seen[13][2])  # make the worst cell the best
    _, qk, _ = _rhos(seen)
    assert round(qk.correlation, 6) != 0.717647


def test_the_beta_rho_sign_is_decided_by_a_single_cell():
    """FINDING. `-0.0324` carries C11 and the section-2.2 strike of the M1 transfer.

    Its magnitude is already known to be non-significant (Inspector it.11: Fisher-z
    95% CI [-0.520, +0.471]). Stronger: its SIGN is an artifact of one cell. Drop
    seed 2 — the single crossing cell, the one the round treats as exceptional
    everywhere else — and rho(beta, nrmse) flips to +0.089286, while rho(qk, .)
    survives at +0.657143, p = 0.007770.
    """
    seen = _cells()
    _, qk_all, beta_all = _rhos(seen)
    n15, qk15, beta15 = _rhos(seen, drop=(2,))
    assert n15 == 15
    assert beta_all.correlation < 0.0 < beta15.correlation
    assert round(beta15.correlation, 6) == 0.089286
    assert round(qk15.correlation, 6) == 0.657143 and qk15.pvalue < 0.01


def test_the_dedup_is_an_unbound_preprocessing_choice_worth_0_18_of_rho():
    """FINDING. it.8 states the dedup in prose and no node enforces it.

    The banked record carries 20 smp_values rows, not 16. On the raw 20,
    rho(qk) = +0.538578 and rho(beta) = -0.154312 — the headline moves 0.179069.
    The dedup is correct (duplicates are bitwise equal, asserted in `_cells`); it
    is simply not bound anywhere outside this file.
    """
    raw = []
    for name in SOURCES:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            smp = (rec.get("manifest") or {}).get("smp_values")
            if smp is not None:
                raw.append((rec["eval_nrmse"], smp["qk"], smp["beta"]))
    assert len(raw) == 20
    y = np.array([r[0] for r in raw])
    rho_qk = spearmanr(np.array([r[1] for r in raw]), y).correlation
    rho_b = spearmanr(np.array([r[2] for r in raw]), y).correlation
    assert round(rho_qk, 6) == 0.538578 and round(rho_b, 6) == -0.154312
    assert round(0.717647 - rho_qk, 6) == 0.179069
