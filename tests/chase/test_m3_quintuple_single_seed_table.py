"""RED-first bind: the per-cell summary table survives a ONE-seed run.

DEFECT. `scale/m3_quintuple.py::main` computes each cell's across-seed spread as

    sd_ = (sum((v - m) ** 2 for v in ev) / (len(ev) - 1)) ** 0.5

in the PER-CELL SUMMARY TABLE. With one seed behind a cell -- exactly what a
`--cells twin --seeds 0` probe run asks for, and what every future single-unit
smoke run of the cuda lane will ask for -- `len(ev) - 1 == 0` and the run dies
with `ZeroDivisionError` AFTER the unit is journalled: the work is banked, the
summary never prints, and the process exits non-zero on a completed bucket.

THE CONTRACT UNDER TEST.
  1. A single-cell, single-seed run completes: `main()` returns 0, the summary
     table prints, and the spread column reads `n/a` where a variance over one
     observation is undefined.
  2. THE BYTE FORMAT FOR n >= 2 IS UNCHANGED. A five-seed row must print the
     same six-decimal mean/sd columns it always printed, so no downstream grep
     or reader of past logs can move.

ISOLATION. `scale.bucket.RESULTS` and `Q.WEIGHTS_DIR` are repointed at the test
temp dir BEFORE `main()` runs, so neither the registered CPU journal
`results/m3_quintuple_v2.jsonl` nor its weights directory is ever touched; the
line count of the registered journal is asserted unchanged around the run.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import bucket as B                                    # noqa: E402
from scale import m3_quintuple as Q                              # noqa: E402

REGISTERED_JOURNAL = ROOT / "results" / "m3_quintuple_v2.jsonl"


def _run_main(tmp_path, monkeypatch, capsys, seeds: list[int]) -> str:
    """Drive `main()` end to end at a toy geometry, isolated from results/."""
    monkeypatch.setattr(B, "RESULTS", tmp_path)
    monkeypatch.setattr(Q, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(sys, "argv", [
        "m3_quintuple.py",
        "--cells", "twin",
        "--ks", "4",
        "--seeds", *[str(s) for s in seeds],
        "--task", "e3_t1",
        "--s", "48", "--d", "8", "--steps", "2",
        "--n-train", "16", "--n-eval", "16",
        "--t-max", "3",
        "--budget", "600",
    ])
    before = (REGISTERED_JOURNAL.read_text(encoding="utf-8").count("\n")
              if REGISTERED_JOURNAL.exists() else 0)
    try:
        rc = Q.main()
    finally:
        out = capsys.readouterr().out
    after = (REGISTERED_JOURNAL.read_text(encoding="utf-8").count("\n")
             if REGISTERED_JOURNAL.exists() else 0)
    assert after == before, (
        f"the registered CPU journal changed size under a test run "
        f"({before} -> {after} lines)")
    assert rc == 0, f"main() returned {rc}"
    return out


def test_one_cell_one_seed_completes_and_prints_n_a(tmp_path, monkeypatch,
                                                    capsys):
    """The smallest entry point that reproduces the crash: `main()` itself,
    one cell, one seed, toy geometry drawn by the registered builder."""
    out = _run_main(tmp_path, monkeypatch, capsys, [0])

    assert "PER-CELL, ALL FIVE SEEDS" in out, "the summary table never printed"
    rows = [ln for ln in out.splitlines()
            if ln.strip().startswith("twin ") and "n_params" not in ln]
    assert rows, f"no twin summary row in output:\n{out[-2000:]}"
    assert "n/a" in rows[0], (
        f"one-seed row does not carry the n/a spread marker:\n{rows[0]}")

    jl = tmp_path / "m3_quintuple_v2.jsonl"
    assert jl.exists(), "the unit was never journalled"
    lines = [l for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = __import__("json").loads(lines[0])
    assert rec["key"] == ("twin_k4_s48_d8_st2_ntr16_nev16_b3_sd0_taske3_t1")


def test_two_seed_row_keeps_the_legacy_byte_format(tmp_path, monkeypatch,
                                                   capsys):
    """The guard must not touch the n >= 2 path: mean AND sd both print as
    six-decimal columns exactly as every previously published log line."""
    out = _run_main(tmp_path, monkeypatch, capsys, [0, 1])

    rows = [ln for ln in out.splitlines()
            if ln.strip().startswith("twin ") and "n_params" not in ln]
    assert rows, f"no twin summary row in output:\n{out[-2000:]}"
    assert "n/a" not in rows[0], "the n/a marker leaked into an n >= 2 row"

    # columns: cell k <two six-decimal eval NRMSEs> <mean> <sd> <0-step> <params>
    parts = rows[0].split()
    floats = [p for p in parts if p.replace(".", "").replace("-", "").isdigit()]
    six_dec = [p for p in parts
               if "." in p and len(p.split(".")[1]) == 6]
    assert len(six_dec) == 5, (
        f"expected eval x2 + mean + sd + 0-step = 5 six-decimal fields, "
        f"got {six_dec} in:\n{rows[0]}")
