"""it.13 MERCURY — the Phase C cost basis, read off the cells rather than asserted.

Nothing here interprets. It reads `secs` out of the journals, evaluates two
formulas other offices wrote, and counts files. Every figure in
V20_R15_IT13_MERCURY.md that is arithmetic comes from this module.
"""
from __future__ import annotations

import json
import pathlib
import re

HASH = "5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309"

#: The 40 cells JUPITER prices a re-take of at V20_R15_IT9_JUPITER.md:239.
#: retake 24 + it.6 seeds 8-15 + it.8 seeds 8-15. The seed>=8 filter drops the
#: 0/1 in-run reproduction controls, which were never part of the 40.
BANKED_40 = (
    ("results/v17k_r4_retake.jsonl", 0),
    ("results/v20_r15_it6_seeds8_15.jsonl", 8),
    ("results/v20_r15_it8_armpl_b.jsonl", 8),
)

#: Measured per-cell means at s=64, retake subset, one process, one device.
MEAN_SECS = {"arm_smprime": 16.161, "arm_pl": 1.780, "softmax": 1.681}


def banked_cells(root: pathlib.Path) -> list[dict]:
    out = []
    for rel, min_seed in BANKED_40:
        path = root / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("t") == "cell" and "secs" in r and r.get("seed", -1) >= min_seed:
                r["_file"] = rel
                out.append(r)
    return out


def _sweep(exponent: float) -> float:
    """JUPITER's own shape: 3 seeds, S in {32,64,128}, 3 arms, cost scaling S**e
    relative to the measured s=64 point."""
    weights = sum((s / 64.0) ** exponent for s in (32, 64, 128))
    return 3 * weights * sum(MEAN_SECS.values())


def jupiter_it8_formula() -> float:
    """`3 x (0.25 + 1 + 4) x (16.16 + 1.78 + 1.68)`, V20_R15_IT8_JUPITER.md:216,
    labelled there as 275 GPU-s."""
    return _sweep(2.0)


def s_exponent_band() -> tuple[float, float, float]:
    """The exponent is the unknown the sweep exists to measure, so the price of
    the sweep is a band, not a point."""
    return _sweep(1.0), _sweep(2.0), _sweep(3.0)


def hash_reader_census(root: pathlib.Path) -> dict[str, list[str]]:
    """Files that mention `instrument_hash`, per directory. SATURN's split."""
    out = {}
    for d in ("ceq", "scale", "scripts"):
        hits = []
        for p in sorted((root / d).rglob("*.py")):
            if "instrument_hash" in p.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(p.relative_to(root)).replace("\\", "/"))
        out[d] = hits
    return out


#: Words that a chess bed cannot be built without.
_CHESS = re.compile(
    r"\bchess\b|\bfen\b|\bpgn\b|stockfish|\buci\b|en_passant|castling|legal_move",
    re.IGNORECASE,
)


def chess_producer_census(root: pathlib.Path) -> dict[str, list[str]]:
    """Clause (4)'s bed, searched for as code and as corpus."""
    code, corpus, control = [], [], []
    for d in ("ceq", "scale", "scripts", "tests", "colab", "kaggle"):
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if _CHESS.search(p.read_text(encoding="utf-8", errors="ignore")):
                code.append(str(p.relative_to(root)).replace("\\", "/"))
            if p.name == "v15_r1.py":
                control.append(str(p.relative_to(root)).replace("\\", "/"))
    for d in ("data", "results"):
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and _CHESS.search(p.name):
                corpus.append(str(p.relative_to(root)).replace("\\", "/"))
    from ceq import kdata
    ev = [p for p in sorted((root / "ceq").rglob("*.py"))
          if re.search(r"eval_delta|cp_delta|centipawn", p.read_text(encoding="utf-8", errors="ignore"))]
    eng = [p for p in sorted(root.rglob("*.py")) if not str(p).replace("\\", "/").endswith(("kaggle/snapshot", "phase_c_price.py"))
           and "kaggle/snapshot/" not in str(p).replace("\\", "/")
           and re.search(r"stockfish|SimpleEngine|popen_uci", p.read_text(encoding="utf-8", errors="ignore"))]
    consumers = [p for p in sorted((root / "scripts").rglob("*.py"))
                 if re.search(r"label_plies|iter_games|split_by_game", p.read_text(encoding="utf-8", errors="ignore"))]
    return {"code": code, "corpus": corpus, "control_v15_r1": control,
            "registered_beds": list(kdata.BED_SPECS),
            "eval_delta_producers": [str(p.relative_to(root)) for p in ev],
            "engines": [str(p.relative_to(root)) for p in eng],
            "arm_consumers": [str(p.relative_to(root)) for p in consumers]}


def chess_oracle_cost(root: pathlib.Path) -> tuple[int, float]:
    """Legality + next-FEN, recomputed. GPU-seconds: zero, it is python-chess."""
    import time

    from ceq import kdata

    fx = root / "tests" / "gate0" / "fixtures" / "games.pgn"
    t0 = time.perf_counter()
    rows = []
    for _key, game in kdata.iter_games(fx):
        rows += kdata.label_plies(game)
    return len(rows), time.perf_counter() - t0
