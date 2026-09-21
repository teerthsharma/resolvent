"""C26(1): exact-match floor sweep. Read-only, no GPU, no writes outside
scratchpad. For every exact-match figure this project has published (tests/,
docs/, results/capability.json), report n, 1/n, 3/n, and whether the figure
or any swing quoted around it survives that floor (a swing smaller than 3/n
is not a swing).

    python c26_c27.py

Sources read (all repo-relative, none written):
  results/capability.json          -- softmax/sgate exact_match, n_eval, per_seed
  tests/chase/cogs/cogs_curve.jsonl -- the measured control curve + eval_seconds
  docs/PHASE_I.md, docs/PHASE_I1.md -- the prose-cited swings (-0.11328, 0.113)

L-REPRO: every number below is read straight from the file named next to it,
not recomputed or re-derived, so the sweep cannot itself be a new source of
drift.
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)")


def floor(n: int) -> tuple[float, float]:
    return 1 / n, 3 / n


def survives(value: float, n: int) -> bool:
    """A figure/swing survives the quantisation floor when it is >= 3/n --
    i.e. it represents at least 3 flipped items, not noise at the subsample's
    own resolution."""
    return abs(value) >= 3 / n


def sweep_capability_json(path: pathlib.Path) -> list[dict]:
    rows = []
    d = json.loads(path.read_text(encoding="utf-8"))
    for split, row in d.items():
        if not isinstance(row, dict) or "arms" not in row:
            continue
        for kind, arm in row["arms"].items():
            n = arm.get("n_eval")
            em = arm.get("exact_match")
            if n is None or em is None:
                continue
            f1, f3 = floor(n)
            rows.append(dict(source=path.name, split=split, arm=kind, n=n,
                             value=em, floor_1_over_n=round(f1, 6),
                             floor_3_over_n=round(f3, 6),
                             survives=survives(em, n) if em else True,
                             note="exact_match, not a swing"))
            per_seed = arm.get("per_seed") or []
            if len(per_seed) > 1:
                swing = max(per_seed) - min(per_seed)
                rows.append(dict(source=path.name, split=split,
                                 arm=f"{kind}.per_seed_swing", n=n,
                                 value=round(swing, 6),
                                 floor_1_over_n=round(f1, 6),
                                 floor_3_over_n=round(f3, 6),
                                 survives=survives(swing, n),
                                 note=f"max-min across seeds {per_seed}"))
    return rows


def sweep_cogs_curve(path: pathlib.Path) -> list[dict]:
    """tests/chase/cogs/cogs_curve.jsonl: same seed, four checkpoints. Every
    consecutive-checkpoint delta on `generalization` (n=512) and
    `in_distribution` (n=256), checked against its own split's floor."""
    if not path.exists():
        return []
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for a, b in zip(lines, lines[1:]):
        for field, n in (("generalization", 512), ("in_distribution", 256)):
            delta = b[field] - a[field]
            f1, f3 = floor(n)
            rows.append(dict(source=path.name,
                             split=f"cogs.{field} step{a['step']}->{b['step']}",
                             arm="softmax", n=n, value=round(delta, 6),
                             floor_1_over_n=round(f1, 6),
                             floor_3_over_n=round(f3, 6),
                             survives=survives(delta, n),
                             note="consecutive-checkpoint delta, same seed"))
    return rows


# Prose-cited swings this project already discusses (docs/PHASE_I.md,
# docs/PHASE_I1.md) -- pulled by hand since they live in prose, not JSON, and
# checked against the same floor rule for consistency.
PROSE_CITED = [
    dict(source="docs/PHASE_I.md:467-469", split="cogs.in_distribution",
        arm="softmax reproduction-check miss", n=256, value=-0.11328,
        note="two-sided in-distribution miss that halted the reproduction run"),
    dict(source="docs/PHASE_I1.md:279,508-534",
        split="cogs.generalization (ceq.lm SDPA path)", arm="softmax",
        n=512, value=0.113,
        note="the known instance -- fixed-seed swing across 4 CUDA runs of the "
             "identical recipe; the doc already computes 58 items and the "
             "1/512=0.00195 floor for it"),
]
for r in PROSE_CITED:
    f1, f3 = floor(r["n"])
    r["floor_1_over_n"] = round(f1, 6)
    r["floor_3_over_n"] = round(f3, 6)
    r["survives"] = survives(r["value"], r["n"])


def main() -> None:
    rows = []
    rows += sweep_capability_json(REPO / "results" / "capability.json")
    rows += sweep_cogs_curve(REPO / "tests" / "chase" / "cogs" / "cogs_curve.jsonl")
    rows += PROSE_CITED

    print(f"{'source':34s} {'split/field':38s} {'n':>6s} {'value':>10s} "
          f"{'1/n':>9s} {'3/n':>9s} {'survives':>9s}")
    for r in rows:
        print(f"{r['source']:34s} {r['split']:38s} {r['n']:6d} "
              f"{r['value']:10.5f} {r['floor_1_over_n']:9.5f} "
              f"{r['floor_3_over_n']:9.5f} {str(r['survives']):>9s}")

    out = pathlib.Path(__file__).with_name("c26_sweep.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    borderline = [r for r in rows if r["survives"] and
                  abs(r["value"]) < 1.5 * r["floor_3_over_n"]]
    if borderline:
        print("\nBORDERLINE (survives 3/n but by less than 1.5x):")
        for r in borderline:
            print(f"  {r['source']} {r['split']}/{r['arm']}: "
                  f"{r['value']} vs floor {r['floor_3_over_n']}")


def _selfcheck():
    """The smallest thing that fails if the floor logic breaks."""
    assert floor(512) == (1 / 512, 3 / 512)
    assert survives(0.113, 512) is True          # 57.9 items, well above 3
    assert survives(0.005859375, 512) is True     # exactly 3/512, boundary is inclusive
    assert survives(0.0009765625, 512) is False   # 0.5 items, below floor
    assert survives(-0.11328, 256) is True         # sign must not matter
    print("selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
    main()
