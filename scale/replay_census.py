"""Replay EVERY journalled m2 unit, not the one the rotation happened to pick.

The inspector's replay check samples ONE unit per run (`keys[iteration % len]`).
Two consecutive rotations landed on `dense_signed__at_pivots` units and BOTH
failed; a third landed on `pivot_signed__in_P/s2048/b11` and passed. One sample
cannot say whether that is one bad unit or a bad cell. Append-only so a death
leaves evidence -- ADR-001.
"""
import json, pathlib, sys, time
import torch
ROOT = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)")
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
from scale.m2_units import compute, units


def main(argv: list[str] | None = None) -> int:
    """The census run. Under a `__main__` guard because this module WRITES.

    Before this guard existed the body ran at module level, and the first two
    statements were `OUT = pathlib.Path(sys.argv[1])` and `OUT.open("w")`.
    So merely IMPORTING this module read argv[1] -- under pytest, a test file
    path -- and truncated it. That is not hypothetical: it is the measurement
    quoted in tests/loop/test_no_module_writes_a_file_at_import.py, where
    importing this file truncated a 37-byte file and wrote 251 bytes of census
    output into it. A module that destroys a file on import cannot be imported
    by any tool that walks the tree, which is every linter, every type checker
    and every test collector.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m scale.replay_census OUTPUT.txt", file=sys.stderr)
        return 2
    OUT = pathlib.Path(args[0])
    journal = {}
    for line in (ROOT / "results/m2.jsonl").read_text().splitlines():
        if line.strip():
            j = json.loads(line); journal[j["key"]] = j
    u = dict(units())
    keys = sorted(k for k in u if k in journal)
    with OUT.open("w", encoding="utf-8") as f:
        f.write(f"{len(keys)} journalled+live units, threads={torch.get_num_threads()}\n")
        f.flush()
        nbad = 0
        for i, key in enumerate(keys):
            t = time.perf_counter()
            got = compute(u[key]); want = journal[key]["value"]
            same = json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
            fields = [k for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)]
            nbad += (not same)
            f.write(f"[{i:>2}] {'MATCH' if same else 'DRIFT'}  {key}  "
                    f"({time.perf_counter()-t:.1f}s)"
                    + ("" if same else f"  fields={fields}"
                       f"  got={ {k: got.get(k) for k in fields} }"
                       f"  want={ {k: want.get(k) for k in fields} }") + "\n")
            f.flush()
        f.write(f"TOTAL DRIFT: {nbad}/{len(keys)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
