"""A journal scan that cannot silently fail to find what it is searching for.

THE FAILURE THIS EXISTS TO PREVENT, with its cost. At iteration 4 a scan reported
"zero NRMSE readings above 1.0 in any results/*.jsonl", and that claim was used to
strike a colleague's evidence. It was false: 22 of 68 readings are at or above 1.0.
The scan iterated the top level of each record while the journals nest their payload
under `value`, so it never descended and could not have returned a hit on any input.

A search structurally incapable of finding a thing, reporting none, is not evidence
of absence. That was the twelfth vacuous control struck in this campaign and the
first authored here that shipped rather than being caught on re-reading.

WHY THIS IS A MECHANISM RATHER THAN MORE CARE. Eleven earlier vacuous controls were
caught by a colleague or by re-reading; the twelfth was not, and discipline that
works eleven times out of twelve is not a guarantee. `scan_journals` refuses to
report an absence it has not earned: the caller supplies a witness that must be
found, the scan verifies it before returning, and a scan whose own witness is
missing raises. A wrong predicate then fails loudly at the point of use rather than
quietly, two iterations later, inside someone else's argument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

__all__ = ["Hit", "ScanWitnessError", "walk_numbers", "scan_journals"]


class ScanWitnessError(RuntimeError):
    """Raised when a scan's own witness was not found at the value it claims.

    This is the whole point of the module: it converts "the scan found nothing"
    from an unfalsifiable statement into one that had to survive a planted case.
    """


@dataclass(frozen=True)
class Hit:
    journal: str
    key: str
    path: str
    value: float


def walk_numbers(obj, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Every numeric leaf in a nested record, with its dotted path.

    Booleans are excluded deliberately. `isinstance(True, int)` is True in Python,
    so a journal carrying flags would otherwise pollute every numeric scan with
    ones and zeros — and a flag counted as a reading is how a threshold claim goes
    wrong without anyone editing a number.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{prefix}{i}.")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix[:-1], float(obj)


def scan_journals(paths: Iterable[Path], *, key_endswith: str,
                  predicate: Callable[[float], bool],
                  witness: tuple[str, float] | None) -> list[Hit]:
    """Numeric leaves whose dotted path ends `key_endswith` and satisfy `predicate`.

    `witness` is a `(dotted_path, value)` pair that MUST be present among the
    records read, whether or not it satisfies the predicate. Two things are
    required of it, and the second is the one that matters:

      * the traversal must actually reach that path, so a reader that stops at the
        wrong nesting level is caught — that was the shipped defect; and
      * the path must also satisfy `key_endswith`, so a selector that matches
        nothing is caught too.

    The second was missing from the first draft, and its own test found the gap:
    a transposed selector (`nrmse_eval` for `eval_nrmse`) passed, because the
    witness was verified against every path walked rather than against the paths
    the filter would keep. A witness that cannot fail on a wrong selector is
    exactly the vacuous control this module was written to abolish.

    Passing `witness=None` opts out, and is appropriate only where an empty result
    is not itself the claim — a caller reporting an absence must supply one.
    """
    hits: list[Hit] = []
    # Paths seen at all, and (path, value) pairs seen at all. Keeping only the
    # LAST value per path was the first draft and its own test caught it: a
    # witness that appears in an earlier record than the final one would be
    # reported missing, which would fire ScanWitnessError on a correct scan and
    # train the reader to ignore it. A check that cries wolf is the failure this
    # module exists to prevent, one level up.
    seen_paths: set[str] = set()
    seen_pairs: set[tuple[str, float]] = set()
    for p in paths:
        p = Path(p)
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("key", "") if isinstance(rec, dict) else ""
            for path, val in walk_numbers(rec):
                seen_paths.add(path)
                seen_pairs.add((path, val))
                if path.endswith(key_endswith) and predicate(val):
                    hits.append(Hit(p.name, key, path, val))
    if witness is not None:
        wpath, wval = witness
        if wpath not in seen_paths:
            raise ScanWitnessError(
                f"witness path {wpath!r} was never reached: the traversal did not "
                f"visit it, so an empty or partial result is not evidence of "
                f"absence. Paths seen: {sorted(seen_paths)[:8]}")
        if not wpath.endswith(key_endswith):
            raise ScanWitnessError(
                f"witness path {wpath!r} does not satisfy key_endswith="
                f"{key_endswith!r}, so this selector could not have returned the "
                f"witness and an empty result proves nothing about it")
        if (wpath, wval) not in seen_pairs:
            got = sorted(v for p_, v in seen_pairs if p_ == wpath)
            raise ScanWitnessError(
                f"witness {wpath!r} never read {wval!r}; values at that path: "
                f"{got[:8]}")
    return hits
