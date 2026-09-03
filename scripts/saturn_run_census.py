"""THE `[RUN]` CENSUS (it.12, SATURN).

The it.11 Inspector's systemic count -- `92 of 153 [RUN] markers name nothing
runnable` -- was a paragraph.  This makes it a command.

A marker is RUNNABLE iff the marker text, or the line carrying it, names a
thing a reader can execute AND that thing exists on disk: a `tests/…::node`
id, a `.py`/`.sh`/`.lean` path under the tree, or a literal `pytest`/`python`/
`bash` command line.  Everything else -- `[RUN: +20.87 on path-product data]`,
`[RUN anchors: …]`, `[RUN]` followed by a number -- is a printed result with
no route back to its producer, which is the class being counted.

No office is exempt.  SATURN's own it.2 and it.3 markers are in the
denominator; both were struck by the Inspector and both are counted here.

    python scripts/saturn_run_census.py
"""
from __future__ import annotations

import pathlib
import re
import sys

#: `[RUN]`, `[RUN: ...]`, `[RUN anchors: ...]` -- the marker and its payload.
MARKER = re.compile(r"\[RUN\b[^\]]*\]")
#: A node id or a path that a reader could paste into a shell.
NODE = re.compile(r"(?:tests|scripts|ceq|scale|lean)/[\w./-]+\.(?:py|sh|lean)"
                  r"(?:::[\w:\[\]-]+)?")
CMD = re.compile(r"\b(?:pytest|python -m pytest|python |bash )\S")
REPORT = re.compile(r"^V20_R15_IT(\d+[_\dA-Z]*)_([A-Z_]+)\.md$")


def _runnable(marker: str, line: str, root: pathlib.Path) -> bool:
    for blob in (marker, line):
        for m in NODE.finditer(blob):
            if (root / m.group(0).split("::")[0]).is_file():
                return True
        if CMD.search(blob) and NODE.search(blob):
            return True
    return False


def census(root: pathlib.Path) -> dict:
    by_office: dict[str, dict[str, int]] = {}
    markers = runnable = 0
    unrunnable_sites: list[str] = []
    for path in sorted(root.glob("V20_R15_IT*.md")):
        m = REPORT.match(path.name)
        if not m:
            continue
        office = f"{m.group(2)}@it{m.group(1)}"
        slot = by_office.setdefault(office, {"markers": 0, "runnable": 0})
        for i, line in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(), 1):
            for mk in MARKER.finditer(line):
                markers += 1
                slot["markers"] += 1
                if _runnable(mk.group(0), line, root):
                    runnable += 1
                    slot["runnable"] += 1
                else:
                    unrunnable_sites.append(f"{path.name}:{i}")
    return {
        "markers": markers,
        "runnable": runnable,
        "unrunnable": markers - runnable,
        "by_office": by_office,
        "unrunnable_sites": unrunnable_sites,
    }


def mutation_reds(root: pathlib.Path) -> list[str]:
    """REDs made by mutating the ASSERTION rather than the code.  `and False`
    appended to a passing predicate is a red against the test, not the tree."""
    out = []
    pat = re.compile(r"assert[^\n#]*\b(?:and False|or True)\b")
    for path in sorted((root / "tests").rglob("test_*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(), 1):
            if pat.search(line):
                out.append(f"{path.relative_to(root).as_posix()}:{i}")
    return out


def red_evidence(root: pathlib.Path) -> dict:
    """Nodes cited in reports, against nodes ever claimed RED.  A node that
    only ever appears green is GREEN-ONLY: nothing shows it can fail."""
    cited: set[str] = set()
    red: set[str] = set()
    node = re.compile(r"[\w./-]+\.py::[\w:\[\]-]+")
    for path in sorted(root.glob("V20_R15_IT*.md")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            for n in node.finditer(line):
                cited.add(n.group(0))
                window = " ".join(lines[max(0, i - 6):i + 7])
                if re.search(r"\bRED\b|\bFAILED\b|\bfailed\b|E\s+assert", window):
                    red.add(n.group(0))
    return {"cited": len(cited), "with_red": len(red),
            "green_only": len(cited) - len(red)}


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    rep = census(root)
    print(f"[RUN] markers            {rep['markers']}")
    print(f"  runnable (resolves)    {rep['runnable']}")
    print(f"  names nothing runnable {rep['unrunnable']}")
    ev = red_evidence(root)
    print(f"test nodes cited         {ev['cited']}")
    print(f"  ever shown RED         {ev['with_red']}")
    print(f"  GREEN-ONLY             {ev['green_only']}")
    mut = mutation_reds(root)
    print(f"REDs made by mutating the assertion: {len(mut)}")
    for s in mut:
        print(f"  {s}")
    print("\nper office (markers / runnable):")
    for office, slot in sorted(rep["by_office"].items(),
                               key=lambda kv: -kv[1]["markers"]):
        print(f"  {office:24s} {slot['markers']:3d} / {slot['runnable']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
