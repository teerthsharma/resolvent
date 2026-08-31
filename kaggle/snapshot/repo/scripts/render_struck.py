"""Render STRUCK.md from the STRUCK registry. `python scripts/render_struck.py > STRUCK.md`.

STRUCK.md is a POINTER to `tests/loop/test_no_struck_constant_ships.py:47`, not a
second copy of it: two hand-maintained lists of struck numbers is how a strike
gets dropped from one and survives in the other. This script is what keeps the
rendered table honest -- edit the dict, re-run this, never edit the table.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = ROOT / "tests/loop/test_no_struck_constant_ships.py"

HEADER = """# STRUCK

Every constant this project has struck: withdrawn from the record, never
silently replaced. A number enters the list the moment an audit strikes it.

**This file is a pointer, not a copy.** The registry is a module-level dict in
`tests/loop/test_no_struck_constant_ships.py:47`, asserted non-empty by
`inspector.py:459` and scanned against the shipped code and the lead documents
by that same test -- a second hand-maintained strike list is exactly how a
strike gets dropped from one copy and survives in the other.

Regenerate this file from the registry; never edit the table by hand:

```
python scripts/render_struck.py > STRUCK.md
```
"""


def main() -> None:
    # the registry text is UTF-8 (em dashes); Windows stdout defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    src = REG.read_text(encoding="utf-8").splitlines()
    spec = importlib.util.spec_from_file_location("_struck_registry", REG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # the literal as written, and the line it is written on, so the cite resolves
    at = {}
    for i, line in enumerate(src):
        m = re.match(r"    (-?\d[\d.eE+-]*):", line)
        if m:
            at[float(m.group(1))] = (i + 1, m.group(1))

    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
    out = [HEADER,
           "Rendered from `{}` at `{}`. {} entries.\n".format(
               REG.relative_to(ROOT).as_posix(), rev, len(mod.STRUCK)),
           "| constant | cite | why it was struck |",
           "| --- | --- | --- |"]
    for k, why in mod.STRUCK.items():
        ln, lit = at.get(k, (47, repr(k)))
        out.append("| `{}` | `{}:{}` | {} |".format(
            lit, REG.relative_to(ROOT).as_posix(), ln,
            " ".join(why.split()).replace("|", "\|")))
    print("\n".join(out))


if __name__ == "__main__":
    main()
