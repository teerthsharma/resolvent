"""MARS / MORIARTY, R9 iteration 3 -- the mechanized catch for the new class.

THE CLASS. **The control constructs its own input, so it certifies the
instrument's predicate over a domain production never chose.**

Every instance passed its own control throughout. Each control was real,
planted and non-degenerate, so vacuity rule 5 was satisfied in all of them. The
failure is one level up: the control entered the instrument BELOW the stage that
selects what the instrument looks at, so the predicate was exercised and the
selection was not. A selection stage that returns nothing then reads as "nothing
is wrong" rather than as "nothing was looked at".

THE CATCH, and why it is static rather than dynamic. The three instances live in
three different subsystems and share no runtime object, so no single fixture can
drive all of them. What they do share is a shape visible in the source: a module
owns a SELECTOR (it walks the filesystem, or picks the objects to test) and owns
a CONTROL, and the control never reaches the selector. `entry_point_report`
below is that check. It is a lead generator, not a verdict -- a module may
legitimately separate them -- so the shipped assertion is the narrow one that
cannot produce a false alarm: the repaired instance stays repaired.

RUN IT DIRECTLY for the survey: `python tests/mars/test_mars_control_entry_point.py`
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: A function is a SELECTOR if it decides WHAT the instrument will look at.
_WALKERS = ("rglob", "glob", "iterdir", "walk", "listdir", "scandir")
#: A function is a CONTROL if its job is to prove the instrument can fire.
_CONTROL_NAMES = ("control", "must_fire", "musfire", "planted", "red")


def _calls(node: ast.AST) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _module_report(path: pathlib.Path) -> dict | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not funcs:
        return None
    selectors = {
        name for name, n in funcs.items()
        if _calls(n) & set(_WALKERS)
        or name.startswith("collect_") or name.endswith("_targets")
    }
    controls = {
        name for name in funcs
        if any(k in name.lower() for k in _CONTROL_NAMES)
    }
    if not (selectors and controls):
        return None

    def reaches(start: str, seen=None) -> bool:
        """Does `start` reach any selector, following intra-module calls?"""
        seen = seen or set()
        if start in seen:
            return False
        seen.add(start)
        called = _calls(funcs[start]) if start in funcs else set()
        if called & selectors:
            return True
        return any(reaches(c, seen) for c in called if c in funcs)

    blind = sorted(c for c in controls if not reaches(c))
    return dict(path=path.relative_to(ROOT).as_posix(), selectors=sorted(selectors),
                controls=sorted(controls), blind=blind)


def entry_point_report(root: pathlib.Path | None = None) -> list[dict]:
    """Every module owning both a selector and a control, with the controls that
    never reach the selector."""
    root = root or ROOT
    out = []
    for p in sorted((root / "scale").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        r = _module_report(p)
        if r and r["blind"]:
            out.append(r)
    return out


# ------------------------------------------------------------------- the catch --
def _selector_has_a_control_somewhere(selector: str) -> list[str]:
    """Files under `tests/` that call `selector` by name."""
    hits = []
    for p in sorted((ROOT / "tests").rglob("test_*.py")):
        if "__pycache__" in p.parts:
            continue
        if selector + "(" in p.read_text(encoding="utf-8", errors="replace"):
            hits.append(p.relative_to(ROOT).as_posix())
    return hits


def test_the_repaired_instance_stays_repaired():
    """INSTANCE A, and the only assertion this file ships as a gate.

    `scale/chase_struck_coverage.py` at `8b40e16^` printed `SCANNING 0 PATHS`,
    `uncovered .md: 0, uncovered .py: 0` and returned 0, while its own
    `control()` returned True -- re-verified by executing that file's blob in
    this worktree: 375 candidates, the shipped absolute-parts filter kept **0**,
    the relative-parts filter kept **374**.

    THE FIRST VERSION OF THIS GATE WAS A FALSE ALARM AND IS WORTH RECORDING.
    It asserted that the module's OWN `control()` reaches `collect_targets`. It
    still does not -- `control()` at HEAD is byte-identical to the pre-fix one --
    and the gate failed on a file that is correctly repaired, because the
    coverage control Jupiter added lives in `tests/jupiter/`, not in the module.
    The invariant is not "the module's control reaches the selector"; it is
    "the selector is driven by SOME control", and where that control lives is
    not the point. A catch that legislates the location rather than the
    coverage is the same mistake one level along.
    """
    r = _module_report(ROOT / "scale" / "chase_struck_coverage.py")
    assert r is not None, "the module no longer owns both a selector and a control"
    assert "collect_targets" in r["selectors"], r["selectors"]
    covering = _selector_has_a_control_somewhere("collect_targets")
    assert covering, (
        "the target selector has no control anywhere in tests/: the stage that "
        "returned 0 of 375 paths would fail silently again")


def test_the_catch_fires_on_the_shape_it_is_written_for():
    """RED. A catch that only ever saw the repaired file could not tell the two
    apart. This builds the pre-fix shape in memory -- a module whose control
    feeds the matcher directly and never calls the walker -- and requires the
    report to name it.
    """
    src = (
        "import pathlib\n"
        "def scan_text(t):\n    return '1.471448' in t\n"
        "def collect_targets(root):\n    return list(pathlib.Path(root).rglob('*.md'))\n"
        "def control():\n    return scan_text('the tail norm is 1.471448 flat out')\n"
    )
    tmp = ROOT / "scale" / "_mars_red_probe.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        r = _module_report(tmp)
        assert r is not None
        assert r["blind"] == ["control"], r
        #: and the GREEN half: routing the control through the selector clears it.
        tmp.write_text(src.replace(
            "def control():\n    return scan_text(",
            "def control():\n    collect_targets('.')\n    return scan_text("),
            encoding="utf-8")
        assert _module_report(tmp)["blind"] == []
    finally:
        tmp.unlink(missing_ok=True)


def test_the_survey_is_a_lead_generator_and_not_a_verdict():
    """The repo-wide survey runs and returns structured leads. Deliberately NOT
    asserted empty: separating a selector from a control is legitimate, as the
    repaired instance itself shows. Every lead must still be covered.
    """
    rows = entry_point_report()
    assert isinstance(rows, list)
    for r in rows:
        assert r["blind"] and r["selectors"]
        for sel in r["selectors"]:
            assert _selector_has_a_control_somewhere(sel), (r["path"], sel)


def test_instance_B_the_published_interval_binder_does_not_scan_the_documents():
    """INSTANCE B -- Neptune's finding, re-verified here by execution rather
    than taken from his report.

    `tests/cameron/test_published_intervals_have_producers.py` exists to stop a
    published interval from losing its estimator. Its subject list is the literal
    dict `PUBLISHED` at `:67` and it never opens a document, so an interval added
    to any of the three files it names is neither bound nor flagged -- the exact
    condition the file exists to end.

    Measured: the binder is parameterised over 3 intervals; the three documents
    it names print 45 distinct `[lo, hi]` pairs. 45 is an UPPER BOUND on the
    population -- not every bracketed pair is a bootstrap interval -- but 3 is
    exact, and the mechanism does not depend on the denominator.
    """
    src = (ROOT / "tests" / "cameron"
           / "test_published_intervals_have_producers.py").read_text(
        encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    published = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "PUBLISHED" for t in n.targets)]
    assert published, "PUBLISHED is no longer a module-level literal"
    n_bound = len(published[0].value.keys)

    #: The selection stage that does not exist: no document is opened anywhere.
    assert not (set(_calls(tree)) & {"read_text", "rglob", "iterdir", "open"}), (
        "the binder now reads documents; this instance is repaired")

    import re
    pat = re.compile(r"\[\s*([+-]?\d+\.\d{4,})\s*,\s*([+-]?\d+\.\d{4,})\s*\]")
    printed = 0
    for doc in ("README.md", "ceq/hf_artifact/README.md", "CHECKLIST.md"):
        p = ROOT / doc
        if p.exists():
            printed += len(set(pat.findall(
                p.read_text(encoding="utf-8", errors="replace"))))
    assert n_bound == 3, n_bound
    assert printed > 10 * n_bound, (n_bound, printed)


if __name__ == "__main__":                                   # pragma: no cover
    rows = entry_point_report()
    print(f"=== CONTROL-ENTRY-POINT SURVEY: {len(rows)} module(s) whose control "
          f"never reaches its own selector ===")
    for r in rows:
        print(f"  {r['path']}")
        print(f"      selectors {r['selectors']}")
        print(f"      blind controls {r['blind']}")
    if not rows:
        print("  none -- every module owning both routes its control through "
              "the selector")
