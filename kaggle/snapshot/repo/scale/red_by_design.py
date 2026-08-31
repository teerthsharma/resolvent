"""Route 3 of T-b: a red is BY DESIGN when a DECLARATION covering it says so.

WHY THIS FILE EXISTS. The iteration-2 T-b enumerated two routes to red-by-design --
the `tests/chase/conftest.py:285` KNOWN_RED ledger and a `test_claim_*` prefix -- and
both are NAME routes. `tests/foreman/conftest.py` is a sys.path shim with no ledger at
all, and the chase ledger is scoped to `tests/chase`, so every foreman red is
unledgered by design. MERCURY measured 13 `tests/foreman` files carrying an in-band
`# CLAIM AS WRITTEN -- RED` banner of which only 12 also use the `test_claim_` prefix:
banner and prefix are DIFFERENT SETS. A classifier keyed on the name cannot see the
banner. Three iteration-2 rows were failed by that blind spot and the census was
rejected on two of them.

WHAT A DECLARATION IS. A present-tense statement, IN BAND, that this artifact's red is
its deliverable. Present tense is the whole of it: `RED-first` is the single largest RED
form in the tree (68 lines across `tests/`, four times every declaration form combined)
and it records AUTHORING ORDER -- the test was written before the fix. A red-first test
that has since been fixed is green; one that is red TODAY is a defect, and route 3 must
not rescue it. `must FAIL` is excluded for the same reason in the other direction: it
states what the SUBJECT must do, and it is asserted by a GREEN test.

THE THREE BANDS, narrowest first. A test is covered by the narrowest band that exists
over it, and a narrower non-declaring band OVERRIDES a wider declaring one:

  T  the test function's own docstring;
  B  the section banner it sits under -- the nearest column-0 comment banner above it,
     running to the next banner or EOF;
  F  the module docstring.

The override is what gives route 3 its rejection region and it is not theoretical.
`tests/foreman/test_r1_settling.py` opens a `# CLAIM AS WRITTEN -- RED` section at :56
and CLOSES it at :165 with `# CALIBRATION AND INSTRUMENTS -- GREEN`. Ten of the
thirteen banner files are built this way. A failure below the GREEN banner is not
rescued, in the file route 3 exists to rescue.

WHAT THIS ROUTE COSTS, stated because it cuts against the change. The override reads any
framed column-0 banner as scoping, so a file whose module docstring declares wholesale
but which uses framed banners for mere layout will have its tests read as UNDECLARED and
its reds FAIL. That error runs toward rejecting the sheet, which is the direction an
audit instrument should fail in, and `--survey` reports the size of it rather than
hiding it.

ponytail: regex over three bands, not a semantic read. It is exact on the forms measured
in this tree (`--survey` prints them and their counts); a NEW declaration form invented
after this file is written will read as undeclared until it is added here, and the demo
pins the forms so that adding one cannot silently widen the rescue.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A present-tense declaration that this artifact's RED is its deliverable.
#: Case-sensitive on the token RED: the tree's convention is uppercase for a verdict.
DECLARE = re.compile(
    r"CLAIM AS WRITTEN -- RED"          # foreman section banner (13 files)
    r"|RED (?:ON PURPOSE|on purpose)"   # chase, cameron; module and test docstrings
    r"|ALL RED"                         # chase/test_m2_instrument_binds.py:1
    r"|CURRENTLY RED"                   # chase/test_m2_instrument_binds.py, m2_verdict_nan
    r"|\b[Ii]s RED\b|\bIS RED\b"        # cameron/test_r3,r4,r6: "It is RED."
    r"|^RED[.,]"                        # docstring opening on the verdict
    r"|RED[^.\n]{0,40}by design"        # cameron/test_harmonic_attribution.py:358
    r"|RED tests\b"                     # cameron/test_r5_aggregator_red.py:1,:80
    r"|RED is its deliverable",
    re.M,
)

#: Authoring order, not present state. Excluded, and the exclusion is the load-bearing
#: half of route 3: without it the route rescues most of `tests/chase` and certifies
#: nothing.
NOT_DECLARE = re.compile(r"RED[- ]first|written first \(RED on purpose\)", re.I)

RULE = re.compile(r"^#\s*[=-]{5,}\s*$")          # a bare rule line
INLINE_RULE = re.compile(r"^#.*[=-]{5,}")        # a rule with the title on the line


#: A backticked span is a CITATION, not an utterance. The repo's own convention, and
#: the same normalisation `scale/spotcheck_draw2.py:rule_key` applies to reason cells.
#: Without this, `tests/loop/test_attic_never_removes_the_last_must_fire.py` rescues
#: itself by QUOTING the foreman banner in its module docstring -- caught by the demo
#: below, which is the reason the demo exists.
QUOTED = re.compile(r"`[^`]*`", re.S)


def declares(text: str) -> bool:
    """True iff `text` declares, in the present tense, that its red is the deliverable."""
    text = QUOTED.sub("@", text)
    return bool(DECLARE.search(text)) and not NOT_DECLARE.search(text)


def banners(lines: list[str]) -> list[tuple[int, bool]]:
    """(lineno, declares) for every column-0 comment that is a framed SECTION banner.

    Framed means: the comment carries an inline rule of >=5 '=' or '-', or it is
    adjacent to a bare rule line. Both forms are in use; `#: MEASURED ...` notes and
    `# noqa` are neither, and do not scope anything.
    """
    out = []
    for i, ln in enumerate(lines):
        if not ln.startswith("#") or RULE.match(ln):
            continue
        above = lines[i - 1] if i else ""
        below = lines[i + 1] if i + 1 < len(lines) else ""
        framed = (INLINE_RULE.match(ln) or RULE.match(above) or RULE.match(below))
        if not framed:
            continue
        title = ln.lstrip("#").strip(" =-\t")
        if title:
            out.append((i + 1, declares(title)))
    return out


def tests_in(path: pathlib.Path):
    """(name, first_line, own_docstring) per collected top-level test function."""
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    lines = src.splitlines()
    mod_doc = ast.get_docstring(tree) or ""
    secs = banners(lines)
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        top = min([d.lineno for d in node.decorator_list] + [node.lineno])
        out.append((node.name, top, ast.get_docstring(node) or ""))
    return mod_doc, secs, out


def band(path: pathlib.Path, name: str) -> tuple[bool, str]:
    """(by_design, band) for one test function. band is T / B / F / '-' (uncovered)."""
    mod_doc, secs, funcs = tests_in(path)
    hit = [f for f in funcs if f[0] == name]
    if not hit:
        return False, "?"                      # not a top-level def we can see
    _n, line, doc = hit[0]
    if declares(doc):
        return True, "T"
    prior = [d for ln, d in secs if ln < line]
    if prior:                                  # the narrowest band that exists decides
        return prior[-1], "B"
    return declares(mod_doc), "F"


# --------------------------------------------------------------- the whole of T-b

def known_red(path: pathlib.Path) -> dict:
    """The chase ledger, read from the conftest that owns it."""
    if "chase" not in path.parts:
        return {}
    ns: dict = {}
    src = (ROOT / "tests" / "chase" / "conftest.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "KNOWN_RED":
            return ast.literal_eval(node.value)
    return ns


def route(path: pathlib.Path, name: str) -> tuple[bool, str]:
    """Which route, if any, makes this failure red-by-design. Routes are tried in
    order and the FIRST that fires is reported, so a rescue always names its ground."""
    bare = name.split("[")[0]
    if bare in known_red(path) or name in known_red(path):
        return True, "R1 chase KNOWN_RED ledger"
    if bare.startswith("test_claim_"):
        return True, "R2 test_claim_ refutation instrument"
    ok, b = band(path, bare)
    return (ok, f"R3 declaration, band {b}") if ok else (False, f"NOT by design (band {b})")


# --------------------------------------------------------------------- reporting

def survey(paths) -> None:
    """The structural rejection region: every test function route 3 does NOT rescue."""
    tot = cov = 0
    per_band: dict[str, int] = {}
    uncovered_files: list[tuple[str, int, int]] = []
    for p in paths:
        try:
            _m, _s, funcs = tests_in(p)
        except SyntaxError:
            continue
        n = u = 0
        for name, _l, _d in funcs:
            ok, b = band(p, name)
            tot += 1
            n += 1
            per_band[b if ok else "-"] = per_band.get(b if ok else "-", 0) + 1
            if ok:
                cov += 1
            else:
                u += 1
        if n:
            uncovered_files.append((str(p).replace("\\", "/"), n, u))
    rescued = tot - per_band.get("-", 0)
    print(f"test functions seen: {tot}")
    print(f"route 3 rescues:     {rescued}  ({rescued / tot:.1%})")
    print(f"route 3 REJECTS:     {per_band.get('-', 0)}  "
          f"({per_band.get('-', 0) / tot:.1%})  <- the rejection region")
    print(f"by band: {dict(sorted(per_band.items()))}")
    part = [f for f in uncovered_files if 0 < f[2] < f[1]]
    print(f"files route 3 splits (some tests rescued, some not): {len(part)}")
    for f, n, u in sorted(part)[:20]:
        print(f"  {f}: {n - u}/{n} rescued, {u} REJECTED")


def demo() -> None:
    """Pins the ground truths that make route 3 non-vacuous. Each assert is a case
    the iteration-2 T-b got wrong, or a case route 3 must keep getting wrong."""
    # NOT a drawn row: the demo may not pin its semantics on a file whose disposition
    # this iteration is about to rule on. `test_topology_washout.py` is the stronger
    # case anyway -- it runs GREEN, then RED, then GREEN, so it pins that the banner
    # scopes FORWARD ONLY and that a declaring banner does not bleed backward.
    f = ROOT / "tests" / "foreman" / "test_topology_washout.py"
    assert band(f, "test_instrument_permutation_invariance") == (False, "B"), \
        "a RED banner at :137 rescues a test at :89 ABOVE it; the band is not scoped"
    assert band(f, "test_claim_the_circle_survives_the_occupancy_operator") == (True, "B"), \
        "the foreman RED banner does not rescue its own section"
    # ... and the GREEN section BELOW the same banner is not rescued. This is the
    # rejection region inside the very files route 3 exists to rescue.
    assert band(f, "test_the_resolvent_reads_the_graph_while_the_barcode_reads_the_metric") \
        == (False, "B"), \
        "route 3 rescues a test under a '-- GREEN' banner: it has no rejection region"

    g = ROOT / "tests" / "chase" / "test_structural_zero_guard.py"
    assert band(g, "test_m2_not_in_P_arm_is_a_structural_zero") == (True, "T"), \
        "a test whose own docstring says RED ON PURPOSE is not rescued"
    assert band(g, "test_m4_eviction_control_is_a_structural_zero") == (True, "T")
    assert band(g, "test_calibration_m2_in_P_control_can_pass")[0] is False, \
        "route 3 rescues a calibration test the file itself declares must be GREEN"

    # authoring order is not a declaration
    assert not declares("RED-first bind for D-1, D-2 and H-1 in `PREREGISTRATION_HOLE_AUDIT.md`.")
    assert not declares('"""RED first. A notebook that will not open is a notebook nobody runs."""')
    assert not declares("the affine update must fail it.")
    # present-tense declaration is
    assert declares("CHECKLIST kill clauses that cannot fire. RED ON PURPOSE.")
    assert declares("# CLAIM AS WRITTEN -- RED")
    assert declares("It is RED. Measured on this file, 24 contexts")
    assert declares("CAMERON round 5 -- the RED tests behind the aggregator finding.")

    # the banner form must NOT fire from inside a docstring that merely quotes it,
    # or the loop meta-test rescues itself by talking about the banner.
    q = ROOT / "tests" / "loop" / "test_attic_never_removes_the_last_must_fire.py"
    _m, _s, funcs = tests_in(q)
    assert funcs, "no tests parsed out of the meta-test"
    assert all(band(q, n)[0] is False for n, _l, _d in funcs), \
        "quoting the banner in prose rescues the quoting file"
    print("demo OK: route 3 rescues its sections, rejects the GREEN half of the same "
          "files, and is not fired by RED-first or by quoted banners")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--exclude", default="", help="comma-separated paths held out")
    ap.add_argument("nodeid", nargs="*", help="path::testname to classify")
    ns = ap.parse_args()
    if ns.demo:
        demo()
    elif ns.survey:
        held = {x.strip() for x in ns.exclude.split(",") if x.strip()}
        paths = [p for p in sorted(ROOT.glob("tests/**/test_*.py"))
                 if str(p.relative_to(ROOT)).replace("\\", "/") not in held]
        survey(paths)
    else:
        for nid in ns.nodeid:
            p, _, n = nid.partition("::")
            ok, why = route(ROOT / p, n)
            print(f"{'BY-DESIGN' if ok else 'FAIL     '}  {nid}  [{why}]")
        sys.exit(0)
