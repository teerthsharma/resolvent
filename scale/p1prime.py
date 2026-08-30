"""P1' measurement. Does the plant enter at the instrument's front door?

P1 (replaced) keyed on WHO built the input. Every planted positive is constructed;
that is what planting means, so P1 fired on cures as readily as diseases.

P1' keys on WHERE the plant ENTERS. A control is vacuous-by-scope when its planted
input enters BELOW the front door, so stages the real input traverses go unexercised.
Measured here, per file, with the stages named.
"""
import ast, json, pathlib, re, subprocess, sys

# Provenance for a shipped number lands in the repo, never in temp.
# Rescued from the session scratchpad (R10 P0 it.2); paths are now repo-relative.
R = pathlib.Path(__file__).resolve().parent.parent
OUT = R / "results"

# THE INPUT IS PINNED, R10 P0 it.3. This read the WORKING TREE, so its output drifted
# with every test file anyone added: the sheet recorded 191/46/145, the Inspector
# re-ran it mid-round and got 193/46/147, and by it.3 the live tree held 202 test
# files. None of those is a disagreement about a row -- they are three different
# denominators. A census is OF A COMMIT. `scale/spotcheck_draw.py` has the same
# shape of defect and the same shape of fix: there the seed was pinned so the draw
# could not be re-rolled, here the tree is pinned so the census cannot be re-taken.
#
# 06a180c is the commit that wrote 191/46/145 into AUDIT.md. Pass a rev to re-take
# the census somewhere else -- `python scale/p1prime.py HEAD` measures the drift --
# but the shipped number is the one at PIN.
PIN = "06a180c"
REV = next((a for a in sys.argv[1:] if not a.startswith("-")), PIN)


def _git(*args, text=True):
    return subprocess.run(("git",) + args, cwd=R, capture_output=True,
                          text=text, check=True).stdout


def tree_files(rev):
    """Every tracked `tests/**.py` at `rev`. The path restriction is what used to
    need a `.claude/worktrees` filter; a worktree cannot appear under `tests/`."""
    return sorted(p for p in _git("ls-tree", "-r", "--name-only", rev, "--",
                                  "tests/").splitlines() if p.endswith(".py"))


def blobs(rev, paths):
    """Contents of every path at `rev`, in ONE `git cat-file --batch`.

    One subprocess rather than one per file: 191 `git show` spawns cost seconds on
    Windows and buy nothing. The batch protocol is `<sha> blob <size>\\n<bytes>\\n`.
    """
    stdin = "".join("%s:%s\n" % (rev, p) for p in paths).encode()
    raw = subprocess.run(["git", "cat-file", "--batch"], cwd=R, input=stdin,
                         capture_output=True, check=True).stdout
    out, i = {}, 0
    for p in paths:
        j = raw.index(b"\n", i)
        header = raw[i:j].split()
        assert len(header) == 3 and header[1] == b"blob", (p, raw[i:j])
        n = int(header[2])
        out[p] = raw[j + 1:j + 1 + n].decode("utf-8", errors="replace")
        i = j + 1 + n + 1                       # skip the blob's trailing newline
    return out


tests = tree_files(REV)
SRC = blobs(REV, tests)

# A PRODUCTION BUILDER is a callable that manufactures the object production reads.
# Measured by name against the scale/ package's own exported builder vocabulary.
BUILDER = re.compile(
    r"\b(M3_TASKS|TASKS|make_\w*batch|make_batch|build_\w+|draw_\w+|\w+_oracle|oracle"
    r"|units\(\)|make_\w+|\w+_batch|read_journal|journalled\(\)|Journal\(|load_\w+"
    r"|run_arm|build_arm|select_pivots|contrast\(|SHIPPED_CASE|shipped_\w+)\b")

rows = {}
for t in tests:
    src = SRC[t]
    # which scale modules are imported, and under what local alias
    aliases = set()
    for m in re.finditer(r"^\s*from\s+scale(?:\.(\w+))?\s+import\s+([^\n#]+)", src, re.M):
        for piece in m.group(2).replace("(", "").replace(")", "").split(","):
            piece = piece.strip()
            if not piece:
                continue
            name = piece.split(" as ")[-1].strip() if " as " in piece else piece
            aliases.add(name)
    for m in re.finditer(r"^\s*import\s+scale\.(\w+)(?:\s+as\s+(\w+))?", src, re.M):
        aliases.add(m.group(2) or m.group(1))

    # front-door calls: a BUILDER token reached through one of those aliases,
    # or a bare imported builder name that came from scale
    front = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for a in aliases:
            if re.search(r"\b%s\s*[.\[(]" % re.escape(a), line) and BUILDER.search(line):
                front.append((i, a, line.strip()[:90]))
                break
        else:
            if aliases and BUILDER.search(line) and any(re.search(r"\b%s\b" % re.escape(a), line) for a in aliases):
                front.append((i, "direct", line.strip()[:90]))
    rows[t] = {
        "aliases": sorted(aliases),
        "front_door": front[:4],
        "n_front": len(front),
        "n_tests": len(re.findall(r"^\s*def test_", src, re.M)),
    }

callers = sorted(t for t, v in rows.items() if v["n_front"] > 0)
print("rev:", REV, "" if REV == PIN else "(NOT the pin -- this number is not the shipped one)")
print("test .py files:", len(tests))
print("files calling a production builder (front door):", len(callers))
print("files with NO front-door call:", len(tests) - len(callers))

# ONLY THE PIN WRITES. Caught by `git diff` in it.3: an exploratory
# `python scale/p1prime.py HEAD` had already overwritten both shipped files with
# HEAD's 201/47/154, which is the drift defect wearing a different hat -- the
# provenance for a shipped number was silently replaced by a number that is not
# shipped. A run at any other rev now prints and writes nothing.
if REV == PIN:
    json.dump(rows, open(OUT / "p1prime_rows.json", "w"), indent=0)
    (OUT / "p1prime_front_door.txt").write_text(chr(10).join(callers) + chr(10),
                                                encoding="utf-8")
    print("list written:", OUT / "p1prime_front_door.txt")
else:
    print("nothing written: only the pin", PIN, "may overwrite", OUT.name)

#: What AUDIT.md ships, at PIN. The assert below is the whole point of the pin.
SHIPPED = (191, 46, 145)

if "--demo" in sys.argv:
    assert REV == PIN, "run the self-check on the pin, not on %s" % REV
    got = (len(tests), len(callers), len(tests) - len(callers))
    assert got == SHIPPED, "pinned census moved: %s, AUDIT.md ships %s" % (got, SHIPPED)
    assert tree_files(PIN) == tests, "the pinned file list is not stable"
    live = len(tree_files("HEAD"))
    print("demo OK: pin reproduces AUDIT.md %s; HEAD holds %d test files (drift %+d)"
          % (SHIPPED, live, live - SHIPPED[0]))
