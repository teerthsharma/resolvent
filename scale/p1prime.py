"""P1' measurement. Does the plant enter at the instrument's front door?

P1 (replaced) keyed on WHO built the input. Every planted positive is constructed;
that is what planting means, so P1 fired on cures as readily as diseases.

P1' keys on WHERE the plant ENTERS. A control is vacuous-by-scope when its planted
input enters BELOW the front door, so stages the real input traverses go unexercised.
Measured here, per file, with the stages named.
"""
import ast, json, pathlib, re, subprocess

# Provenance for a shipped number lands in the repo, never in temp.
# Rescued from the session scratchpad (R10 P0 it.2); paths are now repo-relative.
R = pathlib.Path(__file__).resolve().parent.parent
OUT = R / "results"

tests = [p for p in subprocess.run(["git", "ls-files", "tests/*.py"], cwd=R,
                                   capture_output=True, text=True).stdout.splitlines()
         if p.strip() and not p.startswith(".claude/worktrees")]

# A PRODUCTION BUILDER is a callable that manufactures the object production reads.
# Measured by name against the scale/ package's own exported builder vocabulary.
BUILDER = re.compile(
    r"\b(M3_TASKS|TASKS|make_\w*batch|make_batch|build_\w+|draw_\w+|\w+_oracle|oracle"
    r"|units\(\)|make_\w+|\w+_batch|read_journal|journalled\(\)|Journal\(|load_\w+"
    r"|run_arm|build_arm|select_pivots|contrast\(|SHIPPED_CASE|shipped_\w+)\b")

rows = {}
for t in tests:
    src = (R / t).read_text(encoding="utf-8", errors="replace")
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

json.dump(rows, open(OUT / "p1prime_rows.json", "w"), indent=0)
callers = sorted(t for t, v in rows.items() if v["n_front"] > 0)
(OUT / "p1prime_front_door.txt").write_text(chr(10).join(callers) + chr(10), encoding="utf-8")
print("test .py files:", len(tests))
print("files calling a production builder (front door):", len(callers))
print("files with NO front-door call:", len(tests) - len(callers))
print("list written:", OUT / "p1prime_front_door.txt")
