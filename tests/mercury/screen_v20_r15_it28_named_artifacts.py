"""MERCURY it.28 named-artifact screen. Written to a file so the denominator is reproducible.

    python tests/mercury/screen_v20_r15_it28_named_artifacts.py V20_R15_THEORY_TABLE.md

Prints the table digest, then per scope: unit name, pointer-shaped token count, and the
ordered list of ARTIFACT-CANDIDATE tokens with their line numbers. The screen does NOT
rule CITED/UNCITED -- that is the hand read. It fixes the population only.
"""
import hashlib, re, sys
sys.stdout.reconfigure(encoding="utf-8")  # table is UTF-8; cp1252 stdout otherwise dies on U+2212

POINTER = re.compile(r"[\w./-]+\.(py|lean|md|jsonl|toml|json|txt|lean)\s*:\s*[\d*]")
BARE_LINE = re.compile(r"^:\d+$")
TICK = re.compile(r"`([^`]+)`")

def units(lines):
    """Yield (scope, unit, first_line_no, [lines]) for scopes 2, 3, 5."""
    scope = unit = None; buf = []; start = 0
    for i, ln in enumerate(lines, 1):
        h2 = re.match(r"^## §(\d)", ln)
        h3 = re.match(r"^### (.+)", ln)
        if h2:
            if unit: yield (scope, unit, start, buf)
            scope, unit, buf, start = h2.group(1), None, [], i
            if scope in "235": unit, buf, start = f"§{scope} PREAMBLE", [], i
            continue
        if h3 and scope in "235":
            if unit: yield (scope, unit, start, buf)
            unit, buf, start = h3.group(1).strip(), [], i
            continue
        if unit: buf.append((i, ln))
    if unit: yield (scope, unit, start, buf)

def main(path):
    raw = open(path, "rb").read()
    print(f"file    {path}")
    print(f"digest  {hashlib.sha256(raw).hexdigest()[:16]}   bytes {len(raw)}")
    lines = raw.decode("utf-8").splitlines()
    for scope, unit, start, buf in units(lines):
        if scope not in "235": continue
        ptr = 0; cands = []
        for i, ln in buf:
            for tok in TICK.findall(ln):
                t = tok.strip()
                if POINTER.search(t) or BARE_LINE.match(t): ptr += 1; continue
                cands.append((i, t))
        print(f"\n=== §{scope} :: {unit}   (lines {start}-{buf[-1][0] if buf else start})")
        print(f"    pointer-shaped tokens: {ptr}   artifact-candidate tokens: {len(cands)}")
        for i, t in cands: print(f"    {i}: {t}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V20_R15_THEORY_TABLE.md")
