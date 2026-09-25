"""Build docs/assets/viz/lean_atlas.json: every theorem/lemma (and the defs/structures they use)
in lean/CEQ.lean and lean/CEQ/*.lean, with statement, file:line, and the project declarations
each body references. Run from the repo root: python scripts/lean_atlas.py"""
import json, re, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = [ROOT / "lean/CEQ.lean"] + sorted((ROOT / "lean/CEQ").glob("*.lean"))
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|nonrec|unsafe|partial)\s+)*"
                  r"(theorem|lemma|def|abbrev|structure|instance|inductive|class)\s+([^\s:({\[]+)", re.M)
STOP = re.compile(r"^(?:#\w+|@\[|(?:example|end|namespace|section|open|variable|set_option|theorem|lemma|def|abbrev|structure|instance|inductive|class|private|protected|noncomputable))", re.M)
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)


def strip_comments(src):
    # keep line numbers stable: replace comment text by spaces/newlines
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)


def main():
    decls = []
    for f in FILES:
        raw = f.read_text(encoding="utf-8")
        src = strip_comments(raw)
        ms = list(DECL.finditer(src))
        for i, m in enumerate(ms):
            nl = src.find("\n", m.start())
            nxt = STOP.search(src, nl + 1) if nl >= 0 else None
            end = min(ms[i + 1].start() if i + 1 < len(ms) else len(src), nxt.start() if nxt else len(src))
            body = src[m.start():end]
            head = body.split(":=", 1)[0] if m.group(1) in ("theorem", "lemma", "def", "abbrev", "instance") else body.split(" where", 1)[0]
            decls.append({
                "name": m.group(2), "kind": m.group(1),
                "file": str(f.relative_to(ROOT)).replace("\\", "/"),
                "line": src.count("\n", 0, m.start()) + 1,
                "statement": re.sub(r"\s+", " ", head).strip()[:1200],
                "_body": body,
            })
    names = {d["name"] for d in decls}
    short = {}
    for n in names:
        short.setdefault(n.split(".")[-1], set()).add(n)
    for d in decls:
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*", d.pop("_body")))
        uses = set()
        for t in toks:
            for cand in (t, t.split(".")[-1]):
                for n in short.get(cand, ()):
                    if n != d["name"]:
                        uses.add(n)
        d["uses"] = sorted(uses)
    for d in decls:
        d["usedBy"] = sorted(e["name"] for e in decls if d["name"] in e["uses"])
    thm = [d for d in decls if d["kind"] in ("theorem", "lemma")]
    out = {"files": [str(f.relative_to(ROOT)).replace("\\", "/") for f in FILES],
           "counts": {"theorem": sum(d["kind"] == "theorem" for d in decls),
                      "lemma": sum(d["kind"] == "lemma" for d in decls), "all": len(decls)},
           "decls": decls}
    dst = ROOT / "docs/assets/viz/lean_atlas.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    # self-check: every edge endpoint is a node; theorem+lemma count equals a raw line grep
    raw_count = sum(len(re.findall(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected)\s+)?(?:theorem|lemma)\s", strip_comments(f.read_text(encoding="utf-8")), re.M)) for f in FILES)
    assert all(u in names for d in decls for u in d["uses"]), "dangling edge"
    assert len(thm) == raw_count, (len(thm), raw_count)
    print(f"{len(thm)} theorems+lemmas, {len(decls)} declarations, "
          f"{sum(len(d['uses']) for d in decls)} edges -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
