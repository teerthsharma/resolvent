"""Count the Lean declarations in lean/CEQ/*.lean and lean/CEQ.lean.

A raw grep miscounts in both directions: it matches comment lines that open with
the word "theorem", and it misses declarations carrying an attribute such as
`@[simp] lemma`. This strips `--` and nested `/- -/` comments first and admits
attribute and modifier prefixes. Run: python scripts/lean_count.py
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFIX = r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable)\s+)*"


def strip_comments(src):
    out, i, depth = [], 0, 0
    while i < len(src):
        if src.startswith("/-", i):
            depth, i = depth + 1, i + 2
        elif depth and src.startswith("-/", i):
            depth, i = depth - 1, i + 2
        elif depth:
            out.append("\n" if src[i] == "\n" else " ")
            i += 1
        elif src.startswith("--", i):
            j = src.find("\n", i)
            i = len(src) if j < 0 else j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def count(src, kind):
    return len(re.findall(PREFIX + kind + r"\s", strip_comments(src), re.M))


def main():
    planted = ("/- theorem in_a_block -/\n-- theorem in_a_line\n"
               "@[simp] lemma a : True := trivial\ntheorem b : True := trivial\n")
    assert (count(planted, "theorem"), count(planted, "lemma")) == (1, 1), "counter miscounts the planted file"
    files = sorted((ROOT / "lean" / "CEQ").glob("*.lean")) + [ROOT / "lean" / "CEQ.lean"]
    t = sum(count(f.read_text(encoding="utf-8"), "theorem") for f in files)
    l = sum(count(f.read_text(encoding="utf-8"), "lemma") for f in files)
    print(f"{len(files)} files: {t} theorems + {l} lemmas = {t + l}")


if __name__ == "__main__":
    main()
