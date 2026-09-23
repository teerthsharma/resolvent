# Foreman K1 bar binding every lower-bound quote in FOREMAN_REPORT.md to a saved artifact. Written 2026-09-23 03:02 IST,
# BEFORE quotes.json exists. Never edited after its RED. Reads quotes.json: [{"file": "sources/<id>.txt", "quote": ...}].
#
# foreman.k1.sources_verbatim : every quote occurs verbatim in its saved pdftotext file after collapsing whitespace
#     (runs of spaces/newlines -> one space) and joining hyphenated line breaks; every cited .txt has its .pdf beside it
#     and that .pdf's sha256 is the one in sources/MANIFEST.txt. At least 4 quotes, from at least 2 papers.
import hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
Q = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "quotes.json")).read_text())
man = (HERE / "sources" / "MANIFEST.txt").read_text()
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


def norm(s):
    s = re.sub(r"-\s*\n\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip()


papers = set()
for i, q in enumerate(Q):
    txt = HERE / q["file"]
    pdf = txt.with_suffix(".pdf")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else "missing"
    ok = norm(q["quote"]) in norm(txt.read_text(encoding="utf-8", errors="replace")) and sha in man
    papers.add(q["file"])
    check(f"foreman.k1.sources_verbatim quote {i} ({q['file']})", ok, norm(q["quote"])[:90])
check("foreman.k1.sources_verbatim >= 4 quotes from >= 2 papers", len(Q) >= 4 and len(papers) >= 2, (len(Q), len(papers)))
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
