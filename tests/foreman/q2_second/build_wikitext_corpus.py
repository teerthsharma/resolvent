"""Build a scratchpad-local WikiText-103-raw text corpus, document-separated
by blank lines (DocByteBatches splits on "\n\n"), one WikiText ARTICLE per
document (paragraph breaks WITHIN an article are collapsed to single
newlines so DocByteBatches's "\n\n" split lands on article boundaries only --
the same granularity TinyStories gets from one-story-per-blank-line-run).
Streamed from HF just far enough to comfortably exceed train_with_eval's
default max_bytes=64MiB after truncation. Reports bytes_total (wc -c
equivalent), digest, licence.
"""
import hashlib
import io
import re
import time

from datasets import load_dataset

TARGET_RAW_CHARS = 100 * 1024 * 1024  # gather generously more than the 64MiB max_bytes cap
OUT = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\wikitext103_corpus.txt"

TITLE_RE = re.compile(r"^\s*=\s+[^=].*[^=]\s+=\s*$")  # top-level " = Title = ", not " = = Section = = "

def main():
    t0 = time.time()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    docs = []
    cur = []
    total_chars = 0
    n_lines = 0
    for row in ds:
        line = row["text"].rstrip("\n")
        n_lines += 1
        if TITLE_RE.match(line):
            if cur:
                doc = "\n".join(cur).strip()
                if doc:
                    docs.append(doc)
                    total_chars += len(doc)
            cur = [line]
        elif line.strip():  # drop blank lines WITHIN an article -- doc boundary is "\n\n" only
            cur.append(line)
        if total_chars >= TARGET_RAW_CHARS:
            break
    if cur:
        doc = "\n".join(cur).strip()
        if doc:
            docs.append(doc)
            total_chars += len(doc)

    text = "\n\n".join(docs)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)

    raw_bytes = text.encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    print("lines_scanned", n_lines)
    print("n_documents", len(docs))
    print("chars_total", len(text))
    print("bytes_total", len(raw_bytes))
    print("sha256", digest)
    print("elapsed_s", time.time() - t0)
    print("out_path", OUT)

if __name__ == "__main__":
    main()
