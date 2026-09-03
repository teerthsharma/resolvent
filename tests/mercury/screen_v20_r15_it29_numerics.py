"""MERCURY it.29 screen. Numeric tokens in a line range of a markdown file,
split float-shaped / integer-shaped, with `path:line` pointers removed first.

    python tests/mercury/screen_v20_r15_it29_numerics.py <file> <first> <last>

Prints every surviving token with its line number, then the two counts.
It fixes the POPULATION only. CITED/UNCITED is a hand read, as in it.28.
"""
import hashlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")  # table is UTF-8; Windows stdout is cp1252

# a pointer: any dotted-extension path followed by one or more :line or :*
POINTER = re.compile(r"[\w./\-]+\.(?:py|md|lean|jsonl|toml|txt|json)(?::(?:\d+(?:-\d+)?|\*))*")
# a numeric token: optional ascii/unicode minus, digits, optional fraction, optional exponent
NUMBER = re.compile(r"[-\u2212+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def screen(path, first, last):
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    lines = raw.decode("utf-8").splitlines()
    floats, ints = [], []
    for n in range(first, last + 1):
        text = POINTER.sub(" ", lines[n - 1])
        for m in NUMBER.finditer(text):
            tok = m.group(0)
            (floats if ("." in tok or "e" in tok or "E" in tok) else ints).append((n, tok))
    return digest, floats, ints


if __name__ == "__main__":
    f, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    digest, floats, ints = screen(f, a, b)
    print(f"{f} lines {a}-{b}  digest {digest}")
    print(f"FLOAT-SHAPED  {len(floats)}")
    for n, t in floats:
        print(f"  :{n}  {t}")
    print(f"INTEGER-SHAPED  {len(ints)}")
    for n, t in ints:
        print(f"  :{n}  {t}")
