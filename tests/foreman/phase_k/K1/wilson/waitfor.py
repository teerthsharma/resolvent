"""python waitfor.py FILE PATTERN TIMEOUT_S -> blocks until PATTERN appears in FILE (or timeout); prints the last 15 lines; exit 0 if found, 2 if not."""
import os, sys, time
f, pat, tmo = sys.argv[1], sys.argv[2], float(sys.argv[3])
t0 = time.time()
txt = lambda: open(f, errors="ignore").read() if os.path.exists(f) else ""
while pat not in txt() and time.time() - t0 < tmo:
    time.sleep(15)
print("\n".join(txt().splitlines()[-15:]))
sys.exit(0 if pat in txt() else 2)
