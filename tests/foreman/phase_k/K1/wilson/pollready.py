"""python pollready.py TIMEOUT_S -> checks SP/phase_k/K1/chase/READY every 120 s; prints it and exits 0 when present, 2 at timeout."""
import os, sys, time, datetime
P = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/chase/READY"
t0 = time.time()
while True:
    print(datetime.datetime.now().isoformat(timespec="seconds"), "READY" if os.path.exists(P) else "absent", flush=True)
    if os.path.exists(P):
        print(open(P, errors="ignore").read()); sys.exit(0)
    if time.time() - t0 + 120 > float(sys.argv[1]):
        sys.exit(2)
    time.sleep(120)
