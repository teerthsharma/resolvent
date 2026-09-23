# Inspector pass 5: log every change of gpu.lock.d (exists / owner.txt text) with a timestamp, 1 s poll, read-only.
import datetime, os, sys, time
L = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/gpu.lock.d"
end = time.time() + float(sys.argv[1])
prev = None
while time.time() < end:
    try:
        s = open(L + "/owner.txt").read().strip() if os.path.isdir(L) else "FREE"
    except OSError:
        s = "DIR-NO-OWNER" if os.path.isdir(L) else "FREE"
    if s != prev:
        print(datetime.datetime.now().isoformat(timespec="milliseconds"), s, flush=True)
        prev = s
    time.sleep(1)
