"""Run test files one after another, each in a fresh python process: python chain.py <test.py> <log> [<test.py> <log> ...]"""
import subprocess, sys

rc = 0
for t, log in zip(sys.argv[1::2], sys.argv[2::2]):
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, t], stdout=f, stderr=subprocess.STDOUT).returncode
    with open(log, "a") as f:
        f.write(f"exit {r}\n")
    rc |= r
sys.exit(rc)
