"""Run one command in a fresh subprocess under the house GPU lock: python gpurun.py <tag> <cmd...>"""
import subprocess, sys
from gpulock import gpu_lock

with gpu_lock(sys.argv[1]):
    rc = subprocess.run(sys.argv[2:]).returncode
sys.exit(rc)
