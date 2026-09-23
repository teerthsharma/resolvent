# Foreman K1: run Cameron's rdepth.py (read-only) as ONE fresh subprocess under the house GPU lock; outputs in my lane.
#   python gpu_run.py <tag> <rdepth args...>
import os, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from twin import acquire, LOCK  # noqa: E402  (atomic mkdir lock; stale-lock rule; owner.txt)
import shutil  # noqa: E402

RDEPTH = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/cameron/rdepth.py"

tag, args = sys.argv[1], sys.argv[2:]
acquire(tag)
try:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with open(HERE / "runs" / f"{tag}.stdout", "w") as f:
        rc = subprocess.run([sys.executable, RDEPTH, *args], stdout=f, stderr=subprocess.STDOUT, env=env,
                            timeout=45 * 60).returncode
    print(tag, "exit", rc)
finally:
    shutil.rmtree(LOCK, ignore_errors=True)
