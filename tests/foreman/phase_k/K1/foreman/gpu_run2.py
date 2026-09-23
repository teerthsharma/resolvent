# Foreman K1: gpu_run.py plus the Dispatcher's fairness rule (05:3x): yield to Cameron's far-band arms and Chase's
# checkpoint reader, and never re-acquire within 3 minutes of my own release.  python gpu_run2.py <tag> <rdepth args>
import datetime, json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
RDEPTH = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/cameron/rdepth.py"
LOCK = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/gpu.lock.d"
LAST = HERE / "runs" / "last_release.txt"


def higher_priority_waiting():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True).stdout.lower()
    return "jobs_far" in out or "chase" in out


def main():
    tag, args = sys.argv[1], sys.argv[2:]
    while True:
        cool = LAST.exists() and time.time() - float(LAST.read_text()) < 180
        if not cool and not higher_priority_waiting():
            try:
                os.mkdir(LOCK)
                break
            except FileExistsError:
                pass
        time.sleep(60)
    with open(LOCK + "/owner.txt", "w") as f:
        f.write(f"seat=Foreman pid={os.getpid()} start={datetime.datetime.now().isoformat(timespec='seconds')} tag={tag}\n")
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with open(HERE / "runs" / f"{tag}.stdout", "w") as f:
            rc = subprocess.run([sys.executable, RDEPTH, *args], stdout=f, stderr=subprocess.STDOUT, env=env,
                                timeout=45 * 60).returncode
        print(tag, "exit", rc, flush=True)
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)
        LAST.write_text(str(time.time()))


if __name__ == "__main__":
    main()
