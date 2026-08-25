"""Bucketed, resumable measurement units. See ADR-001.

WHY. M2 was launched as one long process and died silently after 588 s with one
row of six complete and no artifact. Three failures combined:
  1. the harness caps a Bash call at 600 000 ms, and the run was killed by that
     cap -- any measurement over 10 minutes CANNOT complete in one call;
  2. `python ... | tee f` reports TEE's exit status, so a killed process was
     logged as **exit code 0** -- a truncated run presented as a complete one;
  3. all state lived in memory, so 588 s of correct work, including a full
     16 384-draw cell, was destroyed instead of banked.

THE UNIT. The smallest independently meaningful measurement, keyed by its FULL
parameter tuple. Two units with the same key must produce the same number; that
is asserted on replay rather than assumed, which turns resume into a free
determinism audit.

THE BUCKET. However many whole units fit inside a wall-clock budget set
comfortably below the platform cap. One bucket per loop iteration.

NEVER PIPE THROUGH `tee`. Write the journal from inside the process and let the
real exit code out.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class Journal:
    """Append-only record of completed units. One JSON object per line.

    Append-only because a recorded result is evidence: rewriting the file to
    "clean it up" is how a number quietly changes between the run that produced
    it and the document that cites it.
    """

    def __init__(self, name: str):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{name}.jsonl"
        self.lock = RESULTS / f"{name}.lock"

    # -- duplicate-run prevention -------------------------------------------
    def acquire(self) -> None:
        """Refuse to start if another live process holds the lock.

        A `nohup` orphan silently doubled an earlier M2 run: two processes, same
        output file, every cell computed twice, and the only symptom was that
        progress looked slow. A lock makes that impossible rather than unlikely.
        """
        if self.lock.exists():
            try:
                pid = int(self.lock.read_text().strip())
            except Exception:
                pid = -1
            if pid > 0 and _alive(pid):
                raise SystemExit(
                    f"REFUSING TO START: {self.lock} is held by live PID {pid}. "
                    f"Another run of this measurement is in flight. Kill it or "
                    f"wait; do not start a second copy."
                )
            print(f"  (stale lock from dead PID {pid}, reclaiming)")
        self.lock.write_text(str(os.getpid()))

    def release(self) -> None:
        try:
            self.lock.unlink()
        except FileNotFoundError:
            pass

    # -- unit records --------------------------------------------------------
    def done(self) -> dict:
        if not self.path.exists():
            return {}
        out = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["key"]] = rec
        return out

    def append(self, key: str, value: dict, meta: dict | None = None) -> None:
        rec = {"key": key, "value": value, "meta": meta or {}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _alive(pid: int) -> bool:
    """Is this PID a live process?

    ON WINDOWS `tasklist` IS THE PRIMARY PATH, NOT A FALLBACK. `os.kill(pid, 0)`
    is the POSIX idiom and it does not port: on win32 it raised

        SystemError: <built-in function kill> returned a result with an
                     exception set

    for a recycled PID, which is neither OSError nor PermissionError, so it
    escaped the original handler and crashed `acquire()` outright -- the lock
    meant to prevent a duplicate run instead prevented ANY run. Caught here as
    the bare `Exception` it is, because the question "is this process alive" has
    exactly two useful answers and an exception is not one of them.
    """
    if sys.platform == "win32":
        return _alive_win(pid)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return _alive_win(pid)


def _alive_win(pid: int) -> bool:
    import subprocess
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                           capture_output=True, text=True, timeout=20)
        return str(pid) in r.stdout
    except Exception:
        return False


def run_bucket(name: str, units: list, compute, *, budget_s: float = 420.0,
               verify: int = 1) -> dict:
    """Execute whole units until `budget_s` is spent, then stop cleanly.

    `units`   : list of (key, params) -- params is a JSON-serialisable dict
    `compute` : params -> JSON-serialisable dict
    `budget_s`: wall-clock budget. Default 420 s, comfortably under the 600 s cap
                so an in-flight unit cannot be truncated by the platform.
    `verify`  : recompute this many ALREADY-DONE units and require a bitwise
                match. Resume then doubles as a determinism audit instead of a
                leap of faith.

    Returns an accounting dict. It reports done/total and NEVER decides anything;
    a verdict on a partial set is the failure this whole file exists to prevent.
    """
    j = Journal(name)
    j.acquire()
    try:
        done = j.done()
        todo = [(k, p) for k, p in units if k not in done]
        print(f"[{name}] {len(done)}/{len(units)} units already journalled, "
              f"{len(todo)} remaining, budget {budget_s:.0f}s")

        # --- determinism audit on resume ---------------------------------
        checked = []
        for key, params in units:
            if len(checked) >= verify or key not in done:
                continue
            got = compute(params)
            want = done[key]["value"]
            same = json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
            checked.append((key, same))
            print(f"  replay {key}: {'MATCH' if same else 'DRIFT'}")
            if not same:
                raise SystemExit(
                    f"NONDETERMINISM: unit {key} recomputed to {got}, journal "
                    f"holds {want}. Every number from this instrument is void "
                    f"until the cause is found."
                )

        t0, ran = time.time(), 0
        for key, params in todo:
            spent = time.time() - t0
            if spent >= budget_s:
                print(f"  budget spent ({spent:.0f}s); stopping cleanly with "
                      f"{len(todo) - ran} units left")
                break
            u0 = time.time()
            value = compute(params)
            j.append(key, value, {"seconds": round(time.time() - u0, 2)})
            ran += 1
            print(f"  [{len(done)+ran}/{len(units)}] {key} -> {value} "
                  f"({time.time()-u0:.0f}s)", flush=True)

        final = j.done()
        acc = dict(name=name, total=len(units), done=len(final),
                   remaining=len(units) - len(final), ran_this_bucket=ran,
                   verified=checked)
        print(f"[{name}] bucket end: {acc['done']}/{acc['total']} done, "
              f"{acc['remaining']} remaining")
        return acc
    finally:
        j.release()


def require_complete(name: str, units: list) -> dict:
    """Return unit values, or REFUSE if the set is partial.

    Aggregating a partial set and printing a verdict is precisely how a
    truncated run becomes a reported result.
    """
    done = Journal(name).done()
    missing = [k for k, _ in units if k not in done]
    if missing:
        raise SystemExit(
            f"REFUSING TO AGGREGATE: {len(missing)} of {len(units)} units are "
            f"missing, e.g. {missing[:4]}. No verdict may be computed on a "
            f"partial set."
        )
    return {k: done[k]["value"] for k, _ in units}
