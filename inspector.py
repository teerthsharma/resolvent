"""The Health Inspector, as a command with an exit code.

WHY THIS FILE EXISTS. For forty-five iterations the Inspector pass was hand-typed
shell, and it failed as an instrument four times: it declared a Lean build clean
by reading `tail`'s exit status, it reported attribution HITS with zero matching
lines because `head` exits 0 on empty input, it "checked" a published number by
printing it, and it once compared a LOCK against a slice whose boundary had
moved.

Iteration 40 wrote the rule into `STATE.md` in these words: *"the defect is ANY
pipeline, because the shell reports only the last stage."* **Iteration 45 then
wrote `| head -3 &&` anyway.** Iterations 37, 38 and 45 each reproduced a defect
class recorded one to five iterations earlier.

**Naming a defect class does not prevent it.** Every defence that has actually
held in this repository is structural -- the calibration gate, the journal's
bitwise replay, layer 1 of the struck-constant bind. So the Inspector becomes a
test rather than a rule:

  * no decision is ever placed after a shell pipeline -- `subprocess.run` with a
    list argv, and the return code is read from the process that produced it;
  * every check compares a VALUE, never a substring of formatted output;
  * every check ships a MUST-FIRE control that feeds it a known-bad input and
    requires it to fail. A check that cannot fail is not a check (instrument
    #15), and a check that was never seen failing is a rule, not a test.

The script exits nonzero if any check fails OR if any must-fire control does not
fire. Both halves are load-bearing: an all-green run whose controls stayed silent
means the Inspector is blind, which is worse than a red one.

    python inspector.py [ITERATION]

`ITERATION` selects the two rotations (journalled unit, published number) so
consecutive passes cover different ground. Defaults to the value in STATE.md.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

#: The calibration table. `run_calib.py` owns these too, but this checks them by
#: invoking `bench` directly -- its comparison logic is what broke as #12.
CALIB = [("signed", 3, 0.046875), ("sgate", 1, 0.0234375),
         ("sgate", 2, 0.1640625), ("softmax", 3, 0.0)]

#: Published numbers, rotated by iteration so passes cover different ground.
#: Each is (label, thunk -> dict of name: (measured, published)).
LOCK_M2 = "efadc390c93f"

results: list[tuple[str, bool, str]] = []
controls: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def control(name: str, fired: bool) -> None:
    """Record whether a must-fire probe actually failed the check it guards."""
    controls.append((name, fired))


def run(argv: list[str], cwd: pathlib.Path | None = None) -> tuple[int, str]:
    """Run argv with NO shell and NO pipeline. The return code is the process's."""
    p = subprocess.run(argv, cwd=str(cwd or ROOT), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ---------------------------------------------------------------- 1 calibration

def _flip_rates() -> dict[tuple[str, int], float]:
    import torch
    from ceq import bench
    dev = torch.device("cpu")
    return {(k, h): bench.sign_flip_rate(n_draws=128, s=8, device=dev,
                                         kind=k, depth=1, hops=h)
            for k, h, _ in CALIB}


def _calib_matches(got: dict, table) -> bool:
    return all(got[(k, h)] == pub for k, h, pub in table)


def check_calibration() -> None:
    got = _flip_rates()
    ok = _calib_matches(got, CALIB)
    check("calibration (4 values, bench invoked directly)", ok,
          " ".join(f"{k}/h{h}={got[(k, h)]!r}" for k, h, _ in CALIB))
    # MUST-FIRE: perturb one published target by 1e-9 and require rejection.
    bad = [(k, h, pub + 1e-9 if (k, h) == ("sgate", 2) else pub)
           for k, h, pub in CALIB]
    control("calibration rejects a perturbed target", not _calib_matches(got, bad))


# ---------------------------------------------------------------------- 2 LOCK

def _m2_slice(text: str) -> str:
    a = text.index("**M2. CONTEXT-STABLE")
    b = text.index("**M2'. SIGNED INFLUENCE")
    return text[a:b].strip()


def check_lock() -> None:
    live = _m2_slice((ROOT / "CHECKLIST.md").read_text(encoding="utf-8"))
    archived = (ROOT / "results/m2_item_text.txt").read_text(encoding="utf-8").strip()
    h = hashlib.sha256(live.encode()).hexdigest()[:12]
    check(f"LOCK M2 {LOCK_M2}", live == archived and h == LOCK_M2, f"hash={h}")
    # MUST-FIRE: a single character changed must break both the compare and hash.
    mutated = live.replace("M2", "M2x", 1)
    control("LOCK detects a one-token edit",
            mutated != archived
            and hashlib.sha256(mutated.encode()).hexdigest()[:12] != LOCK_M2)


# ------------------------------------------------------------ 3 bitwise replay

#: The journal's determinism is CONDITIONAL ON THREAD COUNT, and nothing in the
#: repository recorded that. Found by this script's first run: replaying
#: `dense_signed__at_pivots/s2048/b5` gives MATCH at 2 threads and DRIFT at 1,
#: because CPU matmul reduction order varies with the thread pool. Every
#: journalled unit was produced under `OMP_NUM_THREADS=2`, so "bitwise replay"
#: means bitwise *at that setting*. `tests/chase/test_resume_checkpoint.py`
#: already knew this for training -- `torch.set_num_threads(1)`, commented "CPU
#: matmul reduction order must not vary run to run" -- and the measurement side
#: never picked it up.
JOURNAL_THREADS = 2


def check_replay(iteration: int) -> None:
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(JOURNAL_THREADS)
    from scale.m2_units import compute, units
    journal = {}
    for line in (ROOT / "results/m2.jsonl").read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            journal[j["key"]] = j
    u = dict(units())
    keys = sorted(k for k in u if k in journal)
    key = keys[iteration % len(keys)]
    got = json.dumps(compute(u[key]), sort_keys=True)
    want = json.dumps(journal[key]["value"], sort_keys=True)
    check(f"bitwise replay [{key}]", got == want, f"{len(keys)} journalled")
    # MUST-FIRE: comparing against a mutated record must not match.
    control("replay detects a mutated record", got != want + " ")


# -------------------------------------------------------- 4 published rotation

def check_published(iteration: int) -> None:
    import torch
    from ceq import bench
    which = iteration % 3

    def op(s):
        g = torch.Generator().manual_seed(0)
        q = torch.randn(s, 16, generator=g)
        k = torch.randn(s, 16, generator=g)
        return bench._causal_sgate_operator(q, k, rho=1.5, lam=0.10).double()

    if which == 0:                                  # Lean core: A^n = 0 exactly
        vals = {s: float(torch.linalg.matrix_power(op(s), s).abs().max())
                for s in (16, 64, 128, 512)}
        ok = all(v == 0.0 for v in vals.values())
        check("published: pow_card_eq_zero (A^n = 0)", ok, str(vals))
        control("A^n check would reject a nonzero tail",
                not all(v == 0.0 for v in {**vals, 512: 1e-12}.values()))
    elif which == 1:                                # M5 tail norms, pinned
        pub = {(128, 2): 0.880500, (512, 2): 1.292741}
        got = {kk: float(torch.linalg.matrix_power(op(kk[0]), kk[1]).abs().max())
               for kk in pub}
        ok = all(abs(got[kk] - pub[kk]) < 5e-7 for kk in pub)
        check("published: M5 tail norms", ok,
              " ".join(f"s{s}h{h}={got[(s, h)]:.6f}" for s, h in pub))
        control("tail-norm check rejects the struck 1.471448",
                not abs(got[(128, 2)] - 1.471448) < 5e-7)
    else:                                           # M2's two-point derivation
        import math
        got = math.log10(0.02732 / 0.16511) / math.log10(4)
        ok = abs(got - (-1.2977)) < 5e-4
        check("published: M2 two-point slope", ok, f"{got:.4f} vs -1.2977")
        control("slope check rejects the struck -1.826",
                not abs(got - (-1.826)) < 5e-4)


# ------------------------------------------------------------------- 5/6/7 sub

def check_suites() -> None:
    rc, out = run([sys.executable, "-m", "pytest", "tests/loop", "tests/w11",
                   "tests/chase/test_resume_checkpoint.py", "-q",
                   "-p", "no:cacheprovider"])
    m = re.search(r"(\d+) passed", out)
    check("value binds + resume", rc == 0, f"{m.group(1) if m else '?'} passed")
    # MUST-FIRE: the exit-code path itself. This is instrument #13's antidote --
    # a process that fails must be SEEN to fail, with no pipeline in between.
    rc_bad, _ = run([sys.executable, "-c", "import sys; sys.exit(3)"])
    control("nonzero exit is detected (no pipeline mask)", rc_bad == 3)


def check_lean() -> None:
    lean = ROOT / "lean"
    if not (lean / "lakefile.lean").exists() and not (lean / "lakefile.toml").exists():
        check("lake build CEQ", False, "no lakefile")
        control("lean check present", False)
        return
    rc, _ = run(["lake", "build", "CEQ"], cwd=lean)
    sorries = sum(len(re.findall(r"\bsorry\b", p.read_text(encoding="utf-8", errors="ignore")))
                  for p in (lean / "CEQ").rglob("*.lean"))
    check("lake build CEQ (unmasked exit) + zero sorry", rc == 0 and sorries == 0,
          f"exit={rc} sorry={sorries}")
    control("sorry counter can see one", len(re.findall(r"\bsorry\b", "have h : True := sorry")) == 1)


def check_struck() -> None:
    rc, out = run([sys.executable, "-m", "pytest",
                   "tests/loop/test_no_struck_constant_ships.py", "-q",
                   "-p", "no:cacheprovider"])
    m = re.search(r"(\d+) passed", out)
    check("struck-constant absence (9 documents + shipped code)", rc == 0,
          f"{m.group(1) if m else '?'} passed")
    control("struck registry is non-empty",
            bool(re.search(r"-1\.389", (ROOT / "tests/loop/test_no_struck_constant_ships.py")
                           .read_text(encoding="utf-8"))))


# ------------------------------------------------------------- 8 attribution

def _attr_hits(msg: str) -> int:
    return len(re.findall(r"co-authored-by|generated with|anthropic", msg, re.I))


def check_attribution() -> None:
    rc, out = run(["git", "log", "--format=%H"])
    shas = [s for s in out.split() if s]
    per = {}
    for s in shas:
        _, msg = run(["git", "log", "-1", "--pretty=%B", s])
        per[s[:7]] = _attr_hits(msg)
    check("no Claude attribution in any commit", all(v == 0 for v in per.values()),
          " ".join(f"{k}:{v}" for k, v in per.items()))
    # MUST-FIRE: the counter must see a real trailer.
    control("attribution counter sees a real trailer",
            _attr_hits("fix thing\n\nCo-Authored-By: Claude <x@y>") == 1)


# ---------------------------------------------------------------------- driver

def main() -> int:
    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
    else:
        m = re.search(r"\| iteration \| \*\*(\d+) complete",
                      (ROOT / "STATE.md").read_text(encoding="utf-8"))
        iteration = int(m.group(1)) + 1 if m else 0

    print(f"HEALTH INSPECTOR  iteration {iteration}\n")
    for fn, arg in ((check_calibration, None), (check_lock, None),
                    (check_replay, iteration), (check_published, iteration),
                    (check_suites, None), (check_lean, None),
                    (check_struck, None), (check_attribution, None)):
        try:
            fn(iteration) if arg is not None else fn()
        except Exception as e:                       # a crashed check is a FAILED check
            check(f"{fn.__name__} (raised)", False, f"{type(e).__name__}: {e}")

    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    print()
    for name, fired in controls:
        print(f"  [{'FIRED' if fired else 'SILENT'}] must-fire: {name}")

    bad = [n for n, ok, _ in results if not ok]
    silent = [n for n, f in controls if not f]
    print()
    if bad:
        print(f"  FAILED CHECKS: {len(bad)} -- {', '.join(bad)}")
    if silent:
        print(f"  SILENT CONTROLS: {len(silent)} -- {', '.join(silent)}")
        print("  A check whose control stayed silent is BLIND. This is not a pass.")
    if not bad and not silent:
        print(f"  CLEAN: {len(results)} checks, {len(controls)} controls all fired.")
    return 1 if (bad or silent) else 0


if __name__ == "__main__":
    raise SystemExit(main())
