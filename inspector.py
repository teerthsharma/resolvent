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

  * every check reports one of THREE states. Iteration 11 ran the Inspector
    under four concurrent agents and got `[FAIL] value binds + resume -- ?
    passed`, exit 1; standalone the same command gave `107 passed`, exit 0. The
    `? passed` is the tell -- the regex found no count, so the subprocess was
    KILLED, not failed. **A checker that cannot separate "it failed" from "I
    could not measure it" is worse than one that fails cleanly**, because the
    first response to a red run is to go hunting a defect that is not there.

The script exits nonzero if any check FAILS, if any check is INDETERMINATE, or if
any must-fire control does not fire. All three halves are load-bearing: an
all-green run whose controls stayed silent means the Inspector is blind, which is
worse than a red one, and **an unmeasured check is not a clean one.**

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

#: The three states. INDET is not a softer FAIL -- it is a REFUSAL to report, and
#: it exits nonzero exactly like a FAIL. The only thing it changes is what the
#: reader goes looking for afterwards.
PASS, FAIL, INDET = "PASS", "FAIL", "INDET"

results: list[tuple[str, str, str]] = []
controls: list[tuple[str, bool]] = []


def check(name: str, state, detail: str = "") -> str:
    """`state` is PASS/FAIL/INDET, or a bool for a check that CANNOT be
    indeterminate -- one computed in-process from data already in hand, where
    there is no subprocess to be killed and no measurement to go missing."""
    if state is True:
        state = PASS
    elif state is False:
        state = FAIL
    results.append((name, state, detail))
    return state


def control(name: str, fired: bool) -> None:
    """Record whether a must-fire probe actually failed the check it guards."""
    controls.append((name, fired))


def run(argv: list[str], cwd: pathlib.Path | None = None) -> tuple[int, str]:
    """Run argv with NO shell and NO pipeline. The return code is the process's."""
    p = subprocess.run(argv, cwd=str(cwd or ROOT), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


#: A pytest run states its verdict in TWO places, and the state is decided only
#: when both are present, because each alone has a blind spot that has already
#: fired in this repository.
#:
#: The SUMMARY LINE alone is the trap that matters. Matching only `passed` refiles
#: `3 failed in 1.2s` -- a TOTAL failure, with no `passed` anywhere in the output
#: -- as INDETERMINATE: the new state absorbing exactly the case it must never
#: absorb. So the regex accepts `passed|failed|error`; its only job is to answer
#: "did pytest reach the end and report", never "what did it report".
#:
#: The EXIT CODE alone is the other blind spot. On Windows a killed child does not
#: come back as `-SIGKILL`; it comes back as a large positive code (0xC000013A =
#: 3221225786 for a console interrupt), indistinguishable from an arbitrary
#: failure unless checked against pytest's own documented set. pytest defines
#: 0 (all passed) and 1 (tests failed); 2/3/4/5 are interrupted / internal error /
#: usage error / nothing collected, and every one of those is a run that did not
#: measure the thing.
SUMMARY = re.compile(r"\d+ (?:passed|failed|error)")


def pytest_state(rc: int, out: str) -> tuple[str, str]:
    """(state, detail) for a pytest subprocess. PASS/FAIL only on rc 0/1 WITH a
    summary line; everything else is INDETERMINATE."""
    m = SUMMARY.search(out)
    if m is None:
        return INDET, f"exit={rc}, NO pytest summary line -- killed, or never started"
    if rc == 0:
        return PASS, m.group(0)
    if rc == 1:
        return FAIL, m.group(0)
    return INDET, f"exit={rc} outside pytest's 0/1, summary was {m.group(0)!r}"


def verdict(states: list[tuple[str, str, str]],
            fired: list[tuple[str, bool]]) -> int:
    """The exit code, as a FUNCTION of the two lists. Extracted out of `main` so
    the must-fire controls can hand it known-bad input: a decision rule that only
    ever runs on real data is a rule, and this file exists because rules do not
    hold here."""
    bad = [n for n, s, _ in states if s == FAIL]
    unk = [n for n, s, _ in states if s == INDET]
    silent = [n for n, f in fired if not f]
    return 1 if (bad or unk or silent) else 0


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
    check("value binds + resume", *pytest_state(rc, out))
    # MUST-FIRE: the exit-code path itself. This is instrument #13's antidote --
    # a process that fails must be SEEN to fail, with no pipeline in between.
    rc_bad, _ = run([sys.executable, "-c", "import sys; sys.exit(3)"])
    control("nonzero exit is detected (no pipeline mask)", rc_bad == 3)


def check_lean() -> None:
    lean = ROOT / "lean"
    # Fired FIRST and in EVERY branch. It is a pure-function control over a
    # literal, so it does not depend on the toolchain being present -- the old
    # code let it go SILENT when the lakefile was missing, which forced a nonzero
    # exit for the right reason by the wrong mechanism: a blind-Inspector alarm
    # standing in for an unmeasured check.
    control("sorry counter can see one",
            len(re.findall(r"\bsorry\b", "have h : True := sorry")) == 1)
    if not (lean / "lakefile.lean").exists() and not (lean / "lakefile.toml").exists():
        check("lake build CEQ (unmasked exit) + zero sorry", INDET,
              "no lakefile -- NOT MEASURED")
        return
    try:
        rc, _ = run(["lake", "build", "CEQ"], cwd=lean)
    except OSError as e:                 # lake absent from PATH: unmeasured
        check("lake build CEQ (unmasked exit) + zero sorry", INDET,
              f"lake not runnable: {type(e).__name__}: {e}")
        return
    sorries = sum(len(re.findall(r"\bsorry\b", p.read_text(encoding="utf-8", errors="ignore")))
                  for p in (lean / "CEQ").rglob("*.lean"))
    check("lake build CEQ (unmasked exit) + zero sorry", rc == 0 and sorries == 0,
          f"exit={rc} sorry={sorries}")


def check_struck() -> None:
    rc, out = run([sys.executable, "-m", "pytest",
                   "tests/loop/test_no_struck_constant_ships.py", "-q",
                   "-p", "no:cacheprovider"])
    check("struck-constant absence (9 documents + shipped code)",
          *pytest_state(rc, out))
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


# ------------------------------------------------- 9 the tri-state logic itself

def check_tri_state() -> None:
    """The classifier and the verdict, checked as VALUES against inputs whose
    right answer is known. Nine of this repository's failed instruments compared
    STRUCTURE and all nine gave a false reading; three compared VALUES and none
    ever has. The new state is decision logic, so it is checked the same way."""
    cases = [((0, "107 passed in 387.03s"), PASS),   # the clean standalone run
             ((1, "3 failed in 1.2s"), FAIL),        # total failure, no `passed`
             ((1, "1 failed, 106 passed in 390.10s"), FAIL),
             ((0, ""), INDET),                       # exited 0, reported nothing
             ((5, "no tests ran in 0.01s"), INDET),  # collected nothing
             ((3221225786, ""), INDET),              # Windows kill
             ((-9, ""), INDET)]                      # POSIX SIGKILL
    got = [pytest_state(rc, out)[0] for (rc, out), _ in cases]
    want = [w for _, w in cases]
    check("tri-state classifier (7 inputs with known answers)", got == want,
          f"got={got}")

    # The five controls that guard the new state. Each feeds a KNOWN-BAD input
    # and requires the answer that is hard, not the one that is convenient.
    control("classifier files a total failure as FAIL, not INDETERMINATE",
            pytest_state(1, "3 failed in 1.2s")[0] == FAIL)
    control("classifier files a killed run as INDETERMINATE, not FAIL",
            pytest_state(3221225786, "")[0] == INDET)
    control("classifier files 'nothing collected' (exit 5) as INDETERMINATE, not PASS",
            pytest_state(5, "no tests ran in 0.01s")[0] == INDET)
    control("one INDETERMINATE and zero FAIL still exits NONZERO",
            verdict([("u", INDET, "")], [("c", True)]) == 1)
    control("a SILENT control alone still exits NONZERO",
            verdict([("p", PASS, "")], [("c", False)]) == 1)


# ------------------------------------------- 10 the rotation selector, unmasked

#: FOUND BY THIS FILE'S OWN VERIFICATION RUN, and it is instrument class #9 all
#: over again -- a regex compared against PROSE. The old pattern required the
#: word `complete` to follow the digits immediately:
#:
#:     r"\| iteration \| \*\*(\d+) complete"
#:
#: The round-four rewrite of STATE.md made the line read
#: `| iteration | **0 (round 4) complete, 1 next** |`. The pattern stopped
#: matching and `int(m.group(1)) + 1 if m else 0` SILENTLY fell back to 0 --
#: a legal iteration number, so nothing looked wrong while the journal replay and
#: the published-number rotation both selected ground nobody chose. A default
#: that is indistinguishable from a reading is the same defect as a killed
#: subprocess reported as a failure, so it gets the same state.
ITERATION_RE = re.compile(r"\| iteration \| \*\*(\d+)\b")


def _iteration_from_state(text: str) -> tuple[int, bool]:
    """(iteration, resolved). `resolved` False means STATE.md did not parse and
    the returned 0 is a DEFAULT, not a reading."""
    m = ITERATION_RE.search(text)
    return (int(m.group(1)) + 1, True) if m else (0, False)


def check_rotation(iteration: int, resolved: bool) -> None:
    check("iteration resolved (selects the replay + published rotations)",
          PASS if resolved else INDET,
          f"iteration={iteration}"
          + ("" if resolved else " is a DEFAULT -- STATE.md did not parse"))
    # MUST-FIRE: the selector must refuse prose it cannot read rather than
    # default, and must still read the round-four wording.
    control("rotation selector refuses an unparseable STATE.md",
            _iteration_from_state("| iteration | in flight |") == (0, False)
            and _iteration_from_state(
                "| iteration | **0 (round 4) complete, 1 next** |") == (1, True))


# ---------------------------------------------------------------------- driver

def main() -> int:
    if len(sys.argv) > 1:
        iteration, resolved = int(sys.argv[1]), True
    else:
        iteration, resolved = _iteration_from_state(
            (ROOT / "STATE.md").read_text(encoding="utf-8"))

    print(f"HEALTH INSPECTOR  iteration {iteration}\n")
    check_rotation(iteration, resolved)
    for fn, arg in ((check_calibration, None), (check_lock, None),
                    (check_replay, iteration), (check_published, iteration),
                    (check_suites, None), (check_lean, None),
                    (check_struck, None), (check_attribution, None),
                    (check_tri_state, None)):
        try:
            fn(iteration) if arg is not None else fn()
        except Exception as e:                       # a crashed check is a FAILED check
            check(f"{fn.__name__} (raised)", False, f"{type(e).__name__}: {e}")

    for name, state, detail in results:
        print(f"  [{state:>5}] {name}" + (f"  -- {detail}" if detail else ""))
    print()
    for name, fired in controls:
        print(f"  [{'FIRED' if fired else 'SILENT'}] must-fire: {name}")

    bad = [n for n, s, _ in results if s == FAIL]
    unk = [n for n, s, _ in results if s == INDET]
    silent = [n for n, f in controls if not f]
    print()
    if bad:
        print(f"  FAILED CHECKS: {len(bad)} -- {', '.join(bad)}")
    if unk:
        print(f"  INDETERMINATE CHECKS: {len(unk)} -- {', '.join(unk)}")
        print("  These did not measure. An unmeasured check is NOT a clean one --")
        print("  and it is not a defect either, so do not go looking for one.")
    if silent:
        print(f"  SILENT CONTROLS: {len(silent)} -- {', '.join(silent)}")
        print("  A check whose control stayed silent is BLIND. This is not a pass.")
    if not bad and not unk and not silent:
        print(f"  CLEAN: {len(results)} checks, {len(controls)} controls all fired.")
    return verdict(results, controls)


if __name__ == "__main__":
    raise SystemExit(main())
