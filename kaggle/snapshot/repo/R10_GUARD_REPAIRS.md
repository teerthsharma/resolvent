# R10 — the guard-repair pass, and what repairing a guard taught about guards

The Health Inspector's Phase 1a audit ended with the round's harshest strike:

> "1 of 11 phase-1a findings has any runnable binding; 0 were RED before repair.
> No test file exists for any phase-1a module."

This is the record of clearing that, and of what each repair turned out to be. The
pattern that emerged is worth stating first: **five of the seven defects fixed here
were in the guards, not in the code the guards watch.** The instruments were less
carefully reasoned about than the things they measure, and nothing about a red test
distinguishes a guard that found everything from one that found half.

---

## 1. Three self-checks that nothing collected

`scale/it11_verdict.py`, `scale/idle_gate.py` and `scale/vram_gate.py` each ended
in an assert-based `demo()` behind a `__main__` guard. All three passed. None ran
unless a human typed the path.

A `demo()` behind `if __name__ == "__main__":` is a binding in the sense that a
fire extinguisher in a locked cupboard is fire protection.

`tests/loop/test_phase1a_modules_are_bound.py` gives each a collectible node id —
six nodes, all passing. Three assert the demo passes **and reaches its own OK
line**, so an early return cannot masquerade as a pass. Three plant a failure into
a copy of the tree and require the copy to exit non-zero:

| module | planted failure | guarantee disabled |
|---|---|---|
| `idle_gate.py` | `is_idle` → always `True` | UNKNOWN counts as busy |
| `it11_verdict.py` | the `n < REQUIRED_SEEDS` refusal | never rules on an unregistered N |
| `vram_gate.py` | the prices-nothing raise | refuses to gate over no resource |

## 2. Three vacuous must-fires, found by the Inspector

Each was **advertised by name** and did not fire.

**`idle_gate` MUST-FIRE 3.** `assert wait_until_idle(lambda: next(unknowns), ...)`
returns `True` whether or not the unknowns released the wait — it never looked at
how many readings were consumed. Reinstating the fail-open defect at the loop layer
gave:

    SHIPPED: True, 6 readings [-1, -1, -1, 0, 0, 0]
    BROKEN : True, 3 readings [-1, -1, -1]

and `demo()` exited 0 printing *"consecutive unknowns do not release"*. A boolean
return could never distinguish those two behaviours. Now checks the consumed count.

**`it11_verdict.demo()` passed on zero journals.** Six `print(verdict(...))` lines
assert nothing, so with no `results/` directory the gate deciding *"the region is
learnable"* exited 0 saying "demo OK" — it could not tell its journals from an
empty folder. Now asserts `cells()` is non-empty and that the published
[0.9462, 0.9591] interval reproduces.

**`vram_gate`'s UNKNOWN assertions were unreachable.** They sat inside
`if card is None:` / `if host is None:` — branches that never execute on any
machine with `nvidia-smi` and `psutil`, which is every machine this has run on. So
the module that *defines* "report UNKNOWN, never GREEN" had no executable evidence
it obeyed its own rule. Now forces both readers to fail through a planted
unreadable.

All three broken copies now exit 1.

## 3. Three modules that wrote at import

`tests/loop/test_no_module_writes_a_file_at_import.py`: **155 passed**, from RED
across several iterations.

- **`scale/it13_mars_binding.py`** — no `__main__` guard at all. Importing it ran
  600 draws (~73 s) and overwrote `results/r10_it13_saturn_binding.jsonl`, the
  evidence file for the it.13 verdict published from it.
- **`scale/replay_census.py`** — the destructive one. `OUT = pathlib.Path(sys.argv[1])`
  then `OUT.open("w")` at module level, so importing it read `argv[1]` (under
  pytest, a test file path) and **truncated that file**. That is the measurement
  quoted in the guard's own assertion: a 37-byte file truncated, 251 bytes written
  into it. Verified inert after repair.
- **`scale/p1prime.py`** — bound `REV` from `sys.argv` at module level and ran a
  full git census on import. Its writes were already gated behind `REV == PIN`
  after an earlier incident; that gate is kept. Verified still working: at HEAD it
  prints 195/46/149 against the shipped 191/46/145, labels itself *not the pin*,
  and writes nothing.

## 4. A guard with two defects, the first hiding the second

`test_conftest_import_is_order_dependent.py`. Catalogued as instance 20.

**Defect one:** a hardcoded roster of five offenders. The tree had moved past it in
both directions — three of the five retired to `attic/` at iteration 4, and five
live offenders never listed. It condemned five while five more went unreported:
MISTAKES.md V-14, in a guard.

Replacing the roster with a scan found **ten**.

**Defect two, visible only once the first was fixed:** the scan matched
`^\s*from conftest import`; the assertion used `line.startswith(...)` with no
`lstrip()`. Five of the ten newly-collected cases were parametrised in and **could
never fail**. The original roster happened to contain only unindented importers, so
the assertion's blindness had nothing to miss.

**Had the roster been fixed without re-running and counting, the guard would have
gone from reporting 5 of 10 to reporting 5 of 10 while appearing repaired.**

## 5. Two remedies with no code path to green

**The manifest-absence guard** offered *"record the coverage in the manifest, or
refuse a record missing a declared field"* — while asserting on the **record**,
which recording in the manifest cannot change. `manifest()` now returns
`absent: ['device']`, outside the hash inputs so no published hash moves
(verified: 15 of 16 covered, hash `210b72c2f070d334`). Rewriting the shipped `.pt`
files was refused: published artifacts, L-G2.

This is the same defect SATURN found in the attic guard at iteration 4 — an
`assert False` whose message named a remedy nothing in the function could reach.
Twice now, in guards written about vacuity.

**The identity-manifest guard** was RED because `kind` and `torch_version` were
declared collision-bearing and no case mutated them. Two parametrize entries added;
**both pass**, so the refusal worked all along and only the coverage was missing.
Its own **premise test then fired** — it pins the expected mutation set and refuses
to trust its verdict when that set changes underneath it. That is the behaviour
wanted: a guard whose premise silently absorbs a change to the thing it measures
cannot tell a repair from a regression.

---

## What stays RED, and why

| finding | why it stays |
|---|---|
| 10 conftest importers | repair priced (1 module, 11 sites, 6 function-local) and deferred: verification needs a full-suite run on a quiet tree |
| corpus provenance | `data/README.md` says 20,000 lines; the file has 211,766 and **no** countable unit equals 20,000. Repair needs the upstream split/revision/slice. Editing the number would fabricate provenance. **Artifacts verified intact: 11/11 corpora, 0 drifted.** |
| `device` in `CONFIG_FIELDS` | a declared discriminator absent from 60/60 records, while `identity_manifest.py:144-147` argues device must *not* affect identity. Removing it is hash-neutral but is a contract decision, not a defect to resolve mid-round. |
| variance law `:367` | one stale sentence against five; `Var(y) = t*` settled by measurement at 9.8 half-widths |

Each is a defect that is still real. None is a test waiting to be silenced.
