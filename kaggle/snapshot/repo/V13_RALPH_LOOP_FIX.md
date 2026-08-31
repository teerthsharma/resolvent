# ralph-loop: a requested iteration cap was silently discarded

Evidence classes used throughout: **RUN** (executed this session, output pasted),
**READ** (exact file quoted with `path:line`), **DERIVED** (steps shown).
Nothing below is asserted without one of the three.

## Mechanism, one sentence

The setup script recognises a ceiling only in the literal form
`--max-iterations <n>`; the mount that lost its cap expressed it as prose
(`max_iterations 60`), which fell through the catch-all argument branch into the
prompt text, leaving `MAX_ITERATIONS` at its initialiser `0` — the same byte the
script writes when no cap was requested at all, and the byte the stop hook read
as infinity.

## Which copy of the plugin is live

READ — two trees hold the plugin:

- `C:\Users\seal\.claude\plugins\cache\claude-plugins-official\ralph-loop\1.0.0\`
- `C:\Users\seal\.claude\plugins\marketplaces\claude-plugins-official\plugins\ralph-loop\`

RUN — `installed_plugins.json` records the cache path as the install root:

```
"ralph-loop@claude-plugins-official": [{"scope": "user",
  "installPath": "C:\\Users\\seal\\.claude\\plugins\\cache\\claude-plugins-official\\ralph-loop\\1.0.0",
  "version": "1.0.0"}]
```

RUN — `diff` shows the two trees are not identical. `commands/ralph-loop.md`
differs: the cache copy quotes `"$ARGUMENTS"` and carries a seven-line paragraph
explaining why; the marketplace copy passes `$ARGUMENTS` unquoted and has no such
paragraph. That paragraph appears verbatim in the live session transcript
(evidence below), so the **cache copy is the tree Claude Code executed**. DERIVED
— the cache copy therefore already carried at least one local hand patch before
this one; `scripts/setup-ralph-loop.sh` in the cache also contains a 37-line
argument-unpacking block (`setup-ralph-loop.sh:13-48`) absent from the
marketplace copy.

## The mechanism, by line

All line numbers below refer to the pre-fix files, preserved at
`scripts/setup-ralph-loop.sh.bak` and `hooks/stop-hook.sh.bak`.

READ — `setup-ralph-loop.sh:10`, the initialiser. This is the byte that means
both "no cap requested" and, after a failed parse, "cap requested and lost":

```bash
MAX_ITERATIONS=0
```

READ — `setup-ralph-loop.sh:25`, the only place a cap is lifted out of the
single quoted argument blob that the slash command delivers:

```bash
    if [[ "$RAW_ARGS" =~ --max-iterations[[:space:]]+([0-9]+) ]]; then
```

The regex requires the literal two-dash flag followed by whitespace and digits.
`max_iterations 60`, `--max-iterations=60` and `--max-iterations sixty` all fail
to match, and no branch reports the miss.

READ — `setup-ralph-loop.sh:141-145`, the catch-all that swallows the unmatched
text into the prompt, which is why the failure is silent:

```bash
    *)
      # Non-option argument - collect all as prompt parts
      PROMPT_PARTS+=("$1")
      shift
      ;;
```

READ — `setup-ralph-loop.sh:182` writes the unchanged initialiser into the state
file, and `setup-ralph-loop.sh:195` renders the same `0` in the banner as the
word `unlimited`:

```bash
max_iterations: $MAX_ITERATIONS
...
Max iterations: $(if [[ $MAX_ITERATIONS -gt 0 ]]; then echo $MAX_ITERATIONS; else echo "unlimited"; fi)
```

READ — `stop-hook.sh:61` completes the chain by treating `0` as no ceiling:

```bash
if [[ $MAX_ITERATIONS -gt 0 ]] && [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
```

READ — `stop-hook.sh:13-18` tests only that the state file exists, so `active:`
in the frontmatter was decorative and the only manual stop was deleting or
renaming the file:

```bash
RALPH_STATE_FILE=".claude/ralph-loop.local.md"

if [[ ! -f "$RALPH_STATE_FILE" ]]; then
  # No active loop - allow exit
  exit 0
fi
```

DERIVED — of the four candidate causes named in the brief, the operative one is
"the flag is parsed but from the wrong form". The argument does reach the script
as a single quoted blob and the blob **is** split correctly
(`setup-ralph-loop.sh:19-48`), the front-matter writer does not substitute a
default, and no shell quoting failure occurs. What fails is that the cap was
never written in the one syntax the parser accepts, and no code path notices.

## The actual invocation that lost the cap

READ — `C:\Users\seal\.claude\projects\C--Users-seal-Desktop-New-folder--32-\cb61a629-d4b4-41bf-b17a-262f36c95d52.jsonl`,
line 113, the `Skill` tool call that mounted the loop. The `args` field, verbatim
in its opening line and its first requirement bullet:

```
Mount CEQ R10 continuation, max_iterations 60, hard cap, terminate at 60.
...
- max_iterations MUST be 60, never 0. 0 disables the ceiling entirely and is
  what made the last loop unstoppable.
```

The string `--max-iterations` does not occur anywhere in that blob. READ — line
115 of the same transcript, the expanded command output:

```
Iteration: 1
Max iterations: unlimited
Completion promise: none (runs forever)
```

DERIVED — the mount asked for a hard cap of 60 four separate times in prose and
received an unbounded loop, because prose is not the flag.

## Reproduction

RUN, in a scratch directory, against an unmodified copy of the shipped script,
with the prose form the transcript records:

```
$ bash setup-ralph-loop.sh "Mount CEQ R10 continuation, max_iterations 60, hard cap, terminate at 60. - max_iterations MUST be 60, never 0."
🔄 Ralph loop activated in this session!

Iteration: 1
Max iterations: unlimited
Completion promise: none (runs forever)

$ sed -n '1,8p' .claude/ralph-loop.local.md
---
active: true
iteration: 1
session_id: cb61a629-d4b4-41bf-b17a-262f36c95d52
max_iterations: 0
completion_promise: null
started_at: "2026-08-30T19:51:15Z"
---
```

That is the observed failure byte for byte: banner `unlimited`, state file
`max_iterations: 0`, `completion_promise: null`, exit status 0.

RUN — the same script, same session, two further forms. The distinction the
brief asked about does exist:

```
### --max-iterations=60 (equals form)
Max iterations: unlimited          <- silently dropped
max_iterations: 0

### --max-iterations 60 (documented form)
Max iterations: 60                 <- honoured
max_iterations: 60
```

DERIVED — the cap parses correctly when and only when written as
`--max-iterations` + whitespace + digits. Every other spelling, including the
common `=` form, is discarded without a diagnostic.

## The fix

Applied to the live install at
`C:\Users\seal\.claude\plugins\cache\claude-plugins-official\ralph-loop\1.0.0\`.
Originals preserved as `scripts/setup-ralph-loop.sh.bak` and
`hooks/stop-hook.sh.bak` alongside the patched files.

Three shape changes, matching the three required properties:

1. **The ceiling is asked for by name.** `--max-iterations` becomes mandatory and
   its value must be a positive integer or the literal `unlimited`. A mount with
   no ceiling flag, a malformed value, or a bare `0` exits 1 and writes no state
   file. Unbounded is now something a caller declares, never something a parser
   defaults to.
2. **`active: false` halts.** The stop hook parses the field and allows exit when
   it is anything other than `true`, leaving the state file in place so the flag
   can be flipped back.
3. **`unlimited` and a parse failure are different bytes.** `unlimited` is the
   only value that disables the ceiling. A numeric `0` is a ceiling of zero and
   stops on the first check; anything unparseable trips the existing
   corrupted-state-file branch.

```diff
--- setup-ralph-loop.sh.orig
+++ setup-ralph-loop.sh
@@ -7,7 +7,7 @@
 # Parse arguments
 PROMPT_PARTS=()
-MAX_ITERATIONS=0
+MAX_ITERATIONS=""
 COMPLETION_PROMISE="null"
@@ -22,7 +22,7 @@
   else
-    if [[ "$RAW_ARGS" =~ --max-iterations[[:space:]]+([0-9]+) ]]; then
+    if [[ "$RAW_ARGS" =~ --max-iterations[[:space:]=]+([0-9]+|unlimited) ]]; then
       MAX_ITERATIONS="${BASH_REMATCH[1]}"
@@ -61,7 +61,7 @@
 OPTIONS:
-  --max-iterations <n>           Maximum iterations before auto-stop (default: unlimited)
+  --max-iterations <n>           REQUIRED. Ceiling, or the literal 'unlimited'
@@ -79,12 +79,12 @@
-  /ralph-loop Refactor cache layer  (runs forever)
+  /ralph-loop Refactor cache layer --max-iterations unlimited  (runs forever)

 STOPPING:
-  Only by reaching --max-iterations or detecting --completion-promise
-  No manual stop - Ralph runs infinitely by default!
+  By reaching --max-iterations, by detecting --completion-promise, or by
+  setting 'active: false' in .claude/ralph-loop.local.md
@@ -95,6 +95,10 @@
       exit 0
       ;;
+    --max-iterations=*)
+      MAX_ITERATIONS="${1#*=}"
+      shift
+      ;;
     --max-iterations)
@@ -102,18 +106,18 @@
-        echo "     --max-iterations 0  (unlimited)" >&2
+        echo "     --max-iterations unlimited  (no ceiling)" >&2
-      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
-        echo "❌ Error: --max-iterations must be a positive integer or 0, got: $2" >&2
+      if [[ "$2" != "unlimited" ]] && ! [[ "$2" =~ ^[0-9]+$ ]]; then
+        echo "❌ Error: --max-iterations must be a positive integer or 'unlimited', got: $2" >&2
-        echo "     --max-iterations 0  (unlimited)" >&2
+        echo "     --max-iterations unlimited  (no ceiling)" >&2
@@ -149,6 +153,50 @@
 PROMPT="${PROMPT_PARTS[*]:-}"

+# An unconsumed --max-iterations left in the prompt means the value was
+# malformed. Separate that from "no cap requested" so the two are not one byte.
+if [[ -z "$MAX_ITERATIONS" ]] && [[ "$PROMPT" == *--max-iterations* ]]; then
+  echo "❌ Error: --max-iterations value could not be parsed" >&2
+  echo "" >&2
+  echo "   The flag survived into the prompt text, so its value was not a" >&2
+  echo "   positive integer or the literal 'unlimited'." >&2
+  echo "" >&2
+  echo "   Nothing was mounted." >&2
+  exit 1
+fi
+
+# The ceiling must be asked for by name. A cap written any other way (prose,
+# "max_iterations 60") is not parsed, and silently defaulting to unbounded made
+# "no cap requested" and "cap requested and lost" the same mount.
+if [[ -z "$MAX_ITERATIONS" ]]; then
+  echo "❌ Error: --max-iterations is required" >&2
+  echo "" >&2
+  echo "   Every loop must declare its ceiling explicitly:" >&2
+  echo "     --max-iterations 60          stop after 60 iterations" >&2
+  echo "     --max-iterations unlimited   no ceiling (say so on purpose)" >&2
+  echo "" >&2
+  echo "   A cap written as prose (e.g. 'max_iterations 60') is not parsed." >&2
+  echo "   Nothing was mounted." >&2
+  exit 1
+fi
+
+if [[ "$MAX_ITERATIONS" == "0" ]]; then
+  echo "❌ Error: --max-iterations 0 is not unlimited" >&2
+  echo "" >&2
+  echo "   0 is a ceiling of zero iterations. For no ceiling, pass:" >&2
+  echo "     --max-iterations unlimited" >&2
+  echo "" >&2
+  echo "   Nothing was mounted." >&2
+  exit 1
+fi
+
+if [[ "$MAX_ITERATIONS" != "unlimited" ]] && ! [[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
+  echo "❌ Error: --max-iterations must be a positive integer or 'unlimited', got: $MAX_ITERATIONS" >&2
+  echo "" >&2
+  echo "   Nothing was mounted." >&2
+  exit 1
+fi
+
 # Validate prompt is non-empty
@@ -192,7 +240,7 @@
 Iteration: 1
-Max iterations: $(if [[ $MAX_ITERATIONS -gt 0 ]]; then echo $MAX_ITERATIONS; else echo "unlimited"; fi)
+Max iterations: $MAX_ITERATIONS
@@ -201,8 +249,8 @@
-⚠️  WARNING: This loop cannot be stopped manually! It will run infinitely
-    unless you set --max-iterations or --completion-promise.
+⚠️  Ceiling: $MAX_ITERATIONS. To stop the loop by hand, set 'active: false'
+    in .claude/ralph-loop.local.md (or delete the file).
```

```diff
--- stop-hook.sh.orig
+++ stop-hook.sh
@@ -19,8 +19,9 @@
 FRONTMATTER=$(sed -n '/^---$/,/^---$/{ /^---$/d; p; }' "$RALPH_STATE_FILE")
-ITERATION=$(echo "$FRONTMATTER" | grep '^iteration:' | sed 's/iteration: *//')
-MAX_ITERATIONS=$(echo "$FRONTMATTER" | grep '^max_iterations:' | sed 's/max_iterations: *//')
+ITERATION=$(echo "$FRONTMATTER" | grep '^iteration:' | sed 's/iteration: *//' || true)
+MAX_ITERATIONS=$(echo "$FRONTMATTER" | grep '^max_iterations:' | sed 's/max_iterations: *//' || true)
+ACTIVE=$(echo "$FRONTMATTER" | grep '^active:' | sed 's/active: *//' || true)
@@ -34,6 +35,15 @@
   exit 0
 fi

+# 'active' used to be decorative: the only test was that the state file exists,
+# so a loop could be stopped only by deleting or renaming the file. A state file
+# with no 'active' field is treated as active (preserves old behavior).
+if [[ -n "$ACTIVE" ]] && [[ "$ACTIVE" != "true" ]]; then
+  echo "🛑 Ralph loop: active is '$ACTIVE' (not 'true'). Loop stopped." >&2
+  echo "   Set 'active: true' in $RALPH_STATE_FILE to resume." >&2
+  exit 0
+fi
+
 # Validate numeric fields before arithmetic operations
@@ -46,10 +56,10 @@
-if [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
+if [[ "$MAX_ITERATIONS" != "unlimited" ]] && [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
   echo "⚠️  Ralph loop: State file corrupted" >&2
   echo "   File: $RALPH_STATE_FILE" >&2
-  echo "   Problem: 'max_iterations' field is not a valid number (got: '$MAX_ITERATIONS')" >&2
+  echo "   Problem: 'max_iterations' is neither a number nor 'unlimited' (got: '$MAX_ITERATIONS')" >&2
@@ -57,8 +67,9 @@
-# Check if max iterations reached
-if [[ $MAX_ITERATIONS -gt 0 ]] && [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
+# Check if max iterations reached. Only the literal 'unlimited' disables the
+# ceiling; a numeric 0 is a ceiling of zero and stops on the first check.
+if [[ "$MAX_ITERATIONS" != "unlimited" ]] && [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
   echo "🛑 Ralph loop: Max iterations ($MAX_ITERATIONS) reached."
```

The `|| true` additions on the two field extractions are a second, smaller defect
found while reading. DERIVED — under `set -euo pipefail` (`stop-hook.sh:7`) a
state file missing `iteration:` or `max_iterations:` made the command
substitution return non-zero and killed the hook before it could print any of its
diagnostics. `session_id` at `stop-hook.sh:31` already carried `|| true`; the
other two did not.

## Verification

Every run below executed against the **installed** files, in a scratch
directory, never in the repository.

### Mount: a requested cap reaches the state file, a malformed one is refused

RUN:

```
### V1 prose cap, the observed invocation
exit=1
❌ Error: --max-iterations is required
(no state file written)

### V2 --max-iterations 60
exit=0
Max iterations: 60
max_iterations: 60

### V3 --max-iterations=60
exit=0
Max iterations: 60
max_iterations: 60

### V4 --max-iterations=abc
exit=1
❌ Error: --max-iterations value could not be parsed
(no state file written)

### V5 --max-iterations sixty
exit=1
❌ Error: --max-iterations value could not be parsed
(no state file written)

### V6 --max-iterations 0
exit=1
❌ Error: --max-iterations 0 is not unlimited
(no state file written)

### V7 --max-iterations unlimited
exit=0
Max iterations: unlimited
max_iterations: unlimited

### V8 --help still works
Ralph Loop - Interactive self-referential development loop
USAGE:
```

The invocation that produced the unbounded loop (V1) now exits 1 and mounts
nothing. `0` is refused at the mount (V6) rather than becoming infinity.

### Stop hook: ceiling, `active: false`, and the sentinel

RUN, each case a freshly written state file, hook input carrying a matching
`session_id` and a two-line fake transcript:

```
### H1 at ceiling (active=true iteration=60 max=60)
exit=0
🛑 Ralph loop: Max iterations (60) reached.
state: FILE REMOVED

### H2 under ceiling (active=true iteration=3 max=60)
exit=0
{"decision": "block", "reason": "\nDo the task.",
 "systemMessage": "🔄 Ralph iteration 4 | No completion promise set - loop runs infinitely"}
state: iteration: 4, active: true

### H3 active:false -> halts
exit=0
stdout: []
🛑 Ralph loop: active is 'false' (not 'true'). Loop stopped.
   Set 'active: true' in .claude/ralph-loop.local.md to resume.
active: false
iteration: 3

### H4 legacy max_iterations 0 (active=true iteration=1 max=0)
exit=0
🛑 Ralph loop: Max iterations (0) reached.
state: FILE REMOVED

### H5 max_iterations unlimited (active=true iteration=900)
exit=0
{"decision": "block", "reason": "\nDo the task.",
 "systemMessage": "🔄 Ralph iteration 901 | No completion promise set - loop runs infinitely"}
state: iteration: 901, active: true

### H6 max_iterations sixty (active=true iteration=3)
exit=0
⚠️  Ralph loop: State file corrupted
   Problem: 'max_iterations' is neither a number nor 'unlimited' (got: 'sixty')
state: FILE REMOVED
```

H3 is the decisive one: no `decision: block` JSON is emitted, so the session is
free to exit, and the state file survives with `active: false` intact.

### End to end: a cap of 3 actually terminates

RUN — one mount followed by three consecutive stop-hook invocations:

```
### E2E on installed files: cap 3 honoured
Max iterations: 3
  stop #1: {  "decision": "block",  "reason": "\nDo the task", "systemMessage…
  stop #2: {  "decision": "block",  "reason": "\nDo the task", "systemMessage…
  stop #3: 🛑 Ralph loop: Max iterations (3) reached.
  terminated at ceiling 3 (good)
```

### Backward compatibility, including the loop currently running

RUN — a state file with no `active:` field at all still runs, so pre-existing
state files are not broken by the new check:

```
### legacy state file with NO active: field still runs
{
  "decision": "block",
  "reason": "\nDo the task.",
```

RUN — the frontmatter of the live loop (`active: true`, `iteration: 2`,
`max_iterations: 60`, quoted multi-line `completion_promise`) was copied into the
scratch directory and fed to the patched hook with a matching `session_id`. The
real file was read only, never written:

```
active: true
iteration: 2
max_iterations: 60
{
  "decision": "block",
  "reason": "\nplaceholder prompt body",
  "systemMessage": "🔄 Ralph iteration 3 | To stop: output <promise>V13 COMPLETE: …</promise>
                    (ONLY when statement is TRUE - do not lie to exit!)"
}
-> iteration now: iteration: 3
```

DERIVED — the running loop's state shape is handled identically by the patched
hook: it continues, the iteration advances, and the completion promise still
parses. Its numeric `max_iterations: 60` is unaffected by the sentinel change.

## Property scorecard

| Required property | Status | Evidence |
|---|---|---|
| A requested cap reaches the state file, or the mount fails loudly | Achieved | RUN V1–V7: 60 reaches the file; prose, `abc`, `sixty` and `0` all exit 1 with no state file |
| `active: false` actually stops the loop | Achieved | RUN H3: no block JSON, session free to exit, file preserved |
| `unlimited` distinguishable from a parse failure | Achieved | RUN V6/V7 at the mount, H4/H5/H6 at the hook: `unlimited` is the only infinity, `0` is a ceiling of zero, junk is corruption |

## Limits

- **The install is not durable.** The patched files live under
  `plugins\cache\claude-plugins-official\ralph-loop\1.0.0\`, which a plugin
  update overwrites. READ — the marketplace tree carries `.gcs-sha`
  `0fc2bb13a805969c14b0fe9398bad41db346d84e` and is the source the cache is
  refreshed from. DERIVED — a refresh would revert this fix and also the earlier
  hand patch already present in the cache (the `"$ARGUMENTS"` quoting in
  `commands/ralph-loop.md` and the 37-line unpack block in the setup script,
  neither of which exists in the marketplace copy). A durable home is upstream in
  the `claude-plugins-official` marketplace source for the `ralph-loop` plugin;
  a durable local home would be a user-owned copy of `stop-hook.sh` registered as
  a `Stop` hook in `~/.claude/settings.json` together with a user-level command
  shadowing `/ralph-loop`. Neither was done here: the first is not reachable from
  this machine, and the second changes user configuration, which was outside the
  scope given. Backups sit at `scripts/setup-ralph-loop.sh.bak` and
  `hooks/stop-hook.sh.bak` next to the patched files.
- **The marketplace copy was left unpatched.** It is a different, older base for
  `setup-ralph-loop.sh`, and it is not the tree that executes. Patching it would
  extend survival only until the next GCS refresh, which rewrites it anyway.
- **Behaviour change for existing state files carrying `max_iterations: 0`.**
  Such a file now stops at the first hook invocation instead of running forever
  (RUN H4). That is the intended direction, but it is a change: any loop
  deliberately mounted unbounded under the old semantics must be rewritten to
  `max_iterations: unlimited` to keep running.
- **`--max-iterations` is now mandatory**, so every previously valid invocation
  that omitted it — including the documented example
  `/ralph-loop Refactor cache layer` — fails until `--max-iterations unlimited`
  is added. The help text was updated to match (RUN V8), but external
  documentation and muscle memory were not.
- **The stop hook was not observed firing inside a real Claude Code session.**
  All hook verification used hand-constructed hook input JSON and a two-line fake
  transcript in a scratch directory. The transcript-parsing and promise-detection
  paths (`stop-hook.sh:79-142`) were exercised only in the no-promise and
  single-text-block cases; the `<promise>` detection path was not re-tested, and
  it was not modified.
- **The live loop was not restarted.** The patched hook was proved correct
  against a copy of its frontmatter in scratch (RUN, above), not against the
  session itself. Whether Claude Code caches hook scripts for the duration of a
  session was not determined, so the running loop may still be executing the
  pre-patch hook until the session ends.
- **The `--max-iterations=*` argv branch trusts the value only as far as the
  final gate.** A value arriving that way is validated by the catch-all check
  before the state file is written (`--max-iterations=abc` refused, RUN V4), but
  it does not produce the flag-specific error text the spaced form produces.
- **No test file was added to the plugin.** The verification suite exists only as
  shell run in this session, reproduced in full above; nothing was left behind in
  the plugin directory to guard against regression.
- **`completion_promise` was not touched.** The same mount also requested a
  promise in prose and got `completion_promise: null`; that half of the observed
  failure has the same mechanism but was outside the stated scope, and a prose
  promise still silently becomes `null`.
- **The repository was not modified.** `.claude\ralph-loop.local.md` was read,
  never written; no git command that writes was run.
