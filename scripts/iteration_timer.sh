#!/usr/bin/env bash
# Iteration wall-clock cap for the CEQ ralph loop.
#
# The ralph stop hook counts ITERATIONS and knows nothing about TIME, so a single
# iteration that dispatches four planets can burn twenty minutes of wall clock and
# the loop cannot tell. This is the missing half: a per-iteration deadline that
# something other than the agent's own judgement enforces.
#
#   bash scripts/iteration_timer.sh start [minutes]   stamp the deadline, arm the watchdog
#   bash scripts/iteration_timer.sh check             elapsed / remaining / OVERDUE (exit 1 if overdue)
#   bash scripts/iteration_timer.sh stop              disarm, clear the marker
#
# `check` is the enforcement point and it is a SHELL exit code, not a request:
# exit 0 = inside the budget, exit 1 = OVERDUE, stop dispatching and file the record.
#
# The watchdog does NOT sweep-kill python. It kills only PIDs written to
# .claude/iteration.pids by the caller, because a blind `pkill python` on this box
# would take the harness with it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.claude/iteration.start"
PIDFILE="$ROOT/.claude/iteration.pids"
WATCHDOG="$ROOT/.claude/iteration.watchdog"
OVERDUE="$ROOT/.claude/ITERATION_OVERDUE"
CLOSED="$ROOT/.claude/ITERATION_CLOSED"
BEAT="$ROOT/.claude/iteration.beat"
EVENTS="$ROOT/house-events.jsonl"
DEFAULT_MIN=20

mkdir -p "$ROOT/.claude"

# Every line carries the time it was written. Without it the log records THAT the
# clock was armed and never WHEN, so two arms of one iteration read exactly like
# one arm -- which is why the it.21 re-arm could not be corroborated from here and
# the auditor had to fall back on wall times he had written down by hand.
log_event() {
  printf '{"t":"%s","agent":"timer","ts":"%s","text":"%s"}\n' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" >> "$EVENTS" 2>/dev/null || true
}

# Is this pid still the watchdog THIS iteration armed? The registration file is
# not evidence: it names a number, the OS recycles numbers, and it.19 retired
# $PIDFILE for exactly that -- reading a process fact it could not source. The
# pid's OWN record can be sourced, so the probe reads it: the watchdog is a
# `bash -c` whose argv carries $STATE, and a recycled pid running something else
# does not carry it. No readable procfs, no signal -- the destructive branch
# fails CLOSED, which is affordable only because a watchdog that outlives its
# disarm is inert (see the start-stamp corroboration in the body below).
_owns_watchdog() {
  local c="/proc/$1/cmdline"
  [[ -r "$c" ]] || return 1
  tr '\0' ' ' < "$c" 2>/dev/null | grep -qF -- "$STATE"
}

disarm() {
  if [[ -f "$WATCHDOG" ]]; then
    local wpid
    wpid="$(cat "$WATCHDOG" 2>/dev/null || true)"
    if [[ "$wpid" =~ ^[0-9]+$ ]] && _owns_watchdog "$wpid"; then
      kill "$wpid" 2>/dev/null
    fi
    rm -f "$WATCHDOG"
  fi
}

# Current loop iteration, so the marker names which one blew its budget.
loop_iteration() {
  local f="$ROOT/.claude/ralph-loop.local.md"
  [[ -f "$f" ]] && sed -n 's/^iteration: *//p' "$f" | head -1 || echo "?"
}

case "${1:-check}" in

  start)
    MINUTES="${2:-$DEFAULT_MIN}"
    [[ "$MINUTES" =~ ^[0-9]+$ ]] || { echo "minutes must be an integer, got: $MINUTES" >&2; exit 2; }

    # A concurrent process must not re-arm the clock under an agent still running.
    # The Inspector was stood in this hole twice: the it.2 audit had the timer
    # DISARMED under it, and after that was fixed the it.3 audit opened at
    # "iteration 5: 1m36s" and closed at "iteration 6: 0m20s, 19m40s left" because
    # `start` overwrote STATE unconditionally. A re-armed clock reads as a fresh
    # budget to everyone still inside the old one.
    # it.22: the refusal is no longer conditional on the iteration being INSIDE its
    # budget. That test -- `_e -lt _b` -- made the guard quit at the exact moment the
    # cap fires: an OVERDUE iteration fails it, so `start` fell straight through and
    # re-armed, silently, in the one state where a fresh clock does the most damage
    # (the room past cap, an auditor mid-audit holding an OVERDUE reading). A cap
    # that resets is worse than no cap, because the room believes it. An existing
    # $STATE now refuses, full stop; a malformed one refuses too, because the elapsed
    # arithmetic is only ever used for the message.
    if [[ -f "$STATE" ]] && [[ "${2:-}" != "--force" ]] && [[ "${3:-}" != "--force" ]]; then
      _s="$(sed -n 's/^start=//p' "$STATE")"; _b="$(sed -n 's/^budget_s=//p' "$STATE")"
      [[ "$_s" =~ ^[0-9]+$ ]] || _s=0
      [[ "$_b" =~ ^[0-9]+$ ]] || _b=0
      _e=$(( $(date +%s) - _s ))
      if [[ $_e -lt $_b ]]; then _w="is still live"; else _w="is OVERDUE and NOT CLOSED"; fi
      echo "REFUSED: iteration $(sed -n 's/^iteration=//p' "$STATE") $_w -- $((_e/60))m$((_e%60))s of $((_b/60))m used." >&2
      echo "   Re-arming would hand every agent still inside that budget a fresh clock." >&2
      echo "   Close it first:  bash scripts/iteration_timer.sh stop" >&2
      echo "   Or override on purpose:  bash scripts/iteration_timer.sh start $MINUTES --force" >&2
      exit 2
    fi
    disarm
    rm -f "$OVERDUE" "$PIDFILE" "$CLOSED" "$BEAT"
    NOW="$(date +%s)"
    IT="$(loop_iteration)"
    printf 'start=%s\nbudget_s=%s\niteration=%s\n' "$NOW" "$((MINUTES*60))" "$IT" > "$STATE"

    # Watchdog: sleep to the deadline, then raise the marker and kill only what the
    # caller registered. Detached so it survives this shell.
    #
    # It clears its own registration on exit ($8) and fires only if $STATE still
    # carries the start stamp it was armed against ($9). Before it.21 neither held:
    # the registration was removed in exactly one place, inside `disarm`, so a
    # watchdog that ran to completion left a file naming a dead pid for the next
    # `stop` to signal (MARS it.20 STRIKE 2). The stamp is the corroboration MARS
    # asked for, moved from the killer to the killed: a watchdog that survives its
    # disarm cannot raise OVERDUE against an iteration it was not armed for, so
    # `_owns_watchdog` declining to signal costs a stale process, not a wrong verdict.
    nohup bash -c '
      # $8 is $WATCHDOG and $9 is the start stamp this watchdog was armed against.
      trap "rm -f \"$8\"" EXIT
      sleep "$1"
      if [ -f "$2" ] && [ "$(sed -n "s/^start=//p" "$2")" = "$9" ]; then
        date -u +%Y-%m-%dT%H:%M:%SZ > "$3"
        echo "iteration $4 exceeded ${5} min" >> "$3"
        if [ -f "$6" ]; then
          while read -r p || [ -n "$p" ]; do
            [ -n "$p" ] && kill "$p" 2>/dev/null
          done < "$6"
        fi
        printf "{\"t\":\"finding\",\"agent\":\"timer\",\"ts\":\"%s\",\"text\":\"ITERATION %s OVERDUE: exceeded %s min wall clock\"}\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" "$5" >> "$7" 2>/dev/null
      fi
    ' _ "$((MINUTES*60))" "$STATE" "$OVERDUE" "$IT" "$MINUTES" "$PIDFILE" "$EVENTS" \
      "$WATCHDOG" "$NOW" \
      >/dev/null 2>&1 &
    echo $! > "$WATCHDOG"
    disown 2>/dev/null || true

    log_event "dispatch" "iteration $IT armed with a ${MINUTES} min cap"
    echo "iteration $IT armed: ${MINUTES} min cap, deadline $(date -d "@$((NOW+MINUTES*60))" -u +%H:%M:%SZ 2>/dev/null || echo "+${MINUTES}m")"
    ;;

  check)
    # A coordinator 'stop' must not read as 'never armed' to an agent still running.
    # The Inspector hit exactly that at it.2: it.3 disarmed the timer under him, and
    # a bare "NO TIMER ARMED" is indistinguishable from a harness fault. Closed is a
    # state, not an absence.
    if [[ ! -f "$STATE" ]] && [[ -f "$CLOSED" ]]; then
      echo "iteration CLOSED by the coordinator at $(cat "$CLOSED") - you are past the"
      echo "record; finish the sentence you are on and file. Do not start new work."
      exit 0
    fi
    if [[ ! -f "$STATE" ]]; then
      echo "NO TIMER ARMED - run: bash scripts/iteration_timer.sh start 20"
      exit 2
    fi
    START="$(sed -n 's/^start=//p' "$STATE")"
    BUDGET="$(sed -n 's/^budget_s=//p' "$STATE")"
    IT="$(sed -n 's/^iteration=//p' "$STATE")"
    ELAPSED=$(( $(date +%s) - START ))
    LEFT=$(( BUDGET - ELAPSED ))
    # THE ARM THIS READING CAME FROM. Elapsed and remaining describe a clock without
    # ever saying WHICH clock, so two readings taken across a re-arm are indistinguishable
    # from one monotonic clock -- the it.21 position, where the only way to catch the reset
    # was to write down the wall time of every call and subtract by hand. Printing the arm's
    # own stamp makes every re-arm route visible at once (`--force`, a `stop`+`start` pair,
    # and whatever the next one is), and lets a reading be DATED: armed-at plus elapsed is
    # the wall time of the call, with no arithmetic left to the reader.
    ARMED="$(date -d "@$START" -u +%H:%M:%SZ 2>/dev/null || echo "epoch $START")"
    if [[ -f "$OVERDUE" || $LEFT -le 0 ]]; then
      _last="$(cat "$BEAT" 2>/dev/null || echo "$START")"; _gap=$(( $(date +%s) - _last ))
      date +%s > "$BEAT"   # OVERDUE used to READ the beat and never WRITE it, so the gap
                           # grew without bound and every overrun past cap+5m announced
                           # itself as an interruption -- with the --force command under
                           # it. That defeated the cap it was built to make readable.
      echo "OVERDUE - iteration $IT ran $((ELAPSED/60))m$((ELAPSED%60))s against a $((BUDGET/60))m cap, armed $ARMED."
      echo "   STOP DISPATCHING. TaskStop every running agent, file the record, end the turn."
      if [[ $_gap -gt 300 ]]; then
        echo "   NOTE: $((_gap/60))m$((_gap%60))s since the last check call -- SPACING, not liveness."
        echo "   Nothing registers pids, so this instrument cannot corroborate an interruption."
        echo "   It is not a licence to re-arm. OVERDUE means stop."
      fi
      echo "STOP DISPATCHING. TaskStop every running agent, file the iteration record, end the turn."
      exit 1
    fi
    # Heartbeat. `check` is called throughout a live iteration, so the gap between
    # the last beat and now is dead time -- a suspended session, a quota stall, a
    # crash. Without this the timer reports wall clock and a reader cannot tell an
    # overrun from an interruption: it.16 read 645m50s against a 20m cap because the
    # session was suspended for ten hours, and the instrument had no way to say so.
    LAST="$(cat "$BEAT" 2>/dev/null || echo "$START")"
    NOWS="$(date +%s)"; GAP=$(( NOWS - LAST ))
    echo "$NOWS" > "$BEAT"
    # THE BEAT MEASURES CHECK-CALL SPACING AND NOTHING ELSE. it.18 tried to promote
    # it to a liveness verdict by corroborating against $PIDFILE -- but nothing in
    # this repo writes $PIDFILE, and the missing-file arm set ANY_DEAD=1, the
    # PERMISSIVE value, so "no evidence" and "evidence of death" printed the same
    # sentence. The predicate was `exists dead` and never cleared besides, so one
    # finished nurse of four licensed the interruption verdict while three ran.
    # A check nobody feeds launders unknown as confirmed. The channel is retired and
    # the output now names the quantity it actually holds.
    # ponytail: no liveness verdict at all. If one is ever genuinely wanted, give
    # $PIDFILE a writer FIRST, then predicate on `all dead AND >=1 registered` --
    # never `exists dead`, and never with a missing file as the permissive default.
    echo "iteration $IT: $((ELAPSED/60))m$((ELAPSED%60))s elapsed, $((LEFT/60))m$((LEFT%60))s left of $((BUDGET/60))m, armed $ARMED"
    if [[ $GAP -gt 300 ]]; then
      echo "   $((GAP/60))m$((GAP%60))s since the last check call. That is CHECK-CALL SPACING,"
      echo "   not liveness: nothing registers pids, so this instrument cannot tell a"
      echo "   suspended session from a busy one. Not evidence, and not a licence."
    fi
    exit 0
    ;;

  stop)
    disarm
    if [[ -f "$STATE" ]]; then
      START="$(sed -n 's/^start=//p' "$STATE")"
      IT="$(sed -n 's/^iteration=//p' "$STATE")"
      ELAPSED=$(( $(date +%s) - START ))
      log_event "done" "iteration $IT closed at $((ELAPSED/60))m$((ELAPSED%60))s"
      printf "%sm%ss (iteration %s)" "$((ELAPSED/60))" "$((ELAPSED%60))" "$IT" > "$CLOSED"
      echo "iteration $IT closed at $((ELAPSED/60))m$((ELAPSED%60))s"
    else
      echo "no timer was armed"
    fi
    rm -f "$STATE" "$OVERDUE" "$PIDFILE"
    ;;

  *)
    echo "usage: iteration_timer.sh {start [minutes]|check|stop}" >&2
    exit 2
    ;;
esac
