#!/usr/bin/env bash
# Poll the v17-K kernel and pull its output the moment it reaches a terminal
# state. Deliberately a plain script and not an agent: it must keep running
# when no session is open, and it must do exactly one thing.
#
#   ./kaggle/watch_v17k.sh [interval_seconds]      # default 3h after the first hour
#
# Kaggle exposes NO incremental log while a kernel runs -- `kernels output`
# returns nothing until the run finishes -- so status is all there is to poll.
# The first hour is polled every 15 min because the cheap failures live there:
# the environment gate, the SHA marker, the dataset hashes, and K-CERT's
# delta/tol HALT. After that the run is training and 3h is plenty.
set -u
KERNEL="melowdramtic/ceq-v17-k"
INTERVAL="${1:-10800}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$DIR/results/v17k_watch.log"
OUT="$DIR/results/kaggle_v17k_output"
mkdir -p "$(dirname "$LOG")" "$OUT"

# ONE INSTANCE ONLY. The Scheduled Task re-runs this every 3h so a reboot or a
# closed terminal cannot silently end the watch -- but a second live loop would
# double every log line, so a running instance wins and the new one exits.
PIDFILE="$DIR/results/.v17k_watch.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "=== watch started $(date -u +%FT%TZ) pid $$ kernel $KERNEL ===" >> "$LOG"
i=0
while :; do
    ts="$(date -u +%FT%TZ)"
    s="$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)"
    echo "$ts  $s" >> "$LOG"
    case "$s" in
        *COMPLETE*|*ERROR*|*CANCEL*)
            echo "$ts  terminal state -- pulling output" >> "$LOG"
            kaggle kernels output "$KERNEL" -p "$OUT" >> "$LOG" 2>&1
            echo "$ts  output in $OUT" >> "$LOG"
            exit 0
            ;;
    esac
    # first four polls at 15 min, then the requested interval
    if [ "$i" -lt 4 ]; then sleep 900; else sleep "$INTERVAL"; fi
    i=$((i + 1))
done
