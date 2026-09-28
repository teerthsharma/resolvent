#!/bin/bash
# one job at a time; wait for >= 1.5 GB free host RAM before each start
cd "$(dirname "$0")"
run() { log=$1; shift
  while true; do
    free=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory")
    free=${free//[$'\r\n ']}
    [ "$free" -ge 1572864 ] && break
    echo "$(date +%T) wait free=${free}KB" >> queue_log.txt; sleep 30
  done
  echo "$(date +%T) start $* -> $log (free=${free}KB)" >> queue_log.txt
  python r4.py "$@" > "$log" 2>&1 && echo "$(date +%T) ok $log" >> queue_log.txt || echo "$(date +%T) FAIL $log" >> queue_log.txt
}
run log_lin2.txt lin2
run log_cells.txt cell 0.9 0.8
echo "$(date +%T) queue done" >> queue_log.txt
