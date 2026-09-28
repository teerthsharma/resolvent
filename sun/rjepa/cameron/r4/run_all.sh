#!/bin/sh
# sequential phases, one job at a time; waits for >= 1.5 GB free host RAM before each
cd "$(dirname "$0")"
for m in fixes sharp rev; do
  while [ "$(powershell -NoProfile -c '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)')" -lt 1500 ]; do echo "wait mem $(date +%T)"; sleep 30; done
  echo "== $m $(date +%T)"; python r4.py $m > $m.log 2>&1 || { echo "FAIL $m"; exit 1; }
done
echo "== done $(date +%T)"
