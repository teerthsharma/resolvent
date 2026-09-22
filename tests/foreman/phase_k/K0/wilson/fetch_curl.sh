#!/bin/bash
# curl fetch of FineWeb-Edu sample/10BT shards at pinned revision; resumable (-C -), retried.
REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
cd /c/Users/seal/datasets/fineweb_edu/raw/sample/10BT
for i in "$@"; do
  f=$(printf "%03d_00000.parquet" $i)
  ( for try in 1 2 3 4 5 6 7 8; do
      curl -sSL -C - --retry 5 --retry-delay 3 --speed-limit 100000 --speed-time 60 -o $f "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/$REV/sample/10BT/$f" && break
      echo "$f retry $try"; sleep 3
    done; echo "$f done $(stat -c %s $f) $(date +%T)" ) &
done
wait
echo ALLDONE $(date +%T)
