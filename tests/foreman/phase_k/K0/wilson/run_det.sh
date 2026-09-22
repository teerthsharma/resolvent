#!/bin/bash
cd "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/wilson"
C="--data_dir C:/Users/seal/datasets/fineweb_edu --rung R0 --ctx 1024 --batch 8 --eval_every 100 --eval_batches 20 --steps 400 --deterministic"
run() { name=$1; shift; echo "START $name $(date +%T)"; python train_ladder.py $C "$@" >> runs/$name.out 2>&1; echo "END $name rc=$? $(date +%T)"; }
run det_R0_A --out_dir runs/det_R0_A
run det_R0_B --stop_at 200 --out_dir runs/det_R0_B
run det_R0_B --resume --out_dir runs/det_R0_B
echo DETDONE $(date +%T)
