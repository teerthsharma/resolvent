#!/bin/bash
# Real FineWeb-Edu shards: R0 400 steps straight vs 200 + resume 200 (fresh process); R1 100 steps. Each run polls the card.
cd "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/wilson"
C="--data_dir C:/Users/seal/datasets/fineweb_edu --ctx 1024 --batch 8 --eval_every 100 --eval_batches 20"
run() { name=$1; shift; echo "START $name $(date +%T)"; python train_ladder.py $C "$@" >> runs/$name.out 2>&1; echo "END $name rc=$? $(date +%T)"; }
run real_R0_A --rung R0 --steps 400 --out_dir runs/real_R0_A
run real_R0_B --rung R0 --steps 400 --stop_at 200 --out_dir runs/real_R0_B
run real_R0_B --rung R0 --steps 400 --resume --out_dir runs/real_R0_B
run real_R1   --rung R1 --steps 100 --out_dir runs/real_R1
echo REALDONE $(date +%T)
