#!/bin/bash
# Supplementary sweep: equal micro-batch across rungs (4096 tokens) to separate rung from memory pressure (R2 b8 ran at 7.2-7.5 GiB of 8).
cd "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/wilson"
D=tmp_shards
run() { name=$1; shift; echo "START $name $(date +%T)"; python train_ladder.py --data_dir $D --out_dir runs/mfu_$name --eval_every 1000000 --eval_batches 1 "$@" > runs/mfu_$name.out 2>&1; echo "END $name rc=$? $(date +%T)"; }
run R0_c1024_b4 --rung R0 --ctx 1024 --batch 4 --steps 300
run R0_c4096_b1 --rung R0 --ctx 4096 --batch 1 --steps 300
run R1_c1024_b4 --rung R1 --ctx 1024 --batch 4 --steps 100
run R1_c4096_b1 --rung R1 --ctx 4096 --batch 1 --steps 100
run R2_c1024_b4 --rung R2 --ctx 1024 --batch 4 --steps 40
run R2_c4096_b1 --rung R2 --ctx 4096 --batch 1 --steps 40
echo SWEEPDONE2 $(date +%T)
