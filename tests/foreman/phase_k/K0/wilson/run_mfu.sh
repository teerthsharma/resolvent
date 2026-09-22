#!/bin/bash
# MFU sweep: each measurement is a fresh process that polls the card first. Synthetic shards (throughput is data-independent).
cd "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/wilson"
D=tmp_shards
python peak.py 4096 >> peak.jsonl 2>peak.err
python peak.py 8192 >> peak.jsonl 2>>peak.err
run() { name=$1; shift; echo "START $name $(date +%T)"; python train_ladder.py --data_dir $D --out_dir runs/mfu_$name --eval_every 1000000 --eval_batches 1 "$@" > runs/mfu_$name.out 2>&1; echo "END $name rc=$? $(date +%T)"; }
run R0_c1024_b8  --rung R0 --ctx 1024 --batch 8 --steps 300
run R0_c1024_b16 --rung R0 --ctx 1024 --batch 16 --steps 300
run R0_c4096_b2  --rung R0 --ctx 4096 --batch 2 --steps 300
run R1_c1024_b8  --rung R1 --ctx 1024 --batch 8 --steps 100
run R1_c4096_b2  --rung R1 --ctx 4096 --batch 2 --steps 100
run R2_c1024_b8  --rung R2 --ctx 1024 --batch 8 --steps 40
run R2_c4096_b2  --rung R2 --ctx 4096 --batch 2 --steps 40
run R3_c1024_b8  --rung R3 --ctx 1024 --batch 8 --steps 30
run R3_c4096_b2  --rung R3 --ctx 4096 --batch 2 --steps 30
run R3_c1024_b4  --rung R3 --ctx 1024 --batch 4 --steps 30
run R3_c4096_b1  --rung R3 --ctx 4096 --batch 1 --steps 30
echo SWEEPDONE $(date +%T)
