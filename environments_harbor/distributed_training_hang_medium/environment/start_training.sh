#!/bin/bash

echo "Starting distributed training with 4 processes"

export MASTER_ADDR=localhost
export MASTER_PORT=29500

for i in 0 1 2 3
do
    echo "Launching process with RANK=$i"
    export RANK=$i
    export WORLD_SIZE=4
    python train.py &
done

echo "All processes launched, waiting for completion..."
wait

echo "Distributed training completed"