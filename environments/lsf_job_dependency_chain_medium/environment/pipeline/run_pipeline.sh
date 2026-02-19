#!/bin/bash

echo "Starting pipeline..."

./process_stage1.sh &
./process_stage2.sh &
./process_stage3.sh &
./process_stage4.sh &

wait

echo "Pipeline completed"