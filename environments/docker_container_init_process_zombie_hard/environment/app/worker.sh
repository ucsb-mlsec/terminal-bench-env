#!/bin/bash

TASK_ID=${1:-"unknown"}

echo "[Worker] Starting task: $TASK_ID"
sleep 2
echo "[Worker] Processing complete for task: $TASK_ID"

exit 0