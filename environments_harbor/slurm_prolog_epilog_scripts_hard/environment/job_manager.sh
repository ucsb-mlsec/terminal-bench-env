#!/bin/bash

# SLURM Job Manager Script - Prolog/Epilog Handler
# Reads environment and state files, outputs actions to be taken

# Check arguments
if [ $# -ne 2 ]; then
    echo "ERROR: Usage: $0 <env_file> <state_file>" >&2
    exit 1
fi

ENV_FILE="$1"
STATE_FILE="$2"

# Validate input files exist
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file not found: $ENV_FILE" >&2
    exit 1
fi

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: State file not found: $STATE_FILE" >&2
    exit 1
fi

# Source environment variables
source "$ENV_FILE"

# Determine if this is prolog or epilog
IS_PROLOG=0
IS_EPILOG=0

if [ -n "$SLURM_PROLOG" ]; then
    IS_PROLOG=1
elif [ -n "$SLURM_EPILOG" ]; then
    IS_EPILOG=1
fi

# Parse JSON state file (using grep/sed for portability)
# Extract running_processes array
RUNNING_PROCESSES=$(grep -oP '"pid":\s*\K[0-9]+' "$STATE_FILE" 2>/dev/null)

# Extract temp_files array
TEMP_FILES=$(grep -oP '"path":\s*"\K[^"]+' "$STATE_FILE" 2>/dev/null)

# Extract user from state if present
STATE_USER=$(grep -oP '"user":\s*"\K[^"]+' "$STATE_FILE" 2>/dev/null | head -1)

# PROLOG handling (job starting)
if [ $IS_PROLOG -eq 1 ]; then
    # Check for conflicting processes and clean them up
    if [ -n "$RUNNING_PROCESSES" ]; then
        for pid in $RUNNING_PROCESSES; do
            echo "ACTION: cleanup_process pid=$pid"
        done
    fi
    
    # Set CUDA_VISIBLE_DEVICES based on SLURM_JOB_GPUS
    if [ -n "$SLURM_JOB_GPUS" ]; then
        # Convert SLURM_JOB_GPUS format to CUDA_VISIBLE_DEVICES format
        GPU_LIST=$(echo "$SLURM_JOB_GPUS" | tr ',' ',')
        echo "ACTION: set_env_var name=CUDA_VISIBLE_DEVICES value=$GPU_LIST"
    fi
    
    # Clean up old temp files
    if [ -n "$TEMP_FILES" ]; then
        for temp_file in $TEMP_FILES; do
            echo "ACTION: cleanup_temp_file path=$temp_file"
        done
    fi
    
    # Log job start
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "ACTION: log_event message=Job started successfully"
    else
        echo "ACTION: log_event message=Prolog executed"
    fi
fi

# EPILOG handling (job ending)
if [ $IS_EPILOG -eq 1 ]; then
    # Clean up processes belonging to the job user
    if [ -n "$RUNNING_PROCESSES" ]; then
        for pid in $RUNNING_PROCESSES; do
            # Check if process belongs to job user
            if [ -n "$SLURM_JOB_USER" ] && [ -n "$STATE_USER" ]; then
                if [ "$STATE_USER" = "$SLURM_JOB_USER" ]; then
                    echo "ACTION: cleanup_process pid=$pid"
                fi
            else
                echo "ACTION: cleanup_process pid=$pid"
            fi
        done
    fi
    
    # Remove temp files matching the job ID pattern
    if [ -n "$TEMP_FILES" ]; then
        for temp_file in $TEMP_FILES; do
            # Check if temp file matches job ID pattern
            if [ -n "$SLURM_JOB_ID" ]; then
                if echo "$temp_file" | grep -q "$SLURM_JOB_ID"; then
                    echo "ACTION: cleanup_temp_file path=$temp_file"
                fi
            else
                echo "ACTION: cleanup_temp_file path=$temp_file"
            fi
        done
    fi
    
    # Log cleanup completion
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "ACTION: log_event message=Cleanup completed for job $SLURM_JOB_ID"
    else
        echo "ACTION: log_event message=Epilog cleanup completed"
    fi
fi

exit 0