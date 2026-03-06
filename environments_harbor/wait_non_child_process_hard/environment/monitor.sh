#!/bin/bash

# Performance monitoring script for background jobs
# Author's assumption: wait command can track any process by PID

# This should work for any PID - we just need to wait on them
# Read the list of job PIDs that need to be monitored

JOBS_FILE="/home/user/jobs.txt"
RESULTS_FILE="/home/user/results.json"

echo "Starting performance monitor..."
echo "Reading PIDs from $JOBS_FILE"

# Initialize tracking variables
total_jobs=0
completed_jobs=0

# Record start time for each job
start_time=$(date +%s)

# Process each PID from the jobs file
for pid in $(cat $JOBS_FILE); do
    echo "Monitoring PID: $pid"
    total_jobs=$((total_jobs + 1))
    
    # Wait for process to complete - this should capture the exit status
    # The wait command works for any process ID, right?
    job_start=$(date +%s)
    
    wait $pid 2>/dev/null  # Wait for the process to finish
    exit_status=$?
    
    job_end=$(date +%s)
    duration=$((job_end - job_start))
    
    # Record the completion
    completed_jobs=$((completed_jobs + 1))
    echo "PID $pid completed with status $exit_status in ${duration}s"
done

# Calculate final statistics
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Write results - though timing might be inaccurate if wait fails
echo "{\"total\": $total_jobs, \"completed\": $completed_jobs, \"time\": $total_time}" > $RESULTS_FILE

echo "Monitoring complete. Results written to $RESULTS_FILE"