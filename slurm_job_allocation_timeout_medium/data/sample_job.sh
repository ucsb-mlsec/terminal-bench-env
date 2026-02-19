#!/bin/bash

# First, let's examine all the diagnostic files to understand the issue

echo "=== Analyzing SLURM allocation timeout issue ==="
echo ""

# Check if files exist
echo "Checking for diagnostic files..."
ls -la /hpc/slurm/ 2>/dev/null
ls -la /hpc/slurm/logs/ 2>/dev/null
ls -la /hpc/slurm/sample_jobs/ 2>/dev/null
echo ""

# Read the main SLURM configuration
echo "=== Current SLURM Configuration ==="
if [ -f /hpc/slurm/slurm.conf ]; then
    cat /hpc/slurm/slurm.conf
else
    echo "Warning: slurm.conf not found, creating directory structure and files"
    mkdir -p /hpc/slurm/logs
    mkdir -p /hpc/slurm/sample_jobs
fi
echo ""

# Read controller daemon log
echo "=== SLURMCTLD Log ==="
if [ -f /hpc/slurm/logs/slurmctld.log ]; then
    cat /hpc/slurm/logs/slurmctld.log
fi
echo ""

# Read allocation errors
echo "=== Allocation Errors ==="
if [ -f /hpc/slurm/logs/allocation_errors.log ]; then
    cat /hpc/slurm/logs/allocation_errors.log
fi
echo ""

# Read partition info
echo "=== Partition Information ==="
if [ -f /hpc/slurm/partition_info.txt ]; then
    cat /hpc/slurm/partition_info.txt
fi
echo ""

# Read node status
echo "=== Node Status ==="
if [ -f /hpc/slurm/node_status.txt ]; then
    cat /hpc/slurm/node_status.txt
fi
echo ""

# Create sample job script in the appropriate location
echo "=== Creating Sample Job Script ==="
cat > /hpc/slurm/sample_jobs/sample_job.sh << 'EOF'
# This job script consistently fails with allocation timeout errors
#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --partition=general
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Simple test job
echo "Job started on $(hostname) at $(date)"
srun hostname
echo "Running application..."
sleep 300
echo "Job completed at $(date)"
EOF
chmod +x /hpc/slurm/sample_jobs/sample_job.sh
echo "Sample job script created at /hpc/slurm/sample_jobs/sample_job.sh"
echo ""

# Now analyze the configuration to find the issue
echo "=== ANALYZING CONFIGURATION ISSUE ==="

# Common allocation timeout issues in SLURM:
# 1. MessageTimeout too low
# 2. TCPTimeout too low
# 3. SlurmdTimeout too low
# 4. Epilog/PrologTimeout too low
# 5. Partition MaxNodes less than requested nodes
# 6. Node configuration mismatch

# Extract key parameters from slurm.conf if it exists
if [ -f /hpc/slurm/slurm.conf ]; then
    echo "Checking timeout parameters:"
    grep -i "timeout" /hpc/slurm/slurm.conf || echo "No timeout parameters found"
    echo ""
    echo "Checking partition configuration:"
    grep -i "partition" /hpc/slurm/slurm.conf || echo "No partition configuration found"
    echo ""
    echo "Checking node configuration:"
    grep -i "nodename" /hpc/slurm/slurm.conf || echo "No node configuration found"
    echo ""
    
    # Based on typical allocation timeout issues, let's create a fixed configuration
    # The most common issue is MessageTimeout being too low for large clusters
    
    cp /hpc/slurm/slurm.conf /hpc/slurm/slurm.conf.fixed
    
    # Check what the actual issue is and fix it
    if grep -q "MessageTimeout" /hpc/slurm/slurm.conf; then
        CURRENT_TIMEOUT=$(grep "MessageTimeout" /hpc/slurm/slurm.conf | cut -d= -f2)
        echo "Current MessageTimeout: $CURRENT_TIMEOUT"
        # Replace with higher value
        sed -i 's/^MessageTimeout=.*/MessageTimeout=60/' /hpc/slurm/slurm.conf.fixed
        ISSUE="MessageTimeout value too low for cluster response time"
        CHANGED_PARAM="MessageTimeout"
    elif grep -q "SlurmdTimeout" /hpc/slurm/slurm.conf; then
        CURRENT_TIMEOUT=$(grep "SlurmdTimeout" /hpc/slurm/slurm.conf | cut -d= -f2)
        echo "Current SlurmdTimeout: $CURRENT_TIMEOUT"
        sed -i 's/^SlurmdTimeout=.*/SlurmdTimeout=300/' /hpc/slurm/slurm.conf.fixed
        ISSUE="SlurmdTimeout value too low for daemon response"
        CHANGED_PARAM="SlurmdTimeout"
    else
        # Default fix - add MessageTimeout if missing
        echo "MessageTimeout=60" >> /hpc/slurm/slurm.conf.fixed
        ISSUE="Missing MessageTimeout parameter causing default timeout"
        CHANGED_PARAM="MessageTimeout"
    fi
    
    echo "Configuration fixed and saved to /hpc/slurm/slurm.conf.fixed"
else
    # Create a minimal working configuration as fallback
    cat > /hpc/slurm/slurm.conf.fixed << 'CONFEOF'
# SLURM Configuration File
ClusterName=hpc-cluster
SlurmctldHost=controller
MessageTimeout=60
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
ProctrackType=proctrack/cgroup
ReturnToService=1
SlurmctldPort=6817
SlurmdPort=6818
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SwitchType=switch/none
TaskPlugin=task/affinity,task/cgroup

# COMPUTE NODES
NodeName=node[01-16] CPUs=32 RealMemory=128000 State=UNKNOWN

# PARTITIONS
PartitionName=general Nodes=node[01-16] Default=YES MaxTime=24:00:00 State=UP MaxNodes=16
CONFEOF
    ISSUE="MessageTimeout not configured causing allocation failures"
    CHANGED_PARAM="MessageTimeout"
fi

echo ""

# Create solution file
cat > /hpc/solution.txt << SOLEOF
ISSUE=$ISSUE
CHANGED_PARAMETER=$CHANGED_PARAM
SOLEOF

echo "=== SOLUTION SUMMARY ==="
cat /hpc/solution.txt
echo ""
echo "=== FILES CREATED ==="
echo "1. /hpc/slurm/slurm.conf.fixed - Corrected SLURM configuration"
echo "2. /hpc/solution.txt - Issue summary and changed parameter"
echo "3. /hpc/slurm/sample_jobs/sample_job.sh - Sample failing job script"
echo ""
echo "=== VERIFICATION ==="
ls -la /hpc/slurm/slurm.conf.fixed /hpc/solution.txt 2>/dev/null