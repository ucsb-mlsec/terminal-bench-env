#!/bin/bash
#SBATCH --job-name=climate_analysis
#SBATCH --output=climate_logs_%A_%a.out
#SBATCH --error=climate_logs_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=0-149

# Climate data analysis job array
# Processing 150 climate model simulations

echo "Starting climate analysis job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Set up working directory
WORK_DIR="/scratch/climate_data/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Load climate model parameters for this task
MODEL_ID=$((SLURM_ARRAY_TASK_ID + 1))
SCENARIO="RCP${MODEL_ID}"

echo "Processing climate scenario: ${SCENARIO}"

# Simulate climate data processing
sleep $((5 + SLURM_ARRAY_TASK_ID % 10))

# Generate climate statistics
echo "Computing temperature anomalies..."
for year in {2020..2030}; do
    temp_anomaly=$(echo "scale=2; ($SLURM_ARRAY_TASK_ID * 0.01) + ($year - 2020) * 0.05" | bc)
    echo "${year},${temp_anomaly}" >> temperature_data_${MODEL_ID}.csv
done

# Calculate precipitation patterns
echo "Analyzing precipitation patterns..."
for month in {1..12}; do
    precip=$(echo "scale=1; 50 + ($SLURM_ARRAY_TASK_ID % 30) + ($month * 2)" | bc)
    echo "${month},${precip}" >> precipitation_${MODEL_ID}.csv
done

# Compile summary statistics
echo "Task ${SLURM_ARRAY_TASK_ID} summary:" > summary_${MODEL_ID}.txt
echo "Model scenario: ${SCENARIO}" >> summary_${MODEL_ID}.txt
echo "Years analyzed: 2020-2030" >> summary_${MODEL_ID}.txt
echo "Data points generated: $(wc -l < temperature_data_${MODEL_ID}.csv)" >> summary_${MODEL_ID}.txt

# Copy results to shared storage
RESULTS_DIR="/home/results/climate_analysis"
mkdir -p ${RESULTS_DIR}
cp *.csv *.txt ${RESULTS_DIR}/

echo "Climate analysis task ${SLURM_ARRAY_TASK_ID} completed at $(date)"
echo "Results saved to ${RESULTS_DIR}"