#!/bin/bash
#SBATCH --job-name=neural_net_training
#SBATCH --output=logs/nn_training_%A_%a.out
#SBATCH --error=logs/nn_training_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-899

# Neural network hyperparameter optimization job array
# Each array task trains a model with different hyperparameters

echo "Starting neural network training job ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# Load required modules
module load python/3.9
module load cuda/11.7
module load cudnn/8.5

# Activate virtual environment
source /home/user/venvs/ml_env/bin/activate

# Set hyperparameters based on array task ID
LEARNING_RATE=$(python -c "print(10 ** (-4 - ${SLURM_ARRAY_TASK_ID} % 3))")
BATCH_SIZE=$((32 * (1 + ${SLURM_ARRAY_TASK_ID} % 4)))
NUM_LAYERS=$((3 + ${SLURM_ARRAY_TASK_ID} % 5))
DROPOUT=$(python -c "print(0.1 + (${SLURM_ARRAY_TASK_ID} % 5) * 0.1)")

echo "Hyperparameters for this run:"
echo "Learning rate: ${LEARNING_RATE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Number of layers: ${NUM_LAYERS}"
echo "Dropout rate: ${DROPOUT}"

# Create output directory
mkdir -p /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID}

# Run the neural network training script
python /home/user/scripts/train_neural_net.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --learning-rate ${LEARNING_RATE} \
    --batch-size ${BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --dropout ${DROPOUT} \
    --epochs 100 \
    --dataset cifar100 \
    --optimizer adam \
    --output-dir /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID} \
    --save-checkpoints \
    --early-stopping-patience 10

# Evaluate the trained model
python /home/user/scripts/evaluate_model.py \
    --model-path /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID}/best_model.pth \
    --test-dataset cifar100 \
    --output-file /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID}/evaluation.json

# Generate visualizations
python /home/user/scripts/plot_training_curves.py \
    --log-file /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID}/training.log \
    --output-dir /home/user/ml_results/nn_training/run_${SLURM_ARRAY_TASK_ID}/plots

echo "Job ${SLURM_ARRAY_TASK_ID} completed at: $(date)"