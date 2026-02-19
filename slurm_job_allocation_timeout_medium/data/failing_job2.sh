#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --partition=general
#SBATCH --nodes=8
#SBATCH --ntasks=256
#SBATCH --time=02:00:00
#SBATCH --output=mpi_job_%j.out

# Allocation fails with: Unable to allocate resources: Timeout waiting for nodes

# MPI job that fails to allocate
module load openmpi
echo "Starting MPI job at $(date)"
srun ./my_mpi_application
echo "Job finished at $(date)"