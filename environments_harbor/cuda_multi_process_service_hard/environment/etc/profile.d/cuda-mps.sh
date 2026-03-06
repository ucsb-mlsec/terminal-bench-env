#!/bin/bash
# CUDA Multi-Process Service (MPS) configuration
# This configures environment variables for CUDA MPS to enable GPU sharing

export CUDA_MPS_PIPE_DIRECTORY=/opt/mps-pipes
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
export PATH=/usr/local/cuda/bin:$PATH