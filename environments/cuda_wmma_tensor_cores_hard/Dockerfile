FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_kernels

COPY cuda_kernels/matmul_wmma_basic.cu /workspace/cuda_kernels/matmul_wmma_basic.cu
COPY cuda_kernels/matmul_wmma_optimized.cu /workspace/cuda_kernels/matmul_wmma_optimized.cu
COPY cuda_kernels/matmul_standard.cu /workspace/cuda_kernels/matmul_standard.cu
COPY cuda_kernels/matmul_broken_wmma.cu /workspace/cuda_kernels/matmul_broken_wmma.cu
COPY cuda_kernels/matmul_naive.cu /workspace/cuda_kernels/matmul_naive.cu
COPY cuda_kernels/matmul_wmma_advanced.cu /workspace/cuda_kernels/matmul_wmma_advanced.cu

WORKDIR /workspace