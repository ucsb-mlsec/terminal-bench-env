FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_analysis/data

COPY data/gpu_specs.txt /workspace/cuda_analysis/gpu_specs.txt
COPY kernel_small_grid.cu /workspace/cuda_analysis/kernel_small_grid.cu
COPY kernel_large_grid.cu /workspace/cuda_analysis/kernel_large_grid.cu
COPY kernel_max_threads.cu /workspace/cuda_analysis/kernel_max_threads.cu
COPY kernel_2d_valid.cu /workspace/cuda_analysis/kernel_2d_valid.cu
COPY kernel_3d_invalid.cu /workspace/cuda_analysis/kernel_3d_invalid.cu
COPY kernel_edge_case.cu /workspace/cuda_analysis/kernel_edge_case.cu

WORKDIR /workspace/cuda_analysis