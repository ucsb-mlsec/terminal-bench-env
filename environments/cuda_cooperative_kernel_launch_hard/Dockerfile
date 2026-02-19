FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_kernels /workspace/data

COPY data/device_specs.txt /workspace/device_specs.txt
COPY cuda_kernels/README.md /workspace/cuda_kernels/README.md
COPY cuda_kernels/matrix_mult.cu /workspace/cuda_kernels/matrix_mult.cu
COPY cuda_kernels/reduction_coop.cu /workspace/cuda_kernels/reduction_coop.cu
COPY cuda_kernels/prefix_sum.cu /workspace/cuda_kernels/prefix_sum.cu
COPY cuda_kernels/histogram_grid.cu /workspace/cuda_kernels/histogram_grid.cu
COPY cuda_kernels/scan_block.cu /workspace/cuda_kernels/scan_block.cu
COPY cuda_kernels/transpose.cu /workspace/cuda_kernels/transpose.cu
COPY cuda_kernels/global_barrier.cu /workspace/cuda_kernels/global_barrier.cu
COPY cuda_kernels/vector_add.cu /workspace/cuda_kernels/vector_add.cu

WORKDIR /workspace