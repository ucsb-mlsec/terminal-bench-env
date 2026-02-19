FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cuda_project

COPY cuda_project/kernel_main.cu /workspace/cuda_project/kernel_main.cu
COPY cuda_project/processing.cu /workspace/cuda_project/processing.cu
COPY cuda_project/compute_kernels.cu /workspace/cuda_project/compute_kernels.cu
COPY cuda_project/utilities.cu /workspace/cuda_project/utilities.cu
COPY cuda_project/advanced_ops.cu /workspace/cuda_project/advanced_ops.cu
COPY cuda_project/special_kernels.cu /workspace/cuda_project/special_kernels.cu

WORKDIR /workspace