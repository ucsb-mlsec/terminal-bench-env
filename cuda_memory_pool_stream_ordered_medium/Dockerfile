FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_sources /workspace/solution

COPY cuda_sources/kernel_optimized.cu /workspace/cuda_sources/kernel_optimized.cu
COPY cuda_sources/stream_handler.cpp /workspace/cuda_sources/stream_handler.cpp
COPY cuda_sources/legacy_code.cu /workspace/cuda_sources/legacy_code.cu
COPY cuda_sources/old_kernel.cu /workspace/cuda_sources/old_kernel.cu
COPY cuda_sources/utils.h /workspace/cuda_sources/utils.h
COPY cuda_sources/constants.cpp /workspace/cuda_sources/constants.cpp
COPY cuda_sources/mixed_allocations.cu /workspace/cuda_sources/mixed_allocations.cu
COPY cuda_sources/device_functions.cu /workspace/cuda_sources/device_functions.cu
COPY cuda_sources/memory_pool.cpp /workspace/cuda_sources/memory_pool.cpp
COPY cuda_sources/helper_functions.h /workspace/cuda_sources/helper_functions.h
COPY cuda_sources/traditional_malloc.c /workspace/cuda_sources/traditional_malloc.c

WORKDIR /workspace