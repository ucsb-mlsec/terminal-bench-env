FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_project /tmp

COPY cuda_project/matrix_multiply.cu /workspace/cuda_project/matrix_multiply.cu
COPY cuda_project/fft_processor.cu /workspace/cuda_project/fft_processor.cu
COPY cuda_project/data_loader.cpp /workspace/cuda_project/data_loader.cpp
COPY cuda_project/reduction.cu /workspace/cuda_project/reduction.cu
COPY cuda_project/convolution.cu /workspace/cuda_project/convolution.cu
COPY cuda_project/solver.cpp /workspace/cuda_project/solver.cpp
COPY cuda_project/memory_manager.cu /workspace/cuda_project/memory_manager.cu
COPY cuda_project/utils.cpp /workspace/cuda_project/utils.cpp
COPY cuda_project/histogram.cu /workspace/cuda_project/histogram.cu
COPY cuda_project/main.cpp /workspace/cuda_project/main.cpp
COPY cuda_project/header.h /workspace/cuda_project/header.h

WORKDIR /workspace