FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install CMake, CUDA toolkit, and build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/ml-project

# Copy project files to container
COPY ml-project/CMakeLists.txt /workspace/ml-project/CMakeLists.txt
COPY ml-project/main.cpp /workspace/ml-project/main.cpp
COPY ml-project/kernel.cu /workspace/ml-project/kernel.cu
COPY ml-project/neural_net.h /workspace/ml-project/neural_net.h

WORKDIR /workspace