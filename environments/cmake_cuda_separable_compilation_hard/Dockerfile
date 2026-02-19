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
    cmake \
    nvidia-cuda-toolkit \
    g++ \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake build-essential nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/cuda_project/src /home/user/cuda_project/include

COPY CMakeLists.txt /home/user/cuda_project/CMakeLists.txt
COPY include/kernels.h /home/user/cuda_project/include/kernels.h
COPY include/utils.h /home/user/cuda_project/include/utils.h
COPY src/main.cu /home/user/cuda_project/src/main.cu
COPY src/kernels.cu /home/user/cuda_project/src/kernels.cu
COPY src/utils.cu /home/user/cuda_project/src/utils.cu

WORKDIR /home/user/cuda_project