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
    apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib /workspace/app

COPY CMakeLists.txt /workspace/
COPY mathlib/CMakeLists.txt /workspace/mathlib/
COPY mathlib/math_operations.h /workspace/mathlib/
COPY mathlib/add.cpp /workspace/mathlib/
COPY mathlib/subtract.cpp /workspace/mathlib/
COPY mathlib/multiply.cpp /workspace/mathlib/
COPY mathlib/divide.cpp /workspace/mathlib/
COPY app/CMakeLists.txt /workspace/app/
COPY app/main.cpp /workspace/app/

WORKDIR /workspace