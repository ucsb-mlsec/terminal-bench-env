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
    cmake \
    g++ \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/src /workspace/mathlib/tests

COPY mathlib/CMakeLists.txt /workspace/mathlib/CMakeLists.txt
COPY mathlib/src/CMakeLists.txt /workspace/mathlib/src/CMakeLists.txt
COPY mathlib/src/math_functions.cpp /workspace/mathlib/src/math_functions.cpp
COPY mathlib/src/math_functions.h /workspace/mathlib/src/math_functions.h
COPY mathlib/tests/CMakeLists.txt /workspace/mathlib/tests/CMakeLists.txt
COPY mathlib/tests/test_main.cpp /workspace/mathlib/tests/test_main.cpp

WORKDIR /workspace