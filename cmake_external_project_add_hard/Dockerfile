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
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/myapp \
    /workspace/external_libs/mathutils

COPY myapp/main.cpp /workspace/myapp/main.cpp
COPY myapp/CMakeLists.txt /workspace/myapp/CMakeLists.txt
COPY external_libs/mathutils/mathutils.h /workspace/external_libs/mathutils/mathutils.h
COPY external_libs/mathutils/mathutils.c /workspace/external_libs/mathutils/mathutils.c
COPY external_libs/mathutils/CMakeLists.txt /workspace/external_libs/mathutils/CMakeLists.txt

WORKDIR /workspace