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
    ninja-build \
    g++ \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake ninja-build g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/lib1 /workspace/lib2 /workspace/app

COPY CMakeLists.txt /workspace/CMakeLists.txt
COPY lib1/CMakeLists.txt /workspace/lib1/CMakeLists.txt
COPY lib1/lib1.h /workspace/lib1/lib1.h
COPY lib1/lib1.cpp /workspace/lib1/lib1.cpp
COPY lib2/CMakeLists.txt /workspace/lib2/CMakeLists.txt
COPY lib2/lib2.h /workspace/lib2/lib2.h
COPY lib2/lib2.cpp /workspace/lib2/lib2.cpp
COPY app/CMakeLists.txt /workspace/app/CMakeLists.txt
COPY app/main.cpp /workspace/app/main.cpp

WORKDIR /workspace