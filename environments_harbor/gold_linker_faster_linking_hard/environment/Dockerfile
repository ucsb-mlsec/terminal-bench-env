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
    binutils \
    lld \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/include /workspace/mathlib/src

COPY mathlib/CMakeLists.txt /workspace/mathlib/CMakeLists.txt
COPY mathlib/Makefile /workspace/mathlib/Makefile
COPY mathlib/include/math_ops.h /workspace/mathlib/include/math_ops.h
COPY mathlib/src/add.cpp /workspace/mathlib/src/add.cpp
COPY mathlib/src/subtract.cpp /workspace/mathlib/src/subtract.cpp
COPY mathlib/src/multiply.cpp /workspace/mathlib/src/multiply.cpp
COPY mathlib/src/divide.cpp /workspace/mathlib/src/divide.cpp
COPY mathlib/src/main.cpp /workspace/mathlib/src/main.cpp
COPY mathlib/test.sh /workspace/mathlib/test.sh

RUN chmod +x /workspace/mathlib/test.sh

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
