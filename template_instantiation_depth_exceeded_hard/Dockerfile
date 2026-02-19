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
    apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_lib

COPY workspace/matrix_lib/matrix.hpp /workspace/matrix_lib/matrix.hpp
COPY workspace/matrix_lib/operations.hpp /workspace/matrix_lib/operations.hpp
COPY workspace/matrix_lib/test_matrix.cpp /workspace/matrix_lib/test_matrix.cpp
COPY workspace/matrix_lib/Makefile /workspace/matrix_lib/Makefile

WORKDIR /workspace/matrix_lib