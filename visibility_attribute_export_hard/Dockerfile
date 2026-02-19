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
    apt-get install -y build-essential gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/libmath

COPY libmath/math_operations.c /workspace/libmath/math_operations.c
COPY libmath/math_operations.h /workspace/libmath/math_operations.h
COPY libmath/internal_helpers.c /workspace/libmath/internal_helpers.c
COPY libmath/Makefile /workspace/libmath/Makefile
COPY libmath/.gitignore /workspace/libmath/.gitignore

WORKDIR /workspace