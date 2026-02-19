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
    clang \
    llvm-14 \
    llvm-14-dev \
    llvm-14-tools \
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y clang llvm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

COPY vector.c /workspace/vector.c
COPY matrix.c /workspace/matrix.c
COPY stats.c /workspace/stats.c
COPY mathlib.h /workspace/mathlib.h
COPY Makefile /workspace/Makefile
COPY test.c /workspace/test.c

WORKDIR /workspace