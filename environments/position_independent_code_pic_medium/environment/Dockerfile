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
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/src

COPY mathlib/Makefile /workspace/mathlib/Makefile
COPY mathlib/src/add.c /workspace/mathlib/src/add.c
COPY mathlib/src/subtract.c /workspace/mathlib/src/subtract.c
COPY mathlib/src/multiply.c /workspace/mathlib/src/multiply.c
COPY mathlib/src/divide.c /workspace/mathlib/src/divide.c
COPY mathlib/src/power.c /workspace/mathlib/src/power.c
COPY mathlib/src/math.h /workspace/mathlib/src/math.h

WORKDIR /workspace/mathlib

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
