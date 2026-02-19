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
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake libtool make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/src

COPY workspace/mathlib/configure.ac /workspace/mathlib/
COPY workspace/mathlib/Makefile.am /workspace/mathlib/
COPY workspace/mathlib/src/Makefile.am /workspace/mathlib/src/
COPY workspace/mathlib/src/mathops.c /workspace/mathlib/src/
COPY workspace/mathlib/src/utils.c /workspace/mathlib/src/
COPY workspace/mathlib/src/mathops.h /workspace/mathlib/src/
COPY workspace/mathlib/src/utils.h /workspace/mathlib/src/
COPY workspace/mathlib/configure /workspace/mathlib/
COPY workspace/mathlib/AUTHORS /workspace/mathlib/
COPY workspace/mathlib/NEWS /workspace/mathlib/
COPY workspace/mathlib/ChangeLog /workspace/mathlib/

WORKDIR /workspace/mathlib