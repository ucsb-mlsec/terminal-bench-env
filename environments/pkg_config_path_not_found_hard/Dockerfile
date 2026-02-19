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
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator && \
    mkdir -p /opt/mathutils/include && \
    mkdir -p /opt/mathutils/lib/pkgconfig

COPY workspace/calculator/main.cpp /workspace/calculator/main.cpp
COPY workspace/calculator/Makefile /workspace/calculator/Makefile
COPY workspace/calculator/README.md /workspace/calculator/README.md
COPY opt/mathutils/include/mathutils.h /opt/mathutils/include/mathutils.h
COPY opt/mathutils/lib/libmathutils.so /opt/mathutils/lib/libmathutils.so
COPY opt/mathutils/lib/libmathutils.a /opt/mathutils/lib/libmathutils.a
COPY opt/mathutils/lib/pkgconfig/mathutils.pc /opt/mathutils/lib/pkgconfig/mathutils.pc

WORKDIR /workspace/calculator