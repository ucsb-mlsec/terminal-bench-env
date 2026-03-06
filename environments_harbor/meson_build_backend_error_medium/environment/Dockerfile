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
    meson \
    ninja-build \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y meson ninja-build g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/src

COPY meson.build /workspace/meson.build
COPY src/meson.build /workspace/src/meson.build
COPY src/lib.cpp /workspace/src/lib.cpp
COPY src/lib.h /workspace/src/lib.h
COPY src/main.cpp /workspace/src/main.cpp

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
