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
    g++ \
    make \
    ccache \
    time && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential ccache time && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy_system/src /workspace/legacy_system/include

COPY legacy_system/src/main.cpp /workspace/legacy_system/src/
COPY legacy_system/src/compute_engine.cpp /workspace/legacy_system/src/
COPY legacy_system/src/data_processor.cpp /workspace/legacy_system/src/
COPY legacy_system/src/io_handler.cpp /workspace/legacy_system/src/
COPY legacy_system/src/validator.cpp /workspace/legacy_system/src/
COPY legacy_system/include/compute_engine.h /workspace/legacy_system/include/
COPY legacy_system/include/data_processor.h /workspace/legacy_system/include/
COPY legacy_system/include/io_handler.h /workspace/legacy_system/include/
COPY legacy_system/include/validator.h /workspace/legacy_system/include/
COPY legacy_system/Makefile /workspace/legacy_system/

ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR="/workspace/.ccache"

WORKDIR /workspace/legacy_system

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
