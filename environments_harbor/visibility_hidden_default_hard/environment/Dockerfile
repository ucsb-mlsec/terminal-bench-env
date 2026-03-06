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
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

WORKDIR /workspace

COPY Makefile /workspace/Makefile
COPY core.cpp /workspace/core.cpp
COPY utils.cpp /workspace/utils.cpp
COPY plugin.cpp /workspace/plugin.cpp
COPY test_core.cpp /workspace/test_core.cpp
COPY test_utils.cpp /workspace/test_utils.cpp
COPY test_plugin.cpp /workspace/test_plugin.cpp
COPY core.h /workspace/core.h
COPY utils.h /workspace/utils.h
COPY plugin.h /workspace/plugin.h

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
