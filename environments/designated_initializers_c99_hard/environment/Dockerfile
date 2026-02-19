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
    gcc \
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/config_manager

COPY config_manager/config_core.c /workspace/config_manager/config_core.c
COPY config_manager/config_loader.c /workspace/config_manager/config_loader.c
COPY config_manager/device_manager.c /workspace/config_manager/device_manager.c
COPY config_manager/protocol_handler.c /workspace/config_manager/protocol_handler.c
COPY config_manager/system_init.c /workspace/config_manager/system_init.c
COPY config_manager/config_types.h /workspace/config_manager/config_types.h
COPY config_manager/Makefile /workspace/config_manager/Makefile

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
