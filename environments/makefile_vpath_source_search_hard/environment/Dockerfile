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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/include \
    /workspace/src/x86 \
    /workspace/src/arm \
    /workspace/src/common

COPY include/memory.h /workspace/include/memory.h
COPY include/utils.h /workspace/include/utils.h
COPY include/processor.h /workspace/include/processor.h
COPY src/x86/memory.c /workspace/src/x86/memory.c
COPY src/x86/processor.c /workspace/src/x86/processor.c
COPY src/arm/memory.c /workspace/src/arm/memory.c
COPY src/arm/processor.c /workspace/src/arm/processor.c
COPY src/common/utils.c /workspace/src/common/utils.c
COPY src/common/memory.c /workspace/src/common/memory.c
COPY src/common/main.c /workspace/src/common/main.c
COPY Makefile /workspace/Makefile

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
