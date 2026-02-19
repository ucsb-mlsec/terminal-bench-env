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
    make \
    gdb \
    valgrind && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/queue_system

COPY lockfree_queue.h /workspace/queue_system/lockfree_queue.h
COPY lockfree_queue.c /workspace/queue_system/lockfree_queue.c
COPY test_harness.c /workspace/queue_system/test_harness.c
COPY Makefile /workspace/queue_system/Makefile
COPY README.md /workspace/queue_system/README.md

WORKDIR /workspace