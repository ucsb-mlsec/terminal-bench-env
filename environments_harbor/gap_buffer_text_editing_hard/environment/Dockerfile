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

RUN mkdir -p /workspace/editor /workspace/solution

COPY editor/text_buffer.h /workspace/editor/text_buffer.h
COPY editor/implementation_a.c /workspace/editor/implementation_a.c
COPY editor/implementation_b.c /workspace/editor/implementation_b.c
COPY editor/performance_test.c /workspace/editor/performance_test.c
COPY editor/Makefile /workspace/editor/Makefile

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
