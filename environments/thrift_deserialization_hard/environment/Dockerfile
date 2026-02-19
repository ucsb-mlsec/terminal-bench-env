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
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages thrift

RUN mkdir -p /workspace/logs /workspace/schema

COPY data/schema/metrics.thrift /workspace/schema/metrics.thrift
COPY data/logs/server_001.bin /workspace/logs/server_001.bin
COPY data/logs/server_002.bin /workspace/logs/server_002.bin
COPY data/logs/server_003.bin /workspace/logs/server_003.bin
COPY data/logs/server_004.bin /workspace/logs/server_004.bin
COPY data/logs/server_005.bin /workspace/logs/server_005.bin
COPY data/logs/server_006.bin /workspace/logs/server_006.bin
COPY data/logs/server_007.bin /workspace/logs/server_007.bin
COPY data/logs/server_008.bin /workspace/logs/server_008.bin
COPY data/logs/server_009.bin /workspace/logs/server_009.bin
COPY data/logs/server_010.bin /workspace/logs/server_010.bin
COPY data/logs/server_011.bin /workspace/logs/server_011.bin
COPY data/logs/server_012.bin /workspace/logs/server_012.bin

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
