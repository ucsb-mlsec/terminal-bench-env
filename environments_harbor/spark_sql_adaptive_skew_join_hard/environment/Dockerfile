FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/analysis /workspace/solution /workspace/data/analysis

COPY data/analysis/partition_stats.json /workspace/analysis/partition_stats.json
COPY data/analysis/cluster_config.json /workspace/analysis/cluster_config.json
COPY data/analysis/current_config.json /workspace/analysis/current_config.json
COPY data/analysis/partition_stats.json /workspace/data/analysis/partition_stats.json
COPY data/analysis/cluster_config.json /workspace/data/analysis/cluster_config.json
COPY data/analysis/current_config.json /workspace/data/analysis/current_config.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
