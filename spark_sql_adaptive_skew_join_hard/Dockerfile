FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /analysis /solution /data/analysis

COPY data/analysis/partition_stats.json /analysis/partition_stats.json
COPY data/analysis/cluster_config.json /analysis/cluster_config.json
COPY data/analysis/current_config.json /analysis/current_config.json
COPY data/analysis/partition_stats.json /data/analysis/partition_stats.json
COPY data/analysis/cluster_config.json /data/analysis/cluster_config.json
COPY data/analysis/current_config.json /data/analysis/current_config.json

WORKDIR /workspace