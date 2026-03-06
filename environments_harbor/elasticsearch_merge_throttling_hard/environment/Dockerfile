FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /home/agent

COPY data/cluster_settings.json /data/cluster_settings.json
COPY data/index_settings.json /data/index_settings.json
COPY data/segments_stats.json /data/segments_stats.json
COPY data/node_stats.json /data/node_stats.json
COPY data/indexing_metrics.log /data/indexing_metrics.log

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
