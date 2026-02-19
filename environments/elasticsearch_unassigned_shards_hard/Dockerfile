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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/elasticsearch_diagnostics /tmp

COPY data/cluster_health.json /var/elasticsearch_diagnostics/cluster_health.json
COPY data/cat_shards.json /var/elasticsearch_diagnostics/cat_shards.json
COPY data/allocation_explain.json /var/elasticsearch_diagnostics/allocation_explain.json
COPY data/cluster_settings.json /var/elasticsearch_diagnostics/cluster_settings.json
COPY data/nodes_stats.json /var/elasticsearch_diagnostics/nodes_stats.json

WORKDIR /workspace