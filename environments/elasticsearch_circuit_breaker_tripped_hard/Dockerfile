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

RUN mkdir -p /etc/elasticsearch /var/log/elasticsearch /home/user

COPY config/elasticsearch.yml /etc/elasticsearch/elasticsearch.yml
COPY data/cluster_stats.json /var/log/elasticsearch/cluster_stats.json
COPY data/elasticsearch.log /var/log/elasticsearch/elasticsearch.log
COPY data/heap_usage.json /var/log/elasticsearch/heap_usage.json

WORKDIR /home/user