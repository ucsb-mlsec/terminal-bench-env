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
    openjdk-11-jdk-headless \
    libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /solution /config /data

COPY config/core-site.xml /config/core-site.xml
COPY config/hdfs-site.xml /config/hdfs-site.xml
COPY config/yarn-site.xml /config/yarn-site.xml
COPY config/mapred-site.xml /config/mapred-site.xml
COPY data/connection_timeout_logs.txt /data/connection_timeout_logs.txt
COPY data/cluster_metrics.csv /data/cluster_metrics.csv

WORKDIR /workspace