FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /logs /config /data /solution

COPY data/logs/datanode1.log /logs/datanode1.log
COPY data/logs/datanode2.log /logs/datanode2.log
COPY data/logs/datanode3.log /logs/datanode3.log
COPY config/hdfs-site.xml /config/hdfs-site.xml
COPY data/block_manifest.txt /data/block_manifest.txt

WORKDIR /workspace