FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /hadoop_data/datanode_reports /solution /config

COPY data/namenode.log /hadoop_data/namenode.log
COPY data/fsimage_summary.txt /hadoop_data/fsimage_summary.txt
COPY data/datanode_reports/datanode-1.status /hadoop_data/datanode_reports/datanode-1.status
COPY data/datanode_reports/datanode-2.status /hadoop_data/datanode_reports/datanode-2.status
COPY data/datanode_reports/datanode-3.status /hadoop_data/datanode_reports/datanode-3.status
COPY data/datanode_reports/datanode-4.status /hadoop_data/datanode_reports/datanode-4.status
COPY data/datanode_reports/datanode-5.status /hadoop_data/datanode_reports/datanode-5.status
COPY config/hdfs-site.xml /hadoop_data/hdfs-site.xml

WORKDIR /workspace