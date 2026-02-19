FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/hadoop/diagnostics /root/solution

COPY data/namenode.log /var/hadoop/diagnostics/namenode.log
COPY data/fsck_output.txt /var/hadoop/diagnostics/fsck_output.txt
COPY data/datanode_status.txt /var/hadoop/diagnostics/datanode_status.txt
COPY data/hdfs_config.xml /var/hadoop/diagnostics/hdfs_config.xml
COPY data/block_report.txt /var/hadoop/diagnostics/block_report.txt

WORKDIR /root