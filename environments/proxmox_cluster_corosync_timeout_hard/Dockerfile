FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /root /etc/pve /var/log/cluster

COPY data/network_latency.txt /root/network_latency.txt
COPY config/corosync.conf /etc/pve/corosync.conf
COPY data/corosync.log /var/log/cluster/corosync.log

WORKDIR /root