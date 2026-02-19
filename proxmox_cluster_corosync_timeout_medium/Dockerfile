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
    apt-get install -y \
    vim \
    nano \
    net-tools \
    iputils-ping && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/pve /etc/corosync /var/log/corosync /data /root

COPY config/corosync.conf /etc/corosync/corosync.conf
COPY data/network_latency.log /data/network_latency.log
COPY data/corosync.log /data/corosync.log
COPY data/system_load.log /data/system_load.log

WORKDIR /root