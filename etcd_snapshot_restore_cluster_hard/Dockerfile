FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /backup /etc/etcd/original /etc/etcd/new-cluster /solution

COPY data/etcd-export.json /backup/etcd-export.json
COPY config/original/node1.conf /etc/etcd/original/node1.conf
COPY config/original/node2.conf /etc/etcd/original/node2.conf
COPY config/original/node3.conf /etc/etcd/original/node3.conf
COPY config/new-cluster/node1.conf /etc/etcd/new-cluster/node1.conf
COPY config/new-cluster/node2.conf /etc/etcd/new-cluster/node2.conf

WORKDIR /workspace