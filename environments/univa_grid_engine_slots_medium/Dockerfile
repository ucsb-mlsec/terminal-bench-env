FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/uge_configs/queues /var/uge_configs/hosts /tmp

COPY data/var/uge_configs/queues/all.q /var/uge_configs/queues/all.q
COPY data/var/uge_configs/queues/parallel.q /var/uge_configs/queues/parallel.q
COPY data/var/uge_configs/queues/serial.q /var/uge_configs/queues/serial.q
COPY data/var/uge_configs/hosts/node01.cluster.local /var/uge_configs/hosts/node01.cluster.local
COPY data/var/uge_configs/hosts/node02.cluster.local /var/uge_configs/hosts/node02.cluster.local
COPY data/var/uge_configs/hosts/node03.cluster.local /var/uge_configs/hosts/node03.cluster.local
COPY data/var/uge_configs/hosts/node04.cluster.local /var/uge_configs/hosts/node04.cluster.local
COPY data/var/uge_configs/hosts/node05.cluster.local /var/uge_configs/hosts/node05.cluster.local
COPY data/var/uge_configs/hosts/node06.cluster.local /var/uge_configs/hosts/node06.cluster.local

WORKDIR /var/uge_configs