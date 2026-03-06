FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /cluster-data /root/solution

COPY data/cluster-data/pvcs.yaml /cluster-data/pvcs.yaml
COPY data/cluster-data/pvs.yaml /cluster-data/pvs.yaml
COPY data/cluster-data/storage-classes.yaml /cluster-data/storage-classes.yaml
COPY data/cluster-data/events.log /cluster-data/events.log
COPY data/cluster-data/quotas.yaml /cluster-data/quotas.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
