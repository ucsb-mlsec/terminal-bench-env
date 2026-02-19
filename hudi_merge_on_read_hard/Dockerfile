FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/transactions /workspace

COPY data/transactions/.manifest.json /data/transactions/.manifest.json
COPY data/transactions/snapshot_001.snapshot /data/transactions/snapshot_001.snapshot
COPY data/transactions/snapshot_002.snapshot /data/transactions/snapshot_002.snapshot
COPY data/transactions/delta_001.delta /data/transactions/delta_001.delta
COPY data/transactions/delta_002.delta /data/transactions/delta_002.delta
COPY data/transactions/delta_003.delta /data/transactions/delta_003.delta
COPY data/transactions/delta_004.delta /data/transactions/delta_004.delta

WORKDIR /workspace