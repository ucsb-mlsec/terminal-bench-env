FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/tailscale /tmp

COPY data/status_snapshot_001.json /var/log/tailscale/status_snapshot_001.json
COPY data/status_snapshot_002.json /var/log/tailscale/status_snapshot_002.json
COPY data/status_snapshot_003.txt /var/log/tailscale/status_snapshot_003.txt
COPY data/status_snapshot_004.json /var/log/tailscale/status_snapshot_004.json
COPY data/status_snapshot_005.txt /var/log/tailscale/status_snapshot_005.txt
COPY data/status_snapshot_006.json /var/log/tailscale/status_snapshot_006.json
COPY data/status_snapshot_007.txt /var/log/tailscale/status_snapshot_007.txt
COPY data/status_snapshot_008.json /var/log/tailscale/status_snapshot_008.json

WORKDIR /workspace