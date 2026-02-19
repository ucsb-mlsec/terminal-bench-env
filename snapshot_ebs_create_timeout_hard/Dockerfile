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
    apt-get install -y --no-install-recommends \
    python3-pip \
    jq && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy scipy

RUN mkdir -p /var/log/aws \
    /var/log/snapshots \
    /var/log/backup-system \
    /etc/ebs \
    /root/solution

COPY data/var/log/aws/describe_volumes.json /var/log/aws/describe_volumes.json
COPY data/var/log/aws/describe_snapshots.json /var/log/aws/describe_snapshots.json
COPY data/var/log/aws/snapshot_change_blocks.json /var/log/aws/snapshot_change_blocks.json
COPY data/var/log/snapshots/snapshot_timing.csv /var/log/snapshots/snapshot_timing.csv
COPY data/var/log/snapshots/failed_snapshots.log /var/log/snapshots/failed_snapshots.log
COPY data/var/log/backup-system/orchestration.log /var/log/backup-system/orchestration.log
COPY data/var/log/backup-system/system_metrics.csv /var/log/backup-system/system_metrics.csv
COPY data/etc/ebs/volume_configs.json /etc/ebs/volume_configs.json

WORKDIR /root