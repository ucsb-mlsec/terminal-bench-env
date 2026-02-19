FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y coreutils grep gawk util-linux && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/pipeline \
    && mkdir -p /var/incoming/logs \
    && mkdir -p /mnt/backup/logs \
    && mkdir -p /tmp/pipeline \
    && mkdir -p /var/pipeline/output \
    && mkdir -p /home/agent

COPY opt/pipeline/process_logs.sh /opt/pipeline/process_logs.sh
COPY data/incoming_logs/server_001.log /var/incoming/logs/server_001.log
COPY data/incoming_logs/server_002.log /var/incoming/logs/server_002.log
COPY data/incoming_logs/server_003.log /var/incoming/logs/server_003.log
COPY mnt/backup/logs/old_backup_001.log /mnt/backup/logs/old_backup_001.log
COPY mnt/backup/logs/old_backup_002.log /mnt/backup/logs/old_backup_002.log
COPY mnt/backup/logs/old_backup_003.log /mnt/backup/logs/old_backup_003.log
COPY tmp/pipeline/stage1.log /tmp/pipeline/stage1.log
COPY var/pipeline/output/stage2.log /var/pipeline/output/stage2.log

RUN chmod +x /opt/pipeline/process_logs.sh

WORKDIR /home/agent