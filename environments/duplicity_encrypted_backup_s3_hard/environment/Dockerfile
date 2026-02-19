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
    apt-get install -y gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /backups/encrypted /restore/output /root

COPY root/restore_key.asc /root/restore_key.asc
COPY data/backup_20240115_143022.tar.gz.gpg /backups/encrypted/backup_20240115_143022.tar.gz.gpg
COPY data/backup_20240128_091545.tar.gz.gpg /backups/encrypted/backup_20240128_091545.tar.gz.gpg
COPY data/backup_20240210_162038.tar.gz.gpg /backups/encrypted/backup_20240210_162038.tar.gz.gpg
COPY data/backup_20240203_201512.tar.gz.gpg /backups/encrypted/backup_20240203_201512.tar.gz.gpg

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
