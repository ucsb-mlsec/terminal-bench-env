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
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/backups_old /data/temp /opt/webapp

COPY data/backup_20240110_093000.tar.gz /data/backup_20240110_093000.tar.gz
COPY data/backup_20240112_140000.tar.gz /data/backup_20240112_140000.tar.gz
COPY data/backup_20240115_143022.tar.gz /data/backup_20240115_143022.tar.gz
COPY data/backup_20240113_100000.tar.gz /data/backup_20240113_100000.tar.gz
COPY data/backup_20240108_080000.tar.gz /data/backup_20240108_080000.tar.gz
COPY data/backups_old/backup_20240105_120000.tar.gz /data/backups_old/backup_20240105_120000.tar.gz
COPY data/temp/backup_20240114_160000.tar.gz /data/temp/backup_20240114_160000.tar.gz
COPY data/backup_20240111_170000.tar.gz /data/backup_20240111_170000.tar.gz

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
