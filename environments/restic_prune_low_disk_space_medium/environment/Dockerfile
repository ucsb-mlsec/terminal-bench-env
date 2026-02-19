FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y tar gzip && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /var/backups/daily /var/backups/.cache /var/log /solution

# Copy generated backup files to their appropriate locations
COPY data/backup-2024-01-15.tar.gz /var/backups/daily/backup-2024-01-15.tar.gz
COPY data/backup-2024-01-20.tar.gz /var/backups/daily/backup-2024-01-20.tar.gz
COPY data/backup-2024-01-25.tar.gz /var/backups/daily/backup-2024-01-25.tar.gz
COPY data/backup-2024-01-27.tar.gz /var/backups/daily/backup-2024-01-27.tar.gz
COPY data/backup-2024-01-28.tar.gz /var/backups/daily/backup-2024-01-28.tar.gz

# Copy cache files
COPY data/cache-temp-001.dat /var/backups/.cache/cache-temp-001.dat
COPY data/cache-temp-002.dat /var/backups/.cache/cache-temp-002.dat
COPY data/cache-extract-old.tmp /var/backups/.cache/cache-extract-old.tmp

# Copy log file
COPY data/backup-errors.log /var/log/backup-errors.log

WORKDIR /var/backups

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
