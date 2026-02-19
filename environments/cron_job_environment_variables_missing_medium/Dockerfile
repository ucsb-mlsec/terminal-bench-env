FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cron mysql-client curl && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /etc/cron.d /opt/backup /data /home/agent

# Copy generated files to their expected locations
COPY config/backup_job /etc/cron.d/backup_job
COPY daily_backup.sh /opt/backup/daily_backup.sh
COPY data/backup.log /data/backup.log

# Set proper permissions
RUN chmod 0644 /etc/cron.d/backup_job && \
    chmod +x /opt/backup/daily_backup.sh && \
    chmod 644 /data/backup.log

# Create backupuser if not exists
RUN useradd -m -s /bin/bash backupuser || true

WORKDIR /home/agent