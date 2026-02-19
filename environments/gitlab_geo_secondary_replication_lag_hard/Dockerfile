FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/diagnostic_data /tmp

COPY data/primary_replication_slots.json /var/diagnostic_data/primary_replication_slots.json
COPY data/standby_status.json /var/diagnostic_data/standby_status.json
COPY data/wal_stats.json /var/diagnostic_data/wal_stats.json
COPY data/system_metrics.json /var/diagnostic_data/system_metrics.json
COPY data/postgres_config.json /var/diagnostic_data/postgres_config.json

WORKDIR /workspace