FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/corrupted_logs /solution

COPY data/corrupted_logs/system_events_2024_01.log.gz /data/corrupted_logs/system_events_2024_01.log.gz
COPY data/corrupted_logs/system_events_2024_02.log.gz /data/corrupted_logs/system_events_2024_02.log.gz
COPY data/corrupted_logs/system_events_2024_03.log.gz /data/corrupted_logs/system_events_2024_03.log.gz
COPY data/corrupted_logs/application_metrics_2024_01.log.gz /data/corrupted_logs/application_metrics_2024_01.log.gz
COPY data/corrupted_logs/application_metrics_2024_02.log.gz /data/corrupted_logs/application_metrics_2024_02.log.gz
COPY data/corrupted_logs/database_queries_2024_01.log.gz /data/corrupted_logs/database_queries_2024_01.log.gz
COPY data/corrupted_logs/database_queries_2024_02.log.gz /data/corrupted_logs/database_queries_2024_02.log.gz
COPY data/corrupted_logs/network_traffic_2024_01.log.gz /data/corrupted_logs/network_traffic_2024_01.log.gz

WORKDIR /workspace