FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/monitor /var/logs/test

COPY opt/monitor/process_logs.sh /opt/monitor/process_logs.sh
COPY data/server_valid.log /var/logs/test/server_valid.log
COPY data/server_corrupted.log /var/logs/test/server_corrupted.log

RUN chmod +x /opt/monitor/process_logs.sh

WORKDIR /opt/monitor

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
