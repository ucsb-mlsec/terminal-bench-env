FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/scripts /var/log/application /solution /data

COPY opt/scripts/cleanup.sh /opt/scripts/cleanup.sh
COPY opt/scripts/analyze_errors.sh /opt/scripts/analyze_errors.sh
COPY opt/scripts/disk_report.sh /opt/scripts/disk_report.sh
COPY data/generate_test_logs.sh /data/generate_test_logs.sh

RUN chmod +x /opt/scripts/cleanup.sh /opt/scripts/analyze_errors.sh /opt/scripts/disk_report.sh /data/generate_test_logs.sh

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
