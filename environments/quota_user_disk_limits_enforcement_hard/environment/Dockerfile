FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/quota_reports

COPY data/server1_quota.txt /tmp/quota_reports/server1_quota.txt
COPY data/server2_quota.txt /tmp/quota_reports/server2_quota.txt
COPY data/server3_quota.txt /tmp/quota_reports/server3_quota.txt

WORKDIR /tmp

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
