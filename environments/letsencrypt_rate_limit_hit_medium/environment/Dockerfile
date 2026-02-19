FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /home/agent

COPY data/cert_renewal.log /data/cert_renewal.log
COPY data/certificate_status.json /data/certificate_status.json
COPY data/rate_limit_tracker.json /data/rate_limit_tracker.json

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
