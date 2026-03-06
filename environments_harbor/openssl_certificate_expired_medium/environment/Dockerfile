FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/certificates /tmp

COPY data/api-server.pem /opt/certificates/api-server.pem
COPY data/mail-gateway.pem /opt/certificates/mail-gateway.pem
COPY data/web-portal.pem /opt/certificates/web-portal.pem
COPY data/database-proxy.pem /opt/certificates/database-proxy.pem
COPY data/auth-service.pem /opt/certificates/auth-service.pem
COPY data/legacy-service.pem /opt/certificates/legacy-service.pem
COPY data/monitoring-dashboard.pem /opt/certificates/monitoring-dashboard.pem
COPY data/readme.txt /opt/certificates/readme.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
