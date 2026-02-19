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
    apt-get install -y openssl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

RUN mkdir -p /opt/ssl/certs /tmp

COPY data/server.crt /opt/ssl/certs/server.crt
COPY data/intermediate1.crt /opt/ssl/certs/intermediate1.crt
COPY data/root.crt /opt/ssl/certs/root.crt
COPY data/intermediate3.crt /opt/ssl/certs/intermediate3.crt
COPY data/expired_intermediate.crt /opt/ssl/certs/expired_intermediate.crt
COPY data/old_server.crt /opt/ssl/certs/old_server.crt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
