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
    apt-get install -y openssl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

RUN mkdir -p /opt/ssl_certs/production && \
    mkdir -p /opt/ssl_certs/backup && \
    mkdir -p /etc/service && \
    mkdir -p /var/log && \
    mkdir -p /tmp

COPY data/opt/ssl_certs/production/server.crt /opt/ssl_certs/production/server.crt
COPY data/opt/ssl_certs/production/intermediate1.crt /opt/ssl_certs/production/intermediate1.crt
COPY data/opt/ssl_certs/production/intermediate2.crt /opt/ssl_certs/production/intermediate2.crt
COPY data/opt/ssl_certs/production/intermediate3.crt /opt/ssl_certs/production/intermediate3.crt
COPY data/opt/ssl_certs/production/root-ca.crt /opt/ssl_certs/production/root-ca.crt

COPY data/opt/ssl_certs/backup/server.crt /opt/ssl_certs/backup/server.crt
COPY data/opt/ssl_certs/backup/intermediate1.crt /opt/ssl_certs/backup/intermediate1.crt
COPY data/opt/ssl_certs/backup/intermediate2.crt /opt/ssl_certs/backup/intermediate2.crt
COPY data/opt/ssl_certs/backup/intermediate3.crt /opt/ssl_certs/backup/intermediate3.crt
COPY data/opt/ssl_certs/backup/root-ca.crt /opt/ssl_certs/backup/root-ca.crt

COPY config/etc/service/ssl_config.conf /etc/service/ssl_config.conf
COPY data/var/log/cert_validation.log /var/log/cert_validation.log

WORKDIR /workspace