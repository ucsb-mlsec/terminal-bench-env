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

RUN mkdir -p /opt/ca /opt/gateway /opt/certificates/subdir1 /opt/certificates/subdir2 /opt/certificates/subdir3 /tmp

COPY data/production-ca.crt /opt/ca/production-ca.crt
COPY data/auth_test.enc /opt/gateway/auth_test.enc

COPY data/cert1.crt /opt/certificates/subdir1/cert1.crt
COPY data/cert2.pem /opt/certificates/subdir1/cert2.pem
COPY data/gateway-api.crt /opt/certificates/subdir2/gateway-api.crt
COPY data/client-legacy.crt /opt/certificates/subdir2/client-legacy.crt
COPY data/vendor-cert.pem /opt/certificates/subdir2/vendor-cert.pem
COPY data/backup-2023.crt /opt/certificates/subdir3/backup-2023.crt
COPY data/production-api-client.crt /opt/certificates/subdir3/production-api-client.crt
COPY data/orphan-cert.pem /opt/certificates/orphan-cert.pem

COPY data/key1.key /opt/certificates/subdir1/key1.key
COPY data/key2.pem /opt/certificates/subdir1/key2.pem
COPY data/gateway-api.key /opt/certificates/subdir2/gateway-api.key
COPY data/client-legacy.key /opt/certificates/subdir2/client-legacy.key
COPY data/vendor-cert.key /opt/certificates/subdir3/vendor-cert.key
COPY data/backup-2023.key /opt/certificates/subdir3/backup-2023.key
COPY data/production-api-client.key /opt/certificates/subdir3/production-api-client.key
COPY data/random-key.pem /opt/certificates/random-key.pem

WORKDIR /opt

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
