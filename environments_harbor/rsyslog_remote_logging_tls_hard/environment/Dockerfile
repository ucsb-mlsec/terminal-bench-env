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
    apt-get install -y rsyslog rsyslog-gnutls && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/rsyslog-certs /var/log/remote /solution

COPY data/ca-cert.pem /etc/rsyslog-certs/ca-cert.pem
COPY data/server-cert.pem /etc/rsyslog-certs/server-cert.pem
COPY data/server-key.pem /etc/rsyslog-certs/server-key.pem
COPY data/client-cert.pem /etc/rsyslog-certs/client-cert.pem
COPY data/client-key.pem /etc/rsyslog-certs/client-key.pem

RUN chmod 600 /etc/rsyslog-certs/server-key.pem /etc/rsyslog-certs/client-key.pem && \
    chmod 644 /etc/rsyslog-certs/ca-cert.pem /etc/rsyslog-certs/server-cert.pem /etc/rsyslog-certs/client-cert.pem

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
