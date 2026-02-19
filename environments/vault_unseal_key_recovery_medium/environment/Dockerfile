FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /tmp

COPY data/vault_init.log /data/vault_init.log
COPY data/bash_history.txt /data/bash_history.txt
COPY data/syslog /data/syslog
COPY data/tmp_vault_session.txt /data/tmp_vault_session.txt
COPY data/vault_audit.log /data/vault_audit.log
COPY data/process_dump.txt /data/process_dump.txt
COPY data/journal_extract.log /data/journal_extract.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
