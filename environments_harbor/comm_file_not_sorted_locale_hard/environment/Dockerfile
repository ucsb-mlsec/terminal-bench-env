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
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 C.UTF-8 en_GB.UTF-8 de_DE.UTF-8 ja_JP.UTF-8 es_ES.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/master /data/regions/us-east /data/regions/us-west /data/regions/eu-central /data/regions/asia-pacific /data/regions/latam /opt/pipeline /solution

COPY data/master/expected_transactions.txt /data/master/expected_transactions.txt
COPY data/regions/us-east/transactions.txt /data/regions/us-east/transactions.txt
COPY data/regions/us-west/transactions.txt /data/regions/us-west/transactions.txt
COPY data/regions/eu-central/transactions.txt /data/regions/eu-central/transactions.txt
COPY data/regions/asia-pacific/transactions.txt /data/regions/asia-pacific/transactions.txt
COPY data/regions/latam/transactions.txt /data/regions/latam/transactions.txt
COPY reconcile.sh /opt/pipeline/reconcile.sh

RUN chmod +x /opt/pipeline/reconcile.sh

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
