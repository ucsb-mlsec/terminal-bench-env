FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/kafka-diagnostics/broker-configs && \
    mkdir -p /var/kafka-diagnostics/broker-logs && \
    mkdir -p /home

COPY data/var/kafka-diagnostics/cluster-metadata.json /var/kafka-diagnostics/cluster-metadata.json
COPY data/var/kafka-diagnostics/broker-configs/broker-1.properties /var/kafka-diagnostics/broker-configs/broker-1.properties
COPY data/var/kafka-diagnostics/broker-configs/broker-2.properties /var/kafka-diagnostics/broker-configs/broker-2.properties
COPY data/var/kafka-diagnostics/broker-configs/broker-3.properties /var/kafka-diagnostics/broker-configs/broker-3.properties
COPY data/var/kafka-diagnostics/broker-logs/broker-1.log /var/kafka-diagnostics/broker-logs/broker-1.log
COPY data/var/kafka-diagnostics/broker-logs/broker-2.log /var/kafka-diagnostics/broker-logs/broker-2.log
COPY data/var/kafka-diagnostics/broker-logs/broker-3.log /var/kafka-diagnostics/broker-logs/broker-3.log

WORKDIR /var/kafka-diagnostics