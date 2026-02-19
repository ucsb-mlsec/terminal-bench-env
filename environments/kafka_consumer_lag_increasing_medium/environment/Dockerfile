FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/kafka-diagnostics /tmp

COPY data/consumer.properties /opt/kafka-diagnostics/consumer.properties
COPY data/application.log /opt/kafka-diagnostics/application.log
COPY data/consumer_group_metrics.json /opt/kafka-diagnostics/consumer_group_metrics.json
COPY data/gc.log /opt/kafka-diagnostics/gc.log
COPY data/system_resources.txt /opt/kafka-diagnostics/system_resources.txt

WORKDIR /opt/kafka-diagnostics

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
