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
    apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages nats-py

RUN mkdir -p /opt/nats /var/log/nats /tmp

COPY config/server.conf /opt/nats/server.conf
COPY data/server.log /var/log/nats/server.log
COPY consumer.py /opt/nats/consumer.py

WORKDIR /opt/nats

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
