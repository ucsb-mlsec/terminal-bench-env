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
    apt-get install -y wget curl && \
    wget https://github.com/nats-io/nats-server/releases/download/v2.10.7/nats-server-v2.10.7-linux-amd64.tar.gz && \
    tar -xzf nats-server-v2.10.7-linux-amd64.tar.gz && \
    mv nats-server-v2.10.7-linux-amd64/nats-server /usr/local/bin/ && \
    rm -rf nats-server-v2.10.7-linux-amd64* && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/nats /solution

COPY config/nats-server.conf /etc/nats/nats-server.conf

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
