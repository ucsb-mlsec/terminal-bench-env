FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/nats-incident/subscriber-configs

COPY data/nats-incident/nats-server.log /workspace/nats-incident/nats-server.log
COPY data/nats-incident/subscriber-configs/subscriber-a.conf /workspace/nats-incident/subscriber-configs/subscriber-a.conf
COPY data/nats-incident/subscriber-configs/subscriber-b.conf /workspace/nats-incident/subscriber-configs/subscriber-b.conf
COPY data/nats-incident/subscriber-configs/subscriber-c.conf /workspace/nats-incident/subscriber-configs/subscriber-c.conf
COPY data/nats-incident/server.conf /workspace/nats-incident/server.conf

WORKDIR /workspace