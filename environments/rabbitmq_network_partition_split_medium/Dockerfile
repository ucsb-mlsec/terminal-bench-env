FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/rabbitmq/partition_event /tmp

COPY data/node1_status.txt /var/log/rabbitmq/partition_event/node1_status.txt
COPY data/node2_status.txt /var/log/rabbitmq/partition_event/node2_status.txt
COPY data/node3_status.txt /var/log/rabbitmq/partition_event/node3_status.txt

WORKDIR /workspace