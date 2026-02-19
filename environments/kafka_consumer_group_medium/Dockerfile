FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/partitions /workspace/consumer_group /workspace/results

COPY data/partitions/partition_0.txt /workspace/partitions/partition_0.txt
COPY data/partitions/partition_1.txt /workspace/partitions/partition_1.txt
COPY data/partitions/partition_2.txt /workspace/partitions/partition_2.txt
COPY broken_consumer.py /workspace/broken_consumer.py

WORKDIR /workspace