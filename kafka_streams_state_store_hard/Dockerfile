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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/streaming-app/checkpoints/.metadata && \
    mkdir -p /var/streaming-app/logs && \
    mkdir -p /solution

COPY data/checkpoints/partition-0.checkpoint /var/streaming-app/checkpoints/partition-0.checkpoint
COPY data/checkpoints/partition-1.checkpoint /var/streaming-app/checkpoints/partition-1.checkpoint
COPY data/checkpoints/partition-2.checkpoint /var/streaming-app/checkpoints/partition-2.checkpoint
COPY data/checkpoints/partition-3.checkpoint /var/streaming-app/checkpoints/partition-3.checkpoint
COPY data/checkpoints/partition-4.checkpoint /var/streaming-app/checkpoints/partition-4.checkpoint
COPY data/checkpoints/partition-5.checkpoint /var/streaming-app/checkpoints/partition-5.checkpoint
COPY data/checkpoints/partition-6.checkpoint /var/streaming-app/checkpoints/partition-6.checkpoint
COPY data/checkpoints/partition-8.checkpoint /var/streaming-app/checkpoints/partition-8.checkpoint
COPY data/checkpoints/partition-3.tmp /var/streaming-app/checkpoints/partition-3.tmp
COPY data/checkpoints/partition-7.tmp /var/streaming-app/checkpoints/partition-7.tmp
COPY data/checkpoints/.metadata/cluster.json /var/streaming-app/checkpoints/.metadata/cluster.json
COPY data/logs/application.log /var/streaming-app/logs/application.log

WORKDIR /solution