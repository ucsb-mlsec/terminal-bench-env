FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data/kinesis_stream /workspace/data/logs /workspace/data/checkpoint /workspace/solution

COPY data/kinesis_stream/shard_0.jsonl /workspace/data/kinesis_stream/shard_0.jsonl
COPY data/kinesis_stream/shard_1.jsonl /workspace/data/kinesis_stream/shard_1.jsonl
COPY data/kinesis_stream/shard_2.jsonl /workspace/data/kinesis_stream/shard_2.jsonl
COPY data/kinesis_stream/shard_0.jsonl /workspace/data/kinesis_stream/shard-0.jsonl
COPY data/kinesis_stream/shard_1.jsonl /workspace/data/kinesis_stream/shard-1.jsonl
COPY data/kinesis_stream/shard_2.jsonl /workspace/data/kinesis_stream/shard-2.jsonl
COPY data/logs/processing.log /workspace/data/logs/processing.log
COPY data/checkpoint/state.json /workspace/data/checkpoint/state.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
