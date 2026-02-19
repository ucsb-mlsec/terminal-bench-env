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
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages boto3

RUN mkdir -p /workspace/config /workspace/mock_data /workspace/data /workspace/solution

COPY kinesis_monitor.py /workspace/kinesis_monitor.py
COPY config/shard_metadata.json /workspace/config/shard_metadata.json
COPY config/stream_config.json /workspace/config/stream_config.json
COPY data/shard_metrics.json /workspace/data/shard_metrics.json
COPY data/traffic_analysis.json /workspace/data/traffic_analysis.json

WORKDIR /workspace