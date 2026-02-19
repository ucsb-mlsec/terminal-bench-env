FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace

COPY data/cache_config.json /workspace/cache_config.json
COPY data/access_logs.txt /workspace/access_logs.txt
COPY data/test_scenarios.json /workspace/test_scenarios.json
COPY shard_manager.py /workspace/shard_manager.py

WORKDIR /workspace