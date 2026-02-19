FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/positions

COPY data/positions/position_001.json /workspace/positions/position_001.json
COPY data/positions/position_007.json /workspace/positions/position_007.json
COPY data/positions/position_008.json /workspace/positions/position_008.json
COPY data/positions/position_009.json /workspace/positions/position_009.json
COPY data/positions/position_010.json /workspace/positions/position_010.json

WORKDIR /workspace