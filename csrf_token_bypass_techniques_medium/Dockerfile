FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/traffic /workspace

COPY data/traffic/request_001.txt /workspace/traffic/request_001.txt
COPY data/traffic/request_002.txt /workspace/traffic/request_002.txt
COPY data/traffic/request_003.txt /workspace/traffic/request_003.txt
COPY data/traffic/request_004.txt /workspace/traffic/request_004.txt
COPY data/traffic/request_005.txt /workspace/traffic/request_005.txt
COPY data/traffic/request_006.txt /workspace/traffic/request_006.txt
COPY data/traffic/request_007.txt /workspace/traffic/request_007.txt
COPY data/traffic/request_008.txt /workspace/traffic/request_008.txt

WORKDIR /workspace