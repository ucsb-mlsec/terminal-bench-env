FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/network/capture_stats /solution

COPY data/session_001.log /var/log/network/capture_stats/session_001.log
COPY data/session_002.log /var/log/network/capture_stats/session_002.log
COPY data/session_003.log /var/log/network/capture_stats/session_003.log
COPY data/session_004.log /var/log/network/capture_stats/session_004.log
COPY data/session_005.log /var/log/network/capture_stats/session_005.log
COPY data/session_006.log /var/log/network/capture_stats/session_006.log
COPY data/session_007.log /var/log/network/capture_stats/session_007.log
COPY data/session_008.log /var/log/network/capture_stats/session_008.log
COPY data/session_009.log /var/log/network/capture_stats/session_009.log
COPY data/session_010.log /var/log/network/capture_stats/session_010.log

WORKDIR /workspace